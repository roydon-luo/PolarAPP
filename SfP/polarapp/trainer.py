import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from polarapp.checkpoints import (
    find_checkpoint,
    load_checkpoint,
    load_inference_checkpoints,
    save_training_checkpoints,
)
from polarapp.config import get_device, set_seed
from polarapp.data import build_eval_loader, build_training_loaders
from polarapp.evaluation import evaluate
from polarapp.losses import feature_cosine_loss, polar_loss, task_loss
from polarapp.models import build_models
from polarapp.operations import (
    EITransformer,
    clear_meta_parameters,
    get_coordinate,
    inner_update_from_loss,
    inter_data_process,
    outer_update,
    sync_meta_parameters,
)
from utils.init_interp import init_interp


def _set_trainable(module, trainable):
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _offset(length, crop_size):
    return random.randint(0, max(length - crop_size, 0))


def _crop_location(mask, crop_size, minimum_coverage=0.5, attempts=100):
    height, width = mask.shape[-2:]
    location = (0, 0)
    for _ in range(attempts):
        location = (_offset(height, crop_size), _offset(width, crop_size))
        top, left = location
        if (
            mask[:, :, top : top + crop_size, left : left + crop_size].mean()
            > minimum_coverage
        ):
            break
    return location


def _alignment_loss(fa_model, dem_features, task_features):
    transformed, generated = fa_model(dem_features, task_features)
    return (
        sum(
            feature_cosine_loss(transformed[f"TF{level}"], generated[f"MF{level}"])
            for level in (1, 2, 3)
        )
        / 3
    )


def train_feature_alignment(
    config,
    dataloader,
    dem_model,
    task_model,
    fa_model,
    optimizer_fa,
    device,
    interpolator,
    equivariant_interpolator,
    epoch,
):
    dem_model.train()
    task_model.train()
    fa_model.train()
    _set_trainable(fa_model, True)
    optimizer_fa.zero_grad(set_to_none=True)
    learning_rate = optimizer_fa.param_groups[0]["lr"]
    target_steps = config.max_meta_step
    if target_steps <= 0:
        target_steps = len(dataloader) // 2

    dem_losses, task_losses, alignment_losses = [], [], []
    outer_steps = 0
    progress = tqdm(
        total=target_steps,
        desc=f"Feature alignment {epoch + 1}/{config.epochs}",
    )
    for index, (pol, normal, mask, _) in enumerate(dataloader):
        if outer_steps >= target_steps:
            break
        pol = pol.to(device)
        normal = normal.to(device)
        mask = mask.to(device)
        height, width = pol.shape[-2:]
        crop_size = min(config.img_size, height, width)

        if index % 2 == 0:
            sync_meta_parameters(dem_model)
            sync_meta_parameters(task_model)
            top, left = _offset(height, crop_size), _offset(width, crop_size)
            pol_crop = pol[:, :, top : top + crop_size, left : left + crop_size]
            coordinate = get_coordinate(crop_size, crop_size).to(device)
            coordinate = coordinate.unsqueeze(0).expand(pol.shape[0], -1, -1, -1)
            pol_pred, dem_features = dem_model(interpolator(pol_crop), ELT_state=False)
            normal_pred, task_features = task_model(
                inter_data_process(pol_pred.detach(), coordinate)
            )
            loss_alignment = _alignment_loss(fa_model, dem_features, task_features)
            inner_update_from_loss(
                dem_model, loss_alignment, learning_rate, retain_graph=True
            )
            inner_update_from_loss(
                task_model, loss_alignment, learning_rate, retain_graph=True
            )
            alignment_losses.append(loss_alignment.item())
            continue

        resized_pol = F.interpolate(
            pol,
            size=(config.resize_h, config.resize_w),
            mode="bilinear",
            align_corners=False,
        )
        resized_normal = F.normalize(
            F.interpolate(normal, size=(config.resize_h, config.resize_w), mode="area"),
            p=2,
            dim=1,
        )
        resized_mask = F.interpolate(
            mask, size=(config.resize_h, config.resize_w), mode="nearest"
        )
        coordinate = get_coordinate(config.resize_h, config.resize_w).to(device)
        coordinate = coordinate.unsqueeze(0).expand(pol.shape[0], -1, -1, -1)

        pol_pred, _ = dem_model(interpolator(resized_pol), ELT_state=False, meta=True)
        transformed_pol = EITransformer().apply(pol_pred.detach())
        transformed_pred = dem_model(
            equivariant_interpolator(transformed_pol.detach()),
            ELT_state=True,
            meta=True,
        )
        normal_pred, _ = task_model(
            inter_data_process(pol_pred.detach(), coordinate), meta=True
        )
        loss_dem = polar_loss(
            transformed_pol, transformed_pred, device=device
        ) + 2 * polar_loss(pol_pred, resized_pol, device=device)
        loss_task = task_loss(normal_pred, resized_normal, resized_mask)
        loss_outer = loss_dem + config.task_loss_weight * loss_task
        loss_outer.backward()
        torch.nn.utils.clip_grad_norm_(fa_model.parameters(), max_norm=1.0)
        outer_update(fa_model, learning_rate)
        clear_meta_parameters(dem_model)
        clear_meta_parameters(task_model)
        optimizer_fa.zero_grad(set_to_none=True)
        dem_model.zero_grad(set_to_none=True)
        task_model.zero_grad(set_to_none=True)

        dem_losses.append(loss_dem.item())
        task_losses.append(loss_task.item())
        outer_steps += 1
        progress.update(1)
        progress.set_postfix(
            dem=f"{_mean(dem_losses):.4f}",
            task=f"{_mean(task_losses):.5f}",
            alignment=f"{_mean(alignment_losses):.6f}",
        )
    progress.close()


def train_joint(
    config,
    dataloader,
    dem_model,
    task_model,
    fa_model,
    optimizer_dem,
    optimizer_task,
    device,
    interpolator,
    equivariant_interpolator,
    epoch,
):
    dem_model.train()
    task_model.train()
    fa_model.eval()
    _set_trainable(fa_model, False)
    optimizer_dem.zero_grad(set_to_none=True)
    optimizer_task.zero_grad(set_to_none=True)
    max_steps = config.max_stage2_step
    total = len(dataloader) if max_steps <= 0 else min(len(dataloader), max_steps)
    dem_losses, task_losses, alignment_losses = [], [], []
    progress = tqdm(total=total, desc=f"Joint training {epoch + 1}/{config.epochs}")

    for index, (pol, normal, mask, _) in enumerate(dataloader):
        if max_steps > 0 and index >= max_steps:
            break
        pol = pol.to(device)
        normal = normal.to(device)
        mask = mask.to(device)
        height, width = pol.shape[-2:]
        crop_size = min(config.img_size, height, width)
        top, left = _crop_location(mask, crop_size)
        pol_crop = pol[:, :, top : top + crop_size, left : left + crop_size]
        normal_crop = normal[:, :, top : top + crop_size, left : left + crop_size]
        mask_crop = mask[:, :, top : top + crop_size, left : left + crop_size]
        coordinate = get_coordinate(crop_size, crop_size).to(device)
        coordinate = coordinate.unsqueeze(0).expand(pol.shape[0], -1, -1, -1)

        pol_pred, dem_features = dem_model(interpolator(pol_crop), ELT_state=False)
        transformed_pol = EITransformer().apply(pol_pred.detach())
        transformed_pred = dem_model(
            equivariant_interpolator(transformed_pol.detach()), ELT_state=True
        )
        normal_pred, task_features = task_model(
            inter_data_process(pol_pred, coordinate)
        )
        loss_dem = polar_loss(
            transformed_pol, transformed_pred, device=device
        ) + 2 * polar_loss(pol_pred, pol_crop, device=device)
        loss_task = task_loss(normal_pred, normal_crop, mask_crop)
        loss_alignment = _alignment_loss(fa_model, dem_features, task_features)
        loss = (
            loss_dem
            + config.task_loss_weight * loss_task
            + config.fa_loss_weight * loss_alignment
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dem_model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(task_model.parameters(), max_norm=1.0)
        optimizer_dem.step()
        optimizer_task.step()
        optimizer_dem.zero_grad(set_to_none=True)
        optimizer_task.zero_grad(set_to_none=True)

        dem_losses.append(loss_dem.item())
        task_losses.append(loss_task.item())
        alignment_losses.append(loss_alignment.item())
        progress.update(1)
        progress.set_postfix(
            dem=f"{_mean(dem_losses):.4f}",
            task=f"{_mean(task_losses):.5f}",
            alignment=f"{_mean(alignment_losses):.6f}",
        )
    progress.close()


def refine_task(
    config,
    dataloader,
    dem_model,
    task_model,
    optimizer_task,
    device,
    epoch,
):
    dem_model.eval()
    task_model.train()
    optimizer_task.zero_grad(set_to_none=True)
    max_steps = config.max_refine_step
    total = len(dataloader) if max_steps <= 0 else min(len(dataloader), max_steps)
    losses = []
    progress = tqdm(total=total, desc=f"Task refinement {epoch + 1}/{config.epochs}")

    for index, (pol, normal, mask, _) in enumerate(dataloader):
        if max_steps > 0 and index >= max_steps:
            break
        pol = pol.to(device)
        normal = normal.to(device)
        mask = mask.to(device)
        height, width = pol.shape[-2:]
        crop_size = min(config.img_size, height, width)
        top, left = _offset(height, crop_size), _offset(width, crop_size)
        pol_crop = pol[:, :, top : top + crop_size, left : left + crop_size]
        coordinate = get_coordinate(2 * crop_size, 2 * crop_size).to(device)
        coordinate = coordinate.unsqueeze(0).expand(pol.shape[0], -1, -1, -1)

        with torch.no_grad():
            pol_pred, _ = dem_model(pol_crop, ELT_state=False)
        normal_pred, _ = task_model(inter_data_process(pol_pred, coordinate))
        normal_gt = normal[:, :, top : top + crop_size, left : left + crop_size]
        normal_gt = F.normalize(
            F.interpolate(
                normal_gt,
                size=(2 * crop_size, 2 * crop_size),
                mode="bilinear",
                align_corners=False,
            ),
            p=2,
            dim=1,
        )
        mask_crop = mask[:, :, top : top + crop_size, left : left + crop_size]
        mask_up = F.interpolate(mask_crop, scale_factor=2, mode="nearest")
        loss = task_loss(normal_pred, normal_gt, mask_up)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(task_model.parameters(), max_norm=1.0)
        optimizer_task.step()
        optimizer_task.zero_grad(set_to_none=True)
        losses.append(loss.item())
        progress.update(1)
        progress.set_postfix(loss=f"{_mean(losses):.5f}")
    progress.close()


def _build_optimizers(config, dem_model, task_model, fa_model):
    optimizers = [
        optim.Adam(dem_model.parameters(), lr=config.lr),
        optim.Adam(task_model.parameters(), lr=config.lr),
        optim.Adam(fa_model.parameters(), lr=config.lr),
    ]
    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=1e-6
        )
        for optimizer in optimizers
    ]
    return optimizers, schedulers


def _resume_training(config, models, optimizers, schedulers, device):
    if not config.resume_state:
        return 0
    prefixes = ("DemNet", "TaskNet", "FANet")
    folders = (
        config.Dem_pretain_path,
        config.Task_pretain_path,
        config.FA_pretain_path,
    )
    checkpoints = []
    for prefix, folder, model, optimizer, scheduler in zip(
        prefixes, folders, models, optimizers, schedulers
    ):
        path = find_checkpoint(folder, prefix, getattr(config, "resume_epoch", None))
        checkpoints.append(load_checkpoint(model, path, device, optimizer, scheduler))
    start_epoch = int(checkpoints[0]["epoch"]) + 1
    print(f"Resuming from epoch {start_epoch}")
    return start_epoch


def fit(config, device):
    loaders = build_training_loaders(config)
    dem_model, task_model, fa_model = build_models(device)
    optimizers, schedulers = _build_optimizers(config, dem_model, task_model, fa_model)
    start_epoch = _resume_training(
        config,
        (dem_model, task_model, fa_model),
        optimizers,
        schedulers,
        device,
    )
    optimizer_dem, optimizer_task, optimizer_fa = optimizers
    scheduler_dem, scheduler_task, scheduler_fa = schedulers
    interpolator = init_interp(phase="train")
    equivariant_interpolator = init_interp(phase="train")
    checkpoint_root = getattr(
        config, "save_ckpt_dir", str(Path(config.save_dir) / "checkpoints")
    )

    for epoch in range(start_epoch, config.epochs):
        train_feature_alignment(
            config,
            loaders["meta"],
            dem_model,
            task_model,
            fa_model,
            optimizer_fa,
            device,
            interpolator,
            equivariant_interpolator,
            epoch,
        )
        train_joint(
            config,
            loaders["train"],
            dem_model,
            task_model,
            fa_model,
            optimizer_dem,
            optimizer_task,
            device,
            interpolator,
            equivariant_interpolator,
            epoch,
        )
        refine_task(
            config,
            loaders["refine"],
            dem_model,
            task_model,
            optimizer_task,
            device,
            epoch,
        )
        for scheduler in schedulers:
            scheduler.step()

        if config.save_freq > 0 and (epoch + 1) % config.save_freq == 0:
            save_training_checkpoints(
                checkpoint_root,
                epoch,
                (
                    ("DemNet", dem_model, optimizer_dem, scheduler_dem),
                    ("TaskNet", task_model, optimizer_task, scheduler_task),
                    ("FANet", fa_model, optimizer_fa, scheduler_fa),
                ),
            )
        if config.val_freq > 0 and (epoch + 1) % config.val_freq == 0:
            evaluate(
                loaders["val"],
                dem_model,
                task_model,
                device,
                Path(config.save_dir) / "images" / "val",
                tag=f"epoch_{epoch + 1:03d}",
                epoch=epoch + 1,
            )
        if config.test_freq > 0 and (epoch + 1) % config.test_freq == 0:
            evaluate(
                loaders["test"],
                dem_model,
                task_model,
                device,
                Path(config.save_dir) / "images" / "test",
                tag=f"epoch_{epoch + 1:03d}",
                epoch=epoch + 1,
            )


def evaluate_only(config, device):
    dem_model, task_model, _ = build_models(device, include_alignment=False)
    load_inference_checkpoints(
        dem_model,
        task_model,
        config.eval_ckpt_dir,
        device,
        getattr(config, "eval_epoch", None),
    )
    dataloader = build_eval_loader(
        config.test_data_path, config.test_batch_size, config.seed
    )
    return evaluate(
        dataloader,
        dem_model,
        task_model,
        device,
        Path(config.save_dir) / "images" / "test",
        tag=getattr(config, "eval_tag", "evaluation"),
        epoch=getattr(config, "eval_epoch", None),
        log_path=getattr(config, "eval_log_path", None),
    )


def run(config):
    set_seed(config.seed)
    device = get_device(getattr(config, "device", None))
    print(f"Using device: {device}")
    if getattr(config, "eval_only", False):
        return evaluate_only(config, device)
    return fit(config, device)
