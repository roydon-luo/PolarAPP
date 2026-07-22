"""Unified training, refinement, and evaluation workflow for PolarAPP-DfP."""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.func import functional_call
from tqdm import tqdm
from utils.init_interp import init_interp

from polarapp.checkpoints import (
    find_checkpoint,
    load_checkpoint,
    load_inference_checkpoints,
    save_training_checkpoints,
)
from polarapp.config import get_device, set_seed
from polarapp.data import build_eval_loader, build_training_loaders
from polarapp.evaluation import evaluate
from polarapp.losses import DfPTaskLoss, feature_alignment_loss, polar_loss
from polarapp.models import build_models
from polarapp.operations import EITransformer, inter_data_process


def _set_trainable(module, trainable):
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _random_crop(*tensors, size):
    height, width = tensors[0].shape[-2:]
    if height < size or width < size:
        target_size = (max(height, size), max(width, size))
        tensors = tuple(
            F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)
            for tensor in tensors
        )
        height, width = target_size
    top = random.randint(0, height - size)
    left = random.randint(0, width - size)
    return tuple(
        tensor[..., top : top + size, left : left + size] for tensor in tensors
    )


def _updated_parameters(model, loss, learning_rate, retain_graph):
    parameters = dict(model.named_parameters())
    trainable = {name: value for name, value in parameters.items() if value.requires_grad}
    gradients = torch.autograd.grad(
        loss,
        tuple(trainable.values()),
        create_graph=True,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    updated = dict(parameters)
    for (name, parameter), gradient in zip(trainable.items(), gradients):
        if gradient is not None:
            updated[name] = parameter - learning_rate * gradient
    return updated


def _demosaicking_loss(
    dem_model,
    polar,
    interpolator,
    equivariant_interpolator,
    parameters=None,
):
    demosaicker_input = interpolator(polar)
    if parameters is None:
        prediction, features = dem_model(demosaicker_input, ELT_state=False)
    else:
        prediction, features = functional_call(
            dem_model,
            parameters,
            (demosaicker_input,),
            {"ELT_state": False},
        )
    transformed = EITransformer().apply(prediction.detach())
    transformed_input = equivariant_interpolator(transformed.detach())
    if parameters is None:
        transformed_prediction = dem_model(transformed_input, ELT_state=True)
    else:
        transformed_prediction = functional_call(
            dem_model,
            parameters,
            (transformed_input,),
            {"ELT_state": True},
        )
    loss = polar_loss(transformed, transformed_prediction, device=str(polar.device))
    loss = loss + 2 * polar_loss(prediction, polar, device=str(polar.device))
    return prediction, loss, features


def _next_cropped_batch(iterator, dataloader, device, crop_size):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(dataloader)
        batch = next(iterator)
    polar = batch["polar"].to(device, non_blocking=device.type == "cuda")
    target = batch["gt_rgb"].to(device, non_blocking=device.type == "cuda")
    polar, target = _random_crop(polar, target, size=crop_size)
    return iterator, polar, target


def train_feature_alignment(
    config,
    dataloader,
    dem_model,
    task_model,
    fa_model,
    optimizer_fa,
    criterion,
    interpolator,
    equivariant_interpolator,
    device,
    epoch,
):
    target_steps = config.max_meta_step
    if target_steps <= 0:
        return
    dem_model.train()
    task_model.train()
    fa_model.train()
    _set_trainable(fa_model, True)
    iterator = iter(dataloader)
    learning_rate = optimizer_fa.param_groups[0]["lr"]
    dem_losses, task_losses, alignment_losses = [], [], []

    progress = tqdm(
        range(target_steps),
        desc=f"Feature alignment {epoch + 1}/{config.epochs}",
    )
    for _ in progress:
        iterator, inner_polar, inner_target = _next_cropped_batch(
            iterator, dataloader, device, config.img_size
        )
        inner_prediction, _, dem_features = _demosaicking_loss(
            dem_model,
            inner_polar,
            interpolator,
            equivariant_interpolator,
        )
        _, task_features, _, _ = task_model(
            inter_data_process(inner_prediction), inner_target
        )
        alignment_loss = feature_alignment_loss(fa_model, dem_features, task_features)
        dem_parameters = _updated_parameters(
            dem_model, alignment_loss, learning_rate, retain_graph=True
        )
        task_parameters = _updated_parameters(
            task_model, alignment_loss, learning_rate, retain_graph=True
        )

        iterator, outer_polar, outer_target = _next_cropped_batch(
            iterator, dataloader, device, config.img_size
        )
        outer_prediction, dem_loss, _ = _demosaicking_loss(
            dem_model,
            outer_polar,
            interpolator,
            equivariant_interpolator,
            parameters=dem_parameters,
        )
        task_prediction, _, _, _ = functional_call(
            task_model,
            task_parameters,
            (inter_data_process(outer_prediction), outer_target),
        )
        task_objective = criterion(task_prediction, outer_target)
        outer_loss = dem_loss + config.task_loss_weight * task_objective
        optimizer_fa.zero_grad(set_to_none=True)
        dem_model.zero_grad(set_to_none=True)
        task_model.zero_grad(set_to_none=True)
        outer_loss.backward()
        torch.nn.utils.clip_grad_norm_(fa_model.parameters(), max_norm=1.0)
        optimizer_fa.step()
        dem_model.zero_grad(set_to_none=True)
        task_model.zero_grad(set_to_none=True)

        dem_losses.append(dem_loss.item())
        task_losses.append(task_objective.item())
        alignment_losses.append(alignment_loss.item())
        progress.set_postfix(
            dem=f"{_mean(dem_losses):.4f}",
            task=f"{_mean(task_losses):.4f}",
            alignment=f"{_mean(alignment_losses):.6f}",
        )


def train_joint(
    config,
    dataloader,
    dem_model,
    task_model,
    fa_model,
    optimizer_dem,
    optimizer_task,
    criterion,
    interpolator,
    equivariant_interpolator,
    device,
    epoch,
):
    dem_model.train()
    task_model.train()
    fa_model.eval()
    _set_trainable(fa_model, False)
    max_steps = config.max_stage2_step
    total = len(dataloader) if max_steps <= 0 else min(len(dataloader), max_steps)
    dem_losses, task_losses, alignment_losses = [], [], []
    progress = tqdm(total=total, desc=f"Joint training {epoch + 1}/{config.epochs}")

    for index, batch in enumerate(dataloader):
        if max_steps > 0 and index >= max_steps:
            break
        polar = batch["polar"].to(device, non_blocking=device.type == "cuda")
        target = batch["gt_rgb"].to(device, non_blocking=device.type == "cuda")
        polar, target = _random_crop(polar, target, size=config.img_size)
        prediction, dem_loss, dem_features = _demosaicking_loss(
            dem_model,
            polar,
            interpolator,
            equivariant_interpolator,
        )
        task_prediction, task_features, _, _ = task_model(
            inter_data_process(prediction), target
        )
        task_objective = criterion(task_prediction, target)
        alignment_loss = feature_alignment_loss(fa_model, dem_features, task_features)
        loss = (
            dem_loss
            + config.task_loss_weight * task_objective
            + config.fa_loss_weight * alignment_loss
        )
        optimizer_dem.zero_grad(set_to_none=True)
        optimizer_task.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dem_model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(task_model.training_parameters(), max_norm=1.0)
        optimizer_dem.step()
        optimizer_task.step()

        dem_losses.append(dem_loss.item())
        task_losses.append(task_objective.item())
        alignment_losses.append(alignment_loss.item())
        progress.update(1)
        progress.set_postfix(
            dem=f"{_mean(dem_losses):.4f}",
            task=f"{_mean(task_losses):.4f}",
            alignment=f"{_mean(alignment_losses):.6f}",
        )
    progress.close()


def _clean_rgb_from_polar(polar):
    return sum(torch.chunk(polar, 4, dim=1)) / 4.0


def refine_task(
    config,
    dataloader,
    dem_model,
    task_model,
    fa_model,
    optimizer_task,
    criterion,
    device,
    epoch,
):
    """Refine TaskNet while keeping the demosaicker and FANet frozen."""

    max_steps = config.max_refine_step
    if max_steps <= 0:
        return
    dem_model.eval()
    task_model.train()
    fa_model.eval()
    _set_trainable(fa_model, False)
    iterator = iter(dataloader)
    task_losses, alignment_losses = [], []
    progress = tqdm(
        range(max_steps),
        desc=f"Task refinement {epoch + 1}/{config.epochs}",
    )

    for _ in progress:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        reflected, clean = _random_crop(
            batch["polar"].to(device, non_blocking=device.type == "cuda"),
            batch["gt_polar"].to(device, non_blocking=device.type == "cuda"),
            size=config.img_size,
        )
        with torch.no_grad():
            restored_reflected, dem_features = dem_model(reflected, ELT_state=False)
            restored_clean, _ = dem_model(clean, ELT_state=False)
            pseudo_reference = _clean_rgb_from_polar(restored_clean).clamp(0, 1)
        prediction, task_features, _, _ = task_model(
            inter_data_process(restored_reflected), pseudo_reference
        )
        task_objective = criterion(prediction, pseudo_reference)
        alignment_loss = feature_alignment_loss(fa_model, dem_features, task_features)
        loss = (
            config.task_loss_weight * task_objective
            + config.fa_loss_weight * alignment_loss
        )
        optimizer_task.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(task_model.training_parameters(), max_norm=1.0)
        optimizer_task.step()

        task_losses.append(task_objective.item())
        alignment_losses.append(alignment_loss.item())
        progress.set_postfix(
            task=f"{_mean(task_losses):.4f}",
            alignment=f"{_mean(alignment_losses):.6f}",
        )


def _build_optimizers(config, dem_model, task_model, fa_model):
    optimizers = [
        optim.Adam(dem_model.parameters(), lr=config.lr),
        optim.Adam(task_model.training_parameters(), lr=config.lr),
        optim.Adam(fa_model.parameters(), lr=config.lr),
    ]
    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(config.epochs, 1), eta_min=1e-6
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
    checkpoint_models = (models[0], models[1].net_g, models[2])
    checkpoints = []
    for prefix, folder, model, optimizer, scheduler in zip(
        prefixes, folders, checkpoint_models, optimizers, schedulers
    ):
        path = find_checkpoint(folder, prefix, getattr(config, "resume_epoch", None))
        checkpoints.append(load_checkpoint(model, path, device, optimizer, scheduler))
    start_epoch = int(checkpoints[0].get("epoch", -1)) + 1
    print(f"Resuming from epoch {start_epoch + 1}")
    return start_epoch


def _load_all_checkpoints(config, dem_model, task_model, fa_model, device):
    epoch = getattr(config, "eval_epoch", None)
    load_inference_checkpoints(
        dem_model, task_model, config.eval_ckpt_dir, device, epoch
    )
    fa_path = find_checkpoint(Path(config.eval_ckpt_dir) / "FANet", "FANet", epoch)
    load_checkpoint(fa_model, fa_path, device)


def fit(config, device):
    loaders = build_training_loaders(config)
    dem_model, task_model, fa_model = build_models(
        config.dm_config_path,
        config.polarfree_checkpoint_dir,
        device,
        load_generator=not config.resume_state,
    )
    optimizers, schedulers = _build_optimizers(
        config, dem_model, task_model, fa_model
    )
    start_epoch = _resume_training(
        config,
        (dem_model, task_model, fa_model),
        optimizers,
        schedulers,
        device,
    )
    optimizer_dem, optimizer_task, optimizer_fa = optimizers
    scheduler_dem, scheduler_task, scheduler_fa = schedulers
    criterion = DfPTaskLoss(config.perceptual_weight).to(device)
    interpolator = init_interp(phase="train").to(device)
    equivariant_interpolator = init_interp(phase="train").to(device)
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
            criterion,
            interpolator,
            equivariant_interpolator,
            device,
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
            criterion,
            interpolator,
            equivariant_interpolator,
            device,
            epoch,
        )
        refine_task(
            config,
            loaders["refine"],
            dem_model,
            task_model,
            fa_model,
            optimizer_task,
            criterion,
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
                    ("TaskNet", task_model.net_g, optimizer_task, scheduler_task),
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
                full_metrics=config.full_metrics,
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
                full_metrics=config.full_metrics,
            )


def evaluate_only(config, device):
    dem_model, task_model, _ = build_models(
        config.dm_config_path,
        config.polarfree_checkpoint_dir,
        device,
        load_generator=False,
        include_alignment=False,
    )
    load_inference_checkpoints(
        dem_model,
        task_model,
        config.eval_ckpt_dir,
        device,
        getattr(config, "eval_epoch", None),
    )
    dataloader = build_eval_loader(
        config.test_data_path,
        config.test_batch_size,
        config.seed,
        config.workers,
        getattr(config, "eval_limit", None),
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
        full_metrics=config.full_metrics,
    )


def refine_only(config, device):
    loaders = build_training_loaders(config)
    dem_model, task_model, fa_model = build_models(
        config.dm_config_path,
        config.polarfree_checkpoint_dir,
        device,
        load_generator=False,
    )
    _load_all_checkpoints(config, dem_model, task_model, fa_model, device)
    optimizer_task = optim.Adam(task_model.training_parameters(), lr=config.lr)
    scheduler_task = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_task, T_max=1, eta_min=1e-6
    )
    criterion = DfPTaskLoss(config.perceptual_weight).to(device)
    checkpoint_epoch = int(getattr(config, "eval_epoch", 0) or 0)
    refine_task(
        config,
        loaders["refine"],
        dem_model,
        task_model,
        fa_model,
        optimizer_task,
        criterion,
        device,
        0,
    )
    scheduler_task.step()
    save_training_checkpoints(
        config.save_ckpt_dir,
        checkpoint_epoch,
        (("TaskNet", task_model.net_g, optimizer_task, scheduler_task),),
    )


def run(config):
    set_seed(config.seed)
    device = get_device(getattr(config, "device", None))
    print(f"Using device: {device}")
    if getattr(config, "eval_only", False):
        return evaluate_only(config, device)
    if getattr(config, "refine_only", False):
        return refine_only(config, device)
    return fit(config, device)
