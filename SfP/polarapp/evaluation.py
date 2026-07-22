import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm
from utils.init_interp import init_interp

from polarapp.operations import (
    calculate_stokes,
    get_coordinate,
    inter_data_process,
    save_normal,
    visualize_aop_dop,
)


def predict_batch(pol, dem_model, task_model, imaging_operator=None):
    demosaicker_input = imaging_operator(pol) if imaging_operator is not None else pol
    height, width = demosaicker_input.shape[-2:]
    coordinate = get_coordinate(2 * height, 2 * width).to(pol.device)
    coordinate = coordinate.unsqueeze(0).expand(pol.shape[0], -1, -1, -1)
    pol_pred, _ = dem_model(demosaicker_input, ELT_state=False)
    normal_pred, _ = task_model(inter_data_process(pol_pred, coordinate))
    return pol_pred, normal_pred


def save_prediction(pol_pred, normal_pred, output_dir, mask=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    i0, i45, i90, i135, _, _, _, aop, dop = calculate_stokes(pol_pred)
    aop_map, dop_map = visualize_aop_dop(
        aop.permute(1, 2, 0).detach().cpu().numpy(),
        dop.permute(1, 2, 0).detach().cpu().numpy(),
    )
    for angle, image in zip(("0", "45", "90", "135"), (i0, i45, i90, i135)):
        save_image(image, output_dir / f"{angle}.png", normalize=False)
    cv2.imwrite(str(output_dir / "AoP.png"), aop_map)
    cv2.imwrite(str(output_dir / "DoP.png"), dop_map)

    normal = normal_pred.permute(1, 2, 0).detach().cpu().numpy()
    if mask is None:
        mask = np.ones_like(normal)
    else:
        mask = np.repeat(mask[..., None], 3, axis=-1)
    save_normal(normal, mask, str(output_dir / "normal.png"))


def _sample_metrics(normal_pred, normal_gt, mask):
    prediction = normal_pred.permute(1, 2, 0).detach().cpu().numpy()
    target = normal_gt.permute(1, 2, 0).detach().cpu().numpy()
    valid = mask.squeeze().detach().cpu().numpy().astype(bool)
    error = np.rad2deg(np.arccos((prediction * target).sum(axis=-1).clip(-1, 1)))
    angles = error[valid]
    return {
        "mean": float(np.mean(angles)),
        "median": float(np.median(angles)),
        "rmse": float(np.sqrt(np.mean(angles**2))),
        "acc_11_25": float(np.mean(angles < 11.25)),
        "acc_22_50": float(np.mean(angles < 22.5)),
        "acc_30_00": float(np.mean(angles < 30.0)),
    }


def _average_metrics(samples):
    return {
        key: float(np.mean([sample[key] for sample in samples])) for key in samples[0]
    }


def _print_metrics(metrics):
    print("Evaluation results:")
    print(f"Mean angle error: {metrics['mean']:.4f}")
    print(f"Median angle error: {metrics['median']:.4f}")
    print(f"RMSE angle error: {metrics['rmse']:.4f}")
    print(f"11.25 degree accuracy: {metrics['acc_11_25'] * 100:.2f}%")
    print(f"22.5 degree accuracy: {metrics['acc_22_50'] * 100:.2f}%")
    print(f"30 degree accuracy: {metrics['acc_30_00'] * 100:.2f}%")


def _append_metrics(path, epoch, tag, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("epoch", "tag", *metrics.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow({"epoch": epoch, "tag": tag, **metrics})


def evaluate(
    dataloader,
    dem_model,
    task_model,
    device,
    output_dir,
    tag="evaluation",
    epoch=None,
    log_path=None,
):
    dem_model.eval()
    task_model.eval()
    sample_results = []
    output_dir = Path(output_dir) / tag
    imaging_operator = init_interp(phase="train").to(device)

    with torch.inference_mode():
        for pol, normal, mask, names in tqdm(dataloader, desc="Evaluating"):
            pol = pol.to(device)
            normal = normal.to(device)
            mask = mask.to(device)
            pol_pred, normal_pred = predict_batch(
                pol, dem_model, task_model, imaging_operator
            )
            normal_gt = F.normalize(normal, p=2, dim=1)
            if normal_pred.shape[-2:] != normal_gt.shape[-2:]:
                raise RuntimeError(
                    "Evaluation output must match native task GT resolution: "
                    f"prediction={tuple(normal_pred.shape[-2:])}, "
                    f"GT={tuple(normal_gt.shape[-2:])}"
                )
            for index, name in enumerate(names):
                sample_results.append(
                    _sample_metrics(
                        normal_pred[index], normal_gt[index], mask[index]
                    )
                )
                sample_mask = mask[index].squeeze().detach().cpu().numpy()
                save_prediction(
                    pol_pred[index],
                    normal_pred[index],
                    output_dir / name,
                    sample_mask,
                )

    metrics = _average_metrics(sample_results)
    _print_metrics(metrics)
    if log_path:
        _append_metrics(log_path, "" if epoch is None else epoch, tag, metrics)
    return metrics


def infer(dataloader, dem_model, task_model, device, output_dir):
    dem_model.eval()
    task_model.eval()
    output_dir = Path(output_dir)
    with torch.inference_mode():
        for pol, names in tqdm(dataloader, desc="Inferring"):
            pol_pred, normal_pred = predict_batch(pol.to(device), dem_model, task_model)
            for index, name in enumerate(names):
                save_prediction(pol_pred[index], normal_pred[index], output_dir / name)
    return output_dir
