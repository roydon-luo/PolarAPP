import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

from polarapp.losses import ssim
from polarapp.operations import inter_data_process


def predict_batch(polar, dem_model, task_model):
    restored, _ = dem_model(polar, ELT_state=False)
    prediction, _, _, _ = task_model(inter_data_process(restored), phase="val")
    return restored, prediction.clamp(0, 1)


def _psnr(prediction, target):
    mse = F.mse_loss(prediction, target).item()
    return float("inf") if mse == 0 else -10 * math.log10(mse)


def _optional_metrics(device):
    metrics = {}
    try:
        import lpips

        metrics["lpips"] = lpips.LPIPS(net="alex").to(device).eval()
    except Exception as error:
        print(f"LPIPS unavailable: {error}")
    try:
        import pyiqa

        metrics["musiq"] = pyiqa.create_metric("musiq", device=device)
    except Exception as error:
        print(f"MUSIQ unavailable: {error}")
    return metrics


def _append_metrics(path, epoch, tag, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("epoch", "tag", *metrics.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow({"epoch": "" if epoch is None else epoch, "tag": tag, **metrics})


def evaluate(
    dataloader,
    dem_model,
    task_model,
    device,
    output_dir,
    tag="evaluation",
    epoch=None,
    log_path=None,
    full_metrics=False,
):
    dem_model.eval()
    task_model.eval()
    output_dir = Path(output_dir) / tag
    optional = _optional_metrics(device) if full_metrics else {}
    rows = []
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating DfP"):
            polar = batch["polar"].to(device, non_blocking=device.type == "cuda")
            target = batch["gt_rgb"].to(device, non_blocking=device.type == "cuda")
            _, prediction = predict_batch(polar, dem_model, task_model)
            evaluated = F.interpolate(
                prediction, size=target.shape[-2:], mode="bilinear", align_corners=False
            ).clamp(0, 1)
            for index, (scene, prefix) in enumerate(zip(batch["scene"], batch["prefix"])):
                scene_dir = output_dir / scene
                scene_dir.mkdir(parents=True, exist_ok=True)
                save_image(prediction[index].cpu(), scene_dir / f"{prefix}_rgb.png")
                pred_item = evaluated[index : index + 1]
                target_item = target[index : index + 1]
                row = {
                    "scene": scene,
                    "prefix": prefix,
                    "psnr": _psnr(pred_item, target_item),
                    "ssim": ssim(pred_item, target_item).item(),
                }
                if "lpips" in optional:
                    row["lpips"] = optional["lpips"](
                        pred_item * 2 - 1, target_item * 2 - 1
                    ).mean().item()
                if "musiq" in optional:
                    row["musiq"] = optional["musiq"](pred_item).mean().item()
                rows.append(row)
    if not rows:
        raise RuntimeError("No DfP samples were evaluated")
    metric_names = [name for name in ("psnr", "ssim", "lpips", "musiq") if name in rows[0]]
    averages = {name: sum(row[name] for row in rows) / len(rows) for name in metric_names}
    print("Evaluation results: " + " ".join(f"{k.upper()}={v:.4f}" for k, v in averages.items()))
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("scene", "prefix", *metric_names))
        writer.writeheader()
        writer.writerows(rows)
    if log_path:
        _append_metrics(log_path, epoch, tag, averages)
    return averages


def infer(dataloader, dem_model, task_model, device, output_dir, full_metrics=True):
    return evaluate(
        dataloader,
        dem_model,
        task_model,
        device,
        output_dir,
        tag="inference",
        full_metrics=full_metrics,
    )

