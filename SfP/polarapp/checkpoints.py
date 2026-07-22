from pathlib import Path

import torch


def find_checkpoint(folder, prefix):
    path = Path(folder) / f"{prefix}.pth"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_checkpoint(model, path, device, optimizer=None, scheduler=None):
    print(f"Loading checkpoint: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def load_inference_checkpoints(dem_model, task_model, root, device):
    root = Path(root)
    dem_path = find_checkpoint(root / "DemNet", "DemNet")
    task_path = find_checkpoint(root / "TaskNet", "TaskNet")
    load_checkpoint(dem_model, dem_path, device)
    load_checkpoint(task_model, task_path, device)
    return dem_path, task_path


def _save_checkpoint(path, epoch, model, optimizer, scheduler):
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(checkpoint, path)


def save_training_checkpoints(root, epoch, components):
    root = Path(root)
    for name, model, optimizer, scheduler in components:
        _save_checkpoint(
            root / name / f"{name}.pth",
            epoch,
            model,
            optimizer,
            scheduler,
        )
