from pathlib import Path

import torch

STATE_KEYS = ("model_state_dict", "params_ema", "params", "state_dict")


def _load_file(path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_state(checkpoint, path="checkpoint"):
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a checkpoint dictionary in {path}")
    for key in STATE_KEYS:
        state = checkpoint.get(key)
        if isinstance(state, dict):
            checkpoint = state
            break
    state = {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
        if isinstance(key, str) and torch.is_tensor(value)
    }
    if not state:
        raise ValueError(f"No model state dictionary found in {path}")
    return state


def load_state(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return extract_state(_load_file(path), path)


def find_checkpoint(folder, prefix):
    path = Path(folder) / f"{prefix}.pth"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_checkpoint(model, path, device, optimizer=None, scheduler=None):
    print(f"Loading checkpoint: {path}")
    checkpoint = _load_file(path, device)
    model.load_state_dict(extract_state(checkpoint, path), strict=True)
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
    load_checkpoint(task_model.net_g, task_path, device)
    return dem_path, task_path


def _save_checkpoint(path, epoch, model, optimizer, scheduler):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path,
    )


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
