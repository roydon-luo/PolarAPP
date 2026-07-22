"""Build the PolarAPP-DfP models with explicit device and weight paths."""

from __future__ import annotations

from pathlib import Path

import torch
from archs.diffusion import DiffusionModel
from omegaconf import OmegaConf
from utils.utils_net import FeatureAlignment, PIDNet


def get_device(name: str | None) -> torch.device:
    requested = name or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {requested} requested, but CUDA is unavailable")
        index = device.index or 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {requested} does not exist; found {torch.cuda.device_count()} device(s)"
            )
        device = torch.device(f"cuda:{index}")
        torch.cuda.set_device(device)
    return device


def load_diffusion_config(config_path: Path, checkpoint_dir: Path, load_generator: bool):
    config = OmegaConf.load(Path(config_path))
    checkpoint_dir = Path(checkpoint_dir).resolve()
    config.path.pretrain_network_le = None
    config.path.pretrain_network_le_dm = str(checkpoint_dir / "net_le_dm_latest.pth")
    config.path.pretrain_network_d = str(checkpoint_dir / "net_d_latest.pth")
    config.path.pretrain_network_g = (
        str(checkpoint_dir / "net_g_latest.pth") if load_generator else None
    )
    required = [
        Path(config.path.pretrain_network_le_dm),
        Path(config.path.pretrain_network_d),
    ]
    if load_generator:
        required.append(Path(config.path.pretrain_network_g))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing PolarFree checkpoint(s): " + ", ".join(missing))
    return config


def build_models(
    config_path: Path,
    polarfree_checkpoint_dir: Path,
    device: torch.device,
    load_generator: bool,
    include_alignment: bool = True,
):
    config = load_diffusion_config(
        config_path, polarfree_checkpoint_dir, load_generator=load_generator
    )
    demosaicker = PIDNet().to(device)
    task_model = DiffusionModel(config, device=device).to(device)
    feature_alignment = FeatureAlignment().to(device) if include_alignment else None
    return demosaicker, task_model, feature_alignment
