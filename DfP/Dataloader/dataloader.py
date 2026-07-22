"""Dataset loader for the official Hugging Face PolaRGB layout."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

POLAR_SUFFIXES = ("000", "045", "090", "135")


def _load_rgb(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = Image.open(path).convert("RGB")
    return pil_to_tensor(image).float().div_(255.0)


class PolaRGBDataset(Dataset):
    """Return reflected polarization inputs and the scene-level clean target.

    Each item contains ``polar`` (12 channels), ``rgb`` (the reflected RGB
    capture), ``gt_polar`` (12 clean channels), and ``gt_rgb`` (clean RGB).
    The four clean images are shared by every capture in the same scene.
    """

    def __init__(self, root: str | Path, split: str) -> None:
        self.root = Path(root)
        if split == "train":
            subsets = (self.root / "train" / "easy", self.root / "train" / "hard")
        elif split == "test":
            subsets = (self.root / "test",)
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        self.samples: list[tuple[Path, Path, str, str, str]] = []
        for subset in subsets:
            input_root, gt_root = subset / "input", subset / "gt"
            if not input_root.is_dir() or not gt_root.is_dir():
                continue
            for scene_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
                gt_dir = gt_root / scene_dir.name
                if not gt_dir.is_dir():
                    continue
                gt_prefixes = sorted(
                    path.name.removesuffix("_000.png")
                    for path in gt_dir.glob("*_000.png")
                    if not path.name.endswith(".part")
                )
                if not gt_prefixes:
                    continue
                gt_prefix = gt_prefixes[0]
                prefixes = sorted(
                    path.name.removesuffix("_000.png")
                    for path in scene_dir.glob("*_000.png")
                    if not path.name.endswith(".part")
                )
                for prefix in prefixes:
                    required = [scene_dir / f"{prefix}_{suffix}.png" for suffix in POLAR_SUFFIXES]
                    required.append(scene_dir / f"{prefix}_rgb.png")
                    required.extend(
                        gt_dir / f"{gt_prefix}_{suffix}.png" for suffix in POLAR_SUFFIXES
                    )
                    required.append(gt_dir / f"{gt_prefix}_rgb.png")
                    if all(path.is_file() and path.stat().st_size > 0 for path in required):
                        self.samples.append(
                            (scene_dir, gt_dir, prefix, gt_prefix, scene_dir.name)
                        )

        if not self.samples:
            raise RuntimeError(f"No complete PolaRGB samples found for {split=} under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        input_dir, gt_dir, prefix, gt_prefix, scene = self.samples[index]
        polar = torch.cat(
            [_load_rgb(input_dir / f"{prefix}_{suffix}.png") for suffix in POLAR_SUFFIXES]
        )
        gt_polar = torch.cat(
            [_load_rgb(gt_dir / f"{gt_prefix}_{suffix}.png") for suffix in POLAR_SUFFIXES]
        )
        return {
            "polar": polar,
            "rgb": _load_rgb(input_dir / f"{prefix}_rgb.png"),
            "gt_polar": gt_polar,
            "gt_rgb": _load_rgb(gt_dir / f"{gt_prefix}_rgb.png"),
            "prefix": prefix,
            "scene": scene,
        }
