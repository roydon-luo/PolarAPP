"""Polarization features and equivalent-imaging transformations for DfP."""

from __future__ import annotations

import random

import kornia
import torch


def _rgb_to_gray(image: torch.Tensor) -> torch.Tensor:
    red, green, blue = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    return 0.299 * red + 0.587 * green + 0.114 * blue


def inter_data_process(polarization: torch.Tensor) -> list[torch.Tensor]:
    """Build the PolarFree RGB/intensity/AoP/DoLP input list."""

    image_0, image_45, image_90, image_135 = torch.chunk(polarization, 4, dim=1)
    rgb = torch.clamp((image_0 + image_45 + image_90 + image_135) / 4, 0, 1)
    gray = [_rgb_to_gray(image) for image in (image_0, image_45, image_90, image_135)]
    gray_0, gray_45, gray_90, gray_135 = gray
    s0 = (gray_0 + gray_45 + gray_90 + gray_135) / 2
    s1 = gray_0 - gray_90
    s2 = gray_45 - gray_135
    dolp = torch.clamp(
        torch.sqrt(s1.square() + s2.square() + 1e-5) / (s0 + 1e-5), 0, 1
    )
    aop = 0.5 * torch.atan2(s2 + 1e-5, s1 + 1e-5)
    return [rgb, gray_0, gray_45, gray_90, gray_135, aop, dolp]


class EITransformer:
    """Generate translation, 10-degree rotation, and flip variants."""

    def __init__(self, shift_count: int = 1, rotation_count: int = 1, flip_count: int = 1):
        self.shift_count = shift_count
        self.rotation_count = rotation_count
        self.flip_count = flip_count

    def apply(self, image: torch.Tensor) -> torch.Tensor:
        variants = [image]
        if self.shift_count:
            variants.append(_random_shift(image, self.shift_count))
        if self.rotation_count:
            variants.append(_rotate(image, self.rotation_count))
        if self.flip_count:
            variants.append(_flip(image, self.flip_count))
        return torch.cat(variants, dim=0)


def _random_shift(image: torch.Tensor, count: int) -> torch.Tensor:
    height, width = image.shape[-2:]
    if count > min(height, width) - 1:
        raise ValueError("Shift count must be smaller than the image dimensions")
    rows = random.sample(list(range(1 - height, 0)) + list(range(1, height)), count)
    columns = random.sample(list(range(1 - width, 0)) + list(range(1, width)), count)
    return torch.cat(
        [
            torch.roll(image, shifts=(row, column), dims=(-2, -1))
            for row, column in zip(rows, columns)
        ],
        dim=0,
    )


def _rotate(image: torch.Tensor, count: int) -> torch.Tensor:
    step = max(1, 360 // count)
    angles = range(10, 360, step)
    return torch.cat(
        [
            kornia.geometry.transform.rotate(image, image.new_tensor([angle]))
            for angle in angles
        ],
        dim=0,
    )


def _flip(image: torch.Tensor, count: int) -> torch.Tensor:
    if count not in (1, 2, 3):
        raise ValueError("Flip count must be between 1 and 3")
    variants = [kornia.geometry.transform.hflip(image)]
    if count >= 2:
        vertical = kornia.geometry.transform.vflip(image)
        variants.append(vertical)
    if count == 3:
        variants.append(kornia.geometry.transform.hflip(vertical))
    return torch.cat(variants, dim=0)

