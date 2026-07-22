"""Demosaicking, DfP task, and feature-alignment objectives."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.vgg_arch import VGGFeatureExtractor


def _image_gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sobel_x = image.new_tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = image.new_tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_x = sobel_x.expand(image.shape[1], 1, 3, 3)
    sobel_y = sobel_y.expand(image.shape[1], 1, 3, 3)
    return (
        F.conv2d(image, sobel_x, padding=1, groups=image.shape[1]),
        F.conv2d(image, sobel_y, padding=1, groups=image.shape[1]),
    )


def _gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_x, pred_y = _image_gradients(prediction)
    target_x, target_y = _image_gradients(target)
    return 0.8 * (F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)) + 0.4 * F.l1_loss(
        prediction, target
    )


def ssim(
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    coordinates = torch.arange(window_size, device=image_a.device, dtype=image_a.dtype)
    gaussian = torch.exp(-((coordinates - window_size // 2) ** 2) / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    window = torch.outer(gaussian, gaussian).expand(
        image_a.shape[1], 1, window_size, window_size
    )
    mean_a = F.conv2d(image_a, window, padding=window_size // 2, groups=image_a.shape[1])
    mean_b = F.conv2d(image_b, window, padding=window_size // 2, groups=image_b.shape[1])
    mean_a_sq, mean_b_sq, mean_ab = mean_a.square(), mean_b.square(), mean_a * mean_b
    variance_a = (
        F.conv2d(image_a.square(), window, padding=window_size // 2, groups=image_a.shape[1])
        - mean_a_sq
    )
    variance_b = (
        F.conv2d(image_b.square(), window, padding=window_size // 2, groups=image_b.shape[1])
        - mean_b_sq
    )
    covariance = (
        F.conv2d(image_a * image_b, window, padding=window_size // 2, groups=image_a.shape[1])
        - mean_ab
    )
    numerator = (2 * mean_ab + 0.01**2) * (2 * covariance + 0.03**2)
    denominator = (mean_a_sq + mean_b_sq + 0.01**2) * (
        variance_a + variance_b + 0.03**2
    )
    return (numerator / denominator).mean()


def polar_loss(
    output: torch.Tensor, target: torch.Tensor, device: str | None = None
) -> torch.Tensor:
    """Composite polarization reconstruction loss used by the demosaicker."""

    del device  # Retained for compatibility with the original training call.
    output_0, output_45, output_90, output_135 = torch.chunk(output, 4, dim=1)
    target_0, target_45, target_90, target_135 = torch.chunk(target, 4, dim=1)
    output_s0 = (output_0 + output_45 + output_90 + output_135) / 2
    output_s1, output_s2 = output_0 - output_90, output_45 - output_135
    target_s0 = (target_0 + target_45 + target_90 + target_135) / 2
    target_s1, target_s2 = target_0 - target_90, target_45 - target_135
    epsilon = 1e-5
    output_aop = torch.atan2(output_s2 + epsilon, output_s1 + epsilon) / 2
    target_aop = torch.atan2(target_s2 + epsilon, target_s1 + epsilon) / 2
    output_dolp = torch.clamp(
        torch.sqrt(output_s1.square() + output_s2.square() + epsilon)
        / (output_s0 + epsilon),
        0,
        1,
    )
    target_dolp = torch.clamp(
        torch.sqrt(target_s1.square() + target_s2.square() + epsilon)
        / (target_s0 + epsilon),
        0,
        1,
    )
    gradient = (
        _gradient_loss(output, target)
        + _gradient_loss(output_s0, target_s0)
        + 10 * _gradient_loss(output_s1, target_s1)
        + 10 * _gradient_loss(output_s2, target_s2)
    )
    stokes = F.l1_loss(output_s1, target_s1) + F.l1_loss(output_s2, target_s2)
    angular = F.l1_loss(output_aop, target_aop) + 10 * F.l1_loss(
        output_dolp, target_dolp
    )
    physical = F.l1_loss(output_0 + output_90, output_45 + output_135)
    structural = 4 - sum(
        ssim(prediction, ground_truth)
        for prediction, ground_truth in (
            (output, target),
            (output_s0, target_s0),
            (output_s1, target_s1),
            (output_s2, target_s2),
        )
    )
    dolp_structural = 1 - ssim(output_dolp, target_dolp)
    return 0.1 * gradient + 10 * stokes + angular + physical + structural + 2 * dolp_structural


class PhaseLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor) -> torch.Tensor:
        gray_a = image_a.mean(dim=1, keepdim=True)
        gray_b = image_b.mean(dim=1, keepdim=True)
        phase_a = torch.angle(torch.fft.fft2(gray_a + self.epsilon))
        phase_b = torch.angle(torch.fft.fft2(gray_b + self.epsilon))
        return F.mse_loss(phase_a, phase_b)


class PerceptualLoss(nn.Module):
    def __init__(self, weight: float) -> None:
        super().__init__()
        self.weight = weight
        self.extractor = VGGFeatureExtractor(
            layer_name_list=["conv5_4"],
            vgg_type="vgg19",
            use_input_norm=True,
            range_norm=True,
        )
        self.extractor.requires_grad_(False)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction_features = self.extractor(prediction)
        target_features = self.extractor(target.detach())
        return self.weight * F.l1_loss(
            prediction_features["conv5_4"], target_features["conv5_4"]
        )


class DfPTaskLoss(nn.Module):
    """PolarFree task loss with one reusable VGG feature extractor."""

    def __init__(self, perceptual_weight: float = 0.1) -> None:
        super().__init__()
        self.phase = PhaseLoss()
        self.perceptual = (
            PerceptualLoss(perceptual_weight) if perceptual_weight > 0 else None
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.l1_loss(prediction, target) + (1 - ssim(prediction, target))
        loss = loss + self.phase(prediction, target)
        if self.perceptual is not None:
            loss = loss + self.perceptual(prediction, target)
        return loss


def feature_alignment_loss(
    alignment: nn.Module,
    dem_features: dict[str, torch.Tensor],
    task_features: dict[str, torch.Tensor],
) -> torch.Tensor:
    transformed, matched = alignment(dem_features, task_features)
    return sum(
        1
        - F.cosine_similarity(
            transformed[f"TF{level}"], matched[f"MF{level}"], dim=1
        ).mean()
        for level in (1, 2, 3)
    ) / 3
