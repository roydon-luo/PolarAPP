import torch
import torch.nn.functional as F
from einops import rearrange


def feature_cosine_loss(feature_a, feature_b, margin=0.0):
    feature_a = rearrange(feature_a, "b c h w -> b c (h w)")
    feature_b = rearrange(feature_b, "b c h w -> b c (h w)")
    similarity = F.cosine_similarity(feature_a, feature_b)
    return F.relu(1 - margin - similarity).mean()


def task_loss(prediction, target, mask):
    cosine_error = 1 - torch.sum(prediction * target, dim=1, keepdim=True)
    mae = torch.abs(cosine_error * mask).mean()

    prediction = F.normalize(prediction, p=2, dim=1)
    target = F.normalize(target, p=2, dim=1)
    cosine = F.cosine_similarity(prediction, target, dim=1, eps=1e-8)
    angles = torch.acos(torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7))
    valid = mask.squeeze(1) > 0.5
    angular = angles[valid].mean() if valid.any() else prediction.sum() * 0
    return mae + 2 * angular


def _image_gradients(image):
    sobel_x = image.new_tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = image.new_tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_x = sobel_x.expand(image.shape[1], 1, 3, 3)
    sobel_y = sobel_y.expand(image.shape[1], 1, 3, 3)
    return (
        F.conv2d(image, sobel_x, padding=1, groups=image.shape[1]),
        F.conv2d(image, sobel_y, padding=1, groups=image.shape[1]),
    )


def _gradient_loss(prediction, target):
    pred_x, pred_y = _image_gradients(prediction)
    target_x, target_y = _image_gradients(target)
    image_loss = F.l1_loss(prediction, target)
    return (
        0.8 * (F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y))
        + 0.4 * image_loss
    )


def _ssim(image_a, image_b, window_size=11, sigma=1.5):
    coordinates = torch.arange(window_size, device=image_a.device, dtype=image_a.dtype)
    gaussian = torch.exp(-((coordinates - window_size // 2) ** 2) / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    window = torch.outer(gaussian, gaussian)
    window = window.expand(image_a.shape[1], 1, window_size, window_size)
    mean_a = F.conv2d(
        image_a, window, padding=window_size // 2, groups=image_a.shape[1]
    )
    mean_b = F.conv2d(
        image_b, window, padding=window_size // 2, groups=image_b.shape[1]
    )
    mean_a_sq = mean_a.square()
    mean_b_sq = mean_b.square()
    mean_ab = mean_a * mean_b
    variance_a = (
        F.conv2d(
            image_a.square(), window, padding=window_size // 2, groups=image_a.shape[1]
        )
        - mean_a_sq
    )
    variance_b = (
        F.conv2d(
            image_b.square(), window, padding=window_size // 2, groups=image_b.shape[1]
        )
        - mean_b_sq
    )
    covariance = (
        F.conv2d(
            image_a * image_b,
            window,
            padding=window_size // 2,
            groups=image_a.shape[1],
        )
        - mean_ab
    )
    numerator = (2 * mean_ab + 0.01**2) * (2 * covariance + 0.03**2)
    denominator = (mean_a_sq + mean_b_sq + 0.01**2) * (
        variance_a + variance_b + 0.03**2
    )
    return (numerator / denominator).mean()


def polar_loss(output, target, device=None):
    output_angles = torch.chunk(output, 4, dim=1)
    target_angles = torch.chunk(target, 4, dim=1)
    output_0, output_45, output_90, output_135 = output_angles
    target_0, target_45, target_90, target_135 = target_angles

    output_s0 = sum(output_angles) / 2
    output_s1 = output_0 - output_90
    output_s2 = output_45 - output_135
    target_s0 = sum(target_angles) / 2
    target_s1 = target_0 - target_90
    target_s2 = target_45 - target_135

    epsilon = 1e-6
    output_aop = torch.atan2(output_s2 + epsilon, output_s1 + epsilon) / 2
    target_aop = torch.atan2(target_s2 + epsilon, target_s1 + epsilon) / 2
    output_dop = torch.clamp(
        torch.sqrt(output_s1.square() + output_s2.square() + epsilon)
        / (output_s0 + epsilon),
        0,
        1,
    )
    target_dop = torch.clamp(
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
    aop = F.l1_loss(output_aop, target_aop) + 10 * F.l1_loss(output_dop, target_dop)
    polarization = F.l1_loss(output_0 + output_90, output_45 + output_135)
    ssim = 4 - sum(
        _ssim(prediction, ground_truth)
        for prediction, ground_truth in (
            (output, target),
            (output_s0, target_s0),
            (output_s1, target_s1),
            (output_s2, target_s2),
        )
    )
    dop_ssim = 1 - _ssim(output_dop, target_dop)
    return 0.1 * gradient + 10 * stokes + aop + polarization + 2 * dop_ssim + ssim
