import random

import cv2
import kornia
import numpy as np
import torch


def visualize_aop_dop(aop, dop):
    aop = (aop + np.pi / 2) / np.pi * 255
    dop = np.clip(dop, 0, 0.3) / 0.3
    aop_map = cv2.applyColorMap(
        cv2.cvtColor(aop.astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.COLORMAP_JET
    )
    dop_map = cv2.applyColorMap(
        cv2.cvtColor((dop * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY),
        cv2.COLORMAP_JET,
    )
    return aop_map, dop_map


def calculate_stokes(image):
    image_0, image_45, image_90, image_135 = torch.chunk(image, 4, dim=0)
    s0 = (image_0 + image_45 + image_90 + image_135) * 0.5
    s1 = image_0 - image_90
    s2 = image_45 - image_135
    dop = torch.sqrt(s1.square() + s2.square()) / (s0 + 1e-8)
    aop = 0.5 * torch.atan2(s2 + 1e-8, s1 + 1e-8)
    return image_0, image_45, image_90, image_135, s0, s1, s2, aop, dop


def get_coordinate(height, width):
    horizontal = (np.tile(np.arange(width), [height, 1]) - 0.5 * width) / (0.5 * width)
    vertical = (np.tile(np.arange(height)[..., None], [1, width]) - 0.5 * height) / (
        0.5 * height
    )
    coordinate = np.stack((horizontal, vertical, np.ones((height, width))), axis=0)
    return torch.from_numpy(coordinate).float()


def save_normal(normal, mask, path):
    encoded = np.clip((normal + 1) * 0.5 * mask, 0, 1)
    cv2.imwrite(str(path), (encoded[..., ::-1] * 65535).astype(np.uint16))


def _rgb_to_gray(image):
    red, green, blue = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    return 0.299 * red + 0.587 * green + 0.114 * blue


def inter_data_process(polarization, coordinate):
    image_0, image_45, image_90, image_135 = torch.chunk(polarization, 4, dim=1)
    image_0 = _rgb_to_gray(image_0)
    image_45 = _rgb_to_gray(image_45)
    image_90 = _rgb_to_gray(image_90)
    image_135 = _rgb_to_gray(image_135)
    s0 = (image_0 + image_45 + image_90 + image_135) / 2
    s1 = image_0 - image_90
    s2 = image_45 - image_135
    dop = torch.clamp(torch.sqrt(s1.square() + s2.square() + 1e-5) / (s0 + 1e-5), 0, 1)
    aop = 0.5 * torch.atan2(s2 + 1e-5, s1 + 1e-5)
    return torch.cat(
        (s0, dop, torch.sin(2 * aop), torch.cos(2 * aop), coordinate), dim=1
    )


class EITransformer:
    def __init__(self, shift_count=1, rotation_count=1, flip_count=1):
        self.shift_count = shift_count
        self.rotation_count = rotation_count
        self.flip_count = flip_count

    def apply(self, image):
        variants = [image]
        if self.shift_count:
            variants.append(_random_shift(image, self.shift_count))
        if self.rotation_count:
            variants.append(_rotate(image, self.rotation_count))
        if self.flip_count:
            variants.append(_flip(image, self.flip_count))
        return torch.cat(variants, dim=0)


def _random_shift(image, count):
    height, width = image.shape[-2:]
    if count > min(height, width) - 1:
        raise ValueError("Shift count must be smaller than the image dimensions")
    row_shifts = random.sample(
        list(range(1 - height, 0)) + list(range(1, height)), count
    )
    column_shifts = random.sample(
        list(range(1 - width, 0)) + list(range(1, width)), count
    )
    return torch.cat(
        [
            torch.roll(image, shifts=(row, column), dims=(-2, -1))
            for row, column in zip(row_shifts, column_shifts)
        ],
        dim=0,
    )


def _rotate(image, count):
    angles = np.arange(10, 360, int(360 / count))
    return torch.cat(
        [
            kornia.geometry.transform.rotate(image, image.new_tensor([angle]))
            for angle in angles
        ],
        dim=0,
    )


def _flip(image, count):
    if count not in (1, 2, 3):
        raise ValueError("Flip count must be between 1 and 3")
    variants = [kornia.geometry.transform.hflip(image)]
    if count >= 2:
        vertical = kornia.geometry.transform.vflip(image)
        variants.append(vertical)
    if count == 3:
        variants.append(kornia.geometry.transform.hflip(vertical))
    return torch.cat(variants, dim=0)


def _named_parameters(module, prefix=""):
    memo = set()
    if hasattr(module, "named_leaves"):
        for name, parameter in module.named_leaves():
            if parameter is not None and parameter not in memo:
                memo.add(parameter)
                yield prefix + ("." if prefix else "") + name, parameter
    for child_name, child in module.named_children():
        child_prefix = prefix + ("." if prefix else "") + child_name
        yield from _named_parameters(child, child_prefix)
    if not hasattr(module, "named_leaves"):
        for name, parameter in module.named_parameters(recurse=False):
            yield prefix + ("." if prefix else "") + name, parameter


def _set_parameter(module, name, parameter):
    if "." not in name:
        setattr(module, name, parameter)
        return
    child_name, remainder = name.split(".", 1)
    _set_parameter(getattr(module, child_name), remainder, parameter)


def sync_meta_parameters(module):
    for name, parameter in _named_parameters(module):
        meta = parameter.detach().clone().requires_grad_(True)
        _set_parameter(module, f"{name}_meta", meta)


def inner_update_from_loss(module, loss, learning_rate, retain_graph=True):
    parameters = list(_named_parameters(module))
    gradients = torch.autograd.grad(
        loss,
        [parameter for _, parameter in parameters],
        create_graph=True,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    for (name, parameter), gradient in zip(parameters, gradients):
        if gradient is not None:
            _set_parameter(module, f"{name}_meta", parameter - learning_rate * gradient)


def clear_meta_parameters(module):
    for name, _ in _named_parameters(module):
        parts = f"{name}_meta".split(".")
        target = module
        for child_name in parts[:-1]:
            target = getattr(target, child_name)
        if hasattr(target, parts[-1]):
            meta = getattr(target, parts[-1])
            if torch.is_tensor(meta):
                setattr(target, parts[-1], meta.detach())


def outer_update(module, learning_rate):
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        return
    max_gradient = max(gradient.abs().max().item() for gradient in gradients)
    if max_gradient == 0:
        return
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.grad is not None:
                parameter.add_(-learning_rate * parameter.grad / max_gradient)
