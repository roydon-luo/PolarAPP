from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class PolarizationDataset(Dataset):
    """Load aligned polarization images, surface normals, and masks."""

    polarization_folders = ("pol000", "pol045", "pol090", "pol135")
    image_suffixes = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    def __init__(
        self,
        root_dir,
        transform=None,
        color_mode="RGB",
        normal_color_mode="RGB",
        mask_color_mode="L",
        crop_size=None,
        crop_alpha_only=False,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.color_mode = color_mode
        self.normal_color_mode = normal_color_mode
        self.mask_color_mode = mask_color_mode
        self.crop_size = crop_size
        self.crop_alpha_only = crop_alpha_only

        for name, mode in (
            ("color_mode", color_mode),
            ("normal_color_mode", normal_color_mode),
            ("mask_color_mode", mask_color_mode),
        ):
            if mode not in ("RGB", "L"):
                raise ValueError(f"{name} must be 'RGB' or 'L', got {mode!r}")

        reference_folder = self.root_dir / self.polarization_folders[0]
        if not reference_folder.is_dir():
            raise FileNotFoundError(f"Missing dataset folder: {reference_folder}")
        self.image_bases = sorted(
            path.stem
            for path in reference_folder.iterdir()
            if path.is_file() and path.suffix.lower() in self.image_suffixes
        )
        if not self.image_bases:
            raise ValueError(f"No images found in {reference_folder}")

    def _find_image(self, folder, image_base):
        for suffix in self.image_suffixes:
            path = self.root_dir / folder / f"{image_base}{suffix}"
            if path.exists():
                return path
        raise FileNotFoundError(f"Missing {folder} image for sample {image_base}")

    def _load_image(self, folder, image_base, mode):
        path = self._find_image(folder, image_base)
        try:
            return Image.open(path).convert(mode)
        except OSError as error:
            raise RuntimeError(f"Failed to load image: {path}") from error

    def _center_crop(self, images, image_base):
        if self.crop_size is None:
            return images
        if self.crop_alpha_only and not image_base[:1].isalpha():
            return images

        sizes = {image.size for image in images}
        if len(sizes) != 1:
            raise ValueError(f"Images have different sizes for sample {image_base}")
        width, height = images[0].size
        crop_width = min(self.crop_size, width)
        crop_height = min(self.crop_size, height)
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        box = (left, top, left + crop_width, top + crop_height)
        return [image.crop(box) for image in images]

    def __len__(self):
        return len(self.image_bases)

    def __getitem__(self, index):
        image_base = self.image_bases[index]
        polarization = [
            self._load_image(folder, image_base, self.color_mode)
            for folder in self.polarization_folders
        ]
        normal = self._load_image("normal", image_base, self.normal_color_mode)
        mask = self._load_image("mask", image_base, self.mask_color_mode)
        images = self._center_crop(polarization + [normal, mask], image_base)
        polarization, normal, mask = images[:4], images[4], images[5]

        if self.transform is None:
            raise ValueError("A tensor transform is required for PolarizationDataset")
        polarization = [self.transform(image) for image in polarization]
        normal = self.transform(normal) * 2 - 1
        mask = self.transform(mask)
        return torch.cat(polarization, dim=0), normal, mask, image_base
