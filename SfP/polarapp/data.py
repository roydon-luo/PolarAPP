from pathlib import Path

import torch
from Dataloader.dataloader import PolarizationDataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from polarapp.config import seed_worker

POLARIZATION_FOLDERS = ("pol000", "pol045", "pol090", "pol135")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _loader(dataset, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_eval_loader(path, batch_size, seed, center_crop=None):
    transform_list = []
    if center_crop:
        transform_list.append(transforms.CenterCrop(center_crop))
    transform_list.append(transforms.ToTensor())
    dataset = PolarizationDataset(path, transform=transforms.Compose(transform_list))
    return _loader(dataset, batch_size, False, seed)


def build_training_loaders(config):
    train_transform = transforms.ToTensor()
    train_data = PolarizationDataset(
        config.train_data_path,
        transform=train_transform,
        crop_size=getattr(config, "dataset_crop_size", None),
        crop_alpha_only=getattr(config, "dataset_crop_alpha_only", False),
    )
    loaders = {
        "meta": _loader(train_data, config.meta_batch_size, True, config.seed),
        "train": _loader(train_data, config.train_batch_size, True, config.seed + 1),
        "refine": _loader(train_data, config.refine_batch_size, True, config.seed + 2),
        "val": build_eval_loader(
            config.val_data_path,
            config.val_batch_size,
            config.seed + 3,
            center_crop=512,
        ),
        "test": build_eval_loader(
            config.test_data_path, config.test_batch_size, config.seed + 4
        ),
    }
    return loaders


class PolarizationInputDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.to_tensor = transforms.ToTensor()
        reference = self.root_dir / POLARIZATION_FOLDERS[0]
        if not reference.is_dir():
            raise FileNotFoundError(f"Missing input folder: {reference}")
        self.samples = sorted(
            path.stem
            for path in reference.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.samples:
            raise ValueError(f"No polarization images found in {reference}")

    def _find_image(self, folder, stem):
        for suffix in IMAGE_SUFFIXES:
            path = self.root_dir / folder / f"{stem}{suffix}"
            if path.exists():
                return path
        raise FileNotFoundError(f"Missing {folder} image for sample {stem}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        stem = self.samples[index]
        images = [
            Image.open(self._find_image(folder, stem)).convert("RGB")
            for folder in POLARIZATION_FOLDERS
        ]
        if len({image.size for image in images}) != 1:
            raise ValueError(
                f"Polarization images have different sizes for sample {stem}"
            )
        return torch.cat([self.to_tensor(image) for image in images], dim=0), stem


def build_inference_loader(path, batch_size):
    return DataLoader(
        PolarizationInputDataset(path),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
