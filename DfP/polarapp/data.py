from Dataloader.dataloader import PolaRGBDataset, PolaRGBInferenceDataset
from torch.utils.data import DataLoader, Subset

from polarapp.config import seed_worker


def _loader(dataset, batch_size, shuffle, seed, workers):
    import torch

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle and len(dataset) >= batch_size,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_eval_loader(path, batch_size, seed, workers=0, limit=None):
    dataset = PolaRGBDataset(path, "test")
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    return _loader(dataset, batch_size, False, seed, workers)


def build_training_loaders(config):
    train_data = PolaRGBDataset(config.train_data_path, "train")
    return {
        "meta": _loader(
            train_data, config.meta_batch_size, True, config.seed, config.workers
        ),
        "train": _loader(
            train_data,
            config.train_batch_size,
            True,
            config.seed + 1,
            config.workers,
        ),
        "refine": _loader(
            train_data,
            config.refine_batch_size,
            True,
            config.seed + 2,
            config.workers,
        ),
        "val": build_eval_loader(
            config.val_data_path,
            config.val_batch_size,
            config.seed + 3,
            config.workers,
            getattr(config, "val_limit", None),
        ),
        "test": build_eval_loader(
            config.test_data_path,
            config.test_batch_size,
            config.seed + 4,
            config.workers,
            getattr(config, "test_limit", None),
        ),
    }


def build_inference_loader(path, batch_size, workers=0, limit=None):
    dataset = PolaRGBInferenceDataset(path)
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    return _loader(dataset, batch_size, False, 0, workers)
