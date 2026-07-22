import argparse
import random
from types import SimpleNamespace

import numpy as np
import torch

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None


def load_config(path):
    if OmegaConf is not None:
        return OmegaConf.load(path)

    import yaml

    with open(path, "r", encoding="utf-8") as file:
        return SimpleNamespace(**yaml.safe_load(file))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(device_name=None):
    requested = device_name or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {requested} was requested, but CUDA is unavailable."
            )
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {requested} does not exist; found {torch.cuda.device_count()} device(s)."
            )
        torch.cuda.set_device(index)
        device = torch.device(f"cuda:{index}")
    return device


def build_train_parser():
    parser = argparse.ArgumentParser(description="Train or evaluate PolarAPP.")
    parser.add_argument("--config", default="./configs/train_config.yaml")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--save-ckpt-dir", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-tag", default=None)
    parser.add_argument("--eval-log-path", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-meta-step", type=int, default=None)
    parser.add_argument("--max-stage2-step", type=int, default=None)
    parser.add_argument("--max-refine-step", type=int, default=None)
    parser.add_argument("--fa-loss-weight", type=float, default=None)
    parser.add_argument("--task-loss-weight", type=float, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--meta-batch-size", type=int, default=None)
    parser.add_argument("--refine-batch-size", type=int, default=None)
    parser.add_argument("--save-freq", type=int, default=None)
    parser.add_argument("--val-freq", type=int, default=None)
    parser.add_argument("--test-freq", type=int, default=None)
    return parser


def apply_train_overrides(config, args):
    if args.eval_only:
        config.eval_only = True
        config.resume_state = False
    if args.ckpt_dir is not None:
        config.eval_ckpt_dir = args.ckpt_dir
        config.Dem_pretain_path = f"{args.ckpt_dir}/DemNet"
        config.Task_pretain_path = f"{args.ckpt_dir}/TaskNet"
        config.FA_pretain_path = f"{args.ckpt_dir}/FANet"
    if args.save_ckpt_dir is not None:
        config.save_ckpt_dir = args.save_ckpt_dir
    if args.data_path is not None:
        config.test_data_path = args.data_path
    if args.device is not None:
        config.device = args.device
    if args.eval_tag is not None:
        config.eval_tag = args.eval_tag
    if args.eval_log_path is not None:
        config.eval_log_path = args.eval_log_path

    overrides = {
        "epochs": args.epochs,
        "max_meta_step": args.max_meta_step,
        "max_stage2_step": args.max_stage2_step,
        "max_refine_step": args.max_refine_step,
        "fa_loss_weight": args.fa_loss_weight,
        "task_loss_weight": args.task_loss_weight,
        "train_batch_size": args.train_batch_size,
        "meta_batch_size": args.meta_batch_size,
        "refine_batch_size": args.refine_batch_size,
        "save_freq": args.save_freq,
        "val_freq": args.val_freq,
        "test_freq": args.test_freq,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(config, name, value)
    return config


def parse_train_config(argv=None):
    args = build_train_parser().parse_args(argv)
    return apply_train_overrides(load_config(args.config), args)
