import argparse
import random

import numpy as np
import torch
from omegaconf import OmegaConf


def load_config(path):
    return OmegaConf.load(path)


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
                f"CUDA device {requested} does not exist; "
                f"found {torch.cuda.device_count()} device(s)."
            )
        torch.cuda.set_device(index)
        device = torch.device(f"cuda:{index}")
    return device


def build_train_parser():
    parser = argparse.ArgumentParser(description="Train, refine, or evaluate PolarAPP-DfP.")
    parser.add_argument("--config", default="./configs/train_config.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--eval-only", action="store_true")
    mode.add_argument("--refine-only", action="store_true")
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--save-ckpt-dir", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dm-config", default=None)
    parser.add_argument("--polarfree-checkpoint-dir", default=None)
    parser.add_argument("--eval-tag", default=None)
    parser.add_argument("--eval-log-path", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-meta-step", type=int, default=None)
    parser.add_argument("--max-stage2-step", type=int, default=None)
    parser.add_argument("--max-refine-step", type=int, default=None)
    parser.add_argument("--fa-loss-weight", type=float, default=None)
    parser.add_argument("--task-loss-weight", type=float, default=None)
    parser.add_argument("--perceptual-weight", type=float, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--meta-batch-size", type=int, default=None)
    parser.add_argument("--refine-batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--save-freq", type=int, default=None)
    parser.add_argument("--val-freq", type=int, default=None)
    parser.add_argument("--test-freq", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--full-metrics", action="store_true")
    return parser


def apply_train_overrides(config, args):
    config.eval_only = args.eval_only
    config.refine_only = args.refine_only
    if args.eval_only or args.refine_only:
        config.resume_state = False
    if args.resume:
        config.resume_state = True
    if args.ckpt_dir is not None:
        config.eval_ckpt_dir = args.ckpt_dir
        config.Dem_pretain_path = f"{args.ckpt_dir}/DemNet"
        config.Task_pretain_path = f"{args.ckpt_dir}/TaskNet"
        config.FA_pretain_path = f"{args.ckpt_dir}/FANet"
    if args.save_ckpt_dir is not None:
        config.save_ckpt_dir = args.save_ckpt_dir
    if args.data_path is not None:
        config.train_data_path = args.data_path
        config.val_data_path = args.data_path
        config.test_data_path = args.data_path
    if args.device is not None:
        config.device = args.device
    if args.dm_config is not None:
        config.dm_config_path = args.dm_config
    if args.polarfree_checkpoint_dir is not None:
        config.polarfree_checkpoint_dir = args.polarfree_checkpoint_dir
    if args.eval_tag is not None:
        config.eval_tag = args.eval_tag
    if args.eval_log_path is not None:
        config.eval_log_path = args.eval_log_path
    config.full_metrics = args.full_metrics or getattr(config, "full_metrics", False)
    if args.limit is not None:
        config.eval_limit = args.limit

    overrides = {
        "epochs": args.epochs,
        "max_meta_step": args.max_meta_step,
        "max_stage2_step": args.max_stage2_step,
        "max_refine_step": args.max_refine_step,
        "fa_loss_weight": args.fa_loss_weight,
        "task_loss_weight": args.task_loss_weight,
        "perceptual_weight": args.perceptual_weight,
        "train_batch_size": args.train_batch_size,
        "meta_batch_size": args.meta_batch_size,
        "refine_batch_size": args.refine_batch_size,
        "workers": args.workers,
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
