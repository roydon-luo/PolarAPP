import argparse

from polarapp.checkpoints import load_inference_checkpoints
from polarapp.config import get_device
from polarapp.data import build_inference_loader
from polarapp.evaluation import infer
from polarapp.models import build_models


def build_parser():
    parser = argparse.ArgumentParser(description="Run PolarAPP-DfP on PolaRGB inputs.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--ckpt-dir", default="./experiments/checkpoints/polarapp")
    parser.add_argument(
        "--polarfree-checkpoint-dir",
        default="./experiments/checkpoints/polarfree",
    )
    parser.add_argument("--dm-config", default="./configs/dm_config.yaml")
    parser.add_argument("--output-dir", default="./experiments/inference")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--basic-metrics", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")
    dem_model, task_model, _ = build_models(
        args.dm_config,
        args.polarfree_checkpoint_dir,
        device,
        load_generator=False,
    )
    load_inference_checkpoints(dem_model, task_model, args.ckpt_dir, device)
    dataloader = build_inference_loader(
        args.input_dir, args.batch_size, args.workers, args.limit
    )
    metrics = infer(
        dataloader,
        dem_model,
        task_model,
        device,
        args.output_dir,
        full_metrics=not args.basic_metrics,
    )
    print(f"Results saved to {args.output_dir}: {metrics}")


if __name__ == "__main__":
    main()
