import argparse

from polarapp.checkpoints import load_inference_checkpoints
from polarapp.config import get_device
from polarapp.data import build_inference_loader
from polarapp.evaluation import infer
from polarapp.models import build_models


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run PolarAPP on four polarization-angle image folders."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--ckpt-dir", default="./experiments/checkpoints")
    parser.add_argument("--output-dir", default="./experiments/inference")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser


def main():
    args = build_parser().parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")
    dem_model, task_model, _ = build_models(device, include_alignment=False)
    load_inference_checkpoints(
        dem_model,
        task_model,
        args.ckpt_dir,
        device,
    )
    dataloader = build_inference_loader(args.input_dir, args.batch_size)
    output_dir = infer(dataloader, dem_model, task_model, device, args.output_dir)
    print(f"Results saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
