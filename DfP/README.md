# PolarAPP-DfP

PolarAPP-DfP is the de-reflection-from-polarization branch of PolarAPP. PIDNet reconstructs full-resolution polarization images and the PolarFree-based TaskNet removes reflections. FANet aligns intermediate features during training and is used only by the training workflow.

## Installation

The reference environment uses Python 3.10. Install a PyTorch/torchvision pair matching the CUDA version on your machine, then install the remaining dependencies:

```bash
conda create -n polarapp-dfp python=3.10 -y
conda activate polarapp-dfp
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The CUDA wheel URL is an example. CPU execution is suitable for small functional checks; full-resolution inference and training are intended for a CUDA GPU.

## Data and checkpoints

Download PolaRGB into `Datasets/PolaRGB`:

```bash
huggingface-cli download Mingde/PolaRGB --repo-type dataset \
  --local-dir ./Datasets/PolaRGB
```

For resumable HTTPS download in proxy-constrained environments:

```bash
python scripts/download_polargb.py --output ./Datasets/PolaRGB
```

See [`Datasets/README.md`](./Datasets/README.md) for the expected data layout and [`experiments/checkpoints/README.md`](./experiments/checkpoints/README.md) for the model layout. Dataset files and model weights are distributed separately.

## Code structure

- `train.py` is the unified training, refinement, and evaluation entry point.
- `infer.py` runs prediction and metric evaluation on PolaRGB inputs.
- `polarapp/trainer.py` coordinates feature alignment, joint training, and TaskNet refinement.
- `polarapp/evaluation.py` contains prediction, metrics, and output routines.
- `polarapp/data.py` and `polarapp/checkpoints.py` handle data and model state.
- `archs/` contains the PolarFree-based task architecture and diffusion wrapper.

This layout matches the SfP branch. `polarapp/trainer.py` contains the refinement training stage.

## Inference

```bash
python infer.py \
  --input-dir ./Datasets/PolaRGB \
  --polarfree-checkpoint-dir ./experiments/checkpoints/polarfree \
  --ckpt-dir ./experiments/checkpoints/polarapp \
  --output-dir ./experiments/inference \
  --device cuda:0
```

Predictions are saved by scene. `metrics.csv` always contains PSNR and SSIM. Install `lpips` and `pyiqa` for the optional LPIPS and MUSIQ metrics. Use `--limit 1 --basic-metrics` for a short end-to-end smoke test.

## Evaluation

```bash
python train.py --eval-only \
  --data-path ./Datasets/PolaRGB \
  --polarfree-checkpoint-dir ./experiments/checkpoints/polarfree \
  --ckpt-dir ./experiments/checkpoints/polarapp \
  --device cuda:0
```

`--data-path` may point directly at the local public PolaRGB root used for the evaluation run; evaluation limits such as the 100-image protocol are independent of training orchestration.

## Training and refinement

The default command runs the paper workflow in one trainer: second-order feature-alignment meta-learning, joint DemNet/TaskNet optimization, then TaskNet refinement with DemNet and FANet frozen.

```bash
python train.py --device cuda:0
```

To resume all three components:

```bash
python train.py --resume \
  --ckpt-dir ./experiments/checkpoints/polarapp \
  --device cuda:0
```

To run only the integrated refinement stage from existing checkpoints:

```bash
python train.py --refine-only \
  --data-path ./Datasets/PolaRGB \
  --polarfree-checkpoint-dir ./experiments/checkpoints/polarfree \
  --ckpt-dir ./experiments/checkpoints/polarapp \
  --save-ckpt-dir ./experiments/checkpoints/refined \
  --max-refine-step 1000 \
  --device cuda:0
```

The paper uses `lambda_task=20`; the DfP alignment sweep peaks at `lambda_fa=10`. Both are configurable in `configs/train_config.yaml` or through command-line overrides.

## Release note

The DfP architecture is adapted from PolarFree. Confirm upstream redistribution permission before publishing these derived files; see the repository-level `THIRD_PARTY_NOTICES.md` and `RELEASE_CHECKLIST.md`.
