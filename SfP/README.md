# PolarAPP

PolarAPP-SfP is a full-resolution shape-from-polarization pipeline. It takes four polarization-angle images as input, reconstructs full-resolution polarization observations with PIDNet, and estimates surface normals with a TaskNet designed specifically for PolarAPP.

## Installation

```bash
conda create -n polarapp-sfp python=3.10
conda activate polarapp-sfp
# Install a matching torch/torchvision pair for your CUDA version first.
pip install -r requirements.txt
```

Install the PyTorch and torchvision builds that match your CUDA version from
the official PyTorch instructions.

## Data And Weights

Dataset files are distributed separately. See `Datasets/README.md` for the expected directory layout.

Model checkpoints are available from [Roydon728/PolarAPP](https://huggingface.co/Roydon728/PolarAPP). See `experiments/checkpoints/README.md` for download instructions and the expected layout.

## Code Structure

- `train.py` is the training and evaluation entry point.
- `infer.py` runs prediction without ground-truth normals or masks.
- `polarapp/trainer.py` coordinates feature alignment, joint training, and TaskNet refinement.
- `polarapp/evaluation.py` contains prediction, metric, and output routines.
- `polarapp/data.py` and `polarapp/checkpoints.py` handle data and model state.

## Inference

Inference requires four folders named `pol000`, `pol045`, `pol090`, and `pol135` with matching RGB image filenames.
Task annotations are not required. The saved surface normal keeps the full-resolution
TaskNet output at twice the input height and width, and no metrics are computed.

```bash
python infer.py --input-dir ./Datasets/Testsets --ckpt-dir ./experiments/checkpoints/huggingface/SfP --device cuda:0
```

## Evaluation

Evaluation additionally requires matching `normal` and `mask` folders. Angular
metrics use the simulated-imaging evaluation path and are computed directly at
the task ground-truth resolution. The saved surface normal has the same height
and width as the task ground truth.

```bash
python train.py --eval-only --device cuda:0 --ckpt-dir ./experiments/checkpoints/huggingface/SfP --data-path ./Datasets/Testsets
```

Results from the released epoch-201 checkpoints on all 100 evaluation samples:

| Mean (deg) | Median (deg) | RMSE (deg) | < 11.25 deg | < 22.5 deg | < 30 deg |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6.7265 | 2.6533 | 12.5278 | 80.57% | 92.97% | 95.43% |

## Training

```bash
python train.py --device cuda:0
```

Common settings can be overridden from the command line:

```bash
python train.py --device cuda:1 --save-ckpt-dir ./experiments/checkpoints_new
```
