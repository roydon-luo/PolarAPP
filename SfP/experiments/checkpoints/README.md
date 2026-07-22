# Checkpoints

Pretrained checkpoints are distributed separately through the project page or a link provided by the authors.

Place the checkpoint files as follows:

```text
experiments/checkpoints/
  DemNet/
    DemNet_<epoch>.pth
  TaskNet/
    TaskNet_<epoch>.pth
  FANet/
    FANet_<epoch>.pth
```

The inference script automatically selects the latest matching checkpoint unless `--ckpt-epoch` is supplied.
