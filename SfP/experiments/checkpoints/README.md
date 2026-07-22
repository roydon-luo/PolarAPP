# Checkpoints

Pretrained checkpoints are available from
[Roydon728/PolarAPP](https://huggingface.co/Roydon728/PolarAPP).

Download the SfP files from the `SfP/` directory:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Roydon728/PolarAPP', allow_patterns='SfP/*', local_dir='./experiments/checkpoints/huggingface')"
```

The download command creates this layout:

```text
experiments/checkpoints/huggingface/SfP/
  DemNet/DemNet.pth
  TaskNet/TaskNet.pth
  FANet/FANet.pth
```

Pass `--ckpt-dir ./experiments/checkpoints/huggingface/SfP` to `infer.py` or
`train.py --eval-only`.

Training updates these fixed filenames and stores the current epoch inside each checkpoint.

The published SHA-256 checksums are listed in
[`SHA256SUMS.txt`](https://huggingface.co/Roydon728/PolarAPP/blob/main/SHA256SUMS.txt).
