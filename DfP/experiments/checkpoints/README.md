# DfP checkpoints

PolarAPP checkpoints are available from
[Roydon728/PolarAPP](https://huggingface.co/Roydon728/PolarAPP).

Download the DfP files from the `DfP/` directory:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Roydon728/PolarAPP', allow_patterns='DfP/*', local_dir='./experiments/checkpoints/huggingface')"
```

The downloaded PolarAPP layout is:

```text
experiments/checkpoints/
|-- huggingface/DfP/
|   |-- DemNet/
|   |   `-- DemNet.pth
|   |-- TaskNet/
|   |   `-- TaskNet.pth
|   `-- FANet/
|       `-- FANet.pth
```

Pass `--ckpt-dir ./experiments/checkpoints/huggingface/DfP` to the DfP entry
points. DfP also requires the PolarFree initialization files obtained from the
upstream project; place them under `experiments/checkpoints/polarfree` and pass
that directory through `--polarfree-checkpoint-dir`.

Checkpoint loaders accept a raw state dictionary or a dictionary containing `model_state_dict`, `state_dict`, `params`, or `params_ema`. Training updates the fixed PolarAPP component filenames.

The published SHA-256 checksums are listed in
[`SHA256SUMS.txt`](https://huggingface.co/Roydon728/PolarAPP/blob/main/SHA256SUMS.txt).
