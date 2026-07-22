# DfP checkpoints

PolarAPP checkpoints are available from
[Roydon728/PolarAPP](https://huggingface.co/Roydon728/PolarAPP).

Download the DfP files from the `DfP/` directory:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Roydon728/PolarAPP', allow_patterns='DfP/*', local_dir='./experiments/checkpoints/huggingface')"
```

The complete local layout is:

```text
experiments/checkpoints/
|-- huggingface/DfP/
|   |-- DemNet/
|   |   `-- DemNet.pth
|   |-- TaskNet/
|   |   `-- TaskNet.pth
|   `-- FANet/
|       `-- FANet.pth
`-- polarfree/
    |-- net_le_dm_latest.pth
    |-- net_d_latest.pth
    `-- net_g_latest.pth
```

Pass `--ckpt-dir ./experiments/checkpoints/huggingface/DfP` to the DfP entry
points. The PolarFree diffusion-prior files retain their upstream names and
location.

`net_g_latest.pth` initializes TaskNet for a fresh training run. Inference, evaluation, refinement, and resumed training load `TaskNet.pth`.

Checkpoint loaders accept a raw state dictionary or a dictionary containing `model_state_dict`, `state_dict`, `params`, or `params_ema`. Training updates the fixed component filenames and stores the current epoch inside each checkpoint.

The published SHA-256 checksums are listed in
[`SHA256SUMS.txt`](https://huggingface.co/Roydon728/PolarAPP/blob/main/SHA256SUMS.txt).
