# DfP checkpoints

Weights are distributed separately. Place them in the following layout:

```text
experiments/checkpoints/
|-- polarapp/
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

`net_g_latest.pth` initializes TaskNet for a fresh training run. Inference, evaluation, refinement, and resumed training load `TaskNet.pth`.

Checkpoint loaders accept a raw state dictionary or a dictionary containing `model_state_dict`, `state_dict`, `params`, or `params_ema`. Training updates the fixed component filenames and stores the current epoch inside each checkpoint.
