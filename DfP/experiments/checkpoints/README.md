# DfP checkpoints

Weights are not included. Place them in the following layout:

```text
experiments/checkpoints/
|-- polarapp/
|   |-- DemNet/
|   |   `-- DemNet_200.pth
|   |-- TaskNet/
|   |   `-- TaskNet_200.pth
|   `-- FANet/
|       `-- FANet_200.pth
`-- polarfree/
    |-- net_le_dm_latest.pth
    |-- net_d_latest.pth
    `-- net_g_latest.pth
```

`net_g_latest.pth` initializes TaskNet for a fresh training run. It is not required when a PolarAPP TaskNet checkpoint is loaded for inference, evaluation, refinement, or resumed training.

Checkpoint loaders accept a raw state dictionary or a dictionary containing `model_state_dict`, `state_dict`, `params`, or `params_ema`. Epoch filenames may use padded or unpadded numbers.
