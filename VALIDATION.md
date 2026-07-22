# Validation record

Release-candidate checks were run on 2026-07-22 with Python 3.10.19,
PyTorch 2.9.1 + CUDA 12.8, and an NVIDIA RTX PRO 6000 Blackwell GPU.

The following checks passed:

- release audit (no weights, datasets, generated results, private absolute
  paths, hard-coded CUDA visibility, or files over 10 MiB);
- Python compilation for every source file;
- dependency-free release-audit unit tests;
- command-line help for the aligned SfP and DfP `infer.py` and `train.py`
  entry points;
- SfP inference on one held-out sample with an existing checkpoint;
- SfP evaluation on one held-out sample;
- one SfP feature-alignment step, joint-training step, and refinement step;
- DfP inference/evaluation on one PolaRGB sample;
- one DfP second-order alignment step and joint-training step;
- one full-resolution DfP TaskNet-refinement step.
- DfP dataset discovery returned the paper counts (6,312 train, 188 test);
- SfP evaluation passed on the fixed 100-image test set.

These checks establish functional execution of the release workflows. Full
numerical reproduction of the paper tables requires the exact paper
checkpoints, configuration mapping, and third-party redistribution clearance
listed in `RELEASE_CHECKLIST.md`.
