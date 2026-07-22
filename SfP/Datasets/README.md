# SfP Data

The following SfP dataset links are distributed through the [SfPUEL repository](https://github.com/YouweiLyu/SfPUEL):

- Evaluation data and synthetic test samples: [Google Drive](https://drive.google.com/file/d/1iHEjg90X2bOSkdt9uBCEd76SqzHj5pPC/view?usp=drive_link)
- Training data: [SfPUEL training dataset](https://huggingface.co/datasets/Youwei2768/SfPUEL-training)

Please follow the original dataset terms and cite SfPUEL when using these data.

Expected layout:

```text
Datasets/
  Trainsets/
    pol000/
    pol045/
    pol090/
    pol135/
    normal/
    mask/
  Testsets/
    pol000/
    pol045/
    pol090/
    pol135/
    normal/
    mask/
```

Each sample should have the same filename across the six subfolders. Rename or
reorganize the downloaded folders to match this layout before running PolarAPP.
The data loader checks that the required folders and matching files exist when
training or evaluation starts.
