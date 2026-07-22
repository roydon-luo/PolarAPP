# PolaRGB layout

Expected layout:

```text
Datasets/PolaRGB/
|-- train/
|   |-- easy/
|   |   |-- input/<scene>/<capture>_{000,045,090,135,rgb}.png
|   |   `-- gt/<scene>/<reference>_{000,045,090,135,rgb}.png
|   `-- hard/
|       |-- input/<scene>/<capture>_{000,045,090,135,rgb}.png
|       `-- gt/<scene>/<reference>_{000,045,090,135,rgb}.png
`-- test/
    |-- input/<scene>/<capture>_{000,045,090,135,rgb}.png
    `-- gt/<scene>/<reference>_{000,045,090,135,rgb}.png
```

Some dataset releases use three-digit capture names (`000_000.png`) and some
local copies use four digits (`0000_000.png`). The loader discovers either
form by matching the `_000.png` suffix. The loader discovers the reference
prefix dynamically from each ground-truth scene folder.

The data loader checks each selected sample for all polarization-angle, RGB,
and ground-truth files when training or evaluation starts. A separate data
validation command is therefore unnecessary.

For annotation-free inference, `infer.py` only requires the four polarization
images for each capture. It accepts the dataset root shown above, an `input`
directory, or a directory that directly contains scene folders:

```text
input/
`-- <scene>/
    |-- <capture>_000.png
    |-- <capture>_045.png
    |-- <capture>_090.png
    `-- <capture>_135.png
```

PolaRGB is available separately from
<https://huggingface.co/datasets/Mingde/PolaRGB> under CC BY-NC 4.0.
