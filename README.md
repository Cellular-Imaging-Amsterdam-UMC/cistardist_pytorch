# cistardist_pytorch

PyTorch-only StarDist 2D inference for existing TensorFlow/Keras StarDist `.h5`
weights. The first target model is `models/SD_Nuclei_Versatile`.

This repo intentionally does not depend on TensorFlow, Keras, CSBDeep, or the
upstream `stardist` Python package.

## Install

The package is published on PyPI:
https://pypi.org/project/cistardist-pytorch/

Install the library with:

```bash
pip install cistardist-pytorch
```

PyTorch is intentionally not pinned by the package, so you can choose the CPU or
CUDA build that matches your environment. For this repo's CUDA 12.6 setup, use:

```bash
conda create -n cistardist_pytorch python=3.10
conda activate cistardist_pytorch
pip install -r requirements.txt
pip install -e . --no-deps
```

## Convert and Predict

```bash
cistardist-convert models/SD_Nuclei_Versatile
cistardist-predict models/SD_Nuclei_Versatile data/nuclei.tif --out outputs/nuclei_labels.tif
```

The converter reads `config.json`, `thresholds.json`, and `weights_best.h5`.
It writes a PyTorch checkpoint next to the source model as `weights_best.pt`.

## Python API

```python
import tifffile
from cistardist_pytorch import StarDist2D

model = StarDist2D.from_folder("models/SD_Nuclei_Versatile")
image = tifffile.imread("data/nuclei.tif")
labels, details = model.predict_instances(image)
```

## Current Scope

- 2D grayscale inference
- Keras `.h5` Conv2D weight conversion via `h5py`
- StarDist-style polygon postprocessing with vendored BSD-compatible 2D geometry
- No training, no TensorFlow reference tests, no 3D, no multiclass models

## Attribution

The 2D geometry and NMS behavior follows the BSD-3-Clause upstream StarDist
project: https://github.com/stardist/stardist
