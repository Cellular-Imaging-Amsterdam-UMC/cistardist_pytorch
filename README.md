# cistardist_pytorch

PyTorch-only StarDist 2D inference for existing TensorFlow/Keras StarDist `.h5`
weights. The first target model is `models/SD_Nuclei_Versatile`.

This repo intentionally does not depend on TensorFlow, Keras, CSBDeep, or the
upstream `stardist` Python package.

**GitHub:** https://github.com/Cellular-Imaging-Amsterdam-UMC/cistardist_pytorch

## Install

The package is published on PyPI:
https://pypi.org/project/cistardist-pytorch/

Install the library with:

```bash
pip install cistardist-pytorch
```

To use the Zenodo DOI download feature, also install `zenodo-get`:

```bash
pip install zenodo-get
```

PyTorch is intentionally not pinned by the package, so you can choose the CPU or
CUDA build that matches your environment. For this repo's CUDA 12.6 setup, use:

```bash
conda create -n cistardist_pytorch python=3.10
conda activate cistardist_pytorch
pip install -r requirements.txt
pip install -e . --no-deps
```

The package includes a compiled StarDist-style C++ NMS extension for fast
polygon suppression. When building from source, a C++ compiler is needed to
compile it; if the extension is not available, inference automatically falls
back to the slower pure-Python NMS implementation. See `COMPILE_NMS.md` for
detailed build instructions.

## Convert and Predict (local model folder)

```bash
cistardist-convert models/SD_Nuclei_Versatile
cistardist-predict models/SD_Nuclei_Versatile data/nuclei.tif --out outputs/nuclei_labels.tif
```

The converter reads `config.json`, `thresholds.json`, and `weights_best.h5`.
It writes a PyTorch checkpoint next to the source model as `weights_best.pt`.

## Predict from a Zenodo DOI

Download a self-contained `.pt` checkpoint directly from Zenodo and run
inference in a single command:

```bash
cistardist-predict-fromdoi 10.5281/zenodo.20038194 data/nuclei.tif --out outputs/nuclei_labels.tif
```

All files for the record are downloaded with `zenodo_get` and cached in
`~/.cistardist_pytorch/models/10.5281_zenodo.20038194/` (the DOI with `/`
replaced by `_`). A `title.txt` file is also saved there with the record
title from the Zenodo API. Subsequent calls reuse the cache; pass `--no-cache`
to force a fresh download.

Additional options mirror `cistardist-predict`:

```
--device        cpu / cuda:0 / auto (default: auto)
--prob-thresh   override probability threshold
--nms-thresh    override NMS threshold
--no-normalize  skip percentile normalization
--models-dir    override cache base directory
--no-cache      always re-download
```

## Python API

### Load from a local model folder

```python
import tifffile
from cistardist_pytorch import StarDist2D

model = StarDist2D.from_folder("models/SD_Nuclei_Versatile")
image = tifffile.imread("data/nuclei.tif")
labels, details = model.predict_instances(image)
```

### Load from a Zenodo DOI and predict over a folder

The snippet below downloads the model once (cached automatically), then runs
inference on every `.tif` image in `inputfolder/` and saves the label images
to `masksfolder/`.

```python
from pathlib import Path

import numpy as np
import tifffile

from cistardist_pytorch.cli import (
    _default_models_dir,
    _doi_to_folder_name,
    _download_doi,
    _fetch_zenodo_title,
    _find_pt_files,
)
from cistardist_pytorch.model import StarDist2D

DOI = "10.5281/zenodo.20038194"
INPUT_FOLDER = Path("inputfolder")
MASKS_FOLDER = Path("masksfolder")

# --- resolve / download model -------------------------------------------
models_dir = _default_models_dir()
model_folder = models_dir / _doi_to_folder_name(DOI)
pt_files = _find_pt_files(model_folder)

if not pt_files:
    title = _fetch_zenodo_title(DOI)
    if title:
        model_folder.mkdir(parents=True, exist_ok=True)
        (model_folder / "title.txt").write_text(title, encoding="utf-8")
        print(f"Title: {title}")
    _download_doi(DOI, model_folder)
    pt_files = _find_pt_files(model_folder)

pt_path = pt_files[0]
print(f"Model: {pt_path.stem}")

# --- load model ---------------------------------------------------------
model = StarDist2D.from_checkpoint(pt_path, device="auto")

# --- batch predict ------------------------------------------------------
MASKS_FOLDER.mkdir(parents=True, exist_ok=True)

for image_path in sorted(INPUT_FOLDER.glob("*.tif")):
    image = tifffile.imread(image_path)
    labels, _ = model.predict_instances(image)
    dtype = np.uint16 if int(labels.max(initial=0)) <= np.iinfo(np.uint16).max else np.uint32
    out_path = MASKS_FOLDER / image_path.name
    tifffile.imwrite(out_path, labels.astype(dtype, copy=False))
    print(f"  {image_path.name} -> {out_path.name}")
```

## Current Scope

- 2D grayscale inference
- Keras `.h5` Conv2D weight conversion via `h5py`
- Zenodo DOI-based model download and caching via `zenodo-get`
- StarDist-style polygon postprocessing with compiled C++ NMS and vendored
  BSD-compatible 2D geometry
- No training, no TensorFlow reference tests, no 3D, no multiclass models

## Attribution

The 2D geometry and NMS behavior follows the BSD-3-Clause upstream StarDist
project: https://github.com/stardist/stardist
