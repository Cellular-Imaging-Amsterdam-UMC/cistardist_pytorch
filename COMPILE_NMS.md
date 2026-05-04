# Compiling the C++ NMS Extension

`cistardist_pytorch` includes an optional compiled C++ extension for fast
StarDist-style 2D non-maximum suppression:

```text
cistardist_pytorch._c_nms
```

When this extension is available, `cistardist_pytorch.nms.non_maximum_suppression`
uses it automatically. If it is not built or cannot be imported, the package
falls back to the pure-Python NMS implementation.

## Requirements

- Python environment with this package's dependencies installed
- `numpy` installed in the build environment
- A working C++ compiler
- On Windows: Microsoft Visual Studio C++ Build Tools or Visual Studio with the
  Desktop development with C++ workload
- On Linux/macOS: a C++ compiler such as GCC or Clang

The extension vendors the same style of components used by upstream StarDist:
Clipper for polygon intersection and nanoflann for neighbor lookup.

## Windows Conda Build

For the local `sdcpsam` environment:

```powershell
cd C:\rahoebe\Python\cistardist_pytorch
C:\Users\p000881\AppData\Local\miniconda3\envs\sdcpsam\python.exe setup.py build_ext --inplace
```

This should create a compiled extension next to the package sources, for
example:

```text
cistardist_pytorch\_c_nms.cp311-win_amd64.pyd
```

If the command cannot find a compiler, install Visual Studio Build Tools 2022
and include the C++ build tools workload.

## Linux/macOS Build

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python setup.py build_ext --inplace
```

The build tries to enable OpenMP for parallel NMS. If OpenMP flags are not
accepted by the compiler, the build retries without OpenMP.

## Verify The Extension

Run:

```bash
python - <<'PY'
from cistardist_pytorch._c_nms import c_non_max_suppression_inds
print("compiled NMS OK:", c_non_max_suppression_inds)
PY
```

Or with the local Windows conda environment:

```powershell
C:\Users\p000881\AppData\Local\miniconda3\envs\sdcpsam\python.exe -c "from cistardist_pytorch._c_nms import c_non_max_suppression_inds; print('compiled NMS OK:', c_non_max_suppression_inds)"
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

The compiled NMS-specific test is skipped when the extension is not built.

## Benchmark

Use the profiling script:

```bash
python test.py --images data/nuclei.tif data/nuclei_medium.tif --devices cpu cuda:0 --runs 1 --warmup-forwards 1
```

With the local Windows conda environment:

```powershell
C:\Users\p000881\AppData\Local\miniconda3\envs\sdcpsam\python.exe test.py --images data\nuclei.tif data\nuclei_medium.tif --devices cpu cuda:0 --runs 1 --warmup-forwards 1
```

The output includes a `non_maximum_suppression` timing row. If the compiled
extension is active, NMS should be in milliseconds for the included example
images instead of several seconds.

## Clean And Rebuild

To force a clean rebuild:

```bash
rm -rf build
python setup.py build_ext --inplace
```

On Windows PowerShell:

```powershell
Remove-Item -Recurse -Force build
C:\Users\p000881\AppData\Local\miniconda3\envs\sdcpsam\python.exe setup.py build_ext --inplace
```

## Notes

- The compiled extension accelerates NMS on CPU. Neural network inference still
  runs on CPU or GPU according to the selected PyTorch device.
- Label rendering is still CPU-side and can become a visible cost after NMS is
  accelerated.
- Source distributions include the C++ files and vendored headers through
  `MANIFEST.in`.
