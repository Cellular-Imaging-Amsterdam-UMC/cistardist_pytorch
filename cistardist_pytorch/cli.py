from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import tifffile

from .converter import convert_model_folder
from .model import StarDist2D


def main_convert() -> None:
    parser = argparse.ArgumentParser(description="Convert a Keras StarDist model folder to a PyTorch checkpoint.")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--weights", default=None, help="Weights filename inside model_dir. Defaults to config train_checkpoint.")
    parser.add_argument("--out", default=None, help="Output .pt filename inside model_dir. Defaults to weights stem + .pt.")
    args = parser.parse_args()

    output_path, report = convert_model_folder(args.model_dir, weights_name=args.weights, output_name=args.out)
    print(f"Wrote {output_path}")
    print(f"Converted {report['n_layers']} layers from {report['source']}")


def main_predict() -> None:
    parser = argparse.ArgumentParser(description="Run PyTorch StarDist 2D inference on a TIFF image.")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prob-thresh", type=float, default=None)
    parser.add_argument("--nms-thresh", type=float, default=None)
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    model = StarDist2D.from_folder(args.model_dir, device=args.device)
    image = tifffile.imread(args.image)
    labels, details = model.predict_instances(
        image,
        prob_thresh=args.prob_thresh,
        nms_thresh=args.nms_thresh,
        normalize=not args.no_normalize,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.uint16 if int(labels.max(initial=0)) <= np.iinfo(np.uint16).max else np.uint32
    tifffile.imwrite(args.out, labels.astype(dtype, copy=False))
    print(f"Wrote {args.out}")
    print(f"Instances: {len(details['prob'])}")


# ---------------------------------------------------------------------------
# DOI helpers
# ---------------------------------------------------------------------------

def _doi_to_folder_name(doi: str) -> str:
    """Convert a DOI to a safe folder name by replacing '/' with '_'."""
    return doi.replace("/", "_")


def _default_models_dir() -> Path:
    return Path.home() / ".cistardist_pytorch" / "models"


def _fetch_zenodo_title(doi: str, timeout: float = 25.0) -> str | None:
    """Return the title of a Zenodo record for *doi*, or None on any error."""
    doi_url = doi if doi.startswith("http") else f"https://doi.org/{doi}"
    req = urllib.request.Request(doi_url, headers={"User-Agent": "cistardist_pytorch"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            record_url = resp.url
    except Exception:
        return None
    record_id = record_url.rstrip("/").split("/")[-1]
    api_url = f"https://zenodo.org/api/records/{record_id}"
    req2 = urllib.request.Request(api_url, headers={"User-Agent": "cistardist_pytorch"})
    try:
        with urllib.request.urlopen(req2, timeout=timeout) as resp2:
            data = json.loads(resp2.read().decode("utf-8"))
        return data.get("metadata", {}).get("title")
    except Exception:
        return None


def _download_doi(doi: str, output_dir: Path) -> None:
    """Download all files for a Zenodo DOI into *output_dir* using zenodo_get."""
    try:
        from zenodo_get import download as zget_download  # type: ignore[import]
    except ImportError:
        raise RuntimeError("zenodo_get is not installed. Install it with: pip install zenodo-get")
    output_dir.mkdir(parents=True, exist_ok=True)
    zget_download(doi=doi, output_dir=output_dir)


def _find_pt_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.pt"))


def main_predict_fromdoi() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download a StarDist model from Zenodo by DOI and run PyTorch StarDist 2D "
            "inference on a TIFF image. Files are downloaded via zenodo_get and cached "
            "in a folder named after the DOI (with '/' replaced by '_')."
        )
    )
    parser.add_argument(
        "doi",
        help="Zenodo DOI, e.g. 10.5281/zenodo.20038194",
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prob-thresh", type=float, default=None)
    parser.add_argument("--nms-thresh", type=float, default=None)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help=f"Base directory for cached models. Defaults to {_default_models_dir()}",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Always re-download even if the folder already contains .pt files.",
    )
    args = parser.parse_args()

    models_dir = args.models_dir or _default_models_dir()
    folder_name = _doi_to_folder_name(args.doi)
    model_folder = models_dir / folder_name

    pt_files = _find_pt_files(model_folder)
    if args.no_cache or not pt_files:
        print(f"Downloading {args.doi} to {model_folder}")
        title = _fetch_zenodo_title(args.doi)
        if title:
            model_folder.mkdir(parents=True, exist_ok=True)
            (model_folder / "title.txt").write_text(title, encoding="utf-8")
            print(f"Title: {title}")
        _download_doi(args.doi, model_folder)
        pt_files = _find_pt_files(model_folder)
    else:
        title_path = model_folder / "title.txt"
        if title_path.exists():
            print(f"Title: {title_path.read_text(encoding='utf-8').strip()}")
        print(f"Using cached model folder: {model_folder}")

    if not pt_files:
        raise FileNotFoundError(f"No .pt file found in {model_folder} after download.")
    if len(pt_files) > 1:
        names = [p.name for p in pt_files]
        raise RuntimeError(f"Multiple .pt files found in {model_folder}: {names}. Specify which to use.")

    pt_path = pt_files[0]
    print(f"Model: {pt_path.stem}")

    model = StarDist2D.from_checkpoint(pt_path, device=args.device)
    image = tifffile.imread(args.image)
    labels, details = model.predict_instances(
        image,
        prob_thresh=args.prob_thresh,
        nms_thresh=args.nms_thresh,
        normalize=not args.no_normalize,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.uint16 if int(labels.max(initial=0)) <= np.iinfo(np.uint16).max else np.uint32
    tifffile.imwrite(args.out, labels.astype(dtype, copy=False))
    print(f"Wrote {args.out}")
    print(f"Instances: {len(details['prob'])}")

