from __future__ import annotations

import argparse
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
