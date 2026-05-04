from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tifffile
import torch

from cistardist_pytorch.geometry import dist_to_coord, polygons_to_label
from cistardist_pytorch.model import StarDist2D, _crop_prediction
from cistardist_pytorch.nms import _candidate_mask, non_maximum_suppression


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / "models" / "SD_Nuclei_Versatile"
DEFAULT_IMAGES = (ROOT / "data" / "nuclei.tif", ROOT / "data" / "nuclei_medium.tif")


@dataclass
class ProfileResult:
    device: str
    image: Path
    run: int
    timings: dict[str, float]
    metadata: dict[str, Any]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(
    timings: dict[str, float],
    name: str,
    device: torch.device,
    func: Callable[[], Any],
) -> Any:
    synchronize(device)
    start = time.perf_counter()
    value = func()
    synchronize(device)
    timings[name] = time.perf_counter() - start
    return value


def timed_forward(
    model: StarDist2D,
    tensor: torch.Tensor,
    timings: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, float | None]:
    device = model.device
    synchronize(device)

    start_event = end_event = None
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    start = time.perf_counter()
    with torch.no_grad():
        prob_t, dist_t = model.net(tensor)

    if end_event is not None:
        end_event.record()
    synchronize(device)

    timings["network_forward_wall"] = time.perf_counter() - start
    gpu_seconds = None
    if start_event is not None and end_event is not None:
        gpu_seconds = start_event.elapsed_time(end_event) / 1000.0
        timings["network_forward_cuda_event"] = gpu_seconds

    return prob_t, dist_t, gpu_seconds


def query_nvidia_smi() -> str | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def warmup_forward(model: StarDist2D, image_path: Path, normalize: bool, count: int) -> None:
    if count <= 0:
        return

    image = tifffile.imread(image_path)
    x, _original_shape, _pad = model._prepare_image(image, normalize=normalize)
    tensor = torch.from_numpy(x[None, None]).to(model.device)

    with torch.no_grad():
        for _ in range(count):
            model.net(tensor)
    synchronize(model.device)


def profile_segmentation(
    model: StarDist2D,
    image_path: Path,
    run: int,
    normalize: bool,
    prob_thresh: float | None,
    nms_thresh: float | None,
    verbose: bool = True,
) -> ProfileResult:
    device = model.device
    timings: dict[str, float] = {}
    metadata: dict[str, Any] = {}

    if verbose:
        print(f"Profiling {image_path.name} on {device} run {run}...", flush=True)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    total_start = time.perf_counter()

    image = timed(timings, "read_image", device, lambda: tifffile.imread(image_path))
    metadata["image_shape"] = tuple(int(v) for v in image.shape)
    metadata["image_dtype"] = str(image.dtype)

    shape = tuple(int(v) for v in np.asarray(image).shape[:2])
    resolved_prob_thresh = model.thresholds["prob"] if prob_thresh is None else float(prob_thresh)
    resolved_nms_thresh = model.thresholds["nms"] if nms_thresh is None else float(nms_thresh)
    metadata["prob_thresh"] = resolved_prob_thresh
    metadata["nms_thresh"] = resolved_nms_thresh

    x, _original_shape, pad = timed(
        timings,
        "prepare_normalize_pad",
        device,
        lambda: model._prepare_image(image, normalize=normalize),
    )
    metadata["padded_shape"] = tuple(int(v) for v in x.shape)
    metadata["pad_yx"] = tuple(int(v) for v in pad)

    tensor = timed(
        timings,
        "numpy_to_tensor_device",
        device,
        lambda: torch.from_numpy(x[None, None]).to(device),
    )
    metadata["tensor_device"] = str(tensor.device)
    metadata["tensor_shape"] = tuple(int(v) for v in tensor.shape)

    prob_t, dist_t, gpu_forward_seconds = timed_forward(model, tensor, timings)
    metadata["network_output_device"] = str(prob_t.device)
    metadata["gpu_forward_seconds"] = gpu_forward_seconds

    def outputs_to_numpy() -> tuple[np.ndarray, np.ndarray]:
        prob = prob_t[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
        dist = np.moveaxis(dist_t[0].detach().cpu().numpy(), 0, -1).astype(np.float32, copy=False)
        prob = _crop_prediction(prob, pad, model.config.grid)
        dist = _crop_prediction(dist, pad, model.config.grid)
        return prob, dist

    prob, dist = timed(timings, "output_to_cpu_crop", device, outputs_to_numpy)
    metadata["prob_map_shape"] = tuple(int(v) for v in prob.shape)
    metadata["dist_map_shape"] = tuple(int(v) for v in dist.shape)

    dist = timed(timings, "distance_clip", device, lambda: np.maximum(dist, 1e-3))

    candidate_mask = timed(
        timings,
        "candidate_mask",
        device,
        lambda: _candidate_mask(prob, prob_thresh=resolved_prob_thresh, b=2),
    )
    metadata["candidate_pixels"] = int(np.count_nonzero(candidate_mask))

    if verbose:
        print(
            f"  entering NMS with {metadata['candidate_pixels']} candidates "
            f"on {image_path.name} / {device}",
            flush=True,
        )

    points, probi, disti = timed(
        timings,
        "non_maximum_suppression",
        device,
        lambda: non_maximum_suppression(
            dist,
            prob,
            grid=model.config.grid,
            prob_thresh=resolved_prob_thresh,
            nms_thresh=resolved_nms_thresh,
        ),
    )
    metadata["instances_after_nms"] = int(len(points))

    labels = timed(
        timings,
        "render_labels",
        device,
        lambda: polygons_to_label(disti, points, shape=shape, prob=probi),
    )
    metadata["labels_dtype"] = str(labels.dtype)
    metadata["label_max"] = int(labels.max(initial=0))

    timed(
        timings,
        "build_details_coord",
        device,
        lambda: dist_to_coord(disti, points)
        if len(points)
        else np.zeros((0, 2, model.config.n_rays), dtype=np.float32),
    )

    synchronize(device)
    timings["total"] = time.perf_counter() - total_start

    if device.type == "cuda":
        metadata["cuda_peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024**2)

    return ProfileResult(
        device=str(device),
        image=image_path,
        run=run,
        timings=timings,
        metadata=metadata,
    )


def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 0.001:
        return f"{seconds * 1_000_000:8.1f} us"
    if seconds < 1:
        return f"{seconds * 1000:8.2f} ms"
    return f"{seconds:8.3f} s "


def print_result(result: ProfileResult) -> None:
    meta = result.metadata
    print()
    print(f"{result.image.name} | {result.device} | run {result.run}")
    print(f"  image={meta['image_shape']} {meta['image_dtype']}  padded={meta['padded_shape']}  pad={meta['pad_yx']}")
    print(f"  tensor_device={meta['tensor_device']}  output_device={meta['network_output_device']}")
    if meta.get("cuda_peak_memory_mb") is not None:
        print(f"  cuda_peak_memory={meta['cuda_peak_memory_mb']:.1f} MiB")
    print(
        f"  candidates={meta['candidate_pixels']}  instances={meta['instances_after_nms']}  "
        f"prob_thresh={meta['prob_thresh']:.6g}  nms_thresh={meta['nms_thresh']:.6g}"
    )
    print("  step                         time")
    print("  ---------------------------  ------------")
    for name, seconds in sorted(result.timings.items(), key=lambda item: item[1], reverse=True):
        print(f"  {name:<27}  {format_seconds(seconds)}")


def print_summary(results: list[ProfileResult]) -> None:
    if not results:
        return

    print()
    print("Summary")
    print("-------")
    by_image: dict[Path, dict[str, list[ProfileResult]]] = {}
    for result in results:
        by_image.setdefault(result.image, {}).setdefault(result.device, []).append(result)

    for image, by_device in by_image.items():
        print(image.name)
        means: dict[str, float] = {}
        for device, device_results in by_device.items():
            total = sum(r.timings["total"] for r in device_results) / len(device_results)
            forward = sum(r.timings["network_forward_wall"] for r in device_results) / len(device_results)
            nms = sum(r.timings["non_maximum_suppression"] for r in device_results) / len(device_results)
            render = sum(r.timings["render_labels"] for r in device_results) / len(device_results)
            means[device] = total
            print(
                f"  {device:<8} total={format_seconds(total).strip():>10}  "
                f"forward={format_seconds(forward).strip():>10}  "
                f"nms={format_seconds(nms).strip():>10}  render={format_seconds(render).strip():>10}"
            )

        cpu_total = means.get("cpu")
        cuda_total = means.get("cuda:0")
        if cpu_total is not None and cuda_total is not None and cuda_total > 0:
            print(f"  CPU/GPU total speedup: {cpu_total / cuda_total:.2f}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile cistardist_pytorch segmentation step-by-step on CPU and GPU. "
            "Use it with the sdcpsam conda env, for example: "
            r"C:\Users\p000881\AppData\Local\miniconda3\envs\sdcpsam\python.exe test.py"
        )
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--images", type=Path, nargs="+", default=list(DEFAULT_IMAGES))
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda:0"])
    parser.add_argument("--runs", type=int, default=1, help="Timed full segmentation runs per image/device.")
    parser.add_argument("--warmup-forwards", type=int, default=1, help="Untimed network-only warmup forwards per image/device.")
    parser.add_argument("--prob-thresh", type=float, default=None)
    parser.add_argument("--nms-thresh", type=float, default=None)
    parser.add_argument("--no-normalize", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    print(f"Python: {sys.executable}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
        smi = query_nvidia_smi()
        if smi:
            print(f"nvidia-smi: {smi}")

    results: list[ProfileResult] = []
    normalize = not args.no_normalize

    for device_name in args.devices:
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            print()
            print(f"Skipping {device_name}: torch.cuda.is_available() is False")
            continue

        device = torch.device(device_name)
        print()
        print(f"Loading model on {device} from {args.model_dir}")
        load_start = time.perf_counter()
        model = StarDist2D.from_folder(args.model_dir, device=device)
        synchronize(model.device)
        load_seconds = time.perf_counter() - load_start
        first_param = next(model.net.parameters())
        print(f"Model loaded in {format_seconds(load_seconds).strip()} on parameter_device={first_param.device}")

        for image_path in args.images:
            if not image_path.exists():
                print(f"Missing image: {image_path}")
                continue

            warmup_forward(model, image_path, normalize=normalize, count=args.warmup_forwards)
            for run in range(1, args.runs + 1):
                result = profile_segmentation(
                    model=model,
                    image_path=image_path,
                    run=run,
                    normalize=normalize,
                    prob_thresh=args.prob_thresh,
                    nms_thresh=args.nms_thresh,
                    verbose=True,
                )
                print_result(result)
                results.append(result)

    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
