from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from .config import StarDist2DConfig, load_thresholds
from .net import StarDist2DNet


def _decode_h5_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def list_h5_layers(h5_path: str | Path) -> list[str]:
    with h5py.File(h5_path, "r") as h5:
        names = h5.attrs.get("layer_names")
        if names is not None:
            return [_decode_h5_attr(name) for name in names]
        return list(h5.keys())


def _find_dataset(group: h5py.Group, suffix: str) -> np.ndarray:
    matches: list[h5py.Dataset] = []

    def visitor(_name: str, obj: h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset) and _name.endswith(suffix):
            matches.append(obj)

    group.visititems(visitor)
    if len(matches) != 1:
        raise KeyError(f"Expected one dataset ending with {suffix!r}, found {len(matches)}.")
    return np.asarray(matches[0])


def _load_keras_conv(h5: h5py.File, layer_name: str) -> tuple[np.ndarray, np.ndarray]:
    if layer_name not in h5:
        raise KeyError(f"Layer {layer_name!r} is missing from the H5 file.")
    group = h5[layer_name]
    return _find_dataset(group, "kernel:0"), _find_dataset(group, "bias:0")


def convert_h5_to_state_dict(h5_path: str | Path, config: StarDist2DConfig) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    net = StarDist2DNet(config)
    state = net.state_dict()
    consumed: list[str] = []

    with h5py.File(h5_path, "r") as h5:
        available = set(list_h5_layers(h5_path))
        missing = [name for name in net.keras_layer_names if name not in available]
        if missing:
            raise KeyError(f"The H5 file is missing expected StarDist layers: {missing}")

        for layer_name in net.keras_layer_names:
            kernel, bias = _load_keras_conv(h5, layer_name)
            weight_key = f"layers.{layer_name}.weight"
            bias_key = f"layers.{layer_name}.bias"

            if kernel.ndim != 4:
                raise ValueError(f"{layer_name}: expected Conv2D kernel with 4 dims, got {kernel.shape}.")
            torch_kernel = np.transpose(kernel, (3, 2, 0, 1))
            if tuple(torch_kernel.shape) != tuple(state[weight_key].shape):
                raise ValueError(
                    f"{layer_name}: converted kernel shape {torch_kernel.shape} "
                    f"does not match PyTorch shape {tuple(state[weight_key].shape)}."
                )
            if tuple(bias.shape) != tuple(state[bias_key].shape):
                raise ValueError(
                    f"{layer_name}: bias shape {bias.shape} does not match "
                    f"PyTorch shape {tuple(state[bias_key].shape)}."
                )

            state[weight_key] = torch.from_numpy(np.ascontiguousarray(torch_kernel)).to(dtype=state[weight_key].dtype)
            state[bias_key] = torch.from_numpy(np.ascontiguousarray(bias)).to(dtype=state[bias_key].dtype)
            consumed.append(layer_name)

    report = {
        "source": str(h5_path),
        "converted_layers": consumed,
        "n_layers": len(consumed),
    }
    return state, report


def convert_model_folder(
    model_dir: str | Path,
    weights_name: str | None = None,
    output_name: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {model_dir}. V1 cannot convert model folders without config.json.")

    config = StarDist2DConfig.from_json(config_path)
    weights_name = weights_name or config.train_checkpoint
    h5_path = model_dir / weights_name
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing Keras weights file: {h5_path}")

    output_name = output_name or f"{h5_path.stem}.pt"
    output_path = model_dir / output_name
    state_dict, report = convert_h5_to_state_dict(h5_path, config)
    checkpoint = {
        "state_dict": state_dict,
        "config": config.as_dict(),
        "thresholds": load_thresholds(model_dir / "thresholds.json"),
        "conversion": report,
    }
    torch.save(checkpoint, output_path)
    return output_path, report
