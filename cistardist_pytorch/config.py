from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StarDist2DConfig:
    n_dim: int
    axes: str
    n_channel_in: int
    n_channel_out: int
    n_rays: int
    grid: tuple[int, int]
    backbone: str
    unet_n_depth: int
    unet_kernel_size: tuple[int, int]
    unet_n_filter_base: int
    unet_n_conv_per_depth: int
    unet_pool: tuple[int, int]
    unet_activation: str
    unet_last_activation: str
    unet_batch_norm: bool
    unet_dropout: float
    net_conv_after_unet: int
    train_checkpoint: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StarDist2DConfig":
        if int(data.get("n_dim", 0)) != 2:
            raise ValueError("Only 2D StarDist models are supported in V1.")
        if str(data.get("backbone", "")).lower() != "unet":
            raise ValueError("Only the upstream StarDist 2D U-Net backbone is supported in V1.")
        if bool(data.get("unet_batch_norm", False)):
            raise ValueError("Batch-normalized StarDist models are not supported in V1.")
        if float(data.get("unet_dropout", 0.0)) != 0.0:
            raise ValueError("Dropout StarDist models are not supported in V1 inference graph.")
        if data.get("unet_activation", "relu") != "relu" or data.get("unet_last_activation", "relu") != "relu":
            raise ValueError("Only ReLU StarDist U-Net activations are supported in V1.")

        return cls(
            n_dim=2,
            axes=str(data["axes"]),
            n_channel_in=int(data["n_channel_in"]),
            n_channel_out=int(data["n_channel_out"]),
            n_rays=int(data["n_rays"]),
            grid=tuple(int(v) for v in data["grid"]),
            backbone=str(data["backbone"]).lower(),
            unet_n_depth=int(data["unet_n_depth"]),
            unet_kernel_size=tuple(int(v) for v in data["unet_kernel_size"]),
            unet_n_filter_base=int(data["unet_n_filter_base"]),
            unet_n_conv_per_depth=int(data["unet_n_conv_per_depth"]),
            unet_pool=tuple(int(v) for v in data["unet_pool"]),
            unet_activation=str(data["unet_activation"]),
            unet_last_activation=str(data["unet_last_activation"]),
            unet_batch_norm=bool(data["unet_batch_norm"]),
            unet_dropout=float(data["unet_dropout"]),
            net_conv_after_unet=int(data["net_conv_after_unet"]),
            train_checkpoint=str(data.get("train_checkpoint", "weights_best.h5")),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "StarDist2DConfig":
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def axes_div_by(self) -> tuple[int, int]:
        return tuple((p**self.unet_n_depth) * g for p, g in zip(self.unet_pool, self.grid))

    def as_dict(self) -> dict[str, Any]:
        data = self.__dict__.copy()
        for key in ("grid", "unet_kernel_size", "unet_pool"):
            data[key] = list(data[key])
        return data


def load_thresholds(path: str | Path) -> dict[str, float]:
    path = Path(path)
    if not path.exists():
        return {"prob": 0.5, "nms": 0.4}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    prob = float(data.get("prob", 0.5))
    nms = float(data.get("nms", 0.4))
    if not 0.0 < prob < 1.0:
        prob = 0.5
    if not 0.0 < nms < 1.0:
        nms = 0.4
    return {"prob": prob, "nms": nms}
