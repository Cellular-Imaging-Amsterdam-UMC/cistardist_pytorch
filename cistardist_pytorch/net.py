from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from .config import StarDist2DConfig


class StarDist2DNet(nn.Module):
    """PyTorch graph matching upstream StarDist2D's Keras U-Net inference model."""

    def __init__(self, config: StarDist2DConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleDict()
        self.layer_specs: OrderedDict[str, tuple[int, int, int]] = OrderedDict()
        self._build_layers()

    def _add_conv(self, name: str, in_channels: int, out_channels: int, kernel_size: int) -> nn.Conv2d:
        padding = kernel_size // 2
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.layers[name] = conv
        self.layer_specs[name] = (in_channels, out_channels, kernel_size)
        return conv

    def _build_layers(self) -> None:
        cfg = self.config
        if cfg.unet_kernel_size[0] != cfg.unet_kernel_size[1]:
            raise ValueError("Only square 2D U-Net kernels are supported in V1.")
        if cfg.unet_pool != (2, 2):
            raise ValueError("Only U-Net pool=(2, 2) is supported in V1.")
        if cfg.grid != (2, 2):
            raise ValueError("V1 currently targets StarDist2D models with grid=(2, 2).")

        k = cfg.unet_kernel_size[0]
        base = cfg.unet_n_filter_base
        expansion = 2

        in_ch = cfg.n_channel_in
        pooled = [1, 1]
        stem_index = 1
        while tuple(pooled) != cfg.grid:
            pool = [1 + int(g > p) for g, p in zip(cfg.grid, pooled)]
            for _ in range(cfg.unet_n_conv_per_depth):
                self._add_conv(f"conv2d_{stem_index}", in_ch, base, k)
                in_ch = base
                stem_index += 1
            pooled = [p * s for p, s in zip(pooled, pool)]

        ch = base
        for depth in range(cfg.unet_n_depth):
            out_ch = int(base * expansion**depth)
            for idx in range(cfg.unet_n_conv_per_depth):
                self._add_conv(f"down_level_{depth}_no_{idx}", ch, out_ch, k)
                ch = out_ch

        for idx in range(cfg.unet_n_conv_per_depth - 1):
            out_ch = int(base * expansion**cfg.unet_n_depth)
            self._add_conv(f"middle_{idx}", ch, out_ch, k)
            ch = out_ch
        out_ch = int(base * expansion ** max(0, cfg.unet_n_depth - 1))
        self._add_conv(f"middle_{cfg.unet_n_conv_per_depth}", ch, out_ch, k)
        ch = out_ch

        for depth in reversed(range(cfg.unet_n_depth)):
            skip_ch = int(base * expansion**depth)
            ch += skip_ch
            out_ch = skip_ch
            for idx in range(cfg.unet_n_conv_per_depth - 1):
                self._add_conv(f"up_level_{depth}_no_{idx}", ch, out_ch, k)
                ch = out_ch
            out_ch = int(base * expansion ** max(0, depth - 1))
            self._add_conv(f"up_level_{depth}_no_{cfg.unet_n_conv_per_depth}", ch, out_ch, k)
            ch = out_ch

        if cfg.net_conv_after_unet > 0:
            self._add_conv("features", ch, cfg.net_conv_after_unet, k)
            ch = cfg.net_conv_after_unet

        self._add_conv("prob", ch, 1, 1)
        self._add_conv("dist", ch, cfg.n_rays, 1)

    @property
    def keras_layer_names(self) -> list[str]:
        return list(self.layer_specs.keys())

    def _relu_conv(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.layers[name](x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config

        pooled = [1, 1]
        stem_index = 1
        while tuple(pooled) != cfg.grid:
            pool = [1 + int(g > p) for g, p in zip(cfg.grid, pooled)]
            for _ in range(cfg.unet_n_conv_per_depth):
                x = self._relu_conv(f"conv2d_{stem_index}", x)
                stem_index += 1
            x = F.max_pool2d(x, kernel_size=tuple(pool), stride=tuple(pool))
            pooled = [p * s for p, s in zip(pooled, pool)]

        skips: list[torch.Tensor] = []
        for depth in range(cfg.unet_n_depth):
            for idx in range(cfg.unet_n_conv_per_depth):
                x = self._relu_conv(f"down_level_{depth}_no_{idx}", x)
            skips.append(x)
            x = F.max_pool2d(x, kernel_size=cfg.unet_pool, stride=cfg.unet_pool)

        for idx in range(cfg.unet_n_conv_per_depth - 1):
            x = self._relu_conv(f"middle_{idx}", x)
        x = self._relu_conv(f"middle_{cfg.unet_n_conv_per_depth}", x)

        for depth in reversed(range(cfg.unet_n_depth)):
            x = F.interpolate(x, scale_factor=cfg.unet_pool, mode="nearest")
            skip = skips[depth]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            for idx in range(cfg.unet_n_conv_per_depth - 1):
                x = self._relu_conv(f"up_level_{depth}_no_{idx}", x)
            x = self._relu_conv(f"up_level_{depth}_no_{cfg.unet_n_conv_per_depth}", x)

        if "features" in self.layers:
            x = self._relu_conv("features", x)

        prob = torch.sigmoid(self.layers["prob"](x))
        dist = self.layers["dist"](x)
        return prob, dist
