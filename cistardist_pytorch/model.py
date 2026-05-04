from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import StarDist2DConfig, load_thresholds
from .converter import convert_model_folder
from .geometry import dist_to_coord, polygons_to_label
from .net import StarDist2DNet
from .nms import non_maximum_suppression


def normalize_percentile(image: np.ndarray, pmin: float = 1.0, pmax: float = 99.8, eps: float = 1e-20) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    lo, hi = np.percentile(image, (pmin, pmax))
    return (image - lo) / (hi - lo + eps)


def _auto_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _pad_to_div_by(image: np.ndarray, div_by: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    pad_y = (div_by[0] - image.shape[0] % div_by[0]) % div_by[0]
    pad_x = (div_by[1] - image.shape[1] % div_by[1]) % div_by[1]
    if pad_y == 0 and pad_x == 0:
        return image, (0, 0)
    mode = "reflect" if image.shape[0] > 1 and image.shape[1] > 1 else "edge"
    padded = np.pad(image, ((0, pad_y), (0, pad_x)), mode=mode)
    return padded, (pad_y, pad_x)


def _crop_prediction(array: np.ndarray, pad: tuple[int, int], grid: tuple[int, int]) -> np.ndarray:
    crop_y = pad[0] // grid[0]
    crop_x = pad[1] // grid[1]
    y_slice = slice(None, -crop_y if crop_y else None)
    x_slice = slice(None, -crop_x if crop_x else None)
    return array[y_slice, x_slice, ...] if array.ndim == 3 else array[y_slice, x_slice]


class StarDist2D:
    def __init__(
        self,
        net: StarDist2DNet,
        config: StarDist2DConfig,
        thresholds: dict[str, float] | None = None,
        device: str | torch.device = "auto",
    ):
        self.config = config
        self.thresholds = thresholds or {"prob": 0.5, "nms": 0.4}
        self.device = _auto_device(device)
        self.net = net.to(self.device).eval()

    @classmethod
    def from_folder(cls, model_dir: str | Path, device: str | torch.device = "auto") -> "StarDist2D":
        model_dir = Path(model_dir)
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json in {model_dir}. V1 requires config.json.")

        config = StarDist2DConfig.from_json(config_path)
        pt_path = model_dir / f"{Path(config.train_checkpoint).stem}.pt"
        if not pt_path.exists():
            convert_model_folder(model_dir)

        checkpoint = torch.load(pt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        thresholds = checkpoint.get("thresholds") or load_thresholds(model_dir / "thresholds.json")
        net = StarDist2DNet(config)
        net.load_state_dict(state_dict)
        return cls(net=net, config=config, thresholds=thresholds, device=device)

    def _prepare_image(self, image: np.ndarray, normalize: bool) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        image = np.asarray(image)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]
        if image.ndim != 2:
            raise ValueError("V1 supports 2D grayscale images or YXC images with one channel.")

        x = normalize_percentile(image) if normalize else image.astype(np.float32, copy=False)
        original_shape = tuple(int(v) for v in x.shape)
        x, pad = _pad_to_div_by(x, self.config.axes_div_by())
        return x.astype(np.float32, copy=False), original_shape, pad

    @torch.no_grad()
    def predict(self, image: np.ndarray, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
        x, _original_shape, pad = self._prepare_image(image, normalize=normalize)
        tensor = torch.from_numpy(x[None, None]).to(self.device)
        prob_t, dist_t = self.net(tensor)
        prob = prob_t[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
        dist = np.moveaxis(dist_t[0].detach().cpu().numpy(), 0, -1).astype(np.float32, copy=False)
        prob = _crop_prediction(prob, pad, self.config.grid)
        dist = _crop_prediction(dist, pad, self.config.grid)
        return prob, dist

    def predict_instances(
        self,
        image: np.ndarray,
        prob_thresh: float | None = None,
        nms_thresh: float | None = None,
        normalize: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        image_array = np.asarray(image)
        shape = image_array.shape[:2]
        prob_thresh = self.thresholds["prob"] if prob_thresh is None else float(prob_thresh)
        nms_thresh = self.thresholds["nms"] if nms_thresh is None else float(nms_thresh)

        prob, dist = self.predict(image_array, normalize=normalize)
        dist = np.maximum(dist, 1e-3)
        points, probi, disti = non_maximum_suppression(
            dist,
            prob,
            grid=self.config.grid,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
        )
        labels = polygons_to_label(disti, points, shape=shape, prob=probi)
        details = {
            "coord": dist_to_coord(disti, points) if len(points) else np.zeros((0, 2, self.config.n_rays), dtype=np.float32),
            "points": points,
            "prob": probi,
            "dist": disti,
            "prob_map": prob,
            "dist_map": dist,
        }
        return labels, details
