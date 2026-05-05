from __future__ import annotations

import numpy as np
from skimage.draw import polygon

try:
    import cv2 as _cv2
    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False


def ray_angles(n_rays: int = 32) -> np.ndarray:
    return np.linspace(0, 2 * np.pi, int(n_rays), endpoint=False)


def dist_to_coord(dist: np.ndarray, points: np.ndarray, scale_dist: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """Convert StarDist ray distances and center points to polygon coordinates.

    Adapted from the BSD-3-Clause StarDist 2D geometry implementation.
    Coordinates are returned as ``(n_polygons, 2, n_rays)`` in ``(row, col)``
    order.
    """

    dist = np.asarray(dist)
    points = np.asarray(points)
    if dist.ndim != 2 or points.ndim != 2 or points.shape[1] != 2 or len(dist) != len(points):
        raise ValueError("dist must be (n_polys, n_rays) and points must be (n_polys, 2).")

    phis = ray_angles(dist.shape[1])
    coord = (dist[:, np.newaxis] * np.array([np.sin(phis), np.cos(phis)])).astype(np.float32)
    coord *= np.asarray(scale_dist, dtype=np.float32).reshape(1, 2, 1)
    coord += points[..., np.newaxis]
    return coord


def polygons_to_label(
    dist: np.ndarray,
    points: np.ndarray,
    shape: tuple[int, int],
    prob: np.ndarray | None = None,
    scale_dist: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """Render StarDist polygons into a 2D instance-label image.

    Lower-probability polygons are drawn first so higher-probability polygons
    overwrite them in overlaps.
    """

    dist = np.asarray(dist)
    points = np.asarray(points)
    if len(points) == 0:
        return np.zeros(shape, dtype=np.int32)

    prob = np.ones(len(points), dtype=np.float32) if prob is None else np.asarray(prob, dtype=np.float32)
    order = np.argsort(prob, kind="stable")
    coord = dist_to_coord(dist[order], points[order], scale_dist=scale_dist)

    labels = np.zeros(shape, dtype=np.int32)
    if _HAVE_CV2:
        # Fast path: cv2.fillPoly is C++ and handles the whole polygon in one call.
        for label_id, poly in enumerate(coord, start=1):
            # poly is (2, n_rays) with row/col; cv2 wants (n_pts, 1, 2) as x(col),y(row)
            pts = np.stack([poly[1], poly[0]], axis=-1).astype(np.int32).reshape(-1, 1, 2)
            _cv2.fillPoly(labels, [pts], color=label_id)
    else:
        # Fallback: skimage polygon rasterization
        for label_id, poly in enumerate(coord, start=1):
            rr, cc = polygon(poly[0], poly[1], shape)
            labels[rr, cc] = label_id
    return labels
