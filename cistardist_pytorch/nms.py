from __future__ import annotations

import numpy as np
from skimage.draw import polygon

from .geometry import dist_to_coord


def _candidate_mask(prob: np.ndarray, prob_thresh: float, b: int | None = 2) -> np.ndarray:
    mask = prob > prob_thresh
    if b is not None and b > 0:
        inner = np.zeros_like(mask, dtype=bool)
        if prob.shape[0] > 2 * b and prob.shape[1] > 2 * b:
            inner[b:-b, b:-b] = True
        mask &= inner
    return mask


def _bbox(poly: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    r0 = max(0, int(np.floor(np.min(poly[0]))))
    r1 = min(shape[0], int(np.ceil(np.max(poly[0]))) + 1)
    c0 = max(0, int(np.floor(np.min(poly[1]))))
    c1 = min(shape[1], int(np.ceil(np.max(poly[1]))) + 1)
    if r1 <= r0 or c1 <= c0:
        return None
    return r0, r1, c0, c1


def _rasterize(poly: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, r1, c0, c1 = bbox
    rr, cc = polygon(poly[0] - r0, poly[1] - c0, (r1 - r0, c1 - c0))
    mask = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    mask[rr, cc] = True
    return mask


def _overlap_smaller_denominator(
    poly_a: np.ndarray,
    bbox_a: tuple[int, int, int, int],
    area_a: int,
    poly_b: np.ndarray,
    bbox_b: tuple[int, int, int, int],
    area_b: int,
) -> float:
    r0 = max(bbox_a[0], bbox_b[0])
    r1 = min(bbox_a[1], bbox_b[1])
    c0 = max(bbox_a[2], bbox_b[2])
    c1 = min(bbox_a[3], bbox_b[3])
    if r1 <= r0 or c1 <= c0:
        return 0.0

    inter_bbox = (r0, r1, c0, c1)
    mask_a = _rasterize(poly_a, inter_bbox)
    mask_b = _rasterize(poly_b, inter_bbox)
    intersection = int(np.count_nonzero(mask_a & mask_b))
    denom = max(1, min(area_a, area_b))
    return intersection / denom


def non_maximum_suppression(
    dist: np.ndarray,
    prob: np.ndarray,
    grid: tuple[int, int] = (1, 1),
    prob_thresh: float = 0.5,
    nms_thresh: float = 0.5,
    b: int | None = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure Python 2D StarDist polygon NMS.

    The overlap criterion follows StarDist's convention: intersection area is
    divided by the smaller polygon area. This implementation is intentionally
    dependency-light and optimized for correctness/readability in V1.
    """

    prob = np.asarray(prob)
    dist = np.asarray(dist)
    if prob.ndim != 2 or dist.ndim != 3 or prob.shape != dist.shape[:2]:
        raise ValueError("prob must be (Y, X) and dist must be (Y, X, n_rays).")

    mask = _candidate_mask(prob, prob_thresh=prob_thresh, b=b)
    candidate_grid_points = np.stack(np.where(mask), axis=1)
    if len(candidate_grid_points) == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, dist.shape[-1]), dtype=np.float32),
        )

    scores = prob[mask].astype(np.float32, copy=False)
    candidate_dist = dist[mask].astype(np.float32, copy=False)
    points = candidate_grid_points * np.asarray(grid, dtype=np.float32).reshape(1, 2)

    order = np.argsort(scores)[::-1]
    points = points[order]
    scores = scores[order]
    candidate_dist = candidate_dist[order]
    coords = dist_to_coord(candidate_dist, points)

    canvas_shape = tuple(int(s) for s in (np.asarray(prob.shape) * np.asarray(grid)))
    kept: list[int] = []
    bboxes: list[tuple[int, int, int, int]] = []
    areas: list[int] = []

    for idx, coord in enumerate(coords):
        bbox = _bbox(coord, canvas_shape)
        if bbox is None:
            continue
        area = int(np.count_nonzero(_rasterize(coord, bbox)))
        if area == 0:
            continue

        suppress = False
        for kept_pos, kept_idx in enumerate(kept):
            overlap = _overlap_smaller_denominator(
                coord,
                bbox,
                area,
                coords[kept_idx],
                bboxes[kept_pos],
                areas[kept_pos],
            )
            if overlap > nms_thresh:
                suppress = True
                break

        if not suppress:
            kept.append(idx)
            bboxes.append(bbox)
            areas.append(area)

    keep = np.asarray(kept, dtype=np.int64)
    return points[keep], scores[keep], candidate_dist[keep]
