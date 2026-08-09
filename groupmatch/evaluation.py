from __future__ import annotations

import numpy as np

from .geometry import polygon_iou, polygon_union_iou


def group_iou(
    predicted: dict[str, np.ndarray], ground_truth: dict[str, np.ndarray]
) -> float:
    """Intersection-over-union of predicted and ground-truth tile unions."""

    if not predicted or not ground_truth:
        return 0.0
    return polygon_union_iou(
        predicted.values(), ground_truth.values()
    )


def tile_iou_summary(
    predicted: dict[str, np.ndarray],
    ground_truth: dict[str, np.ndarray],
    *,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Minimum, mean, and weighted-mean TileIoU over ground-truth tiles."""

    if not ground_truth:
        raise ValueError("ground_truth must contain at least one tile")
    names = sorted(ground_truth)
    values = np.asarray(
        [polygon_iou(predicted[name], ground_truth[name]) if name in predicted else 0.0 for name in names],
        dtype=np.float64,
    )
    tile_weights = np.asarray(
        [float(weights[name]) if weights is not None else 1.0 for name in names], dtype=np.float64
    )
    if np.any(tile_weights < 0) or float(tile_weights.sum()) <= 0:
        raise ValueError("TileIoU weights must be non-negative with a positive sum")
    return {
        "minimum": float(values.min()),
        "mean": float(values.mean()),
        "weighted_mean": float(np.average(values, weights=tile_weights)),
    }
