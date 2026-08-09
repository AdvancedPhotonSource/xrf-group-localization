from __future__ import annotations

from collections.abc import Iterable, Sequence

import cv2
import numpy as np

from .types import XRFTile


def transform_points(transform: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    mapped = (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T
    return (mapped[:, :2] / mapped[:, 2:3]).astype(np.float64)


def image_corners(shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    return np.asarray(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float64,
    )


def tile_pixel_to_physical(tile: XRFTile) -> np.ndarray:
    height, width = tile.image.shape
    sx, sy = tile.step_um
    cx, cy = tile.center_um
    return np.asarray(
        [
            [sx, 0.0, cx - sx * (width - 1.0) / 2.0],
            [0.0, sy, cy - sy * (height - 1.0) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def tile_footprint(tile: XRFTile, physical_to_reference: np.ndarray) -> np.ndarray:
    return transform_points(
        np.asarray(physical_to_reference, dtype=np.float64) @ tile_pixel_to_physical(tile),
        image_corners(tile.image.shape),
    )


def polygon_area(polygon: np.ndarray) -> float:
    return float(abs(cv2.contourArea(np.asarray(polygon, dtype=np.float32))))


def visible_fraction(polygon: np.ndarray, reference_shape: tuple[int, int]) -> float:
    area = polygon_area(polygon)
    if area <= 1e-12:
        return 0.0
    height, width = reference_shape
    boundary = np.asarray(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float32,
    )
    intersection_area, _ = cv2.intersectConvexConvex(
        np.asarray(polygon, dtype=np.float32), boundary
    )
    return float(np.clip(float(intersection_area) / area, 0.0, 1.0))


def polygon_iou(first: np.ndarray, second: np.ndarray) -> float:
    area_first = polygon_area(first)
    area_second = polygon_area(second)
    intersection, _ = cv2.intersectConvexConvex(
        np.asarray(first, dtype=np.float32), np.asarray(second, dtype=np.float32)
    )
    union = area_first + area_second - float(intersection)
    return float(intersection / union) if union > 1e-12 else 0.0


def footprint_center(polygon: np.ndarray) -> np.ndarray:
    return np.asarray(polygon, dtype=np.float64).mean(axis=0)


def tile_weights(tiles: Sequence[XRFTile], mode: str) -> dict[str, float]:
    if mode == "uniform":
        return {tile.name: 1.0 for tile in tiles}
    if mode != "physical_sqrt":
        raise ValueError(f"Unsupported tile weight mode: {mode!r}")
    raw = {
        tile.name: float(
            np.sqrt(tile.image.shape[0] * tile.image.shape[1] * abs(tile.step_um[0] * tile.step_um[1]))
        )
        for tile in tiles
    }
    mean = float(np.mean(list(raw.values())))
    return {name: value / mean for name, value in raw.items()}


def _rasterized_union(polygons: Iterable[np.ndarray], bounds: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, width, height = bounds
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        shifted = np.rint(np.asarray(polygon) - np.asarray([x0, y0])).astype(np.int32)
        cv2.fillPoly(mask, [shifted], 1)
    return mask


def polygon_union_iou(first: Iterable[np.ndarray], second: Iterable[np.ndarray]) -> float:
    first_list = [np.asarray(p, dtype=np.float64) for p in first]
    second_list = [np.asarray(p, dtype=np.float64) for p in second]
    if not first_list or not second_list:
        return 0.0
    all_points = np.vstack(first_list + second_list)
    x0, y0 = np.floor(all_points.min(axis=0)).astype(int)
    x1, y1 = np.ceil(all_points.max(axis=0)).astype(int)
    width = max(1, int(x1 - x0 + 3))
    height = max(1, int(y1 - y0 + 3))
    if width * height > 100_000_000:
        raise ValueError("Footprint extent is too large for rasterized GroupIoU")
    bounds = (int(x0 - 1), int(y0 - 1), width, height)
    first_mask = _rasterized_union(first_list, bounds)
    second_mask = _rasterized_union(second_list, bounds)
    intersection = int(np.count_nonzero(first_mask & second_mask))
    union = int(np.count_nonzero(first_mask | second_mask))
    return float(intersection / union) if union else 0.0
