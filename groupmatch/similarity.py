from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import image_corners, polygon_iou, transform_points
from .preprocessing import apply_feature, rotate_bound
from .types import MetricName, TemplateSearchConfig


@dataclass(frozen=True)
class SearchCandidate:
    transform: np.ndarray
    footprint: np.ndarray
    score: float
    angle_deg: float
    scale: float

    @property
    def center_xy(self) -> tuple[float, float]:
        center = self.footprint.mean(axis=0)
        return float(center[0]), float(center[1])


def _values(start: float, half_range: float, step: float) -> list[float]:
    if half_range <= 0 or step <= 0:
        return [float(start)]
    count = int(np.floor((2.0 * half_range) / step))
    return [float(start - half_range + index * step) for index in range(count + 1)]


def scale_candidates(center: float, span_fraction: float, count: int) -> list[float]:
    if center <= 0 or not np.isfinite(center):
        raise ValueError("Scale centre must be finite and positive")
    if count <= 1:
        return [float(center)]
    low = center * max(1e-6, 1.0 - float(span_fraction))
    high = center * (1.0 + float(span_fraction))
    return np.geomspace(low, high, int(count)).astype(float).tolist()


def normalized_mutual_information(
    first: np.ndarray, second: np.ndarray, mask: np.ndarray | None = None, bins: int = 64
) -> float:
    a = np.asarray(first, dtype=np.float32).ravel()
    b = np.asarray(second, dtype=np.float32).ravel()
    if mask is not None:
        keep = np.asarray(mask, dtype=np.float32).ravel() > 0.5
        a = a[keep]
        b = b[keep]
    finite = np.isfinite(a) & np.isfinite(b)
    a = np.clip(a[finite], 0.0, 1.0)
    b = np.clip(b[finite], 0.0, 1.0)
    if a.size < 4:
        return float("-inf")
    histogram, _, _ = np.histogram2d(a, b, bins=int(bins), range=((0.0, 1.0), (0.0, 1.0)))
    total = float(histogram.sum())
    if total <= 0:
        return float("-inf")
    joint = histogram / total
    pa = joint.sum(axis=1)
    pb = joint.sum(axis=0)

    def entropy(probability: np.ndarray) -> float:
        positive = probability[probability > 0]
        return float(-np.sum(positive * np.log(positive)))

    joint_entropy = entropy(joint)
    if joint_entropy <= 1e-12:
        return 0.0
    return float((entropy(pa) + entropy(pb)) / joint_entropy)


def aligned_score(
    reference_patch: np.ndarray,
    template: np.ndarray,
    metric: MetricName,
    mask: np.ndarray | None = None,
) -> float:
    reference = np.asarray(reference_patch, dtype=np.float32)
    target = np.asarray(template, dtype=np.float32)
    if reference.shape != target.shape:
        raise ValueError(f"Aligned score shape mismatch: {reference.shape} != {target.shape}")
    if metric == "mutual_information":
        return normalized_mutual_information(reference, target, mask)
    if metric != "ncc":
        raise ValueError(f"Unsupported metric: {metric!r}")
    weights = np.ones_like(target, dtype=np.float32) if mask is None else np.asarray(mask, dtype=np.float32)
    keep = weights > 0.5
    if int(np.count_nonzero(keep)) < 4:
        return float("-inf")
    a = reference[keep].astype(np.float64)
    b = target[keep].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else 0.0


def _response_peaks(response: np.ndarray, count: int) -> list[tuple[int, int, float]]:
    flat = np.asarray(response, dtype=np.float32).ravel()
    take = min(int(count), flat.size)
    if take <= 0:
        return []
    indices = np.argpartition(flat, flat.size - take)[-take:]
    indices = indices[np.argsort(flat[indices])[::-1]]
    width = response.shape[1]
    return [(int(index % width), int(index // width), float(flat[index])) for index in indices]


def _candidate_transform(
    raw_to_rotated: np.ndarray, scale: float, top_left_xy: tuple[int, int]
) -> np.ndarray:
    scale_transform = np.diag([float(scale), float(scale), 1.0]).astype(np.float64)
    translation = np.eye(3, dtype=np.float64)
    translation[:2, 2] = np.asarray(top_left_xy, dtype=np.float64)
    return translation @ scale_transform @ raw_to_rotated


def search_template(
    reference: np.ndarray,
    template: np.ndarray,
    valid_mask: np.ndarray,
    *,
    expected_scale: float,
    config: TemplateSearchConfig,
    reference_offset_xy: tuple[int, int] = (0, 0),
    angle_center_deg: float | None = None,
    angle_range_deg: float | None = None,
    scale_span_fraction: float | None = None,
    scale_count: int | None = None,
    candidate_count: int | None = None,
) -> list[SearchCandidate]:
    """Search one template and return non-overlapping candidates in reference coordinates."""

    reference_feature = apply_feature(reference, config.feature)
    template_feature = apply_feature(template, config.feature)
    angles = _values(
        config.angle_center_deg if angle_center_deg is None else angle_center_deg,
        config.angle_range_deg if angle_range_deg is None else angle_range_deg,
        config.angle_step_deg,
    )
    scales = scale_candidates(
        expected_scale,
        config.scale_span_fraction if scale_span_fraction is None else scale_span_fraction,
        config.scale_count if scale_count is None else scale_count,
    )
    requested = int(candidate_count or config.candidates_per_tile)
    raw_candidates: list[SearchCandidate] = []
    offset = np.asarray(reference_offset_xy, dtype=np.float64)

    for angle in angles:
        rotated_template, raw_to_rotated = rotate_bound(template_feature, angle)
        rotated_mask, _ = rotate_bound(valid_mask, angle, interpolation=cv2.INTER_NEAREST)
        for scale in scales:
            width = max(1, int(round(rotated_template.shape[1] * scale)))
            height = max(1, int(round(rotated_template.shape[0] * scale)))
            if height > reference_feature.shape[0] or width > reference_feature.shape[1]:
                continue
            scaled_template = cv2.resize(rotated_template, (width, height), interpolation=cv2.INTER_LINEAR)
            scaled_mask = cv2.resize(rotated_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            search_mask = np.clip(scaled_mask, 0.0, 1.0).astype(np.float32)
            if bool(np.all(search_mask > 0.999)):
                response = cv2.matchTemplate(
                    reference_feature, scaled_template, cv2.TM_CCORR_NORMED
                )
            else:
                response = cv2.matchTemplate(
                    reference_feature,
                    scaled_template,
                    cv2.TM_CCORR_NORMED,
                    mask=search_mask,
                )
            response = np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0)
            peak_count = requested
            if config.metric == "mutual_information":
                peak_count = max(peak_count, int(config.mutual_information_candidates))
            for x, y, _ in _response_peaks(response, peak_count):
                patch = reference_feature[y : y + height, x : x + width]
                score = aligned_score(patch, scaled_template, config.metric, scaled_mask)
                transform = _candidate_transform(raw_to_rotated, scale, (x, y))
                transform[:2, 2] += offset
                footprint = transform_points(transform, image_corners(template.shape))
                raw_candidates.append(
                    SearchCandidate(
                        transform=transform,
                        footprint=footprint,
                        score=float(score),
                        angle_deg=float(angle),
                        scale=float(scale),
                    )
                )

    selected: list[SearchCandidate] = []
    for candidate in sorted(raw_candidates, key=lambda item: item.score, reverse=True):
        if not np.isfinite(candidate.score):
            continue
        if all(polygon_iou(candidate.footprint, kept.footprint) < 0.25 for kept in selected):
            selected.append(candidate)
        if len(selected) >= requested:
            break
    return selected
