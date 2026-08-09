from __future__ import annotations

import cv2
import numpy as np

from .types import FeatureName


def normalize_percentile(
    image: np.ndarray,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        raise ValueError("Image contains no finite values")
    values = array[finite]
    low = float(np.percentile(values, low_percentile))
    high = float(np.percentile(values, high_percentile))
    clean = np.nan_to_num(array, nan=low, posinf=high, neginf=low)
    if high <= low:
        low = float(values.min())
        high = float(values.max())
    if high <= low:
        return np.zeros_like(clean, dtype=np.float32)
    return np.clip((clean - low) / (high - low), 0.0, 1.0).astype(np.float32)


def apply_feature(image: np.ndarray, feature: FeatureName) -> np.ndarray:
    image01 = np.asarray(image, dtype=np.float32)
    if image01.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image01.shape}")
    if feature == "raw":
        return image01
    if feature == "gradient_magnitude":
        gx = cv2.Sobel(image01, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image01, cv2.CV_32F, 0, 1, ksize=3)
        return normalize_percentile(cv2.magnitude(gx, gy), 2.0, 98.0)
    raise ValueError(f"Unsupported feature: {feature!r}")


def rotate_bound(
    image: np.ndarray,
    angle_deg: float,
    *,
    interpolation: int = cv2.INTER_LINEAR,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate an image without clipping and return raw-to-rotated transform."""

    height, width = image.shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    cosine = abs(float(matrix[0, 0]))
    sine = abs(float(matrix[0, 1]))
    out_width = max(1, int(np.ceil(height * sine + width * cosine)))
    out_height = max(1, int(np.ceil(height * cosine + width * sine)))
    matrix[0, 2] += (out_width - 1) / 2.0 - center[0]
    matrix[1, 2] += (out_height - 1) / 2.0 - center[1]
    rotated = cv2.warpAffine(
        image,
        matrix,
        (out_width, out_height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    transform = np.eye(3, dtype=np.float64)
    transform[:2, :] = matrix
    return rotated.astype(np.float32), transform


def apply_orientation(
    image: np.ndarray,
    rotation_deg: float,
    flip_horizontal: bool,
    flip_vertical: bool,
    *,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    oriented = np.asarray(image)
    if flip_horizontal:
        oriented = np.fliplr(oriented)
    if flip_vertical:
        oriented = np.flipud(oriented)
    if abs(float(rotation_deg)) > 1e-12:
        oriented, _ = rotate_bound(oriented, float(rotation_deg), interpolation=interpolation)
    return np.ascontiguousarray(oriented)
