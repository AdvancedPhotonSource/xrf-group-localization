from __future__ import annotations

from pathlib import Path
from typing import Sequence
import xml.etree.ElementTree as ET

import cv2
import h5py
import numpy as np
from PIL import Image

from .preprocessing import apply_orientation, normalize_percentile
from .types import OpticalImage, XRFTile


_X_AXIS_KEYS = ("MAPS/x_axis", "MAPS/X_axis", "x_axis", "x_axis_um")
_Y_AXIS_KEYS = ("MAPS/y_axis", "MAPS/Y_axis", "y_axis", "y_axis_um")


def _resolution_pixels_per_um(path: Path) -> float | None:
    with Image.open(path) as image:
        tags = image.tag_v2
        x_resolution = tags.get(282)
        unit = int(tags.get(296, 1))
    if x_resolution is None:
        return None
    value = float(x_resolution)
    if not np.isfinite(value) or value <= 0:
        return None
    if unit == 2:  # pixels per inch
        return value / 25400.0
    if unit == 3:  # pixels per centimetre
        return value / 10000.0
    return None


def _read_orientation(path: Path | None) -> tuple[float, bool, bool]:
    if path is None:
        return 0.0, False, False
    root = ET.parse(path).getroot()

    def flag(name: str) -> bool:
        return str(root.attrib.get(name, "0")).strip().lower() in {"1", "true", "yes", "on"}

    return (
        float(root.attrib.get("rotation", "0")),
        flag("flip_horizontal"),
        flag("flip_vertical"),
    )


def load_optical_image(path: str | Path, *, orientation_xml: str | Path | None = None) -> OpticalImage:
    """Load a TIFF optical reference, preserving scientific bit depth."""

    source = Path(path)
    if source.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError(f"Optical input must be TIFF, got {source}")
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read optical image: {source}")
    if image.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        image = cv2.cvtColor(image, code)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D optical image, got shape {image.shape}")
    angle, flip_h, flip_v = _read_orientation(Path(orientation_xml) if orientation_xml else None)
    oriented = apply_orientation(image, angle, flip_h, flip_v)
    return OpticalImage(
        image=normalize_percentile(oriented, 0.0, 100.0),
        pixels_per_um=_resolution_pixels_per_um(source),
        source_path=source,
    )


def _axis(h5_file: h5py.File, keys: Sequence[str]) -> np.ndarray:
    for key in keys:
        if key in h5_file:
            values = np.asarray(h5_file[key][...], dtype=np.float64)
            if values.ndim == 1 and values.size >= 2 and np.isfinite(values).all():
                return values
    raise ValueError(f"Missing finite one-dimensional axis; tried {list(keys)}")


def _channel_names(h5_file: h5py.File) -> list[str] | None:
    if "MAPS/channel_names" not in h5_file:
        return None
    values = h5_file["MAPS/channel_names"][...]
    return [v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v) for v in values]


def _channel_axis(shape: tuple[int, ...], names: list[str] | None, requested: int | None) -> int:
    if len(shape) == 2:
        return 0
    if len(shape) != 3:
        raise ValueError(f"Expected a 2D or 3D XRF dataset, got shape {shape}")
    if requested is not None:
        axis = int(requested) % 3
    elif names is not None:
        matches = [index for index, size in enumerate(shape) if size == len(names)]
        if len(matches) != 1:
            raise ValueError(f"Cannot infer channel axis from shape {shape} and {len(names)} channel names")
        axis = matches[0]
    else:
        raise ValueError("channel_axis is required when MAPS/channel_names is absent")
    if names is not None and shape[axis] != len(names):
        raise ValueError("Channel name count does not match the selected channel axis")
    return axis


def _select_channels(
    cube: np.ndarray,
    names: list[str] | None,
    channels: Sequence[str | int] | None,
    automatic_channel_count: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if channels is not None:
        indices: list[int] = []
        for channel in channels:
            if isinstance(channel, str):
                if names is None or channel not in names:
                    raise ValueError(f"XRF channel {channel!r} is unavailable")
                indices.append(names.index(channel))
            else:
                index = int(channel)
                if index < 0:
                    index += cube.shape[0]
                if index < 0 or index >= cube.shape[0]:
                    raise IndexError(f"XRF channel index {channel} is out of range")
                indices.append(index)
    elif automatic_channel_count > 0:
        variances = np.nanvar(cube.reshape(cube.shape[0], -1), axis=1)
        indices = np.argsort(variances)[::-1][: int(automatic_channel_count)].tolist()
    else:
        indices = list(range(cube.shape[0]))
    if not indices:
        raise ValueError("At least one XRF channel is required")
    selected_names = tuple(names[i] if names is not None else str(i) for i in indices)
    return cube[np.asarray(indices, dtype=int)], selected_names


def load_xrf_tile(
    path: str | Path,
    *,
    name: str | None = None,
    dataset_key: str = "MAPS/XRF_fits",
    channels: Sequence[str | int] | None = None,
    automatic_channel_count: int = 0,
    channel_axis: int | None = None,
    border_shave: int = 0,
    isotropic: bool = True,
) -> XRFTile:
    """Load and prepare one MAPS XRF tile from an explicit HDF5 dataset."""

    source = Path(path)
    if source.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError(f"XRF input must be HDF5, got {source}")
    with h5py.File(source, "r") as h5_file:
        if dataset_key not in h5_file:
            raise ValueError(f"Dataset {dataset_key!r} not found in {source}")
        dataset = h5_file[dataset_key]
        names = _channel_names(h5_file)
        axis = _channel_axis(tuple(dataset.shape), names, channel_axis)
        raw = np.asarray(dataset[...], dtype=np.float32)
        cube = raw[None, ...] if raw.ndim == 2 else np.moveaxis(raw, axis, 0)
        x_axis = _axis(h5_file, _X_AXIS_KEYS)
        y_axis = _axis(h5_file, _Y_AXIS_KEYS)

    if cube.shape[1:] != (y_axis.size, x_axis.size):
        raise ValueError(
            f"XRF spatial shape {cube.shape[1:]} does not match axes {(y_axis.size, x_axis.size)}"
        )
    shave = int(border_shave)
    if shave > 0:
        if min(cube.shape[1:]) <= 2 * shave:
            raise ValueError("border_shave removes the entire XRF tile")
        cube = cube[:, shave:-shave, shave:-shave]
        x_axis = x_axis[shave:-shave]
        y_axis = y_axis[shave:-shave]

    cube, selected_names = _select_channels(cube, names, channels, automatic_channel_count)
    sx = float(np.median(np.diff(x_axis)))
    sy = float(np.median(np.diff(y_axis)))
    if not np.isfinite([sx, sy]).all() or abs(sx) < 1e-12 or abs(sy) < 1e-12:
        raise ValueError("XRF axes must have finite non-zero pixel steps")

    mask = np.ones(cube.shape[1:], dtype=np.float32)
    if isotropic:
        step = float(np.sqrt(abs(sx * sy)))
        scale_x = abs(sx) / step
        scale_y = abs(sy) / step
        width = max(1, int(round(cube.shape[2] * scale_x)))
        height = max(1, int(round(cube.shape[1] * scale_y)))
        cube = np.stack(
            [cv2.resize(channel, (width, height), interpolation=cv2.INTER_LINEAR) for channel in cube],
            axis=0,
        )
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        sx = np.copysign(step, sx)
        sy = np.copysign(step, sy)

    normalized = np.stack([normalize_percentile(channel, 2.0, 98.0) for channel in cube], axis=0)
    fused = normalized.mean(axis=0).astype(np.float32)
    return XRFTile(
        name=name or source.stem,
        image=fused,
        center_um=(float(np.mean(x_axis)), float(np.mean(y_axis))),
        step_um=(sx, sy),
        valid_mask=np.clip(mask, 0.0, 1.0).astype(np.float32),
        channels=selected_names,
        source_path=source,
    )
