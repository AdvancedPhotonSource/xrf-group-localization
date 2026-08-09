from dataclasses import replace
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image

from groupmatch import (
    group_iou,
    load_optical_image,
    load_xrf_tile,
    localize_structural_bridge,
    localize_with_bridge_prior,
    localize_with_bridge_rule,
)
from groupmatch.types import MANUSCRIPT_MULTISCALE_CONFIG


def _write_tile(
    path: Path,
    image: np.ndarray,
    center_xy: tuple[float, float],
    step: float,
    channel: str,
) -> None:
    half_y = (image.shape[0] - 1) * step / 2.0
    half_x = (image.shape[1] - 1) * step / 2.0
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("MAPS/XRF_fits", data=image[None, ...].astype(np.float32))
        h5_file.create_dataset("MAPS/channel_names", data=np.asarray([channel.encode()]))
        h5_file.create_dataset(
            "MAPS/x_axis", data=np.linspace(center_xy[0] - half_x, center_xy[0] + half_x, image.shape[1])
        )
        h5_file.create_dataset(
            "MAPS/y_axis", data=np.linspace(center_xy[1] - half_y, center_xy[1] + half_y, image.shape[0])
        )


def _square(center_xy: tuple[float, float], size: int, offset_xy: tuple[float, float]) -> np.ndarray:
    half = (size - 1) / 2.0
    x = center_xy[0] + offset_xy[0]
    y = center_xy[1] + offset_xy[1]
    return np.asarray(
        [[x - half, y - half], [x + half, y - half], [x + half, y + half], [x - half, y + half]],
        dtype=np.float64,
    )


def test_multiscale_routes_and_runtime_fallback(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    first = (rng.uniform(size=(21, 21)) > 0.62).astype(np.float32)
    second = (rng.uniform(size=(21, 21)) > 0.62).astype(np.float32)
    first[4:17, 8:13] = 1.0
    second[8:13, 4:17] = 1.0

    bridge = np.zeros((101, 101), dtype=np.float32)
    bridge[45:55, 35:45] = cv2.resize(first, (10, 10), interpolation=cv2.INTER_AREA)
    bridge[45:55, 55:65] = cv2.resize(second, (10, 10), interpolation=cv2.INTER_AREA)
    bridge_scaled = cv2.resize(bridge, (202, 202), interpolation=cv2.INTER_LINEAR)
    optical = bridge_scaled

    optical_path = tmp_path / "optical.tif"
    Image.fromarray(np.uint16(optical / optical.max() * 65535)).save(
        optical_path, dpi=(25400, 25400)
    )
    bridge_path = tmp_path / "bridge.h5"
    first_path = tmp_path / "target_first.h5"
    second_path = tmp_path / "target_second.h5"
    _write_tile(bridge_path, bridge, (100.0, 100.0), 2.0, "P")
    _write_tile(first_path, first, (80.0, 100.0), 1.0, "P")
    _write_tile(second_path, second, (120.0, 100.0), 1.0, "P")

    reference = load_optical_image(optical_path)
    bridge_tile = load_xrf_tile(bridge_path, name="bridge", channels=["P"])
    targets = [
        load_xrf_tile(first_path, name="first", channels=["P"]),
        load_xrf_tile(second_path, name="second", channels=["P"]),
    ]

    structural = localize_structural_bridge(reference, bridge_tile, targets)
    bridge_prior = localize_with_bridge_prior(reference, bridge_tile, targets)
    selected = localize_with_bridge_rule(reference, bridge_tile, targets)
    assert structural is not None
    assert bridge_prior is not None
    assert selected is not None
    assert selected.route == "structural_bridge"

    truth = {
        "first": _square((80.0, 100.0), 21, (0.0, 0.0)),
        "second": _square((120.0, 100.0), 21, (0.0, 0.0)),
    }
    assert group_iou(structural.footprints, truth) > 0.75
    assert group_iou(bridge_prior.footprints, truth) > 0.65

    invalid_structural = replace(
        MANUSCRIPT_MULTISCALE_CONFIG,
        structural_group=replace(
            MANUSCRIPT_MULTISCALE_CONFIG.structural_group,
            minimum_matched_tiles=3,
        ),
    )
    fallback = localize_with_bridge_rule(
        reference, bridge_tile, targets, config=invalid_structural
    )
    assert fallback is not None
    assert fallback.route == "bridge_prior"
    assert fallback.matched_weight_fraction >= 0.60
    assert fallback.visible_fraction >= 0.60
