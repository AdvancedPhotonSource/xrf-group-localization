from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import pytest

from groupmatch import (
    group_iou,
    load_optical_image,
    load_xrf_tile,
    localize_anchor_verified_group,
    localize_group_mosaic,
    localize_independent_tiles,
    tile_iou_summary,
)
from groupmatch.types import MANUSCRIPT_GROUP_CONFIG


def _write_tile(path: Path, image: np.ndarray, center_xy: tuple[float, float], channel: str) -> None:
    half_y = (image.shape[0] - 1) / 2.0
    half_x = (image.shape[1] - 1) / 2.0
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("MAPS/XRF_fits", data=image[None, ...].astype(np.float32))
        h5_file.create_dataset("MAPS/channel_names", data=np.asarray([channel.encode()]))
        h5_file.create_dataset(
            "MAPS/x_axis", data=np.linspace(center_xy[0] - half_x, center_xy[0] + half_x, image.shape[1])
        )
        h5_file.create_dataset(
            "MAPS/y_axis", data=np.linspace(center_xy[1] - half_y, center_xy[1] + half_y, image.shape[0])
        )


def _square(center_xy: tuple[float, float], size: int) -> np.ndarray:
    half = (size - 1) / 2.0
    x, y = center_xy
    return np.asarray(
        [[x - half, y - half], [x + half, y - half], [x + half, y + half], [x - half, y + half]],
        dtype=np.float64,
    )


def test_scientific_file_loading_and_single_scale_methods(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    first = rng.uniform(0.1, 1.0, size=(21, 21)).astype(np.float32)
    second = rng.uniform(0.1, 1.0, size=(21, 21)).astype(np.float32)
    first[5:16, 9:12] += 1.0
    second[9:12, 4:17] += 1.0

    optical = np.zeros((180, 220), dtype=np.float32)
    optical[70:91, 110:131] = first + rng.normal(0.0, 0.025, first.shape).astype(np.float32)
    optical[70:91, 150:171] = second
    optical[70:91, 10:31] = first  # Ambiguous independent match without companion geometry.
    optical_path = tmp_path / "optical.tif"
    Image.fromarray(np.uint16(optical / optical.max() * 65535)).save(
        optical_path, dpi=(25400, 25400)
    )
    orientation_path = tmp_path / "optical.xml"
    orientation_path.write_text(
        '<image rotation="0" flip_horizontal="false" flip_vertical="false"/>', encoding="utf-8"
    )

    first_path = tmp_path / "tile_first.h5"
    second_path = tmp_path / "tile_second.h5"
    _write_tile(first_path, first, (120.0, 80.0), "P")
    _write_tile(second_path, second, (160.0, 80.0), "P")

    reference = load_optical_image(optical_path, orientation_xml=orientation_path)
    tiles = [
        load_xrf_tile(first_path, name="first", channels=["P"]),
        load_xrf_tile(second_path, name="second", channels=["P"]),
    ]
    assert reference.image.dtype == np.float32
    assert reference.pixels_per_um == pytest.approx(1.0, rel=1e-6)
    assert tiles[0].center_um == pytest.approx((120.0, 80.0))
    assert tiles[0].step_um == pytest.approx((1.0, 1.0))

    synthetic_config = replace(
        MANUSCRIPT_GROUP_CONFIG,
        search=replace(
            MANUSCRIPT_GROUP_CONFIG.search,
            feature="raw",
            angle_range_deg=0.0,
        ),
    )
    independent = localize_independent_tiles(reference, tiles, config=synthetic_config)
    anchor_verified = localize_anchor_verified_group(reference, tiles, config=synthetic_config)
    mosaic = localize_group_mosaic(reference, tiles, config=synthetic_config)
    assert independent is not None
    assert anchor_verified is not None
    assert mosaic is not None

    truth = {"first": _square((120.0, 80.0), 21), "second": _square((160.0, 80.0), 21)}
    assert group_iou(anchor_verified.footprints, truth) > 0.90
    assert group_iou(mosaic.footprints, truth) > 0.85
    assert group_iou(anchor_verified.footprints, truth) > group_iou(independent.footprints, truth)
    summary = tile_iou_summary(anchor_verified.footprints, truth)
    assert summary["minimum"] > 0.85
    assert summary["weighted_mean"] > 0.90
