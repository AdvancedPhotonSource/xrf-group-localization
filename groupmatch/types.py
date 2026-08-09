from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


FeatureName = Literal["raw", "gradient_magnitude"]
MetricName = Literal["ncc", "mutual_information"]


@dataclass(frozen=True)
class OpticalImage:
    """Optical reference image and its calibration.

    ``pixels_per_um`` is read from TIFF resolution metadata when available.
    The image is a two-dimensional float32 array normalized to [0, 1].
    """

    image: np.ndarray
    pixels_per_um: float | None = None
    source_path: Path | None = None


@dataclass(frozen=True)
class XRFTile:
    """Prepared XRF image with acquisition geometry in micrometres.

    ``center_um`` is the scan centre (x, y). ``step_um`` is the signed
    physical step per image pixel (x, y). Image coordinates use (x, y), with
    rectangles reported as (x, y, width, height).
    """

    name: str
    image: np.ndarray
    center_um: tuple[float, float]
    step_um: tuple[float, float]
    valid_mask: np.ndarray
    channels: tuple[str, ...] = ()
    source_path: Path | None = None


@dataclass(frozen=True)
class TemplateSearchConfig:
    feature: FeatureName = "gradient_magnitude"
    metric: MetricName = "ncc"
    angle_center_deg: float = 0.0
    angle_range_deg: float = 18.0
    angle_step_deg: float = 3.0
    scale_span_fraction: float = 0.08
    scale_count: int = 1
    candidates_per_tile: int = 20
    mutual_information_candidates: int = 32


@dataclass(frozen=True)
class GroupLocalizationConfig:
    search: TemplateSearchConfig = field(default_factory=TemplateSearchConfig)
    local_radius_px: int = 24
    local_center_penalty: float = 0.25
    group_center_penalty: float = 0.20
    overlap_penalty: float = 0.0
    minimum_tile_score: float = 0.0
    minimum_matched_tiles: int = 2
    minimum_matched_fraction: float = 0.60
    minimum_matched_weight_fraction: float = 0.60
    minimum_tile_visible_fraction: float = 0.05
    minimum_visible_fraction: float = 0.60
    minimum_visible_weight_fraction: float = 0.60
    anchor_candidates: int = 20
    tile_weight_mode: Literal["uniform", "physical_sqrt"] = "physical_sqrt"


@dataclass(frozen=True)
class MultiscaleConfig:
    stage1_search: TemplateSearchConfig = field(
        default_factory=lambda: TemplateSearchConfig(
            feature="gradient_magnitude",
            metric="ncc",
            angle_center_deg=0.0,
            angle_range_deg=60.0,
            angle_step_deg=6.0,
            scale_span_fraction=0.05,
            scale_count=1,
            candidates_per_tile=3,
        )
    )
    structural_group: GroupLocalizationConfig = field(
        default_factory=lambda: GroupLocalizationConfig(
            search=TemplateSearchConfig(
                feature="raw",
                metric="ncc",
                angle_center_deg=0.0,
                angle_range_deg=45.0,
                angle_step_deg=5.0,
                scale_span_fraction=0.05,
                scale_count=1,
            )
        )
    )
    bridge_prior_group: GroupLocalizationConfig = field(
        default_factory=lambda: GroupLocalizationConfig(
            search=TemplateSearchConfig(
                feature="gradient_magnitude",
                metric="mutual_information",
                angle_center_deg=0.0,
                angle_range_deg=18.0,
                angle_step_deg=3.0,
                scale_span_fraction=0.05,
                scale_count=1,
            )
        )
    )
    bridge_prior_weight: float = 0.25
    bridge_prior_radius_px: int = 128


@dataclass(frozen=True)
class LocalizationResult:
    """Successful localization result; methods return ``None`` for no-match.

    Transforms are 3x3 homogeneous source-to-destination matrices.
    ``group_transform`` maps physical micrometre coordinates to optical pixels
    for group methods and is ``None`` for independent per-tile localization.
    """

    group_transform: np.ndarray | None
    tile_transforms: dict[str, np.ndarray]
    footprints: dict[str, np.ndarray]
    score: float
    matched_tiles: tuple[str, ...]
    matched_weight_fraction: float
    visible_fraction: float
    visible_weight_fraction: float
    route: Literal[
        "independent",
        "anchor_verified",
        "mosaic",
        "structural_bridge",
        "bridge_prior",
    ]


MANUSCRIPT_GROUP_CONFIG = GroupLocalizationConfig()
MANUSCRIPT_MULTISCALE_CONFIG = MultiscaleConfig()
