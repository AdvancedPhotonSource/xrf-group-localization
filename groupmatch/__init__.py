"""Reference implementation for acquisition-geometry-assisted XRF localization."""

from .evaluation import group_iou, tile_iou_summary
from .io import load_optical_image, load_xrf_tile
from .multiscale import (
    localize_structural_bridge,
    localize_with_bridge_prior,
    localize_with_bridge_rule,
)
from .single_scale import (
    localize_anchor_verified_group,
    localize_group_mosaic,
    localize_independent_tiles,
)

__all__ = [
    "load_optical_image",
    "load_xrf_tile",
    "localize_independent_tiles",
    "localize_anchor_verified_group",
    "localize_group_mosaic",
    "localize_structural_bridge",
    "localize_with_bridge_prior",
    "localize_with_bridge_rule",
    "group_iou",
    "tile_iou_summary",
]
