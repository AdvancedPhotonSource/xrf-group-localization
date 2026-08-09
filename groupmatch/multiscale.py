from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from .geometry import (
    tile_footprint,
    tile_pixel_to_physical,
    tile_weights,
    transform_points,
    visible_fraction,
)
from .similarity import SearchCandidate, search_template
from .single_scale import _localize_anchor_verified_group
from .types import (
    LocalizationResult,
    MANUSCRIPT_MULTISCALE_CONFIG,
    MultiscaleConfig,
    OpticalImage,
    XRFTile,
)


def _stage1_bridge_candidate(
    optical: OpticalImage, bridge: XRFTile, config: MultiscaleConfig
) -> SearchCandidate | None:
    if optical.pixels_per_um is None or optical.pixels_per_um <= 0:
        raise ValueError("Optical pixels_per_um calibration is required")
    bridge_step = float(np.sqrt(abs(bridge.step_um[0] * bridge.step_um[1])))
    candidates = search_template(
        optical.image,
        bridge.image,
        bridge.valid_mask,
        expected_scale=float(optical.pixels_per_um * bridge_step),
        config=config.stage1_search,
        candidate_count=max(1, config.stage1_search.candidates_per_tile),
    )
    return candidates[0] if candidates else None


def _replace_route(
    result: LocalizationResult,
    *,
    route: str,
    coordinate_transform: np.ndarray | None = None,
) -> LocalizationResult:
    if coordinate_transform is None:
        return replace(result, route=route)
    group_transform = coordinate_transform @ result.group_transform
    return LocalizationResult(
        group_transform=group_transform,
        tile_transforms={
            name: coordinate_transform @ transform for name, transform in result.tile_transforms.items()
        },
        footprints={
            name: transform_points(coordinate_transform, polygon)
            for name, polygon in result.footprints.items()
        },
        score=result.score,
        matched_tiles=result.matched_tiles,
        matched_weight_fraction=result.matched_weight_fraction,
        visible_fraction=result.visible_fraction,
        visible_weight_fraction=result.visible_weight_fraction,
        route=route,
    )


def _final_visibility_valid(
    result: LocalizationResult,
    optical: OpticalImage,
    tiles: Sequence[XRFTile],
    config: MultiscaleConfig,
) -> LocalizationResult | None:
    group_config = config.structural_group if result.route == "structural_bridge" else config.bridge_prior_group
    weights = tile_weights(tiles, group_config.tile_weight_mode)
    predicted = (
        {tile.name: tile_footprint(tile, result.group_transform) for tile in tiles}
        if result.group_transform is not None
        else result.footprints
    )
    visible_names = {
        tile.name
        for tile in tiles
        if tile.name in predicted
        and visible_fraction(predicted[tile.name], optical.image.shape)
        >= group_config.minimum_tile_visible_fraction
    }
    count_fraction = len(visible_names) / len(tiles)
    weight_fraction = sum(weights[name] for name in visible_names) / sum(weights.values())
    if (
        count_fraction < group_config.minimum_visible_fraction
        or weight_fraction < group_config.minimum_visible_weight_fraction
    ):
        return None
    return replace(
        result,
        visible_fraction=float(count_fraction),
        visible_weight_fraction=float(weight_fraction),
    )


def localize_structural_bridge(
    optical: OpticalImage,
    bridge: XRFTile,
    target_tiles: Sequence[XRFTile],
    *,
    config: MultiscaleConfig = MANUSCRIPT_MULTISCALE_CONFIG,
) -> LocalizationResult | None:
    """Localize the target group inside the bridge and compose into optical coordinates."""

    bridge_candidate = _stage1_bridge_candidate(optical, bridge, config)
    if bridge_candidate is None:
        return None
    bridge_step = float(np.sqrt(abs(bridge.step_um[0] * bridge.step_um[1])))
    bridge_reference = OpticalImage(
        image=bridge.image,
        pixels_per_um=1.0 / bridge_step,
        source_path=bridge.source_path,
    )
    target_in_bridge = _localize_anchor_verified_group(
        bridge_reference,
        target_tiles,
        config=config.structural_group,
    )
    if target_in_bridge is None or target_in_bridge.group_transform is None:
        return None
    composed = _replace_route(
        target_in_bridge,
        route="structural_bridge",
        coordinate_transform=bridge_candidate.transform,
    )
    return _final_visibility_valid(composed, optical, target_tiles, config)


def localize_with_bridge_prior(
    optical: OpticalImage,
    bridge: XRFTile,
    target_tiles: Sequence[XRFTile],
    *,
    config: MultiscaleConfig = MANUSCRIPT_MULTISCALE_CONFIG,
) -> LocalizationResult | None:
    """Use bridge localization as a prior for direct optical-frame group localization."""

    bridge_candidate = _stage1_bridge_candidate(optical, bridge, config)
    if bridge_candidate is None:
        return None
    seed_physical_to_optical = bridge_candidate.transform @ np.linalg.inv(
        tile_pixel_to_physical(bridge)
    )
    direct = _localize_anchor_verified_group(
        optical,
        target_tiles,
        config=config.bridge_prior_group,
        seed_physical_to_reference=seed_physical_to_optical,
        seed_weight=config.bridge_prior_weight,
        seed_radius_px=config.bridge_prior_radius_px,
    )
    if direct is None:
        return None
    prior_result = _replace_route(direct, route="bridge_prior")
    return _final_visibility_valid(prior_result, optical, target_tiles, config)


def localize_with_bridge_rule(
    optical: OpticalImage,
    bridge: XRFTile,
    target_tiles: Sequence[XRFTile],
    *,
    config: MultiscaleConfig = MANUSCRIPT_MULTISCALE_CONFIG,
) -> LocalizationResult | None:
    """Run both routes and select structural bridge only when runtime-valid."""

    structural = localize_structural_bridge(
        optical, bridge, target_tiles, config=config
    )
    bridge_prior = localize_with_bridge_prior(
        optical, bridge, target_tiles, config=config
    )
    return structural if structural is not None else bridge_prior
