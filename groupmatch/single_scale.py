from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from itertools import combinations

import cv2
import numpy as np

from .geometry import (
    footprint_center,
    polygon_iou,
    tile_footprint,
    tile_pixel_to_physical,
    tile_weights,
    transform_points,
    visible_fraction,
)
from .similarity import SearchCandidate, search_template
from .types import (
    GroupLocalizationConfig,
    LocalizationResult,
    MANUSCRIPT_GROUP_CONFIG,
    OpticalImage,
    XRFTile,
)


def _require_unique_tiles(tiles: Sequence[XRFTile]) -> None:
    if not tiles:
        raise ValueError("At least one XRF tile is required")
    names = [tile.name for tile in tiles]
    if len(set(names)) != len(names):
        raise ValueError("XRF tile names must be unique")
    for tile in tiles:
        if tile.image.ndim != 2 or tile.valid_mask.shape != tile.image.shape:
            raise ValueError(f"Invalid image or mask shape for tile {tile.name!r}")
        if not np.isfinite(tile.image).all():
            raise ValueError(f"Tile {tile.name!r} contains non-finite values")


def _reference_scale(reference: OpticalImage, tile: XRFTile) -> float:
    if reference.pixels_per_um is None or reference.pixels_per_um <= 0:
        raise ValueError("Optical pixels_per_um calibration is required")
    tile_step = float(np.sqrt(abs(tile.step_um[0] * tile.step_um[1])))
    return float(reference.pixels_per_um * tile_step)


def _decompose_tile_transform(transform: np.ndarray) -> tuple[float, float]:
    linear = np.asarray(transform, dtype=np.float64)[:2, :2]
    scale = float(np.sqrt(abs(np.linalg.det(linear))))
    angle = float(np.rad2deg(np.arctan2(linear[0, 1], linear[0, 0])))
    return scale, angle


def _crop_for_polygon(
    reference: np.ndarray, polygon: np.ndarray, radius: int
) -> tuple[np.ndarray, tuple[int, int]] | None:
    minimum = np.floor(polygon.min(axis=0)).astype(int) - int(radius)
    maximum = np.ceil(polygon.max(axis=0)).astype(int) + int(radius) + 1
    x0 = max(0, int(minimum[0]))
    y0 = max(0, int(minimum[1]))
    x1 = min(reference.shape[1], int(maximum[0]))
    y1 = min(reference.shape[0], int(maximum[1]))
    if x1 <= x0 or y1 <= y0:
        return None
    return reference[y0:y1, x0:x1], (x0, y0)


def _visibility_support(
    footprints: dict[str, np.ndarray],
    tiles: Sequence[XRFTile],
    reference_shape: tuple[int, int],
    weights: dict[str, float],
    minimum_tile_fraction: float,
) -> tuple[float, float]:
    visible_names = {
        tile.name
        for tile in tiles
        if tile.name in footprints
        and visible_fraction(footprints[tile.name], reference_shape) >= minimum_tile_fraction
    }
    count_fraction = len(visible_names) / len(tiles)
    weight_fraction = sum(weights[name] for name in visible_names) / sum(weights.values())
    return float(count_fraction), float(weight_fraction)


def _result_is_runtime_valid(
    *,
    matched_names: Sequence[str],
    matched_weight_fraction: float,
    visible_count_fraction: float,
    visible_weight_fraction: float,
    tile_count: int,
    config: GroupLocalizationConfig,
) -> bool:
    return (
        len(matched_names) >= int(config.minimum_matched_tiles)
        and len(matched_names) / tile_count >= float(config.minimum_matched_fraction)
        and matched_weight_fraction >= float(config.minimum_matched_weight_fraction)
        and visible_count_fraction >= float(config.minimum_visible_fraction)
        and visible_weight_fraction >= float(config.minimum_visible_weight_fraction)
    )


def localize_independent_tiles(
    optical: OpticalImage,
    tiles: Sequence[XRFTile],
    *,
    config: GroupLocalizationConfig = MANUSCRIPT_GROUP_CONFIG,
) -> LocalizationResult | None:
    """Localize every tile separately, without acquisition-geometry coupling."""

    _require_unique_tiles(tiles)
    tile_transforms: dict[str, np.ndarray] = {}
    footprints: dict[str, np.ndarray] = {}
    scores: list[float] = []
    for tile in tiles:
        candidates = search_template(
            optical.image,
            tile.image,
            tile.valid_mask,
            expected_scale=_reference_scale(optical, tile),
            config=replace(config.search, candidates_per_tile=1),
            candidate_count=1,
        )
        if not candidates or candidates[0].score < config.minimum_tile_score:
            continue
        candidate = candidates[0]
        tile_transforms[tile.name] = candidate.transform
        footprints[tile.name] = candidate.footprint
        scores.append(candidate.score)
    if not scores:
        return None
    weights = tile_weights(tiles, config.tile_weight_mode)
    matched_weight = sum(weights[name] for name in tile_transforms) / sum(weights.values())
    visible_count, visible_weight = _visibility_support(
        footprints,
        tiles,
        optical.image.shape,
        weights,
        config.minimum_tile_visible_fraction,
    )
    return LocalizationResult(
        group_transform=None,
        tile_transforms=tile_transforms,
        footprints=footprints,
        score=float(np.mean(scores)),
        matched_tiles=tuple(sorted(tile_transforms)),
        matched_weight_fraction=float(matched_weight),
        visible_fraction=visible_count,
        visible_weight_fraction=visible_weight,
        route="independent",
    )


def _local_candidates(
    reference: np.ndarray,
    tile: XRFTile,
    predicted_transform: np.ndarray,
    predicted_footprint: np.ndarray,
    config: GroupLocalizationConfig,
) -> list[tuple[SearchCandidate, float, float]]:
    cropped = _crop_for_polygon(reference, predicted_footprint, config.local_radius_px)
    if cropped is None:
        return []
    crop, offset = cropped
    expected_scale, expected_angle = _decompose_tile_transform(predicted_transform)
    candidates = search_template(
        crop,
        tile.image,
        tile.valid_mask,
        expected_scale=expected_scale,
        config=config.search,
        reference_offset_xy=offset,
        angle_center_deg=expected_angle,
        angle_range_deg=0.0,
        scale_span_fraction=0.0,
        scale_count=1,
        candidate_count=8,
    )
    predicted_center = footprint_center(predicted_footprint)
    diagonal = max(1.0, float(np.linalg.norm(predicted_footprint[0] - predicted_footprint[2])))
    ranked: list[tuple[SearchCandidate, float, float]] = []
    for candidate in candidates:
        residual = float(np.linalg.norm(footprint_center(candidate.footprint) - predicted_center) / diagonal)
        penalized = float(candidate.score - config.local_center_penalty * residual)
        ranked.append((candidate, residual, penalized))
    return sorted(ranked, key=lambda item: item[2], reverse=True)


def _overlap_disagreement(
    predicted: dict[str, np.ndarray], verified: dict[str, np.ndarray]
) -> float:
    names = sorted(set(predicted) & set(verified))
    values = [
        abs(polygon_iou(predicted[a], predicted[b]) - polygon_iou(verified[a], verified[b]))
        for a, b in combinations(names, 2)
    ]
    return float(np.mean(values)) if values else 0.0


def _localize_anchor_verified_group(
    optical: OpticalImage,
    tiles: Sequence[XRFTile],
    *,
    config: GroupLocalizationConfig,
    seed_physical_to_reference: np.ndarray | None = None,
    seed_weight: float = 0.0,
    seed_radius_px: int | None = None,
) -> LocalizationResult | None:
    _require_unique_tiles(tiles)
    if len(tiles) < 2:
        raise ValueError("Anchor-verified localization requires at least two tiles")
    weights = tile_weights(tiles, config.tile_weight_mode)
    total_weight = float(sum(weights.values()))
    best: LocalizationResult | None = None

    for anchor in tiles:
        search_reference = optical.image
        offset = (0, 0)
        expected_scale = _reference_scale(optical, anchor)
        angle_center = config.search.angle_center_deg
        if seed_physical_to_reference is not None:
            seed_tile_transform = seed_physical_to_reference @ tile_pixel_to_physical(anchor)
            seed_footprint = tile_footprint(anchor, seed_physical_to_reference)
            cropped = _crop_for_polygon(
                optical.image,
                seed_footprint,
                int(seed_radius_px if seed_radius_px is not None else config.local_radius_px),
            )
            if cropped is None:
                continue
            search_reference, offset = cropped
            expected_scale, angle_center = _decompose_tile_transform(seed_tile_transform)

        anchor_candidates = search_template(
            search_reference,
            anchor.image,
            anchor.valid_mask,
            expected_scale=expected_scale,
            config=config.search,
            reference_offset_xy=offset,
            angle_center_deg=angle_center,
            candidate_count=config.anchor_candidates,
        )
        for anchor_candidate in anchor_candidates:
            physical_to_reference = anchor_candidate.transform @ np.linalg.inv(tile_pixel_to_physical(anchor))
            predicted = {tile.name: tile_footprint(tile, physical_to_reference) for tile in tiles}
            verified: dict[str, np.ndarray] = {anchor.name: anchor_candidate.footprint}
            tile_transforms: dict[str, np.ndarray] = {anchor.name: anchor_candidate.transform}
            raw_scores: dict[str, float] = {anchor.name: anchor_candidate.score}
            residuals: dict[str, float] = {anchor.name: 0.0}

            for tile in tiles:
                if tile.name == anchor.name:
                    continue
                predicted_transform = physical_to_reference @ tile_pixel_to_physical(tile)
                local = _local_candidates(
                    optical.image, tile, predicted_transform, predicted[tile.name], config
                )
                if not local or local[0][0].score < config.minimum_tile_score:
                    continue
                candidate, residual, _ = local[0]
                verified[tile.name] = candidate.footprint
                tile_transforms[tile.name] = candidate.transform
                raw_scores[tile.name] = candidate.score
                residuals[tile.name] = residual

            matched = tuple(sorted(verified))
            matched_weight = sum(weights[name] for name in matched)
            matched_weight_fraction = matched_weight / total_weight
            visible_count, visible_weight = _visibility_support(
                predicted,
                tiles,
                optical.image.shape,
                weights,
                config.minimum_tile_visible_fraction,
            )
            if not _result_is_runtime_valid(
                matched_names=matched,
                matched_weight_fraction=matched_weight_fraction,
                visible_count_fraction=visible_count,
                visible_weight_fraction=visible_weight,
                tile_count=len(tiles),
                config=config,
            ):
                continue

            similarity_score = sum(weights[name] * raw_scores[name] for name in matched) / total_weight
            companion_names = [name for name in matched if name != anchor.name]
            if companion_names:
                residual_score = sum(weights[name] * residuals[name] for name in companion_names) / sum(
                    weights[name] for name in companion_names
                )
            else:
                residual_score = 0.0
            score = float(
                similarity_score
                - config.group_center_penalty * residual_score
                - config.overlap_penalty * _overlap_disagreement(predicted, verified)
            )
            if seed_physical_to_reference is not None and seed_weight > 0:
                predicted_seed = tile_footprint(anchor, seed_physical_to_reference)
                distance = np.linalg.norm(
                    footprint_center(anchor_candidate.footprint) - footprint_center(predicted_seed)
                )
                normalizer = max(1.0, np.linalg.norm(predicted_seed[0] - predicted_seed[2]))
                score -= float(seed_weight) * float(distance / normalizer)

            result = LocalizationResult(
                group_transform=physical_to_reference,
                tile_transforms=tile_transforms,
                footprints=verified,
                score=score,
                matched_tiles=matched,
                matched_weight_fraction=float(matched_weight_fraction),
                visible_fraction=visible_count,
                visible_weight_fraction=visible_weight,
                route="anchor_verified",
            )
            if best is None or result.score > best.score:
                best = result
    return best


def localize_anchor_verified_group(
    optical: OpticalImage,
    tiles: Sequence[XRFTile],
    *,
    config: GroupLocalizationConfig = MANUSCRIPT_GROUP_CONFIG,
) -> LocalizationResult | None:
    """Localize a tile group using anchor candidates and local verification."""

    return _localize_anchor_verified_group(optical, tiles, config=config)


def _build_mosaic(tiles: Sequence[XRFTile]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    physical_corners = [
        transform_points(tile_pixel_to_physical(tile), np.asarray([[0, 0], [tile.image.shape[1] - 1, 0], [tile.image.shape[1] - 1, tile.image.shape[0] - 1], [0, tile.image.shape[0] - 1]]))
        for tile in tiles
    ]
    bounds = np.vstack(physical_corners)
    minimum = bounds.min(axis=0)
    maximum = bounds.max(axis=0)
    common_step = min(float(np.sqrt(abs(tile.step_um[0] * tile.step_um[1]))) for tile in tiles)
    width = max(1, int(np.ceil((maximum[0] - minimum[0]) / common_step)) + 1)
    height = max(1, int(np.ceil((maximum[1] - minimum[1]) / common_step)) + 1)
    physical_to_mosaic = np.asarray(
        [[1.0 / common_step, 0.0, -minimum[0] / common_step], [0.0, 1.0 / common_step, -minimum[1] / common_step], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    total = np.zeros((height, width), dtype=np.float32)
    support = np.zeros((height, width), dtype=np.float32)
    for tile in tiles:
        transform = physical_to_mosaic @ tile_pixel_to_physical(tile)
        warped = cv2.warpPerspective(tile.image, transform, (width, height), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(tile.valid_mask, transform, (width, height), flags=cv2.INTER_LINEAR)
        total += warped * mask
        support += mask
    mosaic = np.divide(total, support, out=np.zeros_like(total), where=support > 1e-6)
    return mosaic, np.clip(support, 0.0, 1.0), physical_to_mosaic


def localize_group_mosaic(
    optical: OpticalImage,
    tiles: Sequence[XRFTile],
    *,
    config: GroupLocalizationConfig = MANUSCRIPT_GROUP_CONFIG,
) -> LocalizationResult | None:
    """Fuse acquisition geometry into one mosaic and localize it as a template."""

    _require_unique_tiles(tiles)
    if len(tiles) < 2:
        raise ValueError("Mosaic localization requires at least two tiles")
    if optical.pixels_per_um is None:
        raise ValueError("Optical pixels_per_um calibration is required")
    mosaic, mask, physical_to_mosaic = _build_mosaic(tiles)
    common_step = 1.0 / float(physical_to_mosaic[0, 0])
    candidates = search_template(
        optical.image,
        mosaic,
        mask,
        expected_scale=float(optical.pixels_per_um * common_step),
        config=replace(config.search, candidates_per_tile=1),
        candidate_count=1,
    )
    if not candidates or candidates[0].score < config.minimum_tile_score:
        return None
    candidate = candidates[0]
    physical_to_reference = candidate.transform @ physical_to_mosaic
    footprints = {tile.name: tile_footprint(tile, physical_to_reference) for tile in tiles}
    tile_transforms = {
        tile.name: physical_to_reference @ tile_pixel_to_physical(tile) for tile in tiles
    }
    weights = tile_weights(tiles, config.tile_weight_mode)
    visible_count, visible_weight = _visibility_support(
        footprints,
        tiles,
        optical.image.shape,
        weights,
        config.minimum_tile_visible_fraction,
    )
    if visible_count < config.minimum_visible_fraction or visible_weight < config.minimum_visible_weight_fraction:
        return None
    return LocalizationResult(
        group_transform=physical_to_reference,
        tile_transforms=tile_transforms,
        footprints=footprints,
        score=candidate.score,
        matched_tiles=tuple(sorted(footprints)),
        matched_weight_fraction=1.0,
        visible_fraction=visible_count,
        visible_weight_fraction=visible_weight,
        route="mosaic",
    )
