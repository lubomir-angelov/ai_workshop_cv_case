"""Shelf and surface region configuration, validation, and expansion."""

from __future__ import annotations

import logging
import math
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from pickup_putdown.common.exceptions import ConfigError

log = logging.getLogger(__name__)

# A polygon is a list of (x, y) point pairs in source-frame pixel coordinates.
Polygon = list[tuple[float, float]]


class RegionType(StrEnum):
    SHELF = "shelf"
    SURFACE = "surface"


class ExpansionMode(StrEnum):
    PIXEL = "pixel"
    NORMALIZED = "normalized"


class ExpansionConfig(BaseModel):
    mode: ExpansionMode = ExpansionMode.PIXEL
    # For pixel mode: number of pixels. For normalized: fraction of max(width, height).
    value: float = Field(gt=0, default=20.0)


class ShelfRegion(BaseModel):
    region_id: str
    type: RegionType
    # Stored as [[x, y], ...] so YAML round-trips cleanly with flow-style inner lists.
    polygon: list[list[float]]

    @field_validator("polygon")
    @classmethod
    def _validate_polygon_structure(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) < 3:
            raise ValueError(f"polygon must have at least 3 points, got {len(v)}")
        for pt in v:
            if len(pt) != 2:
                raise ValueError(f"each polygon point must be [x, y], got {pt!r}")
        return v

    @property
    def points(self) -> Polygon:
        return [(p[0], p[1]) for p in self.polygon]


class CameraShelfConfig(BaseModel):
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    expansion: ExpansionConfig = Field(default_factory=ExpansionConfig)
    regions: list[ShelfRegion]

    @model_validator(mode="after")
    def _validate_region_geometry(self) -> CameraShelfConfig:
        w, h = self.source_width, self.source_height
        seen_ids: set[str] = set()
        for region in self.regions:
            if region.region_id in seen_ids:
                raise ValueError(f"duplicate region_id: '{region.region_id}'")
            seen_ids.add(region.region_id)

            pts = region.points
            for x, y in pts:
                if not (0 <= x < w and 0 <= y < h):
                    raise ValueError(
                        f"region '{region.region_id}': point ({x}, {y}) is outside "
                        f"image bounds {w}x{h}"
                    )
            if _all_collinear(pts):
                raise ValueError(
                    f"region '{region.region_id}': all polygon points are collinear "
                    f"(polygon has zero area)"
                )
        return self


class ShelfConfig(BaseModel):
    cameras: dict[str, CameraShelfConfig]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _all_collinear(points: Polygon) -> bool:
    """Return True if every point lies on the line through the first two."""
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    dx, dy = x1 - x0, y1 - y0
    return all(abs((x - x0) * dy - (y - y0) * dx) <= 1e-6 for x, y in points[2:])


def expand_polygon(
    points: Polygon,
    expansion: ExpansionConfig,
    image_width: int,
    image_height: int,
) -> Polygon:
    """Expand a polygon outward from its centroid by the configured margin.

    Clamps any expanded vertex that would exceed image bounds and logs a warning
    so the caller is never silently given out-of-frame coordinates.
    """
    if expansion.mode is ExpansionMode.PIXEL:
        margin_px = expansion.value
    else:
        # Normalized: fraction of the longer image dimension.
        margin_px = expansion.value * max(image_width, image_height)

    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)

    expanded: Polygon = []
    for x, y in points:
        dx, dy = x - cx, y - cy
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            expanded.append((x, y))
            continue
        scale = (dist + margin_px) / dist
        ex = cx + dx * scale
        ey = cy + dy * scale
        needs_clamp = ex < 0 or ex >= image_width or ey < 0 or ey >= image_height
        if needs_clamp:
            log.warning(
                "Expanded polygon vertex (%.1f, %.1f) exceeds image bounds %dx%d; clamping.",
                ex,
                ey,
                image_width,
                image_height,
            )
        ex = max(0.0, min(float(image_width - 1), ex))
        ey = max(0.0, min(float(image_height - 1), ey))
        expanded.append((ex, ey))

    return expanded


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_shelf_config(path: Path | str) -> ShelfConfig:
    """Load and validate shelf configuration from a YAML file.

    Raises ConfigError for missing files, unparseable YAML, or invalid geometry.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Shelf config not found: {path}")

    try:
        with path.open() as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse shelf config {path}: {exc}") from exc

    if not isinstance(raw, dict) or "cameras" not in raw:
        raise ConfigError(f"Shelf config {path} must have a top-level 'cameras' key")

    try:
        config = ShelfConfig.model_validate(raw)
    except Exception as exc:
        raise ConfigError(f"Invalid shelf config {path}: {exc}") from exc

    log.info("Loaded shelf config from %s (%d camera(s))", path, len(config.cameras))
    return config


def get_regions_for_camera(config: ShelfConfig, camera_id: str) -> CameraShelfConfig:
    """Return the config for *camera_id*, or raise ConfigError if absent."""
    if camera_id not in config.cameras:
        available = sorted(config.cameras)
        raise ConfigError(f"Camera '{camera_id}' not in shelf config. Available: {available}")
    return config.cameras[camera_id]


def get_expanded_regions(camera_config: CameraShelfConfig) -> dict[str, Polygon]:
    """Return a mapping of region_id → expanded polygon for every region."""
    return {
        region.region_id: expand_polygon(
            region.points,
            camera_config.expansion,
            camera_config.source_width,
            camera_config.source_height,
        )
        for region in camera_config.regions
    }
