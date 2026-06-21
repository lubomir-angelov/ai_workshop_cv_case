"""Tests for shelf region configuration loading, validation, and expansion."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from pickup_putdown.common.exceptions import ConfigError
from pickup_putdown.perception.shelf_regions import (
    CameraShelfConfig,
    ExpansionConfig,
    ExpansionMode,
    RegionType,
    ShelfConfig,
    ShelfRegion,
    _all_collinear,
    expand_polygon,
    get_expanded_regions,
    get_regions_for_camera,
    load_shelf_config,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_W, _H = 1920, 1080

_SQUARE = [[100, 100], [500, 100], [500, 400], [100, 400]]
_TRIANGLE = [[200, 200], [600, 200], [400, 500]]


def _make_yaml(cameras: dict, tmp_path: Path) -> Path:
    p = tmp_path / "shelves.yaml"
    with p.open("w") as fh:
        yaml.dump({"cameras": cameras}, fh)
    return p


def _cam(regions: list[dict], w: int = _W, h: int = _H, exp_val: float = 10) -> dict:
    return {
        "source_width": w,
        "source_height": h,
        "expansion": {"mode": "pixel", "value": exp_val},
        "regions": regions,
    }


def _region(
    region_id: str = "shelf_01",
    rtype: str = "shelf",
    polygon: list | None = None,
) -> dict:
    return {
        "region_id": region_id,
        "type": rtype,
        "polygon": polygon or _SQUARE,
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class TestLoadShelfConfig:
    def test_load_valid_config(self, tmp_path: Path) -> None:
        p = _make_yaml({"cam1": _cam([_region()])}, tmp_path)
        cfg = load_shelf_config(p)
        assert isinstance(cfg, ShelfConfig)
        assert "cam1" in cfg.cameras
        assert len(cfg.cameras["cam1"].regions) == 1

    def test_missing_file_raises(self) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_shelf_config(Path("/nonexistent/shelves.yaml"))

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("{invalid yaml: [unclosed")
        with pytest.raises(ConfigError, match="parse"):
            load_shelf_config(p)

    def test_missing_cameras_key_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "nocams.yaml"
        p.write_text("foo: bar\n")
        with pytest.raises(ConfigError, match="cameras"):
            load_shelf_config(p)

    def test_multiple_cameras_loaded(self, tmp_path: Path) -> None:
        p = _make_yaml(
            {
                "cam_a": _cam([_region("shelf_01")]),
                "cam_b": _cam([_region("shelf_02")]),
            },
            tmp_path,
        )
        cfg = load_shelf_config(p)
        assert set(cfg.cameras) == {"cam_a", "cam_b"}

    def test_expansion_defaults_applied(self, tmp_path: Path) -> None:
        cam = {
            "source_width": _W,
            "source_height": _H,
            "regions": [_region()],
        }
        p = _make_yaml({"cam1": cam}, tmp_path)
        cfg = load_shelf_config(p)
        assert cfg.cameras["cam1"].expansion.mode == ExpansionMode.PIXEL
        assert cfg.cameras["cam1"].expansion.value == 20.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestPolygonValidation:
    def test_too_few_points_raises(self, tmp_path: Path) -> None:
        p = _make_yaml({"cam1": _cam([_region(polygon=[[100, 100], [200, 200]])])}, tmp_path)
        with pytest.raises(ConfigError, match="3 points"):
            load_shelf_config(p)

    def test_malformed_point_raises(self, tmp_path: Path) -> None:
        p = _make_yaml(
            {"cam1": _cam([_region(polygon=[[100, 100], [200], [300, 300]])])}, tmp_path
        )
        with pytest.raises(ConfigError):
            load_shelf_config(p)

    def test_out_of_bounds_point_raises(self, tmp_path: Path) -> None:
        p = _make_yaml(
            {"cam1": _cam([_region(polygon=[[0, 0], [9999, 0], [9999, 9999]])])}, tmp_path
        )
        with pytest.raises(ConfigError, match="outside image bounds"):
            load_shelf_config(p)

    def test_negative_coordinate_raises(self, tmp_path: Path) -> None:
        p = _make_yaml({"cam1": _cam([_region(polygon=[[-1, 0], [100, 0], [50, 100]])])}, tmp_path)
        with pytest.raises(ConfigError, match="outside image bounds"):
            load_shelf_config(p)

    def test_all_collinear_raises(self, tmp_path: Path) -> None:
        p = _make_yaml(
            {"cam1": _cam([_region(polygon=[[0, 0], [100, 0], [200, 0], [300, 0]])])},
            tmp_path,
        )
        with pytest.raises(ConfigError, match="collinear"):
            load_shelf_config(p)

    def test_duplicate_region_id_raises(self, tmp_path: Path) -> None:
        p = _make_yaml({"cam1": _cam([_region("shelf_01"), _region("shelf_01")])}, tmp_path)
        with pytest.raises(ConfigError, match="duplicate region_id"):
            load_shelf_config(p)

    def test_valid_triangle_passes(self, tmp_path: Path) -> None:
        p = _make_yaml({"cam1": _cam([_region(polygon=_TRIANGLE)])}, tmp_path)
        cfg = load_shelf_config(p)
        assert cfg.cameras["cam1"].regions[0].type == RegionType.SHELF


# ---------------------------------------------------------------------------
# Collinearity helper
# ---------------------------------------------------------------------------


class TestAllCollinear:
    def test_three_collinear_points(self) -> None:
        assert _all_collinear([(0, 0), (1, 0), (2, 0)])

    def test_three_non_collinear_points(self) -> None:
        assert not _all_collinear([(0, 0), (1, 0), (0, 1)])

    def test_square_is_not_collinear(self) -> None:
        pts = [(p[0], p[1]) for p in _SQUARE]
        assert not _all_collinear(pts)

    def test_fewer_than_three_points(self) -> None:
        assert _all_collinear([(0, 0), (1, 1)])


# ---------------------------------------------------------------------------
# Polygon expansion
# ---------------------------------------------------------------------------


class TestExpandPolygon:
    def _square_pts(self) -> list[tuple[float, float]]:
        return [(p[0], p[1]) for p in _SQUARE]

    def test_pixel_expansion_increases_area(self) -> None:
        pts = self._square_pts()
        exp_cfg = ExpansionConfig(mode=ExpansionMode.PIXEL, value=10)
        expanded = expand_polygon(pts, exp_cfg, _W, _H)
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        for (ox, oy), (ex, ey) in zip(pts, expanded, strict=False):
            orig_dist = math.hypot(ox - cx, oy - cy)
            exp_dist = math.hypot(ex - cx, ey - cy)
            assert exp_dist > orig_dist - 1e-6

    def test_normalized_expansion(self) -> None:
        pts = self._square_pts()
        exp_cfg = ExpansionConfig(mode=ExpansionMode.NORMALIZED, value=0.01)
        expanded = expand_polygon(pts, exp_cfg, _W, _H)
        assert len(expanded) == len(pts)

    def test_expansion_stays_within_bounds(self) -> None:
        pts = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)]
        exp_cfg = ExpansionConfig(mode=ExpansionMode.PIXEL, value=50)
        expanded = expand_polygon(pts, exp_cfg, _W, _H)
        for x, y in expanded:
            assert 0 <= x <= _W
            assert 0 <= y <= _H

    def test_expansion_clamped_produces_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        # A near-edge triangle that would expand outside the frame.
        pts = [(1.0, 1.0), (5.0, 1.0), (3.0, 3.0)]
        exp_cfg = ExpansionConfig(mode=ExpansionMode.PIXEL, value=200)
        import logging

        with caplog.at_level(logging.WARNING, logger="pickup_putdown.perception.shelf_regions"):
            expanded = expand_polygon(pts, exp_cfg, _W, _H)
        assert any("clamping" in msg.lower() for msg in caplog.messages)
        for x, y in expanded:
            assert 0 <= x <= _W
            assert 0 <= y <= _H

    def test_degenerate_centroid_vertex_unchanged(self) -> None:
        # All points at the centroid → no direction → vertex unchanged.
        pts = [(100.0, 100.0), (100.0, 100.0), (100.0, 100.0)]
        exp_cfg = ExpansionConfig(mode=ExpansionMode.PIXEL, value=10)
        expanded = expand_polygon(pts, exp_cfg, _W, _H)
        for x, y in expanded:
            assert x == pytest.approx(100.0)
            assert y == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------


class TestCameraHelpers:
    def _config(self) -> ShelfConfig:
        return ShelfConfig(
            cameras={
                "cam1": CameraShelfConfig(
                    source_width=_W,
                    source_height=_H,
                    expansion=ExpansionConfig(),
                    regions=[
                        ShelfRegion(region_id="shelf_01", type=RegionType.SHELF, polygon=_SQUARE),
                        ShelfRegion(
                            region_id="counter_01", type=RegionType.SURFACE, polygon=_TRIANGLE
                        ),
                    ],
                )
            }
        )

    def test_get_regions_for_camera(self) -> None:
        cfg = self._config()
        cam_cfg = get_regions_for_camera(cfg, "cam1")
        assert cam_cfg.source_width == _W

    def test_missing_camera_raises(self) -> None:
        cfg = self._config()
        with pytest.raises(ConfigError, match="cam99"):
            get_regions_for_camera(cfg, "cam99")

    def test_get_expanded_regions_returns_all(self) -> None:
        cfg = self._config()
        cam_cfg = get_regions_for_camera(cfg, "cam1")
        expanded = get_expanded_regions(cam_cfg)
        assert set(expanded) == {"shelf_01", "counter_01"}
        for pts in expanded.values():
            assert len(pts) >= 3

    def test_expanded_polygons_have_same_vertex_count(self) -> None:
        cfg = self._config()
        cam_cfg = get_regions_for_camera(cfg, "cam1")
        expanded = get_expanded_regions(cam_cfg)
        for region in cam_cfg.regions:
            assert len(expanded[region.region_id]) == len(region.polygon)
