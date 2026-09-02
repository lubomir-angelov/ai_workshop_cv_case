"""Equivalence tests for the shared Numba geometry kernels.

The kernels in ``pickup_putdown.common.geometry`` replaced pure-Python
implementations previously duplicated in proposals, sampling, and Track A
inference. These tests pin the kernels against reference reimplementations
of the originals, including both boundary semantics.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pickup_putdown.common.geometry import (
    BOUNDARY_EXCLUSIVE,
    point_in_polygon,
    point_to_polygon_distance,
    point_to_segment_distance,
    points_in_polygon,
    points_to_polygon_distance,
    polygon_to_array,
    trajectory_speeds,
    velocity_reversals,
)

# ---------------------------------------------------------------------------
# Reference implementations (verbatim behavior of the replaced pure Python)
# ---------------------------------------------------------------------------


def _ref_segment_distance(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def _ref_point_in_polygon_exclusive(px, py, polygon):
    if len(polygon) < 3:
        return False
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > py) != (y2 > py):
            ix = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < ix:
                inside = not inside
    return inside


def _ref_point_in_polygon_inclusive(px, py, polygon, tol=1e-9):
    if len(polygon) < 3:
        return False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if _ref_segment_distance(px, py, x1, y1, x2, y2) <= tol:
            return True
    return _ref_point_in_polygon_exclusive(px, py, polygon)


def _ref_polygon_distance(px, py, polygon):
    if not polygon:
        return float("inf")
    n = len(polygon)
    return min(_ref_segment_distance(px, py, *polygon[i], *polygon[(i + 1) % n]) for i in range(n))


def _ref_velocity_reversal(points, index, window, threshold):
    if index < 2 * window:
        return False
    first = points[index - 2 * window]
    middle = points[index - window]
    current = points[index]
    prev_dx = middle[0] - first[0]
    prev_dy = middle[1] - first[1]
    cur_dx = current[0] - middle[0]
    cur_dy = current[1] - middle[1]
    prev_mag = math.hypot(prev_dx, prev_dy)
    cur_mag = math.hypot(cur_dx, cur_dy)
    if prev_mag <= 1e-9 or cur_mag <= 1e-9:
        return False
    cosine = (prev_dx * cur_dx + prev_dy * cur_dy) / (prev_mag * cur_mag)
    cosine = max(-1.0, min(1.0, cosine))
    return cosine < -threshold


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
CONCAVE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (5.0, 5.0), (0.0, 10.0)]


def _random_polygon(rng, n_vertices):
    angles = np.sort(rng.uniform(0.0, 2.0 * math.pi, n_vertices))
    radii = rng.uniform(1.0, 50.0, n_vertices)
    return [
        (50.0 + r * math.cos(a), 50.0 + r * math.sin(a))
        for r, a in zip(radii, angles, strict=True)
    ]


# ---------------------------------------------------------------------------
# Scalar kernel equivalence
# ---------------------------------------------------------------------------


def test_segment_distance_matches_reference():
    rng = np.random.default_rng(7)
    for _ in range(500):
        px, py, x1, y1, x2, y2 = rng.uniform(-100.0, 100.0, 6)
        assert point_to_segment_distance(px, py, x1, y1, x2, y2) == pytest.approx(
            _ref_segment_distance(px, py, x1, y1, x2, y2), abs=1e-12
        )


def test_segment_distance_degenerate_segment():
    assert point_to_segment_distance(3.0, 4.0, 0.0, 0.0, 0.0, 0.0) == pytest.approx(5.0)


@pytest.mark.parametrize("polygon", [SQUARE, CONCAVE])
def test_point_in_polygon_matches_references(polygon):
    rng = np.random.default_rng(11)
    poly_arr = polygon_to_array(polygon)
    for _ in range(1000):
        px, py = rng.uniform(-5.0, 15.0, 2)
        assert point_in_polygon(px, py, poly_arr, 1e-9) == _ref_point_in_polygon_inclusive(
            px, py, polygon
        )
        assert point_in_polygon(
            px, py, poly_arr, BOUNDARY_EXCLUSIVE
        ) == _ref_point_in_polygon_exclusive(px, py, polygon)


def test_point_in_polygon_random_polygons():
    rng = np.random.default_rng(13)
    for n_vertices in (3, 4, 7, 12):
        polygon = _random_polygon(rng, n_vertices)
        poly_arr = polygon_to_array(polygon)
        for _ in range(200):
            px, py = rng.uniform(-10.0, 110.0, 2)
            assert point_in_polygon(px, py, poly_arr, 1e-9) == _ref_point_in_polygon_inclusive(
                px, py, polygon
            )
            assert point_in_polygon(
                px, py, poly_arr, BOUNDARY_EXCLUSIVE
            ) == _ref_point_in_polygon_exclusive(px, py, polygon)


def test_point_on_boundary_semantics():
    poly_arr = polygon_to_array(SQUARE)
    # On an edge: inclusive says inside, matches the proposals implementation.
    assert point_in_polygon(5.0, 0.0, poly_arr, 1e-9) is True
    # On a vertex: inclusive says inside.
    assert point_in_polygon(0.0, 0.0, poly_arr, 1e-9) is True
    # Exclusive semantics follow ray-cast parity, matching the Track A originals.
    assert point_in_polygon(5.0, 0.0, poly_arr, BOUNDARY_EXCLUSIVE) is (
        _ref_point_in_polygon_exclusive(5.0, 0.0, SQUARE)
    )


def test_point_in_polygon_degenerate_inputs():
    for polygon in ([], [(0.0, 0.0)], [(0.0, 0.0), (1.0, 1.0)]):
        assert point_in_polygon(0.5, 0.5, polygon_to_array(polygon), 1e-9) is False
        assert point_in_polygon(0.5, 0.5, polygon_to_array(polygon), BOUNDARY_EXCLUSIVE) is False


def test_polygon_distance_matches_reference():
    rng = np.random.default_rng(17)
    for polygon in (SQUARE, CONCAVE, _random_polygon(rng, 9)):
        poly_arr = polygon_to_array(polygon)
        for _ in range(300):
            px, py = rng.uniform(-20.0, 120.0, 2)
            assert point_to_polygon_distance(px, py, poly_arr) == pytest.approx(
                _ref_polygon_distance(px, py, polygon), abs=1e-12
            )


def test_polygon_distance_empty_polygon_is_inf():
    assert point_to_polygon_distance(1.0, 2.0, polygon_to_array([])) == float("inf")


# ---------------------------------------------------------------------------
# Batch kernel equivalence
# ---------------------------------------------------------------------------


def test_batch_kernels_match_scalar():
    rng = np.random.default_rng(19)
    poly_arr = polygon_to_array(CONCAVE)
    xs = rng.uniform(-5.0, 15.0, 400)
    ys = rng.uniform(-5.0, 15.0, 400)

    inside_incl = points_in_polygon(xs, ys, poly_arr, 1e-9)
    inside_excl = points_in_polygon(xs, ys, poly_arr, BOUNDARY_EXCLUSIVE)
    distances = points_to_polygon_distance(xs, ys, poly_arr)

    for i in range(xs.shape[0]):
        assert inside_incl[i] == point_in_polygon(xs[i], ys[i], poly_arr, 1e-9)
        assert inside_excl[i] == point_in_polygon(xs[i], ys[i], poly_arr, BOUNDARY_EXCLUSIVE)
        assert distances[i] == pytest.approx(
            point_to_polygon_distance(xs[i], ys[i], poly_arr), abs=1e-12
        )


def test_batch_kernels_empty_points():
    poly_arr = polygon_to_array(SQUARE)
    empty = np.empty(0, dtype=np.float64)
    assert points_in_polygon(empty, empty, poly_arr, 1e-9).shape == (0,)
    assert points_to_polygon_distance(empty, empty, poly_arr).shape == (0,)
    assert trajectory_speeds(empty, empty, empty).shape == (0,)
    assert velocity_reversals(empty, empty, 1, 0.5).shape == (0,)


# ---------------------------------------------------------------------------
# Trajectory kernels
# ---------------------------------------------------------------------------


def test_trajectory_speeds_matches_reference():
    rng = np.random.default_rng(23)
    xs = rng.uniform(0.0, 100.0, 50)
    ys = rng.uniform(0.0, 100.0, 50)
    ts = np.cumsum(rng.uniform(0.0, 0.5, 50))
    ts[10] = ts[9]  # duplicate timestamp -> dt == 0 -> speed 0

    speeds = trajectory_speeds(xs, ys, ts)

    assert speeds[0] == 0.0
    for i in range(1, 50):
        dt = ts[i] - ts[i - 1]
        expected = math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) / dt if dt > 0.0 else 0.0
        assert speeds[i] == pytest.approx(expected, abs=1e-12)


def test_velocity_reversals_matches_reference():
    rng = np.random.default_rng(29)
    for window in (1, 2, 3):
        xs = rng.uniform(0.0, 100.0, 60)
        ys = rng.uniform(0.0, 100.0, 60)
        points = list(zip(xs, ys, strict=True))
        reversals = velocity_reversals(xs, ys, window, 0.5)
        for i in range(60):
            assert reversals[i] == _ref_velocity_reversal(points, i, window, 0.5)


def test_velocity_reversals_detects_direction_flip():
    # Move right, then move back left: cosine == -1.
    xs = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    ys = np.zeros(3, dtype=np.float64)
    assert velocity_reversals(xs, ys, 1, 0.5)[2]
    # Continue straight: no reversal.
    xs = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    assert not velocity_reversals(xs, ys, 1, 0.5)[2]


def test_velocity_reversals_stationary_is_not_reversal():
    xs = np.zeros(5, dtype=np.float64)
    ys = np.zeros(5, dtype=np.float64)
    assert not velocity_reversals(xs, ys, 1, 0.5).any()


# ---------------------------------------------------------------------------
# polygon_to_array
# ---------------------------------------------------------------------------


def test_polygon_to_array_shapes():
    assert polygon_to_array(None).shape == (0, 2)
    assert polygon_to_array([]).shape == (0, 2)
    arr = polygon_to_array(SQUARE)
    assert arr.shape == (4, 2)
    assert arr.dtype == np.float64
    assert arr.flags["C_CONTIGUOUS"]
