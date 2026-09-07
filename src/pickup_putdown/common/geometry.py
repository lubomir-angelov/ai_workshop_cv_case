from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numba import njit

# Sentinel disabling the boundary-inclusion pass in point-in-polygon tests.
BOUNDARY_EXCLUSIVE: float = -1.0


def polygon_to_array(polygon: Sequence[Sequence[float]] | None) -> np.ndarray:
    """Convert a polygon given as a sequence of (x, y) points to float64 (V, 2)."""
    if polygon is None or len(polygon) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.ascontiguousarray(np.asarray(polygon, dtype=np.float64).reshape(len(polygon), 2))


@njit(cache=True)
def point_to_segment_distance(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """Minimum distance from a point to a line segment."""
    dx = x2 - x1
    dy = y2 - y1
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-12:
        return math.hypot(px - x1, py - y1)

    projection = ((px - x1) * dx + (py - y1) * dy) / length_squared
    projection = max(0.0, min(1.0, projection))
    projected_x = x1 + projection * dx
    projected_y = y1 + projection * dy
    return math.hypot(px - projected_x, py - projected_y)


@njit(cache=True)
def point_in_polygon(px: float, py: float, polygon: np.ndarray, boundary_tol: float) -> bool:
    """
    Check if point inside polygon - also known as: Ray casting algorithm
    """
    n = polygon.shape[0]
    if n < 3:
        return False

    # For a given dimention (verticall or horizontall) interp. the point as a ray continuing to move
    # it is easy to visuallize that if the point is inside the figure it will "colide" odd number of times with edges of the polygon
    if boundary_tol >= 0.0:
        for index in range(n):
            next_index = (index + 1) % n
            if (
                point_to_segment_distance(
                    px,
                    py,
                    polygon[index, 0],
                    polygon[index, 1],
                    polygon[next_index, 0],
                    polygon[next_index, 1],
                )
                <= boundary_tol  # To use Ray casting we need to make sure that we are inside the bound of poly_edge_y1 and poly_edge_y2
                # In reality we can have cases where we are exactly or very very close to that edge{y_i} and point lies direcly on a segment or a vertex of the edge
                # to stop us from rounding error we exect that a point so close "lives inside"
            ):
                return True

    inside = False
    previous_index = n - 1
    for index in range(n):
        current_x = polygon[index, 0]
        current_y = polygon[index, 1]
        previous_x = polygon[previous_index, 0]
        previous_y = polygon[previous_index, 1]
        if (current_y > py) != (previous_y > py):
            x_at_y = (previous_x - current_x) * (py - current_y) / (
                previous_y - current_y
            ) + current_x
            if px < x_at_y:
                inside = not inside
        previous_index = index
    return inside


@njit(cache=True)
def point_to_polygon_distance(px: float, py: float, polygon: np.ndarray) -> float:
    """Minimum distance from a point to a polygon's edges (inf for empty polygon)."""
    n = polygon.shape[0]
    if n == 0:
        return np.inf

    minimum = np.inf
    for index in range(n):
        next_index = (index + 1) % n  # realistically here mainly for the N <-> 0 point edge
        distance = point_to_segment_distance(
            px,
            py,  # input vars
            polygon[index, 0],
            polygon[index, 1],  # first point for the curr polygon edge
            polygon[next_index, 0],
            polygon[next_index, 1],  # second point -> both creating an edge
        )
        minimum = min(minimum, distance)
    return minimum


@njit(cache=True)
def points_in_polygon(
    xs: np.ndarray,
    ys: np.ndarray,
    polygon: np.ndarray,
    boundary_tol: float,
) -> np.ndarray:
    """Vectorized :func:`point_in_polygon` over point arrays."""
    result = np.empty(xs.shape[0], dtype=np.bool_)
    for index in range(xs.shape[0]):
        result[index] = point_in_polygon(xs[index], ys[index], polygon, boundary_tol)
    return result


@njit(cache=True)
def points_to_polygon_distance(
    xs: np.ndarray,
    ys: np.ndarray,
    polygon: np.ndarray,
) -> np.ndarray:
    """Vectorized point_to_polygon_distance over point arrays."""
    result = np.empty(xs.shape[0], dtype=np.float64)
    for index in range(xs.shape[0]):
        result[index] = point_to_polygon_distance(xs[index], ys[index], polygon)
    return result


@njit(cache=True)
def trajectory_speeds(xs: np.ndarray, ys: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Per-observation speed against the previous observation (0.0 when undefined)."""
    n = xs.shape[0]
    speeds = np.zeros(n, dtype=np.float64)
    for index in range(1, n):
        time_difference = ts[index] - ts[index - 1]
        if time_difference > 0.0:
            speeds[index] = (
                math.hypot(xs[index] - xs[index - 1], ys[index] - ys[index - 1]) / time_difference
            )
            # pythogorian theor. for straight line difference / time delta == speed calculation
    return speeds


@njit(cache=True)
def velocity_reversals(
    xs: np.ndarray,
    ys: np.ndarray,
    window: int,
    reversal_threshold: float,
) -> np.ndarray:
    """Detect direction reversals via the cosine between two windowed displacements."""

    # By calculating the cosine we can register sharp U turns in 2D trajectory
    # For a given interval Window we calculate the trajectory of N points - (t-W, t-2*W)
    n = xs.shape[0]
    reversals = np.zeros(n, dtype=np.bool_)
    for index in range(2 * window, n):
        first = index - 2 * window
        middle = index - window

        # Computing the transitioning from interval index to window and from window to window*2
        # Realistically two moment depicted as a magnetute of vectors
        previous_dx = xs[middle] - xs[first]
        previous_dy = ys[middle] - ys[first]
        current_dx = xs[index] - xs[middle]
        current_dy = ys[index] - ys[middle]

        # represetnation as vector len
        previous_magnitude = math.hypot(previous_dx, previous_dy)
        current_magnitude = math.hypot(current_dx, current_dy)

        if previous_magnitude <= 1e-9 or current_magnitude <= 1e-9:
            continue

        # using cosine formula on two vectors so that we can check the angle between them
        # cos(a') = (v1 . v2) / |v1|*|v2|
        cosine = (previous_dx * current_dx + previous_dy * current_dy) / (
            previous_magnitude * current_magnitude
        )
        cosine = max(-1.0, min(1.0, cosine))
        reversals[index] = cosine < -reversal_threshold

    # in the end we are interested in how much are the directions of the movements shifting
    return reversals
