"""Tests for Track B1 decoding: threshold regions, boundary modes, same-type merging."""

from __future__ import annotations

import numpy as np
import pytest

from pickup_putdown.layer1.track_b1.inference import (
    InferenceConfig,
    WindowPrediction,
    _find_regions_above_threshold,
    detect_score_peaks,
    merge_same_type_regions,
)

WINDOW_S = 2.5
STRIDE_S = 0.5


def _window(center: float, pickup: float = 0.0, putdown: float = 0.0) -> WindowPrediction:
    probs = np.array([max(0.0, 1.0 - pickup - putdown), pickup, putdown])
    return WindowPrediction(
        window_start_s=center - WINDOW_S / 2,
        window_end_s=center + WINDOW_S / 2,
        window_center_s=center,
        probs=probs,
        predicted_class=int(np.argmax(probs)),
        confidence=float(probs.max()),
    )


def _series(hot_centers: set[float], klass: str = "pickup") -> list[WindowPrediction]:
    windows = []
    for center in np.arange(8.0, 14.0, STRIDE_S):
        hot = round(float(center), 3) in hot_centers
        score = 0.9 if hot else 0.05
        windows.append(
            _window(
                float(center),
                pickup=score if klass == "pickup" else 0.0,
                putdown=score if klass == "putdown" else 0.0,
            )
        )
    return windows


# ---------------------------------------------------------------------------
# Boundary modes
# ---------------------------------------------------------------------------


def test_window_span_mode_cannot_be_tighter_than_one_window():
    """The original behaviour, kept as the default; this is its structural limit."""
    regions = _find_regions_above_threshold(
        _series({10.0}),
        class_idx=1,
        threshold=0.5,
        event_type="pickup",
        boundary_mode="window_span",
    )

    assert len(regions) == 1
    assert regions[0].end_s - regions[0].start_s == pytest.approx(WINDOW_S)


def test_centre_mode_produces_a_much_tighter_interval():
    span = _find_regions_above_threshold(
        _series({10.0, 10.5}), 1, 0.5, "pickup", boundary_mode="window_span"
    )[0]
    centres = _find_regions_above_threshold(
        _series({10.0, 10.5}), 1, 0.5, "pickup", boundary_mode="window_centers"
    )[0]

    assert (centres.end_s - centres.start_s) < (span.end_s - span.start_s)
    # Centres of the hot windows are 10.0 and 10.5, widened by half a stride each side.
    assert centres.start_s == pytest.approx(9.75)
    assert centres.end_s == pytest.approx(10.75)


def test_centre_mode_respects_the_minimum_duration_floor():
    regions = _find_regions_above_threshold(
        _series({10.0}),
        1,
        0.5,
        "pickup",
        boundary_mode="window_centers",
        min_duration_s=1.0,
    )

    assert regions[0].end_s - regions[0].start_s == pytest.approx(1.0)


def test_centre_mode_never_emits_a_negative_start():
    windows = [_window(0.1, pickup=0.9), _window(0.6, pickup=0.9)]
    regions = _find_regions_above_threshold(
        windows, 1, 0.5, "pickup", boundary_mode="window_centers", min_duration_s=5.0
    )

    assert regions[0].start_s >= 0.0


def test_unknown_boundary_mode_is_rejected():
    with pytest.raises(ValueError, match="boundary_mode"):
        _find_regions_above_threshold(_series({10.0}), 1, 0.5, "pickup", boundary_mode="nope")


def test_both_modes_agree_on_how_many_regions_exist():
    hot = {10.0, 10.5, 12.5}
    span = _find_regions_above_threshold(_series(hot), 1, 0.5, "pickup", "window_span")
    centres = _find_regions_above_threshold(_series(hot), 1, 0.5, "pickup", "window_centers")

    assert len(span) == len(centres) == 2


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


def test_pickup_and_putdown_use_their_own_thresholds():
    windows = [
        _window(10.0, pickup=0.6, putdown=0.0),
        _window(10.5, pickup=0.0, putdown=0.6),
    ]
    config = InferenceConfig(pickup_threshold=0.5, putdown_threshold=0.8)

    regions = detect_score_peaks(windows, config)

    assert [r.event_type for r in regions] == ["pickup"]


def test_no_regions_when_nothing_clears_threshold():
    assert detect_score_peaks(_series(set()), InferenceConfig()) == []


# ---------------------------------------------------------------------------
# Same-type merging
# ---------------------------------------------------------------------------


def test_adjacent_pickup_and_putdown_are_never_merged():
    """The acceptance criterion: a pickup followed by a putdown stays two events."""
    windows = [
        _window(10.0, pickup=0.9),
        _window(10.5, pickup=0.9),
        _window(11.0, putdown=0.9),
        _window(11.5, putdown=0.9),
    ]
    config = InferenceConfig(
        pickup_threshold=0.5,
        putdown_threshold=0.5,
        smoothing_window=1,
        boundary_mode="window_centers",
        same_type_merge_gap_s=5.0,
    )

    regions = detect_score_peaks(windows, config)
    merged = merge_same_type_regions(regions, config.same_type_merge_gap_s, 0.1)

    assert sorted(r.event_type for r in merged) == ["pickup", "putdown"]


def test_same_type_regions_within_the_gap_are_merged():
    regions = _find_regions_above_threshold(
        _series({10.0, 11.0}), 1, 0.5, "pickup", boundary_mode="window_centers"
    )
    assert len(regions) == 2

    merged = merge_same_type_regions(regions, merge_gap_s=1.0, min_duration_s=0.1)
    assert len(merged) == 1


def test_same_type_regions_beyond_the_gap_stay_separate():
    regions = _find_regions_above_threshold(
        _series({10.0, 13.0}), 1, 0.5, "pickup", boundary_mode="window_centers"
    )

    merged = merge_same_type_regions(regions, merge_gap_s=0.5, min_duration_s=0.1)
    assert len(merged) == 2


def test_one_candidate_can_emit_several_ordered_events():
    windows = [
        _window(9.0, pickup=0.9),
        _window(11.0, putdown=0.9),
        _window(13.0, pickup=0.9),
    ]
    config = InferenceConfig(
        pickup_threshold=0.5,
        putdown_threshold=0.5,
        smoothing_window=1,
        boundary_mode="window_centers",
        same_type_merge_gap_s=0.5,
    )

    merged = merge_same_type_regions(
        detect_score_peaks(windows, config), config.same_type_merge_gap_s, 0.1
    )
    ordered = sorted(merged, key=lambda r: r.start_s)

    assert [r.event_type for r in ordered] == ["pickup", "putdown", "pickup"]
