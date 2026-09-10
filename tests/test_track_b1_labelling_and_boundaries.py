"""Tests for actor-aware window labelling and event-interval boundary construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pickup_putdown.layer1.track_b1.dataset import WindowConfig, build_window_manifest
from pickup_putdown.layer1.track_b1.inference import (
    InferenceConfig,
    WindowPrediction,
    _find_regions_above_threshold,
    detect_score_peaks,
    merge_same_type_regions,
)

# ---------------------------------------------------------------------------
# Actor-aware window labelling
# ---------------------------------------------------------------------------

CLIPS = pd.DataFrame(
    {"clip_id": ["clip_a"], "duration_s": [120.0], "fps": [20.0], "split": ["train"]}
)
NO_IGNORES = pd.DataFrame(columns=["clip_id", "t_start", "t_end"])


def _manifest(events: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    return build_window_manifest(
        candidates_df=candidates,
        events_df=events,
        ignore_intervals_df=NO_IGNORES,
        clips_df=CLIPS,
        config=WindowConfig(window_duration_s=2.0, window_stride_s=1.0),
    )


def _candidate(candidate_id: str, actor_id: str, start: float, end: float) -> dict:
    return {
        "candidate_id": candidate_id, "clip_id": "clip_a", "actor_id": actor_id,
        "region_id": None, "window_start_s": start, "window_end_s": end,
    }


def test_actor_resolved_events_do_not_leak_between_overlapping_actors():
    """Two actors at the same shelf at the same time must not share labels."""
    candidates = pd.DataFrame([
        _candidate("c_a", "actor_1", 10.0, 16.0),
        _candidate("c_b", "actor_2", 10.0, 16.0),
    ])
    events = pd.DataFrame([{
        "event_id": "e1", "clip_id": "clip_a", "actor_id": "actor_1",
        "type": "pickup", "t_start": 12.0, "t_end": 14.0, "confidence": "high",
    }])

    manifest = _manifest(events, candidates)

    assert set(manifest[manifest["actor_id"] == "actor_1"]["label_name"]) == {
        "background", "pickup"
    }
    # Without the actor filter, actor_2 inherits actor_1's pickup here.
    assert set(manifest[manifest["actor_id"] == "actor_2"]["label_name"]) == {"background"}


def test_clip_level_ground_truth_is_unchanged():
    """Ground truth without an actor_id column must behave exactly as before."""
    candidates = pd.DataFrame([
        _candidate("c_a", "actor_1", 10.0, 16.0),
        _candidate("c_b", "actor_2", 10.0, 16.0),
    ])
    events = pd.DataFrame([{
        "event_id": "e1", "clip_id": "clip_a",
        "type": "pickup", "t_start": 12.0, "t_end": 14.0, "confidence": "high",
    }])

    manifest = _manifest(events, candidates)

    for actor in ("actor_1", "actor_2"):
        assert "pickup" in set(manifest[manifest["actor_id"] == actor]["label_name"])


def test_each_actor_keeps_its_own_event_type():
    candidates = pd.DataFrame([
        _candidate("c_a", "actor_1", 10.0, 16.0),
        _candidate("c_b", "actor_2", 10.0, 16.0),
    ])
    events = pd.DataFrame([
        {"event_id": "e1", "clip_id": "clip_a", "actor_id": "actor_1",
         "type": "pickup", "t_start": 12.0, "t_end": 14.0, "confidence": "high"},
        {"event_id": "e2", "clip_id": "clip_a", "actor_id": "actor_2",
         "type": "putdown", "t_start": 12.0, "t_end": 14.0, "confidence": "high"},
    ])

    manifest = _manifest(events, candidates)

    assert set(manifest[manifest["actor_id"] == "actor_1"]["label_name"]) == {
        "background", "pickup"
    }
    assert set(manifest[manifest["actor_id"] == "actor_2"]["label_name"]) == {
        "background", "putdown"
    }


# ---------------------------------------------------------------------------
# Boundary construction
# ---------------------------------------------------------------------------

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


def _series(hot: set[float]) -> list[WindowPrediction]:
    return [
        _window(float(c), pickup=0.9 if round(float(c), 3) in hot else 0.05)
        for c in np.arange(8.0, 14.0, STRIDE_S)
    ]


def test_default_boundary_mode_preserves_existing_behaviour():
    """window_span is the default, so existing callers see no change."""
    assert InferenceConfig().boundary_mode == "window_span"

    regions = _find_regions_above_threshold(_series({10.0}), 1, 0.5, "pickup")
    assert regions[0].end_s - regions[0].start_s == pytest.approx(WINDOW_S)


def test_centre_mode_yields_a_tighter_interval():
    span = _find_regions_above_threshold(
        _series({10.0, 10.5}), 1, 0.5, "pickup", boundary_mode="window_span"
    )[0]
    centres = _find_regions_above_threshold(
        _series({10.0, 10.5}), 1, 0.5, "pickup", boundary_mode="window_centers"
    )[0]

    assert (centres.end_s - centres.start_s) < (span.end_s - span.start_s)
    assert centres.start_s == pytest.approx(9.75)
    assert centres.end_s == pytest.approx(10.75)


def test_centre_mode_applies_the_minimum_duration_floor():
    regions = _find_regions_above_threshold(
        _series({10.0}), 1, 0.5, "pickup",
        boundary_mode="window_centers", min_duration_s=1.0,
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


def test_both_modes_find_the_same_number_of_regions():
    hot = {10.0, 10.5, 12.5}
    span = _find_regions_above_threshold(_series(hot), 1, 0.5, "pickup", "window_span")
    centres = _find_regions_above_threshold(_series(hot), 1, 0.5, "pickup", "window_centers")
    assert len(span) == len(centres) == 2


def test_adjacent_pickup_and_putdown_are_never_merged():
    windows = [
        _window(10.0, pickup=0.9), _window(10.5, pickup=0.9),
        _window(11.0, putdown=0.9), _window(11.5, putdown=0.9),
    ]
    config = InferenceConfig(
        pickup_threshold=0.5, putdown_threshold=0.5, smoothing_window=1,
        boundary_mode="window_centers", same_type_merge_gap_s=5.0,
    )

    merged = merge_same_type_regions(
        detect_score_peaks(windows, config), config.same_type_merge_gap_s, 0.1
    )
    assert sorted(r.event_type for r in merged) == ["pickup", "putdown"]


def test_config_boundary_mode_reaches_the_region_builder():
    windows = _series({10.0})
    span = detect_score_peaks(windows, InferenceConfig(pickup_threshold=0.5))
    centres = detect_score_peaks(
        windows, InferenceConfig(pickup_threshold=0.5, boundary_mode="window_centers")
    )

    assert (span[0].end_s - span[0].start_s) > (centres[0].end_s - centres[0].start_s)
