"""Tests for turning annotated actor tracks into Track B1 candidates and windows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pickup_putdown.layer1.track_b1.annotation_windows import (
    NegativeSamplingConfig,
    build_candidate_table,
    build_candidates_from_tracks,
    build_verified_negative_candidates,
)
from pickup_putdown.layer1.track_b1.dataset import WindowConfig, build_window_manifest

FPS = 20.0


def _track(clip_id: str, actor_id: str, start_s: float, end_s: float, box=(100, 200, 300, 400)):
    frames = np.arange(int(start_s * FPS), int(end_s * FPS) + 1)
    return pd.DataFrame(
        {
            "clip_id": clip_id,
            "actor_id": actor_id,
            "event_id": f"{clip_id}__{actor_id}__00",
            "source_frame_index": frames,
            "timestamp_s": frames / FPS,
            "person_bbox_x1": float(box[0]),
            "person_bbox_y1": float(box[1]),
            "person_bbox_x2": float(box[2]),
            "person_bbox_y2": float(box[3]),
            "is_padding": False,
        }
    )


@pytest.fixture
def clips() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "clip_id": ["clip_a", "clip_empty"],
            "duration_s": [120.0, 120.0],
            "fps": [FPS, FPS],
            "split": ["train", "train"],
        }
    )


# ---------------------------------------------------------------------------
# Candidates from tracks
# ---------------------------------------------------------------------------


def test_one_candidate_per_actor_track(clips):
    tracks = {
        "clip_a": pd.concat(
            [_track("clip_a", "trk000", 10.0, 14.0), _track("clip_a", "trk001", 40.0, 44.0)]
        ),
        "clip_empty": pd.DataFrame(),
    }

    candidates = build_candidates_from_tracks(tracks, clips)

    assert len(candidates) == 2
    assert set(candidates["actor_id"]) == {"trk000", "trk001"}
    first = candidates[candidates["actor_id"] == "trk000"].iloc[0]
    assert first["window_start_s"] == pytest.approx(10.0)
    assert first["window_end_s"] == pytest.approx(14.0)


def test_candidate_is_clamped_to_clip_duration(clips):
    tracks = {"clip_a": _track("clip_a", "trk000", 115.0, 130.0), "clip_empty": pd.DataFrame()}
    candidates = build_candidates_from_tracks(tracks, clips)

    assert candidates.iloc[0]["window_end_s"] <= 120.0


# ---------------------------------------------------------------------------
# Verified negatives
# ---------------------------------------------------------------------------


def test_cleared_clips_get_candidates_backed_by_borrowed_boxes(clips):
    tracks = {"clip_a": _track("clip_a", "trk000", 10.0, 14.0), "clip_empty": pd.DataFrame()}
    config = NegativeSamplingConfig(windows_per_clip=4, candidate_duration_s=6.0, seed=1)

    candidates, synthetic = build_verified_negative_candidates(tracks, clips, config)

    assert len(candidates) == 4
    assert set(candidates["clip_id"]) == {"clip_empty"}
    assert (candidates["source"] == "verified_negative").all()

    # Borrowed geometry must come from the real annotated pool, so that background
    # windows are framed like event windows.
    boxes = synthetic["clip_empty"][["person_bbox_x1", "person_bbox_y1"]].drop_duplicates()
    assert set(map(tuple, boxes.to_numpy())) <= {(100.0, 200.0)}


def test_negatives_stay_inside_the_clip(clips):
    tracks = {"clip_a": _track("clip_a", "trk000", 10.0, 14.0), "clip_empty": pd.DataFrame()}
    config = NegativeSamplingConfig(windows_per_clip=20, candidate_duration_s=6.0, seed=7)

    candidates, _ = build_verified_negative_candidates(tracks, clips, config)

    assert candidates["window_start_s"].min() >= config.edge_margin_s
    assert candidates["window_end_s"].max() <= 120.0


def test_no_negatives_without_a_box_pool(clips):
    tracks = {"clip_a": pd.DataFrame(), "clip_empty": pd.DataFrame()}
    candidates, synthetic = build_verified_negative_candidates(
        tracks, clips, NegativeSamplingConfig()
    )

    assert candidates.empty
    assert synthetic == {}


def test_negative_sampling_is_seed_reproducible(clips):
    tracks = {"clip_a": _track("clip_a", "trk000", 10.0, 14.0), "clip_empty": pd.DataFrame()}
    config = NegativeSamplingConfig(windows_per_clip=5, seed=42)

    first, _ = build_verified_negative_candidates(tracks, clips, config)
    second, _ = build_verified_negative_candidates(tracks, clips, config)

    pd.testing.assert_frame_equal(first, second)


def test_build_candidate_table_merges_synthetic_tracks(clips):
    tracks = {"clip_a": _track("clip_a", "trk000", 10.0, 14.0), "clip_empty": pd.DataFrame()}

    candidates, merged = build_candidate_table(
        tracks, clips, NegativeSamplingConfig(windows_per_clip=3, seed=3)
    )

    assert len(candidates) == 4  # one annotated + three negatives
    assert not merged["clip_empty"].empty
    # Every candidate must have boxes behind it, or its crop would fall back to
    # the full frame at cache time.
    for candidate_id in candidates["candidate_id"]:
        clip_id, actor_id = candidate_id.split("__")
        assert not merged[clip_id][merged[clip_id]["actor_id"] == actor_id].empty


# ---------------------------------------------------------------------------
# Window labelling
# ---------------------------------------------------------------------------


def _manifest_for(events: pd.DataFrame, candidates: pd.DataFrame, clips: pd.DataFrame):
    return build_window_manifest(
        candidates_df=candidates,
        events_df=events,
        ignore_intervals_df=pd.DataFrame(columns=["clip_id", "t_start", "t_end"]),
        clips_df=clips,
        config=WindowConfig(window_duration_s=2.0, window_stride_s=1.0),
    )


def test_windows_are_labelled_per_actor_not_per_clip(clips):
    """Two actors interact at the same time; each must only see its own event."""
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "c_a",
                "clip_id": "clip_a",
                "actor_id": "trk000",
                "region_id": None,
                "window_start_s": 10.0,
                "window_end_s": 16.0,
            },
            {
                "candidate_id": "c_b",
                "clip_id": "clip_a",
                "actor_id": "trk001",
                "region_id": None,
                "window_start_s": 10.0,
                "window_end_s": 16.0,
            },
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "clip_id": "clip_a",
                "actor_id": "trk000",
                "type": "pickup",
                "t_start": 12.0,
                "t_end": 14.0,
                "confidence": "high",
            },
        ]
    )

    manifest = _manifest_for(events, candidates, clips)

    actor_a = manifest[manifest["actor_id"] == "trk000"]
    actor_b = manifest[manifest["actor_id"] == "trk001"]
    assert set(actor_a["label_name"]) == {"background", "pickup"}
    # Without actor-aware labelling, actor B would inherit actor A's pickup here.
    assert set(actor_b["label_name"]) == {"background"}


def test_adjacent_pickup_and_putdown_keep_distinct_labels(clips):
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "c_a",
                "clip_id": "clip_a",
                "actor_id": "trk000",
                "region_id": None,
                "window_start_s": 10.0,
                "window_end_s": 20.0,
            },
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "clip_id": "clip_a",
                "actor_id": "trk000",
                "type": "pickup",
                "t_start": 12.0,
                "t_end": 13.0,
                "confidence": "high",
            },
            {
                "event_id": "e2",
                "clip_id": "clip_a",
                "actor_id": "trk000",
                "type": "putdown",
                "t_start": 16.0,
                "t_end": 17.0,
                "confidence": "high",
            },
        ]
    )

    manifest = _manifest_for(events, candidates, clips)

    assert "pickup" in set(manifest["label_name"])
    assert "putdown" in set(manifest["label_name"])
    assert manifest[manifest["label_name"] == "pickup"]["event_id"].unique().tolist() == ["e1"]
    assert manifest[manifest["label_name"] == "putdown"]["event_id"].unique().tolist() == ["e2"]


def test_low_confidence_events_are_downweighted(clips):
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "c_a",
                "clip_id": "clip_a",
                "actor_id": "trk000",
                "region_id": None,
                "window_start_s": 10.0,
                "window_end_s": 16.0,
            },
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "clip_id": "clip_a",
                "actor_id": "trk000",
                "type": "pickup",
                "t_start": 12.0,
                "t_end": 14.0,
                "confidence": "low",
            },
        ]
    )

    manifest = _manifest_for(events, candidates, clips)
    positives = manifest[manifest["label_name"] == "pickup"]

    assert not positives.empty
    assert (positives["sample_weight"] < 1.0).all()
