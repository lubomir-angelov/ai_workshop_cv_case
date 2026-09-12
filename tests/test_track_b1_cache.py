"""Tests for the Track B1 per-candidate crop cache and its dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pickup_putdown.layer1.track_b1.cache import (
    CACHE_VERSION,
    CachedCandidate,
    CachedTrackB1Dataset,
    candidate_crop_box,
    load_cache_index,
)

FPS = 20.0


def _write_entry(cache_dir: Path, candidate_id: str, n_frames: int = 100, start_frame: int = 0):
    """Write a cache entry whose frames encode their own position, so order is checkable."""
    frames = np.zeros((n_frames, 8, 8, 3), dtype=np.uint8)
    for position in range(n_frames):
        frames[position] = position  # frame value == position in the array
    np.save(cache_dir / f"{candidate_id}.npy", frames)
    (cache_dir / f"{candidate_id}.json").write_text(
        json.dumps(
            {
                "cache_version": CACHE_VERSION,
                "candidate_id": candidate_id,
                "clip_id": "clip_a",
                "actor_id": "trk000",
                "start_frame": start_frame,
                "n_frames": n_frames,
                "fps": FPS,
                "crop_box": [0, 0, 8, 8],
                "image_size": [8, 8],
                "crop_margin": 0.15,
            }
        )
    )


def _manifest_row(candidate_id: str, start_s: float, end_s: float, label: int = 0):
    return {
        "sample_id": f"s_{candidate_id}_{start_s}",
        "clip_id": "clip_a",
        "candidate_id": candidate_id,
        "actor_id": "trk000",
        "window_start_s": start_s,
        "window_end_s": end_s,
        "label": label,
        "label_name": "background",
        "sample_weight": 1.0,
        "split": "train",
    }


# ---------------------------------------------------------------------------
# Crop box
# ---------------------------------------------------------------------------


def test_crop_box_unions_boxes_over_the_candidate_span():
    track = pd.DataFrame(
        {
            "timestamp_s": [1.0, 2.0, 3.0],
            "person_bbox_x1": [100.0, 80.0, 120.0],
            "person_bbox_y1": [200.0, 200.0, 180.0],
            "person_bbox_x2": [300.0, 320.0, 300.0],
            "person_bbox_y2": [400.0, 400.0, 420.0],
        }
    )

    x1, y1, x2, y2 = candidate_crop_box(track, 1.0, 3.0, margin=0.0, frame_size=(3840, 2160))

    assert (x1, y1, x2, y2) == (80, 180, 320, 420)


def test_crop_box_falls_back_to_the_full_frame_without_boxes():
    empty = pd.DataFrame(
        columns=[
            "timestamp_s",
            "person_bbox_x1",
            "person_bbox_y1",
            "person_bbox_x2",
            "person_bbox_y2",
        ]
    )

    assert candidate_crop_box(empty, 0.0, 1.0, 0.15, (3840, 2160)) == (0, 0, 3840, 2160)


def test_crop_box_margin_is_clamped_to_the_frame():
    track = pd.DataFrame(
        {
            "timestamp_s": [1.0],
            "person_bbox_x1": [0.0],
            "person_bbox_y1": [0.0],
            "person_bbox_x2": [100.0],
            "person_bbox_y2": [100.0],
        }
    )

    x1, y1, x2, y2 = candidate_crop_box(track, 0.0, 2.0, margin=0.5, frame_size=(200, 200))

    assert x1 == 0 and y1 == 0
    assert x2 <= 200 and y2 <= 200


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def test_frame_position_is_relative_and_clamped():
    entry = CachedCandidate(
        candidate_id="c",
        clip_id="clip_a",
        actor_id="trk000",
        start_frame=100,
        n_frames=50,
        fps=FPS,
        crop_box=(0, 0, 8, 8),
        array_path=Path("unused.npy"),
    )

    assert entry.frame_position(100) == 0
    assert entry.frame_position(120) == 20
    assert entry.frame_position(50) == 0  # before the cached span
    assert entry.frame_position(9999) == 49  # after it


def test_load_cache_index_skips_stale_versions(tmp_path: Path):
    _write_entry(tmp_path, "good")
    (tmp_path / "stale.json").write_text(json.dumps({"cache_version": 0, "candidate_id": "stale"}))

    assert set(load_cache_index(tmp_path)) == {"good"}


def test_load_cache_index_skips_entries_without_an_array(tmp_path: Path):
    _write_entry(tmp_path, "good")
    (tmp_path / "orphan.json").write_text(
        json.dumps(
            {
                "cache_version": CACHE_VERSION,
                "candidate_id": "orphan",
                "clip_id": "clip_a",
                "actor_id": "trk000",
                "start_frame": 0,
                "n_frames": 10,
                "fps": FPS,
                "crop_box": [0, 0, 8, 8],
            }
        )
    )

    assert set(load_cache_index(tmp_path)) == {"good"}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def test_dataset_samples_frames_in_chronological_order(tmp_path: Path):
    _write_entry(tmp_path, "c1", n_frames=100)
    manifest = pd.DataFrame([_manifest_row("c1", 0.0, 2.5)])

    dataset = CachedTrackB1Dataset(manifest, tmp_path, num_frames=16)
    sample = dataset[0]

    assert sample["pixel_values"].shape == (16, 3, 8, 8)
    # Each cached frame is filled with its own position, so recovering the positions
    # from the normalized tensor proves time was not reversed or shuffled.
    means = sample["pixel_values"].mean(dim=(1, 2, 3)).tolist()
    assert means == sorted(means)


def test_dataset_respects_the_candidate_start_offset(tmp_path: Path):
    _write_entry(tmp_path, "c1", n_frames=100, start_frame=200)  # candidate starts at 10 s
    manifest = pd.DataFrame([_manifest_row("c1", 10.0, 12.5)])

    dataset = CachedTrackB1Dataset(manifest, tmp_path, num_frames=4)
    positions = dataset._sample_positions(dataset.index["c1"], 10.0, 12.5)

    assert positions[0] == 0  # 10 s is the first cached frame, not array index 200
    assert positions == sorted(positions)


def test_dataset_drops_windows_with_no_cached_candidate(tmp_path: Path, caplog):
    _write_entry(tmp_path, "c1")
    manifest = pd.DataFrame([_manifest_row("c1", 0.0, 2.5), _manifest_row("missing", 0.0, 2.5)])

    dataset = CachedTrackB1Dataset(manifest, tmp_path, num_frames=8)

    assert len(dataset) == 1
    assert dataset.manifest.iloc[0]["candidate_id"] == "c1"


def test_dataset_passes_label_and_weight_through(tmp_path: Path):
    _write_entry(tmp_path, "c1")
    row = _manifest_row("c1", 0.0, 2.5, label=2)
    row["sample_weight"] = 0.5
    dataset = CachedTrackB1Dataset(pd.DataFrame([row]), tmp_path, num_frames=8)

    sample = dataset[0]
    assert sample["label"] == 2
    assert sample["sample_weight"] == pytest.approx(0.5)
    assert sample["actor_id"] == "trk000"
