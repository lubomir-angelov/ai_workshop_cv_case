"""Tests for importing CVAT source-video annotations into canonical tables."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from pickup_putdown.annotation.cvat_import import (
    assign_splits,
    import_export_dir,
    parse_archive,
    recording_day,
    to_actor_track,
    to_events,
    to_ignore_intervals,
)

CLIP_ID = "D2_S20260520140905_E20260520141207_anon"  # 182 s of recording


def _box(frame: int, outside: str = "0", **attributes: str) -> str:
    defaults = {
        "confidence": "high",
        "hard_case": "false",
        "item_count": "1",
        "review_status": "accepted",
        "notes": "",
    }
    defaults.update(attributes)
    rendered = "".join(
        f'<attribute name="{name}">{value}</attribute>' for name, value in defaults.items()
    )
    return (
        f'<box frame="{frame}" keyframe="1" outside="{outside}" occluded="0" '
        f'xtl="100.0" ytl="200.0" xbr="300.0" ybr="400.0" z_order="0">{rendered}</box>'
    )


def _annotations_xml(tracks: str, clip_id: str = CLIP_ID, size: int = 3654) -> str:
    return f"""<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <id>2556446</id>
      <name>{clip_id}</name>
      <size>{size}</size>
      <original_size><width>3840</width><height>2160</height></original_size>
    </task>
  </meta>
  {tracks}
</annotations>"""


def _write_archive(directory: Path, xml: str, name: str = "task.zip") -> Path:
    archive = directory / name
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("annotations.xml", xml)
    return archive


@pytest.fixture
def single_pickup(tmp_path: Path) -> Path:
    tracks = (
        '<track id="0" label="pickup" source="manual">'
        + "".join(_box(frame) for frame in range(100, 120))
        + "</track>"
    )
    _write_archive(tmp_path, _annotations_xml(tracks))
    return tmp_path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_archive_derives_timebase_from_clip_stem(single_pickup: Path):
    clip = parse_archive(next(single_pickup.glob("*.zip")))

    assert clip.clip_id == CLIP_ID
    assert clip.n_frames == 3654
    assert clip.fps == pytest.approx(3654 / 182.0, rel=1e-3)
    assert len(clip.intervals) == 1
    assert clip.intervals[0].start_frame == 100
    assert clip.intervals[0].end_frame == 119


def test_probed_video_fps_overrides_the_stem_estimate(single_pickup: Path, monkeypatch):
    monkeypatch.setattr("pickup_putdown.annotation.cvat_import.probe_fps", lambda path: 20.0)
    clip = parse_archive(next(single_pickup.glob("*.zip")), video_dir=single_pickup)

    assert clip.fps == 20.0
    assert clip.fps_source == "video"


def test_track_interrupted_by_outside_becomes_two_intervals(tmp_path: Path):
    tracks = (
        '<track id="3" label="putdown" source="manual">'
        + "".join(_box(frame) for frame in range(10, 20))
        + _box(20, outside="1")
        + "".join(_box(frame) for frame in range(50, 60))
        + "</track>"
    )
    _write_archive(tmp_path, _annotations_xml(tracks))
    clip = parse_archive(next(tmp_path.glob("*.zip")))

    assert len(clip.intervals) == 2
    assert (clip.intervals[0].start_frame, clip.intervals[0].end_frame) == (10, 19)
    assert (clip.intervals[1].start_frame, clip.intervals[1].end_frame) == (50, 59)
    # Distinct ids matter: two events must not collide into one row downstream.
    assert clip.intervals[0].event_id() != clip.intervals[1].event_id()


# ---------------------------------------------------------------------------
# Canonical conversion
# ---------------------------------------------------------------------------


def test_events_carry_attributes_and_actor_identity(single_pickup: Path):
    clip = parse_archive(next(single_pickup.glob("*.zip")))
    rows = to_events(clip, min_duration_s=0.25, accepted_only=False)

    assert len(rows) == 1
    row = rows[0]
    assert row["type"] == "pickup"
    assert row["confidence"] == "high"
    assert row["hard_case"] is False
    assert row["actor_id"] == "trk000"
    assert row["t_start"] < row["t_end"]


def test_single_frame_mark_is_widened_to_the_minimum_duration(tmp_path: Path):
    tracks = '<track id="0" label="pickup" source="manual">' + _box(100) + "</track>"
    _write_archive(tmp_path, _annotations_xml(tracks))
    clip = parse_archive(next(tmp_path.glob("*.zip")))

    row = to_events(clip, min_duration_s=0.5, accepted_only=False)[0]
    assert row["t_end"] - row["t_start"] == pytest.approx(0.5)


def test_accepted_only_drops_draft_intervals(tmp_path: Path):
    tracks = (
        '<track id="0" label="pickup" source="manual">'
        + "".join(_box(f, review_status="draft") for f in range(10, 20))
        + "</track>"
        '<track id="1" label="pickup" source="manual">'
        + "".join(_box(f) for f in range(30, 40))
        + "</track>"
    )
    _write_archive(tmp_path, _annotations_xml(tracks))
    clip = parse_archive(next(tmp_path.glob("*.zip")))

    assert len(to_events(clip, 0.25, accepted_only=False)) == 2
    assert len(to_events(clip, 0.25, accepted_only=True)) == 1


def test_ignore_tracks_become_ignore_intervals_not_events(tmp_path: Path):
    tracks = (
        '<track id="0" label="ignore" source="manual">'
        + "".join(_box(f, ignore_reason="ACTION_OCCLUDED") for f in range(10, 30))
        + "</track>"
    )
    _write_archive(tmp_path, _annotations_xml(tracks))
    clip = parse_archive(next(tmp_path.glob("*.zip")))

    assert to_events(clip, 0.25, False) == []
    ignores = to_ignore_intervals(clip, 0.25)
    assert len(ignores) == 1
    assert ignores[0]["reason"] == "ACTION_OCCLUDED"


# ---------------------------------------------------------------------------
# Actor tracks
# ---------------------------------------------------------------------------


def test_actor_track_uses_pose_track_column_names(single_pickup: Path):
    clip = parse_archive(next(single_pickup.glob("*.zip")))
    track = to_actor_track(clip)

    for column in ("actor_id", "timestamp_s", "person_bbox_x1", "person_bbox_y2"):
        assert column in track.columns
    assert len(track) == 20
    assert not track["is_padding"].any()


def test_padding_extends_the_track_with_held_boundary_boxes(single_pickup: Path):
    clip = parse_archive(next(single_pickup.glob("*.zip")))
    padded = to_actor_track(clip, context_pad_s=1.0)

    pad_frames = int(round(1.0 * clip.fps))
    assert len(padded) == 20 + 2 * pad_frames
    assert padded["is_padding"].sum() == 2 * pad_frames

    # Held boxes must equal the boundary box, or negatives would be cropped
    # differently from positives and the crop itself would leak the label.
    held = padded[padded["is_padding"]]
    assert (held["person_bbox_x1"] == 100.0).all()
    assert (held["person_bbox_y2"] == 400.0).all()


def test_padding_is_clamped_to_the_clip(tmp_path: Path):
    tracks = (
        '<track id="0" label="pickup" source="manual">'
        + "".join(_box(frame) for frame in range(0, 5))
        + "</track>"
    )
    _write_archive(tmp_path, _annotations_xml(tracks))
    clip = parse_archive(next(tmp_path.glob("*.zip")))

    padded = to_actor_track(clip, context_pad_s=2.0)
    assert padded["source_frame_index"].min() >= 0
    assert padded["source_frame_index"].max() < clip.n_frames


def test_ignore_tracks_contribute_no_crop_geometry(tmp_path: Path):
    tracks = (
        '<track id="0" label="ignore" source="manual">'
        + "".join(_box(f) for f in range(10, 30))
        + "</track>"
    )
    _write_archive(tmp_path, _annotations_xml(tracks))
    clip = parse_archive(next(tmp_path.glob("*.zip")))

    assert to_actor_track(clip, context_pad_s=1.0).empty


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def test_recording_day_is_read_from_the_clip_stem():
    assert recording_day(CLIP_ID) == "20260520"
    assert recording_day("nonsense") == "unknown"


def test_splits_never_put_one_recording_day_in_two_splits():
    clips = pd.DataFrame(
        {
            "clip_id": [
                "D2_S20260520100000_E20260520100100_anon",
                "D2_S20260520110000_E20260520110100_anon",
                "D2_S20260521100000_E20260521100100_anon",
                "D2_S20260522100000_E20260522100100_anon",
            ],
            "duration_s": [60.0] * 4,
        }
    )
    events = pd.DataFrame(
        {"clip_id": clips["clip_id"].repeat([10, 10, 2, 1]), "event_id": range(23)}
    )

    assigned = assign_splits(clips, events, val_days=1, test_days=1)
    per_day = assigned.groupby("recording_day")["split"].nunique()
    assert (per_day == 1).all()

    # The busiest day carries training; the sparse days are held out.
    assert set(assigned[assigned["recording_day"] == "20260520"]["split"]) == {"train"}
    assert set(assigned["split"]) == {"train", "val", "test"}


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_import_export_dir_produces_aligned_tables(tmp_path: Path):
    for index, day in enumerate(("20260520", "20260521", "20260522")):
        clip_id = f"D2_S{day}140905_E{day}141207_anon"
        tracks = (
            f'<track id="{index}" label="pickup" source="manual">'
            + "".join(_box(f) for f in range(100, 120))
            + "</track>"
        )
        _write_archive(tmp_path, _annotations_xml(tracks, clip_id), f"{clip_id}.zip")

    result = import_export_dir(tmp_path, context_pad_s=1.0, val_days=1, test_days=1)

    assert len(result.clips) == 3
    assert len(result.events) == 3
    assert set(result.actor_tracks) == set(result.clips["clip_id"])
    assert set(result.events["clip_id"]) <= set(result.clips["clip_id"])
    assert set(result.clips["split"]) == {"train", "val", "test"}


def test_empty_annotations_still_yield_typed_tables(tmp_path: Path):
    _write_archive(tmp_path, _annotations_xml(""), f"{CLIP_ID}.zip")
    result = import_export_dir(tmp_path)

    # A completed job with no events is a verified negative, not a failure.
    assert result.events.empty
    assert "t_start" in result.events.columns
    assert "reason" in result.ignore_intervals.columns
    assert result.actor_tracks[CLIP_ID].empty
