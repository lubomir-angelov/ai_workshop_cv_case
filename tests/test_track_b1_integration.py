"""Integration risks between the CVAT-conditioned and pose-derived Track B1 routes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from pickup_putdown.annotation.cvat_import import (
    ClipAnnotations,
    TrackInterval,
    assign_splits_from_registry,
    normalize_clip_id,
    to_events,
)
from pickup_putdown.layer1.track_b1.actor_association import (
    AssociationConfig,
    associate_events,
    candidate_coverage,
    events_in_pose_identity,
    window_evidence,
)
from pickup_putdown.layer1.track_b1.cache import (
    CachedTrackB1Dataset,
    build_candidate_cache,
    find_stale_entries,
)
from pickup_putdown.layer1.track_b1.dataset import (
    TrackB1Dataset,
    WindowConfig,
    build_window_manifest,
    window_cache_relpath,
)
from pickup_putdown.layer1.track_b1.embeddings import load_embeddings_for, save_embeddings
from pickup_putdown.layer1.track_b1.inference import (
    InferenceConfig,
    InferenceWindowDataset,
    decode_window_scores,
    evaluate_events,
    suppress_duplicate_events,
)

CLIP = "D2_S20260520135131_E20260520135549_anon"
EMPTY_CLIP = "D2_S20260520140905_E20260520141207_anon"


def _clips(*clip_ids: str, split: str = "val") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "clip_id": list(clip_ids),
            "split": split,
            "duration_s": 60.0,
            "fps": 20.0,
            "width": 64,
            "height": 48,
        }
    )


def _candidate(
    candidate_id: str, actor_id: str, start: float, end: float, clip: str = CLIP
) -> dict:
    return {
        "candidate_id": candidate_id,
        "clip_id": clip,
        "actor_id": actor_id,
        "region_id": None,
        "window_start_s": start,
        "window_end_s": end,
    }


def _event(event_id: str, event_type: str, start: float, end: float, actor_id=None) -> dict:
    return {
        "event_id": event_id,
        "clip_id": CLIP,
        "type": event_type,
        "t_start": start,
        "t_end": end,
        "confidence": "high",
        "actor_id": actor_id,
    }


NO_IGNORES = pd.DataFrame(columns=["clip_id", "t_start", "t_end"])
CONFIG = WindowConfig(window_duration_s=1.0, window_stride_s=0.5)


def _labels(manifest: pd.DataFrame, actor_id: str) -> dict[float, str]:
    rows = manifest[manifest["actor_id"] == actor_id]
    return {
        round((r.window_start_s + r.window_end_s) / 2, 2): r.label_name for r in rows.itertuples()
    }


# ---------------------------------------------------------------------------
# Identity and labels
# ---------------------------------------------------------------------------


def test_disjoint_identity_systems_raise_instead_of_labelling_background():
    candidates = pd.DataFrame([_candidate("c1", "actor_3", 0.0, 10.0)])
    events = pd.DataFrame([_event("e1", "pickup", 4.0, 6.0, actor_id="trk000")])

    with pytest.raises(ValueError, match="different identity systems"):
        build_window_manifest(candidates, events, NO_IGNORES, _clips(CLIP), CONFIG)


def test_overlapping_actors_never_receive_each_others_events():
    candidates = pd.DataFrame(
        [_candidate("a", "actor_1", 0.0, 10.0), _candidate("b", "actor_2", 0.0, 10.0)]
    )
    events = pd.DataFrame(
        [
            _event("e1", "pickup", 4.0, 6.0, actor_id="actor_1"),
            _event("e2", "putdown", 4.5, 5.5, actor_id="actor_2"),
        ]
    )

    manifest = build_window_manifest(candidates, events, NO_IGNORES, _clips(CLIP), CONFIG)

    assert _labels(manifest, "actor_1")[5.0] == "pickup"
    assert _labels(manifest, "actor_2")[5.0] == "putdown"
    assert _labels(manifest, "actor_2")[4.0] == "background"  # actor_1's pickup only


def test_unassociated_events_exclude_windows_rather_than_label_background():
    candidates = pd.DataFrame(
        [_candidate("a", "actor_1", 0.0, 10.0), _candidate("u", "actor_untracked", 0.0, 10.0)]
    )
    events = pd.DataFrame(
        [
            _event("e1", "pickup", 4.0, 6.0, actor_id=None),
            _event("e2", "putdown", 7.0, 8.0, actor_id="actor_1"),
        ]
    )

    manifest = build_window_manifest(candidates, events, NO_IGNORES, _clips(CLIP), CONFIG)
    actor_1, untracked = _labels(manifest, "actor_1"), _labels(manifest, "actor_untracked")

    assert 5.0 not in actor_1  # centre inside the unassociated event: excluded
    assert actor_1[7.5] == "putdown"
    assert actor_1[3.0] == "background"
    assert 7.5 not in untracked and 5.0 not in untracked  # pooled id is not an actor
    assert untracked[2.0] == "background"


def test_unknown_candidate_clip_ids_raise():
    candidates = pd.DataFrame([_candidate("c1", "a", 0.0, 10.0, clip="clip_" + CLIP)])

    with pytest.raises(ValueError, match="missing from clips_df"):
        build_window_manifest(
            candidates, pd.DataFrame(columns=["clip_id"]), NO_IGNORES, _clips(CLIP), CONFIG
        )


def test_clip_id_normalisation_strips_run_prefixes_and_validates():
    assert normalize_clip_id("clip_" + CLIP) == CLIP
    assert normalize_clip_id("b1_" + CLIP) == CLIP
    with pytest.raises(ValueError):
        normalize_clip_id("actor_3")


def test_split_registry_is_explicit():
    clips = pd.DataFrame({"clip_id": [CLIP]})
    assigned = assign_splits_from_registry(
        clips, pd.DataFrame(columns=["clip_id"]), {"20260520": "val"}
    )
    assert assigned["split"].tolist() == ["val"]
    with pytest.raises(ValueError, match="not in the split registry"):
        assign_splits_from_registry(clips, pd.DataFrame(columns=["clip_id"]), {"20260526": "val"})


def test_item_count_expands_into_one_canonical_row_per_item():
    interval = TrackInterval(CLIP, 7, 0, "pickup", 20, 40, {"item_count": "3"})
    clip = ClipAnnotations(CLIP, 1, 20.0, 64, 48, 1200, 60.0, "video", [interval])

    rows = pd.DataFrame(to_events(clip, 0.25, accepted_only=False))

    assert len(rows) == 3
    assert rows["event_id"].is_unique
    assert rows["event_group_id"].nunique() == 1
    assert rows[["t_start", "t_end"]].drop_duplicates().shape[0] == 1


# ---------------------------------------------------------------------------
# CVAT -> pose association
# ---------------------------------------------------------------------------


def _cvat_track(box=(100, 100, 140, 140)) -> pd.DataFrame:
    times = np.arange(4.0, 6.01, 0.05)
    return pd.DataFrame(
        {
            "actor_id": "trk000",
            "timestamp_s": times,
            "is_padding": False,
            "person_bbox_x1": box[0],
            "person_bbox_y1": box[1],
            "person_bbox_x2": box[2],
            "person_bbox_y2": box[3],
        }
    )


def _pose(actor_wrists: dict[str, tuple[float, float]]) -> pd.DataFrame:
    rows = []
    for t in np.arange(3.9, 6.11, 0.15):
        for actor, (x, y) in actor_wrists.items():
            rows.append(
                {
                    "actor_id": actor,
                    "timestamp_s": round(t, 2),
                    "hand_side": "left",
                    "wrist_x": x,
                    "wrist_y": y,
                    "wrist_confidence": 0.9,
                    "is_valid": True,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("wrists", "status", "pose_actor"),
    [
        ({"actor_1": (120, 120), "actor_2": (500, 500)}, "matched", "actor_1"),
        ({"actor_1": (120, 120), "actor_2": (125, 118)}, "ambiguous", None),
        ({"actor_1": (500, 500)}, "unmatched", None),
        ({"actor_untracked": (120, 120)}, "unmatched", None),
    ],
)
def test_association_uses_wrist_evidence_and_flags_ambiguity(wrists, status, pose_actor):
    events = pd.DataFrame([_event("e1", "pickup", 4.0, 6.0, actor_id="trk000")])

    association = associate_events(
        events, {CLIP: _cvat_track()}, {CLIP: _pose(wrists)}, AssociationConfig()
    )

    assert association.loc[0, "association_status"] == status
    assert association.loc[0, "pose_actor_id"] == pose_actor


def test_association_without_pose_is_reported_and_keeps_provenance():
    events = pd.DataFrame([_event("e1", "pickup", 4.0, 6.0, actor_id="trk000")])
    association = associate_events(events, {CLIP: _cvat_track()}, {})
    relabelled = events_in_pose_identity(events, association)

    assert association.loc[0, "association_status"] == "no_pose"
    assert pd.isna(relabelled.loc[0, "actor_id"])
    assert relabelled.loc[0, "cvat_actor_id"] == "trk000"


def test_candidate_coverage_separates_proposal_misses_from_labelled_events():
    events = pd.DataFrame(
        [
            _event("hit", "pickup", 4.0, 6.0, actor_id="actor_1"),
            _event("miss", "putdown", 40.0, 41.0, actor_id="actor_1"),
            _event("other", "pickup", 4.2, 5.8, actor_id="actor_2"),
            _event("failed", "pickup", 4.0, 5.0, actor_id=None),
        ]
    )
    candidates = pd.DataFrame([_candidate("a", "actor_1", 0.0, 10.0)])
    manifest = build_window_manifest(candidates, events, NO_IGNORES, _clips(CLIP), CONFIG)

    coverage = candidate_coverage(events, candidates, manifest).set_index("event_id")["coverage"]

    assert coverage.to_dict() == {
        "hit": "labelled",
        "miss": "no_candidate",
        "other": "other_actor_only",
        "failed": "association_failed",
    }


def test_window_evidence_separates_crop_exclusion_from_sampling_gaps():
    event = pd.Series(_event("e1", "pickup", 4.0, 4.05, actor_id="trk000"))
    track = _cvat_track(box=(100, 100, 140, 140))

    inside = window_evidence(3.5, 4.5, 16, 20.0, (0, 0, 640, 480), event, track)
    cropped_out = window_evidence(3.5, 4.5, 16, 20.0, (300, 300, 640, 480), event, track)
    between_samples = window_evidence(3.5, 4.5, 2, 20.0, (0, 0, 640, 480), event, track)

    assert inside["event_box_in_crop"] == pytest.approx(1.0)
    assert cropped_out["event_box_in_crop"] == pytest.approx(0.0)
    assert inside["sampled_frames_in_event"] >= 1
    assert between_samples["sampled_frames_in_event"] == 0


# ---------------------------------------------------------------------------
# Empty clips, ignore intervals, decoding and evaluation
# ---------------------------------------------------------------------------


def test_ignore_intervals_drop_windows_and_predictions_consistently():
    candidates = pd.DataFrame([_candidate("a", "actor_1", 0.0, 10.0)])
    ignores = pd.DataFrame({"clip_id": [CLIP], "t_start": [4.0], "t_end": [6.0]})
    manifest = build_window_manifest(
        candidates, pd.DataFrame(columns=["clip_id", "type"]), ignores, _clips(CLIP), CONFIG
    )
    centres = set(_labels(manifest, "actor_1"))
    assert not centres & {4.0, 4.5, 5.0, 5.5, 6.0}

    events = pd.DataFrame([_event("e1", "pickup", 4.5, 5.5)])
    preds = pd.DataFrame(
        [
            {
                "pred_id": "p1",
                "clip_id": CLIP,
                "type": "pickup",
                "t_start": 4.6,
                "t_end": 5.4,
                "score": 0.9,
                "model": "m",
            }
        ]
    )
    metrics = evaluate_events(preds, events, ignores, {CLIP: 60.0})
    assert (metrics["tiou@0.5"]["tp"], metrics["tiou@0.5"]["fp"], metrics["tiou@0.5"]["fn"]) == (
        0,
        0,
        0,
    )


def test_empty_reviewed_clips_count_predictions_as_false_positives():
    events = pd.DataFrame([_event("e1", "pickup", 4.0, 5.0)])
    preds = pd.DataFrame(
        [
            {
                "pred_id": "p1",
                "clip_id": CLIP,
                "type": "pickup",
                "t_start": 4.0,
                "t_end": 5.0,
                "score": 0.9,
                "model": "m",
            },
            {
                "pred_id": "p2",
                "clip_id": EMPTY_CLIP,
                "type": "putdown",
                "t_start": 1.0,
                "t_end": 2.0,
                "score": 0.9,
                "model": "m",
            },
        ]
    )

    metrics = evaluate_events(preds, events, NO_IGNORES, {CLIP: 60.0, EMPTY_CLIP: 60.0})

    assert metrics["tiou@0.5"] == pytest.approx(
        {"precision": 0.5, "recall": 1.0, "f1": 2 / 3, "tp": 1, "fp": 1, "fn": 0}
    )
    assert metrics["fp_per_hour"] == pytest.approx(30.0)


def _scores(candidate: str, actor: str, hot: dict[float, str]) -> pd.DataFrame:
    rows = []
    for start in np.arange(0.0, 9.01, 0.25):
        klass = hot.get(round(start + 0.5, 2))
        rows.append(
            {
                "clip_id": CLIP,
                "candidate_id": candidate,
                "actor_id": actor,
                "window_start_s": start,
                "window_end_s": start + 1.0,
                "p_background": 0.05 if klass else 0.9,
                "p_pickup": 0.9 if klass == "pickup" else 0.05,
                "p_putdown": 0.9 if klass == "putdown" else 0.05,
            }
        )
    return pd.DataFrame(rows)


def test_decoding_keeps_adjacent_opposite_types_and_suppresses_duplicate_candidates():
    hot = {2.0: "pickup", 2.25: "pickup", 2.5: "putdown", 2.75: "putdown"}
    scores = pd.concat(
        [_scores("a", "actor_1", hot), _scores("a2", "actor_1", hot), _scores("b", "actor_2", hot)]
    )
    config = InferenceConfig(
        window_duration_s=1.0,
        window_stride_s=0.25,
        smoothing_window=1,
        boundary_mode="window_centers",
    )

    predictions, suppressed = suppress_duplicate_events(decode_window_scores(scores, config))

    assert suppressed == 2  # a2 repeats a's pickup and putdown for the same actor
    for actor in ("actor_1", "actor_2"):
        types = predictions[predictions["actor_id"] == actor].sort_values("t_start")["type"]
        assert types.tolist() == ["pickup", "putdown"]


# ---------------------------------------------------------------------------
# Caches and preprocessing agreement
# ---------------------------------------------------------------------------


def _video(path: Path, n_frames: int = 80) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("cv2 cannot write mp4v here")
    for i in range(n_frames):
        frame = np.full((48, 64, 3), (i * 3) % 256, dtype=np.uint8)
        frame[8:24, 4 + i % 40 : 20 + i % 40] = (255, 0, 0)
        writer.write(frame)
    writer.release()
    return path


def _track(actor_id: str = "trk000", box=(10.0, 5.0, 40.0, 30.0)) -> pd.DataFrame:
    times = np.arange(0.0, 4.0, 0.05)
    return pd.DataFrame(
        {
            "clip_id": CLIP,
            "actor_id": actor_id,
            "timestamp_s": times,
            "person_bbox_x1": box[0],
            "person_bbox_y1": box[1],
            "person_bbox_x2": box[2],
            "person_bbox_y2": box[3],
        }
    )


def test_window_cache_key_is_label_free_and_tracks_preprocessing(tmp_path: Path):
    video, pose = tmp_path / "v.mp4", tmp_path / "p.parquet"
    video.write_bytes(b"x")
    pose.write_bytes(b"y")
    base = WindowConfig()

    def key(config):
        return window_cache_relpath(video, pose, "a", 1.0, 3.5, None, (1.0, 3.5), config)

    # Settings the v1 key already implied keep their v1 digest: existing entries stay valid.
    legacy = {
        "version": 1,
        "video": [str(video.resolve()), 1, video.stat().st_mtime_ns],
        "pose": [str(pose.resolve()), 1, pose.stat().st_mtime_ns],
        "actor": "a",
        "start": 1.0,
        "end": 3.5,
        "frames": 16,
        "size": [224, 224],
        "margin": 0.15,
        "shelf": None,
    }
    digest = hashlib.sha256(json.dumps(legacy, sort_keys=True, default=str).encode()).hexdigest()
    assert key(base) == Path(digest[:2]) / f"{digest}.npy"

    assert key(WindowConfig(crop_margin=0.3)) != key(base)
    assert key(WindowConfig(resize_interpolation="area")) != key(base)
    assert key(WindowConfig(crop_scope="candidate")) != key(base)
    pose.write_bytes(b"changed")
    assert key(base) != Path(digest[:2]) / f"{digest}.npy"


def _manifest(config: WindowConfig) -> pd.DataFrame:
    candidates = pd.DataFrame([_candidate("cand", "trk000", 0.2, 3.6)])
    return build_window_manifest(
        candidates, pd.DataFrame(columns=["clip_id", "type"]), NO_IGNORES, _clips(CLIP), config
    )


def test_candidate_cache_matches_the_pixel_path_and_rebuilds_on_changed_inputs(tmp_path: Path):
    video = _video(tmp_path / f"{CLIP}.mp4")
    tracks_dir = tmp_path / "tracks"
    tracks_dir.mkdir()
    _track().to_parquet(tracks_dir / f"{CLIP}.parquet")
    config = WindowConfig(
        window_duration_s=1.0,
        window_stride_s=0.5,
        num_frames=4,
        image_size=(32, 32),
        crop_scope="candidate",
        resize_interpolation="area",
    )
    manifest = _manifest(config)
    candidate = pd.Series(_candidate("cand", "trk000", 0.2, 3.6))
    cache_dir = tmp_path / "cache"

    assert find_stale_entries(candidate.to_frame().T, tracks_dir, tmp_path, cache_dir, config) == [
        "cand"
    ]
    entry = build_candidate_cache(candidate, _track(), video, cache_dir, config)
    assert (
        find_stale_entries(candidate.to_frame().T, tracks_dir, tmp_path, cache_dir, config) == []
    )

    cached = CachedTrackB1Dataset(manifest, cache_dir, config=config, require_complete=True)
    decoded = TrackB1Dataset(manifest, tmp_path, tracks_dir, {}, config)
    for index in range(len(manifest)):
        assert np.array_equal(
            cached[index]["pixel_values"].numpy(), decoded[index]["pixel_values"].numpy()
        )

    # Changed annotation boxes invalidate the entry; reuse only when inputs match.
    moved = _track(box=(0.0, 0.0, 20.0, 20.0))
    moved.to_parquet(tracks_dir / f"{CLIP}.parquet")
    assert find_stale_entries(candidate.to_frame().T, tracks_dir, tmp_path, cache_dir, config) == [
        "cand"
    ]
    rebuilt = build_candidate_cache(candidate, moved, video, cache_dir, config)
    assert rebuilt.crop_box != entry.crop_box

    with pytest.raises(ValueError, match="different preprocessing"):
        CachedTrackB1Dataset(
            manifest,
            cache_dir,
            config=WindowConfig(num_frames=4, image_size=(32, 32), crop_scope="candidate"),
        )
    with pytest.raises(ValueError, match="not cached"):
        missing = manifest.assign(candidate_id="other")
        CachedTrackB1Dataset(missing, cache_dir, config=config, require_complete=True)


def test_inference_and_training_datasets_produce_identical_inputs(tmp_path: Path):
    video = _video(tmp_path / f"{CLIP}.mp4")
    tracks_dir = tmp_path / "tracks"
    tracks_dir.mkdir()
    _track().to_parquet(tracks_dir / f"{CLIP}.parquet")
    config = InferenceConfig(
        window_duration_s=1.0, window_stride_s=0.5, num_frames=4, image_size=(32, 32)
    )
    manifest = _manifest(config.to_window_config())
    from pickup_putdown.layer1.track_b1.dataset import generate_inference_windows

    windows = generate_inference_windows(
        pd.DataFrame([_candidate("cand", "trk000", 0.2, 3.6)]), config.to_window_config()
    )
    inference = InferenceWindowDataset(windows, video, _track(), None, config)
    training = TrackB1Dataset(
        manifest,
        tmp_path,
        tracks_dir,
        {},
        config.to_window_config(),
        cache_dir=tmp_path / "frames",
    )
    for index in range(len(windows)):
        first = training[index]
        assert np.array_equal(inference[index]["tensor"].numpy(), first["pixel_values"].numpy())
        again = training[index]  # served from the per-window cache
        assert again["cache_hit"] and np.array_equal(
            again["pixel_values"].numpy(), first["pixel_values"].numpy()
        )


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------


def _embedding_setup(tmp_path: Path) -> tuple[pd.DataFrame, dict]:
    manifest = pd.DataFrame(
        {
            "clip_id": CLIP,
            "candidate_id": "c",
            "actor_id": "a",
            "window_start_s": [0.0, 0.5, 1.0],
            "window_end_s": [1.0, 1.5, 2.0],
            "label": [0, 1, 2],
        }
    )
    preprocessing = {"version": 2, "crop_scope": "candidate"}
    save_embeddings(
        tmp_path,
        np.arange(6, dtype=np.float32).reshape(3, 2),
        manifest,
        {"encoder_weights_sha256": "abc", "preprocessing": preprocessing},
    )
    return manifest, preprocessing


def test_embeddings_are_label_free_and_aligned_by_window_key(tmp_path: Path):
    manifest, preprocessing = _embedding_setup(tmp_path)
    assert "label" not in pd.read_parquet(tmp_path / "windows.parquet").columns

    reordered = manifest.iloc[[2, 0, 1]]
    features, _ = load_embeddings_for(reordered, tmp_path, preprocessing, "abc")

    assert features.tolist() == [[4.0, 5.0], [0.0, 1.0], [2.0, 3.0]]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"unfreeze_last_n_blocks": 2}, "pixel path"),
        ({"encoder_weights_sha256": "other"}, "encoder weights"),
        ({"preprocessing": {"version": 1}}, "preprocessing"),
    ],
)
def test_embeddings_refuse_fine_tuning_and_mismatched_provenance(tmp_path: Path, kwargs, message):
    manifest, preprocessing = _embedding_setup(tmp_path)
    arguments = {"preprocessing": preprocessing, "encoder_weights_sha256": "abc", **kwargs}

    with pytest.raises(ValueError, match=message):
        load_embeddings_for(manifest, tmp_path, **arguments)


def test_embeddings_refuse_windows_they_do_not_cover(tmp_path: Path):
    manifest, preprocessing = _embedding_setup(tmp_path)
    extra = pd.concat([manifest, manifest.assign(window_start_s=5.0, window_end_s=6.0).head(1)])

    with pytest.raises(ValueError, match="no embedding"):
        load_embeddings_for(extra, tmp_path, preprocessing, "abc")


def test_shelf_regions_load_from_the_camera_schema_and_are_opt_in(tmp_path: Path):
    from pickup_putdown.layer1.track_b1.dataset import effective_shelf_region, load_shelf_regions

    regions = load_shelf_regions(Path(__file__).resolve().parents[1] / "configs/shelves.yaml")
    assert "Shelf_01" in regions

    assert effective_shelf_region(regions, "Shelf_01", WindowConfig()) is None
    enabled = WindowConfig(include_shelf_region=True)
    assert effective_shelf_region(regions, "Shelf_01", enabled) is regions["Shelf_01"]
    with pytest.raises(KeyError):
        effective_shelf_region(regions, "Shelf_99", enabled)

    empty = tmp_path / "shelves.yaml"
    empty.write_text("cameras: {cam: {regions: []}}\n")
    with pytest.raises(ValueError, match="no shelf regions"):
        load_shelf_regions(empty)
