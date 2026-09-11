"""Align CVAT supervision with pose-derived (deployment-input) Track B1 candidates.

CVAT interaction tracks (``trk007``) are drawn per interaction, around the hands and
the item. Pose actor ids (``actor_12``, ``<clip>:person:3``) identify tracked people.
They are different identity systems and are never compared directly: an event is
attributed to the pose actor whose confident wrist keypoints fall inside the
annotated box while the event is happening (spatial and temporal evidence). Weak or
contested evidence is recorded as such and the event keeps a null pose actor, which
``dataset.build_window_manifest`` excludes rather than labels background.

Also here: per-event candidate coverage (did proposal generation give the event any
chance of being labelled/detected?) and per-window evidence diagnostics that separate
a transfer cropped out of view from one that fell between sampled frames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pickup_putdown.layer1.track_b1.dataset import _compute_frame_indices

logger = logging.getLogger(__name__)

BOX = ["person_bbox_x1", "person_bbox_y1", "person_bbox_x2", "person_bbox_y2"]
ASSOCIATION_COLUMNS = [
    "event_id",
    "clip_id",
    "cvat_actor_id",
    "pose_actor_id",
    "association_status",
    "association_score",
    "runner_up_actor_id",
    "runner_up_score",
    "n_pose_samples",
]


@dataclass
class AssociationConfig:
    """Evidence thresholds for attributing a CVAT event to a pose actor."""

    #: The CVAT box is tight on hands/item; widen it per side before testing wrists.
    box_expand: float = 0.25
    min_wrist_confidence: float = 0.3
    #: Pose is sampled sparsely (~0.15 s here); look slightly beyond the event.
    time_pad_s: float = 0.25
    #: Fraction of pose sample times with a wrist in the box needed to match.
    min_score: float = 0.3
    #: Best actor must beat the runner-up by this much, otherwise "ambiguous".
    min_margin: float = 0.15


def _expanded(box: np.ndarray, fraction: float) -> np.ndarray:
    width, height = box[2] - box[0], box[3] - box[1]
    return box + np.array([-width, -height, width, height]) * fraction


def _event_boxes(cvat_track: pd.DataFrame, event: pd.Series) -> pd.DataFrame:
    """Annotator-drawn (not padded) boxes of the event's CVAT track inside the event."""
    rows = cvat_track[
        (cvat_track["actor_id"] == event["actor_id"])
        & (cvat_track["timestamp_s"] >= event["t_start"])
        & (cvat_track["timestamp_s"] <= event["t_end"])
    ]
    if "is_padding" in rows.columns:
        rows = rows[~rows["is_padding"].astype(bool)]
    return rows.sort_values("timestamp_s")


def score_pose_actors(
    event: pd.Series,
    cvat_track: pd.DataFrame,
    pose_track: pd.DataFrame,
    config: AssociationConfig,
) -> tuple[dict[str, float], int]:
    """Per pose actor: fraction of pose sample times with a wrist inside the event box.

    The denominator is every pose sample time in the (padded) event span, so an actor
    visible for only part of the event scores proportionally lower. Pooled
    ``*untracked`` ids are not identities and are never scored.
    """
    for column in ("wrist_x", "wrist_y", "wrist_confidence", "timestamp_s", "actor_id"):
        if column not in pose_track.columns:
            raise ValueError(f"pose track lacks {column!r}; cannot associate actors")

    boxes = _event_boxes(cvat_track, event)
    start, end = event["t_start"] - config.time_pad_s, event["t_end"] + config.time_pad_s
    pose = pose_track[(pose_track["timestamp_s"] >= start) & (pose_track["timestamp_s"] <= end)]
    if "is_valid" in pose.columns:
        pose = pose[pose["is_valid"].astype(bool)]
    sample_times = np.sort(pose["timestamp_s"].unique())
    if boxes.empty or len(sample_times) == 0:
        return {}, len(sample_times)

    box_times = boxes["timestamp_s"].to_numpy()
    box_values = boxes[BOX].to_numpy(dtype=float)
    nearest = np.clip(
        np.searchsorted(box_times, pose["timestamp_s"].to_numpy()), 0, len(box_times) - 1
    )
    # Nearest in time among the two neighbours of each insertion point.
    previous = np.clip(nearest - 1, 0, len(box_times) - 1)
    use_previous = np.abs(box_times[previous] - pose["timestamp_s"].to_numpy()) < np.abs(
        box_times[nearest] - pose["timestamp_s"].to_numpy()
    )
    nearest = np.where(use_previous, previous, nearest)
    expanded = np.array([_expanded(b, config.box_expand) for b in box_values])[nearest]

    x, y = pose["wrist_x"].to_numpy(), pose["wrist_y"].to_numpy()
    hit = (
        (pose["wrist_confidence"].to_numpy() >= config.min_wrist_confidence)
        & (x >= expanded[:, 0])
        & (x <= expanded[:, 2])
        & (y >= expanded[:, 1])
        & (y <= expanded[:, 3])
    )
    hits = pd.DataFrame(
        {"actor_id": pose["actor_id"].to_numpy(), "t": pose["timestamp_s"].to_numpy(), "hit": hit}
    )
    hits = hits[~hits["actor_id"].astype(str).str.endswith("untracked")]
    per_actor = hits.groupby(["actor_id", "t"])["hit"].any().groupby("actor_id").sum()
    return {str(a): float(n) / len(sample_times) for a, n in per_actor.items()}, len(sample_times)


def associate_events(
    events: pd.DataFrame,
    cvat_tracks: dict[str, pd.DataFrame],
    pose_tracks: dict[str, pd.DataFrame],
    config: AssociationConfig | None = None,
) -> pd.DataFrame:
    """One row per event: the pose actor it is attributed to, or why it is not.

    ``events`` must carry CVAT ``actor_id`` (track id). Status is one of
    ``matched``, ``ambiguous`` (two actors with similar evidence),
    ``unmatched`` (no actor's wrist near the box) or ``no_pose`` (no pose samples).
    """
    config = config or AssociationConfig()
    rows: list[dict] = []
    for _, event in events.iterrows():
        clip_id = event["clip_id"]
        scores, n_samples = score_pose_actors(
            event,
            cvat_tracks.get(clip_id, pd.DataFrame(columns=["actor_id", "timestamp_s", *BOX])),
            pose_tracks.get(
                clip_id,
                pd.DataFrame(
                    columns=["actor_id", "timestamp_s", "wrist_x", "wrist_y", "wrist_confidence"]
                ),
            ),
            config,
        )
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        best_actor, best = ranked[0] if ranked else (None, 0.0)
        runner_actor, runner = ranked[1] if len(ranked) > 1 else (None, 0.0)
        if n_samples == 0:
            status = "no_pose"
        elif best < config.min_score:
            status = "unmatched"
        elif best - runner < config.min_margin:
            status = "ambiguous"
        else:
            status = "matched"
        rows.append(
            {
                "event_id": event["event_id"],
                "clip_id": clip_id,
                "cvat_actor_id": event["actor_id"],
                "pose_actor_id": best_actor if status == "matched" else None,
                "association_status": status,
                "association_score": best,
                "runner_up_actor_id": runner_actor,
                "runner_up_score": runner,
                "n_pose_samples": n_samples,
            }
        )
    result = pd.DataFrame(rows, columns=ASSOCIATION_COLUMNS)
    logger.info("Actor association: %s", result["association_status"].value_counts().to_dict())
    return result


def events_in_pose_identity(events: pd.DataFrame, association: pd.DataFrame) -> pd.DataFrame:
    """Events relabelled into the pose identity system, keeping CVAT provenance.

    ``actor_id`` becomes the associated pose actor (null unless ``matched``); the
    original track id survives as ``cvat_actor_id``.
    """
    merged = events.drop(columns=["actor_id"]).merge(
        association.drop(columns=["clip_id"]), on="event_id", how="left", validate="one_to_one"
    )
    if merged["association_status"].isna().any():
        raise ValueError("every event needs an association row")
    return merged.rename(columns={"pose_actor_id": "actor_id"})


# ============================================================
# COVERAGE AND EVIDENCE DIAGNOSTICS
# ============================================================


def candidate_coverage(
    events: pd.DataFrame,
    candidates: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Why each reviewed event can or cannot be learned/detected from the candidates.

    ``events`` is in the candidates' identity system (``events_in_pose_identity``).
    ``coverage`` is ``labelled`` when at least one window carries the event;
    otherwise the first failing stage: ``association_failed``, ``no_candidate``
    (no candidate of any actor overlaps it — proposal recall miss),
    ``other_actor_only`` or ``no_window_centre`` (a candidate overlaps but no window
    centre lands inside the event). Items of one multi-item action
    (``event_group_id``) share an interval, and a window records only one of their
    ids, so they are counted as a group.
    """
    labelled = (
        manifest["event_id"].dropna().value_counts()
        if "event_id" in manifest
        else pd.Series(dtype=int)
    )
    if "event_group_id" in events.columns:
        group_of = events.set_index("event_id")["event_group_id"]
        labelled = labelled.groupby(labelled.index.map(group_of)).sum()
    rows: list[dict] = []
    for _, event in events.iterrows():
        clip_candidates = candidates[candidates["clip_id"] == event["clip_id"]]
        overlapping = clip_candidates[
            (clip_candidates["window_start_s"] < event["t_end"])
            & (clip_candidates["window_end_s"] > event["t_start"])
        ]
        own = overlapping[overlapping["actor_id"] == event["actor_id"]]
        n_windows = int(labelled.get(event.get("event_group_id", event["event_id"]), 0))
        if n_windows:
            coverage = "labelled"
        elif pd.isna(event["actor_id"]):
            coverage = "association_failed"
        elif overlapping.empty:
            coverage = "no_candidate"
        elif own.empty:
            coverage = "other_actor_only"
        else:
            coverage = "no_window_centre"
        rows.append(
            {
                "event_id": event["event_id"],
                "clip_id": event["clip_id"],
                "type": event["type"],
                "t_start": event["t_start"],
                "t_end": event["t_end"],
                "actor_id": event["actor_id"],
                "association_status": event.get("association_status"),
                "n_overlapping_candidates": len(overlapping),
                "n_actor_candidates": len(own),
                "n_labelled_windows": n_windows,
                "coverage": coverage,
            }
        )
    return pd.DataFrame(rows)


def window_evidence(
    window_start_s: float,
    window_end_s: float,
    num_frames: int,
    fps: float,
    crop_box: tuple[int, int, int, int],
    event: pd.Series | None,
    cvat_track: pd.DataFrame | None,
) -> dict:
    """Separate spatial exclusion from temporal under-sampling for one window.

    ``sampled_frames_in_event``: how many of the model's sampled frames fall inside
    the annotated event (0 means the transfer happened between samples).
    ``event_box_in_crop``: mean fraction of the annotated box area inside the crop
    over the event frames in this window (low means the transfer was cropped out).
    """
    indices = _compute_frame_indices(window_start_s, window_end_s, num_frames, fps)
    result = {"sampled_frames_in_event": None, "event_box_in_crop": None}
    if event is None or cvat_track is None:
        return result
    start_frame, end_frame = event["t_start"] * fps, event["t_end"] * fps
    result["sampled_frames_in_event"] = int(sum(start_frame <= i < end_frame for i in indices))

    boxes = _event_boxes(cvat_track, event)
    boxes = boxes[
        (boxes["timestamp_s"] >= window_start_s) & (boxes["timestamp_s"] <= window_end_s)
    ]
    if boxes.empty:
        return result
    b = boxes[BOX].to_numpy(dtype=float)
    cx1, cy1, cx2, cy2 = crop_box
    inter_w = np.clip(np.minimum(b[:, 2], cx2) - np.maximum(b[:, 0], cx1), 0, None)
    inter_h = np.clip(np.minimum(b[:, 3], cy2) - np.maximum(b[:, 1], cy1), 0, None)
    area = np.clip((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1e-9, None)
    result["event_box_in_crop"] = float(np.mean(inter_w * inter_h / area))
    return result
