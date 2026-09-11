"""Build Track B1 training windows from CVAT source-video annotations.

Track B1 was written against Task 5 candidates plus pose tracks. Neither exists for
the 42 CVAT-annotated clips: annotation was done directly on source video, and the
pose pipeline was never run over them. What the annotation does give is a per-frame
box for every interaction, which serves the same purpose — it says where in the frame
to look and for whom.

This module turns the imported annotation tables (see
``pickup_putdown.annotation.cvat_import``) into the two things
``dataset.build_window_manifest`` expects:

* a candidate table, one row per annotated actor track plus its context padding;
* per-clip actor tracks in pose-track column layout, written by the importer.

Sliding a window across a candidate then yields positives (window centre inside the
event) and hard negatives (centre in the padding, same actor, same crop) in one pass.

Clips whose annotators found nothing are handled separately by
:func:`build_verified_negative_candidates` — they carry no boxes at all, so a crop
region is borrowed from the annotated pool rather than invented.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CANDIDATE_COLUMNS = [
    "candidate_id", "clip_id", "actor_id", "region_id",
    "window_start_s", "window_end_s", "source",
]


@dataclass
class NegativeSamplingConfig:
    """How to draw background windows from clips with no annotated events."""

    #: Windows to synthesise per verified-negative clip.
    windows_per_clip: int = 12

    #: Length of each synthesised candidate, in seconds.
    candidate_duration_s: float = 6.0

    #: Keep synthesised candidates this far from the clip edges.
    edge_margin_s: float = 2.0

    seed: int = 42


# ============================================================
# CANDIDATES FROM ANNOTATED ACTOR TRACKS
# ============================================================


def build_candidates_from_tracks(
    actor_tracks: dict[str, pd.DataFrame],
    clips_df: pd.DataFrame,
) -> pd.DataFrame:
    """One candidate per annotated actor track, spanning the track's padded extent.

    The importer already padded each track with held boundary boxes, so the span here
    is ``event ± context_pad_s``. That is deliberately the unit of candidacy: it is
    exactly the interval over which a crop for this actor is defined.
    """
    rows: list[dict] = []
    durations = clips_df.set_index("clip_id")["duration_s"].to_dict()

    for clip_id, track_df in actor_tracks.items():
        if track_df.empty:
            continue
        clip_duration = durations.get(clip_id)

        for actor_id, actor_df in track_df.groupby("actor_id"):
            start = float(actor_df["timestamp_s"].min())
            end = float(actor_df["timestamp_s"].max())
            if clip_duration is not None:
                end = min(end, clip_duration)
            if end <= start:
                continue

            rows.append(
                {
                    "candidate_id": f"{clip_id}__{actor_id}",
                    "clip_id": clip_id,
                    "actor_id": actor_id,
                    "region_id": None,
                    "window_start_s": start,
                    "window_end_s": end,
                    "source": "annotation",
                }
            )

    candidates = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    logger.info("Built %d candidates from annotated actor tracks", len(candidates))
    return candidates


# ============================================================
# CANDIDATES FROM VERIFIED-NEGATIVE CLIPS
# ============================================================


def build_verified_negative_candidates(
    actor_tracks: dict[str, pd.DataFrame],
    clips_df: pd.DataFrame,
    config: NegativeSamplingConfig,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Synthesise candidates and crop boxes for clips an annotator cleared as empty.

    A completed job with zero events is a verified negative and is worth as much as a
    positive, but it carries no box, so there is no annotated region to crop to.
    A box is therefore borrowed from the pool of real annotated boxes and held fixed
    across a randomly placed interval. The borrowed geometry matches the positive
    class's crop distribution, which is the point: the classifier must not be able to
    separate these clips on framing.

    Returns the synthesised candidates and the synthetic actor tracks to merge into
    the per-clip track files.

    The borrowed box is not where a person actually is in that clip. These windows are
    honest negatives for "is a transfer happening in this crop", not for "is a person
    present"; that is the question Track B1 asks, but the distinction matters when
    reading failure cases.
    """
    box_cols = ["person_bbox_x1", "person_bbox_y1", "person_bbox_x2", "person_bbox_y2"]

    pool = pd.concat(
        [df[box_cols] for df in actor_tracks.values() if not df.empty],
        ignore_index=True,
    ) if any(not df.empty for df in actor_tracks.values()) else pd.DataFrame(columns=box_cols)

    empty_clips = [clip_id for clip_id, df in actor_tracks.items() if df.empty]
    if not empty_clips or pool.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS), {}

    rng = np.random.default_rng(config.seed)
    durations = clips_df.set_index("clip_id")["duration_s"].to_dict()
    fps_by_clip = clips_df.set_index("clip_id")["fps"].to_dict()

    rows: list[dict] = []
    synthetic_tracks: dict[str, pd.DataFrame] = {}

    for clip_id in sorted(empty_clips):
        duration = durations.get(clip_id)
        fps = fps_by_clip.get(clip_id)
        if duration is None or fps is None:
            continue

        usable_start = config.edge_margin_s
        usable_end = duration - config.edge_margin_s - config.candidate_duration_s
        if usable_end <= usable_start:
            logger.warning("clip %s too short for negative sampling", clip_id)
            continue

        track_rows: list[dict] = []
        for index in range(config.windows_per_clip):
            actor_id = f"neg{index:03d}"
            start = float(rng.uniform(usable_start, usable_end))
            end = start + config.candidate_duration_s
            box = pool.iloc[int(rng.integers(len(pool)))]

            rows.append(
                {
                    "candidate_id": f"{clip_id}__{actor_id}",
                    "clip_id": clip_id,
                    "actor_id": actor_id,
                    "region_id": None,
                    "window_start_s": start,
                    "window_end_s": end,
                    "source": "verified_negative",
                }
            )

            for frame in range(int(start * fps), int(end * fps) + 1):
                track_rows.append(
                    {
                        "clip_id": clip_id,
                        "actor_id": actor_id,
                        "event_id": None,
                        "source_frame_index": frame,
                        "timestamp_s": frame / fps,
                        "person_bbox_x1": float(box["person_bbox_x1"]),
                        "person_bbox_y1": float(box["person_bbox_y1"]),
                        "person_bbox_x2": float(box["person_bbox_x2"]),
                        "person_bbox_y2": float(box["person_bbox_y2"]),
                        "is_padding": True,
                    }
                )

        synthetic_tracks[clip_id] = pd.DataFrame(track_rows)

    candidates = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    logger.info(
        "Built %d verified-negative candidates across %d cleared clips",
        len(candidates), len(synthetic_tracks),
    )
    return candidates, synthetic_tracks


# ============================================================
# ASSEMBLY
# ============================================================


def build_candidate_table(
    actor_tracks: dict[str, pd.DataFrame],
    clips_df: pd.DataFrame,
    negative_config: NegativeSamplingConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Full candidate table plus the actor tracks that back every candidate in it."""
    annotated = build_candidates_from_tracks(actor_tracks, clips_df)

    tracks = {clip_id: df.copy() for clip_id, df in actor_tracks.items()}

    if negative_config is None:
        return annotated, tracks

    negatives, synthetic = build_verified_negative_candidates(
        actor_tracks, clips_df, negative_config
    )
    for clip_id, df in synthetic.items():
        existing = tracks.get(clip_id)
        tracks[clip_id] = (
            df if existing is None or existing.empty
            else pd.concat([existing, df], ignore_index=True)
        )

    candidates = pd.concat([annotated, negatives], ignore_index=True)
    return candidates, tracks


def write_actor_tracks(tracks: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Write per-clip actor tracks where ``TrackB1Dataset`` looks for pose tracks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for clip_id, df in tracks.items():
        df.to_parquet(output_dir / f"{clip_id}.parquet", index=False)


def summarize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Label counts per split, for the record and for spotting a degenerate build."""
    if manifest.empty:
        return pd.DataFrame()
    return (
        manifest.pivot_table(
            index="split", columns="label_name", values="sample_id",
            aggfunc="count", fill_value=0,
        )
        .assign(total=lambda df: df.sum(axis=1))
    )
