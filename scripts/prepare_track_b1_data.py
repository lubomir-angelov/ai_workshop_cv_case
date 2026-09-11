#!/usr/bin/env python3
"""Assemble Track B1 training data (docs/PLAN_1B_HUMAN.md phase 3).

Requires: downloaded S3 data + phase 2 converter output (.local/task_7_human).
Per train/val clip: runs `make tasks-3-5` (GPU) unless outputs already exist,
then copies pose/candidate parquets. The B1 candidate set is the locally
regenerated task_5 output (carries actor_id/region_id for crops, current
config fingerprint, reproducible on this device). S3 metadata window drift
and human events outside every candidate window (the generator's verified
recall gap) are logged as per-clip warnings — explicit, never fatal.

Usage: python scripts/prepare_track_b1_data.py
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
HUMAN = REPO / ".local/task_7_human"
VLM = REPO / ".local/task_7_vlm"
META = REPO / ".local/candidate_staging/candidates"
VIDEOS = REPO / ".local/source_videos"
RUNS = REPO / ".local/task_runs"
OUT = REPO / ".local/track_b1_data"
SPLIT_SEED = "b1-human-2026-09-09"


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _day_key(clip_id: str) -> str:
    m = re.search(r"S(\d{8})", clip_id)
    return m.group(1) if m else clip_id


def assign_splits(clips: pd.DataFrame) -> pd.DataFrame:
    """Deterministic recording-day 80/20 split when the registry carries none.

    The S3 registry's split/session_id columns are empty (Task-7 splits were
    never uploaded), so days are ranked by sha256(seed:day) and the lowest
    ~20% (at least 1) go to val. Reproducible: same seed + same clips -> same split.
    """
    if clips["split"].notna().any():
        return clips
    days = sorted({_day_key(c) for c in clips["clip_id"]})
    ranked = sorted(days, key=lambda d: hashlib.sha256(f"{SPLIT_SEED}:{d}".encode()).digest())
    val_days = set(ranked[: max(1, round(len(days) * 0.2))])
    clips = clips.copy()
    clips["split"] = clips["clip_id"].map(lambda c: "val" if _day_key(c) in val_days else "train")
    print(f"split: registry split empty -> assigned by recording day (seed={SPLIT_SEED})")
    for day in days:
        n = sum(1 for c in clips["clip_id"] if _day_key(c) == day)
        print(f"  {day}: {'val' if day in val_days else 'train'} ({n} clips)")
    print(f"  clips: {clips['split'].value_counts().to_dict()}")
    return clips


def run_tasks_3_5(clip_id: str, max_attempts: int = 3) -> Path:
    task5 = RUNS / f"b1_{clip_id}" / "task_5"
    if (task5 / "candidates.parquet").is_file() and (task5 / "tracks_pose.parquet").is_file():
        return task5
    video = VIDEOS / f"{clip_id}.mp4"
    if not video.is_file():
        fail(f"missing source video: {video}")
    # decode-pipeline slot timeouts (frame_pipeline.py, 10 s) flake under load
    # (e.g. concurrent 300 GB S3 sync); retry a few times before failing
    for attempt in range(1, max_attempts + 1):
        try:
            subprocess.run(
                [
                    "make",
                    f"VIDEO={video}",
                    f"RUN_ID=b1_{clip_id}",
                    "RENDER_PREVIEWS=0",
                    "tasks-3-5",
                ],
                cwd=REPO,
                check=True,
            )
            break
        except subprocess.CalledProcessError as e:
            print(
                f"tasks-3-5 attempt {attempt}/{max_attempts} failed for {clip_id}: {e}",
                file=sys.stderr,
            )
            if attempt == max_attempts:
                fail(f"tasks-3-5 failed {max_attempts}x for {clip_id}: {e}")
            time.sleep(30)
    if not (task5 / "candidates.parquet").is_file():
        fail(f"tasks-3-5 produced no candidates.parquet for {clip_id}")
    return task5


def load_s3_candidates(clip_id: str) -> pd.DataFrame:
    path = META / clip_id / f"{clip_id}.json"
    data = json.loads(path.read_text())
    if data.get("source_video_id") and data["source_video_id"] != clip_id:
        fail(f"{path}: source_video_id {data['source_video_id']!r} != dir name {clip_id!r}")
    # S3 metadata carries no actor_id/region_id at all (verified: 0/1527);
    # only window bounds are consumed here (provenance / coverage gate).
    rows = [
        {
            "clip_id": clip_id,
            "candidate_id": c["candidate_id"],
            "actor_id": c.get("actor_id") or "",
            "region_id": c.get("region_id") or "",
            "window_start_s": c["source_start_s"],
            "window_end_s": c["source_end_s"],
        }
        for c in data.get("candidates", [])
    ]
    cols = ["clip_id", "candidate_id", "actor_id", "region_id", "window_start_s", "window_end_s"]
    return pd.DataFrame(rows, columns=cols)


OVERLAP_S = 0.5


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def cross_check(
    clip_id: str, s3: pd.DataFrame, local: pd.DataFrame, clip_events: pd.DataFrame
) -> None:
    """Coverage diagnostics: S3 metadata vs local regen vs human events.

    Candidate ids/partitions legitimately differ across generator configs
    (S3 metadata carries no config fingerprint), and the proposal generator
    does not cover every human event (verified recall gap). All drift is
    logged as warnings — never silent, never fatal: labels come from human
    GT, and events outside every candidate window are inherently out of
    window-based training scope.
    """
    s3_w = list(zip(s3["window_start_s"], s3["window_end_s"], strict=True))
    loc_w = list(zip(local["window_start_s"], local["window_end_s"], strict=True))
    unbacked_local = [
        (row["candidate_id"], row["window_start_s"], row["window_end_s"])
        for _, row in local.iterrows()
        if not any(
            _overlap((row["window_start_s"], row["window_end_s"]), w) > OVERLAP_S for w in s3_w
        )
    ]
    if unbacked_local:
        print(f"WARNING {clip_id}: local candidates not in S3 metadata (kept): {unbacked_local}")
    unbacked_s3 = [
        (r, e) for r, e in s3_w if not any(_overlap((r, e), w) > OVERLAP_S for w in loc_w)
    ]
    if unbacked_s3:
        print(f"WARNING {clip_id}: S3 windows not covered locally (drift): {unbacked_s3}")
    if not clip_events.empty:
        uncovered = [
            (r["event_id"], r["type"], r["t_start"], r["t_end"])
            for _, r in clip_events.iterrows()
            if not any(_overlap((r["t_start"], r["t_end"]), w) > 0.05 for w in loc_w)
        ]
        if uncovered:
            print(
                f"WARNING {clip_id}: {len(uncovered)} human event(s) outside all candidate windows (excluded from training): {uncovered}"
            )


def main() -> None:
    clips_path = HUMAN / "clips.csv"
    if not clips_path.is_file():
        fail(f"missing input: {clips_path} (run phase 2 converter first)")
    clips = pd.read_csv(clips_path)
    for col in ("clip_id", "split"):
        if col not in clips.columns:
            fail(f"{HUMAN / 'clips.csv'}: missing column {col!r}")
    clips = assign_splits(clips)
    work = clips[clips["split"].isin(["train", "val"])]
    if work.empty:
        fail("no clips with split train/val")

    (OUT / "pose_tracks").mkdir(parents=True, exist_ok=True)
    (OUT / "candidates_perclip").mkdir(parents=True, exist_ok=True)
    for name, src in (
        ("events_human.csv", HUMAN / "events.csv"),
        ("events_vlm.csv", VLM / "events.csv"),
        ("ignore_intervals.parquet", HUMAN / "ignore_intervals.parquet"),
    ):
        if not src.is_file():
            fail(f"missing input: {src}")
        shutil.copy2(src, OUT / name)
    clips.to_csv(OUT / "clips.csv", index=False)
    events = pd.read_csv(OUT / "events_human.csv")

    frames: list[pd.DataFrame] = []
    for clip_id in sorted(work["clip_id"]):
        video = VIDEOS / f"{clip_id}.mp4"

        if not video.is_file():
            fail(f"missing source video: {video}")
        meta = META / clip_id / f"{clip_id}.json"
        
        if not meta.is_file():
            fail(f"missing candidate metadata: {meta}")
        task5 = run_tasks_3_5(clip_id)

        pose = pd.read_parquet(task5 / "tracks_pose.parquet")
        local = pd.read_parquet(task5 / "candidates.parquet")

        # RUN_ID=b1_<clip_id> controls the output directory, but dataset identity
        # must remain the original source-video clip_id used by CVAT.
        if "clip_id" in pose.columns:
            pose["clip_id"] = clip_id

        local["clip_id"] = clip_id

        pose.to_parquet(
            OUT / "pose_tracks" / f"{clip_id}.parquet",
            index=False,
        )
        local.to_parquet(
            OUT / "candidates_perclip" / f"{clip_id}.parquet",
            index=False,
        )

        cross_check(
            clip_id, load_s3_candidates(clip_id), local, events[events["clip_id"] == clip_id]
        )
        frames.append(
            local[
                [
                    "clip_id",
                    "candidate_id",
                    "actor_id",
                    "region_id",
                    "window_start_s",
                    "window_end_s",
                ]
            ]
        )
        print(
            f"{clip_id}: {len(local)} candidates (split={work.loc[work['clip_id'] == clip_id, 'split'].iloc[0]})"
        )

    candidates = pd.concat(frames, ignore_index=True)
    candidates.to_parquet(OUT / "candidates.parquet")

    val_ids = set(work.loc[work["split"] == "val", "clip_id"])
    candidates[candidates["clip_id"].isin(val_ids)].to_parquet(OUT / "candidates_val.parquet")

    ignores = pd.read_parquet(OUT / "ignore_intervals.parquet")

    from pickup_putdown.layer1.track_b1.dataset import WindowConfig, build_window_manifest

    manifest = build_window_manifest(
        candidates_df=candidates,
        events_df=events,
        ignore_intervals_df=ignores,
        clips_df=clips,
        config=WindowConfig(),
    )
    for split in ("train", "val"):
        sub = manifest[manifest["split"] == split]
        if sub.empty:
            fail(f"no window samples in {split} split")
        dist = sub["label_name"].value_counts().to_dict()
        print(f"{split}: {len(sub)} windows, labels={dist}")

    no_pose = sorted(
        set(candidates["clip_id"]) - {p.stem for p in (OUT / "pose_tracks").glob("*.parquet")}
    )
    if no_pose:
        fail(f"missing pose tracks for: {no_pose}")

    print(
        f"wrote {OUT / 'candidates.parquet'} ({len(candidates)} candidates, "
        f"{candidates['clip_id'].nunique()} clips) -> {OUT}"
    )


if __name__ == "__main__":
    main()
