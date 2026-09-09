#!/usr/bin/env python3
"""Assemble Track B1 training data (docs/PLAN_1B_HUMAN.md phase 3).

Requires: downloaded S3 data + phase 2 converter output (.local/task_7_human).
Per train/val clip: runs `make tasks-3-5` (GPU) unless outputs already exist,
then copies pose/candidate parquets. candidates.parquet is built from the S3
metadata JSONs (source of truth) and cross-checked against the locally
regenerated task_5 parquets; any mismatch is a hard error.

Usage: python scripts/prepare_track_b1_data.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
HUMAN = REPO / ".local/task_7_human"
VLM = REPO / ".local/task_7_vlm"
META = REPO / ".local/candidate_staging/candidates"
VIDEOS = REPO / ".local/source_videos"
RUNS = REPO / ".local/task_runs"
OUT = REPO / ".local/track_b1_data"
EPS_S = 1e-6


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def run_tasks_3_5(clip_id: str) -> Path:
    task5 = RUNS / f"b1_{clip_id}" / "task_5"
    if (task5 / "candidates.parquet").is_file() and (task5 / "tracks_pose.parquet").is_file():
        return task5
    video = VIDEOS / f"{clip_id}.mp4"
    if not video.is_file():
        fail(f"missing source video: {video}")
    try:
        subprocess.run(
            ["make", f"VIDEO={video}", f"RUN_ID=b1_{clip_id}", "RENDER_PREVIEWS=0", "tasks-3-5"],
            cwd=REPO,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        fail(f"tasks-3-5 failed for {clip_id}: {e}")
    if not (task5 / "candidates.parquet").is_file():
        fail(f"tasks-3-5 produced no candidates.parquet for {clip_id}")
    return task5


def load_s3_candidates(clip_id: str) -> pd.DataFrame:
    path = META / clip_id / f"{clip_id}.json"
    data = json.loads(path.read_text())
    if data.get("source_video_id") and data["source_video_id"] != clip_id:
        fail(f"{path}: source_video_id {data['source_video_id']!r} != dir name {clip_id!r}")
    rows = [
        {
            "clip_id": clip_id,
            "candidate_id": c["candidate_id"],
            "actor_id": c["actor_id"],
            "region_id": c.get("region_id") or "",
            "window_start_s": c["source_start_s"],
            "window_end_s": c["source_end_s"],
        }
        for c in data.get("candidates", [])
    ]
    cols = ["clip_id", "candidate_id", "actor_id", "region_id", "window_start_s", "window_end_s"]
    return pd.DataFrame(rows, columns=cols)


def cross_check(clip_id: str, s3: pd.DataFrame, local: pd.DataFrame) -> None:
    a_ids, b_ids = set(s3["candidate_id"]), set(local["candidate_id"])
    if a_ids != b_ids:
        fail(
            f"{clip_id}: candidate ids differ "
            f"(s3-only={sorted(a_ids - b_ids)} local-only={sorted(b_ids - a_ids)})"
        )
    m = s3.merge(
        local[["candidate_id", "actor_id", "region_id", "window_start_s", "window_end_s"]],
        on="candidate_id",
    )
    # same-name columns from both sides merge as <col>_x (s3) / <col>_y (local)
    for pair in (("window_start_s_x", "window_start_s_y"), ("window_end_s_x", "window_end_s_y")):
        if not (abs(m[pair[0]] - m[pair[1]]) <= EPS_S).all():
            bad = m[abs(m[pair[0]] - m[pair[1]]) > EPS_S]
            fail(f"{clip_id}: window bounds mismatch (s3 vs regenerated):\n{bad.to_string()}")
    for col in ("actor_id", "region_id"):
        if not (m[f"{col}_x"].fillna("") == m[f"{col}_y"].fillna("")).all():
            bad = m[m[f"{col}_x"].fillna("") != m[f"{col}_y"].fillna("")]
            fail(f"{clip_id}: {col} mismatch (s3 vs regenerated):\n{bad.to_string()}")


def main() -> None:
    clips_path = HUMAN / "clips.csv"
    if not clips_path.is_file():
        fail(f"missing input: {clips_path} (run phase 2 converter first)")
    clips = pd.read_csv(clips_path)
    for col in ("clip_id", "split"):
        if col not in clips.columns:
            fail(f"{HUMAN / 'clips.csv'}: missing column {col!r}")
    work = clips[clips["split"].isin(["train", "val"])]
    if work.empty:
        fail("no clips with split train/val")

    (OUT / "pose_tracks").mkdir(parents=True, exist_ok=True)
    (OUT / "candidates_perclip").mkdir(parents=True, exist_ok=True)
    for name, src in (
        ("events_human.csv", HUMAN / "events.csv"),
        ("events_vlm.csv", VLM / "events.csv"),
        ("ignore_intervals.parquet", HUMAN / "ignore_intervals.parquet"),
        ("clips.csv", HUMAN / "clips.csv"),
    ):
        if not src.is_file():
            fail(f"missing input: {src}")
        shutil.copy2(src, OUT / name)

    frames: list[pd.DataFrame] = []
    for clip_id in sorted(work["clip_id"]):
        video = VIDEOS / f"{clip_id}.mp4"
        if not video.is_file():
            fail(f"missing source video: {video}")
        meta = META / clip_id / f"{clip_id}.json"
        if not meta.is_file():
            fail(f"missing candidate metadata: {meta}")
        task5 = run_tasks_3_5(clip_id)
        shutil.copy2(task5 / "tracks_pose.parquet", OUT / "pose_tracks" / f"{clip_id}.parquet")
        shutil.copy2(
            task5 / "candidates.parquet", OUT / "candidates_perclip" / f"{clip_id}.parquet"
        )
        s3 = load_s3_candidates(clip_id)
        cross_check(clip_id, s3, pd.read_parquet(task5 / "candidates.parquet"))
        frames.append(s3)
        print(
            f"{clip_id}: {len(s3)} candidates (split={work.loc[work['clip_id'] == clip_id, 'split'].iloc[0]})"
        )

    candidates = pd.concat(frames, ignore_index=True)
    candidates.to_parquet(OUT / "candidates.parquet")

    val_ids = set(work.loc[work["split"] == "val", "clip_id"])
    candidates[candidates["clip_id"].isin(val_ids)].to_parquet(OUT / "candidates_val.parquet")

    events = pd.read_csv(OUT / "events_human.csv")
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
