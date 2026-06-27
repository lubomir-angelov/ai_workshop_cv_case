#!/usr/bin/env python3
"""Run Track A inference pipeline with stub classifiers (no downloads needed).

Validates full data flow: candidates + poses -> state machine -> output format.
Predictions will be empty (stub classifiers return uniform probabilities),
but the pipeline wiring is proven.
"""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import pandas as pd

from pickup_putdown.common.schemas import Candidate, PoseObservation
from pickup_putdown.layer1.track_a.inference import (
    InferenceConfig,
    TrackAInferencePipeline,
)
from pickup_putdown.layer1.track_a.state_machine import StateMachineConfig
from pickup_putdown.perception.shelf_regions import (
    load_shelf_config,
    get_expanded_regions,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent  # project root
LOCAL = BASE / ".local"
CANDIDATES_PARQUET = LOCAL / "task5_acceptance" / "preview_20260622_100316" / "candidates.parquet"
POSES_PARQUET = LOCAL / "task5_acceptance" / "preview_20260622_100316" / "tracks_pose.parquet"
VIDEO_PATH = LOCAL / "triage_acceptance" / "videos" / "person_clear.mp4"
ARTIFACTS_DIR = LOCAL / "track_a_artifacts"
OUTPUT_DIR = LOCAL / "track_a_output_stub"
SHELVES_CONFIG = BASE / "configs" / "shelves.yaml"


def load_candidates() -> list[Candidate]:
    df = pd.read_parquet(CANDIDATES_PARQUET)
    return [Candidate(**row.to_dict()) for _, row in df.iterrows()]


def load_poses() -> list[PoseObservation]:
    df = pd.read_parquet(POSES_PARQUET)
    return [PoseObservation(**row.to_dict()) for _, row in df.iterrows()]


def load_shelf_regions() -> dict[str, list[tuple[float, float]]]:
    cfg = load_shelf_config(SHELVES_CONFIG)
    cam_cfg = list(cfg.cameras.values())[0]
    expanded = get_expanded_regions(cam_cfg)
    return {rid: list(pts) for rid, pts in expanded.items()}


def run() -> TrackAInferencePipeline:
    candidates = load_candidates()
    poses = load_poses()
    shelf_regions = load_shelf_regions()

    logger.info("Candidates: %d", len(candidates))
    logger.info("Pose observations: %d", len(poses))
    logger.info("Shelf regions: %d", len(shelf_regions))
    logger.info("Video: %s (exists=%s)", VIDEO_PATH, VIDEO_PATH.exists())

    source_videos = {"clip_person_clear": VIDEO_PATH}

    pipeline = TrackAInferencePipeline(
        config=InferenceConfig(),
        state_machine_config=StateMachineConfig(),
    )

    result = pipeline.run(
        candidates=candidates,
        pose_observations=poses,
        source_videos=source_videos,
        hand_classifier_path=ARTIFACTS_DIR / "hand_state.joblib",
        shelf_classifier_path=ARTIFACTS_DIR / "shelf_state.joblib",
        output_dir=OUTPUT_DIR,
        shelf_regions=shelf_regions,
        embedder=None,
        cache_dir=None,
    )

    return result


def main() -> int:
    result = run()

    s = result.summary
    print(f"\n{'='*60}")
    print(f"Track A Inference (stub classifiers)")
    print(f"{'='*60}")
    print(f"Candidates total:       {s.candidates_total}")
    print(f"Candidates processed:   {s.candidates_processed}")
    print(f"Candidates skipped:     {s.candidates_skipped}")
    if s.skip_reasons:
        for reason, count in s.skip_reasons.items():
            print(f"  {reason}: {count}")
    print(f"Total samples:          {s.total_samples}")
    print(f"Raw events emitted:     {s.raw_events_emitted}")
    print(f"Final predictions:      {s.final_events_after_dedup}")
    print(f"  Pickups:              {s.pickup_count}")
    print(f"  Putdowns:             {s.putdown_count}")
    print(f"{'='*60}")

    if result.diagnostics:
        print(f"\nDiagnostics ({len(result.diagnostics)} candidates):")
        for diag in result.diagnostics:
            status = "SKIPPED" if diag.skipped else "PROCESSED"
            reason = f" ({diag.skip_reason})" if diag.skipped else ""
            print(f"  {diag.candidate_id}: {status}{reason}")
            if diag.skipped:
                print(f"    Skip reason: {diag.skip_reason}")

    output_dir = Path(result.output_paths.get("predictions_csv", OUTPUT_DIR))
    print(f"\nOutput files in {OUTPUT_DIR}:")
    if OUTPUT_DIR.exists():
        for f in sorted(OUTPUT_DIR.iterdir()):
            size = f.stat().st_size
            print(f"  {f.name} ({size:,} bytes)")

    pred_file = OUTPUT_DIR / "predictions.csv"
    if pred_file.exists():
        print(f"\npredictions.csv ({pred_file.stat().st_size} bytes):")
        with open(pred_file) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            print(f"  Rows: {len(rows)}")
            if rows:
                for row in rows[:5]:
                    print(f"  {row}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
