#!/usr/bin/env python3
"""Tune Track B1 decision thresholds on validation window scores.

    python scripts/tune_track_b1_thresholds.py

Reads the per-window class probabilities that ``infer_track_b1.py`` writes for the
validation split and re-runs only the decode chain — smooth, peak, same-type merge —
over a grid of pickup/putdown thresholds. The model is never re-run, so a full sweep
costs seconds.

Selection is on validation event-level F1 from the shared Task 8 evaluator at tIoU 0.3.
The chosen thresholds are written out to be recorded in configuration and applied to
test exactly once; nothing here reads the test split.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pickup_putdown.layer1.track_b1.inference import (  # noqa: E402
    InferenceConfig,
    WindowPrediction,
    create_event_predictions,
    detect_score_peaks,
    merge_same_type_regions,
    smooth_predictions,
)

logger = logging.getLogger("tune_track_b1_thresholds")


def decode(scores: pd.DataFrame, config: InferenceConfig) -> pd.DataFrame:
    """Run the decode chain over precomputed window scores for every candidate."""
    rows: list[dict] = []
    for (clip_id, candidate_id, actor_id), group in scores.groupby(
        ["clip_id", "candidate_id", "actor_id"], sort=False
    ):
        group = group.sort_values("window_start_s")
        predictions = [
            WindowPrediction(
                window_start_s=row.window_start_s,
                window_end_s=row.window_end_s,
                window_center_s=(row.window_start_s + row.window_end_s) / 2,
                probs=np.array([row.p_background, row.p_pickup, row.p_putdown]),
                predicted_class=int(
                    np.argmax([row.p_background, row.p_pickup, row.p_putdown])
                ),
                confidence=float(max(row.p_background, row.p_pickup, row.p_putdown)),
            )
            for row in group.itertuples()
        ]
        smoothed = smooth_predictions(predictions, config.smoothing_window)
        regions = detect_score_peaks(smoothed, config)
        merged = merge_same_type_regions(
            regions, config.same_type_merge_gap_s, config.min_event_duration_s
        )
        rows.extend(
            event.to_dict()
            for event in create_event_predictions(
                merged, clip_id, candidate_id, actor_id, config
            )
        )
    return pd.DataFrame(rows)


def score(predictions: pd.DataFrame, truth, ignores, clip_durations, tiou: float) -> dict:
    from pickup_putdown.evaluation.contracts import EvaluationPrediction
    from pickup_putdown.evaluation.metrics import aggregate_metrics

    predicted = (
        [
            EvaluationPrediction(
                pred_id=row.pred_id, clip_id=row.clip_id, type=row.type,
                t_start=float(row.t_start), t_end=float(row.t_end),
                score=float(row.score), model=row.model,
            )
            for row in predictions.itertuples()
        ]
        if not predictions.empty
        else []
    )
    results = aggregate_metrics(
        events=truth, preds=predicted, clip_durations=clip_durations,
        ignores=ignores, tiou_thresholds=(tiou,),
    )
    return results[f"tiou@{tiou}"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / ".local/track_b1_dataset")
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / ".local/track_b1_run")
    parser.add_argument("--split", default="val", choices=["val"])
    parser.add_argument("--tiou", type=float, default=0.3)
    parser.add_argument("--grid", type=float, nargs="+",
                        default=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7])
    parser.add_argument("--smoothing-windows", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--window-duration-s", type=float, default=2.5)
    parser.add_argument("--window-stride-s", type=float, default=0.5)
    parser.add_argument("--same-type-merge-gap-s", type=float, default=0.75)
    parser.add_argument("--min-event-duration-s", type=float, default=0.3)
    parser.add_argument("--boundary-mode", default="window_centers",
                        choices=["window_span", "window_centers"],
                        help="How a run of above-threshold windows becomes an interval")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    from pickup_putdown.evaluation.contracts import EvaluationEvent, EvaluationIgnoreInterval

    scores_path = args.run_dir / "predictions" / f"window_scores_{args.split}.parquet"
    if not scores_path.exists():
        logger.error("no window scores at %s; run infer_track_b1.py first", scores_path)
        return 1
    scores = pd.read_parquet(scores_path)

    clips = pd.read_parquet(args.dataset_dir / "clips.parquet")
    events = pd.read_parquet(args.dataset_dir / "events.parquet")
    ignores_df = pd.read_parquet(args.dataset_dir / "ignore_intervals.parquet")

    split_clips = set(clips[clips["split"] == args.split]["clip_id"])
    truth = [
        EvaluationEvent(
            event_id=row.event_id, clip_id=row.clip_id, type=row.type,
            t_start=float(row.t_start), t_end=float(row.t_end),
        )
        for row in events[events["clip_id"].isin(split_clips)].itertuples()
    ]
    ignores = [
        EvaluationIgnoreInterval(
            clip_id=row.clip_id, t_start=float(row.t_start), t_end=float(row.t_end)
        )
        for row in ignores_df[ignores_df["clip_id"].isin(split_clips)].itertuples()
    ]
    clip_durations = dict(zip(clips["clip_id"], clips["duration_s"].astype(float)))

    results: list[dict] = []
    for smoothing in args.smoothing_windows:
        for pickup_threshold in args.grid:
            for putdown_threshold in args.grid:
                config = InferenceConfig(
                    window_duration_s=args.window_duration_s,
                    window_stride_s=args.window_stride_s,
                    pickup_threshold=pickup_threshold,
                    putdown_threshold=putdown_threshold,
                    smoothing_window=smoothing,
                    same_type_merge_gap_s=args.same_type_merge_gap_s,
                    min_event_duration_s=args.min_event_duration_s,
                    boundary_mode=args.boundary_mode,
                )
                metrics = score(decode(scores, config), truth, ignores, clip_durations, args.tiou)
                results.append(
                    {
                        "smoothing_window": smoothing,
                        "pickup_threshold": pickup_threshold,
                        "putdown_threshold": putdown_threshold,
                        **metrics,
                    }
                )

    table = pd.DataFrame(results).sort_values("f1", ascending=False)
    tag = f"tiou{args.tiou}_{args.boundary_mode}"
    table.to_csv(args.run_dir / f"threshold_sweep_{args.split}_{tag}.csv", index=False)

    best = table.iloc[0]
    print(f"\nTop 10 by validation F1 @ tIoU {args.tiou}:")
    print(table.head(10).to_string(index=False))

    chosen = {
        "tiou": args.tiou,
        "boundary_mode": args.boundary_mode,
        "window_duration_s": args.window_duration_s,
        "window_stride_s": args.window_stride_s,
        "pickup_threshold": float(best["pickup_threshold"]),
        "putdown_threshold": float(best["putdown_threshold"]),
        "smoothing_window": int(best["smoothing_window"]),
        "same_type_merge_gap_s": args.same_type_merge_gap_s,
        "min_event_duration_s": args.min_event_duration_s,
        "val_f1": float(best["f1"]),
        "val_precision": float(best["precision"]),
        "val_recall": float(best["recall"]),
        "selected_on": "validation split only; test not read",
    }
    # Tagged per sweep: a later run at a different tIoU or boundary mode must not
    # silently overwrite the selection an earlier one made.
    (args.run_dir / f"chosen_thresholds_{tag}.json").write_text(json.dumps(chosen, indent=2))
    print(f"\nchosen: {json.dumps(chosen, indent=2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
