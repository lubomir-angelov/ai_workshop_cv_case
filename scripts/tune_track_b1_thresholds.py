#!/usr/bin/env python3
"""Tune Track B1 decision thresholds on validation window scores.

    python scripts/tune_track_b1_thresholds.py --dataset-dir .local/track_b1_dataset \
        --predictions-dir .local/track_b1_run/predictions_annotation

Reads the per-window class probabilities that ``infer_track_b1.py`` writes for the
validation split and re-runs only the decode chain — smooth, peak, same-type merge —
over a grid of pickup/putdown thresholds. The model is never re-run, so a full sweep
costs seconds.

Selection is on the mean of validation event-level F1 at tIoU 0.3 and 0.5 from the
shared Task 8 evaluator (the rule the configured thresholds were chosen by).
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

from pickup_putdown.layer1.track_b1.dataset_dir import load_dataset_dir  # noqa: E402
from pickup_putdown.layer1.track_b1.inference import (  # noqa: E402
    InferenceConfig,
    decode_window_scores,
    evaluate_events,
    suppress_duplicate_events,
)

logger = logging.getLogger("tune_track_b1_thresholds")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / ".local/track_b1_dataset")
    parser.add_argument("--predictions-dir", type=Path, required=True,
                        help="directory holding window_scores_val.parquet from infer_track_b1.py")
    parser.add_argument("--split", default="val", choices=["val"])
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.3, 0.5],
                        help="selection maximises the mean F1 over these")
    parser.add_argument("--grid", type=float, nargs="+",
                        default=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7])
    parser.add_argument("--smoothing-windows", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--same-type-merge-gap-s", type=float, default=0.75)
    parser.add_argument("--min-event-duration-s", type=float, default=0.3)
    parser.add_argument("--boundary-mode", default="window_centers",
                        choices=["window_span", "window_centers"],
                        help="How a run of above-threshold windows becomes an interval")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    dataset_dir = load_dataset_dir(args.dataset_dir)
    window = dataset_dir.metadata["window"]

    scores_path = args.predictions_dir / f"window_scores_{args.split}.parquet"
    if not scores_path.exists():
        logger.error("no window scores at %s; run infer_track_b1.py first", scores_path)
        return 1
    scores = pd.read_parquet(scores_path)

    clips = dataset_dir.table("clips")
    split_clips = set(clips.loc[clips["split"] == args.split, "clip_id"])
    events = dataset_dir.table("events")
    ignores = dataset_dir.table("ignore_intervals")
    truth = events[events["clip_id"].isin(split_clips)]
    split_ignores = ignores[ignores["clip_id"].isin(split_clips)]
    durations = {c: float(d) for c, d in zip(clips["clip_id"], clips["duration_s"], strict=True) if c in split_clips}

    results: list[dict] = []
    for smoothing in args.smoothing_windows:
        for pickup_threshold in args.grid:
            for putdown_threshold in args.grid:
                config = InferenceConfig(
                    window_duration_s=window["window_duration_s"],
                    window_stride_s=window["window_stride_s"],
                    pickup_threshold=pickup_threshold,
                    putdown_threshold=putdown_threshold,
                    smoothing_window=smoothing,
                    same_type_merge_gap_s=args.same_type_merge_gap_s,
                    min_event_duration_s=args.min_event_duration_s,
                    boundary_mode=args.boundary_mode,
                )
                predictions, _ = suppress_duplicate_events(decode_window_scores(scores, config))
                metrics = evaluate_events(predictions, truth, split_ignores, durations, tuple(args.tiou))
                row = {"smoothing_window": smoothing, "pickup_threshold": pickup_threshold,
                       "putdown_threshold": putdown_threshold}
                for tiou in args.tiou:
                    row.update({f"{k}@{tiou}": metrics[f"tiou@{tiou}"][k] for k in ("f1", "precision", "recall")})
                row["objective"] = float(np.mean([row[f"f1@{t}"] for t in args.tiou]))
                results.append(row)

    table = pd.DataFrame(results).sort_values(
        ["objective", "smoothing_window", "pickup_threshold", "putdown_threshold"],
        ascending=[False, True, True, True],
    )
    tag = f"{dataset_dir.input_mode}_{args.boundary_mode}"
    table.to_csv(args.predictions_dir / f"threshold_sweep_{args.split}_{tag}.csv", index=False)
    best = table.iloc[0]
    print(f"\nTop 10 by mean validation F1 over tIoU {args.tiou} ({dataset_dir.input_mode} inputs):")
    print(table.head(10).to_string(index=False))

    chosen = {
        "input_mode": dataset_dir.input_mode,
        "tiou": args.tiou,
        "boundary_mode": args.boundary_mode,
        "window_duration_s": window["window_duration_s"],
        "window_stride_s": window["window_stride_s"],
        "pickup_threshold": float(best["pickup_threshold"]),
        "putdown_threshold": float(best["putdown_threshold"]),
        "smoothing_window": int(best["smoothing_window"]),
        "same_type_merge_gap_s": args.same_type_merge_gap_s,
        "min_event_duration_s": args.min_event_duration_s,
        "val_objective": float(best["objective"]),
        **{k: float(best[k]) for k in best.index if "@" in k},
        "selected_on": "validation split only; test not read",
    }
    # Tagged per sweep so a later sweep cannot silently overwrite an earlier selection.
    (args.predictions_dir / f"chosen_thresholds_{tag}.json").write_text(json.dumps(chosen, indent=2))
    print(f"\nchosen: {json.dumps(chosen, indent=2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
