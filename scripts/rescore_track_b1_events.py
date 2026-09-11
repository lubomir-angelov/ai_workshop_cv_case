#!/usr/bin/env python3
"""Re-score saved Track B1 event predictions under every GT counting policy.

    python scripts/rescore_track_b1_events.py --dataset-dir .local/track_b1_dataset \
        --predictions-dir .local/track_b1_run/predictions_annotation --split val

Reads ``predictions_<split>.csv`` written by ``infer_track_b1.py`` and scores it, unchanged,
against the dataset's reviewed events of that split with the shared Task 8 evaluator,
once per policy in ``inference.COUNTING_POLICIES``: per item (primary) and per event group
(historical comparison with the CVAT branch). Writes
``rescored_by_counting_policy_<split>.json`` next to the predictions; never re-runs a model
and never touches the predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pickup_putdown.layer1.track_b1.dataset_dir import load_dataset_dir  # noqa: E402
from pickup_putdown.layer1.track_b1.inference import (  # noqa: E402
    PRIMARY_COUNTING_POLICY,
    evaluate_events_by_policy,
    ground_truth_counts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.3, 0.5])
    args = parser.parse_args(argv)

    dataset_dir = load_dataset_dir(args.dataset_dir)
    clips = dataset_dir.table("clips")
    split_clips = set(clips.loc[clips["split"] == args.split, "clip_id"])
    events = dataset_dir.table("events")
    truth = events[events["clip_id"].isin(split_clips)]
    ignores = dataset_dir.table("ignore_intervals")
    ignores = ignores[ignores["clip_id"].isin(split_clips)]
    durations = {
        c: float(d)
        for c, d in zip(clips["clip_id"], clips["duration_s"], strict=True)
        if c in split_clips
    }
    predictions = pd.read_csv(args.predictions_dir / f"predictions_{args.split}.csv")
    foreign = sorted(set(predictions["clip_id"]) - split_clips)
    if foreign:
        raise ValueError(f"predictions for clips outside split {args.split}: {foreign[:3]}")

    by_policy = evaluate_events_by_policy(predictions, truth, ignores, durations, tuple(args.tiou))
    record = {
        "dataset_dir": str(dataset_dir.path),
        "input_mode": dataset_dir.input_mode,
        "annotation_conditioned": dataset_dir.input_mode == "annotation",
        "split": args.split,
        "split_registry": dataset_dir.metadata["split_registry"]["name"],
        "predictions": str(args.predictions_dir / f"predictions_{args.split}.csv"),
        "n_predictions": int(len(predictions)),
        "ground_truth_counts": ground_truth_counts(truth),
        "primary_counting_policy": PRIMARY_COUNTING_POLICY,
        "event_metrics_by_counting_policy": by_policy,
    }
    out = args.predictions_dir / f"rescored_by_counting_policy_{args.split}.json"
    out.write_text(json.dumps(record, indent=2, default=str))
    for policy, metrics in by_policy.items():
        print(
            f"{policy:16s} GT rows={metrics['n_ground_truth_rows']:3d}  "
            + "  ".join(f"F1@{t}={metrics[f'tiou@{t}']['f1']:.3f}" for t in args.tiou)
            + "  "
            + "  ".join(f"{k}@0.5={v['f1']:.3f}" for k, v in metrics["per_type"].items())
        )
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
