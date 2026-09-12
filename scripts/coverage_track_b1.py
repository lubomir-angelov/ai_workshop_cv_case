#!/usr/bin/env python3
"""Coverage, crop and error attribution for a deployment-input Track B1 evaluation.

    python scripts/coverage_track_b1.py --dataset-dir .local/track_b1_dataset_deploy \
        --predictions-dir <infer_track_b1.py output dir> --split val

Model-free: reads the dataset (pose candidates, CVAT events associated to pose actors,
CVAT boxes kept for diagnostics) and the ``predictions_<split>.csv`` /
``window_scores_<split>.parquet`` that ``infer_track_b1.py`` wrote. CVAT is used here only
for evaluation and diagnostics; nothing feeds back into candidates, crops or decoding.

Per GT item row (``event_coverage_<split>.csv``): association status, proposal/actor flags,
the ``actor_association.COVERAGE_PARTITION`` label, the fraction of the annotated
interaction box inside the model's crop for windows centred in the event (own actor and
best of any actor), the share of the crop that box occupies (the scale the model sees)
and the crop's share of the frame, and whether it was matched at each tIoU (per-item counting). The
summary (``coverage_<split>.json``) reports every count over item rows and over event
groups, checks that the partition reconciles, attributes false negatives to coverage
categories and false positives to what they overlap, and gives window-level metrics on
windows whose labels are defensible (the dataset's manifest), with exclusion counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pickup_putdown.evaluation.class_aware_matching import (  # noqa: E402
    drop_ignored,
    evaluate_class_aware,
)
from pickup_putdown.evaluation.intervals import Criterion  # noqa: E402
from pickup_putdown.layer1.track_b1.actor_association import (  # noqa: E402
    COVERAGE_PARTITION,
    event_coverage,
    window_evidence,
)
from pickup_putdown.layer1.track_b1.dataset import (  # noqa: E402
    compute_actor_crop_box,
    crop_span,
    effective_shelf_region,
    generate_inference_windows,
)
from pickup_putdown.layer1.track_b1.dataset_dir import load_dataset_dir  # noqa: E402
from pickup_putdown.layer1.track_b1.inference import ground_truth_counts  # noqa: E402

NAMES = ["background", "pickup", "putdown"]
WINDOW_KEY = ["clip_id", "candidate_id", "actor_id", "window_start_s", "window_end_s"]


def crop_coverage(table, windows, dataset_dir, clips) -> pd.DataFrame:
    """Annotated-box-in-crop for windows centred in each event: own actor mean, any-actor best."""
    config = dataset_dir.window_config()
    shelves = dataset_dir.shelf_regions()
    pose, cvat, rows = {}, {}, []
    for _, event in table.iterrows():
        clip_id = event["clip_id"]
        if clip_id not in pose:
            pose[clip_id] = pd.read_parquet(dataset_dir.tracks_dir / f"{clip_id}.parquet")
            cvat[clip_id] = pd.read_parquet(
                dataset_dir.path / "cvat_tracks" / f"{clip_id}.parquet"
            )
        clip = clips.loc[clip_id]
        frame_size = (int(clip["width"]), int(clip["height"]))
        centred = windows[
            (windows["clip_id"] == clip_id)
            & (windows["window_center_s"] >= event["t_start"])
            & (windows["window_center_s"] <= event["t_end"])
        ]
        box_event = event.copy()
        box_event["actor_id"] = event["cvat_actor_id"]
        own, own_share, own_area, best, full_frame = [], [], [], [], []
        for _, window in centred.iterrows():
            track = pose[clip_id][pose[clip_id]["actor_id"] == window["actor_id"]]
            crop = compute_actor_crop_box(
                track,
                effective_shelf_region(shelves, window.get("region_id"), config),
                *crop_span(window, config),
                config.crop_margin,
                frame_size,
            )
            evidence = window_evidence(
                window["window_start_s"],
                window["window_end_s"],
                config.num_frames,
                float(clip["fps"]),
                crop,
                box_event,
                cvat[clip_id],
            )
            share = evidence["event_box_in_crop"]
            if share is None:
                continue
            best.append(share)
            if window["actor_id"] == event["actor_id"]:
                own.append(share)
                own_share.append(evidence["event_box_share_of_crop"])
                own_area.append(
                    (crop[2] - crop[0]) * (crop[3] - crop[1]) / (frame_size[0] * frame_size[1])
                )
                full_frame.append(tuple(crop) == (0, 0, *frame_size))
        rows.append(
            {
                "event_id": event["event_id"],
                "own_box_in_crop_mean": float(np.mean(own)) if own else None,
                "own_box_share_of_crop_mean": float(np.mean(own_share)) if own_share else None,
                "own_crop_share_of_frame_mean": float(np.mean(own_area)) if own_area else None,
                "own_crop_full_frame": any(full_frame) if full_frame else None,
                "best_box_in_crop_any_actor": float(max(best)) if best else None,
            }
        )
    return pd.DataFrame(rows)


def match_flags(truth, predictions, ignores, tiou_thresholds):
    """Per-item match flags at each tIoU and false-positive attribution (shared evaluator)."""
    events = list(truth.itertuples(index=False))
    preds = list(predictions.itertuples(index=False))
    ignored = {e.event_id for e in events} - {e.event_id for e in drop_ignored(events, ignores)}
    flags, fp_attribution = {}, {}
    for thr in tiou_thresholds:
        result = evaluate_class_aware(
            events, preds, Criterion("tiou", tiou_threshold=thr), ignores
        )
        matched = {g.event_id for g, _ in result.matched}
        flags[f"matched@{thr}"] = truth["event_id"].map(
            lambda e, m=matched: "ignored" if e in ignored else e in m
        )
        kinds = {"overlaps_same_type_event": 0, "overlaps_other_type_only": 0, "no_event": 0}
        for p in result.unmatched_pred:
            near = truth[
                (truth["clip_id"] == p.clip_id)
                & (truth["t_start"] < p.t_end)
                & (truth["t_end"] > p.t_start)
            ]
            if (near["type"] == p.type).any():
                kinds["overlaps_same_type_event"] += 1
            elif not near.empty:
                kinds["overlaps_other_type_only"] += 1
            else:
                kinds["no_event"] += 1
        fp_attribution[f"tiou@{thr}"] = kinds
    return pd.DataFrame(flags, index=truth.index), ignored, fp_attribution


def window_metrics(scores, manifest, ignores) -> dict:
    """Argmax window metrics on manifest-labelled windows; exclusions counted, not scored."""
    labelled = scores.merge(
        manifest[[*WINDOW_KEY, "label"]], on=WINDOW_KEY, how="inner", validate="one_to_one"
    )
    if len(labelled) != len(manifest):
        raise ValueError(f"{len(manifest) - len(labelled)} manifest windows have no score")
    centre = (scores["window_start_s"] + scores["window_end_s"]) / 2
    in_ignore = np.zeros(len(scores), dtype=bool)
    for _, ignore in ignores.iterrows():
        in_ignore |= (
            (scores["clip_id"] == ignore["clip_id"])
            & (centre >= ignore["t_start"])
            & (centre <= ignore["t_end"])
        ).to_numpy()
    predicted = labelled[["p_background", "p_pickup", "p_putdown"]].to_numpy().argmax(axis=1)
    truth = labelled["label"].astype(int).to_numpy()
    report = classification_report(
        truth, predicted, labels=[0, 1, 2], target_names=NAMES, output_dict=True, zero_division=0
    )
    return {
        "n_inference_windows": int(len(scores)),
        "n_labelled_windows": int(len(labelled)),
        "n_excluded_ignore_interval": int(in_ignore.sum()),
        "n_excluded_unresolved_actor": int(len(scores) - len(labelled) - in_ignore.sum()),
        "label_counts": {NAMES[k]: int(v) for k, v in pd.Series(truth).value_counts().items()},
        "accuracy": float((truth == predicted).mean()),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {
            n: {k: float(report[n][k]) for k in ("precision", "recall", "f1-score")} for n in NAMES
        },
        "confusion_rows_true_cols_pred": confusion_matrix(
            truth, predicted, labels=[0, 1, 2]
        ).tolist(),
    }


def counts(series: pd.Series) -> dict:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).sort_index().items()}


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
    if dataset_dir.input_mode != "deployment":
        parser.error("coverage attribution needs a deployment-input dataset")
    out_table = args.predictions_dir / f"event_coverage_{args.split}.csv"
    out_summary = args.predictions_dir / f"coverage_{args.split}.json"
    if out_table.exists() or out_summary.exists():
        parser.error(f"{out_summary} exists; not overwriting")

    clips = dataset_dir.table("clips").set_index("clip_id")
    split_clips = set(clips.index[clips["split"] == args.split])
    events = dataset_dir.table("events_pose_identity")
    events = events[events["clip_id"].isin(split_clips)].reset_index(drop=True)
    candidates = dataset_dir.table("candidates")
    candidates = candidates[candidates["clip_id"].isin(split_clips)]
    ignores = dataset_dir.table("ignore_intervals")
    ignores = ignores[ignores["clip_id"].isin(split_clips)]
    windows = pd.DataFrame(
        [w.to_dict() for w in generate_inference_windows(candidates, dataset_dir.window_config())]
    )
    predictions = pd.read_csv(args.predictions_dir / f"predictions_{args.split}.csv")
    scores = pd.read_parquet(args.predictions_dir / f"window_scores_{args.split}.parquet")
    if len(scores) != len(windows):
        raise ValueError(f"{len(scores)} scored windows but {len(windows)} inference windows")

    table = events.merge(event_coverage(events, candidates, windows), on="event_id")
    table = table.merge(crop_coverage(table, windows, dataset_dir, clips), on="event_id")
    flags, ignored, fp_attribution = match_flags(
        table, predictions, list(ignores.itertuples(index=False)), args.tiou
    )
    table = pd.concat([table, flags], axis=1)
    table["ignored"] = table["event_id"].isin(ignored)

    groups = table.groupby(["clip_id", "event_group_id"], sort=True)
    inconsistent = groups[["coverage", "association_status"]].nunique().gt(1).any(axis=1)
    if inconsistent.any():
        raise ValueError(
            f"items of one event group differ in coverage: {inconsistent[inconsistent].index[:3].tolist()}"
        )
    group_table = groups.head(1)

    def summarise(frame: pd.DataFrame) -> dict:
        scored = frame[~frame["ignored"]]
        partition = counts(frame["coverage"])
        return {
            "n": int(len(frame)),
            "n_ignored_by_ignore_intervals": int(frame["ignored"].sum()),
            "association_status": counts(frame["association_status"]),
            "flags": {
                "any_candidate_overlap": int((frame["n_candidates_any_actor"] > 0).sum()),
                "any_window_centre": int(frame["any_window_centre"].sum()),
                "association_matched": int((frame["association_status"] == "matched").sum()),
                "own_candidate_overlap": int(frame["own_candidate_overlap"].sum()),
                "own_window_centre": int(frame["own_window_centre"].sum()),
            },
            "partition": {k: partition.get(k, 0) for k in COVERAGE_PARTITION},
            "partition_reconciles": sum(partition.values()) == len(frame)
            and set(partition) <= set(COVERAGE_PARTITION),
            "matched_by_partition": {
                f"tiou@{t}": {
                    k: {
                        "matched": int((sub[f"matched@{t}"] == True).sum()),  # noqa: E712
                        "missed": int((sub[f"matched@{t}"] == False).sum()),  # noqa: E712
                    }
                    for k in COVERAGE_PARTITION
                    for sub in [scored[scored["coverage"] == k]]
                }
                for t in args.tiou
            },
            "crop": {
                k: {
                    "n_with_own_crop": int(sub["own_box_in_crop_mean"].notna().sum()),
                    "own_box_in_crop_median": _median(sub["own_box_in_crop_mean"]),
                    "own_box_in_crop_below_0.5": int((sub["own_box_in_crop_mean"] < 0.5).sum()),
                    "own_box_share_of_crop_median": _median(sub["own_box_share_of_crop_mean"]),
                    "own_crop_share_of_frame_median": _median(sub["own_crop_share_of_frame_mean"]),
                    "own_crop_full_frame": int((sub["own_crop_full_frame"] == True).sum()),  # noqa: E712
                    "n_with_any_crop": int(sub["best_box_in_crop_any_actor"].notna().sum()),
                    "best_any_actor_box_in_crop_median": _median(
                        sub["best_box_in_crop_any_actor"]
                    ),
                }
                for k in COVERAGE_PARTITION
                for sub in [frame[frame["coverage"] == k]]
            },
        }

    manifest = dataset_dir.table("window_manifest")
    metrics_path = args.predictions_dir / f"metrics_{args.split}.json"
    inference = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    summary = {
        "dataset_dir": str(dataset_dir.path),
        "split": args.split,
        "split_registry": dataset_dir.metadata["split_registry"]["name"],
        "predictions_dir": str(args.predictions_dir),
        "partition_precedence": list(COVERAGE_PARTITION),
        "ground_truth_counts": ground_truth_counts(events),
        "per_item": summarise(table),
        "per_event_group": summarise(group_table),
        "false_positive_attribution": fp_attribution,
        "n_predictions": int(len(predictions)),
        "n_suppressed_duplicates": inference.get("n_suppressed_duplicates"),
        "n_candidates": int(len(candidates)),
        "n_candidates_without_windows": int(
            len(set(candidates["candidate_id"]) - set(windows["candidate_id"]))
        ),
        "window_level": window_metrics(scores, manifest[manifest["split"] == args.split], ignores),
    }
    table.to_csv(out_table, index=False)
    out_summary.write_text(json.dumps(summary, indent=2, default=str))
    for policy in ("per_item", "per_event_group"):
        part = summary[policy]
        print(
            f"[{args.split} {policy}] n={part['n']} reconciles={part['partition_reconciles']} {part['partition']}"
        )
    print(
        f"window-level: {summary['window_level']['n_labelled_windows']} labelled, macro F1 {summary['window_level']['macro_f1']:.4f}"
    )
    print(f"-> {out_summary}")
    return 0


def _median(values: pd.Series) -> float | None:
    values = values.dropna()
    return float(values.median()) if len(values) else None


if __name__ == "__main__":
    sys.exit(main())
