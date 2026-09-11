#!/usr/bin/env python3
"""Run Track B1 sliding-window inference over the cached candidates and evaluate it.

    python scripts/infer_track_b1.py --split val --checkpoint .local/track_b1_run/checkpoints/best.pt

Windows are scored from the crop cache, class probabilities are smoothed over time,
score peaks are turned into regions, and only same-type regions are merged — a pickup
and an adjacent putdown stay two events, which is the behaviour Task 12 asks for.

Predictions are written in canonical form and, when ground truth is available for the
split, scored with the shared Task 8 evaluator.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pickup_putdown.layer1.track_b1.cache import load_cache_index  # noqa: E402
from pickup_putdown.layer1.track_b1.dataset import (  # noqa: E402
    generate_inference_windows_for_candidate,
    normalize_frames,
)
from pickup_putdown.layer1.track_b1.inference import (  # noqa: E402
    InferenceConfig,
    WindowPrediction,
    create_event_predictions,
    detect_score_peaks,
    merge_same_type_regions,
    smooth_predictions,
)
from pickup_putdown.layer1.track_b1.videomae_classifier import (  # noqa: E402
    _resolve_device,
    create_model,
    load_checkpoint,
)


def load_into(model, checkpoint_path: Path, device: torch.device) -> None:
    """Load either a full-model checkpoint or a head-only one into ``model``.

    The head trainer works on cached embeddings and so only ever sees the head; its
    checkpoint carries ``head_state_dict``. The pixel-path trainer saves the whole
    model. Both are valid Track B1 models — the backbone is the same frozen
    pretrained encoder in either case — so inference accepts both.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "head_state_dict" in checkpoint:
        model.head.load_state_dict(checkpoint["head_state_dict"])
        logger.info(
            "loaded head-only checkpoint (epoch %s, val macro F1 %.4f)",
            checkpoint.get("epoch"), checkpoint.get("val_f1_macro", float("nan")),
        )
    else:
        load_checkpoint(checkpoint_path, model, device=str(device))

logger = logging.getLogger("infer_track_b1")


def score_candidate_windows(
    model,
    entry,
    windows,
    num_frames: int,
    batch_size: int,
    device: torch.device,
) -> list[WindowPrediction]:
    """Score one candidate's windows straight out of its cached frames."""
    if not windows:
        return []

    frames_array = np.load(entry.array_path, mmap_mode="r")
    tensors = []
    for window in windows:
        start_frame = int(window.window_start_s * entry.fps)
        end_frame = max(start_frame + 1, int(window.window_end_s * entry.fps))
        wanted = np.linspace(start_frame, end_frame - 1, num_frames)
        positions = [entry.frame_position(int(f)) for f in wanted]
        tensors.append(normalize_frames(np.asarray(frames_array[positions])))

    predictions: list[WindowPrediction] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start : start + batch_size]).to(device)
            probs = torch.softmax(model(batch), dim=-1).cpu().numpy()
            for offset, window_probs in enumerate(probs):
                window = windows[start + offset]
                predicted = int(np.argmax(window_probs))
                predictions.append(
                    WindowPrediction(
                        window_start_s=window.window_start_s,
                        window_end_s=window.window_end_s,
                        window_center_s=(window.window_start_s + window.window_end_s) / 2,
                        probs=window_probs,
                        predicted_class=predicted,
                        confidence=float(window_probs[predicted]),
                    )
                )
    return predictions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / ".local/track_b1_dataset")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / ".local/track_b1_cache")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / ".local/track_b1_run/predictions")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])

    parser.add_argument("--model-name", default="MCG-NJU/videomae-base")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--window-duration-s", type=float, default=2.5)
    parser.add_argument("--window-stride-s", type=float, default=0.5)
    parser.add_argument("--pickup-threshold", type=float, default=0.5)
    parser.add_argument("--putdown-threshold", type=float, default=0.5)
    parser.add_argument("--smoothing-window", type=int, default=3)
    parser.add_argument("--same-type-merge-gap-s", type=float, default=0.75)
    parser.add_argument("--min-event-duration-s", type=float, default=0.3)
    parser.add_argument("--boundary-mode", default="window_centers",
                        choices=["window_span", "window_centers"],
                        help="How a run of above-threshold windows becomes an interval")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.3, 0.5])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    clips = pd.read_parquet(args.dataset_dir / "clips.parquet")
    candidates = pd.read_parquet(args.dataset_dir / "candidates.parquet")
    events = pd.read_parquet(args.dataset_dir / "events.parquet")
    ignores = pd.read_parquet(args.dataset_dir / "ignore_intervals.parquet")

    split_clips = set(clips[clips["split"] == args.split]["clip_id"])
    candidates = candidates[candidates["clip_id"].isin(split_clips)]
    if candidates.empty:
        logger.error("no candidates in split %s", args.split)
        return 1

    cache_index = load_cache_index(args.cache_dir)
    device = _resolve_device(args.device)

    model = create_model(
        model_name=args.model_name, num_classes=3, freeze_backbone=True, device=args.device
    )
    load_into(model, args.checkpoint, device)
    logger.info("loaded checkpoint %s", args.checkpoint)

    config = InferenceConfig(
        window_duration_s=args.window_duration_s,
        window_stride_s=args.window_stride_s,
        num_frames=args.num_frames,
        pickup_threshold=args.pickup_threshold,
        putdown_threshold=args.putdown_threshold,
        smoothing_window=args.smoothing_window,
        same_type_merge_gap_s=args.same_type_merge_gap_s,
        min_event_duration_s=args.min_event_duration_s,
        boundary_mode=args.boundary_mode,
        batch_size=args.batch_size,
    )
    window_config = config.to_window_config()

    all_events: list[dict] = []
    window_rows: list[dict] = []
    skipped = 0

    for position, (_, candidate) in enumerate(candidates.iterrows(), start=1):
        entry = cache_index.get(candidate["candidate_id"])
        if entry is None:
            skipped += 1
            continue

        windows = generate_inference_windows_for_candidate(candidate, window_config)
        predictions = score_candidate_windows(
            model, entry, windows, args.num_frames, args.batch_size, device
        )
        if not predictions:
            continue

        for prediction in predictions:
            window_rows.append(
                {
                    "clip_id": candidate["clip_id"],
                    "candidate_id": candidate["candidate_id"],
                    "actor_id": candidate["actor_id"],
                    "window_start_s": prediction.window_start_s,
                    "window_end_s": prediction.window_end_s,
                    "p_background": prediction.background_prob,
                    "p_pickup": prediction.pickup_prob,
                    "p_putdown": prediction.putdown_prob,
                }
            )

        smoothed = smooth_predictions(predictions, config.smoothing_window)
        regions = detect_score_peaks(smoothed, config)
        merged = merge_same_type_regions(
            regions, config.same_type_merge_gap_s, config.min_event_duration_s
        )
        detected = create_event_predictions(
            merged, candidate["clip_id"], candidate["candidate_id"], candidate["actor_id"], config
        )
        all_events.extend(event.to_dict() for event in detected)

        if position % 25 == 0:
            logger.info("  %d/%d candidates scored", position, len(candidates))

    predictions_df = pd.DataFrame(all_events)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / f"predictions_{args.split}.csv"
    predictions_df.to_csv(predictions_path, index=False)
    pd.DataFrame(window_rows).to_parquet(
        args.output_dir / f"window_scores_{args.split}.parquet", index=False
    )

    print(f"\n{len(predictions_df)} predicted events from {len(candidates)} candidates "
          f"({skipped} uncached) -> {predictions_path}")
    if not predictions_df.empty:
        print("by type:", predictions_df["type"].value_counts().to_dict())

    ground_truth = events[events["clip_id"].isin(split_clips)]
    print(f"ground truth in {args.split}: {len(ground_truth)} events "
          f"{ground_truth['type'].value_counts().to_dict()}")

    split_ignores = ignores[ignores["clip_id"].isin(split_clips)]
    metrics = evaluate(predictions_df, ground_truth, split_ignores, clips, args)
    (args.output_dir / f"metrics_{args.split}.json").write_text(json.dumps(metrics, indent=2))
    return 0


def evaluate(
    predictions_df: pd.DataFrame,
    ground_truth: pd.DataFrame,
    ignore_df: pd.DataFrame,
    clips: pd.DataFrame,
    args,
) -> dict:
    """Score predictions with the shared Task 8 evaluator."""
    from pickup_putdown.evaluation.contracts import (
        EvaluationEvent,
        EvaluationIgnoreInterval,
        EvaluationPrediction,
    )
    from pickup_putdown.evaluation.metrics import aggregate_metrics

    truth = [
        EvaluationEvent(
            event_id=row.event_id, clip_id=row.clip_id, type=row.type,
            t_start=float(row.t_start), t_end=float(row.t_end),
            confidence=row.confidence, hard_case=bool(row.hard_case),
        )
        for row in ground_truth.itertuples()
    ]
    predicted = (
        [
            EvaluationPrediction(
                pred_id=row.pred_id, clip_id=row.clip_id, type=row.type,
                t_start=float(row.t_start), t_end=float(row.t_end),
                score=float(row.score), model=row.model,
            )
            for row in predictions_df.itertuples()
        ]
        if not predictions_df.empty
        else []
    )
    ignores = [
        EvaluationIgnoreInterval(
            clip_id=row.clip_id, t_start=float(row.t_start), t_end=float(row.t_end)
        )
        for row in ignore_df.itertuples()
    ]
    # Only the evaluated split's footage: fp_per_hour divides by this sum, and passing
    # every clip's duration understated the rate ~3.7x (2.31 h against 0.63 h of test).
    split_only = clips[clips["clip_id"].isin(set(ground_truth["clip_id"]) | set(predictions_df.get("clip_id", [])))]
    clip_durations = dict(zip(split_only["clip_id"], split_only["duration_s"].astype(float)))

    results = aggregate_metrics(
        events=truth,
        preds=predicted,
        clip_durations=clip_durations,
        ignores=ignores,
        tiou_thresholds=tuple(args.tiou),
    )

    print("\nEvaluation (shared Task 8 evaluator):")
    for tiou in args.tiou:
        row = results.get(f"tiou@{tiou}", {})
        print(
            f"  tIoU {tiou}: P={row.get('precision', 0):.3f} R={row.get('recall', 0):.3f} "
            f"F1={row.get('f1', 0):.3f}  (tp={row.get('tp')} fp={row.get('fp')} fn={row.get('fn')})"
        )
    for event_type, row in (results.get("per_type") or {}).items():
        print(
            f"  {event_type:8s}: P={row.get('precision', 0):.3f} R={row.get('recall', 0):.3f} "
            f"F1={row.get('f1', 0):.3f}"
        )
    if results.get("start_mae_s") is not None:
        print(f"  boundary MAE: start={results['start_mae_s']:.2f}s end={results['end_mae_s']:.2f}s")
    return results


if __name__ == "__main__":
    sys.exit(main())
