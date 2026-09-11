#!/usr/bin/env python3
"""Score Track B1 windows, decode events and evaluate them, in either input mode.

    python scripts/infer_track_b1.py --dataset-dir .local/track_b1_dataset \
        --checkpoint .local/track_b1_run/checkpoints/head_best.pt --split val
    python scripts/infer_track_b1.py --dataset-dir .local/track_b1_dataset_deploy \
        --checkpoint .local/track_b1_run/checkpoints/head_best.pt --split val

Windows come from the dataset's candidates (CVAT-derived in annotation mode, pose
proposals in deployment mode) and are scored through the same pixel path as
training. Class probabilities are smoothed, turned into regions, merged only with
the same type (a pickup next to a putdown stays two events), and same-type duplicates
of one actor from overlapping candidates are suppressed.

Evaluation uses the shared Task 8 evaluator against every reviewed event of the
split's clips, including events no candidate covers and clips with no events, so
proposal misses count as false negatives. Candidate coverage is reported separately.
Annotation-mode results are annotation-conditioned and are labelled as such.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pickup_putdown.layer1.track_b1.dataset import generate_inference_windows  # noqa: E402
from pickup_putdown.layer1.track_b1.dataset_dir import (  # noqa: E402
    load_dataset_dir,
    open_window_dataset,
)
from pickup_putdown.layer1.track_b1.inference import (  # noqa: E402
    InferenceConfig,
    decode_window_scores,
    evaluate_events,
    suppress_duplicate_events,
)
from pickup_putdown.layer1.track_b1.videomae_classifier import (  # noqa: E402
    _resolve_device,
    create_model,
    load_checkpoint,
)

logger = logging.getLogger("infer_track_b1")

DECODE_KEYS = ("pickup_threshold", "putdown_threshold", "smoothing_window",
               "same_type_merge_gap_s", "min_event_duration_s", "boundary_mode")


def load_into(model, checkpoint_path: Path, device: torch.device) -> dict:
    """Load a full-model or head-only checkpoint into ``model``; return its record.

    The head trainer only ever sees cached embeddings and saves ``head_state_dict``;
    the pixel-path trainer saves the whole model. Both load strictly.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "head_state_dict" in checkpoint:
        model.head.load_state_dict(checkpoint["head_state_dict"], strict=True)
        return {"kind": "head_only", "epoch": checkpoint.get("epoch"),
                "input_mode": checkpoint.get("input_mode")}
    load_checkpoint(checkpoint_path, model, device=str(device), strict=True)
    return {"kind": "full_model", "epoch": checkpoint.get("epoch"),
            "model_config": checkpoint.get("model_config")}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / ".local/track_b1_dataset")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, default=REPO_ROOT / ".local/source_videos")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / ".local/track_b1_cache")
    parser.add_argument("--frame-cache-dir", type=Path, default=REPO_ROOT / ".local/track_b1_frame_cache")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: <checkpoint dir>/../predictions_<input mode>")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--clips", nargs="+", default=None, help="restrict to these clip ids (smoke runs)")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/track_b1.yaml",
                        help="inference section supplies decode defaults")
    parser.add_argument("--model-name", default="MCG-NJU/videomae-base")
    parser.add_argument("--pickup-threshold", type=float, default=None)
    parser.add_argument("--putdown-threshold", type=float, default=None)
    parser.add_argument("--smoothing-window", type=int, default=None)
    parser.add_argument("--same-type-merge-gap-s", type=float, default=None)
    parser.add_argument("--min-event-duration-s", type=float, default=None)
    parser.add_argument("--boundary-mode", default=None, choices=["window_span", "window_centers"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.3, 0.5])
    return parser.parse_args(argv)


def inference_config(args: argparse.Namespace, window: dict) -> InferenceConfig:
    """Decode settings: config file ``inference`` section, then explicit CLI flags."""
    section = (yaml.safe_load(args.config.read_text()) or {}).get("inference") or {}
    values = {k: section[k] for k in DECODE_KEYS if k in section}
    values.update({k: getattr(args, k) for k in DECODE_KEYS if getattr(args, k) is not None})
    window_values = {k: window[k] for k in ("window_duration_s", "window_stride_s", "num_frames",
                                            "image_size", "crop_margin", "crop_scope",
                                            "resize_interpolation", "include_shelf_region")}
    return InferenceConfig(**window_values, **values, batch_size=args.batch_size)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    dataset_dir = load_dataset_dir(args.dataset_dir)
    mode = dataset_dir.input_mode
    config = inference_config(args, dataset_dir.metadata["window"])

    clips = dataset_dir.table("clips")
    split_clips = set(clips.loc[clips["split"] == args.split, "clip_id"])
    if args.clips:
        split_clips &= set(args.clips)
    candidates = dataset_dir.table("candidates")
    candidates = candidates[candidates["clip_id"].isin(split_clips)]

    windows = generate_inference_windows(candidates, config.to_window_config())
    window_table = pd.DataFrame([w.to_dict() for w in windows])
    if window_table.empty:
        logger.error("no inference windows in split %s", args.split)
        return 1
    window_table["sample_id"] = [f"inf_{i:06d}" for i in range(len(window_table))]
    window_table["label"] = -1  # inference windows carry no label
    dataset = open_window_dataset(
        dataset_dir, window_table, args.video_dir, cache_dir=args.cache_dir,
        frame_cache_dir=args.frame_cache_dir,
    )

    device = _resolve_device(args.device)
    model = create_model(model_name=args.model_name, num_classes=3, freeze_backbone=True, device=args.device)
    checkpoint_info = load_into(model, args.checkpoint, device)
    if checkpoint_info.get("input_mode") not in (None, mode):
        logger.warning("checkpoint trained in %s mode, evaluated on %s inputs",
                       checkpoint_info["input_mode"], mode)
    model.eval()

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        multiprocessing_context="spawn" if args.num_workers else None)
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            probabilities.append(torch.softmax(model(batch["pixel_values"].to(device)), dim=-1).cpu().numpy())
            if step % 25 == 0 or step == len(loader):
                logger.info("  %d/%d batches", step, len(loader))
    probs = np.concatenate(probabilities)
    scores = dataset.manifest[["clip_id", "candidate_id", "actor_id", "window_start_s", "window_end_s"]].copy()
    scores[["p_background", "p_pickup", "p_putdown"]] = probs

    decoded = decode_window_scores(scores, config)
    predictions, n_suppressed = suppress_duplicate_events(decoded)

    out = args.output_dir or args.checkpoint.parent.parent / f"predictions_{mode}"
    out.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out / f"window_scores_{args.split}.parquet", index=False)
    predictions.to_csv(out / f"predictions_{args.split}.csv", index=False)

    events = dataset_dir.table("events")
    ignores = dataset_dir.table("ignore_intervals")
    ground_truth = events[events["clip_id"].isin(split_clips)]
    durations = {c: float(d) for c, d in zip(clips["clip_id"], clips["duration_s"], strict=True) if c in split_clips}
    metrics = evaluate_events(predictions, ground_truth, ignores[ignores["clip_id"].isin(split_clips)],
                              durations, tuple(args.tiou))

    coverage = dataset_dir.table("candidate_coverage")
    coverage = coverage[coverage["clip_id"].isin(split_clips)]
    record = {
        "input_mode": mode,
        "annotation_conditioned": mode == "annotation",
        "split": args.split,
        "n_clips": len(split_clips),
        "n_clips_without_events": len(split_clips - set(ground_truth["clip_id"])),
        "n_ground_truth_event_rows": int(len(ground_truth)),
        "n_candidates": int(len(candidates)),
        "n_windows": int(len(scores)),
        "n_predictions": int(len(predictions)),
        "n_suppressed_duplicates": n_suppressed,
        "candidate_coverage": coverage["coverage"].value_counts().to_dict(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_info": checkpoint_info,
        "dataset_dir": str(dataset_dir.path),
        "split_registry": dataset_dir.metadata["split_registry"]["name"],
        "preprocessing": dataset_dir.metadata["preprocessing"],
        "decode": {k: v for k, v in asdict(config).items() if k in DECODE_KEYS},
        "event_metrics": metrics,
    }
    (out / f"metrics_{args.split}.json").write_text(json.dumps(record, indent=2, default=str))

    label = "ANNOTATION-CONDITIONED (GT candidates/crops)" if mode == "annotation" else "DEPLOYMENT-INPUT (pose candidates/crops)"
    print(f"\n[{label}] split={args.split} clips={len(split_clips)} "
          f"GT event rows={len(ground_truth)} predictions={len(predictions)} "
          f"(suppressed duplicates={n_suppressed})")
    print(f"candidate coverage of GT events: {record['candidate_coverage']}")
    for tiou in args.tiou:
        row = metrics[f"tiou@{tiou}"]
        print(f"  tIoU {tiou}: P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f} "
              f"(tp={row['tp']} fp={row['fp']} fn={row['fn']})")
    for event_type, row in metrics["per_type"].items():
        print(f"  {event_type:8s}: P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
