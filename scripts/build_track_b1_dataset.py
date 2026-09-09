#!/usr/bin/env python3
"""Build the Track B1 window dataset from CVAT source-video annotations.

    python scripts/build_track_b1_dataset.py \
        --export-dir .local/cvat_exports/2026-09-09 \
        --video-dir  .local/source_videos \
        --output-dir .local/track_b1_dataset

Writes canonical annotation tables, the actor tracks that supply Track B1's crop
boxes, and the labelled window manifest that training and evaluation both read.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pickup_putdown.annotation.cvat_import import import_export_dir, write_result  # noqa: E402
from pickup_putdown.layer1.track_b1.annotation_windows import (  # noqa: E402
    NegativeSamplingConfig,
    build_candidate_table,
    summarize_manifest,
    write_actor_tracks,
)
from pickup_putdown.layer1.track_b1.dataset import WindowConfig, build_window_manifest  # noqa: E402

logger = logging.getLogger("build_track_b1_dataset")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=REPO_ROOT / ".local/cvat_exports/2026-09-09")
    parser.add_argument("--video-dir", type=Path, default=REPO_ROOT / ".local/source_videos")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / ".local/track_b1_dataset")

    parser.add_argument("--window-duration-s", type=float, default=2.5)
    parser.add_argument("--window-stride-s", type=float, default=0.5)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--crop-margin", type=float, default=0.15)
    parser.add_argument(
        "--context-pad-s", type=float, default=3.0,
        help="Seconds of held-box padding around each event; the source of hard negatives",
    )
    parser.add_argument(
        "--accepted-only", action="store_true",
        help="Drop intervals still marked draft in CVAT",
    )
    parser.add_argument("--val-days", type=int, default=1)
    parser.add_argument("--test-days", type=int, default=1)
    parser.add_argument("--negative-windows-per-clip", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    video_dir = args.video_dir if args.video_dir.exists() else None
    if video_dir is None:
        logger.warning(
            "video dir %s not found; frame rates will be estimated from clip stems",
            args.video_dir,
        )

    logger.info("Importing CVAT exports from %s", args.export_dir)
    result = import_export_dir(
        export_dir=args.export_dir,
        video_dir=video_dir,
        accepted_only=args.accepted_only,
        context_pad_s=args.context_pad_s,
        val_days=args.val_days,
        test_days=args.test_days,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_result(result, args.output_dir)
    logger.info(
        "Imported %d events, %d ignore intervals across %d clips",
        len(result.events), len(result.ignore_intervals), len(result.clips),
    )

    candidates, tracks = build_candidate_table(
        actor_tracks=result.actor_tracks,
        clips_df=result.clips,
        negative_config=NegativeSamplingConfig(
            windows_per_clip=args.negative_windows_per_clip, seed=args.seed
        ),
    )
    write_actor_tracks(tracks, args.output_dir / "actor_tracks")
    candidates.to_parquet(args.output_dir / "candidates.parquet", index=False)

    window_config = WindowConfig(
        window_duration_s=args.window_duration_s,
        window_stride_s=args.window_stride_s,
        num_frames=args.num_frames,
        crop_margin=args.crop_margin,
    )
    manifest = build_window_manifest(
        candidates_df=candidates,
        events_df=result.events,
        ignore_intervals_df=result.ignore_intervals,
        clips_df=result.clips,
        config=window_config,
    )
    manifest.to_parquet(args.output_dir / "window_manifest.parquet", index=False)

    summary = summarize_manifest(manifest)
    print("\nWindow manifest by split:")
    print(summary.to_string() if not summary.empty else "  (empty)")

    print("\nEvents by split:")
    events_by_split = (
        result.events.merge(result.clips[["clip_id", "split"]], on="clip_id")
        .pivot_table(index="split", columns="type", values="event_id", aggfunc="count", fill_value=0)
    )
    print(events_by_split.to_string())

    # Leakage guard: a clip must not appear under two splits, or validation is fiction.
    per_clip_splits = manifest.groupby("clip_id")["split"].nunique()
    straddling = per_clip_splits[per_clip_splits > 1]
    if len(straddling):
        logger.error("clips present in multiple splits: %s", list(straddling.index))
        return 1

    (args.output_dir / "build_metadata.json").write_text(
        json.dumps(
            {
                "export_dir": str(args.export_dir),
                "video_dir": str(video_dir) if video_dir else None,
                "n_events": int(len(result.events)),
                "n_ignore_intervals": int(len(result.ignore_intervals)),
                "n_clips": int(len(result.clips)),
                "n_candidates": int(len(candidates)),
                "n_windows": int(len(manifest)),
                "window": vars(window_config) if hasattr(window_config, "__dict__") else {},
                "context_pad_s": args.context_pad_s,
                "accepted_only": args.accepted_only,
                "seed": args.seed,
                "label_counts": manifest["label_name"].value_counts().to_dict(),
            },
            indent=2,
            default=str,
        )
    )
    logger.info("Wrote dataset to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
