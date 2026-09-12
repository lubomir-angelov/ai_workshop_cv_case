#!/usr/bin/env python3
"""Build a Track B1 window dataset from CVAT source-video annotations.

CVAT is the authoritative supervision in both input modes. The modes differ only in
where candidates (when/who to look at) and crop boxes (where to look) come from:

    # annotation-conditioned: candidates + crops from the CVAT tracks (no pose needed)
    python scripts/build_track_b1_dataset.py --input-mode annotation \
        --output-dir .local/track_b1_dataset

    # deployment-input: candidates + crops from the pose pipeline; CVAT only labels
    python scripts/build_track_b1_dataset.py --input-mode deployment \
        --pose-data-dir .local/track_b1_data --output-dir .local/track_b1_dataset_deploy

Writes canonical CVAT tables (events keep every reviewed event, including ones no
candidate covers), candidates, actor/pose tracks, the labelled window manifest and
build_metadata.json. Deployment mode also writes the CVAT->pose actor association
and per-event candidate coverage.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pickup_putdown.annotation.cvat_import import (  # noqa: E402
    import_export_dir,
    load_split_registry,
    normalize_clip_id,
    write_result,
)
from pickup_putdown.layer1.track_b1.actor_association import (  # noqa: E402
    AssociationConfig,
    associate_events,
    candidate_coverage,
    events_in_pose_identity,
)
from pickup_putdown.layer1.track_b1.annotation_windows import (  # noqa: E402
    NegativeSamplingConfig,
    build_candidate_table,
    summarize_manifest,
    write_actor_tracks,
)
from pickup_putdown.layer1.track_b1.dataset import (  # noqa: E402
    WindowConfig,
    build_window_manifest,
    preprocessing_spec,
)

logger = logging.getLogger("build_track_b1_dataset")

WINDOW_KEYS = (
    "window_duration_s",
    "window_stride_s",
    "num_frames",
    "image_size",
    "crop_margin",
    "crop_scope",
    "resize_interpolation",
    "include_shelf_region",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-mode", choices=["annotation", "deployment"], required=True)
    parser.add_argument(
        "--export-dir", type=Path, default=REPO_ROOT / ".local/annotations/cvat/2026-09-09/raw"
    )
    parser.add_argument("--video-dir", type=Path, default=REPO_ROOT / ".local/source_videos")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/track_b1.yaml",
        help="window + input_modes sections supply the window config",
    )
    parser.add_argument("--splits", type=Path, default=REPO_ROOT / "configs/track_b1_splits.yaml")
    parser.add_argument(
        "--split-registry",
        default=None,
        help="registry name in --splits (default: the file's 'default')",
    )
    parser.add_argument(
        "--pose-data-dir",
        type=Path,
        default=REPO_ROOT / ".local/track_b1_data",
        help="deployment mode: candidates.parquet + pose_tracks/ from "
        "scripts/prepare_track_b1_data.py",
    )
    parser.add_argument("--shelves", type=Path, default=REPO_ROOT / "configs/shelves.yaml")
    for key in ("window_duration_s", "window_stride_s", "crop_margin"):
        parser.add_argument(f"--{key.replace('_', '-')}", type=float, default=None)
    parser.add_argument(
        "--context-pad-s",
        type=float,
        default=None,
        help="annotation mode: held-box padding around each event",
    )
    parser.add_argument(
        "--accepted-only", action="store_true", help="Drop intervals still marked draft in CVAT"
    )
    parser.add_argument("--negative-windows-per-clip", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def window_config_from(config: dict, mode: str, args: argparse.Namespace) -> WindowConfig:
    """Config file window section + per-mode preprocessing + explicit CLI overrides."""
    values = {k: v for k, v in (config.get("window") or {}).items() if k in WINDOW_KEYS}
    values.update((config.get("input_modes") or {}).get(mode) or {})
    for key in ("window_duration_s", "window_stride_s", "crop_margin"):
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    return WindowConfig(**{k: v for k, v in values.items() if k in WINDOW_KEYS})


def load_pose_inputs(
    pose_dir: Path, clip_ids: set[str]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], bool]:
    """Pose-derived candidates and tracks with clip ids validated against CVAT.

    Also returns whether any track file's clip_id column needed normalising.
    """
    candidates = pd.read_parquet(pose_dir / "candidates.parquet")
    candidates["clip_id"] = candidates["clip_id"].map(normalize_clip_id)
    unknown = sorted(set(candidates["clip_id"]) - clip_ids)
    if unknown:
        raise ValueError(f"pose candidates for clips absent from the CVAT export: {unknown[:5]}")
    duplicated = candidates["candidate_id"].duplicated()
    if duplicated.any():
        raise ValueError(
            f"duplicate candidate ids: {candidates.loc[duplicated, 'candidate_id'].head(3).tolist()}"
        )

    tracks: dict[str, pd.DataFrame] = {}
    normalised = False
    for clip_id in sorted(clip_ids):
        path = pose_dir / "pose_tracks" / f"{clip_id}.parquet"
        if not path.is_file():
            if clip_id in set(candidates["clip_id"]):
                raise FileNotFoundError(f"{path} missing but the clip has pose candidates")
            logger.warning("%s: no pose track; its events can only be missed", clip_id)
            continue
        track = pd.read_parquet(path)
        if "clip_id" in track.columns:
            raw = set(track["clip_id"])
            track["clip_id"] = track["clip_id"].map(normalize_clip_id)
            if set(track["clip_id"]) != {clip_id}:
                raise ValueError(f"{path}: clip_id column {set(track['clip_id'])} != {clip_id}")
            normalised |= raw != {clip_id}
        tracks[clip_id] = track
    return candidates, tracks, normalised


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    config = yaml.safe_load(args.config.read_text()) or {}
    annotation_cfg = config.get("annotation") or {}
    window_config = window_config_from(config, args.input_mode, args)
    context_pad_s = (
        args.context_pad_s
        if args.context_pad_s is not None
        else annotation_cfg.get("context_pad_s", 3.0)
    )
    negatives_per_clip = (
        args.negative_windows_per_clip
        if args.negative_windows_per_clip is not None
        else annotation_cfg.get("negative_windows_per_clip", 12)
    )

    splits_doc = yaml.safe_load(args.splits.read_text())
    registry = args.split_registry or splits_doc["default"]
    day_splits = load_split_registry(args.splits, registry)

    video_dir = args.video_dir if args.video_dir.exists() else None
    if video_dir is None:
        logger.warning("video dir %s not found; fps estimated from clip stems", args.video_dir)

    logger.info("Importing CVAT exports from %s (split registry %s)", args.export_dir, registry)
    result = import_export_dir(
        export_dir=args.export_dir,
        video_dir=video_dir,
        accepted_only=args.accepted_only,
        context_pad_s=context_pad_s,
        day_splits=day_splits,
    )
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_result(result, out)
    logger.info(
        "Imported %d event rows, %d ignore intervals across %d clips",
        len(result.events),
        len(result.ignore_intervals),
        len(result.clips),
    )

    extra: dict = {}
    if args.input_mode == "annotation":
        candidates, tracks = build_candidate_table(
            actor_tracks=result.actor_tracks,
            clips_df=result.clips,
            negative_config=NegativeSamplingConfig(
                windows_per_clip=negatives_per_clip, seed=args.seed
            ),
        )
        write_actor_tracks(tracks, out / "actor_tracks")
        labelling_events = result.events  # CVAT track ids on both sides
    else:
        candidates, pose_tracks, normalised = load_pose_inputs(
            args.pose_data_dir, set(result.clips["clip_id"])
        )
        # Crops read the pose files by path, and the per-window frame cache keys on
        # their identity: reference them in place unless their clip ids had to be fixed.
        tracks_dir = args.pose_data_dir / "pose_tracks"
        if normalised:
            tracks_dir = out / "pose_tracks"
            write_actor_tracks(pose_tracks, tracks_dir)
        # CVAT boxes are kept for diagnostics only (crop evidence); never model inputs here.
        write_actor_tracks(result.actor_tracks, out / "cvat_tracks")
        association_config = AssociationConfig()
        association = associate_events(
            result.events, result.actor_tracks, pose_tracks, association_config
        )
        association.to_parquet(out / "actor_association.parquet", index=False)
        labelling_events = events_in_pose_identity(result.events, association)
        labelling_events.to_parquet(out / "events_pose_identity.parquet", index=False)
        extra = {
            "pose_data_dir": str(args.pose_data_dir),
            "tracks_dir": str(tracks_dir.resolve()),
            "association_config": asdict(association_config),
            "association_status": association["association_status"].value_counts().to_dict(),
        }
    candidates.to_parquet(out / "candidates.parquet", index=False)

    manifest = build_window_manifest(
        candidates_df=candidates,
        events_df=labelling_events,
        ignore_intervals_df=result.ignore_intervals,
        clips_df=result.clips,
        config=window_config,
    )
    manifest.to_parquet(out / "window_manifest.parquet", index=False)

    # Leakage guard: a clip must not appear under two splits, or validation is fiction.
    straddling = manifest.groupby("clip_id")["split"].nunique()
    if (straddling > 1).any():
        logger.error(
            "clips present in multiple splits: %s", list(straddling[straddling > 1].index)
        )
        return 1

    split_of = result.clips.set_index("clip_id")["split"]
    coverage = candidate_coverage(
        labelling_events.assign(split=labelling_events["clip_id"].map(split_of)),
        candidates,
        manifest,
    )
    coverage["split"] = coverage["clip_id"].map(split_of)
    coverage.to_parquet(out / "candidate_coverage.parquet", index=False)
    coverage_summary = coverage.pivot_table(
        index="split", columns="coverage", values="event_id", aggfunc="count", fill_value=0
    )

    summary = summarize_manifest(manifest)
    print(f"\n[{args.input_mode}] window manifest by split:")
    print(summary.to_string() if not summary.empty else "  (empty)")
    print(f"\n[{args.input_mode}] reviewed event rows by candidate coverage:")
    print(coverage_summary.to_string())

    (out / "build_metadata.json").write_text(
        json.dumps(
            {
                "input_mode": args.input_mode,
                "export_dir": str(args.export_dir),
                "video_dir": str(video_dir) if video_dir else None,
                "split_registry": {"file": str(args.splits), "name": registry, "days": day_splits},
                "window": asdict(window_config),
                "preprocessing": preprocessing_spec(window_config),
                "shelves_config": str(args.shelves) if args.input_mode == "deployment" else None,
                "context_pad_s": context_pad_s,
                "negative_windows_per_clip": negatives_per_clip,
                "accepted_only": args.accepted_only,
                "seed": args.seed,
                "n_event_rows": int(len(result.events)),
                "n_ignore_intervals": int(len(result.ignore_intervals)),
                "n_clips": int(len(result.clips)),
                "n_candidates": int(len(candidates)),
                "n_windows": int(len(manifest)),
                "label_counts": manifest["label_name"].value_counts().to_dict(),
                "coverage": coverage["coverage"].value_counts().to_dict(),
                **extra,
            },
            indent=2,
            default=str,
        )
    )
    logger.info("Wrote %s dataset to %s", args.input_mode, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
