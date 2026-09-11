#!/usr/bin/env python3
"""Precompute Track B1 crops for a built dataset directory.

    python scripts/build_track_b1_cache.py --dataset-dir .local/track_b1_dataset --workers 6
    python scripts/build_track_b1_cache.py --dataset-dir .local/track_b1_dataset_deploy --workers 8

Annotation mode: decodes each candidate's span once, crops it to the candidate's
CVAT-box region and stores it (``layer1.track_b1.cache``). Deployment mode: fills the
per-window frame cache used by ``TrackB1Dataset``. Both use the preprocessing in the
dataset's build_metadata.json and store pixels only, never labels.

Safe to re-run: valid entries are reused, entries whose inputs or preprocessing
changed are rebuilt.
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pickup_putdown.layer1.track_b1.cache import build_candidate_cache  # noqa: E402
from pickup_putdown.layer1.track_b1.dataset import WindowConfig  # noqa: E402
from pickup_putdown.layer1.track_b1.dataset_dir import (  # noqa: E402
    load_dataset_dir,
    open_window_dataset,
)

logger = logging.getLogger("build_track_b1_cache")


def _cache_one(payload: dict) -> tuple[str, bool]:
    """Worker entry point — rebuilt from plain data so it pickles cheaply."""
    import logging as _logging

    _logging.basicConfig(level=_logging.WARNING)
    candidate = pd.Series(payload["candidate"])
    entry = build_candidate_cache(
        candidate=candidate,
        actor_track=pd.DataFrame(payload["actor_track"]),
        video_path=Path(payload["video_path"]),
        output_dir=Path(payload["output_dir"]),
        config=WindowConfig(**payload["window"]),
        overwrite=payload["overwrite"],
    )
    return candidate["candidate_id"], entry is not None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / ".local/track_b1_dataset")
    parser.add_argument("--video-dir", type=Path, default=REPO_ROOT / ".local/source_videos")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="default: .local/track_b1_cache (annotation) or "
        ".local/track_b1_frame_cache (deployment)",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default=None, help="only cache this split")
    return parser.parse_args(argv)


def cache_candidates(dataset_dir, args, cache_dir: Path) -> int:
    candidates = dataset_dir.table("candidates")
    if args.split:
        clips = dataset_dir.table("clips")
        candidates = candidates[
            candidates["clip_id"].isin(clips.loc[clips["split"] == args.split, "clip_id"])
        ]
    if args.limit:
        candidates = candidates.head(args.limit)
    window = dataset_dir.metadata["window"]

    payloads: list[dict] = []
    for _, candidate in candidates.iterrows():
        clip_id = candidate["clip_id"]
        track = pd.read_parquet(dataset_dir.tracks_dir / f"{clip_id}.parquet")
        track = track[track["actor_id"] == candidate["actor_id"]]
        if track.empty:
            logger.error("no boxes for %s / %s", clip_id, candidate["actor_id"])
            return 1
        payloads.append(
            {
                "candidate": candidate.to_dict(),
                "actor_track": track.to_dict(orient="list"),
                "video_path": str(args.video_dir / f"{clip_id}.mp4"),
                "output_dir": str(cache_dir),
                "window": window,
                "overwrite": args.overwrite,
            }
        )

    logger.info("Caching %d candidates with %d workers", len(payloads), args.workers)
    succeeded, failed = 0, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_cache_one, payload): payload for payload in payloads}
        for done, future in enumerate(as_completed(futures), start=1):
            candidate_id, ok = future.result()
            if ok:
                succeeded += 1
            else:
                failed.append(candidate_id)
            if done % 25 == 0 or done == len(payloads):
                logger.info("  %d/%d cached", done, len(payloads))

    size_mb = sum(p.stat().st_size for p in cache_dir.glob("*.npy")) / 1e6
    print(f"\ncached {succeeded}/{len(payloads)} candidates, {size_mb:.0f} MB in {cache_dir}")
    for candidate_id in failed[:10]:
        print(f"  failed: {candidate_id}")
    return 1 if failed else 0


def cache_windows(dataset_dir, args, cache_dir: Path) -> int:
    manifest = dataset_dir.table("window_manifest")
    if args.split:
        manifest = manifest[manifest["split"] == args.split]
    if args.limit:
        manifest = manifest.head(args.limit)
    dataset = open_window_dataset(dataset_dir, manifest, args.video_dir, frame_cache_dir=cache_dir)
    loader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=args.workers,
        multiprocessing_context="spawn" if args.workers else None,
    )
    hits = 0
    for step, batch in enumerate(loader, start=1):
        hits += int(batch["cache_hit"].sum())
        if step % 50 == 0 or step == len(loader):
            logger.info("  %d/%d batches (%d windows already cached)", step, len(loader), hits)
    print(f"\n{len(dataset)} windows in {cache_dir} ({hits} were already cached)")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    dataset_dir = load_dataset_dir(args.dataset_dir)
    if dataset_dir.input_mode == "annotation":
        cache_dir = args.cache_dir or REPO_ROOT / ".local/track_b1_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_candidates(dataset_dir, args, cache_dir)
    cache_dir = args.cache_dir or REPO_ROOT / ".local/track_b1_frame_cache"
    return cache_windows(dataset_dir, args, cache_dir)


if __name__ == "__main__":
    sys.exit(main())
