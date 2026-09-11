#!/usr/bin/env python3
"""Precompute the Track B1 per-candidate crop cache.

Decodes each candidate's span once, crops it to the candidate's actor-conditioned
region, resizes to the model input size and stores it as uint8. See
``pickup_putdown.layer1.track_b1.cache`` for why training cannot read the 4K sources
directly.

    python scripts/build_track_b1_cache.py --workers 6

Safe to re-run: candidates already cached at the current version are skipped.
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pickup_putdown.layer1.track_b1.cache import build_candidate_cache  # noqa: E402

logger = logging.getLogger("build_track_b1_cache")


def _cache_one(payload: dict) -> tuple[str, bool]:
    """Worker entry point — rebuilt from plain data so it pickles cheaply."""
    import logging as _logging

    _logging.basicConfig(level=_logging.WARNING)
    candidate = pd.Series(payload["candidate"])
    actor_track = pd.DataFrame(payload["actor_track"])
    entry = build_candidate_cache(
        candidate=candidate,
        actor_track=actor_track,
        video_path=Path(payload["video_path"]),
        output_dir=Path(payload["output_dir"]),
        image_size=tuple(payload["image_size"]),
        crop_margin=payload["crop_margin"],
        overwrite=payload["overwrite"],
    )
    return candidate["candidate_id"], entry is not None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / ".local/track_b1_dataset")
    parser.add_argument("--video-dir", type=Path, default=REPO_ROOT / ".local/source_videos")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / ".local/track_b1_cache")
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--crop-margin", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    candidates = pd.read_parquet(args.dataset_dir / "candidates.parquet")
    if args.limit:
        candidates = candidates.head(args.limit)

    tracks_dir = args.dataset_dir / "actor_tracks"
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    payloads: list[dict] = []
    for _, candidate in candidates.iterrows():
        clip_id = candidate["clip_id"]
        track_path = tracks_dir / f"{clip_id}.parquet"
        if not track_path.exists():
            logger.error("no actor track for %s", clip_id)
            continue
        track = pd.read_parquet(track_path)
        track = track[track["actor_id"] == candidate["actor_id"]]
        if track.empty:
            logger.error("no boxes for %s / %s", clip_id, candidate["actor_id"])
            continue

        payloads.append(
            {
                "candidate": candidate.to_dict(),
                "actor_track": track.to_dict(orient="list"),
                "video_path": str(args.video_dir / f"{clip_id}.mp4"),
                "output_dir": str(args.cache_dir),
                "image_size": list(args.image_size),
                "crop_margin": args.crop_margin,
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

    size_mb = sum(p.stat().st_size for p in args.cache_dir.glob("*.npy")) / 1e6
    print(f"\ncached {succeeded}/{len(payloads)} candidates, {size_mb:.0f} MB in {args.cache_dir}")
    if failed:
        print(f"failed: {len(failed)}")
        for candidate_id in failed[:10]:
            print(f"  {candidate_id}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
