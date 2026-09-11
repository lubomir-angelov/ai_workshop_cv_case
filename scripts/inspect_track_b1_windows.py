#!/usr/bin/env python3
"""Gate A — render sampled-frame grids from the Track B1 loader before training.

Task 12 asks for the data loader to be inspected visually, and specifically for
temporal order to be verified by eye. This renders exactly what the model receives:
frames pulled through ``TrackB1Dataset``, after actor-conditioned cropping and
resizing, laid out left-to-right in sampling order with their index drawn on.

    python scripts/inspect_track_b1_windows.py --per-class 4

Grids land in ``--output-dir``, one PNG per sampled window, named by label so a
mislabelled or time-reversed window is obvious at a glance. A ``contact_sheet.png``
stacks them for a single-image review.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pickup_putdown.layer1.track_b1.dataset_dir import (  # noqa: E402
    load_dataset_dir,
    open_window_dataset,
)

logger = logging.getLogger("inspect_track_b1_windows")

# ImageNet statistics, matching normalize_frames() in dataset.py.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize(pixel_values) -> np.ndarray:
    """[T, C, H, W] normalized tensor -> [T, H, W, C] uint8 RGB, for display."""
    frames = pixel_values.numpy().transpose(0, 2, 3, 1)
    frames = frames * STD + MEAN
    return (np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8)


def render_grid(frames: np.ndarray, title: str, columns: int = 8) -> np.ndarray:
    """Lay frames out in sampling order, index-stamped, with a title bar."""
    n_frames, height, width = frames.shape[:3]
    rows = int(np.ceil(n_frames / columns))
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)

    for index, frame in enumerate(frames):
        row, col = divmod(index, columns)
        tile = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
        # The index is the whole point of the grid: it makes time order checkable.
        cv2.putText(tile, str(index), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(tile, str(index), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        canvas[row * height : (row + 1) * height, col * width : (col + 1) * width] = tile

    bar = np.full((36, canvas.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(bar, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return np.vstack([bar, canvas])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / ".local/track_b1_dataset")
    parser.add_argument("--video-dir", type=Path, default=REPO_ROOT / ".local/source_videos")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / ".local/track_b1_inspection")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / ".local/track_b1_cache")
    parser.add_argument("--frame-cache-dir", type=Path, default=None,
                        help="deployment mode: read/write this per-window cache (default: decode)")
    parser.add_argument("--per-class", type=int, default=4, help="Windows to render per label")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare-source", action="store_true",
                        help="annotation mode: also decode each window from the source video "
                             "and fail if it differs from the cached input training uses")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    dataset_dir = load_dataset_dir(args.dataset_dir)
    manifest = dataset_dir.table("window_manifest")

    manifest = manifest[manifest["split"] == args.split]
    if manifest.empty:
        logger.error("no windows in split %s", args.split)
        return 1

    rng = np.random.default_rng(args.seed)
    picked = pd.concat(
        [
            group.sample(min(args.per_class, len(group)), random_state=int(rng.integers(1 << 31)))
            for _, group in manifest.groupby("label_name")
        ]
    ).reset_index(drop=True)

    # The same dataset path training uses, so the grids show the real model input.
    dataset = open_window_dataset(
        dataset_dir, picked, args.video_dir, cache_dir=args.cache_dir,
        frame_cache_dir=args.frame_cache_dir,
    )
    source = None
    if args.compare_source and dataset_dir.input_mode == "annotation":
        source = open_window_dataset(dataset_dir, picked, args.video_dir, from_video=True)
    worst_difference = 0.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grids: list[np.ndarray] = []

    for index in range(len(dataset)):
        sample = dataset[index]
        row = picked.iloc[index]
        frames = denormalize(sample["pixel_values"])
        if source is not None:
            decoded = denormalize(source[index]["pixel_values"]).astype(int)
            difference = float(np.abs(decoded - frames.astype(int)).max())
            worst_difference = max(worst_difference, difference)
            logger.info("%s: max |cache - source decode| = %.0f", row["sample_id"], difference)

        # A window whose frames are all identical means the decode fell back to a
        # repeated frame; that is silent in training but obvious here.
        spread = float(np.std(frames.astype(np.float32).mean(axis=(1, 2, 3))))

        title = (
            f"{row['label_name']}  {row['clip_id'][:34]}  {row['actor_id']}  "
            f"[{row['window_start_s']:.2f}-{row['window_end_s']:.2f}s]  frame-mean sd={spread:.2f}"
        )
        grid = render_grid(frames, title)
        name = f"{row['label_name']}__{row['sample_id']}__{row['clip_id'][:28]}.png"
        cv2.imwrite(str(args.output_dir / name), grid)
        grids.append(grid)
        logger.info("rendered %s (frame-mean sd=%.2f)", name, spread)

    width = max(g.shape[1] for g in grids)
    padded = [
        np.pad(g, ((0, 8), (0, width - g.shape[1]), (0, 0)), constant_values=0) for g in grids
    ]
    cv2.imwrite(str(args.output_dir / "contact_sheet.png"), np.vstack(padded))

    print(f"\nWrote {len(grids)} grids + contact_sheet.png to {args.output_dir} "
          f"({dataset_dir.input_mode} inputs)")
    if source is not None:
        print(f"max |cached input - fresh source decode| over all windows: {worst_difference:.0f}")
        if worst_difference > 0:
            logger.error("cached inputs differ from the source decode; rebuild the cache")
            return 1
    print("Check: frames advance left-to-right, the actor stays in crop, "
          "and the label matches what the hands do.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
