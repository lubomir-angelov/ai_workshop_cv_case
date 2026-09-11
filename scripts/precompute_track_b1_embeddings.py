#!/usr/bin/env python3
"""Precompute frozen-VideoMAE embeddings for every window of a Track B1 dataset.

    python scripts/precompute_track_b1_embeddings.py --dataset-dir .local/track_b1_dataset

With the backbone frozen only the classification head trains, yet each epoch would
re-run all 86M backbone parameters over every window. The backbone output never
changes, so it is computed once. Pixels come from the same dataset path training
uses (per-candidate cache in annotation mode, per-window frames in deployment mode).

Writes embeddings.npy + windows.parquet (window keys, no labels) + metadata.json
(encoder weight hash, preprocessing, input mode). Valid only while the backbone is
frozen; fine-tuning uses the pixel path in train_track_b1.py.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pickup_putdown.layer1.track_b1.dataset_dir import (  # noqa: E402
    load_dataset_dir,
    open_window_dataset,
)
from pickup_putdown.layer1.track_b1.embeddings import save_embeddings  # noqa: E402
from pickup_putdown.layer1.track_b1.videomae_classifier import (  # noqa: E402
    VideoMAEClassifier,
    _resolve_device,
)

logger = logging.getLogger("precompute_track_b1_embeddings")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / ".local/track_b1_dataset")
    parser.add_argument("--video-dir", type=Path, default=REPO_ROOT / ".local/source_videos")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / ".local/track_b1_cache",
                        help="annotation mode: per-candidate crop cache")
    parser.add_argument("--frame-cache-dir", type=Path, default=REPO_ROOT / ".local/track_b1_frame_cache",
                        help="deployment mode: per-window crop cache")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: <dataset-dir>/embeddings")
    parser.add_argument("--model-name", default="MCG-NJU/videomae-base")
    parser.add_argument("--split", nargs="+", default=None, help="restrict to these splits")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    dataset_dir = load_dataset_dir(args.dataset_dir)
    manifest = dataset_dir.table("window_manifest")
    if args.split:
        manifest = manifest[manifest["split"].isin(args.split)].reset_index(drop=True)
    dataset = open_window_dataset(
        dataset_dir, manifest, args.video_dir, cache_dir=args.cache_dir,
        frame_cache_dir=args.frame_cache_dir,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        multiprocessing_context="spawn" if args.num_workers else None,
    )

    device = _resolve_device(args.device)
    model = VideoMAEClassifier(model_name=args.model_name, num_classes=3, freeze_backbone=True).to(device)
    model.eval()

    features: list[np.ndarray] = []
    with torch.inference_mode():
        for index, batch in enumerate(loader, start=1):
            hidden = model.encoder(pixel_values=batch["pixel_values"].to(device)).last_hidden_state
            features.append(model._pool_features(hidden).float().cpu().numpy())
            if index % 25 == 0 or index == len(loader):
                logger.info("  %d/%d batches", index, len(loader))

    embeddings = np.concatenate(features, axis=0)
    output_dir = args.output_dir or dataset_dir.path / "embeddings"
    save_embeddings(
        output_dir,
        embeddings,
        dataset.manifest,
        {
            "model_name": args.model_name,
            "encoder_weights_sha256": model.encoder_weights_sha256,
            "preprocessing": dataset_dir.metadata["preprocessing"],
            "input_mode": dataset_dir.input_mode,
            "dataset_dir": str(dataset_dir.path),
            "hidden_dim": int(embeddings.shape[1]),
            "n_windows": int(embeddings.shape[0]),
            "pooling": "mean over sequence",
        },
    )
    print(f"\nwrote {embeddings.shape} embeddings to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
