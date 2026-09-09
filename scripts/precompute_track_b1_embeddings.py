#!/usr/bin/env python3
"""Precompute frozen-VideoMAE embeddings for every Track B1 window.

With the backbone frozen only the classification head trains — a few thousand
parameters — yet each epoch re-runs all 86M backbone parameters over every window,
which costs about nine minutes an epoch here. The backbone output never changes, so
it is computed once and stored.

    python scripts/precompute_track_b1_embeddings.py

Writes ``embeddings.npy`` ([n_windows, hidden_dim] float32) aligned row-for-row with
``manifest.parquet`` in the output directory. Head training then reads these and runs
an epoch in under a second, which is what makes threshold tuning and honest early
stopping affordable.

This is only valid while the backbone is frozen. Fine-tuning backbone blocks requires
the full pixel path in ``train_track_b1.py``.
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
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pickup_putdown.layer1.track_b1.cache import CachedTrackB1Dataset  # noqa: E402
from pickup_putdown.layer1.track_b1.videomae_classifier import (  # noqa: E402
    VideoMAEClassifier,
    _resolve_device,
)

logger = logging.getLogger("precompute_track_b1_embeddings")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / ".local/track_b1_dataset")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / ".local/track_b1_cache")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / ".local/track_b1_embeddings")
    parser.add_argument("--model-name", default="MCG-NJU/videomae-base")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    manifest = pd.read_parquet(args.dataset_dir / "window_manifest.parquet")
    dataset = CachedTrackB1Dataset(manifest, args.cache_dir, num_frames=args.num_frames)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
    )

    device = _resolve_device(args.device)
    model = VideoMAEClassifier(
        model_name=args.model_name, num_classes=3, freeze_backbone=True
    ).to(device)
    model.eval()

    features: list[np.ndarray] = []
    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            pixel_values = batch["pixel_values"].to(device)
            hidden = model.encoder(pixel_values=pixel_values).last_hidden_state
            features.append(model._pool_features(hidden).float().cpu().numpy())
            if index % 25 == 0:
                logger.info("  %d/%d batches", index, len(loader))

    embeddings = np.concatenate(features, axis=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "embeddings.npy", embeddings)

    # The manifest is written alongside because rows are aligned positionally; a
    # separately rebuilt manifest could order differently and silently mismatch.
    dataset.manifest.to_parquet(args.output_dir / "manifest.parquet", index=False)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "num_frames": args.num_frames,
                "hidden_dim": int(embeddings.shape[1]),
                "n_windows": int(embeddings.shape[0]),
                "cache_dir": str(args.cache_dir),
                "pooling": "mean over sequence",
            },
            indent=2,
        )
    )

    print(f"\nwrote {embeddings.shape} embeddings to {args.output_dir}")
    print(dataset.manifest.groupby("split")["label_name"].value_counts().unstack(fill_value=0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
