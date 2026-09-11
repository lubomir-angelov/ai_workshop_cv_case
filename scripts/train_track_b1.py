#!/usr/bin/env python3
"""Train the Track B1 VideoMAE window classifier through the full pixel path.

    # fine-tune the last two encoder blocks, head warm-started from the frozen probe
    python scripts/train_track_b1.py --dataset-dir .local/track_b1_dataset \
        --unfreeze-last-n-blocks 2 --backbone-lr 5e-5 --learning-rate 1e-3 \
        --init-head-from .local/track_b1_run/checkpoints/head_best.pt \
        --output-dir .local/track_b1_finetune

Reads a dataset directory built by ``build_track_b1_dataset.py`` in either input
mode; pixels come from the same path as every other script (per-candidate cache in
annotation mode, per-window frames in deployment mode). Cached embeddings are never
used here, so fine-tuned encoder blocks always see pixels. Runs Gate B (tiny overfit,
on a copy of the model) unless told otherwise and selects on validation macro F1.

The test split is not touched here.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pickup_putdown.layer1.track_b1.dataset_dir import (  # noqa: E402
    load_dataset_dir,
    open_window_dataset,
)
from pickup_putdown.layer1.track_b1.train import TrainConfig, train  # noqa: E402
from pickup_putdown.layer1.track_b1.videomae_classifier import (  # noqa: E402
    _resolve_device,
    create_model,
)

logger = logging.getLogger("train_track_b1")


def class_weights_from(manifest: pd.DataFrame, num_classes: int = 3) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1.

    Background outnumbers the event classes roughly four to one here, and putdown is
    scarcer than pickup; without reweighting the loss is dominated by background and
    the model can score well while never predicting an event.
    """
    counts = manifest["label"].value_counts().reindex(range(num_classes), fill_value=0)
    counts = counts.replace(0, 1)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor((weights / weights.mean()).values, dtype=torch.float32)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / ".local/track_b1_dataset")
    parser.add_argument("--video-dir", type=Path, default=REPO_ROOT / ".local/source_videos")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / ".local/track_b1_cache")
    parser.add_argument("--frame-cache-dir", type=Path, default=REPO_ROOT / ".local/track_b1_frame_cache")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / ".local/track_b1_run")

    parser.add_argument("--model-name", default="MCG-NJU/videomae-base")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--unfreeze-last-n-blocks", type=int, default=0)
    parser.add_argument(
        "--backbone-lr", type=float, default=None,
        help="Learning rate for unfrozen encoder blocks (default: same as the head). "
             "A pretrained backbone needs a much smaller step than an untrained head.",
    )
    parser.add_argument(
        "--init-head-from", type=Path, default=None,
        help="Warm-start the head from a head_best.pt produced by train_track_b1_head.py, "
             "so fine-tuning starts from a head that already works rather than from noise.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-tiny-overfit", action="store_true")
    parser.add_argument(
        "--balanced-sampler", action="store_true",
        help="Oversample event windows instead of relying on class weights alone",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    dataset_dir = load_dataset_dir(args.dataset_dir)
    manifest = dataset_dir.table("window_manifest")
    train_manifest = manifest[manifest["split"] == "train"]
    val_manifest = manifest[manifest["split"] == "val"]
    if train_manifest.empty or val_manifest.empty:
        logger.error("train or val split is empty")
        return 1

    def open_split(split_manifest: pd.DataFrame):
        return open_window_dataset(
            dataset_dir, split_manifest, args.video_dir, cache_dir=args.cache_dir,
            frame_cache_dir=args.frame_cache_dir,
        )

    train_dataset = open_split(train_manifest)
    val_dataset = open_split(val_manifest)
    logger.info("input mode %s, preprocessing %s", dataset_dir.input_mode,
                dataset_dir.metadata["preprocessing"])

    sampler, shuffle = None, True
    if args.balanced_sampler:
        counts = train_dataset.manifest["label"].value_counts()
        per_sample = train_dataset.manifest["label"].map(lambda label: 1.0 / counts[label])
        sampler = WeightedRandomSampler(
            weights=per_sample.tolist(), num_samples=len(train_dataset), replacement=True
        )
        shuffle = False

    workers = {"num_workers": args.num_workers}
    if args.num_workers:
        workers.update(multiprocessing_context="spawn", persistent_workers=True)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
        pin_memory=False, drop_last=False, **workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=False, **workers,
    )

    device = _resolve_device(args.device)
    model = create_model(
        model_name=args.model_name,
        num_classes=3,
        dropout=args.dropout,
        freeze_backbone=True,
        unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
        device=args.device,
    )

    if args.init_head_from is not None:
        # Fine-tuning from a random head fights itself: the head needs large steps the
        # pretrained backbone cannot tolerate. Starting from a head that already works
        # means the run measures what unfreezing adds, not how fast a head can train.
        checkpoint = torch.load(args.init_head_from, map_location=device, weights_only=False)
        model.head.load_state_dict(checkpoint["head_state_dict"])
        head_mode = checkpoint.get("input_mode")
        if head_mode != dataset_dir.input_mode:
            logger.warning("head was trained in input mode %s, fine-tuning in %s",
                           head_mode, dataset_dir.input_mode)
        if checkpoint.get("encoder_weights_sha256") not in (None, model.encoder_weights_sha256):
            raise ValueError("warm-start head was trained on a different pretrained encoder")
        logger.info(
            "warm-started head from %s (val macro F1 %.4f)",
            args.init_head_from, checkpoint.get("val_f1_macro", float("nan")),
        )

    config = TrainConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        checkpoint_dir=args.output_dir / "checkpoints",
        model_name=args.model_name,
        freeze_backbone=True,
        unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
        dropout=args.dropout,
        device=args.device,
        backbone_lr=args.backbone_lr,
    )

    weights = class_weights_from(train_dataset.manifest).to(device)
    logger.info("class weights (background, pickup, putdown): %s", weights.tolist())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        class_weights=weights,
        skip_tiny_overfit=args.skip_tiny_overfit,
    )

    (args.output_dir / "training_results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                **{k: str(v) for k, v in vars(args).items()},
                "class_weights": weights.tolist(),
                "input_mode": dataset_dir.input_mode,
                "preprocessing": dataset_dir.metadata["preprocessing"],
                "split_registry": dataset_dir.metadata["split_registry"],
                "encoder_weights_sha256": model.encoder_weights_sha256,
                "n_train_windows": len(train_dataset),
                "n_val_windows": len(val_dataset),
                "device": str(device),
            },
            indent=2,
        )
    )

    best = results.get("best_metrics") or {}
    print("\nBest validation metrics:")
    for key in ("f1_macro", "accuracy", "precision_macro", "recall_macro"):
        if key in best:
            print(f"  {key:16s} {best[key]:.4f}")
    print(f"  per-class F1     {best.get('f1_per_class')}")
    print(f"\ncheckpoint: {results.get('best_checkpoint_path')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
