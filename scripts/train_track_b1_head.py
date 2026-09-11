#!/usr/bin/env python3
"""Train the Track B1 classification head on precomputed VideoMAE embeddings.

    python scripts/train_track_b1_head.py --epochs 200

Equivalent to the frozen-backbone path in ``train_track_b1.py``, but reading cached
embeddings instead of re-encoding pixels every epoch. The head is the same module, so
the checkpoint written here drops straight into the full model for inference.

Gate B (tiny overfit) runs on a class-balanced subset first: with cached features it
costs a second, and it still catches a scrambled label mapping or a misaligned
manifest, which is what the gate is for.
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
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pickup_putdown.layer1.track_b1.dataset import LABEL_NAMES  # noqa: E402
from pickup_putdown.layer1.track_b1.dataset_dir import load_dataset_dir  # noqa: E402
from pickup_putdown.layer1.track_b1.embeddings import load_embeddings_for  # noqa: E402
from pickup_putdown.layer1.track_b1.videomae_classifier import (  # noqa: E402
    ClassificationHead,
    file_sha256,
    resolve_weights_path,
)

logger = logging.getLogger("train_track_b1_head")


def macro_f1(true: np.ndarray, predicted: np.ndarray, num_classes: int = 3) -> tuple[float, dict]:
    """Macro F1 plus the per-class breakdown, computed without sklearn."""
    per_class: dict[str, float] = {}
    scores = []
    for label in range(num_classes):
        tp = int(((predicted == label) & (true == label)).sum())
        fp = int(((predicted == label) & (true != label)).sum())
        fn = int(((predicted != label) & (true == label)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[LABEL_NAMES[label]] = f1
        scores.append(f1)
    return float(np.mean(scores)), per_class


def gate_b(head: nn.Module, features: torch.Tensor, labels: torch.Tensor, device) -> bool:
    """Overfit a small class-balanced subset; a head that cannot is a wiring bug."""
    picked: list[int] = []
    for label in sorted(set(labels.tolist())):
        picked.extend(np.where(labels.numpy() == label)[0][:5].tolist())

    x = features[picked].to(device)
    y = labels[picked].to(device)

    probe = ClassificationHead(features.shape[1], 3, dropout=0.0).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    probe.train()
    loss = torch.tensor(float("inf"))
    for _ in range(500):
        optimizer.zero_grad()
        loss = criterion(probe(x), y)
        loss.backward()
        optimizer.step()
        if loss.item() < 0.05:
            break

    accuracy = (probe(x).argmax(dim=-1) == y).float().mean().item()
    passed = loss.item() < 0.05 and accuracy > 0.95
    logger.info(
        "GATE B %s: %d samples, labels=%s, loss=%.4f acc=%.3f",
        "PASSED" if passed else "FAILED",
        len(picked), torch.bincount(y, minlength=3).tolist(), loss.item(), accuracy,
    )
    return passed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / ".local/track_b1_dataset",
                        help="labels/splits come from this dataset's window manifest")
    parser.add_argument("--embeddings-dir", type=Path, default=None,
                        help="default: <dataset-dir>/embeddings")
    parser.add_argument("--model-name", default="MCG-NJU/videomae-base",
                        help="pretrained encoder the embeddings must come from")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / ".local/track_b1_run")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-gate-b", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_dir = load_dataset_dir(args.dataset_dir)
    embeddings_dir = args.embeddings_dir or dataset_dir.path / "embeddings"
    manifest = dataset_dir.table("window_manifest")
    manifest = manifest[manifest["split"].isin(["train", "val"])].reset_index(drop=True)
    encoder_sha = file_sha256(resolve_weights_path(args.model_name))
    embeddings, embedding_metadata = load_embeddings_for(
        manifest, embeddings_dir, dataset_dir.metadata["preprocessing"], encoder_sha
    )
    logger.info("input mode %s: %d windows with cached frozen-encoder features",
                dataset_dir.input_mode, len(manifest))

    device = torch.device(args.device)
    features = torch.from_numpy(embeddings).float()
    labels = torch.from_numpy(manifest["label"].to_numpy()).long()
    weights = torch.from_numpy(manifest["sample_weight"].to_numpy()).float()

    train_mask = (manifest["split"] == "train").to_numpy()
    val_mask = (manifest["split"] == "val").to_numpy()

    x_train, y_train, w_train = features[train_mask].to(device), labels[train_mask].to(device), weights[train_mask].to(device)
    x_val, y_val = features[val_mask].to(device), labels[val_mask].to(device)
    logger.info("train=%d val=%d hidden_dim=%d", len(x_train), len(x_val), features.shape[1])

    if not args.skip_gate_b and not gate_b(nn.Identity(), features[train_mask], labels[train_mask], device):
        logger.error("Gate B failed; not proceeding to full training")
        return 1

    head = ClassificationHead(features.shape[1], 3, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    counts = torch.bincount(y_train, minlength=3).float().clamp(min=1)
    class_weights = (counts.sum() / (3 * counts)).to(device)
    logger.info("class weights: %s", [round(w, 3) for w in class_weights.tolist()])
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")

    best_f1, best_state, best_epoch, since_improved = -1.0, None, 0, 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        head.train()
        order = torch.randperm(len(x_train), device=device)
        epoch_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            optimizer.zero_grad()
            loss = (criterion(head(x_train[batch]), y_train[batch]) * w_train[batch]).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        scheduler.step()

        head.eval()
        with torch.no_grad():
            val_predicted = head(x_val).argmax(dim=-1)
        f1, per_class = macro_f1(y_val.cpu().numpy(), val_predicted.cpu().numpy())
        history.append({"epoch": epoch, "train_loss": epoch_loss / len(order), "val_f1_macro": f1, **per_class})

        if f1 > best_f1 + 1e-4:
            best_f1, best_epoch, since_improved = f1, epoch, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            since_improved += 1

        if epoch % 25 == 0 or epoch == 1:
            logger.info(
                "epoch %d: loss=%.4f val_f1=%.4f (pickup=%.3f putdown=%.3f)",
                epoch, epoch_loss / len(order), f1, per_class["pickup"], per_class["putdown"],
            )
        if since_improved >= args.patience:
            logger.info("early stopping at epoch %d (best %d)", epoch, best_epoch)
            break

    final_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        val_probs = torch.softmax(head(x_val), dim=-1).cpu().numpy()
    final_f1, final_per_class = macro_f1(y_val.cpu().numpy(), val_probs.argmax(axis=1))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    provenance = {
        "hidden_dim": int(features.shape[1]),
        "dropout": args.dropout,
        "metadata": embedding_metadata,
        "input_mode": dataset_dir.input_mode,
        "dataset_dir": str(dataset_dir.path),
        "encoder_weights_sha256": encoder_sha,
    }
    torch.save(
        {"head_state_dict": best_state, "epoch": best_epoch, "val_f1_macro": final_f1, **provenance},
        checkpoint_dir / "head_best.pt",
    )
    torch.save(
        {"head_state_dict": final_state, "epoch": history[-1]["epoch"],
         "val_f1_macro": history[-1]["val_f1_macro"], **provenance},
        checkpoint_dir / "head_final.pt",
    )
    pd.DataFrame(history).to_csv(args.output_dir / "head_training_history.csv", index=False)
    np.save(args.output_dir / "val_probs.npy", val_probs)
    # Keyed per-window export: provenance survives any reordering of the manifest.
    val_rows = manifest[val_mask].reset_index(drop=True)
    val_rows = val_rows.assign(pred_id=val_probs.argmax(axis=1),
                               p_background=val_probs[:, 0], p_pickup=val_probs[:, 1],
                               p_putdown=val_probs[:, 2])
    val_rows.to_csv(args.output_dir / "val_window_predictions.csv", index=False)
    (args.output_dir / "head_results.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "final_epoch": history[-1]["epoch"],
                "val_f1_macro": final_f1,
                "val_f1_per_class": final_per_class,
                "input_mode": dataset_dir.input_mode,
                "n_train": int(len(x_train)),
                "n_val": int(len(x_val)),
                "class_weights": class_weights.tolist(),
                **{k: str(v) for k, v in vars(args).items()},
            },
            indent=2,
        )
    )

    print(f"\nbest epoch {best_epoch}  val macro F1 {final_f1:.4f}")
    for name, score in final_per_class.items():
        print(f"  {name:11s} F1 {score:.4f}")
    print(f"\ncheckpoint: {checkpoint_dir / 'head_best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
