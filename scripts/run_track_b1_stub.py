#!/usr/bin/env python3
"""Run Track B1 inference with frozen VideoMAE + random classification head.

Validates: window extraction -> model forward -> temporal smoothing -> peak detection -> merging.
Predictions will be random (untrained head), but the full pipeline wiring is proven.
"""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import numpy as np
import os
import pandas as pd
import torch

os.environ["HF_HUB_OFFLINE"] = "1"
import torch.nn as nn

from transformers import VideoMAEModel

from pickup_putdown.layer1.track_b1.inference import (
    InferenceConfig,
    infer_candidate,
    save_predictions,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent  # project root
LOCAL = BASE / ".local"
CANDIDATES_PARQUET = LOCAL / "task5_acceptance" / "preview_20260622_100316" / "candidates.parquet"
POSES_PARQUET = LOCAL / "task5_acceptance" / "preview_20260622_100316" / "tracks_pose.parquet"
VIDEO_PATH = LOCAL / "triage_acceptance" / "videos" / "person_clear.mp4"
OUTPUT_DIR = LOCAL / "track_b1_output"
VIDEO_MAE_MODEL = "MCG-NJU/videomae-small-finetuned-ssv2"


class FrozenVideoMAEClassifier(nn.Module):
    """Minimal wrapper: frozen VideoMAE encoder + random linear head."""

    def __init__(self, encoder: VideoMAEModel, num_classes: int = 3):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        hidden_dim = encoder.config.hidden_size

        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hidden = self.encoder(pixel_values=pixel_values).last_hidden_state
        pooled = hidden.mean(dim=1)  # [B, hidden]
        pooled = self.norm(pooled)
        return self.classifier(pooled)


def load_data():
    candidates_df = pd.read_parquet(CANDIDATES_PARQUET)
    pose_df = pd.read_parquet(POSES_PARQUET)
    return candidates_df, pose_df


def create_frozen_model(device: torch.device):
    logger.info(f"Loading VideoMAE encoder from {VIDEO_MAE_MODEL}")
    encoder = VideoMAEModel.from_pretrained(VIDEO_MAE_MODEL)
    encoder = encoder.to(device)
    encoder.eval()

    model = FrozenVideoMAEClassifier(encoder, num_classes=3).to(device)
    model.eval()

    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Backbone frozen. Trainable params: {trainable}")
    return model


def run():
    candidates_df, pose_df = load_data()
    device = torch.device("cuda")  # GPU full (llama-server), use CPU
    logger.info(f"Device: {device}")
    logger.info(f"Candidates: {len(candidates_df)}")
    logger.info(f"Pose observations: {len(pose_df)}")
    logger.info(f"Video: {VIDEO_PATH} (exists={VIDEO_PATH.exists()})")

    model = create_frozen_model(device)
    config = InferenceConfig(
        pickup_threshold=0.5,
        putdown_threshold=0.5,
        smoothing_window=3,
    )

    all_predictions = []
    for idx, candidate in candidates_df.iterrows():
        actor_id = candidate["actor_id"]
        pose_track_df = pose_df[pose_df["actor_id"] == actor_id].copy()

        events = infer_candidate(
            model=model,
            candidate=candidate,
            video_path=VIDEO_PATH,
            pose_track_df=pose_track_df,
            shelf_region=None,
            config=config,
            device=device,
        )

        for event in events:
            all_predictions.append(event.to_dict())

        if (idx + 1) % 5 == 0:
            logger.info(f"Processed {idx + 1}/{len(candidates_df)} candidates")

    predictions_df = pd.DataFrame(all_predictions) if all_predictions else pd.DataFrame(
        columns=["pred_id", "clip_id", "type", "t_start", "t_end", "score", "model"]
    )

    output_path = OUTPUT_DIR / "predictions.csv"
    save_predictions(predictions_df, output_path)

    return predictions_df


def main() -> int:
    predictions_df = run()
    s = predictions_df.shape

    print(f"\n{'='*60}")
    print(f"Track B1 Inference (frozen VideoMAE + random head)")
    print(f"{'='*60}")
    print(f"Candidates processed: {s[0]}")
    print(f"Predictions:          {s[1]}")
    print(f"{'='*60}")

    pred_file = OUTPUT_DIR / "predictions.csv"
    if pred_file.exists():
        print(f"\npredictions.csv ({pred_file.stat().st_size} bytes):")
        with open(pred_file) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            print(f"  Rows: {len(rows)}")
            if rows:
                for row in rows[:5]:
                    print(f"  {row}")

    print(f"\nOutput files in {OUTPUT_DIR}:")
    if OUTPUT_DIR.exists():
        for f in sorted(OUTPUT_DIR.iterdir()):
            print(f"  {f.name} ({f.stat().st_size:,} bytes)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
