"""Frozen-encoder embedding cache for Track B1 head training.

With the encoder frozen its pooled output per window never changes, so it is computed
once. The cache stores features keyed by window identity plus what produced them
(exact pretrained weights, preprocessing, input mode) and never labels: labels are
joined from the current window manifest at training time, so relabelling does not
require recomputing features, and stale features cannot be silently reused.

Embeddings are only valid for the frozen pretrained encoder; any run that fine-tunes
encoder blocks must use the pixel path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_KEY = ["clip_id", "candidate_id", "actor_id", "window_start_s", "window_end_s"]


def save_embeddings(
    output_dir: Path,
    embeddings: np.ndarray,
    windows: pd.DataFrame,
    metadata: dict,
) -> None:
    """Write features, their window keys (row-aligned) and provenance."""
    if len(embeddings) != len(windows):
        raise ValueError(f"{len(embeddings)} embeddings for {len(windows)} windows")
    keys = windows[WINDOW_KEY].reset_index(drop=True)
    if keys.duplicated().any():
        raise ValueError("duplicate window keys; embeddings would be ambiguous")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embeddings)
    keys.to_parquet(output_dir / "windows.parquet", index=False)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))


def load_embeddings_for(
    manifest: pd.DataFrame,
    embeddings_dir: Path,
    preprocessing: dict,
    encoder_weights_sha256: str,
    unfreeze_last_n_blocks: int = 0,
) -> tuple[np.ndarray, dict]:
    """Features aligned row-for-row with ``manifest``, after checking they apply.

    Raises when the encoder would be fine-tuned, when the features came from other
    weights or preprocessing, or when any manifest window has no feature.
    """
    if unfreeze_last_n_blocks:
        raise ValueError(
            "cached embeddings are frozen-encoder features; fine-tuning "
            f"{unfreeze_last_n_blocks} blocks must use the pixel path (train_track_b1.py)"
        )
    metadata = json.loads((embeddings_dir / "metadata.json").read_text())
    if metadata.get("encoder_weights_sha256") != encoder_weights_sha256:
        raise ValueError(
            f"embeddings in {embeddings_dir} came from encoder weights "
            f"{str(metadata.get('encoder_weights_sha256'))[:12]}, not {encoder_weights_sha256[:12]}"
        )
    if metadata.get("preprocessing") != preprocessing:
        raise ValueError(
            f"embeddings in {embeddings_dir} used preprocessing {metadata.get('preprocessing')}, "
            f"dataset uses {preprocessing}"
        )
    embeddings = np.load(embeddings_dir / "embeddings.npy")
    keys = pd.read_parquet(embeddings_dir / "windows.parquet")
    keys["_row"] = np.arange(len(keys))
    joined = manifest[WINDOW_KEY].merge(keys, on=WINDOW_KEY, how="left", validate="many_to_one")
    missing = joined["_row"].isna()
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} manifest windows have no embedding in {embeddings_dir}; "
            "recompute with scripts/precompute_track_b1_embeddings.py"
        )
    return embeddings[joined["_row"].astype(int).to_numpy()], metadata
