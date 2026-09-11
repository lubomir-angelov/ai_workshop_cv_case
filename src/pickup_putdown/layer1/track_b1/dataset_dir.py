"""Open a built Track B1 dataset directory in its declared input mode.

``scripts/build_track_b1_dataset.py`` writes one directory per input mode:

* ``annotation`` — candidates and crop boxes derived from CVAT tracks. Uses ground
  truth to decide where and when to look, so its scores are annotation-conditioned
  classifier diagnostics, not deployment performance.
* ``deployment`` — candidates and crop boxes from the pose pipeline, generated from
  video without ground truth. CVAT supervision only labels the windows.

Every script (cache, embeddings, training, inference, diagnostics, inspection) opens
the directory through here, so all of them read the same window configuration and
build model inputs the same way.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

from pickup_putdown.layer1.track_b1.dataset import (
    TrackB1Dataset,
    WindowConfig,
    load_shelf_regions,
)

logger = logging.getLogger(__name__)

INPUT_MODES = ("annotation", "deployment")
TRACKS_SUBDIR = {"annotation": "actor_tracks", "deployment": "pose_tracks"}


@dataclass
class DatasetDir:
    """A dataset directory plus its ``build_metadata.json``."""

    path: Path
    metadata: dict

    @property
    def input_mode(self) -> str:
        return self.metadata["input_mode"]

    @property
    def tracks_dir(self) -> Path:
        """Actor/pose tracks the crops come from (recorded explicitly when external)."""
        recorded = self.metadata.get("tracks_dir")
        return Path(recorded) if recorded else self.path / TRACKS_SUBDIR[self.input_mode]

    def window_config(self) -> WindowConfig:
        return WindowConfig(**self.metadata["window"])

    def table(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.path / f"{name}.parquet")

    def shelf_regions(self) -> dict[str, dict]:
        shelves = self.metadata.get("shelves_config")
        return load_shelf_regions(Path(shelves)) if shelves else {}


def load_dataset_dir(path: Path) -> DatasetDir:
    meta_path = Path(path) / "build_metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"{meta_path} missing; build with scripts/build_track_b1_dataset.py"
        )
    metadata = json.loads(meta_path.read_text())
    if metadata.get("input_mode") not in INPUT_MODES:
        raise ValueError(f"{meta_path}: input_mode must be one of {INPUT_MODES}")
    return DatasetDir(Path(path), metadata)


def open_window_dataset(
    dataset_dir: DatasetDir,
    manifest: pd.DataFrame,
    video_dir: Path,
    cache_dir: Path | None = None,
    frame_cache_dir: Path | None = None,
    from_video: bool = False,
) -> Dataset:
    """Dataset yielding exactly the pixels training uses for these windows.

    Annotation mode reads the per-candidate cache (``cache_dir``) after checking every
    needed entry is present and fresh; ``from_video=True`` decodes the source instead
    through the same preprocessing (used to verify the cache). Deployment mode decodes
    per window, optionally through the per-window frame cache.
    """
    config = dataset_dir.window_config()
    if dataset_dir.input_mode == "annotation" and not from_video:
        from pickup_putdown.layer1.track_b1.cache import CachedTrackB1Dataset, find_stale_entries

        if cache_dir is None:
            raise ValueError("annotation mode reads the candidate cache; pass cache_dir")
        candidates = dataset_dir.table("candidates")
        candidates = candidates[candidates["candidate_id"].isin(set(manifest["candidate_id"]))]
        stale = find_stale_entries(
            candidates, dataset_dir.tracks_dir, video_dir, cache_dir, config
        )
        if stale:
            raise ValueError(
                f"{len(stale)} candidates missing from or stale in {cache_dir} "
                f"(e.g. {stale[:3]}); run scripts/build_track_b1_cache.py"
            )
        return CachedTrackB1Dataset(manifest, cache_dir, config=config, require_complete=True)

    return TrackB1Dataset(
        window_manifest=manifest,
        video_dir=video_dir,
        pose_tracks_dir=dataset_dir.tracks_dir,
        shelf_regions=dataset_dir.shelf_regions(),
        config=config,
        clips_df=dataset_dir.table("clips"),
        cache_dir=frame_cache_dir,
    )
