"""Per-candidate crop cache for Track B1.

The source clips are 4K H.264. Decoding a window by seeking to each of its 16 frames
costs about 1.3 s per seek, because every seek decodes forward from the preceding
keyframe at full resolution; sequential decoding of the same frames costs under 10 ms
each. Training reads every window many times over many epochs, so the seek-per-frame
path in ``dataset.decode_window_frames`` is roughly four orders of magnitude too slow
to train on.

This module decodes each candidate exactly once, sequentially, crops to the candidate's
actor-conditioned region and resizes to the model's input size, then stores the result
as a uint8 array. Windows are afterwards served by indexing into that array.

The crop is computed once per candidate rather than once per window. Beyond making the
cache possible, this is the better conditioning: within a candidate every window then
shares a frame of reference, so the model cannot read the label off a changing crop
size and only the motion inside the box distinguishes the classes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from pickup_putdown.layer1.track_b1.dataset import (
    INTERPOLATIONS,
    WindowConfig,
    _apply_margin_and_clamp,
    _file_fingerprint,
    _union_actor_boxes,
    normalize_frames,
    preprocessing_spec,
)

logger = logging.getLogger(__name__)

# v3: entries carry a fingerprint (source video, actor-track boxes, candidate span,
# preprocessing) and are rebuilt when it no longer matches.
CACHE_VERSION = 3

BOX_COLUMNS = [
    "timestamp_s",
    "person_bbox_x1",
    "person_bbox_y1",
    "person_bbox_x2",
    "person_bbox_y2",
]


def cache_preprocessing(config: WindowConfig) -> dict:
    """Preprocessing that determines cached pixels. ``num_frames`` is applied at read time."""
    if config.crop_scope != "candidate":
        raise ValueError(
            "the per-candidate cache crops once per candidate; it needs "
            f"crop_scope='candidate', got {config.crop_scope!r}"
        )
    return {k: v for k, v in preprocessing_spec(config).items() if k != "num_frames"}


def candidate_fingerprint(
    candidate: pd.Series,
    actor_track: pd.DataFrame,
    video_path: Path,
    config: WindowConfig,
) -> dict:
    """Everything a cached candidate's pixels depend on. Labels are not part of it."""
    boxes = actor_track.reindex(columns=BOX_COLUMNS).sort_values("timestamp_s")
    track_hash = hashlib.sha256(
        pd.util.hash_pandas_object(boxes, index=False).to_numpy().tobytes()
    ).hexdigest()
    return {
        "cache_version": CACHE_VERSION,
        "video": _file_fingerprint(video_path),
        "track_sha256": track_hash,
        "span": [float(candidate["window_start_s"]), float(candidate["window_end_s"])],
        "preprocessing": cache_preprocessing(config),
    }


@dataclass
class CachedCandidate:
    """Where one candidate's frames live and what time they cover."""

    candidate_id: str
    clip_id: str
    actor_id: str
    start_frame: int
    n_frames: int
    fps: float
    crop_box: tuple[int, int, int, int]
    array_path: Path
    fingerprint: dict | None = None

    def frame_position(self, frame_index: int) -> int:
        """Position of a source frame inside the cached array, clamped to its range."""
        return int(np.clip(frame_index - self.start_frame, 0, self.n_frames - 1))


def candidate_crop_box(
    actor_track: pd.DataFrame,
    start_s: float,
    end_s: float,
    margin: float,
    frame_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Actor-conditioned crop for a whole candidate, as (x1, y1, x2, y2)."""
    box = _union_actor_boxes(actor_track, start_s, end_s)
    if box is None:
        logger.warning("no boxes in candidate span; falling back to the full frame")
        return 0, 0, frame_size[0], frame_size[1]
    return _apply_margin_and_clamp(box, margin, frame_size)


def build_candidate_cache(
    candidate: pd.Series,
    actor_track: pd.DataFrame,
    video_path: Path,
    output_dir: Path,
    config: WindowConfig,
    overwrite: bool = False,
) -> CachedCandidate | None:
    """Decode, crop and cache one candidate's frames. Returns None if it cannot be read.

    An existing entry is reused only when its fingerprint (video, track boxes, span,
    preprocessing) matches; otherwise it is rebuilt.
    """
    candidate_id = candidate["candidate_id"]
    array_path = output_dir / f"{candidate_id}.npy"
    meta_path = output_dir / f"{candidate_id}.json"
    expected = candidate_fingerprint(candidate, actor_track, video_path, config)

    if not overwrite and array_path.exists() and meta_path.exists():
        entry = _entry_from_meta(json.loads(meta_path.read_text()), array_path)
        if entry is not None and entry.fingerprint == expected:
            return entry
        logger.info("rebuilding stale cache entry %s", candidate_id)

    if not video_path.exists():
        logger.error("video not found: %s", video_path)
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("cannot open video: %s", video_path)
        return None

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0:
            logger.error("invalid fps for %s", video_path)
            return None

        start_s = float(candidate["window_start_s"])
        end_s = float(candidate["window_end_s"])
        crop_box = candidate_crop_box(
            actor_track, start_s, end_s, config.crop_margin, (frame_w, frame_h)
        )
        x1, y1, x2, y2 = crop_box
        if x2 <= x1 or y2 <= y1:
            logger.error("degenerate crop for %s: %s", candidate_id, crop_box)
            return None

        start_frame = max(0, int(np.floor(start_s * fps)))
        end_frame = int(np.ceil(end_s * fps))
        n_frames = max(1, end_frame - start_frame + 1)
        image_size = tuple(config.image_size)
        interpolation = INTERPOLATIONS[config.resize_interpolation]

        # One seek, then a sequential run: the only access pattern this codec is fast at.
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = np.zeros((n_frames, image_size[1], image_size[0], 3), dtype=np.uint8)
        last: np.ndarray | None = None
        decoded = 0

        for position in range(n_frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                # Past the end of the clip, or a damaged run: hold the previous frame
                # rather than injecting black, which would read as a scene cut.
                if last is None:
                    break
                frames[position] = last
                continue
            tile = cv2.resize(frame[y1:y2, x1:x2], image_size, interpolation=interpolation)
            frames[position] = tile
            last = tile
            decoded += 1

        if decoded == 0:
            logger.error("decoded no frames for %s", candidate_id)
            return None
        if decoded < n_frames:
            logger.warning("%s: decoded %d/%d frames, tail held", candidate_id, decoded, n_frames)

        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(array_path, frames)
        meta = {
            "cache_version": CACHE_VERSION,
            "candidate_id": candidate_id,
            "clip_id": candidate["clip_id"],
            "actor_id": candidate["actor_id"],
            "start_frame": start_frame,
            "n_frames": n_frames,
            "n_decoded": decoded,
            "fps": fps,
            "crop_box": list(crop_box),
            "fingerprint": expected,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        return _entry_from_meta(meta, array_path)
    finally:
        cap.release()


def _entry_from_meta(meta: dict, array_path: Path) -> CachedCandidate | None:
    if meta.get("cache_version") != CACHE_VERSION:
        return None
    return CachedCandidate(
        candidate_id=meta["candidate_id"],
        clip_id=meta["clip_id"],
        actor_id=meta["actor_id"],
        start_frame=meta["start_frame"],
        n_frames=meta["n_frames"],
        fps=meta["fps"],
        crop_box=tuple(meta["crop_box"]),
        array_path=array_path,
        fingerprint=meta.get("fingerprint"),
    )


def load_cache_index(cache_dir: Path) -> dict[str, CachedCandidate]:
    """Read every current-version candidate cache entry written under ``cache_dir``."""
    index: dict[str, CachedCandidate] = {}
    for meta_path in sorted(cache_dir.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        array_path = cache_dir / f"{meta.get('candidate_id')}.npy"
        entry = _entry_from_meta(meta, array_path)
        if entry is None or not array_path.exists():
            continue
        index[entry.candidate_id] = entry
    return index


def find_stale_entries(
    candidates: pd.DataFrame,
    tracks_dir: Path,
    video_dir: Path,
    cache_dir: Path,
    config: WindowConfig,
) -> list[str]:
    """Candidate ids whose cache entry is missing or no longer matches its inputs.

    Cheap (hashes track rows and stats videos, never decodes), so every consumer of
    the cache can check it before trusting cached pixels.
    """
    index = load_cache_index(cache_dir)
    tracks: dict[str, pd.DataFrame] = {}
    stale: list[str] = []
    for _, candidate in candidates.iterrows():
        clip_id = candidate["clip_id"]
        if clip_id not in tracks:
            tracks[clip_id] = pd.read_parquet(tracks_dir / f"{clip_id}.parquet")
        track = tracks[clip_id]
        track = track[track["actor_id"] == candidate["actor_id"]]
        entry = index.get(candidate["candidate_id"])
        expected = candidate_fingerprint(candidate, track, video_dir / f"{clip_id}.mp4", config)
        if entry is None or entry.fingerprint != expected:
            stale.append(candidate["candidate_id"])
    return stale


# ============================================================
# CACHE-BACKED DATASET
# ============================================================


class CachedTrackB1Dataset(Dataset):
    """Serve Track B1 windows out of the per-candidate crop cache.

    Emits the same keys as :class:`~pickup_putdown.layer1.track_b1.dataset.TrackB1Dataset`
    so training, inference and inspection code is unchanged; only the frame source
    differs. Frames are memory-mapped, so workers share the page cache instead of each
    holding a copy.
    """

    def __init__(
        self,
        window_manifest: pd.DataFrame,
        cache_dir: Path,
        num_frames: int = 16,
        transform: Callable | None = None,
        config: WindowConfig | None = None,
        require_complete: bool = False,
    ) -> None:
        """``config`` (when given) must match the preprocessing the cache was built
        with; ``require_complete`` raises instead of dropping uncached windows, so a
        run cannot silently train or evaluate on a different example set."""
        self.cache_dir = Path(cache_dir)
        self.num_frames = config.num_frames if config is not None else num_frames
        self.transform = transform
        self.index = load_cache_index(self.cache_dir)

        if config is not None:
            wanted = cache_preprocessing(config)
            mismatched = [
                cid
                for cid, entry in self.index.items()
                if (entry.fingerprint or {}).get("preprocessing") != wanted
            ]
            if mismatched:
                raise ValueError(
                    f"{len(mismatched)} cache entries in {self.cache_dir} were built with "
                    f"different preprocessing than {wanted} (e.g. {mismatched[:3]}); "
                    "rebuild with scripts/build_track_b1_cache.py"
                )

        manifest = window_manifest.reset_index(drop=True)
        known = manifest["candidate_id"].isin(self.index)
        if not known.all():
            missing = sorted(set(manifest.loc[~known, "candidate_id"]))
            if require_complete:
                raise ValueError(
                    f"{int((~known).sum())} windows from {len(missing)} candidates are "
                    f"not cached (e.g. {missing[:3]})"
                )
            logger.warning(
                "dropping %d windows from %d uncached candidates (e.g. %s)",
                int((~known).sum()),
                len(missing),
                missing[:3],
            )
        self.manifest = manifest[known].reset_index(drop=True)

        self._arrays: dict[str, np.ndarray] = {}
        logger.info(
            "CachedTrackB1Dataset: %d windows over %d cached candidates",
            len(self.manifest),
            self.manifest["candidate_id"].nunique(),
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def _array(self, candidate_id: str) -> np.ndarray:
        array = self._arrays.get(candidate_id)
        if array is None:
            array = np.load(self.index[candidate_id].array_path, mmap_mode="r")
            self._arrays[candidate_id] = array
        return array

    def _sample_positions(self, entry: CachedCandidate, start_s: float, end_s: float) -> list[int]:
        """Uniformly spaced positions in the cached array, in chronological order."""
        start_frame = int(start_s * entry.fps)
        end_frame = int(end_s * entry.fps)
        if end_frame <= start_frame:
            end_frame = start_frame + 1
        if self.num_frames == 1:
            return [entry.frame_position(start_frame)]
        wanted = np.linspace(start_frame, end_frame - 1, self.num_frames)
        return [entry.frame_position(int(f)) for f in wanted]

    def __getitem__(self, idx: int) -> dict:
        row = self.manifest.iloc[idx]
        entry = self.index[row["candidate_id"]]

        positions = self._sample_positions(
            entry, float(row["window_start_s"]), float(row["window_end_s"])
        )
        frames = np.asarray(self._array(row["candidate_id"])[positions])

        pixel_values = normalize_frames(frames)
        if self.transform is not None:
            pixel_values = self.transform(pixel_values)

        return {
            "pixel_values": pixel_values,
            "label": int(row["label"]),
            "sample_id": row["sample_id"],
            "clip_id": row["clip_id"],
            "actor_id": row["actor_id"],
            "candidate_id": row["candidate_id"],
            "window_start_s": float(row["window_start_s"]),
            "window_end_s": float(row["window_end_s"]),
            "sample_weight": float(row.get("sample_weight", 1.0)),
        }
