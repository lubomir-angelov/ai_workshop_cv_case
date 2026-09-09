"""Track B1 VideoMAE dataset: actor-conditioned window extraction for event classification.

This module handles:
- Sliding window generation over candidate intervals
- Label assignment based on window center vs event intervals
- Actor-conditioned spatial cropping (actor bbox + shelf region)
- On-demand video decoding with uniform frame sampling
- PyTorch Dataset for training/inference

The output tensor shape is [T, C, H, W] where:
- T = num_frames (default 16)
- C = 3 (RGB channels)
- H, W = image_size (default 224x224)
"""

from __future__ import annotations

import logging
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

if TYPE_CHECKING:
    from typing import Callable

logger = logging.getLogger(__name__)


# ============================================================
# LABEL CONSTANTS
# ============================================================

LABEL_BACKGROUND: int = 0
LABEL_PICKUP: int = 1
LABEL_PUTDOWN: int = 2

LABEL_NAMES: dict[int, str] = {
    LABEL_BACKGROUND: "background",
    LABEL_PICKUP: "pickup",
    LABEL_PUTDOWN: "putdown",
}

LABEL_IDS: dict[str, int] = {v: k for k, v in LABEL_NAMES.items()}


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass
class WindowConfig:
    """Configuration for window extraction and frame sampling."""

    # Window parameters
    window_duration_s: float = 2.5
    window_stride_s: float = 0.5
    min_window_duration_s: float = 1.0  # Skip candidates shorter than this

    # Frame sampling
    num_frames: int = 16
    image_size: tuple[int, int] = (224, 224)

    # Actor-conditioned crop
    crop_margin: float = 0.15  # 15% margin around actor+shelf union

    # Label weights by confidence
    weight_high: float = 1.0
    weight_med: float = 1.0
    weight_low: float = 0.5

    # Validation
    min_event_overlap_ratio: float = 0.0  # Minimum overlap for center-based assignment


@dataclass
class WindowSample:
    """A single training/inference window sample."""

    sample_id: str
    clip_id: str
    candidate_id: str
    actor_id: str
    region_id: Optional[str]
    window_start_s: float
    window_end_s: float
    label: int
    label_name: str
    event_id: Optional[str] = None
    event_confidence: Optional[str] = None
    sample_weight: float = 1.0
    split: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            "sample_id": self.sample_id,
            "clip_id": self.clip_id,
            "candidate_id": self.candidate_id,
            "actor_id": self.actor_id,
            "region_id": self.region_id,
            "window_start_s": self.window_start_s,
            "window_end_s": self.window_end_s,
            "label": self.label,
            "label_name": self.label_name,
            "event_id": self.event_id,
            "event_confidence": self.event_confidence,
            "sample_weight": self.sample_weight,
            "split": self.split,
        }


# ============================================================
# WINDOW MANIFEST BUILDER
# ============================================================


def build_window_manifest(
    candidates_df: pd.DataFrame,
    events_df: pd.DataFrame,
    ignore_intervals_df: pd.DataFrame,
    clips_df: pd.DataFrame,
    config: WindowConfig,
    split: Optional[str] = None,
) -> pd.DataFrame:
    """Generate all training/validation/test windows with labels.

    Args:
        candidates_df: DataFrame with candidate intervals (from Task 5).
        events_df: DataFrame with ground-truth events (from Task 7).
        ignore_intervals_df: DataFrame with ignore intervals.
        clips_df: DataFrame with clip metadata including split assignment.
        config: Window extraction configuration.
        split: Optional split filter ("train", "val", "test"). If None, all splits.

    Returns:
        DataFrame with one row per window sample.
    """
    all_samples: list[dict] = []
    sample_counter = 0

    # Filter by split if specified
    if split is not None:
        clip_ids_in_split = set(clips_df[clips_df["split"] == split]["clip_id"])
        candidates_df = candidates_df[candidates_df["clip_id"].isin(clip_ids_in_split)]

    # Process each candidate
    for _, candidate in candidates_df.iterrows():
        clip_id = candidate["clip_id"]
        candidate_id = candidate["candidate_id"]
        actor_id = candidate["actor_id"]
        region_id = candidate.get("region_id")

        # Get candidate interval
        window_start = candidate["window_start_s"]
        window_end = candidate["window_end_s"]
        candidate_duration = window_end - window_start

        # Skip candidates that are too short
        if candidate_duration < config.min_window_duration_s:
            logger.debug(
                f"Skipping candidate {candidate_id}: duration {candidate_duration:.2f}s "
                f"< min {config.min_window_duration_s:.2f}s"
            )
            continue

        # Get events for this clip
        clip_events = events_df[events_df["clip_id"] == clip_id]

        # Get ignore intervals for this clip
        clip_ignores = ignore_intervals_df[ignore_intervals_df["clip_id"] == clip_id]

        # Get clip split
        clip_split = clips_df[clips_df["clip_id"] == clip_id]["split"].iloc[0] if len(
            clips_df[clips_df["clip_id"] == clip_id]
        ) > 0 else None

        # Generate sliding windows for this candidate
        windows = _generate_windows_for_candidate(
            candidate_start=window_start,
            candidate_end=window_end,
            config=config,
        )

        for win_start, win_end in windows:
            # Check if window center is in ignore interval
            window_center = (win_start + win_end) / 2

            if _is_in_ignore_interval(window_center, clip_ignores):
                continue

            # Assign label based on window center
            label, event_id, event_confidence = _assign_window_label(
                window_center=window_center,
                events_df=clip_events,
            )

            # Compute sample weight
            sample_weight = _compute_sample_weight(event_confidence, config)

            # Create sample
            sample_id = f"b1_{sample_counter:06d}"
            sample_counter += 1

            sample = WindowSample(
                sample_id=sample_id,
                clip_id=clip_id,
                candidate_id=candidate_id,
                actor_id=actor_id,
                region_id=region_id,
                window_start_s=win_start,
                window_end_s=win_end,
                label=label,
                label_name=LABEL_NAMES[label],
                event_id=event_id,
                event_confidence=event_confidence,
                sample_weight=sample_weight,
                split=clip_split,
            )

            all_samples.append(sample.to_dict())

    manifest_df = pd.DataFrame(all_samples)

    logger.info(
        f"Built window manifest: {len(manifest_df)} samples "
        f"(split={split or 'all'})"
    )

    if len(manifest_df) > 0:
        label_counts = manifest_df["label_name"].value_counts().to_dict()
        logger.info(f"Label distribution: {label_counts}")

    return manifest_df


def generate_sliding_windows(
    start_s: float,
    end_s: float,
    window_duration_s: float,
    window_stride_s: float,
) -> list[tuple[float, float]]:
    """Generate sliding windows over an interval.

    This is the CORE window generation function, used by both training and inference.
    It contains no label logic - just pure window positioning.

    Args:
        start_s: Interval start time in seconds.
        end_s: Interval end time in seconds.
        window_duration_s: Duration of each window in seconds.
        window_stride_s: Stride between window starts in seconds.

    Returns:
        List of (window_start, window_end) tuples.

    Example:
        >>> generate_sliding_windows(10.0, 25.0, 2.5, 0.5)
        [(10.0, 12.5), (10.5, 13.0), (11.0, 13.5), ...]
    """
    windows: list[tuple[float, float]] = []

    interval_duration = end_s - start_s

    # Handle case where interval is shorter than window duration
    if interval_duration < window_duration_s:
        # Center a single window on the interval
        center = (start_s + end_s) / 2
        win_start = center - window_duration_s / 2
        win_end = center + window_duration_s / 2

        # Clamp to interval bounds (may result in shorter window)
        win_start = max(start_s, win_start)
        win_end = min(end_s, win_end)

        windows.append((win_start, win_end))
        return windows

    # Slide windows with stride
    current_start = start_s

    while current_start + window_duration_s <= end_s:
        win_end = current_start + window_duration_s
        windows.append((current_start, win_end))
        current_start += window_stride_s

    # Add final window if there's remaining content
    if current_start < end_s and len(windows) > 0:
        # Check if last window doesn't already cover the end
        last_win_end = windows[-1][1]
        if last_win_end < end_s - 0.1:  # 100ms tolerance
            # Add window ending at interval end
            final_start = end_s - window_duration_s
            if final_start > windows[-1][0]:  # Don't duplicate
                windows.append((final_start, end_s))

    return windows


def _generate_windows_for_candidate(
    candidate_start: float,
    candidate_end: float,
    config: WindowConfig,
) -> list[tuple[float, float]]:
    """Slide windows over a single candidate interval.

    Wrapper around generate_sliding_windows() that uses WindowConfig.

    Args:
        candidate_start: Start time of candidate in seconds.
        candidate_end: End time of candidate in seconds.
        config: Window configuration.

    Returns:
        List of (window_start, window_end) tuples.
    """
    return generate_sliding_windows(
        start_s=candidate_start,
        end_s=candidate_end,
        window_duration_s=config.window_duration_s,
        window_stride_s=config.window_stride_s,
    )


# ============================================================
# INFERENCE WINDOW GENERATION (NO LABELS)
# ============================================================


@dataclass
class InferenceWindow:
    """A window for inference (no label)."""

    clip_id: str
    candidate_id: str
    actor_id: str
    region_id: Optional[str]
    window_start_s: float
    window_end_s: float
    window_center_s: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "clip_id": self.clip_id,
            "candidate_id": self.candidate_id,
            "actor_id": self.actor_id,
            "region_id": self.region_id,
            "window_start_s": self.window_start_s,
            "window_end_s": self.window_end_s,
            "window_center_s": self.window_center_s,
        }


def generate_inference_windows(
    candidates_df: pd.DataFrame,
    config: WindowConfig,
) -> list[InferenceWindow]:
    """Generate windows for inference (no labels needed).

    This function generates sliding windows over candidates WITHOUT
    any label assignment. Used for inference on new/unlabeled videos.

    Args:
        candidates_df: DataFrame with candidate intervals.
        config: Window configuration.

    Returns:
        List of InferenceWindow objects ready for model prediction.

    Example:
        >>> windows = generate_inference_windows(candidates_df, config)
        >>> for win in windows:
        ...     frames = load_frames(video, win.window_start_s, win.window_end_s)
        ...     prediction = model(frames)
    """
    all_windows: list[InferenceWindow] = []

    for _, candidate in candidates_df.iterrows():
        clip_id = candidate["clip_id"]
        candidate_id = candidate["candidate_id"]
        actor_id = candidate["actor_id"]
        region_id = candidate.get("region_id")

        # Get candidate interval
        cand_start = candidate["window_start_s"]
        cand_end = candidate["window_end_s"]
        candidate_duration = cand_end - cand_start

        # Skip candidates that are too short
        if candidate_duration < config.min_window_duration_s:
            logger.debug(
                f"Skipping candidate {candidate_id}: duration {candidate_duration:.2f}s "
                f"< min {config.min_window_duration_s:.2f}s"
            )
            continue

        # Generate sliding windows (shared logic!)
        windows = generate_sliding_windows(
            start_s=cand_start,
            end_s=cand_end,
            window_duration_s=config.window_duration_s,
            window_stride_s=config.window_stride_s,
        )

        # Create InferenceWindow objects
        for win_start, win_end in windows:
            window = InferenceWindow(
                clip_id=clip_id,
                candidate_id=candidate_id,
                actor_id=actor_id,
                region_id=region_id,
                window_start_s=win_start,
                window_end_s=win_end,
                window_center_s=(win_start + win_end) / 2,
            )
            all_windows.append(window)

    logger.info(f"Generated {len(all_windows)} inference windows from {len(candidates_df)} candidates")

    return all_windows


def generate_inference_windows_for_candidate(
    candidate: pd.Series,
    config: WindowConfig,
) -> list[InferenceWindow]:
    """Generate inference windows for a single candidate.

    Convenience function for processing one candidate at a time.

    Args:
        candidate: Single candidate row from DataFrame.
        config: Window configuration.

    Returns:
        List of InferenceWindow objects for this candidate.
    """
    clip_id = candidate["clip_id"]
    candidate_id = candidate["candidate_id"]
    actor_id = candidate["actor_id"]
    region_id = candidate.get("region_id")

    cand_start = candidate["window_start_s"]
    cand_end = candidate["window_end_s"]

    # Generate sliding windows
    windows = generate_sliding_windows(
        start_s=cand_start,
        end_s=cand_end,
        window_duration_s=config.window_duration_s,
        window_stride_s=config.window_stride_s,
    )

    # Create InferenceWindow objects
    return [
        InferenceWindow(
            clip_id=clip_id,
            candidate_id=candidate_id,
            actor_id=actor_id,
            region_id=region_id,
            window_start_s=win_start,
            window_end_s=win_end,
            window_center_s=(win_start + win_end) / 2,
        )
        for win_start, win_end in windows
    ]


def _assign_window_label(
    window_center: float,
    events_df: pd.DataFrame,
) -> tuple[int, Optional[str], Optional[str]]:
    """Determine label based on window center vs event intervals.

    Uses center-based assignment: the label is determined by which event
    interval (if any) contains the window center.

    Args:
        window_center: Center timestamp of the window in seconds.
        events_df: DataFrame with events for this clip.

    Returns:
        Tuple of (label_id, event_id or None, event_confidence or None).
    """
    if events_df.empty:
        return LABEL_BACKGROUND, None, None

    # Find event whose interval contains the window center
    for _, event in events_df.iterrows():
        event_start = event["t_start"]
        event_end = event["t_end"]

        if event_start <= window_center <= event_end:
            event_type = event["type"]
            event_id = event["event_id"]
            confidence = event.get("confidence", "high")

            if event_type == "pickup":
                return LABEL_PICKUP, event_id, confidence
            elif event_type == "putdown":
                return LABEL_PUTDOWN, event_id, confidence

    return LABEL_BACKGROUND, None, None


def _is_in_ignore_interval(
    timestamp: float,
    ignore_intervals_df: pd.DataFrame,
) -> bool:
    """Check if a timestamp falls within any ignore interval.

    Args:
        timestamp: Timestamp to check in seconds.
        ignore_intervals_df: DataFrame with ignore intervals for the clip.

    Returns:
        True if timestamp is inside an ignore interval.
    """
    if ignore_intervals_df.empty:
        return False

    for _, interval in ignore_intervals_df.iterrows():
        if interval["t_start"] <= timestamp <= interval["t_end"]:
            return True

    return False


def _compute_sample_weight(
    event_confidence: Optional[str],
    config: WindowConfig,
) -> float:
    """Compute sample weight based on event confidence.

    Args:
        event_confidence: Confidence level ("high", "med", "low") or None.
        config: Window configuration with weight settings.

    Returns:
        Sample weight for loss computation.
    """
    if event_confidence is None:
        return config.weight_high  # Background samples get full weight

    confidence_weights = {
        "high": config.weight_high,
        "med": config.weight_med,
        "low": config.weight_low,
    }

    return confidence_weights.get(event_confidence, config.weight_high)


# ============================================================
# VIDEO DECODING
# ============================================================


def decode_window_frames(
    video_path: Path,
    start_s: float,
    end_s: float,
    num_frames: int,
    frame_transform: Optional[Callable] = None,
) -> Optional[np.ndarray]:
    """Seek once, decode forward, and optionally crop each selected frame.

    Sampling indices are unchanged. Failed reads raise instead of silently
    caching black/repeated frames with valid event labels.
    """
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    # FFmpeg decoder threads are independent of cv2.setNumThreads().
    if hasattr(cv2, "CAP_PROP_N_THREADS"):
        cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_N_THREADS, 1])
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(str(video_path))
    else:
        cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid FPS for {video_path}: {fps}")
        indices = _compute_frame_indices(start_s, end_s, num_frames, fps)
        if indices[0] < 0:
            raise ValueError(f"Negative frame index for {video_path}")
        if indices[0] and not cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0]):
            raise RuntimeError(f"Cannot seek to {indices[0]} in {video_path}")
        frames = []
        current = indices[0]
        previous_index = None
        selected = None
        for target in indices:
            if target != previous_index:
                while current < target:
                    if not cap.grab():
                        raise RuntimeError(f"Decode failed at {current} in {video_path}")
                    current += 1
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Decode failed at {target} in {video_path}")
                current += 1
                selected = frame_transform(frame) if frame_transform else frame
                previous_index = target
            frames.append(selected)
        return np.stack(frames)
    finally:
        cap.release()


def _compute_frame_indices(
    start_s: float,
    end_s: float,
    num_frames: int,
    video_fps: float,
) -> list[int]:
    """Calculate which frame indices to extract for uniform sampling.

    Args:
        start_s: Start timestamp in seconds.
        end_s: End timestamp in seconds.
        num_frames: Number of frames to sample.
        video_fps: Video frame rate.

    Returns:
        List of frame indices to extract.
    """
    start_frame = int(start_s * video_fps)
    end_frame = int(end_s * video_fps)

    # Ensure at least 1 frame span
    if end_frame <= start_frame:
        end_frame = start_frame + 1

    # Uniform sampling
    if num_frames == 1:
        return [start_frame]

    # Compute evenly spaced frame indices
    indices = np.linspace(start_frame, end_frame - 1, num_frames)
    return [int(idx) for idx in indices]


# ============================================================
# ACTOR-CONDITIONED CROPPING
# ============================================================


def compute_actor_crop_box(
    pose_track_df: pd.DataFrame,
    shelf_region: Optional[dict],
    start_s: float,
    end_s: float,
    margin: float,
    frame_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Compute crop box as union of actor boxes + shelf region + margin.

    Args:
        pose_track_df: DataFrame with pose observations for this actor.
        shelf_region: Dict with shelf polygon bounds, or None.
        start_s: Window start time in seconds.
        end_s: Window end time in seconds.
        margin: Margin to add as fraction (e.g., 0.15 for 15%).
        frame_size: (width, height) of the video frame.

    Returns:
        Tuple of (x1, y1, x2, y2) crop coordinates.
    """
    frame_w, frame_h = frame_size

    # Get actor bounding boxes in time range
    actor_box = _union_actor_boxes(pose_track_df, start_s, end_s)

    if actor_box is None:
        # Fallback to full frame
        return 0, 0, frame_w, frame_h

    # Expand with shelf region if available
    if shelf_region is not None:
        actor_box = _expand_with_shelf_region(actor_box, shelf_region)

    # Apply margin and clamp to frame
    return _apply_margin_and_clamp(actor_box, margin, frame_size)


def _union_actor_boxes(
    pose_track_df: pd.DataFrame,
    start_s: float,
    end_s: float,
) -> Optional[tuple[float, float, float, float]]:
    """Merge all actor bounding boxes in time range.

    Args:
        pose_track_df: DataFrame with pose observations.
        start_s: Start time in seconds.
        end_s: End time in seconds.

    Returns:
        Tuple of (x1, y1, x2, y2) or None if no boxes found.
    """
    if pose_track_df.empty or "timestamp_s" not in pose_track_df.columns:
        return None

    # Filter to time range
    mask = (pose_track_df["timestamp_s"] >= start_s) & (
        pose_track_df["timestamp_s"] <= end_s
    )
    relevant = pose_track_df[mask]

    if relevant.empty:
        return None

    # Check for bbox columns
    bbox_cols = ["person_bbox_x1", "person_bbox_y1", "person_bbox_x2", "person_bbox_y2"]
    if not all(col in relevant.columns for col in bbox_cols):
        return None

    # Filter out rows with missing bbox
    relevant = relevant.dropna(subset=bbox_cols)
    if relevant.empty:
        return None

    # Compute union
    x1 = relevant["person_bbox_x1"].min()
    y1 = relevant["person_bbox_y1"].min()
    x2 = relevant["person_bbox_x2"].max()
    y2 = relevant["person_bbox_y2"].max()

    return (x1, y1, x2, y2)


def _expand_with_shelf_region(
    actor_box: tuple[float, float, float, float],
    shelf_region: dict,
) -> tuple[float, float, float, float]:
    """Expand actor box to include shelf region bounds.

    Args:
        actor_box: (x1, y1, x2, y2) actor bounding box.
        shelf_region: Dict with 'polygon' key containing list of [x, y] points.

    Returns:
        Expanded (x1, y1, x2, y2) including shelf region.
    """
    ax1, ay1, ax2, ay2 = actor_box

    polygon = shelf_region.get("polygon", [])
    if not polygon:
        return actor_box

    # Get shelf bounds
    shelf_xs = [p[0] for p in polygon]
    shelf_ys = [p[1] for p in polygon]

    sx1 = min(shelf_xs)
    sy1 = min(shelf_ys)
    sx2 = max(shelf_xs)
    sy2 = max(shelf_ys)

    # Union of actor and shelf bounds
    return (
        min(ax1, sx1),
        min(ay1, sy1),
        max(ax2, sx2),
        max(ay2, sy2),
    )


def _apply_margin_and_clamp(
    box: tuple[float, float, float, float],
    margin: float,
    frame_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Add margin percentage and clamp to frame boundaries.

    Args:
        box: (x1, y1, x2, y2) bounding box.
        margin: Margin as fraction (e.g., 0.15 for 15%).
        frame_size: (width, height) of frame.

    Returns:
        Clamped (x1, y1, x2, y2) as integers.
    """
    x1, y1, x2, y2 = box
    frame_w, frame_h = frame_size

    # Compute box dimensions
    box_w = x2 - x1
    box_h = y2 - y1

    # Add margin
    margin_w = box_w * margin
    margin_h = box_h * margin

    x1 = x1 - margin_w
    y1 = y1 - margin_h
    x2 = x2 + margin_w
    y2 = y2 + margin_h

    # Clamp to frame
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(frame_w, int(x2))
    y2 = min(frame_h, int(y2))

    return (x1, y1, x2, y2)


def apply_crop_and_resize(
    frames: np.ndarray,
    crop_box: tuple[int, int, int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    """Crop and resize frames to target resolution.

    Args:
        frames: Array of shape [T, H, W, C] in uint8 format.
        crop_box: (x1, y1, x2, y2) crop coordinates.
        target_size: (width, height) target resolution.

    Returns:
        Array of shape [T, target_H, target_W, C] in uint8 format.
    """
    x1, y1, x2, y2 = crop_box
    target_w, target_h = target_size

    cropped_frames: list[np.ndarray] = []

    for frame in frames:
        # Crop
        cropped = frame[y1:y2, x1:x2]

        # Handle edge case of empty crop
        if cropped.size == 0:
            cropped = frame

        # Resize
        resized = cv2.resize(cropped, (target_w, target_h))
        cropped_frames.append(resized)

    return np.stack(cropped_frames, axis=0)


def normalize_frames(
    frames: np.ndarray,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Normalize frames for model input.

    Converts BGR uint8 to RGB float32 and applies ImageNet normalization.

    Args:
        frames: Array of shape [T, H, W, C] in uint8 BGR format.
        mean: Per-channel mean for normalization.
        std: Per-channel std for normalization.

    Returns:
        Tensor of shape [T, C, H, W] in float32 normalized format.
    """
    # Convert BGR to RGB
    frames = frames[..., ::-1].copy()  # [T, H, W, C] BGR -> RGB

    # Convert to float and scale to [0, 1]
    frames = frames.astype(np.float32) / 255.0

    # Normalize with ImageNet stats
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    frames = (frames - mean) / std

    # Convert to tensor and reorder to [T, C, H, W]
    tensor = torch.from_numpy(frames)
    tensor = tensor.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]

    return tensor


# ============================================================
# PYTORCH DATASET
# ============================================================


class TrackB1Dataset(Dataset):
    """PyTorch Dataset for Track B1 VideoMAE training/inference.

    Each sample returns:
        - pixel_values: Tensor [T, C, H, W] normalized frames
        - label: int class label
        - metadata: dict with clip_id, actor_id, timestamps, etc.
    """

    def __init__(
        self,
        window_manifest: pd.DataFrame,
        video_dir: Path,
        pose_tracks_dir: Path,
        shelf_regions: dict[str, dict],
        config: WindowConfig,
        clips_df: Optional[pd.DataFrame] = None,
        transform: Optional[Callable] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """Initialize dataset with manifest and paths.

        Args:
            window_manifest: DataFrame with window samples.
            video_dir: Directory containing video files.
            pose_tracks_dir: Directory containing pose track parquet files.
            shelf_regions: Dict mapping region_id to region config.
            config: Window extraction configuration.
            clips_df: Optional DataFrame with clip metadata (for video paths).
            transform: Optional additional transform to apply.
        """
        self.manifest = window_manifest.reset_index(drop=True)
        self.video_dir = Path(video_dir)
        self.pose_tracks_dir = Path(pose_tracks_dir)
        self.shelf_regions = shelf_regions
        self.config = config
        self.clips_df = clips_df
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache for pose tracks (loaded on demand)
        self._pose_cache: dict[str, pd.DataFrame] = {}

        # Validate manifest
        self._validate_manifest()

        logger.info(
            f"TrackB1Dataset initialized with {len(self.manifest)} samples"
        )

    def _validate_manifest(self) -> None:
        """Check manifest has required columns."""
        required_cols = [
            "sample_id",
            "clip_id",
            "candidate_id",
            "actor_id",
            "window_start_s",
            "window_end_s",
            "label",
        ]
        missing = [col for col in required_cols if col not in self.manifest.columns]
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        """Load a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dict with:
                - pixel_values: Tensor [T, C, H, W]
                - label: int
                - sample_id: str
                - clip_id: str
                - actor_id: str
                - candidate_id: str
                - window_start_s: float
                - window_end_s: float
                - sample_weight: float
        """
        row = self.manifest.iloc[idx]

        clip_id = row["clip_id"]
        actor_id = row["actor_id"]
        candidate_id = row["candidate_id"]
        region_id = row.get("region_id")
        window_start = row["window_start_s"]
        window_end = row["window_end_s"]
        label = int(row["label"])
        sample_weight = row.get("sample_weight", 1.0)

        # Get video path
        video_path = self._get_video_path(clip_id)

        pose_file = self.pose_tracks_dir / f"{clip_id}.parquet"
        shelf_region = self.shelf_regions.get(region_id) if region_id else None
        cache_path = None
        frames = None
        expected_shape = (self.config.num_frames, self.config.image_size[1],
                          self.config.image_size[0], 3)
        if self.cache_dir is not None:
            # Labels are deliberately excluded: cached pixels never carry GT.
            # Source/pose stat fingerprints assume inputs are stable during a run.
            def fingerprint(path):
                if not path.exists():
                    return [str(path.resolve()), None]
                stat = path.stat()
                return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]

            key = {"version": 1, "video": fingerprint(video_path),
                   "pose": fingerprint(pose_file), "actor": str(actor_id),
                   "start": float(window_start), "end": float(window_end),
                   "frames": self.config.num_frames,
                   "size": list(self.config.image_size),
                   "margin": self.config.crop_margin, "shelf": shelf_region}
            digest = hashlib.sha256(json.dumps(key, sort_keys=True,
                                               default=str).encode()).hexdigest()
            cache_path = self.cache_dir / digest[:2] / f"{digest}.npy"
            if cache_path.exists():
                try:
                    frames = np.load(cache_path, allow_pickle=False)
                    if frames.shape != expected_shape or frames.dtype != np.uint8:
                        raise ValueError("Invalid cached window shape/dtype")
                except (OSError, ValueError, EOFError) as exc:
                    logger.warning("Rebuilding invalid cache %s: %s", cache_path, exc)
                    frames = None

        cache_hit = frames is not None
        if frames is None:
            pose_track = self._load_pose_track(clip_id, actor_id)
            crop_box = None

            def crop_frame(frame):
                nonlocal crop_box
                if crop_box is None:
                    height, width = frame.shape[:2]
                    crop_box = compute_actor_crop_box(
                        pose_track, shelf_region, window_start, window_end,
                        self.config.crop_margin, (width, height))
                return apply_crop_and_resize(
                    frame[None], crop_box, self.config.image_size)[0]

            frames = decode_window_frames(
                video_path, window_start, window_end, self.config.num_frames,
                frame_transform=crop_frame)
            if frames is None:
                raise RuntimeError(f"Cannot decode sample {row['sample_id']}")
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                # Unique temp + atomic replace: concurrent workers may compute
                # the same sample, but readers never see a partial .npy file.
                temp_name = None
                try:
                    with tempfile.NamedTemporaryFile(
                        dir=cache_path.parent, suffix=".tmp", delete=False
                    ) as stream:
                        temp_name = stream.name
                        np.save(stream, frames, allow_pickle=False)
                    os.replace(temp_name, cache_path)
                finally:
                    if temp_name is not None and os.path.exists(temp_name):
                        os.unlink(temp_name)

        # Normalize to tensor
        pixel_values = normalize_frames(frames)

        # Apply additional transform if provided
        if self.transform is not None:
            pixel_values = self.transform(pixel_values)

        return {
            "pixel_values": pixel_values,
            "cache_hit": cache_hit,
            "label": label,
            "sample_id": row["sample_id"],
            "clip_id": clip_id,
            "actor_id": actor_id,
            "candidate_id": candidate_id,
            "window_start_s": window_start,
            "window_end_s": window_end,
            "sample_weight": sample_weight,
        }

    def _get_video_path(self, clip_id: str) -> Path:
        """Resolve video file path from clip_id.

        Args:
            clip_id: Clip identifier.

        Returns:
            Path to video file.
        """
        # Try to get path from clips_df if available
        if self.clips_df is not None and "s3_key" in self.clips_df.columns:
            matches = self.clips_df[self.clips_df["clip_id"] == clip_id]
            if not matches.empty:
                s3_key = matches.iloc[0]["s3_key"]
                # Extract filename from S3 key
                filename = Path(s3_key).name
                video_path = self.video_dir / filename
                if video_path.exists():
                    return video_path

        # Fallback: try common patterns
        for ext in [".mp4", ".avi", ".mov", ".mkv"]:
            video_path = self.video_dir / f"{clip_id}{ext}"
            if video_path.exists():
                return video_path

        # Last resort: search in directory
        for video_file in self.video_dir.glob(f"*{clip_id}*"):
            if video_file.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
                return video_file

        # Return expected path even if not found (will fail gracefully in decode)
        return self.video_dir / f"{clip_id}.mp4"

    def _load_pose_track(
        self,
        clip_id: str,
        actor_id: str,
    ) -> pd.DataFrame:
        """Load actor's pose track for bounding box computation.

        Args:
            clip_id: Clip identifier.
            actor_id: Actor identifier.

        Returns:
            DataFrame with pose observations for this actor.
        """
        cache_key = f"{clip_id}_{actor_id}"

        if cache_key in self._pose_cache:
            return self._pose_cache[cache_key]

        # Load pose track file
        pose_file = self.pose_tracks_dir / f"{clip_id}.parquet"

        if not pose_file.exists():
            logger.warning(f"Pose track file not found: {pose_file}")
            return pd.DataFrame()

        try:
            pose_df = pd.read_parquet(pose_file)

            # Filter to this actor
            if "actor_id" in pose_df.columns:
                pose_df = pose_df[pose_df["actor_id"] == actor_id]

            self._pose_cache[cache_key] = pose_df
            return pose_df

        except Exception as e:
            logger.warning(f"Failed to load pose track {pose_file}: {e}")
            return pd.DataFrame()


# ============================================================
# UTILITIES
# ============================================================


def get_label_weights(
    manifest: pd.DataFrame,
    num_classes: int = 3,
) -> torch.Tensor:
    """Compute inverse frequency class weights for imbalanced labels.

    Args:
        manifest: DataFrame with 'label' column.
        num_classes: Number of classes (default 3: bg/pickup/putdown).

    Returns:
        Tensor of shape [num_classes] with class weights.
    """
    label_counts = manifest["label"].value_counts().sort_index()

    weights = torch.zeros(num_classes)
    total = len(manifest)

    for label_id in range(num_classes):
        count = label_counts.get(label_id, 0)
        if count > 0:
            weights[label_id] = total / (num_classes * count)
        else:
            weights[label_id] = 1.0

    return weights


def get_sample_weights(manifest: pd.DataFrame) -> torch.Tensor:
    """Get per-sample weights for WeightedRandomSampler.

    Combines class balancing with confidence-based weights.

    Args:
        manifest: DataFrame with 'label' and 'sample_weight' columns.

    Returns:
        Tensor of shape [num_samples] with per-sample weights.
    """
    # Class weights (inverse frequency)
    class_weights = get_label_weights(manifest)

    # Per-sample weights
    sample_weights = []
    for _, row in manifest.iterrows():
        label = int(row["label"])
        confidence_weight = row.get("sample_weight", 1.0)
        weight = class_weights[label].item() * confidence_weight
        sample_weights.append(weight)

    return torch.tensor(sample_weights, dtype=torch.float32)


def create_dataloaders(
    train_manifest: pd.DataFrame,
    val_manifest: pd.DataFrame,
    video_dir: Path,
    pose_tracks_dir: Path,
    shelf_regions: dict[str, dict],
    config: WindowConfig,
    batch_size: int = 8,
    num_workers: int = 4,
    clips_df: Optional[pd.DataFrame] = None,
    use_weighted_sampling: bool = True,
    cache_dir: Optional[Path] = None,
) -> tuple[DataLoader, DataLoader]:
    """Factory for train/val dataloaders.

    Args:
        train_manifest: Training window manifest.
        val_manifest: Validation window manifest.
        video_dir: Directory containing video files.
        pose_tracks_dir: Directory containing pose tracks.
        shelf_regions: Dict mapping region_id to region config.
        config: Window configuration.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        clips_df: Optional clips metadata DataFrame.
        use_weighted_sampling: Whether to use weighted random sampling for training.

    Returns:
        Tuple of (train_dataloader, val_dataloader).
    """
    train_dataset = TrackB1Dataset(
        window_manifest=train_manifest,
        video_dir=video_dir,
        pose_tracks_dir=pose_tracks_dir,
        shelf_regions=shelf_regions,
        config=config,
        clips_df=clips_df,
        cache_dir=cache_dir,
    )

    val_dataset = TrackB1Dataset(
        window_manifest=val_manifest,
        video_dir=video_dir,
        pose_tracks_dir=pose_tracks_dir,
        shelf_regions=shelf_regions,
        config=config,
        clips_df=clips_df,
        cache_dir=cache_dir,
    )

    # Training sampler (weighted for class balance)
    if use_weighted_sampling and len(train_manifest) > 0:
        sample_weights = get_sample_weights(train_manifest)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_manifest),
            replacement=True,
        )
        train_shuffle = False
    else:
        sampler = None
        train_shuffle = True

    worker_options = {}
    if num_workers > 0:
        worker_options = dict(
            persistent_workers=True, prefetch_factor=1,
            worker_init_fn=_init_data_worker, multiprocessing_context="spawn")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        **worker_options,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **worker_options,
        pin_memory=True,
        drop_last=False,
    )

    logger.info(
        f"Created dataloaders: train={len(train_dataset)} samples, "
        f"val={len(val_dataset)} samples, batch_size={batch_size}"
    )

    return train_loader, val_loader

def _init_data_worker(worker_id: int) -> None:
    """Keep each worker's CPU libraries from oversubscribing the machine."""
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
