"""Track B1 inference: sliding window prediction and event decoding.

This module provides:
- Sliding window inference over candidate intervals
- Temporal smoothing of prediction scores
- Peak detection for event localization
- Same-type merging (never merge pickup + putdown)
- Canonical predictions.csv output

Inference pipeline:
    1. Generate sliding windows over candidate (reuses dataset.py logic)
    2. Run model on each window → class probabilities
    3. Smooth scores temporally
    4. Detect peaks above threshold
    5. Merge same-type overlapping detections
    6. Output canonical event predictions
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from pickup_putdown.layer1.track_b1.dataset import (
    LABEL_BACKGROUND,
    LABEL_PICKUP,
    LABEL_PUTDOWN,
    InferenceWindow,
    WindowConfig,
    crop_span,
    generate_inference_windows_for_candidate,
    load_shelf_regions,
    normalize_frames,
    prepare_window_frames,
)
from pickup_putdown.layer1.track_b1.videomae_classifier import (
    VideoMAEClassifier,
    load_checkpoint,
)

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass
class InferenceConfig:
    """Inference configuration."""

    # Window parameters (must match training)
    window_duration_s: float = 2.5
    window_stride_s: float = 0.5
    num_frames: int = 16
    image_size: tuple[int, int] = (224, 224)
    crop_margin: float = 0.15
    crop_scope: str = "window"  # see WindowConfig.crop_scope
    resize_interpolation: str = "linear"  # see WindowConfig.resize_interpolation
    include_shelf_region: bool = False  # see WindowConfig.include_shelf_region

    # Thresholds (tune on validation set)
    pickup_threshold: float = 0.5
    putdown_threshold: float = 0.5

    # Temporal smoothing
    smoothing_window: int = 3  # Number of adjacent predictions to average (must be odd)

    # Merging
    same_type_merge_gap_s: float = 0.75  # Merge same-type if gap < this
    min_event_duration_s: float = 0.3  # Discard events shorter than this

    # How a run of above-threshold windows becomes an event interval.
    #   "window_span"    — first window start to last window end (original behaviour)
    #   "window_centers" — first to last window centre, widened by half a stride
    # A window is much longer than a typical event here (2.5 s against a 0.9 s
    # median), so "window_span" cannot emit an interval tighter than one window and
    # caps achievable tIoU at roughly event_duration / window_duration. Centres carry
    # the actual temporal evidence: a window is labelled by what sits at its centre.
    boundary_mode: str = "window_span"

    # Output
    model_name: str = "layer1_track_b1_videomae_window_v1"

    # Processing
    batch_size: int = 8
    num_workers: int = 4  # DataLoader workers for parallel CPU decoding
    pin_memory: bool = True  # Faster CPU→GPU transfer (allocates in page-locked memory)
    prefetch_factor: int = 2  # Batches to prefetch per worker

    def to_window_config(self) -> WindowConfig:
        """Convert to WindowConfig for window generation."""
        return WindowConfig(
            window_duration_s=self.window_duration_s,
            window_stride_s=self.window_stride_s,
            num_frames=self.num_frames,
            image_size=self.image_size,
            crop_margin=self.crop_margin,
            crop_scope=self.crop_scope,
            resize_interpolation=self.resize_interpolation,
            include_shelf_region=self.include_shelf_region,
        )


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass
class WindowPrediction:
    """Prediction for a single window."""

    window_start_s: float
    window_end_s: float
    window_center_s: float
    probs: np.ndarray  # [3] probabilities: [bg, pickup, putdown]
    predicted_class: int
    confidence: float

    @property
    def background_prob(self) -> float:
        return float(self.probs[LABEL_BACKGROUND])

    @property
    def pickup_prob(self) -> float:
        return float(self.probs[LABEL_PICKUP])

    @property
    def putdown_prob(self) -> float:
        return float(self.probs[LABEL_PUTDOWN])


@dataclass
class ScoreRegion:
    """A contiguous region where score exceeds threshold."""

    event_type: str  # "pickup" or "putdown"
    start_s: float
    end_s: float
    peak_score: float
    mean_score: float


@dataclass
class EventPrediction:
    """A detected event in canonical format."""

    pred_id: str
    clip_id: str
    candidate_id: str
    actor_id: str
    event_type: str  # "pickup" or "putdown"
    t_start: float
    t_end: float
    score: float
    model: str

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame/CSV export."""
        return {
            "pred_id": self.pred_id,
            "clip_id": self.clip_id,
            "type": self.event_type,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "score": self.score,
            "model": self.model,
            # Internal fields (not in canonical schema but useful)
            "candidate_id": self.candidate_id,
            "actor_id": self.actor_id,
        }


# ============================================================
# INFERENCE DATASET (for parallel CPU decoding)
# ============================================================


class InferenceWindowDataset(Dataset):
    """PyTorch Dataset for inference windows with parallel frame decoding.

    Each worker process decodes frames independently, enabling CPU/GPU overlap.
    Video captures are lazily initialized per-worker (cv2.VideoCapture can't be pickled).
    """

    def __init__(
        self,
        windows: list[InferenceWindow],
        video_path: Path,
        pose_track_df: pd.DataFrame,
        shelf_region: dict | None,
        config: InferenceConfig,
    ):
        self.windows = windows
        self.video_path = video_path
        self.pose_track_df = pose_track_df
        self.shelf_region = shelf_region
        self.config = config

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        """Prepare single window tensor with metadata.

        Returns:
            Dict with:
                - tensor: [num_frames, C, H, W] normalized tensor
                - window_start_s: float
                - window_end_s: float
                - window_center_s: float
        """
        window = self.windows[idx]
        window_config = self.config.to_window_config()
        span = crop_span(pd.Series(window.to_dict()), window_config)
        shelf = self.shelf_region if window_config.include_shelf_region else None
        frames = prepare_window_frames(
            self.video_path,
            self.pose_track_df,
            shelf,
            window.window_start_s,
            window.window_end_s,
            span,
            window_config,
        )

        # Normalize to tensor
        tensor = normalize_frames(frames)

        return {
            "tensor": tensor,
            "window_start_s": window.window_start_s,
            "window_end_s": window.window_end_s,
            "window_center_s": window.window_center_s,
        }


def inference_collate_fn(batch: list[dict]) -> dict:
    """Custom collate function for inference batches.

    Stacks tensors and preserves window metadata as lists.

    Args:
        batch: List of dicts from InferenceWindowDataset.__getitem__

    Returns:
        Dict with:
            - tensor: [batch_size, num_frames, C, H, W] stacked tensor
            - window_start_s: list[float]
            - window_end_s: list[float]
            - window_center_s: list[float]
    """
    return {
        "tensor": torch.stack([item["tensor"] for item in batch], dim=0),
        "window_start_s": [item["window_start_s"] for item in batch],
        "window_end_s": [item["window_end_s"] for item in batch],
        "window_center_s": [item["window_center_s"] for item in batch],
    }


# ============================================================
# MODEL INFERENCE
# ============================================================


@torch.no_grad()
def predict_windows(
    model: VideoMAEClassifier,
    windows: list[InferenceWindow],
    video_path: Path,
    pose_track_df: pd.DataFrame,
    shelf_region: dict | None,
    config: InferenceConfig,
    device: torch.device,
) -> list[WindowPrediction]:
    """Run model inference on all windows for a candidate.

    Uses DataLoader with multiple workers for parallel CPU frame decoding,
    eliminating GPU idle time while waiting for batches.

    Args:
        model: Trained VideoMAE classifier.
        windows: List of InferenceWindow objects.
        video_path: Path to video file.
        pose_track_df: Pose track DataFrame for this actor.
        shelf_region: Shelf region config dict.
        config: Inference configuration.
        device: Device to run inference on.

    Returns:
        List of WindowPrediction for each window.
    """
    if not windows:
        return []

    model.eval()
    predictions: list[WindowPrediction] = []

    # Create dataset and dataloader for parallel CPU decoding
    dataset = InferenceWindowDataset(
        windows=windows,
        video_path=video_path,
        pose_track_df=pose_track_df,
        shelf_region=shelf_region,
        config=config,
    )

    # Configure DataLoader
    # - num_workers > 0: parallel CPU decoding in separate processes
    # - pin_memory: faster CPU→GPU transfer
    # - prefetch_factor: batches to prepare ahead per worker
    use_multiprocessing = config.num_workers > 0
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,  # Preserve temporal order
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == "cuda",
        prefetch_factor=config.prefetch_factor if use_multiprocessing else None,
        collate_fn=inference_collate_fn,
        persistent_workers=use_multiprocessing,  # Keep workers alive between batches
    )

    # Process batches (workers prefetch in background)
    for batch in loader:
        batch_tensor = batch["tensor"].to(device, non_blocking=config.pin_memory)
        logits = model(batch_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # Create WindowPrediction for each item in batch
        batch_size = len(batch["window_start_s"])
        for i in range(batch_size):
            window_probs = probs[i]
            predicted_class = int(np.argmax(window_probs))
            confidence = float(window_probs[predicted_class])

            predictions.append(
                WindowPrediction(
                    window_start_s=batch["window_start_s"][i],
                    window_end_s=batch["window_end_s"][i],
                    window_center_s=batch["window_center_s"][i],
                    probs=window_probs,
                    predicted_class=predicted_class,
                    confidence=confidence,
                )
            )

    return predictions


# ============================================================
# TEMPORAL SMOOTHING
# ============================================================


def smooth_predictions(
    predictions: list[WindowPrediction],
    window_size: int = 3,
) -> list[WindowPrediction]:
    """Apply temporal smoothing to predictions.

    Averages probabilities over adjacent windows to reduce noise.
    Uses a simple moving average.

    Args:
        predictions: List of window predictions (must be time-ordered).
        window_size: Number of neighbors to average (must be odd).

    Returns:
        New list with smoothed probabilities.
    """
    if len(predictions) <= 1:
        return predictions

    if window_size < 1:
        return predictions

    # Ensure odd window size
    if window_size % 2 == 0:
        window_size += 1

    half_window = window_size // 2
    smoothed: list[WindowPrediction] = []

    for i, pred in enumerate(predictions):
        # Get neighbors
        start_idx = max(0, i - half_window)
        end_idx = min(len(predictions), i + half_window + 1)

        # Average probabilities
        neighbor_probs = [predictions[j].probs for j in range(start_idx, end_idx)]
        avg_probs = np.mean(neighbor_probs, axis=0)

        # Create smoothed prediction
        predicted_class = int(np.argmax(avg_probs))
        confidence = float(avg_probs[predicted_class])

        smoothed.append(
            WindowPrediction(
                window_start_s=pred.window_start_s,
                window_end_s=pred.window_end_s,
                window_center_s=pred.window_center_s,
                probs=avg_probs,
                predicted_class=predicted_class,
                confidence=confidence,
            )
        )

    return smoothed


# ============================================================
# PEAK DETECTION
# ============================================================


def detect_score_peaks(
    predictions: list[WindowPrediction],
    config: InferenceConfig,
) -> list[ScoreRegion]:
    """Find regions where class probability exceeds threshold.

    Processes pickup and putdown separately with their own thresholds.

    Args:
        predictions: Smoothed window predictions.
        config: Configuration with thresholds.

    Returns:
        List of ScoreRegion for detected events.
    """
    regions: list[ScoreRegion] = []

    # Detect pickup regions
    pickup_regions = _find_regions_above_threshold(
        predictions=predictions,
        class_idx=LABEL_PICKUP,
        threshold=config.pickup_threshold,
        event_type="pickup",
        boundary_mode=config.boundary_mode,
        min_duration_s=config.min_event_duration_s,
    )
    regions.extend(pickup_regions)

    # Detect putdown regions
    putdown_regions = _find_regions_above_threshold(
        predictions=predictions,
        class_idx=LABEL_PUTDOWN,
        threshold=config.putdown_threshold,
        event_type="putdown",
        boundary_mode=config.boundary_mode,
        min_duration_s=config.min_event_duration_s,
    )
    regions.extend(putdown_regions)

    return regions


def _find_regions_above_threshold(
    predictions: list[WindowPrediction],
    class_idx: int,
    threshold: float,
    event_type: str,
    boundary_mode: str = "window_span",
    min_duration_s: float = 0.3,
) -> list[ScoreRegion]:
    """Find contiguous runs where a class score exceeds its threshold.

    Args:
        predictions: Window predictions, in time order.
        class_idx: Class index to check (1=pickup, 2=putdown).
        threshold: Score threshold.
        event_type: Event type string for output.
        boundary_mode: "window_span" or "window_centers"; see InferenceConfig.
        min_duration_s: Floor applied in centre mode, where a single above-threshold
            window would otherwise produce a zero-length interval.

    Returns:
        List of detected ScoreRegion.
    """
    if boundary_mode not in ("window_span", "window_centers"):
        raise ValueError(f"unknown boundary_mode {boundary_mode!r}")

    regions: list[ScoreRegion] = []
    if not predictions:
        return regions

    # Half the spacing between consecutive windows: the finest boundary this stride
    # can resolve, and so the right amount to widen a centre-derived interval by.
    if len(predictions) > 1:
        half_stride = abs(predictions[1].window_center_s - predictions[0].window_center_s) / 2
    else:
        half_stride = min_duration_s / 2

    def close(run: list[WindowPrediction], scores: list[float]) -> None:
        if not run:
            return
        if boundary_mode == "window_span":
            start_s, end_s = run[0].window_start_s, run[-1].window_end_s
        else:
            start_s = run[0].window_center_s - half_stride
            end_s = run[-1].window_center_s + half_stride
            if end_s - start_s < min_duration_s:
                centre = (start_s + end_s) / 2
                start_s = centre - min_duration_s / 2
                end_s = centre + min_duration_s / 2
            start_s = max(0.0, start_s)
        regions.append(
            ScoreRegion(
                event_type=event_type,
                start_s=start_s,
                end_s=end_s,
                peak_score=max(scores),
                mean_score=float(np.mean(scores)),
            )
        )

    run: list[WindowPrediction] = []
    scores: list[float] = []

    for prediction in predictions:
        score = float(prediction.probs[class_idx])
        if score > threshold:
            run.append(prediction)
            scores.append(score)
        else:
            close(run, scores)
            run, scores = [], []

    close(run, scores)
    return regions


# ============================================================
# SAME-TYPE MERGING
# ============================================================


def merge_same_type_regions(
    regions: list[ScoreRegion],
    merge_gap_s: float,
    min_duration_s: float,
) -> list[ScoreRegion]:
    """Merge overlapping/adjacent regions of the SAME type.

    CRITICAL: Never merges pickup with putdown - they are different events!

    Args:
        regions: Detected score regions.
        merge_gap_s: Merge if gap between same-type regions < this.
        min_duration_s: Discard regions shorter than this.

    Returns:
        Merged and filtered regions.
    """
    if not regions:
        return []

    merged: list[ScoreRegion] = []

    # Process pickup and putdown separately
    pickup_regions = [r for r in regions if r.event_type == "pickup"]
    putdown_regions = [r for r in regions if r.event_type == "putdown"]

    merged.extend(_merge_regions_of_type(pickup_regions, merge_gap_s))
    merged.extend(_merge_regions_of_type(putdown_regions, merge_gap_s))

    # Filter by minimum duration
    filtered = [r for r in merged if (r.end_s - r.start_s) >= min_duration_s]

    # Sort by start time
    filtered.sort(key=lambda r: r.start_s)

    return filtered


def _merge_regions_of_type(
    regions: list[ScoreRegion],
    merge_gap_s: float,
) -> list[ScoreRegion]:
    """Merge overlapping/adjacent regions of the same type.

    Args:
        regions: Regions of a single type.
        merge_gap_s: Maximum gap to merge.

    Returns:
        Merged regions.
    """
    if not regions:
        return []

    # Sort by start time
    sorted_regions = sorted(regions, key=lambda r: r.start_s)

    merged: list[ScoreRegion] = []
    current = sorted_regions[0]

    for next_region in sorted_regions[1:]:
        gap = next_region.start_s - current.end_s

        if gap <= merge_gap_s:
            # Merge: extend current region
            current = ScoreRegion(
                event_type=current.event_type,
                start_s=current.start_s,
                end_s=max(current.end_s, next_region.end_s),
                peak_score=max(current.peak_score, next_region.peak_score),
                mean_score=(current.mean_score + next_region.mean_score) / 2,
            )
        else:
            # Gap too large: save current and start new
            merged.append(current)
            current = next_region

    # Don't forget last region
    merged.append(current)

    return merged


# ============================================================
# EVENT CREATION
# ============================================================


def create_event_predictions(
    regions: list[ScoreRegion],
    clip_id: str,
    candidate_id: str,
    actor_id: str,
    config: InferenceConfig,
) -> list[EventPrediction]:
    """Convert score regions to canonical event predictions.

    Args:
        regions: Merged score regions.
        clip_id: Clip identifier.
        candidate_id: Candidate identifier.
        actor_id: Actor identifier.
        config: Configuration with model name.

    Returns:
        List of EventPrediction in canonical format.
    """
    predictions: list[EventPrediction] = []

    for i, region in enumerate(regions):
        # Generate unique prediction ID
        pred_id = _generate_pred_id(clip_id, candidate_id, region, i)

        predictions.append(
            EventPrediction(
                pred_id=pred_id,
                clip_id=clip_id,
                candidate_id=candidate_id,
                actor_id=actor_id,
                event_type=region.event_type,
                t_start=region.start_s,
                t_end=region.end_s,
                score=region.peak_score,
                model=config.model_name,
            )
        )

    return predictions


# ============================================================
# DECODING PRECOMPUTED WINDOW SCORES (shared by inference and tuning scripts)
# ============================================================


def decode_window_scores(scores: pd.DataFrame, config: InferenceConfig) -> pd.DataFrame:
    """Smooth -> peaks -> same-type merge per candidate, from a window-score table.

    ``scores`` has one row per window with clip_id, candidate_id, actor_id,
    window_start_s, window_end_s and p_background/p_pickup/p_putdown. Returns
    canonical prediction rows (``EventPrediction.to_dict``), deterministically ordered.
    """
    rows: list[dict] = []
    for (clip_id, candidate_id, actor_id), group in scores.groupby(
        ["clip_id", "candidate_id", "actor_id"], sort=True
    ):
        group = group.sort_values("window_start_s")
        probs = group[["p_background", "p_pickup", "p_putdown"]].to_numpy(dtype=float)
        predictions = [
            WindowPrediction(
                window_start_s=float(start),
                window_end_s=float(end),
                window_center_s=(float(start) + float(end)) / 2,
                probs=prob,
                predicted_class=int(np.argmax(prob)),
                confidence=float(np.max(prob)),
            )
            for start, end, prob in zip(
                group["window_start_s"], group["window_end_s"], probs, strict=True
            )
        ]
        smoothed = smooth_predictions(predictions, config.smoothing_window)
        regions = detect_score_peaks(smoothed, config)
        merged = merge_same_type_regions(
            regions, config.same_type_merge_gap_s, config.min_event_duration_s
        )
        rows.extend(
            event.to_dict()
            for event in create_event_predictions(merged, clip_id, candidate_id, actor_id, config)
        )
    return pd.DataFrame(
        rows,
        columns=[
            "pred_id",
            "clip_id",
            "type",
            "t_start",
            "t_end",
            "score",
            "model",
            "candidate_id",
            "actor_id",
        ],
    )


def suppress_duplicate_events(predictions: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop same-type predictions of one actor that overlap a higher-scoring one.

    Pose proposals can give one actor several overlapping candidates, each of which
    decodes the same transfer; without this, one event yields a TP plus FPs. Opposite
    types and different actors are never suppressed against each other.
    """
    if predictions.empty:
        return predictions, 0
    kept: list[int] = []
    for _, group in predictions.groupby(["clip_id", "actor_id", "type"], sort=False):
        chosen: list[tuple[float, float]] = []
        for index, row in group.sort_values(
            ["score", "pred_id"], ascending=[False, True]
        ).iterrows():
            if all(row["t_end"] <= start or row["t_start"] >= end for start, end in chosen):
                chosen.append((row["t_start"], row["t_end"]))
                kept.append(index)
    result = predictions.loc[sorted(kept)].sort_values(["clip_id", "t_start", "type", "pred_id"])
    return result.reset_index(drop=True), len(predictions) - len(kept)


def _text(value: object, default: str) -> str:
    return value if isinstance(value, str) and value else default


# Ground-truth counting policies. Predictions are identical under both: the classifier
# emits one interval per detection and no item count, so nothing is duplicated to match
# item rows, and the evaluator matches each prediction to at most one GT row.
#   per_item        — primary. One row per item (LABELING_GUIDELINES §5.1): an event with
#                     item_count = N is N rows, so a single detection of it scores 1 TP
#                     and N-1 FN (per-item recall on such events is capped at 1/N).
#   per_event_group — historical comparison (the CVAT branch counted this way). The N
#                     rows of one annotated event collapse to one row per
#                     (clip_id, event_group_id).
COUNTING_POLICIES = ("per_item", "per_event_group")
PRIMARY_COUNTING_POLICY = "per_item"
_GROUP_INVARIANTS = ("type", "t_start", "t_end", "actor_id")


def ground_truth_for_policy(ground_truth: pd.DataFrame, policy: str) -> pd.DataFrame:
    """GT rows to score against under ``policy`` (see ``COUNTING_POLICIES``).

    Groups come only from explicit ``event_group_id`` provenance, scoped by clip; equal
    timestamps never merge rows, so independent simultaneous events, different actors
    and adjacent opposite-type events stay separate. Raises on missing group ids or on
    a group whose rows disagree on type, interval or actor, or whose size contradicts
    ``item_count`` / ``item_index``.
    """
    if policy not in COUNTING_POLICIES:
        raise ValueError(
            f"unknown counting policy {policy!r}; expected one of {COUNTING_POLICIES}"
        )
    if policy == "per_item" or ground_truth.empty:
        return ground_truth
    if "event_group_id" not in ground_truth.columns:
        raise ValueError("per_event_group counting needs an event_group_id column")
    group_ids = ground_truth["event_group_id"]
    missing = group_ids.isna() | (group_ids.astype(str).str.strip() == "")
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} GT rows lack event_group_id "
            f"(e.g. {ground_truth.loc[missing, 'event_id'].head(3).tolist()}); "
            "per_event_group counting will not guess their grouping"
        )
    invariants = [c for c in _GROUP_INVARIANTS if c in ground_truth.columns]
    kept: list[pd.Series] = []
    for (clip_id, group_id), rows in ground_truth.groupby(
        ["clip_id", "event_group_id"], sort=True, dropna=False
    ):
        conflicting = [c for c in invariants if rows[c].nunique(dropna=False) > 1]
        if conflicting:
            raise ValueError(
                f"event group {group_id!r} in {clip_id} is ambiguous: rows differ in {conflicting}"
            )
        if "item_count" in rows.columns:
            counts = set(rows["item_count"].astype(int))
            if counts != {len(rows)}:
                raise ValueError(
                    f"event group {group_id!r} in {clip_id} has {len(rows)} rows "
                    f"but item_count {sorted(counts)}"
                )
        if "item_index" in rows.columns and sorted(rows["item_index"].astype(int)) != list(
            range(len(rows))
        ):
            raise ValueError(f"event group {group_id!r} in {clip_id} has inconsistent item_index")
        order = "item_index" if "item_index" in rows.columns else "event_id"
        kept.append(rows.sort_values(order).iloc[0])
    return pd.DataFrame(kept).reset_index(drop=True)


def ground_truth_counts(ground_truth: pd.DataFrame) -> dict[str, int]:
    """Item rows, annotated event groups and distinct (clip, type, interval) spans."""
    return {
        "item_rows": int(len(ground_truth)),
        "event_groups": int(
            len(ground_truth_for_policy(ground_truth, "per_event_group"))
            if not ground_truth.empty
            else 0
        ),
        "distinct_intervals": int(
            ground_truth[["clip_id", "type", "t_start", "t_end"]].drop_duplicates().shape[0]
        ),
    }


def evaluate_events_by_policy(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    ignores: pd.DataFrame,
    clip_durations: dict[str, float],
    tiou_thresholds: tuple[float, ...] = (0.3, 0.5),
) -> dict[str, dict]:
    """``evaluate_events`` under every counting policy, with the GT counts each used."""
    return {
        policy: {
            "n_ground_truth_rows": int(len(truth)),
            **evaluate_events(predictions, truth, ignores, clip_durations, tiou_thresholds),
        }
        for policy in COUNTING_POLICIES
        for truth in [ground_truth_for_policy(ground_truth, policy)]
    }


def evaluate_events(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    ignores: pd.DataFrame,
    clip_durations: dict[str, float],
    tiou_thresholds: tuple[float, ...] = (0.3, 0.5),
) -> dict:
    """Score canonical predictions with the shared Task 8 evaluator.

    ``ground_truth`` must hold every reviewed event of the evaluated clips, including
    events no candidate covered; ``clip_durations`` must include reviewed clips with no
    events. Ignore intervals are applied inside ``aggregate_metrics``.
    """
    from pickup_putdown.evaluation.contracts import (
        EvaluationEvent,
        EvaluationIgnoreInterval,
        EvaluationPrediction,
    )
    from pickup_putdown.evaluation.metrics import aggregate_metrics

    truth = [
        EvaluationEvent(
            event_id=row.event_id,
            clip_id=row.clip_id,
            type=row.type,
            t_start=float(row.t_start),
            t_end=float(row.t_end),
            confidence=_text(getattr(row, "confidence", None), "high"),
            hard_case=bool(getattr(row, "hard_case", False)),
            group_id=_text(getattr(row, "event_group_id", None), ""),
        )
        for row in ground_truth.itertuples()
    ]
    predicted = [
        EvaluationPrediction(
            pred_id=row.pred_id,
            clip_id=row.clip_id,
            type=row.type,
            t_start=float(row.t_start),
            t_end=float(row.t_end),
            score=float(row.score),
            model=row.model,
        )
        for row in predictions.itertuples()
    ]
    ignore_rows = [
        EvaluationIgnoreInterval(
            clip_id=row.clip_id, t_start=float(row.t_start), t_end=float(row.t_end)
        )
        for row in ignores.itertuples()
    ]
    return aggregate_metrics(
        events=truth,
        preds=predicted,
        clip_durations=clip_durations,
        ignores=ignore_rows,
        tiou_thresholds=tuple(tiou_thresholds),
    )


def _generate_pred_id(
    clip_id: str,
    candidate_id: str,
    region: ScoreRegion,
    index: int,
) -> str:
    """Generate unique prediction ID."""
    payload = f"{clip_id}:{candidate_id}:{region.event_type}:{region.start_s:.3f}:{index}"
    hash_suffix = hashlib.sha256(payload.encode()).hexdigest()[:8]
    return f"b1_{hash_suffix}"


# ============================================================
# SINGLE CANDIDATE INFERENCE
# ============================================================


def infer_candidate(
    model: VideoMAEClassifier,
    candidate: pd.Series,
    video_path: Path,
    pose_track_df: pd.DataFrame,
    shelf_region: dict | None,
    config: InferenceConfig,
    device: torch.device,
) -> list[EventPrediction]:
    """Run full inference pipeline on one candidate.

    This is the main entry point for processing a single candidate.
    It orchestrates: window generation → prediction → smoothing →
    peak detection → merging → event creation.

    Args:
        model: Trained VideoMAE classifier.
        candidate: Candidate row from DataFrame.
        video_path: Path to video file.
        pose_track_df: Pose track for this actor.
        shelf_region: Shelf region config.
        config: Inference configuration.
        device: Device to run on.

    Returns:
        List of detected events (may be empty, one, or multiple).
    """
    clip_id = candidate["clip_id"]
    candidate_id = candidate["candidate_id"]
    actor_id = candidate["actor_id"]

    # 1. Generate windows (reuses dataset.py logic!)
    window_config = config.to_window_config()
    windows = generate_inference_windows_for_candidate(candidate, window_config)

    if not windows:
        logger.debug(f"No windows generated for candidate {candidate_id}")
        return []

    logger.debug(f"Generated {len(windows)} windows for candidate {candidate_id}")

    # 2. Run model predictions
    predictions = predict_windows(
        model=model,
        windows=windows,
        video_path=video_path,
        pose_track_df=pose_track_df,
        shelf_region=shelf_region,
        config=config,
        device=device,
    )

    if not predictions:
        return []

    # 3. Temporal smoothing
    smoothed = smooth_predictions(predictions, config.smoothing_window)

    # 4. Peak detection
    regions = detect_score_peaks(smoothed, config)

    # 5. Same-type merging
    merged = merge_same_type_regions(
        regions,
        config.same_type_merge_gap_s,
        config.min_event_duration_s,
    )

    # 6. Create event predictions
    events = create_event_predictions(
        merged,
        clip_id,
        candidate_id,
        actor_id,
        config,
    )

    logger.debug(
        f"Candidate {candidate_id}: {len(windows)} windows → "
        f"{len(regions)} peaks → {len(merged)} merged → {len(events)} events"
    )

    return events


# ============================================================
# BATCH INFERENCE
# ============================================================


def infer_all_candidates(
    model: VideoMAEClassifier,
    candidates_df: pd.DataFrame,
    video_dir: Path,
    pose_tracks_dir: Path,
    shelf_regions: dict[str, dict],
    clips_df: pd.DataFrame,
    config: InferenceConfig,
    device: torch.device,
) -> pd.DataFrame:
    """Run inference on all candidates.

    Args:
        model: Trained VideoMAE classifier.
        candidates_df: All candidates to process.
        video_dir: Directory with video files.
        pose_tracks_dir: Directory with pose tracks.
        shelf_regions: Dict mapping region_id to region config.
        clips_df: Clip metadata.
        config: Inference configuration.
        device: Device to run on.

    Returns:
        DataFrame with all predictions in canonical format.
    """
    logger.info(f"Running inference on {len(candidates_df)} candidates")

    all_predictions: list[dict] = []
    pose_cache: dict[str, pd.DataFrame] = {}

    for idx, candidate in candidates_df.iterrows():
        clip_id = candidate["clip_id"]
        actor_id = candidate["actor_id"]
        region_id = candidate.get("region_id")

        # Get video path
        video_path = _get_video_path(clip_id, video_dir, clips_df)
        if not video_path.exists():
            logger.warning(f"Video not found for clip {clip_id}: {video_path}")
            continue

        # Load pose track (with caching)
        cache_key = f"{clip_id}_{actor_id}"
        if cache_key not in pose_cache:
            pose_cache[cache_key] = _load_pose_track(clip_id, actor_id, pose_tracks_dir)
        pose_track_df = pose_cache[cache_key]

        # Get shelf region
        shelf_region = shelf_regions.get(region_id) if region_id else None

        # Run inference on this candidate
        events = infer_candidate(
            model=model,
            candidate=candidate,
            video_path=video_path,
            pose_track_df=pose_track_df,
            shelf_region=shelf_region,
            config=config,
            device=device,
        )

        # Collect predictions
        for event in events:
            all_predictions.append(event.to_dict())

        # Progress logging
        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx + 1}/{len(candidates_df)} candidates")

    # Create DataFrame
    if all_predictions:
        predictions_df = pd.DataFrame(all_predictions)
        logger.info(
            f"Inference complete: {len(predictions_df)} predictions from "
            f"{len(candidates_df)} candidates"
        )
    else:
        predictions_df = pd.DataFrame(
            columns=["pred_id", "clip_id", "type", "t_start", "t_end", "score", "model"]
        )
        logger.info("Inference complete: no events detected")

    return predictions_df


def _get_video_path(clip_id: str, video_dir: Path, clips_df: pd.DataFrame) -> Path:
    """Resolve video path from clip_id."""
    # Try to get path from clips_df
    if "s3_key" in clips_df.columns:
        matches = clips_df[clips_df["clip_id"] == clip_id]
        if not matches.empty:
            s3_key = matches.iloc[0]["s3_key"]
            filename = Path(s3_key).name
            video_path = video_dir / filename
            if video_path.exists():
                return video_path

    # Fallback: try common patterns
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        video_path = video_dir / f"{clip_id}{ext}"
        if video_path.exists():
            return video_path

    # Last resort
    return video_dir / f"{clip_id}.mp4"


def _load_pose_track(clip_id: str, actor_id: str, pose_tracks_dir: Path) -> pd.DataFrame:
    """Load pose track for actor."""
    pose_file = pose_tracks_dir / f"{clip_id}.parquet"

    if not pose_file.exists():
        logger.warning(f"Pose track not found: {pose_file}")
        return pd.DataFrame()

    try:
        pose_df = pd.read_parquet(pose_file)
        if "actor_id" in pose_df.columns:
            pose_df = pose_df[pose_df["actor_id"] == actor_id]
        return pose_df
    except Exception as e:
        logger.warning(f"Failed to load pose track {pose_file}: {e}")
        return pd.DataFrame()


# ============================================================
# OUTPUT
# ============================================================


def save_predictions(
    predictions_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Save predictions to canonical CSV format.

    Args:
        predictions_df: DataFrame with predictions.
        output_path: Output file path.

    Returns:
        Path where file was saved.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Select canonical columns (in order)
    canonical_columns = ["pred_id", "clip_id", "type", "t_start", "t_end", "score", "model"]
    available_columns = [col for col in canonical_columns if col in predictions_df.columns]

    predictions_df[available_columns].to_csv(output_path, index=False)
    logger.info(f"Saved {len(predictions_df)} predictions to {output_path}")

    return output_path


# ============================================================
# CLI ENTRY POINT
# ============================================================


def main(
    checkpoint_path: str,
    candidates_path: str,
    clips_path: str,
    video_dir: str,
    pose_tracks_dir: str,
    shelf_regions_path: str,
    output_path: str,
    config_path: str | None = None,
    pickup_threshold: float = 0.5,
    putdown_threshold: float = 0.5,
) -> None:
    """CLI entry point for inference.

    Args:
        checkpoint_path: Path to trained model checkpoint.
        candidates_path: Path to candidates.parquet.
        clips_path: Path to clips.csv.
        video_dir: Directory with video files.
        pose_tracks_dir: Directory with pose tracks.
        shelf_regions_path: Path to shelf regions YAML.
        output_path: Output predictions CSV path.
        config_path: Optional config YAML path.
        pickup_threshold: Threshold for pickup detection.
        putdown_threshold: Threshold for putdown detection.
    """
    import yaml

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("=" * 60)
    logger.info("Track B1 Inference")
    logger.info("=" * 60)

    # Load config
    config = InferenceConfig(
        pickup_threshold=pickup_threshold,
        putdown_threshold=putdown_threshold,
    )

    if config_path is not None:
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    # Load data
    logger.info("Loading data...")
    candidates_df = pd.read_parquet(candidates_path)
    clips_df = pd.read_csv(clips_path)

    # Load shelf regions
    shelf_regions = load_shelf_regions(Path(shelf_regions_path))

    # Load model
    logger.info(f"Loading model from {checkpoint_path}")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    checkpoint_data = load_checkpoint(checkpoint_path, device=str(device))
    model = checkpoint_data["model"]
    model.eval()

    logger.info(f"Model loaded, device={device}")
    logger.info(
        f"Thresholds: pickup={config.pickup_threshold}, putdown={config.putdown_threshold}"
    )

    # Run inference
    predictions_df = infer_all_candidates(
        model=model,
        candidates_df=candidates_df,
        video_dir=Path(video_dir),
        pose_tracks_dir=Path(pose_tracks_dir),
        shelf_regions=shelf_regions,
        clips_df=clips_df,
        config=config,
        device=device,
    )

    # Save predictions
    save_predictions(predictions_df, Path(output_path))

    logger.info("=" * 60)
    logger.info("Inference complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    import typer

    typer.run(main)
