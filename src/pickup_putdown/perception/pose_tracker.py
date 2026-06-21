"""Pose inference on person-active spans for Layer 0B.

Runs a configured YOLO pose model over person-active spans at a configurable
target FPS.  Preserves source timestamps, clamps to clip duration, and records
missing or low-confidence keypoints rather than interpolating them silently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from pickup_putdown.common.schemas import ActiveSpan, PoseObservation
from pickup_putdown.config import PoseConfig
from pickup_putdown.ingestion.video_probe import probe_video

logger = logging.getLogger(__name__)


@dataclass
class PoseTracker:
    """Run YOLO pose detection on a single video file.

    Parameters
    ----------
    video_path : Path
        Path to the source MP4/video file.
    pose_cfg : PoseConfig
        Pose inference configuration.
    active_spans : list[ActiveSpan] | None
        Active spans to restrict pose inference to.  When *None* the entire
        clip is processed.
    """

    video_path: Path
    pose_cfg: PoseConfig
    active_spans: list[ActiveSpan] | None = None

    _model: object | None = field(default=None, repr=False)
    _source_fps: float = field(default=0.0, repr=False)
    _total_frames: int = field(default=0, repr=False)
    _clip_duration_s: float = field(default=0.0, repr=False)
    _vid_stride: int = field(default=1, repr=False)
    _sample_frames: list[int] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self._source_fps = self._read_source_fps()
        self._total_frames = self._read_total_frames()
        self._clip_duration_s = self._total_frames / max(self._source_fps, 1e-9)
        self._vid_stride = max(1, round(self._source_fps / max(self.pose_cfg.target_fps, 1e-9)))
        self._sample_frames = self._compute_sample_frames(self._total_frames, self._vid_stride)
        logger.info(
            "Video %s: fps=%.2f, frames=%d, stride=%d, samples=%d",
            self.video_path,
            self._source_fps,
            self._total_frames,
            self._vid_stride,
            len(self._sample_frames),
        )

    # ------------------------------------------------------------------
    # Video metadata helpers
    # ------------------------------------------------------------------

    def _read_source_fps(self) -> float:
        result = probe_video(self.video_path)
        if not result.decode_ok:
            raise RuntimeError(f"Cannot decode video: {result.probe_error}")
        fps = result.fps or result.probe_fps
        if fps is None or fps <= 0:
            raise RuntimeError(f"Could not determine FPS for {self.video_path}")
        return float(fps)

    def _read_total_frames(self) -> int:
        import cv2

        cap = cv2.VideoCapture(str(self.video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total <= 0:
            raise RuntimeError(f"Could not determine frame count for {self.video_path}")
        return total

    @staticmethod
    def _compute_sample_frames(total_frames: int, vid_stride: int) -> list[int]:
        return list(range(0, total_frames, vid_stride))

    # ------------------------------------------------------------------
    # Active-span filtering
    # ------------------------------------------------------------------

    def _active_frame_indices(self) -> set[int]:
        """Return source frame indices that fall within active spans."""
        if not self.active_spans:
            return set(self._sample_frames)

        active_frames: set[int] = set()
        for span in self.active_spans:
            start_frame = max(0, int(span.t_start * self._source_fps))
            end_frame = min(self._total_frames, int(span.t_end * self._source_fps) + 1)
            for fi in self._sample_frames:
                if start_frame <= fi < end_frame:
                    active_frames.add(fi)
        return active_frames if active_frames else set(self._sample_frames)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> object:
        from ultralytics import YOLO

        if self._model is not None:
            return self._model

        device = self.pose_cfg.device
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading YOLO pose model from %s on %s", self.pose_cfg.model_path, device)
        model = YOLO(self.pose_cfg.model_path)
        self._model = model
        return model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> list[PoseObservation]:
        """Run pose inference and return timestamped wrist observations.

        Returns
        -------
        list[PoseObservation]
            One row per (actor, timestamp, hand_side).  Deterministically
            sorted by (clip_id, timestamp_s, actor_id, hand_side).
        """
        import cv2

        model = self._load_model()
        cap = cv2.VideoCapture(str(self.video_path))

        active_frames = self._active_frame_indices()
        clip_id = self._extract_clip_id()

        all_observations: list[PoseObservation] = []
        # Track which pose detections have been assigned to avoid duplicates.
        # Key: (sample_index, pose_box_index)
        assigned: set[tuple[int, int]] = set()

        for si, src_frame_idx in enumerate(self._sample_frames):
            if src_frame_idx not in active_frames:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, src_frame_idx)
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame %d of %s", src_frame_idx, self.video_path)
                continue

            timestamp_s = src_frame_idx / self._source_fps
            timestamp_s = min(timestamp_s, self._clip_duration_s)

            results = model.track(
                frame,
                persist=True,
                conf=self.pose_cfg.pose_confidence,
                max_det=self.pose_cfg.max_detections,
                classes=[0],  # person class
                verbose=False,
                imgsz=self.pose_cfg.image_size,
                half=self.pose_cfg.half,
            )

            boxes = results[0].boxes
            if boxes is None:
                continue

            n_dets = len(boxes.xyxy) if boxes.xyxy is not None else 0
            if n_dets == 0:
                continue

            for det_idx in range(n_dets):
                track_id = int(boxes.id[det_idx].item()) if boxes.id is not None else None
                conf = float(boxes.conf[det_idx].item())
                xyxy = boxes.xyxy[det_idx].tolist()

                pose_key = (si, det_idx)
                if pose_key in assigned:
                    continue

                keypoints = self._extract_keypoints(results[0], det_idx)
                assigned.add(pose_key)

                actor_id = f"actor_{track_id}" if track_id is not None else "actor_untracked"

                for hand_side, kp_key in [("left", "left_wrist"), ("right", "right_wrist")]:
                    kp = keypoints.get(kp_key)
                    if kp is None:
                        continue
                    kx, ky, kconf = kp
                    if kconf < self.pose_cfg.pose_confidence:
                        continue

                    obs = PoseObservation(
                        clip_id=clip_id,
                        timestamp_s=timestamp_s,
                        source_frame_index=src_frame_idx,
                        sample_index=si,
                        actor_id=actor_id,
                        hand_side=hand_side,
                        wrist_x=kx,
                        wrist_y=ky,
                        wrist_confidence=kconf,
                        person_bbox_x1=xyxy[0],
                        person_bbox_y1=xyxy[1],
                        person_bbox_x2=xyxy[2],
                        person_bbox_y2=xyxy[3],
                        pose_association_confidence=conf,
                        is_valid=True,
                    )
                    all_observations.append(obs)

        cap.release()

        all_observations.sort(key=lambda o: (o.clip_id, o.timestamp_s, o.actor_id, o.hand_side))

        logger.info(
            "Pose complete: %d wrist observations from %d frames",
            len(all_observations),
            len(active_frames),
        )
        return all_observations

    @staticmethod
    def _extract_keypoints(results, det_idx: int) -> dict[str, tuple[float, float, float]]:
        """Extract left/right wrist from YOLO pose keypoints.

        YOLO pose keypoints are stored as [x, y, conf] per keypoint.
        We map the first two available keypoints to left/right wrists.
        """
        keypoints_data = {}
        try:
            kpts = results[0].boxes.keypoints
            if kpts is None:
                return keypoints_data

            kp_tensor = kpts.x if hasattr(kpts, "x") else None
            conf_tensor = kpts.conf if hasattr(kpts, "conf") else None

            if kp_tensor is None or det_idx >= len(kp_tensor):
                return keypoints_data

            kp_x = kp_tensor[det_idx].cpu().numpy().tolist()
            kp_y = kpts.y[det_idx].cpu().numpy().tolist() if hasattr(kpts, "y") else kp_x
            kp_conf = (
                conf_tensor[det_idx].cpu().numpy().tolist()
                if conf_tensor is not None
                else [1.0] * len(kp_x)
            )

            # YOLO pose: index 0 = nose, 5 = left_wrist, 6 = right_wrist
            wrist_map = {5: "left_wrist", 6: "right_wrist"}
            for idx, side in wrist_map.items():
                if idx < len(kp_x):
                    keypoints_data[side] = (
                        float(kp_x[idx]),
                        float(kp_y[idx]) if len(kp_y) > idx else 0.0,
                        float(kp_conf[idx]) if len(kp_conf) > idx else 0.0,
                    )
        except Exception:
            pass

        return keypoints_data

    def _extract_clip_id(self) -> str:
        stem = self.video_path.stem
        return f"clip_{stem}"
