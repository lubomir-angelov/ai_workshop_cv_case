"""Pipelined person tracker with multiprocessed frame decoding.

This module provides PipelinedPersonTracker, a drop-in replacement for
PersonTracker that uses multiple decoder worker processes to decode frames
in parallel. This improves GPU utilization by allowing the GPU to process
frames while the CPU is decoding subsequent frames.

Key features:
- Backward compatible with PersonTracker API
- Multiple decoder workers (configurable, default 2)
- Frames resized to 640x640 in decoder workers (~20MB shared memory vs 300MB for 4K)
- Frame reordering for sequential ByteTrack processing
- Graceful shutdown and error handling
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from pickup_putdown.common.schemas import PersonObservation, TrackSummary
from pickup_putdown.config import AppConfig, TriageConfig
from pickup_putdown.perception.frame_pipeline import DecoderPool, FrameReorderer
from pickup_putdown.perception.person_tracker import PersonTracker

logger = logging.getLogger(__name__)


@dataclass
class PipelinedPersonTracker(PersonTracker):
    """YOLO person detection + ByteTrack with pipelined frame decoding.

    This class extends PersonTracker to use multiple decoder worker processes
    for parallel frame decoding. The main process consumes frames in order
    and runs YOLO inference sequentially (required by ByteTrack).

    Parameters
    ----------
    video_path : Path
        Path to the source MP4/video file.
    triage_cfg : TriageConfig
        Triage configuration with detection and acceptance thresholds.
    tracker_cfg : dict | None
        ByteTrack configuration overrides. If None, uses defaults.
    app_cfg : AppConfig | None
        Full application config (used to resolve tracker config path).
    use_pipeline : bool
        Whether to use pipelined decoding. If False, falls back to
        sequential decoding (same as PersonTracker).
    """

    use_pipeline: bool = True

    _decoder_pool: DecoderPool | None = field(default=None, repr=False)

    def run(self) -> tuple[list[PersonObservation], list[TrackSummary]]:
        """Run person detection and tracking on the video.

        Returns
        -------
        observations : list[PersonObservation]
            Flat list of timestamped person detections.
        summaries : list[TrackSummary]
            Per-tracker summaries with stability flags.
        """
        if not self.use_pipeline:
            return super().run()

        return self._run_pipelined()

    def _run_pipelined(self) -> tuple[list[PersonObservation], list[TrackSummary]]:
        """Run person detection with pipelined frame decoding."""
        import cv2

        model = self._load_model()
        clip_id = self._extract_clip_id()

        all_observations: list[PersonObservation] = []
        track_points: dict[int | str, list[tuple[int, float, list[float], float]]] = {}

        # Get pipeline config from triage_cfg or use defaults
        n_workers = getattr(self.triage_cfg, "pipeline_n_decoders", 2)
        queue_depth = getattr(self.triage_cfg, "pipeline_queue_depth", 16)
        resize_frames = getattr(self.triage_cfg, "pipeline_resize_frames", True)
        frame_size = getattr(self.triage_cfg, "pipeline_frame_size", (640, 640))
        timeout = getattr(self.triage_cfg, "pipeline_frame_timeout_s", 10.0)

        frame_height, frame_width = frame_size

        # If not resizing, we need original frame dimensions
        if not resize_frames:
            cap = cv2.VideoCapture(str(self.video_path))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

        logger.info(
            "Starting pipelined tracking: %d workers, queue_depth=%d, frame_size=%dx%d",
            n_workers,
            queue_depth,
            frame_width,
            frame_height,
        )

        # Create decoder pool
        decoder_pool = DecoderPool(
            n_workers=n_workers,
            queue_depth=queue_depth,
            frame_height=frame_height,
            frame_width=frame_width,
            resize_frames=resize_frames,
        )

        # Create frame reorderer
        reorderer = FrameReorderer(total_frames=len(self._sample_frames))

        # Track completed workers
        completed_workers = 0

        try:
            # Start decoder workers
            decoder_pool.start(
                video_path=self.video_path,
                sample_frames=self._sample_frames,
                source_fps=self._source_fps,
            )

            # Process frames
            while completed_workers < n_workers or reorderer.n_pending > 0:
                # Get next frame from any worker
                result = decoder_pool.get_frame(timeout=timeout)

                if result is None:
                    # A worker has finished
                    completed_workers += 1
                    logger.debug(
                        "Worker completed (%d/%d done)",
                        completed_workers,
                        n_workers,
                    )
                    continue

                frame, metadata = result

                # Add to reorderer and process any frames now in order
                ready_frames = reorderer.add_frame(frame, metadata)

                for ready_frame, ready_meta in ready_frames:
                    # Run YOLO inference on this frame
                    self._process_frame(
                        model=model,
                        frame=ready_frame,
                        sample_index=ready_meta.sample_index,
                        source_frame_index=ready_meta.source_frame_index,
                        timestamp_s=ready_meta.timestamp_s,
                        clip_id=clip_id,
                        all_observations=all_observations,
                        track_points=track_points,
                    )

                    # Return the slot for reuse
                    decoder_pool.return_slot(ready_meta.slot_index)

        except Exception as e:
            logger.error("Pipelined tracking error: %s", e)
            raise
        finally:
            decoder_pool.stop()

        # Mark stable tracks
        summaries = self._compute_summaries(clip_id, track_points)
        stable_tracker_ids = {s.tracker_track_id for s in summaries if s.is_stable}
        for obs in all_observations:
            tid = obs.tracker_track_id if obs.tracker_track_id is not None else -1
            obs.is_stable = tid in stable_tracker_ids

        # Sort observations deterministically
        all_observations.sort(
            key=lambda o: (o.source_frame_index, o.sample_index, o.tracker_track_id or -1)
        )

        logger.info(
            "Pipelined tracking complete: %d observations, %d tracks (%d stable)",
            len(all_observations),
            len(summaries),
            sum(1 for s in summaries if s.is_stable),
        )

        return all_observations, summaries

    def _process_frame(
        self,
        model: object,
        frame,
        sample_index: int,
        source_frame_index: int,
        timestamp_s: float,
        clip_id: str,
        all_observations: list[PersonObservation],
        track_points: dict[int | str, list[tuple[int, float, list[float], float]]],
    ) -> None:
        """Process a single frame through YOLO inference.

        Parameters
        ----------
        model : object
            YOLO model instance.
        frame : np.ndarray
            Frame data (BGR, uint8).
        sample_index : int
            Index in the sample_frames list.
        source_frame_index : int
            Original frame number in the video.
        timestamp_s : float
            Timestamp in seconds.
        clip_id : str
            Clip identifier.
        all_observations : list[PersonObservation]
            List to append observations to.
        track_points : dict
            Dictionary tracking per-track observation data.
        """
        results = model.track(
            frame,
            persist=True,
            conf=self.triage_cfg.detector_confidence,
            iou=self.triage_cfg.detector_iou_threshold,
            max_det=self.triage_cfg.max_detections,
            classes=[0],  # person class
            verbose=False,
            imgsz=self.triage_cfg.image_size,
            half=self.triage_cfg.half and self._device.startswith("cuda"),
            device=self._device,
        )

        boxes = results[0].boxes
        if boxes is None:
            return

        n_dets = len(boxes.xyxy) if boxes.xyxy is not None else 0
        if n_dets == 0:
            return

        for i in range(n_dets):
            track_id = int(boxes.id[i].item()) if boxes.id is not None else None
            conf = float(boxes.conf[i].item())

            if conf < self.triage_cfg.minimum_track_confidence:
                continue

            xyxy = boxes.xyxy[i].tolist()
            bbox = [float(v) for v in xyxy]

            obs = PersonObservation(
                clip_id=clip_id,
                person_track_id=f"{clip_id}:person:{track_id}"
                if track_id is not None
                else f"{clip_id}:person:untracked",
                tracker_track_id=track_id,
                sample_index=sample_index,
                source_frame_index=source_frame_index,
                timestamp_s=timestamp_s,
                bbox_x1=bbox[0],
                bbox_y1=bbox[1],
                bbox_x2=bbox[2],
                bbox_y2=bbox[3],
                confidence=conf,
                is_stable=False,
            )
            all_observations.append(obs)

            tid = track_id if track_id is not None else -1
            if tid not in track_points:
                track_points[tid] = []
            track_points[tid].append((sample_index, timestamp_s, bbox, conf))
