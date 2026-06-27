"""Frame extraction and VLM input preparation for Layer 2 windows.

Extracts evenly-spaced frames from a video clip for each window,
records metadata (frame indices, timestamps, sampling rate, dimensions),
and returns base64-encoded frames suitable for the shared VLM client.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass

import cv2

from pickup_putdown.layer2.window_generator import Window

logger = logging.getLogger(__name__)


@dataclass
class FrameInfo:
    """Metadata for one extracted frame."""

    frame_index: int
    relative_timestamp_s: float
    source_timestamp_s: float


@dataclass
class WindowRender:
    """Rendered input for one window ready for VLM inference."""

    window: Window
    clip_id: str
    video_path: str
    frames: list[bytes]  # raw JPEG bytes
    frame_infos: list[FrameInfo]
    selected_frame_indices: list[int]
    relative_timestamps_s: list[float]
    source_timestamps_s: list[float]
    sampling_rate_fps: float
    render_width: int
    render_height: int
    clip_fps: float
    clip_duration_s: float


def render_window(
    window: Window,
    video_path: str,
    clip_fps: float,
    clip_duration_s: float,
    *,
    max_frame_width: int = 640,
    n_frames: int = 10,
) -> WindowRender | None:
    """Extract frames for one window and prepare VLM input.

    Returns None if the video cannot be opened or no frames can be extracted.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return None

    try:
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or clip_fps
        actual_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(actual_fps, 0.001)

        # Clamp window to actual clip duration
        w_start = max(0.0, window.window_start_s)
        w_end = min(window.window_end_s, clip_duration_s, actual_duration)

        if w_end <= w_start:
            logger.warning(
                "Window %s out of bounds [%s, %s] in clip duration %s",
                window.window_id,
                w_start,
                w_end,
                clip_duration_s,
            )
            return None

        # Compute frame indices within the window
        window_duration = w_end - w_start
        if window_duration <= 0:
            return None

        # Determine sampling: evenly space n_frames across the window
        step = window_duration / max(n_frames, 1)
        selected_indices: list[int] = []
        frame_infos: list[FrameInfo] = []

        for i in range(n_frames):
            t_rel = i * step  # relative timestamp within window
            t_abs = w_start + t_rel  # absolute timestamp in video
            frame_idx = int(round(t_abs * actual_fps))

            # Clamp to valid range
            frame_idx = max(0, min(frame_idx, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Resize for VLM input
            h, w = frame.shape[:2]
            if w > max_frame_width:
                scale = max_frame_width / w
                new_w = max_frame_width
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))

            # Overlay visible timestamp and frame number for VLM reference
            overlay_text = f"frame={frame_idx}  t={t_rel:.1f}s"
            cv2.putText(
                frame,
                overlay_text,
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames_bytes = buf.tobytes()

            selected_indices.append(frame_idx)
            frame_infos.append(
                FrameInfo(
                    frame_index=frame_idx,
                    relative_timestamp_s=round(t_rel, 3),
                    source_timestamp_s=round(t_abs, 3),
                )
            )

        if not selected_indices:
            return None

        # Resize first frame to get render dimensions
        _, first_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        first_frame = cv2.imdecode(first_buf, cv2.IMREAD_COLOR)
        rh, rw = first_frame.shape[:2]

        return WindowRender(
            window=window,
            clip_id=window.clip_id,
            video_path=video_path,
            frames=[frames_bytes],  # keep as list for multi-frame support
            frame_infos=frame_infos,
            selected_frame_indices=selected_indices,
            relative_timestamps_s=[round(fi.relative_timestamp_s, 3) for fi in frame_infos],
            source_timestamps_s=[round(fi.source_timestamp_s, 3) for fi in frame_infos],
            sampling_rate_fps=round(1.0 / step, 2) if step > 0 else clip_fps,
            render_width=rw,
            render_height=rh,
            clip_fps=clip_fps,
            clip_duration_s=clip_duration_s,
        )
    finally:
        cap.release()


def frames_to_base64(frames: list[bytes]) -> list[str]:
    """Convert raw JPEG bytes to base64 strings for VLM API."""
    return [base64.b64encode(f).decode("ascii") for f in frames]


def render_all_windows(
    windows: list[Window],
    video_path: str,
    clip_fps: float,
    clip_duration_s: float,
    *,
    max_frame_width: int = 640,
    n_frames: int = 10,
) -> list[WindowRender]:
    """Render all windows, skipping failures."""
    results: list[WindowRender] = []
    for w in windows:
        render = render_window(
            w,
            video_path,
            clip_fps,
            clip_duration_s,
            max_frame_width=max_frame_width,
            n_frames=n_frames,
        )
        if render is not None:
            results.append(render)
    return results
