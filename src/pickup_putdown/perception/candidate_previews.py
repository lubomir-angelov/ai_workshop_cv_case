"""Candidate preview rendering with overlays for actor boxes, wrists, and regions.

Each preview visibly identifies:
- candidate_id, actor_id, hand_side, region_id
- raw interval and padded interval
- actor bounding box
- wrist positions when valid
- configured region and expanded proposal region

Preview rendering failures are reported without corrupting machine-readable
candidate output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pickup_putdown.common.schemas import Candidate, PoseObservation
from pickup_putdown.perception.shelf_regions import Polygon

logger = logging.getLogger(__name__)

# Colors (BGR)
COLOR_ACTOR_BOX = (0, 255, 0)  # Green
COLOR_WRIST_LEFT = (255, 0, 0)  # Blue
COLOR_WRIST_RIGHT = (0, 165, 255)  # Orange
COLOR_REGION = (0, 128, 255)  # Yellow-orange
COLOR_EXPANDED_REGION = (128, 0, 255)  # Purple
COLOR_RAW_INTERVAL = (255, 255, 0)  # Cyan
COLOR_PADDED_INTERVAL = (0, 255, 255)  # Lime
COLOR_TEXT = (255, 255, 255)  # White
COLOR_BG = (0, 0, 0)  # Black


@dataclass
class CandidateOverlayConfig:
    """Configuration for candidate preview overlays."""

    draw_actor_box: bool = True
    draw_wrist_positions: bool = True
    draw_region_polygons: bool = True
    draw_region_labels: bool = True
    draw_candidate_intervals: bool = True
    text_scale: float = 0.5
    line_thickness: int = 2
    max_output_width: int = 1280
    max_output_height: int = 720
    preview_fps: float = 4.0


def render_candidate_preview(
    video_path: Path,
    candidate: Candidate,
    pose_observations: list[PoseObservation],
    original_polygon: Polygon,
    expanded_polygon: Polygon,
    output_path: Path,
    config: CandidateOverlayConfig | None = None,
    clip_duration_s: float | None = None,
) -> Path:
    """Render a preview clip for a single candidate.

    Parameters
    ----------
    video_path : Path
        Path to the source video.
    candidate : Candidate
        The candidate to preview.
    pose_observations : list[PoseObservation]
        Pose observations for the candidate's clip/actor/hand/region.
    original_polygon : Polygon
        Original shelf/surface region polygon.
    expanded_polygon : Polygon
        Expanded proposal region polygon.
    output_path : Path
        Destination for the preview MP4.
    config : CandidateOverlayConfig | None
        Overlay rendering configuration.
    clip_duration_s : float | None
        Source clip duration for clamping.

    Returns
    -------
    Path
        Path to the rendered preview.
    """
    import cv2

    if config is None:
        config = CandidateOverlayConfig()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        logger.warning(
            "Cannot open video for candidate %s: %s", candidate.candidate_id, video_path
        )
        return output_path

    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if source_fps <= 0:
            raise RuntimeError(f"Invalid source FPS for preview: {video_path}")

        if total_frames <= 0:
            raise RuntimeError(f"No frames in source video: {video_path}")

        clip_dur = clip_duration_s or (total_frames / max(source_fps, 1e-9))

        output_width, output_height = _compute_output_size(source_width, source_height, config)
        scale_x = output_width / source_width
        scale_y = output_height / source_height

        # Frame range for this candidate (padded interval)
        start_frame = max(0, int(candidate.window_start_s * source_fps))
        end_frame = min(total_frames, int(candidate.window_end_s * source_fps) + 1)
        if end_frame <= start_frame:
            end_frame = start_frame + 1

        output_fps = min(config.preview_fps, source_fps)
        if output_fps <= 0:
            output_fps = 1.0

        writer = _open_video_writer(output_path, output_fps, (output_width, output_height))

        rendered_frames = 0
        for frame_idx in range(start_frame, end_frame):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = capture.read()
            if not success:
                continue

            if frame.shape[1] != output_width or frame.shape[0] != output_height:
                frame = cv2.resize(
                    frame, (output_width, output_height), interpolation=cv2.INTER_AREA
                )

            frame_ts = frame_idx / source_fps
            overlay = frame.copy()

            # Draw regions
            if config.draw_region_polygons:
                _draw_polygon(
                    overlay,
                    original_polygon,
                    COLOR_REGION,
                    config.line_thickness,
                    scale_x,
                    scale_y,
                )
                _draw_polygon(
                    overlay,
                    expanded_polygon,
                    COLOR_EXPANDED_REGION,
                    config.line_thickness,
                    scale_x,
                    scale_y,
                )

            if config.draw_region_labels:
                _draw_label(overlay, f"region={candidate.region_id}", 8, 8 + 16, config)
                _draw_label(
                    overlay,
                    "expanded_margin=+",
                    8,
                    24 + 16,
                    config,
                )

            # Draw actor box (from pose observations at this frame)
            if config.draw_actor_box:
                actor_obs = [
                    o
                    for o in pose_observations
                    if o.source_frame_index == frame_idx
                    and o.clip_id == candidate.clip_id
                    and o.actor_id == candidate.actor_id
                ]
                for ao in actor_obs:
                    if ao.person_bbox_x1 is not None:
                        ax1 = int(ao.person_bbox_x1 * scale_x)
                        ay1 = int(ao.person_bbox_y1 * scale_y)
                        ax2 = int(ao.person_bbox_x2 * scale_x)
                        ay2 = int(ao.person_bbox_y2 * scale_y)
                        cv2.rectangle(
                            overlay, (ax1, ay1), (ax2, ay2), COLOR_ACTOR_BOX, config.line_thickness
                        )

            # Draw wrist positions
            if config.draw_wrist_positions:
                wrist_obs = [
                    o
                    for o in pose_observations
                    if o.source_frame_index == frame_idx
                    and o.clip_id == candidate.clip_id
                    and o.actor_id == candidate.actor_id
                ]
                for wo in wrist_obs:
                    wx = int(wo.wrist_x * scale_x)
                    wy = int(wo.wrist_y * scale_y)
                    color = COLOR_WRIST_LEFT if wo.hand_side == "left" else COLOR_WRIST_RIGHT
                    cv2.circle(overlay, (wx, wy), 5, color, -1)

            # Draw interval indicators
            if config.draw_candidate_intervals:
                raw_y = source_height - 40
                raw_start_x = int((candidate.raw_start_s / max(clip_dur, 1e-9)) * output_width)
                raw_end_x = int((candidate.raw_end_s / max(clip_dur, 1e-9)) * output_width)
                padded_start_x = int(
                    (candidate.window_start_s / max(clip_dur, 1e-9)) * output_width
                )
                padded_end_x = int((candidate.window_end_s / max(clip_dur, 1e-9)) * output_width)

                # Padded interval
                cv2.rectangle(
                    overlay,
                    (padded_start_x, raw_y),
                    (padded_end_x, raw_y + 12),
                    COLOR_PADDED_INTERVAL,
                    -1,
                )
                # Raw interval
                cv2.rectangle(
                    overlay,
                    (raw_start_x, raw_y + 14),
                    (raw_end_x, raw_y + 26),
                    COLOR_RAW_INTERVAL,
                    -1,
                )

            # Title bar
            title = (
                f"CANDIDATE: {candidate.candidate_id}  |  "
                f"ACTOR: {candidate.actor_id}  |  "
                f"HAND: {candidate.hand_side}  |  "
                f"REGION: {candidate.region_id}  |  "
                f"RAW: [{candidate.raw_start_s:.2f}, {candidate.raw_end_s:.2f}]  |  "
                f"PADDED: [{candidate.window_start_s:.2f}, {candidate.window_end_s:.2f}]"
            )
            _draw_label(overlay, title, 8, 8, config, multiline=True)

            writer.write(overlay)
            rendered_frames += 1

        if rendered_frames == 0:
            logger.warning("No frames rendered for candidate %s", candidate.candidate_id)
            return output_path

    except Exception as exc:
        logger.warning("Preview render failed for candidate %s: %s", candidate.candidate_id, exc)
        return output_path
    finally:
        capture.release()
        if "writer" in locals():
            writer.release()

    logger.info(
        "Candidate preview rendered: %s (%d frames)",
        output_path,
        rendered_frames,
    )
    return output_path


def _compute_output_size(
    source_width: int, source_height: int, config: CandidateOverlayConfig
) -> tuple[int, int]:
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"Invalid source dimensions: {source_width}x{source_height}")

    width_scale = config.max_output_width / source_width
    height_scale = config.max_output_height / source_height
    scale = min(1.0, width_scale, height_scale)

    output_width = max(2, int(round(source_width * scale)))
    output_height = max(2, int(round(source_height * scale)))

    if output_width % 2:
        output_width -= 1
    if output_height % 2:
        output_height -= 1

    return output_width, output_height


def _draw_polygon(
    frame: np.ndarray,
    polygon: Polygon,
    color: tuple[int, int, int],
    thickness: int,
    scale_x: float,
    scale_y: float,
) -> None:
    import cv2

    pts = []
    for x, y in polygon:
        sx = min(max(0, int(x * scale_x)), frame.shape[1] - 1)
        sy = min(max(0, int(y * scale_y)), frame.shape[0] - 1)
        pts.append((sx, sy))

    if len(pts) >= 3:
        pts_np = np.array(pts, dtype=np.int32)
        pts_np = pts_np.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts_np], True, color, thickness)


def _draw_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    config: CandidateOverlayConfig,
    *,
    multiline: bool = False,
) -> None:
    import cv2

    height, width = frame.shape[:2]
    (tw, th), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        config.text_scale,
        config.line_thickness,
    )

    padding = 4
    bx1 = min(max(0, x), width - 1)
    by1 = max(0, y - th - 2 * padding) if not multiline else max(0, y - th - 2 * padding)
    bx2 = min(width, bx1 + tw + 2 * padding)
    by2 = min(height, by1 + th + 2 * padding + baseline)

    cv2.rectangle(frame, (bx1, by1), (bx2, by2), COLOR_BG, -1)
    cv2.putText(
        frame,
        text,
        (bx1 + padding, by1 + th + padding + baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        config.text_scale,
        COLOR_TEXT,
        config.line_thickness,
        cv2.LINE_AA,
    )


def _open_video_writer(output_path: Path, output_fps: float, output_size: tuple[int, int]):
    import cv2

    for codec in ("mp4v", "avc1"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, output_size)
        if writer.isOpened():
            return writer
        writer.release()

    raise RuntimeError(f"Could not open video writer for {output_path}")
