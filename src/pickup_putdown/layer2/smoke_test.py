#!/usr/bin/env python3
"""Smoke test for multimodal Layer 2 inference via real llama.cpp endpoint.

Usage:
    python -m pickup_putdown.layer2.smoke_test /path/to/candidate.mp4
    python -m pickup_putdown.layer2.smoke_test /path/to/candidate.mp4 --output-dir /tmp/smoke_out

Requirements:
    - A running llama.cpp endpoint at http://127.0.0.1:8000
    - One candidate MP4 video file
    - cv2 (opencv-python) for frame extraction

This script:
    1. Renders up to 8 timestamped frames from the video.
    2. Creates one Layer 2 window covering the full clip.
    3. Calls the real endpoint via the multimodal Qwen client.
    4. Prints attempts, validation errors, and the validated result.
    5. Writes outputs to the specified output directory.

It does NOT modify any annotation dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import cv2

from pickup_putdown.layer2.qwen_client import QwenClientConfig, call_qwen
from pickup_putdown.layer2.window_generator import Window

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("smoke_test")

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
MAX_FRAMES = 8


def render_frames(video_path: str, output_dir: str) -> list[str]:
    """Extract frames from video, save as JPEG, return file paths."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / max(fps, 0.001)

    os.makedirs(output_dir, exist_ok=True)

    # Sample evenly-spaced frames
    step = max(total_frames // MAX_FRAMES, 1)
    frame_paths: list[str] = []

    for i in range(MAX_FRAMES):
        frame_idx = min(i * step, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Resize for VLM input
        h, w = frame.shape[:2]
        max_width = 640
        if w > max_width:
            scale = max_width / w
            frame = cv2.resize(frame, (max_width, int(h * scale)))

        # Overlay timestamp
        t_rel = frame_idx / max(fps, 0.001)
        cv2.putText(
            frame,
            f"frame={frame_idx}  t={t_rel:.1f}s",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        frame_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_paths.append(frame_path)
        logger.info("Saved frame %d -> %s", i, frame_path)

    cap.release()
    return frame_paths, fps, duration_s


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test: multimodal Layer 2 inference")
    parser.add_argument("video_path", help="Path to candidate MP4 video")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for frames and results (default: /tmp/smoke_test_<uuid>)",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"llama.cpp endpoint base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model ID (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    video_path = os.path.abspath(args.video_path)
    if not os.path.isfile(video_path):
        logger.error("Video not found: %s", video_path)
        sys.exit(1)

    output_dir = args.output_dir or f"/tmp/smoke_test_{Path(video_path).stem}"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Render frames
    logger.info("Rendering frames from %s", video_path)
    frame_paths, fps, duration_s = render_frames(video_path, output_dir)
    if not frame_paths:
        logger.error("No frames extracted")
        sys.exit(1)

    logger.info("Extracted %d frames", len(frame_paths))

    # 2. Create one Layer 2 window
    clip_id = Path(video_path).stem
    window = Window(
        clip_id=clip_id,
        active_span_id="smoke",
        window_id="smoke_w0000",
        window_start_s=0.0,
        window_end_s=duration_s,
        duration_s=duration_s,
        overlap_s=0.0,
        source_timestamp_s=0.0,
    )

    # 3. Call the real endpoint
    qwen_config = QwenClientConfig(
        model_id=args.model,
        base_url=args.base_url,
        max_tokens=2048,
        retry_max_tokens=4096,
        max_attempts=2,
        timeout_s=300,
    )

    logger.info("Calling Layer 2 Qwen client (endpoint=%s, model=%s)", args.base_url, args.model)
    result = call_qwen(
        window_id=window.window_id,
        clip_id=window.clip_id,
        window_start_s=window.window_start_s,
        window_end_s=window.window_end_s,
        frame_paths=frame_paths,
        fps=fps,
        qwen_config=qwen_config,
    )

    # 4. Print results
    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)
    print(f"Window: {result.window_id}")
    print(f"Clip:   {result.clip_id}")
    print(f"Frames: {len(frame_paths)}")
    print(f"Attempts: {len(result.attempts)}")

    for i, attempt in enumerate(result.attempts, 1):
        print(f"\n--- Attempt {i} ---")
        print(f"  Success: {attempt.is_success}")
        print(f"  Validation errors: {attempt.validation_errors or 'none'}")
        if attempt.validation_errors:
            for err in attempt.validation_errors:
                print(f"    - {err}")

    if result.validated_response:
        print(f"\nValidated events: {len(result.validated_response.events)}")
        for ev in result.validated_response.events:
            print(
                f"  {ev.event_type}: [{ev.relative_start_s:.1f}s - {ev.relative_end_s:.1f}s] "
                f"items={ev.item_count} conf={ev.confidence:.2f} vis={ev.visibility}"
            )
    else:
        print(f"\nNo validated response. Error: {result.error}")

    # 5. Write outputs
    output_file = os.path.join(output_dir, "smoke_result.json")
    output_data = {
        "window_id": result.window_id,
        "clip_id": result.clip_id,
        "frame_paths": frame_paths,
        "attempts": [
            {
                "attempt_number": a.attempt_number,
                "is_success": a.is_success,
                "validation_errors": a.validation_errors,
                "raw_response_length": len(a.raw_response),
            }
            for a in result.attempts
        ],
        "validated_events": (
            [
                {
                    "event_type": e.event_type,
                    "relative_start_s": e.relative_start_s,
                    "relative_end_s": e.relative_end_s,
                    "item_count": e.item_count,
                    "visibility": e.visibility,
                    "confidence": e.confidence,
                }
                for e in (result.validated_response.events if result.validated_response else [])
            ]
        ),
        "error": result.error,
    }
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nOutput written to: {output_file}")


if __name__ == "__main__":
    main()
