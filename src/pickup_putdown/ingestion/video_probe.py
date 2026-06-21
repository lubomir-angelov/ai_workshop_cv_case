"""Extract video metadata using ffprobe."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Metadata extracted from a video file via ffprobe."""

    duration_s: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    probe_fps: float | None = None
    decode_ok: bool = True
    probe_error: str | None = None
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)


def _find_ffprobe() -> str:
    """Return the ffprobe executable path, raising if not found."""
    path = shutil.which("ffprobe")
    if path is None:
        raise RuntimeError(
            "ffprobe not found on PATH. Install ffmpeg package: "
            "sudo apt-get install ffmpeg (Debian/Ubuntu) or "
            "brew install ffmpeg (macOS)."
        )
    return path


def probe_video(local_path: str | Path) -> ProbeResult:
    """Probe a local video file using ffprobe.

    Returns a ProbeResult with metadata fields populated on success.
    On failure, decode_ok is False and probe_error contains the error message.

    Parameters
    ----------
    local_path : str or Path
        Path to the video file on disk.

    Returns
    -------
    ProbeResult
    """
    local_path = Path(local_path)
    if not local_path.exists():
        return ProbeResult(
            decode_ok=False,
            probe_error=f"File not found: {local_path}",
        )

    ffprobe = _find_ffprobe()

    # Phase 1: metadata extraction
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(local_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return ProbeResult(
                decode_ok=False,
                probe_error=f"ffprobe returned code {result.returncode}: {result.stderr.strip()}",
            )

        data = json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return ProbeResult(decode_ok=False, probe_error="ffprobe timed out after 60s")
    except json.JSONDecodeError as exc:
        return ProbeResult(decode_ok=False, probe_error=f"ffprobe output is not valid JSON: {exc}")
    except FileNotFoundError:
        return ProbeResult(decode_ok=False, probe_error="ffprobe executable not found")

    probe_result = ProbeResult(_raw=data)

    # Extract video stream info
    streams = data.get("streams", [])
    video_stream = None
    for stream in streams:
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        return ProbeResult(
            decode_ok=False,
            probe_error="No video stream found in file",
        )

    # Parse video metadata
    probe_result.width = video_stream.get("width")
    probe_result.height = video_stream.get("height")
    probe_result.video_codec = video_stream.get("codec_name")

    # FPS: prefer avg_frame_rate from stream, fall back to r_frame_rate
    fps_str = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")
    if fps_str and "/" in fps_str:
        try:
            num, den = fps_str.split("/", 2)
            num, den = int(num), int(den)
            if den > 0:
                probe_result.fps = num / den
                probe_result.probe_fps = num / den
        except (ValueError, ZeroDivisionError):
            pass

    # Duration from format
    format_data = data.get("format", {})
    duration_str = format_data.get("duration")
    if duration_str:
        with suppress(ValueError):
            probe_result.duration_s = float(duration_str)

    # Audio codec (optional)
    for stream in streams:
        if stream.get("codec_type") == "audio":
            probe_result.audio_codec = stream.get("codec_name")
            break

    # Phase 2: decode validation — attempt to read first 3 frames or 1 second
    probe_result = _validate_decode(local_path, ffprobe, probe_result)

    logger.debug(
        "Probed %s: %dx%d %sfps duration=%.2fs codec=%s decode_ok=%s",
        local_path.name,
        probe_result.width or "?",
        probe_result.height or "?",
        probe_result.fps or "?",
        probe_result.duration_s or 0,
        probe_result.video_codec or "?",
        probe_result.decode_ok,
    )

    return probe_result


def _validate_decode(local_path: Path, ffprobe: str, result: ProbeResult) -> ProbeResult:
    """Validate that the video can be decoded by reading a short segment."""
    try:
        # Use ffprobe to count decodable video frames in first 1 second
        duration = 1.0
        if result.duration_s is not None and result.duration_s < 1.0:
            duration = result.duration_s

        decode_result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-skip_frame",
                "novideo",
                "-frames:v",
                "3",
                "-ss",
                str(duration),
                "-t",
                "1",
                "-f",
                "null",
                "-",
                str(local_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if decode_result.returncode != 0:
            result.decode_ok = False
            result.probe_error = f"Decode validation failed: {decode_result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        result.decode_ok = False
        result.probe_error = "Decode validation timed out after 30s"
    except FileNotFoundError:
        result.decode_ok = False
        result.probe_error = "ffprobe not found during decode validation"

    return result
