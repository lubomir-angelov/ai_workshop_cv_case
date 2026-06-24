"""VLM-assisted visual annotation pipeline for candidate videos.

Discovers candidate videos, extracts review frames, inspects them visually,
and produces canonical event annotations aligned with the repository schema.

Designed to be run as:
    pickup-putdown annotate-vlm <candidate-dir> --output-dir .local/vlm_annotations
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

from pickup_putdown.annotation.schemas import (
    ConfidenceLevel,
    EventLabel,
)

if TYPE_CHECKING:
    from pickup_putdown.annotation.vlm_client import VlmClientConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schemas for VLM annotation output
# ---------------------------------------------------------------------------


class VlMEventAnnotation(BaseModel):
    """One event annotation produced by VLM visual review."""

    label: EventLabel
    start_s: float = Field(ge=0.0)
    end_s: float = Field(ge=0.0)
    item_count: int = Field(default=1, ge=1)
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH
    hard_case: bool = False
    group_id: str | None = None
    notes: str = ""

    @field_validator("end_s")
    @classmethod
    def end_after_start(cls, v: float, info) -> float:
        start = info.data.get("start_s")
        if start is not None and v <= start:
            raise ValueError("end_s must be greater than start_s")
        return v


class VlMCandidateResult(BaseModel):
    """Normalized result for a single candidate after VLM review."""

    candidate_id: str
    clip_id: str
    video_path: str
    candidate_duration_s: float = Field(ge=0.0)
    source_start_s: float = Field(ge=0.0)
    source_end_s: float = Field(ge=0.0)
    review_status: str = "complete"
    events: list[VlMEventAnnotation] = Field(default_factory=list)
    ignore_intervals: list[dict[str, Any]] = Field(default_factory=list)
    complete_active_span_reviewed: bool = True
    fps: float = 0.0
    notes: str = ""

    @field_validator("source_end_s")
    @classmethod
    def source_end_after_start(cls, v: float, info) -> float:
        start = info.data.get("source_start_s")
        if start is not None and v <= start:
            raise ValueError("source_end_s must be greater than source_start_s")
        return v


class ProcessingRecord(BaseModel):
    """Processing ledger entry for one candidate."""

    candidate_id: str
    video_path: str
    status: str  # success, failure, review_required, skipped
    error: str = ""
    processed_at: str = ""
    frames_extracted: int = 0
    events_found: int = 0


# ---------------------------------------------------------------------------
# Candidate discovery
# ---------------------------------------------------------------------------


def discover_candidates(candidates_dir: str | Path) -> list[dict[str, Any]]:
    """Discover candidate videos from metadata JSON files.

    Walks the candidates directory, finds metadata JSON files (one per source
    clip), and extracts candidate records with video paths and source offsets.

    Returns a sorted list of candidate dicts with keys:
        candidate_id, clip_id, source_start_s, source_end_s,
        candidate_video, duration_s, codec, fps (if available)
    """
    candidates_dir = Path(candidates_dir)
    if not candidates_dir.exists():
        raise FileNotFoundError(f"Candidates directory not found: {candidates_dir}")

    all_candidates: list[dict[str, Any]] = []

    json_files = sorted(candidates_dir.rglob("*.json"))
    for json_file in json_files:
        try:
            content = json.loads(json_file.read_text())
        except json.JSONDecodeError as exc:
            logger.warning("Malformed JSON in %s: %s", json_file, exc)
            continue

        if isinstance(content, list):
            # Flat array of candidates
            for item in content:
                if isinstance(item, dict):
                    all_candidates.append(item)
            continue

        if not isinstance(content, dict):
            continue

        # Source-level metadata with nested candidates
        if "candidates" in content:
            source_video_id = content.get("source_video_id", json_file.stem)
            nested = content.get("candidates", [])
            if not isinstance(nested, list):
                continue

            for cand in nested:
                if not isinstance(cand, dict):
                    continue
                enriched = dict(cand)
                enriched.setdefault("clip_id", source_video_id)
                all_candidates.append(enriched)

    # Sort deterministically
    all_candidates.sort(key=lambda c: (c.get("clip_id", ""), c.get("candidate_id", "")))
    return all_candidates


# ---------------------------------------------------------------------------
# Video probing
# ---------------------------------------------------------------------------


def probe_candidate_video(video_path: str | Path) -> dict[str, Any]:
    """Probe a candidate video with ffprobe.

    Returns dict with duration_s, fps, width, height, codec, nb_frames.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        info = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise RuntimeError(f"ffprobe failed for {video_path}: {exc}") from exc

    video_stream = None
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        raise RuntimeError(f"No video stream found in {video_path}")

    # Parse frame rate
    r_frame_rate = video_stream.get("r_frame_rate", "30/1")
    if "/" in r_frame_rate:
        num, den = r_frame_rate.split("/")
        fps = float(num) / max(1, float(den))
    else:
        fps = float(r_frame_rate)

    duration = float(info.get("format", {}).get("duration", 0.0))

    return {
        "duration_s": duration,
        "fps": fps,
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "codec": video_stream.get("codec_name", ""),
        "nb_frames": int(video_stream.get("nb_frames", 0)),
    }


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def extract_review_frames(
    video_path: str | Path,
    output_dir: str | Path,
    target_fps: float = 5.0,
    max_width: int = 640,
) -> list[Path]:
    """Extract review frames from a candidate video.

    Extracts frames at target_fps, resized to max_width for efficient review.
    Frames are saved as frame_0001.jpg, frame_0002.jpg, etc.

    Returns sorted list of extracted frame paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Calculate stride from actual video FPS
    probe = probe_candidate_video(video_path)
    actual_fps = probe["fps"]
    if actual_fps <= 0:
        actual_fps = 30.0

    stride = max(1, round(actual_fps / target_fps))

    # Extract frames
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"select='not(mod(n,{stride}))',scale={max_width}:-1",
            "-vsync",
            "vfr",
            str(output_dir / "frame_%04d.jpg"),
        ],
        capture_output=True,
        timeout=60,
    )

    frames = sorted(output_dir.glob("frame_*.jpg"))
    return frames


def create_contact_sheet(
    frame_paths: list[Path],
    output_path: str | Path,
    cols: int = 8,
    frame_width: int = 320,
) -> Path:
    """Create a single contact sheet image from extracted frames.

    Arranges frames in a grid with timestamps overlay.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        logger.warning("Pillow not available, skipping contact sheet creation")
        return Path(output_path)

    if not frame_paths:
        return Path(output_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = []
    for fp in frame_paths:
        try:
            img = Image.open(fp)
            if img.width > frame_width:
                ratio = frame_width / img.width
                new_height = max(1, int(img.height * ratio))
                img = img.resize((frame_width, new_height), Image.LANCZOS)
            # Add timestamp from filename
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            frame_num = int(fp.stem.split("_")[1])
            ts = f"#{frame_num}"
            draw.text((5, 5), ts, fill=(255, 255, 0), font=font)
            images.append(img)
        except Exception:
            continue

    if not images:
        return output_path

    # Calculate grid
    rows = (len(images) + cols - 1) // cols
    max_h = max(img.height for img in images)
    max_w = max(img.width for img in images)

    sheet = Image.new("RGB", (cols * max_w, rows * max_h), (240, 240, 240))
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        sheet.paste(img, (col * max_w, row * max_h))

    sheet.save(str(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Frame-to-timestamp conversion
# ---------------------------------------------------------------------------


def frame_index_to_time(frame_idx: int, fps: float, source_start_s: float) -> tuple[float, float]:
    """Convert frame index to (candidate_relative_s, source_absolute_s)."""
    if fps <= 0:
        fps = 30.0
    candidate_rel_s = frame_idx / fps
    source_abs_s = source_start_s + candidate_rel_s
    return candidate_rel_s, source_abs_s


# ---------------------------------------------------------------------------
# VLM annotation analysis (visual inspection)
# ---------------------------------------------------------------------------


def analyze_candidate_frames(
    frame_paths: list[Path],
    fps: float,
    candidate_duration_s: float,
    candidate_id: str,
    contact_sheet_path: Path,
    proposal_info: dict[str, Any] | None = None,
    vlm_config: VlmClientConfig | None = None,
) -> VlMCandidateResult:
    """Analyze extracted frames to determine events via VLM.

    Sends the contact sheet to the VLM for visual inspection and parses
    the structured JSON response into event annotations.
    """
    from pickup_putdown.annotation.vlm_client import (
        call_vlm,
        vlm_result_to_annotations,
    )

    result = VlMCandidateResult(
        candidate_id=candidate_id,
        clip_id="",
        video_path="",
        candidate_duration_s=candidate_duration_s,
        source_start_s=0.0,
        source_end_s=candidate_duration_s,
        fps=fps,
    )

    if vlm_config is None:
        logger.info("VLM disabled for %s, skipping analysis", candidate_id)
        return result

    frame_count = len(frame_paths)
    vlm_response = call_vlm(
        contact_sheet_path,
        frame_count,
        fps,
        candidate_duration_s,
        vlm_config,
    )

    reasoning = vlm_response.get("reasoning", "")
    if reasoning:
        logger.info("VLM reasoning for %s: %s", candidate_id, reasoning[:200])

    annotations = vlm_result_to_annotations(vlm_response, fps)
    events = []
    for ann in annotations:
        try:
            evt = VlMEventAnnotation(**ann)
            events.append(evt)
        except Exception as exc:
            logger.warning("Failed to create VlMEventAnnotation for %s: %s", candidate_id, exc)

    result.events = events
    result.notes = reasoning[:500] if reasoning else ""
    return result


# ---------------------------------------------------------------------------
# Annotation normalization
# ---------------------------------------------------------------------------


def normalize_candidate_result(
    result: VlMCandidateResult,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize VLM candidate result into canonical event dicts and ignore intervals.

    Returns (events, ignore_intervals) as lists of dicts compatible with
    the canonical CSV schema.
    """
    events: list[dict[str, Any]] = []
    ignores: list[dict[str, Any]] = []

    for evt in result.events:
        # Convert candidate-relative timestamps to source-video timestamps
        source_event_start = result.source_start_s + evt.start_s
        source_event_end = result.source_start_s + evt.end_s

        # Generate deterministic event group ID
        group_raw = f"{result.clip_id}:{evt.label}:{evt.start_s:.2f}"
        group_id = f"group_{hashlib.sha256(group_raw.encode()).hexdigest()[:12]}"

        # Handle multi-item: emit separate rows
        for item_idx in range(evt.item_count):
            event_raw = f"{result.clip_id}:{evt.label}:{group_id}:{item_idx}"
            event_id = f"evt_{hashlib.md5(event_raw.encode()).hexdigest()[:12]}"

            events.append(
                {
                    "event_id": event_id,
                    "clip_id": result.clip_id,
                    "type": str(evt.label),
                    "t_start": round(source_event_start, 3),
                    "t_end": round(source_event_end, 3),
                    "hard_case": evt.hard_case,
                    "annotator": "vlm_pipeline",
                    "confidence": str(evt.confidence),
                    "notes": evt.notes,
                }
            )

    # Process ignore intervals
    for ig in result.ignore_intervals:
        ig_start = result.source_start_s + float(ig.get("start_s", 0.0))
        ig_end = result.source_start_s + float(ig.get("end_s", 0.0))
        ig_raw = f"{result.clip_id}:ignore:{ig_start:.2f}"
        ig_id = f"ign_{hashlib.md5(ig_raw.encode()).hexdigest()[:12]}"

        ignores.append(
            {
                "ignore_id": ig_id,
                "clip_id": result.clip_id,
                "t_start": round(ig_start, 3),
                "t_end": round(ig_end, 3),
                "reason": ig.get("reason", "UNLABELABLE"),
                "annotator": "vlm_pipeline",
                "notes": ig.get("notes", ""),
            }
        )

    return events, ignores


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_candidate_annotation(
    result: VlMCandidateResult,
) -> list[str]:
    """Validate a candidate annotation result.

    Returns list of error messages (empty if valid).
    """
    errors: list[str] = []

    # Check timestamps against duration
    for evt in result.events:
        if evt.start_s < 0:
            errors.append(f"Event {evt.label} has negative start_s={evt.start_s}")
        if evt.end_s > result.candidate_duration_s + 0.1:
            errors.append(
                f"Event {evt.label} end_s={evt.end_s} exceeds "
                f"candidate duration {result.candidate_duration_s}"
            )
        if evt.end_s <= evt.start_s:
            errors.append(f"Event {evt.label} has end_s={evt.end_s} <= start_s={evt.start_s}")

    # Check source timestamps
    for evt in result.events:
        source_start = result.source_start_s + evt.start_s
        source_end = result.source_start_s + evt.end_s
        if source_start >= source_end:
            errors.append(
                f"Event {evt.label} source timestamps invalid: {source_start} >= {source_end}"
            )

    # Check label values
    valid_labels = {EventLabel.PICKUP, EventLabel.PUTDOWN}
    for evt in result.events:
        if evt.label not in valid_labels:
            errors.append(f"Invalid event label: {evt.label}")

    # Check confidence values
    valid_confidence = {ConfidenceLevel.HIGH, ConfidenceLevel.MED, ConfidenceLevel.LOW}
    for evt in result.events:
        if evt.confidence not in valid_confidence:
            errors.append(f"Invalid confidence: {evt.confidence}")

    return errors


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Configuration for the VLM annotation pipeline."""

    candidates_dir: str
    output_dir: str
    review_fps: float = 5.0
    max_frame_width: int = 640
    contact_sheet_cols: int = 8
    force: bool = False
    limit: int | None = None
    annotator: str = "vlm_pipeline"
    # VLM client settings
    vlm_base_url: str = "http://localhost:8080"
    vlm_model: str = ""
    vlm_temperature: float = 0.0
    vlm_max_tokens: int = 2048
    vlm_timeout_s: int = 120
    vlm_enabled: bool = True


@dataclass
class PipelineSummary:
    """Summary of pipeline execution."""

    total_candidates: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    review_required: int = 0
    events_found: int = 0
    processing_time_s: float = 0.0
    errors: list[str] = field(default_factory=list)


def run_pipeline(config: PipelineConfig) -> PipelineSummary:
    """Run the VLM annotation pipeline.

    1. Discover candidates
    2. For each candidate: probe, extract frames, analyze, normalize
    3. Write outputs: raw JSON, normalized JSON, events.csv, processing.csv, summary.json
    """
    import time

    start_time = time.time()
    summary = PipelineSummary()

    output_base = Path(config.output_dir)
    raw_dir = output_base / "raw"
    normalized_dir = output_base / "normalized"
    frames_dir = output_base / "review_frames"
    for d in [raw_dir, normalized_dir, frames_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Discover candidates
    candidates = discover_candidates(config.candidates_dir)
    if config.limit:
        candidates = candidates[: config.limit]
    summary.total_candidates = len(candidates)
    logger.info("Discovered %d candidates", len(candidates))

    all_events: list[dict[str, Any]] = []
    all_processing: list[ProcessingRecord] = []

    for idx, cand in enumerate(candidates):
        candidate_id = cand.get("candidate_id", f"unknown_{idx}")
        candidate_key = cand.get("candidate_key", "")
        video_path = Path(candidate_key)

        # Print progress every 10 candidates
        if (idx + 1) % 10 == 0 or idx == 0:
            pct = (idx + 1) / len(candidates) * 100
            print(
                f"Progress: {idx + 1}/{len(candidates)} ({pct:.1f}%)", file=sys.stderr, flush=True
            )

        # Skip if output already exists (unless --force)
        norm_path = normalized_dir / f"{candidate_id}.json"
        if norm_path.exists() and not config.force:
            summary.skipped += 1
            all_processing.append(
                ProcessingRecord(
                    candidate_id=candidate_id,
                    video_path=str(video_path),
                    status="skipped",
                    processed_at=datetime.now(UTC).isoformat(),
                )
            )
            continue

        rec = ProcessingRecord(
            candidate_id=candidate_id,
            video_path=str(video_path),
            status="pending",
            processed_at=datetime.now(UTC).isoformat(),
        )

        try:
            # Probe video
            probe_info = probe_candidate_video(video_path)
            fps = probe_info["fps"]
            duration = probe_info["duration_s"]

            # Extract frames (skip if already extracted)
            cand_frames_dir = frames_dir / candidate_id
            existing_frames = sorted(cand_frames_dir.glob("frame_*.jpg"))
            if existing_frames and not config.force:
                frame_paths = existing_frames
                logger.info("Reusing %d existing frames for %s", len(frame_paths), candidate_id)
            else:
                frame_paths = extract_review_frames(
                    video_path,
                    cand_frames_dir,
                    target_fps=config.review_fps,
                    max_width=config.max_frame_width,
                )
            rec.frames_extracted = len(frame_paths)

            # Create contact sheet (skip if exists)
            contact_sheet_path = cand_frames_dir / "contact_sheet.jpg"
            if not contact_sheet_path.exists() or config.force:
                create_contact_sheet(
                    frame_paths, contact_sheet_path, cols=config.contact_sheet_cols
                )

            # Build VLM client config
            vlm_config: VlmClientConfig | None = None
            if config.vlm_enabled:
                from pickup_putdown.annotation.vlm_client import VlmClientConfig

                vlm_config = VlmClientConfig(
                    base_url=config.vlm_base_url,
                    model=config.vlm_model,
                    temperature=config.vlm_temperature,
                    max_tokens=config.vlm_max_tokens,
                    timeout_s=config.vlm_timeout_s,
                )

            # Analyze via VLM
            analysis_result = analyze_candidate_frames(
                frame_paths=frame_paths,
                fps=fps,
                candidate_duration_s=duration,
                candidate_id=candidate_id,
                contact_sheet_path=contact_sheet_path,
                vlm_config=vlm_config,
            )

            # Merge candidate metadata into analysis result
            result = VlMCandidateResult(
                candidate_id=candidate_id,
                clip_id=cand.get("clip_id", ""),
                video_path=str(video_path),
                candidate_duration_s=duration,
                source_start_s=float(cand.get("source_start_s", 0.0)),
                source_end_s=float(cand.get("source_end_s", 0.0)),
                fps=fps,
                review_status="complete" if vlm_config else "pending_review",
                events=analysis_result.events,
                notes=analysis_result.notes,
            )

            # Save raw result
            raw_path = raw_dir / f"{candidate_id}.json"
            raw_data = result.model_dump()
            raw_data["frame_count"] = len(frame_paths)
            raw_data["contact_sheet"] = str(contact_sheet_path)
            raw_data["probe_info"] = probe_info
            raw_data["metadata"] = cand
            raw_path.write_text(json.dumps(raw_data, indent=2, default=str))

            # Validate
            validation_errors = validate_candidate_annotation(result)
            if validation_errors:
                rec.status = "failure"
                rec.error = "; ".join(validation_errors)
                logger.warning("Validation errors for %s: %s", candidate_id, validation_errors)
            else:
                rec.status = "success"

            rec.events_found = len(result.events)

            # Normalize and save
            events, ignores = normalize_candidate_result(result)
            all_events.extend(events)

            norm_data = result.model_dump()
            norm_data["ignore_intervals"] = ignores
            norm_path.write_text(json.dumps(norm_data, indent=2, default=str))

            summary.processed += 1
            print(
                f"[{idx + 1}/{len(candidates)}] OK {candidate_id} "
                f"({rec.frames_extracted} frames, {rec.events_found} events)",
                file=sys.stderr,
                flush=True,
            )

        except FileNotFoundError as exc:
            rec.status = "failure"
            rec.error = f"File not found: {exc}"
            logger.warning("Failed %s: %s", candidate_id, exc)
            summary.failed += 1
        except Exception as exc:
            rec.status = "failure"
            rec.error = str(exc)
            logger.exception("Failed %s: %s", candidate_id, exc)
            summary.failed += 1

        all_processing.append(rec)

    # Sort events deterministically
    all_events.sort(key=lambda e: (e["clip_id"], e["t_start"], e["type"], e["event_id"]))

    # Write events.csv
    events_csv = output_base / "events.csv"
    if all_events:
        csv_columns = [
            "event_id",
            "clip_id",
            "type",
            "t_start",
            "t_end",
            "hard_case",
            "annotator",
            "confidence",
            "notes",
        ]
        with events_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_events)
    else:
        events_csv.write_text(
            "event_id,clip_id,type,t_start,t_end,hard_case,annotator,confidence,notes\n"
        )

    # Write processing.csv
    processing_csv = output_base / "processing.csv"
    with processing_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidate_id",
                "video_path",
                "status",
                "error",
                "processed_at",
                "frames_extracted",
                "events_found",
            ],
        )
        writer.writeheader()
        for rec in all_processing:
            writer.writerow(rec.model_dump())

    # Write summary.json
    summary.processing_time_s = round(time.time() - start_time, 2)
    summary.events_found = len(all_events)
    summary_json = output_base / "summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "total_candidates": summary.total_candidates,
                "processed": summary.processed,
                "skipped": summary.skipped,
                "failed": summary.failed,
                "review_required": summary.review_required,
                "events_found": summary.events_found,
                "processing_time_s": summary.processing_time_s,
                "annotator": config.annotator,
                "review_fps": config.review_fps,
                "force": config.force,
                "timestamp": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
    )

    logger.info(
        "Pipeline complete: %d processed, %d skipped, %d failed, %d events",
        summary.processed,
        summary.skipped,
        summary.failed,
        summary.events_found,
    )
    return summary
