"""Import CVAT-for-video exports into canonical events, ignore intervals and actor tracks.

Annotation happens on the *source* video (see docs/CVAT_ANNOTATION_SETUP.md), so
every timestamp produced here is already in source-video time — no candidate-relative
conversion is needed.

One CVAT ``<track>`` is one annotated interval. A track carries a box on every frame
it spans, which gives two things at once:

* the temporal extent of the event (first to last frame with ``outside="0"``);
* a per-frame box that Track B1 uses as the actor-conditioning crop, standing in for
  the pose tracks that were never produced for these clips.

A track may be interrupted (``outside="1"``) and resume later. Each contiguous run of
inside frames becomes its own interval, so an interrupted track yields several events
rather than one event spanning the gap.

Emitted artifacts (all in source-video time):

    events.parquet           canonical Event rows (pickup / putdown)
    ignore_intervals.parquet canonical IgnoreInterval rows
    clips.parquet            one row per annotated clip, with fps and split
    actor_tracks/<clip>.parquet  per-frame boxes, pose-track-compatible columns

The actor track columns (``actor_id``, ``timestamp_s``, ``person_bbox_*``) are chosen to
match what ``layer1.track_b1.dataset`` already reads from pose tracks, so the Track B1
dataset consumes them unmodified.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd

logger = logging.getLogger(__name__)

EVENT_LABELS = frozenset({"pickup", "putdown"})
IGNORE_LABEL = "ignore"

#: Attribute values that mean the annotator considered the interval finished.
ACCEPTED_REVIEW_STATUSES = frozenset({"accepted", "reviewed"})

#: Clip stems encode the recording window: D2_S<start>_E<end>_anon.
CLIP_TIME_RE = re.compile(r"_S(\d{14})_E(\d{14})_")

DEFAULT_CONFIDENCE = "high"
DEFAULT_IGNORE_REASON = "UNLABELABLE"


# ============================================================
# PARSED REPRESENTATION
# ============================================================


@dataclass
class TrackInterval:
    """One contiguous run of inside-frames from a CVAT track."""

    clip_id: str
    track_id: int
    segment_index: int
    label: str
    start_frame: int
    end_frame: int
    attributes: dict[str, str]
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)

    @property
    def actor_id(self) -> str:
        """Stable per-track actor identity.

        CVAT tracks are drawn per interaction, not per person, so one track is one
        actor stream. Two overlapping tracks in a clip therefore produce two
        independent actors, which is what Track B1 needs to predict per actor.
        """
        return f"trk{self.track_id:03d}"

    def event_id(self) -> str:
        return f"{self.clip_id}__{self.actor_id}__{self.segment_index:02d}"


@dataclass
class ClipAnnotations:
    """Everything parsed out of one CVAT archive."""

    clip_id: str
    task_id: int
    fps: float
    width: int
    height: int
    n_frames: int
    duration_s: float
    fps_source: str
    intervals: list[TrackInterval]


# ============================================================
# XML PARSING
# ============================================================


def _clip_duration_s(clip_id: str) -> float | None:
    """Recording duration implied by the clip stem, or None if unparseable."""
    match = CLIP_TIME_RE.search(clip_id)
    if match is None:
        return None
    try:
        start = datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
        end = datetime.strptime(match.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None
    seconds = (end - start).total_seconds()
    return seconds if seconds > 0 else None


def probe_fps(video_path: Path) -> float | None:
    """Exact average frame rate of a video, or None if it cannot be probed.

    CVAT stores annotations as frame indices, so the frame->second conversion is only
    as good as the frame rate. The stem-derived estimate is off by ~0.2% here, which
    drifts a third of a second by the end of a clip — more than a typical event is
    long — so the real rate is read from the file whenever it is available.
    """
    if not video_path.exists() or shutil.which("ffprobe") is None:
        return None
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate",
                "-of",
                "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout
        rate = json.loads(out)["streams"][0]["avg_frame_rate"]
    except (subprocess.SubprocessError, KeyError, IndexError, json.JSONDecodeError) as exc:
        logger.warning("ffprobe failed for %s: %s", video_path.name, exc)
        return None

    num, _, den = rate.partition("/")
    try:
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return None
    return fps if fps > 0 else None


def _box_attributes(box: ElementTree.Element) -> dict[str, str]:
    return {attr.get("name", ""): (attr.text or "") for attr in box.findall("attribute")}


def _split_track(
    clip_id: str,
    track: ElementTree.Element,
) -> Iterator[TrackInterval]:
    """Yield one interval per contiguous run of inside-frames in a CVAT track."""
    track_id = int(track.get("id", "-1"))
    label = track.get("label", "")

    run: list[ElementTree.Element] = []
    segment_index = 0

    def flush(run: list[ElementTree.Element], segment_index: int) -> TrackInterval | None:
        if not run:
            return None
        frames = [int(box.get("frame", "0")) for box in run]
        return TrackInterval(
            clip_id=clip_id,
            track_id=track_id,
            segment_index=segment_index,
            label=label,
            start_frame=min(frames),
            end_frame=max(frames),
            attributes=_box_attributes(run[0]),
            boxes=[
                (
                    int(box.get("frame", "0")),
                    float(box.get("xtl", "0")),
                    float(box.get("ytl", "0")),
                    float(box.get("xbr", "0")),
                    float(box.get("ybr", "0")),
                )
                for box in run
            ],
        )

    for box in track.findall("box"):
        if box.get("outside") == "1":
            interval = flush(run, segment_index)
            if interval is not None:
                yield interval
                segment_index += 1
            run = []
            continue
        run.append(box)

    interval = flush(run, segment_index)
    if interval is not None:
        yield interval


def parse_archive(archive: Path, video_dir: Path | None = None) -> ClipAnnotations:
    """Parse one ``CVAT for video 1.1`` archive.

    When ``video_dir`` holds the matching source video its exact frame rate is used;
    otherwise the rate is estimated from the clip stem's recording window.
    """
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if n.endswith("annotations.xml")]
        if not names:
            raise ValueError(f"no annotations.xml in {archive.name}")
        root = ElementTree.fromstring(zf.read(names[0]))

    task = root.find(".//meta/task")
    if task is None:
        raise ValueError(f"no <task> metadata in {archive.name}")

    clip_id = task.findtext("name") or archive.stem
    n_frames = int(task.findtext("size") or 0)
    width = int(task.findtext("original_size/width") or 0)
    height = int(task.findtext("original_size/height") or 0)

    if n_frames <= 0:
        raise ValueError(f"CVAT reports {n_frames} frames for {clip_id}")

    fps = probe_fps(video_dir / f"{clip_id}.mp4") if video_dir is not None else None
    fps_source = "video"
    if fps is None:
        stem_duration = _clip_duration_s(clip_id)
        if stem_duration is None:
            raise ValueError(
                f"cannot derive a timebase for {clip_id}: no video and unparseable stem"
            )
        fps = n_frames / stem_duration
        fps_source = "clip stem"
        logger.warning(
            "%s: no probeable video, estimating fps=%.4f from the clip stem", clip_id, fps
        )
    logger.debug("%s: fps=%.4f (from %s)", clip_id, fps, fps_source)
    duration_s = n_frames / fps

    intervals = [
        interval for track in root.findall(".//track") for interval in _split_track(clip_id, track)
    ]

    return ClipAnnotations(
        clip_id=clip_id,
        task_id=int(task.findtext("id") or 0),
        fps=fps,
        width=width,
        height=height,
        n_frames=n_frames,
        duration_s=duration_s,
        fps_source=fps_source,
        intervals=intervals,
    )


# ============================================================
# CANONICAL CONVERSION
# ============================================================


def _interval_times(
    interval: TrackInterval,
    fps: float,
    min_duration_s: float,
) -> tuple[float, float]:
    """Frame span -> [t_start, t_end), widened so single-frame marks stay non-empty."""
    t_start = interval.start_frame / fps
    t_end = (interval.end_frame + 1) / fps
    if t_end - t_start < min_duration_s:
        centre = (t_start + t_end) / 2
        t_start = max(0.0, centre - min_duration_s / 2)
        t_end = t_start + min_duration_s
    return t_start, t_end


def to_events(
    clip: ClipAnnotations,
    min_duration_s: float,
    accepted_only: bool,
) -> list[dict]:
    """Canonical Event rows for one clip's pickup/putdown intervals.

    ``item_count = N`` yields N rows with the same interval, per
    docs/LABELING_GUIDELINES.md §5.1 ("do not collapse a two-item action into one
    canonical event row"); the evaluator's multi-item metrics rely on it. Rows of one
    interval share ``event_group_id`` and differ in ``item_index``.
    """
    rows: list[dict] = []
    for interval in clip.intervals:
        if interval.label not in EVENT_LABELS:
            continue
        attrs = interval.attributes
        review_status = attrs.get("review_status", "draft")
        if accepted_only and review_status not in ACCEPTED_REVIEW_STATUSES:
            logger.debug("skipping %s: review_status=%s", interval.event_id(), review_status)
            continue

        t_start, t_end = _interval_times(interval, clip.fps, min_duration_s)
        item_count = int(float(attrs.get("item_count") or 1))
        if item_count < 1:
            raise ValueError(f"{interval.event_id()}: item_count {item_count} < 1")
        group_id = interval.event_id()
        for item_index in range(item_count):
            rows.append(
                {
                    "event_id": group_id if item_count == 1 else f"{group_id}__i{item_index}",
                    "clip_id": clip.clip_id,
                    "type": interval.label,
                    "t_start": t_start,
                    "t_end": t_end,
                    "hard_case": attrs.get("hard_case", "false") == "true",
                    "annotator": "cvat",
                    "confidence": attrs.get("confidence") or DEFAULT_CONFIDENCE,
                    "notes": attrs.get("notes") or None,
                    "actor_id": interval.actor_id,
                    "item_count": item_count,
                    "review_status": review_status,
                    "event_group_id": group_id,
                    "item_index": item_index,
                }
            )
    return rows


def to_ignore_intervals(clip: ClipAnnotations, min_duration_s: float) -> list[dict]:
    """Canonical IgnoreInterval rows for one clip."""
    rows: list[dict] = []
    for interval in clip.intervals:
        if interval.label != IGNORE_LABEL:
            continue
        attrs = interval.attributes
        t_start, t_end = _interval_times(interval, clip.fps, min_duration_s)
        rows.append(
            {
                "ignore_id": interval.event_id(),
                "clip_id": clip.clip_id,
                "t_start": t_start,
                "t_end": t_end,
                "reason": attrs.get("ignore_reason") or DEFAULT_IGNORE_REASON,
                "annotator": "cvat",
                "notes": attrs.get("notes") or None,
            }
        )
    return rows


def to_actor_track(clip: ClipAnnotations, context_pad_s: float = 0.0) -> pd.DataFrame:
    """Per-frame boxes for one clip, in the column layout Track B1 reads.

    Only ``pickup``/``putdown`` tracks contribute: ``ignore`` marks a region the model
    must not be scored on, so its geometry is not a conditioning signal.

    ``context_pad_s`` extends each track before and after the annotated interval by
    repeating its boundary box. This exists to stop the crop from leaking the label.
    Boxes are only drawn while an event is happening, so without padding every negative
    window would find no box, fall back to the full 4K frame, and the classifier could
    separate the classes on frame size alone without ever looking at the action. With
    padding, a negative sampled from the run-up to an interaction is cropped to the same
    region as the positive, and only the motion inside it distinguishes them.

    Padded rows are marked ``is_padding=True`` so downstream code can tell an
    annotator-drawn box from a held one.
    """
    pad_frames = int(round(context_pad_s * clip.fps))
    rows: list[dict] = []

    def emit(
        interval: TrackInterval, frame: int, box: tuple[float, float, float, float], padding: bool
    ) -> None:
        rows.append(
            {
                "clip_id": clip.clip_id,
                "actor_id": interval.actor_id,
                "event_id": interval.event_id(),
                "source_frame_index": frame,
                "timestamp_s": frame / clip.fps,
                "person_bbox_x1": box[0],
                "person_bbox_y1": box[1],
                "person_bbox_x2": box[2],
                "person_bbox_y2": box[3],
                "is_padding": padding,
            }
        )

    for interval in clip.intervals:
        if interval.label not in EVENT_LABELS or not interval.boxes:
            continue

        for frame, x1, y1, x2, y2 in interval.boxes:
            emit(interval, frame, (x1, y1, x2, y2), padding=False)

        first, last = interval.boxes[0], interval.boxes[-1]
        for offset in range(1, pad_frames + 1):
            before = first[0] - offset
            if before >= 0:
                emit(interval, before, first[1:], padding=True)
            after = last[0] + offset
            if after < clip.n_frames:
                emit(interval, after, last[1:], padding=True)

    return pd.DataFrame(rows)


# ============================================================
# SPLIT ASSIGNMENT
# ============================================================


def normalize_clip_id(raw: str) -> str:
    """Canonical source-video clip id (``D2_S<start>_E<end>_anon``).

    Pose runs and registries have used ``clip_<id>`` and ``b1_<id>`` (the RUN_ID) as
    well; mixing them with the CVAT ids silently drops every label join, so they are
    stripped here and anything that still does not look like a clip stem raises.
    """
    clip_id = str(raw)
    for prefix in ("b1_", "clip_"):
        clip_id = clip_id.removeprefix(prefix)
    if CLIP_TIME_RE.search(clip_id) is None:
        raise ValueError(f"not a source-video clip id: {raw!r}")
    return clip_id


def recording_day(clip_id: str) -> str:
    """Recording day encoded in the clip stem, used as the split grouping key."""
    match = CLIP_TIME_RE.search(clip_id)
    return match.group(1)[:8] if match else "unknown"


def assign_splits(
    clips_df: pd.DataFrame,
    events_df: pd.DataFrame,
    val_days: int,
    test_days: int,
) -> pd.DataFrame:
    """Assign train/val/test at the recording-day level.

    Splitting by day rather than by clip keeps clips recorded minutes apart — same
    shoppers, same shelf state — out of different splits, which is where leakage
    would otherwise come from. Days are ordered by event count so the held-out days
    are the smaller ones, leaving the bulk of supervision in train.
    """
    clips_df = clips_df.copy()
    clips_df["recording_day"] = clips_df["clip_id"].map(recording_day)

    events_per_clip = (
        events_df.groupby("clip_id").size() if len(events_df) else pd.Series(dtype=int)
    )
    clips_df["n_events"] = clips_df["clip_id"].map(events_per_clip).fillna(0).astype(int)

    per_day = clips_df.groupby("recording_day")["n_events"].sum().sort_values(ascending=True)

    holdout = list(per_day.index)
    val = set(holdout[:val_days])
    test = set(holdout[val_days : val_days + test_days])

    def split_of(day: str) -> str:
        if day in val:
            return "val"
        if day in test:
            return "test"
        return "train"

    clips_df["split"] = clips_df["recording_day"].map(split_of)
    return clips_df


def assign_splits_from_registry(
    clips_df: pd.DataFrame,
    events_df: pd.DataFrame,
    day_splits: dict[str, str],
) -> pd.DataFrame:
    """Assign splits from an explicit recording-day registry.

    Every clip's day must be listed: a new day is a decision to record in the
    registry, not something to place automatically.
    """
    clips_df = clips_df.copy()
    clips_df["recording_day"] = clips_df["clip_id"].map(recording_day)
    events_per_clip = (
        events_df.groupby("clip_id").size() if len(events_df) else pd.Series(dtype=int)
    )
    clips_df["n_events"] = clips_df["clip_id"].map(events_per_clip).fillna(0).astype(int)
    unknown = sorted(set(clips_df["recording_day"]) - set(day_splits))
    if unknown:
        raise ValueError(f"recording days {unknown} are not in the split registry")
    invalid = sorted(set(day_splits.values()) - {"train", "val", "test"})
    if invalid:
        raise ValueError(f"invalid split names in registry: {invalid}")
    clips_df["split"] = clips_df["recording_day"].map(day_splits)
    return clips_df


def load_split_registry(path: Path, name: str) -> dict[str, str]:
    """Day -> split mapping of one named registry in a split-registry YAML file."""
    import yaml

    registries = (yaml.safe_load(path.read_text()) or {}).get("registries", {})
    if name not in registries:
        raise KeyError(f"split registry {name!r} not in {path}; have {sorted(registries)}")
    return {str(day): split for day, split in registries[name]["days"].items()}


# ============================================================
# TOP-LEVEL IMPORT
# ============================================================


@dataclass
class ImportResult:
    events: pd.DataFrame
    ignore_intervals: pd.DataFrame
    clips: pd.DataFrame
    actor_tracks: dict[str, pd.DataFrame]


EVENT_COLUMNS = [
    "event_id",
    "clip_id",
    "type",
    "t_start",
    "t_end",
    "hard_case",
    "annotator",
    "confidence",
    "notes",
    "actor_id",
    "item_count",
    "review_status",
    "event_group_id",
    "item_index",
]
IGNORE_COLUMNS = [
    "ignore_id",
    "clip_id",
    "t_start",
    "t_end",
    "reason",
    "annotator",
    "notes",
]


def import_export_dir(
    export_dir: Path,
    video_dir: Path | None = None,
    min_duration_s: float = 0.25,
    accepted_only: bool = False,
    context_pad_s: float = 3.0,
    val_days: int = 1,
    test_days: int = 1,
    day_splits: dict[str, str] | None = None,
) -> ImportResult:
    """Parse every CVAT archive in ``export_dir`` into canonical tables.

    Splits come from ``day_splits`` (an explicit registry) when given, otherwise
    from :func:`assign_splits` ranking with ``val_days``/``test_days``.
    """
    archives = sorted(export_dir.glob("*.zip"))
    if not archives:
        raise ValueError(f"no CVAT archives found in {export_dir}")

    event_rows: list[dict] = []
    ignore_rows: list[dict] = []
    clip_rows: list[dict] = []
    actor_tracks: dict[str, pd.DataFrame] = {}

    for archive in archives:
        clip = parse_archive(archive, video_dir=video_dir)
        event_rows.extend(to_events(clip, min_duration_s, accepted_only))
        ignore_rows.extend(to_ignore_intervals(clip, min_duration_s))
        actor_tracks[clip.clip_id] = to_actor_track(clip, context_pad_s=context_pad_s)
        clip_rows.append(
            {
                "clip_id": clip.clip_id,
                "s3_key": f"anon/{clip.clip_id}.mp4",
                "task_id": clip.task_id,
                "fps": clip.fps,
                "n_frames": clip.n_frames,
                "duration_s": clip.duration_s,
                "width": clip.width,
                "height": clip.height,
                "fps_source": clip.fps_source,
            }
        )

    events_df = pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
    ignore_df = pd.DataFrame(ignore_rows, columns=IGNORE_COLUMNS)
    clips_df = (
        assign_splits_from_registry(pd.DataFrame(clip_rows), events_df, day_splits)
        if day_splits is not None
        else assign_splits(pd.DataFrame(clip_rows), events_df, val_days, test_days)
    )

    return ImportResult(
        events=events_df,
        ignore_intervals=ignore_df,
        clips=clips_df,
        actor_tracks=actor_tracks,
    )


def write_result(result: ImportResult, output_dir: Path) -> None:
    """Persist canonical tables under ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result.events.to_parquet(output_dir / "events.parquet", index=False)
    result.ignore_intervals.to_parquet(output_dir / "ignore_intervals.parquet", index=False)
    result.clips.to_parquet(output_dir / "clips.parquet", index=False)

    tracks_dir = output_dir / "actor_tracks"
    tracks_dir.mkdir(exist_ok=True)
    for clip_id, track_df in result.actor_tracks.items():
        track_df.to_parquet(tracks_dir / f"{clip_id}.parquet", index=False)
