#!/usr/bin/env python3
"""Convert CVAT 1.1 source-video export to canonical formats.

Annotations are on full source videos, so timestamps are already canonical.
Candidate metadata is loaded for provenance matching only.

Two input formats are auto-detected:
- single-task (job) export: <annotations><meta><task>... root-level <track>
  with per-frame <box> elements (no fps in XML; fps comes from --clips-csv)
- multi-task export: <cvat><Task><Info>... with <Point> elements

Usage (single file):
  python scripts/convert_cvat_source_export.py \
    --cvat-export annotation/cvat_export_source.xml \
    --events-output annotation/exports/events.csv \
    --ignore-output annotation/exports/ignore_intervals.parquet \
    --provenance-output annotation/exports/event_provenance.parquet

Usage (directory of per-job archives, e.g. the S3 raw export):
  python scripts/convert_cvat_source_export.py \
    --cvat-export-dir .local/annotations/cvat_2026-09-07/unzipped \
    --clips-csv .local/task_7_vlm/clips.csv \
    --manifest .local/annotations/cvat_2026-09-07/raw/export_manifest.csv \
    --output-dir .local/task_7_human
"""

import argparse
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def load_candidate_windows(meta_dir: str) -> dict[str, list[dict]]:
    """Load candidate windows per source video.

    Returns:
        { clip_id: [ { candidate_id, source_start_s, source_end_s, ... }, ... ] }
    """
    windows: dict[str, list[dict]] = {}
    base = Path(meta_dir)
    for src_dir in sorted(base.iterdir()):
        if not src_dir.is_dir():
            continue
        meta_file = src_dir / (src_dir.name + ".json")
        if not meta_file.exists():
            continue
        data = json.loads(meta_file.read_text())
        clip_id = data.get("source_video_id", src_dir.name)
        cands: list[dict] = []
        for c in data.get("candidates", []):
            cands.append(
                {
                    "candidate_id": c["candidate_id"],
                    "source_start_s": c["source_start_s"],
                    "source_end_s": c["source_end_s"],
                    "actor_id": c.get("actor_id"),
                    "hand_side": c.get("hand_side"),
                    "region_id": c.get("region_id"),
                }
            )
        if cands:
            windows[clip_id] = cands
    return windows


def find_matching_candidate(
    event_start: float, event_end: float, candidates: list[dict]
) -> dict | None:
    """Find the candidate window with the most overlap for this event."""
    best: dict | None = None
    best_overlap = 0.0
    for c in candidates:
        overlap_start = max(c["source_start_s"], event_start)
        overlap_end = min(c["source_end_s"], event_end)
        overlap = max(0.0, overlap_end - overlap_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best = c
    return best if best and best_overlap > 0.01 else None


def parse_cvat_1_1(xml_path: str) -> list[dict]:
    """Parse CVAT 1.1 XML export.

    Returns list of annotation dicts with:
      clip_id, label, start_s, end_s, fps, attributes
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    results: list[dict] = []

    for task in root.findall(".//Task"):
        fps_elem = task.find("Info/fps")
        fps = 30.0
        if fps_elem is not None and fps_elem.text:
            try:
                fps = float(fps_elem.text)
            except ValueError:
                pass

        video_elem = task.find("Info/file_name")
        clip_id = ""
        if video_elem is not None and video_elem.text:
            clip_id = Path(video_elem.text).stem
        else:
            name_elem = task.find("Info/name")
            if name_elem is not None and name_elem.text:
                clip_id = name_elem.text.strip()

        for track in task.findall(".//Track"):
            label = track.attrib.get("label", "").strip()
            if not label:
                continue

            frames: list[int] = []
            for point in track.findall("Point"):
                frames.append(int(point.attrib["frame"]))

            if not frames:
                continue

            start_frame = min(frames)
            end_frame = max(frames)

            start_s = start_frame / fps
            end_s = (end_frame + 1) / fps

            attributes: dict[str, str] = {}
            attr_elem = track.find("Attributes")
            if attr_elem is not None:
                for attr in attr_elem.findall("Attribute"):
                    attributes[attr.attrib["name"]] = attr.attrib["value"]

            results.append(
                {
                    "clip_id": clip_id,
                    "label": label,
                    "start_s": start_s,
                    "end_s": end_s,
                    "fps": fps,
                    "attributes": attributes,
                }
            )

    return results


def parse_cvat_1_1_job_export(root: ET.Element, fps: float) -> list[dict]:
    """Parse a CVAT 1.1 single-task (job) export.

    Verified layout (2026-09-07 export):
      <annotations><meta><task><name>, <size></task></meta>
      <track label="..."><box frame="..." keyframe="..."><attribute name=.../></box></track>

    This format carries no fps; the caller supplies it from the clip registry.
    Track attributes are read from the first keyframe box.
    """
    task = root.find("meta/task")
    if task is None:
        raise ValueError("job export has no <meta><task> element")
    clip_id = (task.findtext("name") or "").strip()
    if not clip_id:
        raise ValueError("job export task has an empty <name>")
    annotator = task.findtext("assignee/username") or task.findtext("owner/username") or "unknown"

    results: list[dict] = []
    for track in root.findall("track"):
        label = track.attrib.get("label", "").strip()
        if not label:
            continue
        boxes = track.findall("box")
        frames = [int(b.attrib["frame"]) for b in boxes]
        if not frames:
            continue
        keyframe = next((b for b in boxes if b.attrib.get("keyframe") == "1"), boxes[0])
        attributes = {
            a.attrib.get("name", ""): (a.text or "").strip() for a in keyframe.findall("attribute")
        }
        results.append(
            {
                "clip_id": clip_id,
                "label": label,
                "start_s": min(frames) / fps,
                "end_s": (max(frames) + 1) / fps,
                "fps": fps,
                "attributes": attributes,
                "annotator": annotator,
            }
        )
    return results


def iter_job_exports(export_dir: str):
    """Yield (stem, root Element) for each job export in a directory.

    Accepts *.zip archives (single .xml at archive root) and/or
    <stem>/annotations.xml directories, sorted by name.
    """
    base = Path(export_dir)
    if not base.is_dir():
        raise SystemExit(f"Export directory not found: {export_dir}")
    for entry in sorted(base.iterdir()):
        if entry.is_file() and entry.suffix == ".zip":
            with zipfile.ZipFile(entry) as zf:
                xmls = [n for n in zf.namelist() if n.endswith(".xml")]
                if len(xmls) != 1:
                    raise SystemExit(f"{entry.name}: expected exactly one .xml, found {xmls}")
                yield entry.stem, ET.fromstring(zf.read(xmls[0]))
        elif entry.is_dir():
            xml_path = entry / "annotations.xml"
            if xml_path.exists():
                yield entry.name, ET.parse(xml_path).getroot()


def load_clips_registry(path: str) -> dict[str, dict[str, str]]:
    """Load the clip registry CSV, keyed by clip_id (full rows)."""
    registry: dict[str, dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            clip_id = (row.get("clip_id") or "").strip()
            if clip_id:
                registry[clip_id] = row
    return registry


def convert_to_canonical(
    annotations: list[dict], candidate_windows: dict[str, list[dict]]
) -> tuple[list[dict], list[dict], list[dict]]:
    """Convert parsed annotations to canonical events, ignores, and provenance."""
    events: list[dict] = []
    ignores: list[dict] = []
    provenance: list[dict] = []
    event_idx = 0
    ignore_idx = 0

    for ann in annotations:
        clip_id = ann["clip_id"]
        label = ann["label"]
        attrs = ann["attributes"]
        start_s = round(ann["start_s"], 3)
        end_s = round(ann["end_s"], 3)

        if end_s <= start_s:
            continue

        cands = candidate_windows.get(clip_id, [])
        matched = find_matching_candidate(start_s, end_s, cands)

        if label in ("pickup", "putdown"):
            item_count = int(attrs.get("item_count", 1))
            for i in range(item_count):
                event_idx += 1
                h = hashlib.sha256(f"{clip_id}:{start_s:.3f}:{label}:{i}".encode()).hexdigest()[
                    :12
                ]
                event_id = f"evt_{h}"

                events.append(
                    {
                        "event_id": event_id,
                        "clip_id": clip_id,
                        "type": label,
                        "t_start": start_s,
                        "t_end": end_s,
                        "hard_case": attrs.get("hard_case", "false") == "true",
                        "annotator": attrs.get("annotator") or ann.get("annotator", "unknown"),
                        "confidence": attrs.get("confidence", "med"),
                        "notes": attrs.get("notes", ""),
                    }
                )

                provenance.append(
                    {
                        "event_id": event_id,
                        "candidate_id": matched["candidate_id"] if matched else "",
                        "clip_id": clip_id,
                        "actor_id": matched.get("actor_id", "") if matched else "",
                        "hand_side": matched.get("hand_side", "") if matched else "",
                        "region_id": matched.get("region_id", "") if matched else "",
                        "event_group_id": f"grp_{event_idx // max(item_count, 1)}_{event_idx % max(item_count, 1)}",
                        "review_status": attrs.get("review_status", ""),
                    }
                )

        elif label == "ignore":
            ignore_idx += 1
            ignores.append(
                {
                    "ignore_id": f"ign_{ignore_idx:06d}",
                    "clip_id": clip_id,
                    "t_start": start_s,
                    "t_end": end_s,
                    "reason": attrs.get("ignore_reason", "UNLABELABLE"),
                    "annotator": attrs.get("annotator", "unknown"),
                    "notes": attrs.get("notes", ""),
                }
            )

    events.sort(key=lambda e: (e["clip_id"], e["t_start"], e["type"]))
    ignores.sort(key=lambda i: (i["clip_id"], i["t_start"]))

    return events, ignores, provenance


def write_events_csv(events: list[dict], path: str) -> None:
    columns = [
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for e in events:
            writer.writerow(e)


def write_ignore_parquet(ignores: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("ignore_id", pa.string()),
            ("clip_id", pa.string()),
            ("t_start", pa.float64()),
            ("t_end", pa.float64()),
            ("reason", pa.string()),
            ("annotator", pa.string()),
            ("notes", pa.string()),
        ]
    )
    if not ignores:
        table = pa.Table.from_pydict({name: [] for name in schema.names}, schema=schema)
    else:
        table = pa.Table.from_pylist(ignores)
    pq.write_table(table, path)


def write_provenance_parquet(provenance: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("event_id", pa.string()),
            ("candidate_id", pa.string()),
            ("clip_id", pa.string()),
            ("actor_id", pa.string()),
            ("hand_side", pa.string()),
            ("region_id", pa.string()),
            ("event_group_id", pa.string()),
            ("review_status", pa.string()),
        ]
    )
    if not provenance:
        table = pa.Table.from_pydict({name: [] for name in schema.names}, schema=schema)
    else:
        table = pa.Table.from_pylist(provenance)
    pq.write_table(table, path)


def write_clips_csv(clips: list[dict], path: str) -> None:
    if not clips:
        raise ValueError("no clip rows to write")
    columns = list(clips[0].keys())
    for row in clips[1:]:
        for key in row:
            if key not in columns:
                columns.append(key)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in sorted(clips, key=lambda r: r["clip_id"]):
            writer.writerow(row)


REVIEW_MANIFEST_COLUMNS = [
    "candidate_id",
    "clip_id",
    "review_groups",
    "video_path",
    "json_path",
    "event_count",
    "reviewed",
    "review_notes",
]


def build_candidate_review(
    annotations: list[dict],
    candidate_windows: dict[str, list[dict]],
    zero_event_clips: set[str],
    task_job_by_clip: dict[str, str],
    json_dir: Path,
    video_dir: str,
) -> tuple[list[dict], dict[str, dict], int]:
    """Map candidate windows onto clip-level CVAT annotations.

    - candidate overlapping an event track -> positive, JSON events =
      those tracks (candidate-relative, clamped to the candidate window)
    - candidate in a zero-event clip -> verified negative (empty events)
    - anything else (no overlap, ignore overlap) -> unlabeled, no row

    Returns (manifest rows, {candidate_id: json dict}, unlabeled count).
    """
    by_clip: dict[str, list[dict]] = {}
    for ann in annotations:
        by_clip.setdefault(ann["clip_id"], []).append(ann)

    rows: list[dict] = []
    jsons: dict[str, dict] = {}
    unlabeled = 0

    for clip_id in sorted(set(by_clip) | zero_event_clips):
        tracks = by_clip.get(clip_id, [])
        event_tracks = [a for a in tracks if a["label"] in ("pickup", "putdown")]
        zero_event = clip_id in zero_event_clips
        notes = task_job_by_clip.get(clip_id, "")
        for cand in candidate_windows.get(clip_id, []):
            cand_id = cand["candidate_id"]
            c_start = cand["source_start_s"]
            c_end = cand["source_end_s"]
            matched = [
                a
                for a in event_tracks
                if min(a["end_s"], c_end) - max(a["start_s"], c_start) > 0.01
            ]
            if matched:
                groups = "cvat_event"
            elif zero_event:
                groups = "cvat_zero_event"
            else:
                unlabeled += 1
                continue

            events = []
            for a in sorted(matched, key=lambda x: x["start_s"]):
                attrs = a["attributes"]
                events.append(
                    {
                        "label": a["label"],
                        "start_s": round(max(a["start_s"], c_start) - c_start, 3),
                        "end_s": round(min(a["end_s"], c_end) - c_start, 3),
                        "hard_case": attrs.get("hard_case", "false") == "true",
                        "confidence": attrs.get("confidence", "med"),
                        "notes": attrs.get("notes", ""),
                    }
                )

            jsons[cand_id] = {
                "candidate_id": cand_id,
                "clip_id": clip_id,
                "source_start_s": c_start,
                "events": events,
            }
            rows.append(
                {
                    "candidate_id": cand_id,
                    "clip_id": clip_id,
                    "review_groups": groups,
                    "video_path": f"{video_dir.rstrip('/')}/{clip_id}.mp4",
                    "json_path": str(json_dir / f"{cand_id}.json"),
                    "event_count": len(matched),
                    "reviewed": "true",
                    "review_notes": notes,
                }
            )

    return rows, jsons, unlabeled


def write_review_manifest(rows: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_reviewed_jsons(jsons: dict[str, dict], json_dir: Path) -> None:
    json_dir.mkdir(parents=True, exist_ok=True)
    for cand_id in sorted(jsons):
        (json_dir / f"{cand_id}.json").write_text(
            json.dumps(jsons[cand_id], indent=2) + "\n", encoding="utf-8"
        )


def validate_job_annotations(
    annotations: list[dict], registry: dict[str, dict[str, str]]
) -> list[str]:
    """Sanity gates for parsed job exports. Returns a list of error strings."""
    errors: list[str] = []
    for ann in annotations:
        clip_id = ann["clip_id"]
        row = registry.get(clip_id)
        if row is None:
            errors.append(f"{clip_id}: not in clip registry")
            continue
        if ann["label"] not in ("pickup", "putdown", "ignore"):
            errors.append(f"{clip_id}: unexpected label {ann['label']!r}")
            continue
        if not (0 <= ann["start_s"] < ann["end_s"]):
            errors.append(f"{clip_id}: bad interval [{ann['start_s']}, {ann['end_s']}]")
        duration = float(row.get("duration_s") or 0)
        if duration and ann["end_s"] > duration:
            errors.append(f"{clip_id}: end {ann['end_s']}s exceeds duration {duration}s")
        if ann["label"] in ("pickup", "putdown"):
            raw = ann["attributes"].get("item_count", "1")
            try:
                if int(raw) < 1:
                    errors.append(f"{clip_id}: item_count {raw} < 1")
            except ValueError:
                errors.append(f"{clip_id}: non-integer item_count {raw!r}")
    return errors


def run_dir_mode(args) -> None:
    """Convert a directory of per-job archives and write all canonical files."""
    registry = load_clips_registry(args.clips_csv)
    annotations: list[dict] = []
    counts: dict[str, int] = {}
    task_job: dict[str, str] = {}

    for stem, root in iter_job_exports(args.cvat_export_dir):
        clip_id = (root.findtext("meta/task/name") or "").strip()
        if clip_id not in registry:
            raise SystemExit(f"{stem}: clip {clip_id!r} not in {args.clips_csv}")
        fps = float(registry[clip_id]["fps"])
        anns = parse_cvat_1_1_job_export(root, fps)
        annotations.extend(anns)
        counts[clip_id] = len(anns)

    if args.manifest:
        with open(args.manifest, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                clip = Path(row["source_video_filename"]).stem
                if clip in counts and int(row["event_count"]) != counts[clip]:
                    raise SystemExit(
                        f"{clip}: {counts[clip]} tracks != manifest {row['event_count']}"
                    )
                m = re.match(r"^(.+)__task_(\d+)__job_(\d+)$", Path(row["s3_uri"]).name[:-4])
                if m:
                    task_job[clip] = f"cvat task {m.group(2)} job {m.group(3)}"

    errors = validate_job_annotations(annotations, registry)
    if errors:
        print(f"Validation failed with {len(errors)} error(s):")
        for err in errors:
            print(f"  {err}")
        raise SystemExit(1)

    zero_event_clips = {c for c, n in counts.items() if n == 0}

    candidate_windows = load_candidate_windows(args.candidate_meta_dir)
    events, ignores, provenance = convert_to_canonical(annotations, candidate_windows)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_events_csv(events, str(out / "events.csv"))
    write_ignore_parquet(ignores, str(out / "ignore_intervals.parquet"))
    write_provenance_parquet(provenance, str(out / "event_provenance.parquet"))

    clip_rows = []
    for clip_id in sorted(counts):
        row = dict(registry[clip_id])
        row["event_count"] = counts[clip_id]
        row["verified_negative"] = clip_id in zero_event_clips
        clip_rows.append(row)
    write_clips_csv(clip_rows, str(out / "clips.csv"))

    review_rows, jsons, unlabeled = build_candidate_review(
        annotations,
        candidate_windows,
        zero_event_clips,
        task_job,
        out / "reviewed_jsons",
        args.video_dir,
    )
    write_review_manifest(review_rows, str(out / "review_manifest.csv"))
    write_reviewed_jsons(jsons, out / "reviewed_jsons")

    from collections import Counter

    types = Counter(e["type"] for e in events)
    confs = Counter(e["confidence"] for e in events)
    print(f"\nParsed {len(annotations)} tracks from {len(counts)} job exports")
    print(f"By type: {dict(types)}")
    print(f"By confidence: {dict(confs)}")
    print(f"\nOutput ({out}):")
    print(
        f"  Events:        {len(events)} rows (from {len([a for a in annotations if a['label'] != 'ignore'])} tracks, item_count expanded)"
    )
    print(f"  Ignore:        {len(ignores)} rows")
    print(
        f"  Provenance:    {len(provenance)} rows, matched to candidates: "
        f"{sum(1 for p in provenance if p['candidate_id'])}/{len(provenance)}"
    )
    print(f"  Clips:         {len(clip_rows)} ({len(zero_event_clips)} verified-negative)")
    print(
        f"  Review manifest: {len(review_rows)} rows "
        f"({sum(1 for r in review_rows if r['review_groups'] == 'cvat_event')} positive, "
        f"{sum(1 for r in review_rows if r['review_groups'] == 'cvat_zero_event')} negative), "
        f"{unlabeled} candidates left unlabeled"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CVAT source-video export to canonical formats"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cvat-export", help="Path to a single CVAT 1.1 XML export")
    source.add_argument(
        "--cvat-export-dir", help="Directory of per-job archives (*.zip or <stem>/annotations.xml)"
    )
    parser.add_argument(
        "--candidate-meta-dir",
        default=".local/candidate_staging/candidates",
        help="Candidate metadata directory",
    )
    parser.add_argument(
        "--clips-csv",
        default=".local/task_7_vlm/clips.csv",
        help="Clip registry CSV with fps/duration_s "
        "(required for job exports, which carry no fps)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="export_manifest.csv for per-clip cross-checks and task/job notes",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Required with --cvat-export-dir; all canonical files are written under it",
    )
    parser.add_argument(
        "--video-dir",
        default=".local/source_videos",
        help="Video path prefix recorded in the review manifest",
    )
    parser.add_argument(
        "--events-output",
        default="annotation/exports/events.csv",
        help="Output path for canonical events.csv (single-file mode)",
    )
    parser.add_argument(
        "--ignore-output",
        default="annotation/exports/ignore_intervals.parquet",
        help="Output path for ignore_intervals.parquet (single-file mode)",
    )
    parser.add_argument(
        "--provenance-output",
        default="annotation/exports/event_provenance.parquet",
        help="Output path for event_provenance.parquet (single-file mode)",
    )
    args = parser.parse_args()

    if args.cvat_export_dir:
        if not args.output_dir:
            parser.error("--output-dir is required with --cvat-export-dir")
        run_dir_mode(args)
        return

    root = ET.parse(args.cvat_export).getroot()
    if root.tag == "annotations":
        clip_id = (root.findtext("meta/task/name") or "").strip()
        registry = load_clips_registry(args.clips_csv)
        if clip_id not in registry:
            raise SystemExit(
                f"{args.cvat_export}: clip {clip_id!r} not in {args.clips_csv}; "
                "job exports carry no fps"
            )
        annotations = parse_cvat_1_1_job_export(root, float(registry[clip_id]["fps"]))
    else:
        annotations = parse_cvat_1_1(args.cvat_export)

    candidate_windows = load_candidate_windows(args.candidate_meta_dir)
    print(f"Loaded candidate windows for {len(candidate_windows)} source videos")
    print(f"Parsed {len(annotations)} annotations from CVAT export")

    events, ignores, provenance = convert_to_canonical(annotations, candidate_windows)

    write_events_csv(events, args.events_output)
    write_ignore_parquet(ignores, args.ignore_output)
    write_provenance_parquet(provenance, args.provenance_output)

    matched = sum(1 for p in provenance if p["candidate_id"])
    print("\nOutput:")
    print(f"  Events:        {len(events)} rows -> {args.events_output}")
    print(f"  Ignore:        {len(ignores)} rows -> {args.ignore_output}")
    print(f"  Provenance:    {len(provenance)} rows -> {args.provenance_output}")
    print(f"  Matched to candidates: {matched}/{len(provenance)} events")

    from collections import Counter

    types = Counter(e["type"] for e in events)
    confs = Counter(e["confidence"] for e in events)
    print(f"\n  By type: {dict(types)}")
    print(f"  By confidence: {dict(confs)}")


if __name__ == "__main__":
    main()
