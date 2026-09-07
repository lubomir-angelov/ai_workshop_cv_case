#!/usr/bin/env python3
"""Convert CVAT 1.1 source-video export to canonical formats.

Annotations are on full source videos, so timestamps are already canonical.
Candidate metadata is loaded for provenance matching only.

Usage:
  python scripts/convert_cvat_source_export.py \
    --cvat-export annotation/cvat_export_source.xml \
    --events-output annotation/exports/events.csv \
    --ignore-output annotation/exports/ignore_intervals.parquet \
    --provenance-output annotation/exports/event_provenance.parquet
"""

import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
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
            cands.append({
                "candidate_id": c["candidate_id"],
                "source_start_s": c["source_start_s"],
                "source_end_s": c["source_end_s"],
                "actor_id": c.get("actor_id"),
                "hand_side": c.get("hand_side"),
                "region_id": c.get("region_id"),
            })
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

            results.append({
                "clip_id": clip_id,
                "label": label,
                "start_s": start_s,
                "end_s": end_s,
                "fps": fps,
                "attributes": attributes,
            })

    return results


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
                h = hashlib.sha256(
                    f"{clip_id}:{start_s:.3f}:{label}:{i}".encode()
                ).hexdigest()[:12]
                event_id = f"evt_{h}"

                events.append({
                    "event_id": event_id,
                    "clip_id": clip_id,
                    "type": label,
                    "t_start": start_s,
                    "t_end": end_s,
                    "hard_case": attrs.get("hard_case", "false") == "true",
                    "annotator": attrs.get("annotator", "unknown"),
                    "confidence": attrs.get("confidence", "med"),
                    "notes": attrs.get("notes", ""),
                })

                provenance.append({
                    "event_id": event_id,
                    "candidate_id": matched["candidate_id"] if matched else "",
                    "clip_id": clip_id,
                    "actor_id": matched.get("actor_id", "") if matched else "",
                    "hand_side": matched.get("hand_side", "") if matched else "",
                    "region_id": matched.get("region_id", "") if matched else "",
                    "event_group_id": f"grp_{event_idx // max(item_count, 1)}_{event_idx % max(item_count, 1)}",
                })

        elif label == "ignore":
            ignore_idx += 1
            ignores.append({
                "ignore_id": f"ign_{ignore_idx:06d}",
                "clip_id": clip_id,
                "t_start": start_s,
                "t_end": end_s,
                "reason": attrs.get("ignore_reason", "UNLABELABLE"),
                "annotator": attrs.get("annotator", "unknown"),
                "notes": attrs.get("notes", ""),
            })

    events.sort(key=lambda e: (e["clip_id"], e["t_start"], e["type"]))
    ignores.sort(key=lambda i: (i["clip_id"], i["t_start"]))

    return events, ignores, provenance


def write_events_csv(events: list[dict], path: str) -> None:
    columns = [
        "event_id", "clip_id", "type", "t_start", "t_end",
        "hard_case", "annotator", "confidence", "notes",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for e in events:
            writer.writerow(e)


def write_ignore_parquet(ignores: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        ("ignore_id", pa.string()), ("clip_id", pa.string()),
        ("t_start", pa.float64()), ("t_end", pa.float64()),
        ("reason", pa.string()), ("annotator", pa.string()),
        ("notes", pa.string()),
    ])
    if not ignores:
        table = pa.Table.from_pydict({name: [] for name in schema.names}, schema=schema)
    else:
        table = pa.Table.from_pylist(ignores)
    pq.write_table(table, path)


def write_provenance_parquet(provenance: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        ("event_id", pa.string()), ("candidate_id", pa.string()),
        ("clip_id", pa.string()), ("actor_id", pa.string()),
        ("hand_side", pa.string()), ("region_id", pa.string()),
        ("event_group_id", pa.string()),
    ])
    if not provenance:
        table = pa.Table.from_pydict({name: [] for name in schema.names}, schema=schema)
    else:
        table = pa.Table.from_pylist(provenance)
    pq.write_table(table, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CVAT source-video export to canonical formats"
    )
    parser.add_argument("--cvat-export", required=True,
                        help="Path to CVAT 1.1 XML export")
    parser.add_argument("--candidate-meta-dir",
                        default=".local/candidate_staging/candidates",
                        help="Candidate metadata directory")
    parser.add_argument("--events-output",
                        default="annotation/exports/events.csv",
                        help="Output path for canonical events.csv")
    parser.add_argument("--ignore-output",
                        default="annotation/exports/ignore_intervals.parquet",
                        help="Output path for ignore_intervals.parquet")
    parser.add_argument("--provenance-output",
                        default="annotation/exports/event_provenance.parquet",
                        help="Output path for event_provenance.parquet")
    args = parser.parse_args()

    candidate_windows = load_candidate_windows(args.candidate_meta_dir)
    print(f"Loaded candidate windows for {len(candidate_windows)} source videos")

    annotations = parse_cvat_1_1(args.cvat_export)
    print(f"Parsed {len(annotations)} annotations from CVAT export")

    events, ignores, provenance = convert_to_canonical(
        annotations, candidate_windows
    )

    write_events_csv(events, args.events_output)
    write_ignore_parquet(ignores, args.ignore_output)
    write_provenance_parquet(provenance, args.provenance_output)

    matched = sum(1 for p in provenance if p["candidate_id"])
    print(f"\nOutput:")
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
