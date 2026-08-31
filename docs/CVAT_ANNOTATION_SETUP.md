# CVAT Annotation — Source-Video Workflow

> Annotate directly on source videos. Candidate windows are reference overlays.
> No timestamp conversion needed — annotations are already in source-video time.

## Why source-video annotation

| | Per-candidate | Per-source-video |
|---|---|---|
| Tasks | 1527 | 42 |
| Timestamp conversion | required (candidate-relative → source) | none |
| Context | trimmed 4–10 s window | full active span visible |
| Events outside candidates | missed | caught naturally |
| Task mapping sidecar | required | not needed |

---

## 1. Candidate windows as reference

Each candidate metadata file contains `source_start_s` / `source_end_s` —
time windows into the source video where the pose pipeline flagged a possible
interaction.

```
.local/candidate_staging/candidates/<source_id>/
  <source_id>.json        ← candidate windows (source_start_s, source_end_s)
.local/source_videos/
  <source_id>.mp4         ← full source video
```

These windows serve as **annotation guides**, not constraints. Annotator
reviews the full active span and creates, corrects, or deletes events freely.

### Generate candidate window summary per source

```bash
python3 -c "
import json
from pathlib import Path

base = Path('.local/candidate_staging/candidates')
for src_dir in sorted(base.iterdir()):
    meta = src_dir / (src_dir.name + '.json')
    if not meta.exists():
        continue
    data = json.loads(meta.read_text())
    cands = data.get('candidates', [])
    if not cands:
        continue
    windows = []
    for c in sorted(cands, key=lambda x: x['source_start_s']):
        windows.append(f\"  [{c['source_start_s']:.1f}–{c['source_end_s']:.1f}s] {c['candidate_id']}  {c.get('actor_id','?')} {c.get('hand_side','?')} {c.get('region_id','?')}\")
    print(f\"\\n=== {src_dir.name} ({len(cands)} candidates) ===\")
    print('\\n'.join(windows[:5]))
    if len(windows) > 5:
        print(f'  ... and {len(windows)-5} more')
"
```

Use this output to populate the **task description** in CVAT for each source
video, giving annotators the candidate window list as reference.

---

## 2. CVAT project setup

### Create project

1. Log in to CVAT.
2. **New project** → name: `pickup_putdown_source_annotation`.
3. Description: `Temporal annotation on source videos. Candidate windows in task description are reference only.`

### Labels (project-level)

| Label | Color | Description |
|-------|-------|-------------|
| `pickup` | `#4CAF50` | Person removes item from shelf/surface |
| `putdown` | `#2196F3` | Person places item onto shelf/surface |
| `ignore` | `#FF9800` | Transfer evidence unavailable |

### Per-label attributes

**pickup** and **putdown** (add on each label):

| Name | Type | Values | Required |
|------|------|--------|----------|
| `confidence` | Select | `high`, `med`, `low` | yes |
| `hard_case` | Select | `false`, `true` | yes |
| `item_count` | Number | 1–10, default 1 | yes |
| `review_status` | Select | `draft`, `reviewed`, `accepted`, `needs_adjudication` | yes |
| `notes` | Text | free text | no |

**ignore**:

| Name | Type | Values | Required |
|------|------|--------|----------|
| `ignore_reason` | Select | `ACTION_OCCLUDED`, `ACTION_OUT_OF_FRAME`, `CLIP_BOUNDARY`, `UNLABELABLE`, `CORRUPT_SECTION` | no |
| `review_status` | Select | `draft`, `reviewed`, `accepted`, `needs_adjudication` | yes |
| `notes` | Text | free text | no |

---

## 3. Upload source videos and create tasks

### Option A: S3 direct import

1. **Project → Data → Import from storage**.
2. Bucket: `chillnbite-cameras`.
3. Prefix: `anon/source/` (adjust to match your actual source video location).
4. One task per video, frame step = 1.
5. For each task, paste the candidate window summary (from §1) into the
   task description before assigning.

### Option B: Local upload

```bash
# 42 source videos with candidate metadata, all present locally:
ls .local/source_videos/D2_S*.mp4
```

Upload in batches via **Project → Data → Upload files**. Create one task
per video.

### Task naming convention

```
<source_video_id>
```

Example: `D2_S20260520135131_E20260520135549_anon`

This matches the `clip_id` in the canonical schema, so no ID remapping is
needed at export time.

---

## 4. Annotation procedure

### Per task (one source video)

1. **Read the task description** — candidate windows are listed as reference.
2. **Review the active span** — watch from first person appearance to last.
3. **Mark events on the timeline**:
   - Drag to create a temporal segment at the event location.
   - Assign label: `pickup`, `putdown`, or `ignore`.
   - Fill attributes: confidence, hard_case, item_count, review_status.
4. **Events inside candidate windows**: Annotator confirms or corrects.
5. **Events outside candidate windows**: Annotator adds freely. These are
   valid events — the candidate generator missed them, not the annotator.
6. **Zero-event source video**: Leave timeline empty. Add "no_event" to
   task comments to mark as reviewed.
7. **Multiple events in one candidate window**: Create separate segments.
   A single candidate window can contain pickup then putdown.

### Confidence rules

| Condition | Label | Confidence |
|-----------|-------|------------|
| Visible transfer, clear | pickup / putdown | high |
| Visible transfer, likely | pickup / putdown | med |
| Visible transfer, uncertain | pickup / putdown | low |
| Fully occluded / out of frame | ignore | — |

Low-confidence visible events are OFFICIAL events. Ignore is only for
unobservable actions.

### Special cases

- **Immediate pickup → putdown**: Two separate ordered segments.
- **Simultaneous items**: One segment, `item_count = N`.
- **Multiple people**: Annotate each transfer independently.

---

## 5. Export from CVAT

1. **Project → Annotations → Export**.
2. Format: **CVAT 1.1** (preserves attributes, one XML file).
3. Download to `annotation/cvat_export_source.xml`.

---

## 6. Convert to canonical format

Since annotations are on source videos, timestamps are already canonical.
The conversion script:

1. Parses CVAT 1.1 XML export.
2. Extracts `clip_id` from task/video name.
3. Converts frame numbers to seconds using source video FPS.
4. Matches each event to its candidate(s) for provenance.
5. Writes `events.csv`, `ignore_intervals.parquet`, and
   `event_provenance.parquet`.

```bash
python scripts/convert_cvat_source_export.py \
  --cvat-export annotation/cvat_export_source.xml \
  --candidate-meta-dir .local/candidate_staging/candidates \
  --events-output annotation/exports/events.csv \
  --ignore-output annotation/exports/ignore_intervals.parquet \
  --provenance-output annotation/exports/event_provenance.parquet
```

See `scripts/convert_cvat_source_export.py` for the full script.

### Run the conversion

```bash
python convert_cvat_source_export.py \
  --cvat-export annotation/cvat_export_source.xml \
  --candidate-meta-dir .local/candidate_staging/candidates \
  --events-output annotation/exports/events.csv \
  --ignore-output annotation/exports/ignore_intervals.parquet \
  --provenance-output annotation/exports/event_provenance.parquet
```

---

## 7. Validation

```bash
python3 -c "
import csv
rows = list(csv.DictReader(open('annotation/exports/events.csv')))
print(f'Total events: {len(rows)}')
from collections import Counter
print(f'  Pickup:  {Counter(r[\"type\"] for r in rows).get(\"pickup\", 0)}')
print(f'  Putdown: {Counter(r[\"type\"] for r in rows).get(\"putdown\", 0)}')
bad = [r for r in rows if float(r['t_start']) >= float(r['t_end'])]
print(f'Bad timestamps (start >= end): {len(bad)}')
bad_conf = [r for r in rows if r['confidence'] not in ('high','med','low')]
print(f'Missing/invalid confidence: {len(bad_conf)}')
"
```

---

## 8. Files

| File | Commit? | Notes |
|------|---------|-------|
| `docs/CVAT_ANNOTATION_SETUP.md` | yes | This guide |
| `convert_cvat_source_export.py` | yes | Conversion script |
| `annotation/cvat_export_source.xml` | no | CVAT export |
| `annotation/exports/events.csv` | no | Canonical events |
| `annotation/exports/ignore_intervals.parquet` | no | Ignore intervals |
| `annotation/exports/event_provenance.parquet` | no | Candidate traceability |
