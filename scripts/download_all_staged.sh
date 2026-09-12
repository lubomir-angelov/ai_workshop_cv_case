#!/usr/bin/env bash
# Staged S3 download for Track B1 (docs/PLAN_1B_HUMAN.md phase 1b).
# Small files first, large videos last; idempotent (sync/cp skip existing).
#
# Run:      nohup bash scripts/download_all_staged.sh > .local/downloads.log 2>&1 &
# Monitor:  tail -f .local/downloads.log
set -Eeuo pipefail

BUCKET="s3://chillnbite-cameras/anon"
export AWS_REGION="${AWS_REGION:-eu-central-1}"
LOCAL=".local"
MISSING="$LOCAL/missing_videos.txt"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >&2; }
trap 'log "ERROR: failed at line $LINENO"' ERR

mkdir -p "$LOCAL"

log "step 1/6: CVAT exports (2026-09-07 + 2026-09-09, ~5 MB)"
aws s3 sync "$BUCKET/annotations/cvat" "$LOCAL/annotations/cvat"

log "step 2/6: VLM results (~5 MB)"
aws s3 sync "$BUCKET/vlm/2026-06-26/vlm_annotations" "$LOCAL/vlm_annotations"
aws s3 sync "$BUCKET/vlm/2026-06-26/task_7_vlm" "$LOCAL/task_7_vlm"
aws s3 sync "$BUCKET/vlm/2026-06-26/task_7_review" "$LOCAL/task_7_review"

log "step 3a/6: candidate metadata (75 JSONs)"
aws s3 sync "$BUCKET/candidates/metadata" "$LOCAL/candidate_staging/candidates"

log "step 3b/6: referenced source videos"
# Referenced clips = export manifests + candidate metadata + VLM registry,
# in that priority order (B1 only needs the manifest clips).
: > "$MISSING"
collect_clips() {
    python3 - "$1" <<'EOF'
import csv
import json
import sys
from pathlib import Path

label = sys.argv[1]
out = []

def add(key, name):
    key = key.strip()
    if key:
        out.append((key, Path(name).name))

if label == "manifest":
    for p in sorted(Path(".local/annotations/cvat").glob("*/raw/export_manifest.csv")):
        for row in csv.DictReader(p.open()):
            fn = Path(row["source_video_filename"]).name
            add(fn, fn)
elif label == "metadata":
    for p in sorted(Path(".local/candidate_staging/candidates").glob("*/*.json")):
        data = json.loads(p.read_text())
        vid = data.get("source_video_id") or p.parent.name
        add(vid + ".mp4", vid + ".mp4")
elif label == "registry":
    # registry s3_keys carry a stale "source_videos/" prefix; actual objects are
    # at the bucket root, so resolve bare names against the local inventory
    inv_root = set()
    inv = Path(".local/s3_inventory.txt")
    if inv.is_file():
        inv_root = {
            Path(line.split()[-1]).name
            for line in inv.read_text().splitlines()
            if line.strip()
        }
    p = Path(".local/task_7_vlm/clips.csv")
    if p.exists():
        for row in csv.DictReader(p.open()):
            key = (row.get("s3_key") or row.get("clip_id") or "").strip()
            if key and not key.endswith(".mp4"):
                key += ".mp4"
            if key and Path(key).name in inv_root:
                key = Path(key).name
            if key:
                add(key, Path(key).name)

for key, name in out:
    print(f"{key}\t{name}")
EOF
}
mkdir -p "$LOCAL/source_videos"
for src in manifest metadata registry; do
    while IFS=$'\t' read -r key name; do
        if [ -f "$LOCAL/source_videos/$name" ]; then
            continue
        fi
        case "$key" in
            s3://*) uri="$key" ;;
            *) uri="$BUCKET/$key" ;;
        esac
        if aws s3 cp "$uri" "$LOCAL/source_videos/$name" --only-show-errors; then
            log "  downloaded $name"
        else
            log "  MISSING $key"
            echo "$key" >> "$MISSING"
        fi
    done < <(collect_clips "$src" | sort -u)
done

log "step 4/6: track A1 results - prefix absent from S3 inventory, skipping"

log "step 5/6: candidate videos (1527 MP4s, ~32 GB)"
aws s3 sync "$BUCKET/candidates/videos" "$LOCAL/candidate_staging/videos"

log "step 6/6: remaining root source videos (~275 GB)"
aws s3 sync "$BUCKET/" "$LOCAL/source_videos/" \
    --include "*.mp4" \
    --exclude "vlm/*" --exclude "candidates/*" --exclude "annotations/*"

if [ -s "$MISSING" ]; then
    log "ERROR: referenced source videos missing from S3:"
    cat "$MISSING" >&2
    exit 1
fi
log "DONE: all steps completed, all referenced videos present"
