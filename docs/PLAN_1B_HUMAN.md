# Plan: restore S3 data + Track B1 (VideoMAE) training on this device

> **Status:** not started. **Branch:** `feature/human_annotation_track_b`.
> **Goal:** download all S3 data/models to this (fresh) device, build the human-annotation
> dataset, then train + evaluate Track B1 on human GT first, VLM GT second.

## Repo state

- Branch `feature/human_annotation_track_b` (1 commit ahead of master, clean tree). `.local/`
  absent — this device has no data at all.
- "Track 1a" = Track A (state machine), "track 1b" = Track B1 (VideoMAE window classifier,
  `src/pickup_putdown/layer1/track_b1/`). B1 is fully implemented (dataset/train/inference)
  but was never trained here; no trained checkpoints exist anywhere.
- B1 training entry (src/pickup_putdown/layer1/track_b1/train.py:803):

  ```bash
  python -m pickup_putdown.layer1.track_b1.train \
    <candidates.parquet> <events.csv> <clips.csv> <video_dir> <pose_tracks_dir> \
    <shelves.yaml> <output_dir> [--ignore-intervals-path ...]
  ```

  Needs: candidates with `clip_id/candidate_id/actor_id/region_id/window_start_s/window_end_s`;
  events with `event_id/clip_id/type/t_start/t_end/confidence`; `clips.csv` with `split` +
  `s3_key`; source videos as `<clip_id>.mp4`; pose parquets **flat** at
  `pose_tracks_dir/<clip_id>.parquet`.
- CVAT→canonical converter (`scripts/convert_cvat_source_export.py`) already produces exactly
  the B1-compatible schemas (events.csv + ignore_intervals.parquet) — verified
  column-for-column. No adapter code needed.
- Trained Track A artifacts from the dead device are **not** on S3 (per `docs/LOCAL.md`:
  "No classifier artifacts, no feature datasets, no trained checkpoints"). Rebuildable later
  if needed; not needed for B1.

## Phase 0 — Environment (verify, ~10 min)

1. `df -h .` — check free disk vs. S3 total size (step 1a). **Stop and ask if < ~1.5× total.**
2. `nvidia-smi` — GPU required for pose generation + training.
3. AWS: creds in env vars. Verify: `aws s3 ls s3://chillnbite-cameras/anon/` (bucket region
   `eu-central-1` — set `AWS_REGION` if unset).
4. Fresh venv: `python -m venv .venv && .venv/bin/pip install -e ".[dev,track_b1]"`
   (torch CUDA build if GPU).
5. `make models` — YOLO11n person + pose detectors (~10 MB each, from ultralytics, not S3).
6. VideoMAE-small pulls from HuggingFace automatically on first model load (~90 MB).

## Phase 1 — S3 inventory + staged downloads (one monitored background job)

**1a. Inventory first** (resolves the two unknown prefixes: CVAT raw export path and
"track A1 results" path; expected CVAT prefix from docstrings:
`anon/annotations/cvat_2026-09-07/`):

```bash
mkdir -p .local
aws s3 ls s3://chillnbite-cameras/anon/ --recursive > .local/s3_inventory.txt
aws s3 ls s3://chillnbite-cameras/anon/ --recursive --summarize
```

**1b. One script, sequential steps, nohup + log** (repo pattern, survives SSH disconnect):
new `scripts/download_all_staged.sh` with `set -Eeuo pipefail`, steps:

| # | Command (sync → local) | Size |
|---|---|---|
| 1 | `anon/<cvat export prefix>/` → `.local/annotations/cvat_2026-09-07/` | small |
| 2 | `anon/vlm/2026-06-26/{vlm_annotations,task_7_vlm,task_7_review}/` → `.local/vlm_annotations/`, `.local/task_7_vlm/`, `.local/task_7_review/` | small (~MBs) |
| 3 | Source videos **referenced by the CVAT export** (clip list from export manifest + `s3_key` in `clips.csv` + `source_key` in candidate metadata) → `.local/source_videos/` | large, first |
| 4 | Track A1 results prefix (if exists) → `.local/track_a1_results/` | unknown |
| 5 | `anon/candidates/` → `.local/candidate_staging/` (1527 candidate MP4s + 75 metadata JSONs) | medium |
| 6 | Remaining root `anon/*.mp4` → `.local/source_videos/`; then any leftover prefixes → `.local/s3_mirror/` | large, last |

Run: `nohup bash scripts/download_all_staged.sh > .local/downloads.log 2>&1 &`.
`aws s3 sync` is resumable/skips-existing, so steps are idempotent.
Monitor: `tail -f .local/downloads.log`.

While steps 1–2 land, work continues (phases 2–3 only need the small files + referenced
source videos).

## Phase 2 — Human dataset from CVAT export (~5 min)

Layout fix first: converter expects metadata as `<dir>/<clip_id>/<clip_id>.json`
(scripts/convert_cvat_source_export.py:48); S3 stores flat `metadata/<clip_id>.json` →
restructure into `.local/candidate_staging/candidates/<clip_id>/<clip_id>.json` (also the
layout Track A CLIs expect).

```bash
python scripts/convert_cvat_source_export.py \
  --cvat-export-dir .local/annotations/cvat_2026-09-07/unzipped \
  --clips-csv .local/task_7_vlm/clips.csv \
  --manifest .local/annotations/cvat_2026-09-07/raw/export_manifest.csv \
  --output-dir .local/task_7_human
```

→ `.local/task_7_human/{events.csv, ignore_intervals.parquet, event_provenance.parquet,
clips.csv, review_manifest.csv, reviewed_jsons/}`. Built-in validation gates (clip registry
membership, duration bounds, manifest cross-check) run automatically.

**Risk handled at impl time:** if the CVAT export contains clips missing from
`task_7_vlm/clips.csv`, the converter exits. Fix: extend the clips registry with
`video_probe` for the missing clips (existing `pickup_putdown.ingestion.video_probe`),
re-run.

## Phase 3 — Track B1 data prep (new small script `scripts/prepare_track_b1_data.py`)

1. Read `clips.csv` (human) → clips in `train`/`val` splits.
2. For each such clip:
   `make tasks-3-5 VIDEO=.local/source_videos/<clip>.mp4 RUN_ID=b1_<clip>` (existing
   Makefile target; runs YOLO triage + pose, GPU), then copy:
   - `.../task_5/tracks_pose.parquet` → `.local/track_b1_data/pose_tracks/<clip_id>.parquet`
     (flat layout B1 requires)
   - `.../task_5/candidates.parquet` → `.local/track_b1_data/candidates_perclip/<clip_id>.parquet`
3. Build `.local/track_b1_data/candidates.parquet` from the **S3 metadata JSONs** (canonical:
   `source_start_s`→`window_start_s`, `source_end_s`→`window_end_s`, `actor_id`, `region_id`),
   cross-checked against regenerated per-clip task_5 parquets (same candidate ids/counts;
   mismatch → hard error, per repo convention). Also write
   `.local/track_b1_data/candidates_val.parquet` (val-clip candidates only, for Phase 5 eval).
4. Copy `events_human.csv` (= `task_7_human/events.csv`), `events_vlm.csv`
   (= `task_7_vlm/events.csv`), `ignore_intervals.parquet`, `clips.csv` into
   `.local/track_b1_data/`.
5. Print summary: candidates/clip, per-split window label distribution, missing videos/poses.

~80 lines, stdlib + pandas only. This is the only non-trivial new code.

## Phase 4 — Train B1 (human first, then VLM)

```bash
# Run 1: human GT
python -m pickup_putdown.layer1.track_b1.train \
  .local/track_b1_data/candidates.parquet .local/track_b1_data/events_human.csv \
  .local/track_b1_data/clips.csv .local/source_videos .local/track_b1_data/pose_tracks \
  configs/shelves.yaml .local/track_b1_output_human \
  --ignore-intervals-path .local/track_b1_data/ignore_intervals.parquet
# Run 2: VLM GT → .local/track_b1_output_vlm (events_vlm.csv, no ignore intervals)
```

- Gate B (tiny overfit) runs automatically before each full run.
- Hyperparams = `TrainConfig` defaults (frozen backbone, lr 1e-4, 20 epochs, batch 8) —
  matches task_12 step 7 ("start frozen, unfreeze later"). **Known wart:**
  `configs/track_b1.yaml` is nested but `train.py`'s loader reads flat keys, so the yaml is
  silently ignored (train.py:840-845). No src change now; actual hyperparams are recorded in
  `training_results.json`. Unfreeze-last-2-blocks run = follow-up decision after run 1
  stabilizes.
- Runs go under nohup + log (hours on GPU), one at a time.

## Phase 5 — Inference + evaluation (val split)

```bash
python -m pickup_putdown.layer1.track_b1.inference \
  .local/track_b1_output_human/checkpoints/best_model.pt \
  .local/track_b1_data/candidates_val.parquet .local/track_b1_data/clips.csv \
  .local/source_videos .local/track_b1_data/pose_tracks configs/shelves.yaml \
  .local/track_b1_output_human/predictions_val.csv
```

Then Task-8 shared evaluator (`pickup_putdown.evaluation.aggregate_metrics`) on predictions
vs `.local/task_7_human/events.csv` (filtered to val clips) → `metrics.json`
(P/R/F1 @ tIoU 0.3/0.5). Same for the VLM run. Report both side by side.

## Files changed

| File | Change |
|---|---|
| `scripts/download_all_staged.sh` | **new** (~50 lines, bash strict mode, sequential `aws s3 sync` steps, log lines) |
| `scripts/prepare_track_b1_data.py` | **new** (~80 lines, stdlib + pandas) |
| `Makefile` | **no change** (reuse `tasks-3-5`, `models`) |
| `src/**` | **no changes** |

## Tests

- No unit tests added (both scripts are glue). Checks instead:
  - `shellcheck scripts/download_all_staged.sh`
  - `ruff check scripts/prepare_track_b1_data.py && ruff format --check scripts/prepare_track_b1_data.py`
  - `prepare_track_b1_data.py` self-checks: split membership, non-empty train+val manifests,
    label distribution printed, pose files present for every candidate clip — exits non-zero
    on any violation.
  - Real-data smoke: build manifest → `len > 0` and label counts match events (sanity:
    pickup+putdown windows ≈ event count × ~5 windows/event).
- Regression gate (no src changes, should be green): `python -m pytest`, `ruff check .`,
  `ruff format --check .`.

## Verification (per phase)

- **P0:** `df -h .`, `nvidia-smi`, `aws s3 ls` success,
  `.venv/bin/python -c "import torch; print(torch.cuda.is_available())"`.
- **P1:** per-step log lines; final `find .local -name '*.mp4' | wc -l` vs inventory;
  re-run sync = "0 transferred".
- **P2:** converter prints event/ignore/manifest counts;
  `head .local/task_7_human/events.csv` shows `t_start/t_end/type/confidence`; compare event
  count to Track A run-2 on the dead device if the A1 results sync reveals it.
- **P3:** summary printout; spot-check one pose parquet has `person_bbox_x1..y2` + `actor_id`.
- **P4:** Gate B passes; `training_results.json` val F1; `best_model.pt` exists.
- **P5:** `predictions_val.csv` non-empty; `metrics.json` written for both runs.

## Risks

1. **Disk:** full bucket total unknown until inventory; largest sources are 2.9 GB each.
   Mitigation: `df -h` gate + staged order (small→large); if tight, stop and ask before step 6.
2. **No GPU / small GPU** on this box: pose generation over ~15–30 clips and 20-epoch VideoMAE
   training need CUDA; CPU fallback is impractical. Check in P0 before committing.
3. **CVAT export layout** (zip vs unzipped, exact prefix) unknown until inventory — converter
   handles both zip and unzipped dirs; prefix resolved from 1a.
4. **CVAT clips outside `task_7_vlm/clips.csv` registry** → converter hard-exits; fix = extend
   registry via existing video probe (small, documented above).
5. **Candidate drift:** locally regenerated task_5 candidates should match S3 metadata JSONs;
   prep script cross-checks and fails loudly on mismatch (S3 JSONs are source of truth).
6. **Track A1 results prefix may not exist** on S3 (docs say no trained artifacts uploaded).
   If absent: nothing lost for B1; Track A artifacts are rebuildable from downloaded data
   later.
7. **`configs/track_b1.yaml` silently ignored** by flat config loader — runs use `TrainConfig`
   defaults; provenance recorded in results JSON; fix deferred.
8. **WSL path:** `.local/` stays on the Linux FS (`/home/ubuntu/repos/...`), not `/mnt/c`.

## Skipped (ponytail)

- No `track-b1-train`/`infer` CLI commands or Makefile targets — module entry points exist
  and are enough for two runs. Add when this becomes a repeated workflow.
- No code fix for the nested-yaml config mismatch, no unfreeze run, no Track A rebuild — all
  follow-ups, not blockers.
