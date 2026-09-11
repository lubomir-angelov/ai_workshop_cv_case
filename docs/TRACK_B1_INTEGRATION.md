# Track B1 integration: CVAT supervision, two input modes

This branch (`feature/unify_track_b_trainings_and_eval`) combines the two Track B1
(VideoMAE window classifier, task_12) routes into one implementation. It is the
reference for how the combined pipeline works, what its numbers mean, and how to run it.
Older route-specific write-ups (`TRACK_B1_CVAT.md`,
`results/TRACK_B1_APPROACH_COMPARISON.md`, `results/TRACK_B1_METRICS_EXPLAINED.md`,
`results/TRACK_1B_HUMAN_ANNOTATION.md`) are kept as historical records and carry
correction notes pointing here.

## 1. Provenance

| Source | Commit | Role |
|---|---|---|
| this branch before integration | `9a10b5d` | base |
| `origin/feature/cvat_annotation` | `3e0e747` (local branch was `dde4ab9`, one commit behind) | primary implementation baseline, merged first (`9e86ed7`) |
| `origin/feature/human_annotation_track_b` | `c847509` (local = remote) | fixes, diagnostics, pose data path, merged second (`f2000f7`) |

Integration changes are committed on top of the two merges. Merge conflicts:
`annotation/import_export.py` (deleted here with Label Studio, modified on the CVAT
branch: deletion kept) and the sample-weight transfer in `track_b1/train.py` (kept both
the float32 cast and `non_blocking`). A stray `from xml.parsers.expat import model`
from the human branch was dropped.

## 2. What each branch contributed and how overlaps were resolved

| Capability | Kept from | Resolution |
|---|---|---|
| CVAT export ingestion, interrupted tracks → separate intervals, ffprobe-measured fps | CVAT | kept; `item_count = N` now expands to N canonical rows (see §4.1) |
| Annotation-conditioned candidates/crops, held-box padding, verified-negative sampling | CVAT | kept as `--input-mode annotation` |
| Actor-aware window labelling | CVAT | kept, made identity-safe (§4.2) |
| Sequential per-candidate decode + frame cache | CVAT | kept; entries now fingerprinted and rebuilt when stale (§5) |
| Frozen-encoder embedding cache | CVAT | kept; keyed by window, label-free, tied to encoder weights + preprocessing, refused for fine-tuning (§5) |
| Balanced tiny-overfit (Gate B), visual inspection (Gate A) | CVAT | kept; Gate B no longer discards a warm-started head; Gate A renders the training path and can diff it against a fresh source decode |
| Last-2-block fine-tuning, warm-started head, head/backbone learning rates | CVAT | kept (`train_track_b1.py`), now also on deployment inputs |
| `boundary_mode: window_centers`, shared Task 8 evaluation at tIoU 0.3/0.5 | CVAT | kept; decode/evaluate moved into `inference.py` and shared by inference and tuning |
| Day-based splits | CVAT (ranking) + human (hash) | replaced by an explicit registry, `configs/track_b1_splits.yaml`, with both historical registries recorded (§4.4) |
| Clip-ID normalisation and validation | human | `cvat_import.normalize_clip_id`; the dataset builder and `build_window_manifest` reject unknown clip ids |
| Pose-derived candidates and crops | human | kept as `--input-mode deployment` (reads `scripts/prepare_track_b1_data.py` output) |
| Strict pretrained encoder loading, legacy attention-bias conversion | human | kept and made layout-adaptive (§6) |
| Seek-once decoding, per-window frame cache, worker/progress/timing logs | human | kept; the per-window cache key is extended but stays valid for existing entries (§5) |
| Per-window prediction export, confusion matrix, per-class metrics, review manifest | human | `scripts/diagtnostics_track_b1.py` → `scripts/diagnose_track_b1.py`, now for either mode plus the legacy baseline layout, with crop/sampling evidence columns |
| Source-versus-crop inspection | — | no such tool was committed on either branch; `inspect_track_b1_windows.py --compare-source` and the diagnostics evidence columns cover the check |

One implementation per component: a single dataset builder, preprocessing function
(`dataset.prepare_window_frames`, which the candidate cache reproduces bit-for-bit,
asserted by a test), model, trainer pair (embeddings head / pixel path), decoder and
evaluator. `InferenceWindowDataset` now delegates to the shared preprocessing and no
longer substitutes black frames on decode failure.

## 3. The two input modes

CVAT exports are the only supervision in both modes: event types, timestamps, boxes,
ignore intervals, review status, and completed clips with no events.

| | annotation (`.local/track_b1_dataset`) | deployment (`.local/track_b1_dataset_deploy`) |
|---|---|---|
| candidates (who/when) | padded extent of each CVAT track + verified-negative intervals | Task 5 pose proposals |
| crop (where) | CVAT box ∪ over the candidate, fixed per candidate, `INTER_AREA` | pose person box ∪ over each window, `INTER_LINEAR` |
| needs pose | no | yes (`prepare_track_b1_data.py`, runs tasks 3-5 only where outputs are missing) |
| uses GT at inference | **yes**, where and roughly when to look | no |
| labels | CVAT track id = candidate actor | CVAT events associated to pose actors (§4.2) |
| valid for | classifier diagnostics, supervised training | deployment-style end-to-end evaluation, training on realistic inputs |

Evaluation limits:

* **Annotation-mode event scores are annotation-conditioned.** Candidates exist exactly
  where events are, so recall is not limited by proposals and false positives are
  limited to ±3 s around real events and a few borrowed-box negatives. These numbers are
  upper bounds on classifier quality given perfect conditioning, never deployment
  performance. `infer_track_b1.py` labels them as such in its output and `metrics_*.json`.
* **Deployment-mode evaluation keeps every reviewed event.** Ground truth is all CVAT
  event rows of the split's clips (including events no proposal covers and clips with
  no events), so proposal and association misses are false negatives.
* **Coverage is reported separately** from classifier and event metrics:
  `candidate_coverage.parquet` per dataset (why each event can or cannot be learned or
  detected) and the `candidate_coverage` block in every `metrics_<split>.json`.
* **Compare model variants only within one mode, split registry, window config and
  decode config.** The modes crop differently, so the same checkpoint sees different
  inputs in each. The dataset's `build_metadata.json` records everything that must match.

## 4. Data semantics

### 4.1 Events

* `item_count = N` → N canonical rows with one interval, a shared `event_group_id` and
  `item_index` (docs/LABELING_GUIDELINES.md §5.1; the evaluator's multi-item recall
  relies on it). The CVAT branch collapsed these into one row: the 2026-09-09 export
  gives 283 event rows (179 pickup / 104 putdown) in 259 groups, where the CVAT
  documents report 259 events. Window labels are unaffected (same interval, same actor);
  event-level recall denominators change, so CVAT-branch event metrics are not directly
  comparable to new runs.
* The older converter that produced `.local/track_b1_data/events_human.csv` (281 rows)
  treats a CVAT track as one interval from its first to last box, including an
  `outside` box; the importer splits interrupted tracks and ends at the last inside
  frame (1 track in this export is interrupted).
* Ignore intervals: windows whose centre falls in one are excluded from manifests;
  events and predictions overlapping one are dropped by the shared evaluator.
* Seven completed clips have no events. They produce verified-negative candidates in
  annotation mode and contribute their duration to FP/hour in both modes.

### 4.2 Actor identity

CVAT interaction-track ids (`trk007`, one per interaction) and pose ids (`actor_12`,
`<clip>:person:3`, plus pooled `*untracked` ids) are different identity systems.

* `build_window_manifest` raises when event and candidate actor ids share no values,
  instead of silently labelling every window background (which the CVAT branch's
  actor filter would have done on pose candidates).
* Deployment mode associates each CVAT event with a pose actor
  (`actor_association.associate_events`): the fraction of pose sample times, over the
  event ±0.25 s, with a confident wrist (≥0.3) inside the CVAT box widened by 25%.
  Matched needs ≥0.3 and a 0.15 lead over the runner-up; otherwise `ambiguous`,
  `unmatched` or `no_pose`. Provenance (`cvat_actor_id`, scores, runner-up) is kept in
  `actor_association.parquet` and `events_pose_identity.parquet`.
* A window takes a label only from its own actor's events. A window centred inside an
  event whose association failed is **excluded**, not labelled background; pooled
  `*untracked` candidates are excluded inside every event.
* Clip-level labelling (events without `actor_id`) is still accepted for reproducing the
  historical baseline, and logs how many overlapping candidate pairs of different actors
  are exposed to cross-labelling (134 among the baseline's 410 candidates).

Measured on the 2026-09-09 export with the existing pose outputs (1.5 s windows, default
registry): association 197 matched / 69 unmatched / 14 no pose / 3 ambiguous of 283
rows; matched events lead the runner-up by a median 0.83. Coverage: 140 rows labelled,
86 association failed, 46 `other_actor_only` (the wrist-evidence actor has no
overlapping candidate while another actor's does), 6 `no_window_centre`, 5 with no
candidate at all. Annotation mode labels all 283 by construction.

### 4.3 Crops and shelf regions

`configs/shelves.yaml` nests regions under `cameras.<camera>.regions`, but every pose
route reader looked for a top-level `regions` key, so **no shelf region was ever applied**:
historical pose-route crops are the person box + 15% margin (the comparison document's
"pose box ∪ shelf region" is inaccurate). `load_shelf_regions` now reads the real schema
and fails on an empty file; including the shelf polygon is an explicit
`include_shelf_region` setting, default off to match what was run. Turning it on would
enlarge crops considerably (Shelf_01 covers roughly half the frame) and is unevaluated.

### 4.4 Splits

`configs/track_b1_splits.yaml` holds named day→split registries; the builder fails on
any day not listed.

| registry | train | val | test | status |
|---|---|---|---|---|
| `cvat_b1_2026_09_09` (default) | 0520, 0521, 0522 | 0523 | 0526 | historical; reproduces the CVAT ranking exactly |
| `human_b1_2026_09_09` | 0520–0523 | 0526 | — | historical; the frozen-head baseline |

**20260526 is not an untouched test day.** It is the baseline's validation day (model
selection and manual window review) and the CVAT branch read it twice as test. For
combined development treat it as a second development day; a clean held-out estimate
needs new recording days or grouped cross-validation over days. Splits were not
reshuffled.

## 5. Preprocessing and caches

`WindowConfig` (window, stride, frames, size, margin, `crop_scope`,
`resize_interpolation`, `include_shelf_region`) is set once per dataset from
`configs/track_b1.yaml` (`window` + `input_modes.<mode>`, overridable on the builder CLI)
and stored in `build_metadata.json`; cache, embeddings, training, inference, diagnostics
and inspection all read it from there via `dataset_dir.open_window_dataset`.
`preprocessing_spec` (with `PREPROCESSING_VERSION`) lists every setting that affects
pixels and is part of every cache key and run record. Labels are never part of a cache.

| cache | key / validity | invalidated by |
|---|---|---|
| per-candidate crops (`.local/track_b1_cache`, annotation) | fingerprint in each entry: video path+size+mtime, SHA-256 of the actor's box rows, candidate span, preprocessing | any of those; rebuilt by `build_track_b1_cache.py`, and every consumer refuses stale or missing entries (`find_stale_entries`, `require_complete`) |
| per-window crops (`.local/track_b1_frame_cache`, deployment) | SHA-256 of video and pose-file identity, actor, window, shelf region, preprocessing | any of those. Window-scope / linear / no-shelf settings keep the human branch's v1 key, so the existing 14 GB of entries stay valid (verified byte-identical to a fresh decode) |
| embeddings (`<dataset>/embeddings`) | `windows.parquet` keys (clip, candidate, actor, window) + pretrained-weights SHA-256 + preprocessing | different weights or preprocessing, missing windows, or any fine-tuned blocks (raises: fine-tuning always uses pixels) |

Deployment datasets reference pose tracks in place (recorded as `tracks_dir`) unless
their clip ids needed normalising, because the per-window key includes the pose file's
identity.

## 6. Encoder loading

The MCG-NJU pretraining checkpoints store attention biases as `q_bias`/`v_bias`.
transformers ≥5 builds `query/key/value.bias`, and `VideoMAEModel.from_pretrained`
then reports those as MISSING and initialises them fresh (verified with
`MCG-NJU/videomae-base` and transformers 5.16.1: loaded `query.bias` all zeros against a
checkpoint mean |q_bias| of 0.26). The CVAT branch avoided this by pinning
transformers <5 in a Mac venv; the human branch converted explicitly.

The combined loader (`videomae_classifier.convert_encoder_state`) converts to whatever
layout the installed transformers builds (legacy layout passes through), sets the key
bias to zero (what the legacy layout computes), discards the decoder and loads with
`strict=True`. On this machine (torch 2.14 + CUDA, transformers 5.16.1) all 184
encoder tensors match the checkpoint (160 identical, 24 renamed biases). The weights'
SHA-256 is recorded on the model, in checkpoints and in embedding metadata. No
environment change is needed on either platform.

## 7. Results and their status

### 7.1 Verified: the historical frozen-head baseline (reproduced)

Pose-route inputs, clip-level labels from `events_human.csv`, `human_b1_2026_09_09`
split, 2.5 s / 0.5 s windows, frozen VideoMAE-base + head, best epoch 6.
Reproduced on the integrated code (`diagnose_track_b1.py --legacy-data-dir`), identical
to the checkpoint's stored metrics (difference 5.6e-17):

| validation windows | macro F1 | background F1 | pickup F1 | putdown F1 |
|---|---|---|---|---|
| 1,391 | 0.4240668 | 0.8982630 | 0.2612613 | 0.1126761 |

Confusion matrix (rows = true, columns = predicted; background, pickup, putdown):

| true \ pred | background | pickup | putdown |
|---|---|---|---|
| background | 1086 | 103 | 95 |
| pickup | 39 | 29 | 14 |
| putdown | 9 | 8 | 8 |

Evidence on the same windows (CVAT boxes vs the crop actually used): 17 of 107 event
windows keep under half of the annotated box inside the crop; the four reviewed windows
noted as "event outside of crop" score 0.00, 0.09, 0.31 and 0.00, the reviewed
"hand and item visible" ones 0.87–1.00. Every labelled event window has ≥5 of its 16
sampled frames inside the annotated event. 17 windows fell back to a full-frame crop
(no pose box for the actor inside the window).

### 7.2 Smoke evaluations on the integrated pipeline (not model results)

Same baseline checkpoint, deployment inputs, `human_b1_2026_09_09` validation day
(the checkpoint's own selection day), 2.5 s / 0.5 s windows:

| | value |
|---|---|
| window macro F1 with actor-resolved CVAT labels (1,372 windows: 1294 / 55 / 23) | 0.4177 (background 0.9056, pickup 0.2316, putdown 0.1159) |
| event F1 @0.3 / @0.5, historical decode (0.5 thresholds, smoothing 3, window_span) | 0.000 / 0.000 — no window's pickup or putdown probability exceeds 0.49 |
| event F1 @0.3 / @0.5, configured decode (0.40 / 0.45, smoothing 5, window_centers) | 0.043 / 0.043 |
| event F1 @0.3 / @0.5, thresholds tuned on this same day (optimistic) | 0.154 / 0.092 |
| candidate coverage of the day's 35 event rows | 26 labelled, 7 association failed, 2 no window centre |

The window-level number differs from 7.1 only through label semantics and exclusions,
so the two are not a controlled comparison of anything but labelling. Inference and
diagnostics produce the same probabilities for shared windows (max difference 7e-8).

Annotation mode: the built dataset reproduces the CVAT branch's 1.5 s validation
manifest exactly (973 windows, 814 background). Candidate-cache pixels equal a fresh
source decode (max difference 0), embeddings (973 × 768) and head training run end to
end; those smoke scores come from a 4-clip scratch split and mean nothing.

### 7.3 Reported by the CVAT branch, not reproduced here

Annotation-conditioned, `cvat_b1_2026_09_09`, run on a Mac with transformers <5. Window
macro F1: frozen probe 0.615, fine-tuned (last 2 blocks) 0.743. Fine-tuned validation
event F1 is reported as **0.806 / 0.746** (tIoU 0.3 / 0.5; `TRACK_B1_CVAT.md`,
`configs/track_b1.yaml`) and as **0.783 / 0.638** (comparison and metrics documents).
**Unresolved:** the run directories that would settle it (`.local/track_b1_run*`,
`.local/track_b1_finetune`, `.local/track_b1_dataset_w15`) are not on this machine.
Plausibly (unverified) the first pair is after validation threshold tuning and the second
before it. These scores also predate the item-count expansion (§4.1) and are
annotation-conditioned; do not compare them with deployment-mode or baseline numbers.

## 8. Commands

All commands run from the repository root in the project environment. The Makefile
mirrors them (`make <target> TRACK_B1_MODE=annotation|deployment`).

```bash
# --- data preparation -------------------------------------------------------
bash scripts/download_all_staged.sh                 # S3 → .local (CVAT exports, videos, metadata)
python scripts/build_track_b1_dataset.py --input-mode annotation \
    --output-dir .local/track_b1_dataset            # no pose needed
python scripts/prepare_track_b1_data.py             # pose per clip; skips clips with outputs (GPU)
python scripts/build_track_b1_dataset.py --input-mode deployment \
    --pose-data-dir .local/track_b1_data --output-dir .local/track_b1_dataset_deploy
# options: --split-registry human_b1_2026_09_09, --window-duration-s/--window-stride-s,
#          --crop-margin, --accepted-only, --config <yaml with input_modes overrides>

# --- caching + Gate A -------------------------------------------------------
python scripts/build_track_b1_cache.py --dataset-dir .local/track_b1_dataset --workers 8
python scripts/build_track_b1_cache.py --dataset-dir .local/track_b1_dataset_deploy --workers 8
python scripts/inspect_track_b1_windows.py --dataset-dir .local/track_b1_dataset \
    --split train --compare-source --output-dir .local/track_b1_inspection_annotation
python scripts/precompute_track_b1_embeddings.py --dataset-dir .local/track_b1_dataset
python scripts/precompute_track_b1_embeddings.py --dataset-dir .local/track_b1_dataset_deploy

# --- frozen-head training (Gate B runs first) -------------------------------
python scripts/train_track_b1_head.py --dataset-dir .local/track_b1_dataset \
    --output-dir .local/track_b1_run_annotation
python scripts/train_track_b1_head.py --dataset-dir .local/track_b1_dataset_deploy \
    --output-dir .local/track_b1_run_deployment

# --- fine-tuning the last two encoder blocks (pixel path, warm-started head) -
python scripts/train_track_b1.py --dataset-dir .local/track_b1_dataset \
    --unfreeze-last-n-blocks 2 --backbone-lr 5e-5 --learning-rate 1e-3 \
    --init-head-from .local/track_b1_run_annotation/checkpoints/head_best.pt \
    --output-dir .local/track_b1_finetune_annotation --epochs 20 --patience 5
#   (same with .local/track_b1_dataset_deploy / _deployment for deployment inputs)

# --- evaluation --------------------------------------------------------------
python scripts/infer_track_b1.py --dataset-dir .local/track_b1_dataset_deploy \
    --checkpoint .local/track_b1_finetune_deployment/checkpoints/best_model.pt --split val
python scripts/tune_track_b1_thresholds.py --dataset-dir .local/track_b1_dataset_deploy \
    --predictions-dir .local/track_b1_finetune_deployment/predictions_deployment
python scripts/infer_track_b1.py --dataset-dir .local/track_b1_dataset \
    --checkpoint .local/track_b1_run_annotation/checkpoints/head_best.pt --split val

# --- diagnostics -------------------------------------------------------------
python scripts/diagnose_track_b1.py --dataset-dir .local/track_b1_dataset_deploy \
    --checkpoint .local/track_b1_run_deployment/checkpoints/head_best.pt \
    --output-dir .local/track_b1_run_deployment/diagnostics
python scripts/diagnose_track_b1.py --legacy-data-dir .local/track_b1_data \
    --checkpoint .local/track_b1_output_human/checkpoints/best_model.pt \
    --evidence-dataset-dir .local/track_b1_dataset_deploy \
    --output-dir .local/track_b1_output_human/diagnostics_repro
```

To evaluate an annotation-trained checkpoint on deployment inputs with the *same*
preprocessing it was trained on, build a deployment dataset with a config whose
`input_modes.deployment` sets `crop_scope: candidate` and `resize_interpolation: area`;
otherwise the comparison mixes a change of candidates with a change of crop.

## 9. Remaining issues

### Verified (fixed on this branch unless stated)

1. `from_pretrained` under transformers ≥5 re-initialises VideoMAE attention biases (§6).
2. The CVAT actor filter would have labelled every pose-candidate window background
   (identity systems never match). Now association + a raising identity check.
3. Clip-level labels let one actor's event label another actor's crop; the historical
   baseline was trained this way (134 overlapping cross-actor candidate pairs in its
   candidates). Kept only for reproduction.
4. Gate B re-created the model, silently discarding `--init-head-from` unless Gate B was
   skipped. Whether the CVAT fine-tuning run was affected depends on flags not recorded
   here.
5. Shelf regions were never applied (wrong YAML key); documents claimed otherwise (§4.3).
   Not "fixed" in behaviour: kept off and made explicit.
6. `item_count` collapsed into one event row (§4.1).
7. Gate A inspected a different preprocessing path than CVAT training used; now the same
   path, with an exact source comparison.
8. The candidate cache could serve stale pixels after annotation, video or setting
   changes; embeddings were aligned by row position with labels stored alongside.
9. Overlapping same-actor pose candidates (131 pairs) duplicate detections; same-type
   overlapping predictions of one actor are now suppressed and counted.
10. The baseline head never exceeds 0.49 on event classes, so the historical decode
    emits no events (§7.2). Not fixed: a model property.
11. 17 baseline validation windows silently fell back to a full-frame crop. Not fixed;
    flagged by `crop_is_full_frame` in diagnostics.
12. An interrupted smoke run left 888 orphaned per-window cache entries (≈2.1 GB) in
    `.local/track_b1_frame_cache`, keyed to a scratch copy of the pose tracks and written
    on 2026-09-11 between 10:12 and 10:31; nothing will ever hit them. Not deleted here.

### Hypotheses (unverified)

* The 46 `other_actor_only` events reflect pose track fragmentation / id switches
  between proposal generation and the event, rather than association errors.
* The unresolved 0.806/0.746 vs 0.783/0.638 reflects before/after threshold tuning.
* Crop tightness (CVAT hand/item box vs pose person box) is the largest non-conditioning
  contributor to the gap between routes; enabling shelf union would widen it further.
* Transfers falling between sampled frames matter at the instant of contact; the
  event-interval evidence here (≥5 sampled frames per labelled window) neither shows nor
  rules that out. Lengthening windows at 16 frames lowers the sampling rate and is
  unevaluated.
* The CVAT branch's cross-day pickup/putdown direction collapse (its test day) was not
  reproduced here.

### Not done here, by request

No full training run, full-dataset pose regeneration or bulk video download. The
presentation, paper-figure, results-report and reproducibility-bundle scripts from the
CVAT branch still point at its historical run directories.
