# Track B1 close-out (2026-09-11)

Summary of the Track B1 reproduction close-out and the deployment-input evaluation of the
reproduced checkpoints. Details, provenance and commands are in the two reports below.

| report | content |
|---|---|
| `TRACK_B1_REPRODUCTION_CVAT.md` | annotation-conditioned reproduction of the CVAT-branch results (frozen head, warm- and cold-head fine-tuning) |
| `TRACK_B1_DEPLOYMENT_EVAL.md` | the same reproduced checkpoints evaluated end to end on pose-pipeline (deployment) inputs |
| `../TRACK_B1_INTEGRATION.md` §10 | loss weighting of the two trainers |

Commits on `feature/unify_track_b_trainings_and_eval`: `3d22096` (reproduction cleanup),
`12e4871` (deployment-input evaluation).

## 1. Reproduction (annotation-conditioned)

* The frozen-head results reproduce exactly: val window macro F1 0.6147 (historical 0.615);
  val event F1 0.647 / 0.412 (default decode) and 0.585 / 0.523 (tuned) under
  per-event-group counting; test 0.361 / 0.278.
* Warm- and cold-head last-2-block fine-tuning approximately match historical val window
  performance (0.7348 and 0.7220 vs 0.743), but neither reproduces the historical
  fine-tuning experiment (best epoch, epoch-1 profile, event pairs, test row).
* The cold-head run does not support loss of the warm start as a sufficient explanation.
  With one run per initialisation, the warm/cold difference is not an estimate of
  seed-to-seed variance and cannot isolate the cause.
* The cross-day putdown weakness persists. Matching one historical metric in isolation is
  not a reproduction of an experiment.

## 2. Cleanup delivered

* **Counting policies** (`track_b1/inference.py`, `scripts/rescore_track_b1_events.py`):
  per-item primary, per-event-group for historical comparison; groups only from
  `event_group_id` scoped by clip; missing or ambiguous group metadata raises; predictions
  are never duplicated to fill item rows. Reproduces the historical frozen scores above.
* **Split-aware diagnostics** (`scripts/diagnose_track_b1.py`): per-split file names, no
  overwriting, stored checkpoint F1 compared only on the checkpoint's own validation data.
* **Loss weighting documented:** the frozen head combines class × confidence weights; the
  pixel trainer's loss replaces class weights with confidence weights and its balanced
  sampler is off (an inherited inconsistency). Follow-up experiment
  `track_b1_consistent_loss_weighting` specified, not run.
* Small provenance files and exact commands: `provenance/track_b1_repro_cvat_20260911/`.

## 3. Deployment-input evaluation

Per-item event F1 @ tIoU 0.3 / 0.5; same checkpoints, same clips, split registry
`cvat_b1_2026_09_09`. 20260526 is an already-inspected historical test day.

| checkpoint | split | annotation-conditioned | deployment, fixed decoder (A) | deployment, calibrated on deployment val (B) |
|---|---|---|---|---|
| frozen head | val | 0.567 / 0.507 | 0.000 / 0.000 | 0.065 / 0.032 (tuned on this split) |
| frozen head | test | 0.361 / 0.278 | 0.101 / 0.014 | 0.087 / 0.029 |
| fine-tuned (warm head) | val | 0.771 / 0.743 | 0.000 / 0.000 | 0.000 / 0.000 (degenerate grid) |
| fine-tuned (warm head) | test | 0.529 / 0.471 | 0.078 / 0.039 | 0.073 / 0.000 |

* Neither checkpoint transfers; calibration does not recover them. Test FP/h is 49–161.
* Test-day proposals cover all 35 events and 28 have an own-actor window; 24 of those 28
  are still missed. On val, 15 of 34 item rows have no own-actor window (4 no candidate,
  6 unresolved association, 5 other-actor-only).
* Deployment crops contain the annotated box (median 1.0) but it fills a median 2.4 % (val)
  / 4.8 % (test) of the crop, against 59 % in the annotation crops the models were trained
  on. That shift is not yet isolated from candidate timing and window context, so the loss is
  not attributed to crops alone.
* Annotation-conditioned scores are not production performance.

## 4. Open items

* `track_b1_crop_factorization` (next): on the val day, pose candidates × CVAT-box crops and
  CVAT candidates × pose-box crops, same checkpoints and decoders, to separate crop from
  candidate effects.
* `track_b1_consistent_loss_weighting`: multi-seed retraining under one weighting policy.
* A multi-seed fine-tuning comparison to resolve the historical fine-tuning discrepancy.
* A clean held-out estimate needs new recording days.
