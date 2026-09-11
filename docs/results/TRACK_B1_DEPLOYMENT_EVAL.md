# Track B1: reproduced checkpoints on deployment inputs (2026-09-11)

The two reproduced checkpoints from `TRACK_B1_REPRODUCTION_CVAT.md`, unchanged, evaluated
end to end on deployment inputs: pose-pipeline candidates and crops, with CVAT used only
to score and diagnose. Same source clips and split registry as the reproduction.

**Read first**

* Annotation-conditioned scores (CVAT candidates and crops at inference) are upper bounds on
  classifier quality, not production performance. The deployment-input numbers below are the
  end-to-end measurement; they are not a production estimate either (one camera, 12 test
  clips, 0.63 h of test footage).
* **20260526 (test) is an already-inspected historical test day**, not a fresh one (it was the
  `human_b1_2026_09_09` validation day, read twice by the CVAT branch and once per model by
  the reproduction). Nothing here was selected on it.
* No retraining, no weight change, no checkpoint selection on test. Per-item counting is
  primary; per-event-group is supplementary. The cold-head ablation was not evaluated (its
  annotation-mode results are preserved in the reproduction report).

## Summary

| checkpoint | split | annotation-conditioned F1 @0.3 / @0.5 | deployment, fixed decoder (A) | deployment, deployment-val calibrated (B) |
|---|---|---|---|---|
| frozen head | val | 0.567 / 0.507 | 0.000 / 0.000 | 0.065 / 0.032 (tuned on this split) |
| frozen head | test | 0.361 / 0.278 | 0.101 / 0.014 | 0.087 / 0.029 |
| fine-tuned (warm head, last 2 blocks) | val | 0.771 / 0.743 | 0.000 / 0.000 | 0.000 / 0.000 (tuned on this split; degenerate) |
| fine-tuned (warm head, last 2 blocks) | test | 0.529 / 0.471 | 0.078 / 0.039 | 0.073 / 0.000 |

Per-item event F1 at tIoU 0.3 / 0.5, same checkpoints, same footage, same evaluator.

* Neither checkpoint transfers to deployment inputs. On validation, both produce no
  correct detection with their own decoders; on test they recover 3–7 of 35 events at tIoU
  0.3, at 49–161 false positives per hour.
* Decoder calibration on deployment validation does not rescue either model. The frozen
  head's best grid point reaches 0.048 mean validation F1; for the fine-tuned model every
  one of the 200 grid points scores 0, so its "calibrated" decoder is the tie-break corner
  and carries no information.
* Test-day proposals cover every event temporally (35/35) and 28/35 have a window of the
  associated actor centred inside. The losses there are mainly classification failures on
  events the pipeline did present to the model (24 of those 28 missed at tIoU 0.3, fixed
  decoder, both models). On validation, 15/34 item rows had no own-actor window
  (4 without any candidate, 6 unresolved association, 5 other-actor-only).
* The most visible input difference is crop scale. Deployment crops contain the annotated
  interaction box (median 1.0) but the box fills a median 2–5 % of the crop, against 59 %
  in the annotation crops the checkpoints were trained on. That is evidence of a large
  shift, not a proof that crop scale alone causes the loss (candidate timing, window context
  and other-actor windows also change); see Limitations.

## Provenance

| | |
|---|---|
| code | commit `3d22096` plus `configs/track_b1_deployment_candidate_crop.yaml`, `scripts/coverage_track_b1.py` and the `event_coverage` / `event_box_share_of_crop` additions to `track_b1/actor_association.py`, committed unchanged with this report (sha256s in `provenance/track_b1_deploy_eval_20260911/code_state.txt`). Inference, decoding and evaluation code are those of `3d22096` |
| frozen head | `.local/track_b1_repro_cvat_20260911/frozen_head/checkpoints/head_best.pt`, sha256 `83d613f8…c0ba` (epoch 37) |
| fine-tuned | `.local/track_b1_repro_cvat_20260911/finetune_last2/checkpoints/best_model.pt`, sha256 `42f15aec…5bcd` (epoch 8, warm head) |
| input mode | deployment: Task 5 pose proposals (`.local/track_b1_data/candidates.parquet`, generated from pose tracks and shelf regions; no annotation input) and pose person boxes |
| preprocessing | as the checkpoints were trained (`TRACK_B1_INTEGRATION.md` §8): 1.5 s windows, stride 0.25 s, 16 frames, 224², margin 0.15, `crop_scope: candidate`, `INTER_AREA`, no shelf region, preprocessing v2. Crop box = pose person boxes of the candidate's actor, unioned over the candidate |
| dataset | `.local/track_b1_deploy_eval_20260911/dataset` (built from the same CVAT export `2026-09-09/raw`, `accepted_only: false`, 283 event rows, as the reproduction) |
| split registry | `cvat_b1_2026_09_09`: val 20260523 (7 clips, 0.333 h), test 20260526 (12 clips, 0.632 h); clips never overlap in time |
| counting policy | per-item primary; per-event-group supplementary |
| decoders | A: each checkpoint's annotation-validation selection (`frozen_selection.json`). B: frozen in `deploy_calibration_selection.json` (15:23:28, before any test inference; read-only) |
| protocol | `evaluation_protocol.json`, written before any deployment prediction existed |

Copies of the protocol, selection, runner, commands, dataset metadata, calibration sweeps,
all metrics/coverage summaries and the table compiler are in
`docs/results/provenance/track_b1_deploy_eval_20260911/`.

### Protocol

1. Validation inference with decoder A for both checkpoints; coverage on validation.
2. Calibration B on the validation window scores only: grid pickup × putdown thresholds
   {0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7}², smoothing {3, 5}, merge gap
   0.75 s, minimum duration 0.3 s, `window_centers`; objective mean per-item F1 at tIoU 0.3
   and 0.5 after duplicate suppression; ties to lower smoothing, then lower thresholds
   (`tune_track_b1_thresholds.py` defaults, the rule used for decoder A). Selection frozen.
3. Validation decode with B; then test inference with A and with B; coverage for each.
   Nothing was changed after test.

Selected decoders: frozen head A 0.45 / 0.60 / 3, B **0.70 / 0.30 / 5** (pickup threshold at
the grid's upper edge); fine-tuned A 0.50 / 0.20 / 3, B **0.20 / 0.20 / 3** (all 200 points
tied at 0; tie-break corner).

Every eligible deployment candidate of the split was processed (val 84, test 96 candidates;
none lacked windows; 2184 / 3031 inference windows), including those that match no event.
No candidate was dropped for failed CVAT-to-pose matching. Ground truth is every reviewed
CVAT event row of the split's clips, including events with no proposal or an unresolved
actor; none of them overlaps an ignore interval. Window scores are bit-identical between
the A and B runs of each checkpoint (max |Δp| = 0).

## Event-level results

Per-item counting. FP/h uses the split's total footage. Boundary MAE is conditional on
events matched by midpoint (±1 s) and is not comparable across rows with different
match sets.

### Validation (20260523)

| checkpoint | input | decoder (pickup / putdown / smoothing) | predictions | P / R / F1 @0.3 | P / R / F1 @0.5 | pickup / putdown F1 @0.5 | FP/h @0.3 / @0.5 | start / end MAE s | per-event-group F1 @0.3 / @0.5 |
|---|---|---|---|---|---|---|---|---|---|
| frozen head | annotation-conditioned | A (0.45 / 0.60 / 3) | 33 | 0.576 / 0.559 / **0.567** | 0.515 / 0.500 / **0.507** | 0.558 / 0.417 | 42.1 / 48.1 | 0.46 / 0.53 | 0.585 / 0.523 |
| frozen head | deployment | A fixed (0.45 / 0.60 / 3) | 65 | 0.000 / 0.000 / **0.000** | 0.000 / 0.000 / **0.000** | 0.000 / 0.000 | 195.3 / 195.3 | 1.50 / 1.30 | 0.000 / 0.000 |
| frozen head | deployment | B calibrated (0.70 / 0.30 / 5) | 28 | 0.071 / 0.059 / **0.065** | 0.036 / 0.029 / **0.032** | 0.000 / 0.071 | 78.1 / 81.1 | 1.08 / 0.27 | 0.067 / 0.033 |
| fine-tuned | annotation-conditioned | A (0.50 / 0.20 / 3) | 36 | 0.750 / 0.794 / **0.771** | 0.722 / 0.765 / **0.743** | 0.826 / 0.583 | 27.0 / 30.1 | 0.22 / 0.25 | 0.794 / 0.765 |
| fine-tuned | deployment | A fixed (0.50 / 0.20 / 3) | 32 | 0.000 / 0.000 / **0.000** | 0.000 / 0.000 / **0.000** | 0.000 / 0.000 | 96.2 / 96.2 | — | 0.000 / 0.000 |
| fine-tuned | deployment | B calibrated (0.20 / 0.20 / 3) | 41 | 0.000 / 0.000 / **0.000** | 0.000 / 0.000 / **0.000** | 0.000 / 0.000 | 123.2 / 123.2 | — | 0.000 / 0.000 |

B rows on validation are selected on this split and therefore optimistic.

### Test (20260526, already-inspected historical day)

| checkpoint | input | decoder (pickup / putdown / smoothing) | predictions | P / R / F1 @0.3 | P / R / F1 @0.5 | pickup / putdown F1 @0.5 | FP/h @0.3 / @0.5 | start / end MAE s | per-event-group F1 @0.3 / @0.5 |
|---|---|---|---|---|---|---|---|---|---|
| frozen head | annotation-conditioned | A (0.45 / 0.60 / 3) | 37 | 0.351 / 0.371 / **0.361** | 0.270 / 0.286 / **0.278** | 0.316 / 0.133 | 38.0 / 42.7 | 0.97 / 0.85 | 0.361 / 0.278 |
| frozen head | deployment | A fixed (0.45 / 0.60 / 3) | 103 | 0.068 / 0.200 / **0.101** | 0.010 / 0.029 / **0.014** | 0.016 / 0.000 | 151.8 / 161.3 | 0.94 / 0.82 | 0.101 / 0.014 |
| frozen head | deployment | B calibrated (0.70 / 0.30 / 5) | 34 | 0.088 / 0.086 / **0.087** | 0.029 / 0.029 / **0.029** | 0.039 / 0.000 | 49.0 / 52.2 | 0.42 / 0.55 | 0.087 / 0.029 |
| fine-tuned | annotation-conditioned | A (0.50 / 0.20 / 3) | 33 | 0.545 / 0.514 / **0.529** | 0.485 / 0.457 / **0.471** | 0.500 / 0.375 | 23.7 / 26.9 | 0.35 / 0.36 | 0.529 / 0.471 |
| fine-tuned | deployment | A fixed (0.50 / 0.20 / 3) | 67 | 0.060 / 0.114 / **0.078** | 0.030 / 0.057 / **0.039** | 0.045 / 0.000 | 99.6 / 102.8 | 0.60 / 0.43 | 0.078 / 0.039 |
| fine-tuned | deployment | B calibrated (0.20 / 0.20 / 3) | 74 | 0.054 / 0.114 / **0.073** | 0.000 / 0.000 / **0.000** | 0.000 / 0.000 | 110.7 / 117.0 | 0.98 / 0.84 | 0.073 / 0.000 |

Test has no multi-item events, so the two counting policies agree there. No deployment run on
the test day detects a single putdown at tIoU 0.5.

**A versus B.** Calibration trades false positives for recall on the frozen head (test
FP/h @0.5 161 → 52, recall @0.3 0.200 → 0.086) without improving F1 @0.3 (0.101 → 0.087);
@0.5 changes from 0.014 to 0.029, one true positive either way. For the fine-tuned model
the calibration had no signal to select on.

## Coverage (deployment inputs)

GT counts: val 34 item rows / 32 event groups / 32 distinct intervals; test 35 / 35 / 35.

Non-exclusive flags (item rows; event groups in brackets):

| | val (34 [32]) | test (35 [35]) |
|---|---|---|
| proposal temporal coverage: any candidate of any actor overlaps the event | 30 [29] | 35 [35] |
| a window of any actor is centred inside the event | 30 [29] | 35 [35] |
| association matched | 25 [24] | 28 [28] |
| association unmatched / no pose / ambiguous | 5 / 4 / 0 [5 / 3 / 0] | 4 / 2 / 1 [4 / 2 / 1] |
| own-actor coverage: associated actor's candidate overlaps | 19 of 25 matched [18 of 24] | 28 of 28 [28 of 28] |
| own-actor window centred inside | 19 of 25 [18 of 24] | 28 of 28 [28 of 28] |

Partition (`actor_association.COVERAGE_PARTITION`; an event takes the **first** category
that applies, temporal proposal checks before association, so an association failure is
never counted as a proposal miss). Counts reconcile with the totals in every run.

| category (precedence order) | val items [groups] | test items [groups] |
|---|---|---|
| 1. `no_candidate` — no candidate of any actor overlaps | 4 [3] | 0 |
| 2. `no_window_centre` — candidates overlap, no window centred inside | 0 | 0 |
| 3. `association_unresolved` — proposals exist, actor unmatched / ambiguous / no pose | 6 [6] | 7 [7] |
| 4. `other_actor_only` — matched actor has no overlapping candidate, others do | 5 [5] | 0 |
| 5. `own_no_window_centre` — matched actor's candidate overlaps, none of its windows centred inside | 0 | 0 |
| 6. `own_covered` — a window of the matched actor is centred inside | 19 [18] | 28 [28] |
| total | 34 [32] | 35 [35] |

Of the 4 val `no_candidate` rows, 3 also failed association and 1 was matched. Association
failures are unresolved diagnostics: they may be pose misses, hands out of view or
association-rule misses, and are not counted as proposal misses. Scoring is actor-agnostic
(clip, type and time only), so events outside `own_covered` can still be matched by another
candidate's prediction; the categories describe the model's opportunity, not a hard ceiling.

### Crops

Share of the annotated interaction box inside the model's crop, and share of the crop that
box occupies, for windows centred inside the event (own actor), medians over events:

| | box inside crop | box share of crop | crop share of frame |
|---|---|---|---|
| deployment, val `own_covered` (19) | 1.00 (1 event < 0.5) | **0.024** | 0.19 |
| deployment, test `own_covered` (28) | 1.00 (2 events < 0.5) | **0.048** | 0.25 |
| annotation inputs, same val events (32 groups) | 1.00 by construction | **0.592** | 0.014 |
| annotation inputs, same test events (35) | 1.00 by construction | **0.592** | 0.016 |

No deployment crop fell back to full frame. For val events outside `own_covered`, the best
crop of any actor centred in the event contains a median 0.0 of the box (the proposals
present are of people not doing the interaction); for test `association_unresolved` events
it is 1.0 (someone's crop shows it). Visual check (`inspection_val/` vs
`inspection_val_annotation_reference/`): deployment windows show the whole person and
several metres of shelf; annotation windows show the hand and item.

Within deployment `own_covered` test events, matching is rare at every containment level
(fixed decoder, tIoU 0.3: frozen head 0/3, 2/4, 2/21; fine-tuned 0/3, 1/4, 3/21 for box
inside crop < 0.5, 0.5–0.9, ≥ 0.9). Containment is not what separates hits from misses;
scale could not be varied within this run.

### Error attribution (fixed decoder A, tIoU 0.3)

| | val FN by category | test FN by category | FP overlapping a same-type event / other type only / no event |
|---|---|---|---|
| frozen head | no_candidate 4, unresolved 6, other_actor_only 5, own_covered 19 (34 FN) | unresolved 4, own_covered 24 (28 FN) | val 6 / 4 / 55; test 9 / 1 / 86 |
| fine-tuned | same as frozen (34 FN) | unresolved 7, own_covered 24 (31 FN) | val 1 / 3 / 28; test 3 / 0 / 60 |

Most false positives lie on footage with no annotated event, so the loss is not only missed
events. Calibrated B runs have the same structure (details in the provenance summaries).

### Duplicate suppression

Same-type predictions of one pose actor that overlap a higher-scoring one are suppressed;
opposite types and different actors never are, and the classifier emits no item count, so no
multi-item detections exist to suppress. Suppressed: val A 11 (frozen) / 2 (fine-tuned),
val B 4 / 4, test A 4 / 1, test B 1 / 1; re-decoding the saved window scores reproduces
every saved prediction set. Per-item TP with vs without suppression is unchanged in all runs
**except frozen head, val, decoder A, tIoU 0.3: 1 TP without suppression, 0 with** — the
higher-scoring overlapping prediction missed while the suppressed one would have matched.
No GT pair of one actor and type is closer than the 0.75 s merge gap in either split, so
same-type merging did not fuse annotated events here.

## Window level (labelled windows only)

Deployment windows are labelled only where defensible: the manifest keeps a window if its
centre lies in its own actor's event or in no event, and excludes windows centred in an
event whose actor association failed or in an ignore interval. Argmax, no threshold.

| | windows labelled / inferred | excluded unresolved / ignore | labels bg / pickup / putdown | frozen head macro F1 (bg / pu / pd) | fine-tuned macro F1 (bg / pu / pd) |
|---|---|---|---|---|---|
| deployment val | 2154 / 2184 | 30 / 0 | 2066 / 53 / 35 | 0.280 (0.783 / 0.057 / 0.000) | 0.324 (0.928 / 0.043 / 0.000) |
| deployment test | 2968 / 3031 | 50 / 13 | 2804 / 113 / 51 | 0.292 (0.813 / 0.062 / 0.000) | 0.323 (0.906 / 0.062 / 0.000) |
| annotation val (different windows) | 973 | — | 814 / 96 / 63 | 0.615 (0.883 / 0.529 / 0.432) | 0.735 (0.962 / 0.711 / 0.532) |
| annotation test (different windows) | 1551 | — | 1320 / 174 / 57 | 0.480 (0.887 / 0.495 / 0.059) | 0.539 (0.945 / 0.576 / 0.095) |

The annotation and deployment rows score different window populations (different
candidates, class balance and exclusions) and are not a controlled comparison. Within
deployment, no labelled putdown window is predicted putdown by either model (val frozen
head: 35 putdown windows → 15 background, 20 pickup), and mean class probabilities barely
separate pickup windows from background (frozen head, val: p_pickup 0.40 vs 0.33).

## Conclusions (bounded by one camera, two days, fixed checkpoints)

* The annotation-conditioned scores of these checkpoints do not survive the change to
  deployment inputs. With decoders fixed from annotation validation, end-to-end test F1 @0.3
  falls from 0.361 to 0.101 (frozen head) and from 0.529 to 0.078 (fine-tuned), with 100–160
  FP/h; validation falls to zero. The fine-tuning gain seen under annotation conditioning
  does not appear on deployment inputs.
* Deployment-validation calibration does not change that conclusion.
* On the test day, proposal generation is not the limiting factor (every event has an
  overlapping candidate and a centred window); on the validation day it limits own-actor
  opportunity for 15 of 34 item rows, 6 of them through unresolved association.
* The crop presents the interaction at roughly 1/12–1/25 of the relative area the models
  were trained on, and the models show no class separation on these inputs. This is
  consistent with crop scale being a major factor, but this run did not isolate it from
  candidate timing, window context or other-actor windows, so the loss is not attributed to
  crops alone.

## Limitations

* Deployment preprocessing here is the pose pipeline with annotation-mode crop scope and
  interpolation (candidate-wide union, `INTER_AREA`), chosen to hold preprocessing fixed.
  The pose route's own default (per-window union, linear resize) was not evaluated; with
  long pose candidates (val median 5.9 s, up to 43 s; test up to 112 s) the candidate-wide
  union is larger than a per-window one.
* Small samples: 34 / 35 event rows, 0.33 / 0.63 h; single seeds and single checkpoints.
  Differences of one or two events move F1 by several points.
* Association uses the documented wrist-in-box rule; its failures are unresolved, not
  verified pose misses.
* Draft CVAT intervals (val 2, test 8) are scored as in the reproduction.
* The val A coverage summaries were regenerated once to add the crop-scale columns (same
  inputs, no model rerun); the first version is kept under
  `*/A_fixed_decoder/coverage_v1_superseded/`.

## Recommended next experiment

**`track_b1_crop_factorization`** (diagnostic, CVAT-conditioned by design, validation day
only, same two checkpoints, same decoders A): evaluate (a) pose candidates with CVAT-box
candidate crops and (b) CVAT candidates with pose-box candidate crops, alongside the two
existing endpoints. The four cells separate crop source from candidate source. Only if crops
dominate, follow with a deployable wrist-centred crop from pose keypoints at annotation-like
scale, evaluated with this protocol before any test read and, for a clean estimate, on new
recording days.

## Commands and outputs

```bash
X=.local/track_b1_deploy_eval_20260911
python scripts/build_track_b1_dataset.py --input-mode deployment \
    --config configs/track_b1_deployment_candidate_crop.yaml \
    --pose-data-dir .local/track_b1_data --output-dir $X/dataset
python scripts/build_track_b1_cache.py --dataset-dir $X/dataset --split val --workers 10   # then --split test
$X/provenance/run_eval.sh val_A    # infer_track_b1.py + coverage_track_b1.py, both checkpoints
$X/provenance/run_eval.sh tune_B   # tune_track_b1_thresholds.py on the val window scores; then freeze
$X/provenance/run_eval.sh val_B
$X/provenance/run_eval.sh test_A
$X/provenance/run_eval.sh test_B
python $X/provenance/compile_tables.py
```

Per checkpoint `M` in `frozen_head`, `finetune_last2`: `$X/M/A_fixed_decoder/` and
`$X/M/B_deploy_calibrated/` hold `window_scores_<split>.parquet`, `predictions_<split>.csv`,
`metrics_<split>.json` (both counting policies), `event_coverage_<split>.csv` and
`coverage_<split>.json`; `$X/M/B_calibration_sweep/` holds the sweep. Logs, commands and
PIDs (frame cache 28441, val A waiter 27772, test waiter 9631): `$X/logs/`. The per-window
frame cache added 12 GB to `.local/track_b1_frame_cache` (28 GB total). Runtime: frame cache
22 min (val) + 28 min (test) on 10 workers; each inference pass 1.5–3 min on the RTX 5090;
no decode failures.
