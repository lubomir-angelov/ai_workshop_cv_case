# Track B1 reproduction: annotation-conditioned CVAT pipeline (2026-09-11)

Reproduces the `feature/cvat_annotation` Track B1 results (frozen probe, then last-2-block
fine-tune) on the integrated branch. **All scores are annotation-conditioned** (CVAT
candidates and crops at inference): upper bounds on classifier quality, not deployment
performance. **20260526 (test) is not an untouched test set**: it was the
`human_b1_2026_09_09` validation day and the CVAT branch read it twice.

## Verdict

| model | verdict | why |
|---|---|---|
| frozen head | **reproduced exactly** (annotation-conditioned) | val window macro F1 0.6147 vs 0.615 (per class identical to 3 decimals); val event F1 exactly 0.585 / 0.523 (tuned) and 0.647 / 0.412 (default) under per-event-group counting, as the CVAT branch counted; test identical (0.361 / 0.278 / 0.316 / 0.133) |
| fine-tuned (last 2 blocks), warm and cold head | **historical window performance approximately matched; historical fine-tuning experiment not reproduced** | val window macro F1 0.7348 (warm) and 0.7220 (cold) vs 0.743, but best epochs (8, 5 vs 4), epoch-1 behaviour, validation event F1 pairs and every test row differ from the historical run (see the cold-head diagnostic) |

Conclusions, bounded by one run per initialisation setting:

* The annotation-conditioned frozen-head results reproduce exactly.
* Warm-head and cold-head fine-tuning both approximately match historical validation window
  performance, and neither reproduces the historical fine-tuning experiment.
* The cold-head run does not support loss of the warm start as a sufficient explanation of
  the discrepancy: removing the warm start moves validation further from history, not closer.
* There is one run per initialisation. The warm/cold difference is not an estimate of
  seed-to-seed variance and cannot isolate the cause of the discrepancy.
* The cross-day putdown weakness persists in both runs.
* Matching one historical metric in isolation (e.g. the cold run's 0.746) does not
  reproduce an experiment.

## Provenance

* Commit `65a3f07` (`feature/unify_track_b_trainings_and_eval`) plus the artifact-only
  changes below (committed as `f98cff3`); the cold-head run used `4956f77`. Python 3.13.5, torch 2.14.0 (CUDA 13.0, cuDNN 9.24),
  transformers 5.16.1, numpy 2.5.3, pandas 3.0.5, opencv 5.0.0.93, ffprobe 4.4.2;
  RTX 5090 32 GB, WSL2. Full list: `provenance/{git_env,packages}.txt`, `pip_freeze.txt`.
* Encoder `MCG-NJU/videomae-base`, loaded strictly through `convert_encoder_state`
  (weights sha256 `bc053ca2840a…`, recorded in the embeddings and both checkpoints).
* Split registry `cvat_b1_2026_09_09`: train 20260520-22, val 20260523, test 20260526.
* CVAT export `.local/annotations/cvat/2026-09-09/raw`; `accepted_only: false` (drafts kept,
  as on the CVAT branch).
* Experiment root: `.local/track_b1_repro_cvat_20260911/` (below `$E`). Nothing earlier
  was overwritten; the shared candidate cache `.local/track_b1_cache` was extended.

## Data

| split | clips | event rows (pickup/putdown) | unique intervals (pickup/putdown) | extra item rows | draft intervals | windows (bg / pickup / putdown) |
|---|---|---|---|---|---|---|
| train | 23 | 214 (136/78) | 192 (127/65) | 22 | 3 | 5192 (4160 / 596 / 436) |
| val | 7 | 34 (20/14) | 32 (20/12) | 2 | 2 | **973 (814 / 96 / 63)** |
| test | 12 | 35 (23/12) | 35 (23/12) | 0 | 8 | 1551 (1320 / 174 / 57) |

Unique intervals equal the CVAT branch's split table (127/65, 20/12, 23/12; 259 total).
The val manifest is the historical 973 / 814. The rebuilt dataset's tables are identical to
`.local/track_b1_dataset`. 342 candidates (258 annotated tracks + 84 verified negatives),
3 ignore intervals (7 windows excluded). Train pickup/putdown windows carry
confidence weights 0.5-1.0.

## Effective configuration

* Windows (`build_metadata.json`): 1.5 s, stride 0.25 s, 16 frames, 224², margin 0.15,
  `crop_scope: candidate`, `INTER_AREA`, no shelf region, preprocessing version 2, context
  pad 3.0 s, 12 negative windows per cleared clip, seed 42.
* Caches: 342 per-candidate entries, 44 reused (fingerprints matched), 298 built, 0 stale
  after the build (`find_stale_entries`). Embeddings: train+val only, 6165 × 768, eval-mode
  frozen encoder, keyed per window.
* Frozen head (CVAT defaults): 300 epochs max, patience 40, batch 64, AdamW lr 1e-3, wd
  1e-2, dropout 0.2, cosine, class-weighted CE × sample weight, seed 42, CPU.
* Fine-tune (CVAT defaults / integration doc): last 2 blocks (`encoder.encoder.layer.10-11`,
  14.18 M of 86.24 M parameters; embeddings, blocks 0-9 and the final layernorm frozen),
  head warm-started from this experiment's `head_best.pt`, head lr 1e-3 / backbone 5e-5,
  wd 0.01, dropout 0.1, batch 8, 20 epochs max, patience 5 (min delta 0.001), 2 warm-up
  epochs then cosine, grad clip 1.0, Gate B on (runs on a copy), seed 42, CUDA fp32.
* Decode: `window_centers`, merge gap 0.75 s, min duration 0.3 s; thresholds per table
  below; tIoU 0.3 / 0.5 with the shared Task 8 evaluator. Per-type scores are at tIoU 0.5.
* Frozen selection before test: `$E/frozen_selection.json` (checkpoint sha256s, decode).
* Versioned copies of the small provenance files (commands, effective run configs,
  selections, chosen thresholds, checkpoint sha256s, environment) are in
  `docs/results/provenance/track_b1_repro_cvat_20260911/`; `commands.sh` there lists every
  command. Checkpoints, caches, datasets and predictions stay under `.local/`.

## Fine-tune preflight (`$E/provenance/preflight_finetune.json`)

Trainable set = blocks 10-11 + head only; optimizer groups head 1e-3 (4 tensors) /
backbone 5e-5 (32 tensors); data comes from `CachedTrackB1Dataset` pixels, never embeddings;
the warm-started model scores 0.614730298930164 val macro F1 through pixels, identical to
the embedding-trained head; Gate B passes and changes 0 tensors of the trained model; a real
batch gives finite loss 0.493 with finite, non-zero gradients on all 36 trainable tensors and
none on frozen ones. Peak allocated VRAM 4.4 GB.

## Validation results

### Window level (best checkpoints)

| | accuracy | macro F1 | background P / R / F1 | pickup P / R / F1 | putdown P / R / F1 |
|---|---|---|---|---|---|
| frozen head (epoch 37) | 0.789 | **0.6147** | 0.981 / 0.803 / 0.883 | 0.402 / 0.771 / 0.529 | 0.328 / 0.635 / 0.432 |
| historical frozen probe | — | 0.615 | — / — / 0.883 | — / — / 0.529 | — / — / 0.432 |
| fine-tuned (epoch 8) | 0.916 | **0.7348** | 0.947 / 0.978 / 0.962 | 0.693 / 0.729 / 0.711 | 0.806 / 0.397 / 0.532 |
| historical fine-tuned (epoch 4) | — | 0.743 | — | — / — / 0.736 | — / — / 0.539 |

Always-background accuracy is 0.837. Confusion (rows true bg/pickup/putdown):
frozen `[[654,91,69],[9,74,13],[4,19,40]]`, fine-tuned `[[796,12,6],[26,70,0],[19,19,25]]`.

Fine-tune trajectory (val macro F1 per epoch): 0.681, 0.526, 0.513, 0.639, 0.716, 0.670,
0.680, **0.735**, 0.706, 0.725, 0.732, 0.736 (below min delta), 0.732, early stop at 13.
Epochs 2-3 collapse pickup (0.15-0.19), the instability the CVAT documents describe, then
the model overfits (train F1 0.998 by epoch 13).

### Event level, validation (F1 @ tIoU 0.3 / 0.5)

Two counting policies (`inference.COUNTING_POLICIES`, `scripts/rescore_track_b1_events.py`),
applied to the same predictions. **Per-item** (primary) follows the annotation convention of
one row per item: `item_count = N` is N rows (34 val rows). **Per-event-group** (historical
comparison) collapses those rows to one per (clip, `event_group_id`) (32 rows), as the CVAT
branch counted. Predictions are never duplicated: the classifier outputs no item count, so
one detection of a 3-item event scores 1 TP + 2 FN per item and 1 TP per group.
`infer_track_b1.py` now records both; the promoted implementation reproduces the scratch
helper's numbers on all nine prediction sets of this experiment (`$E/logs/24_*`).

| model | decode (pickup / putdown thr, smoothing) | per-item | per-event-group | historical |
|---|---|---|---|---|
| frozen | 0.50 / 0.50, 3 (historical code default) | 0.629 / 0.400 | **0.647 / 0.412** | 0.647 / 0.412 |
| frozen | 0.40 / 0.45, 5 (configured) | 0.571 / 0.286 | 0.587 / 0.293 | — |
| frozen | **0.45 / 0.60, 3 (val-tuned, selected)** | 0.567 / 0.507 | **0.585 / 0.523** | 0.585 / 0.523 (tuned) |
| fine-tuned | 0.50 / 0.50, 3 (historical code default) | 0.783 / 0.696 | 0.806 / 0.716 | 0.783 / 0.638 or 0.806 / 0.746 |
| fine-tuned | 0.40 / 0.45, 5 (configured) | 0.716 / 0.627 | 0.738 / 0.646 | |
| fine-tuned | **0.50 / 0.20, 3 (val-tuned, selected)** | 0.771 / 0.743 | 0.794 / 0.765 | |

Selected-decode per-type F1 @0.5 (per-item): frozen pickup 0.558 / putdown 0.417; fine-tuned
pickup 0.826 / putdown 0.583. Fine-tuned boundary MAE 0.22 s / 0.25 s; frozen 0.50 / 0.48 s
at the default decode (historical 0.48 s). Neither historical fine-tuned pair is reproduced as a
pair by any decode here; which one is authoritative remains unresolved. The fine-tuned
sweep's best putdown threshold (0.20, tied with 0.25) is the lowest grid value.

## Test results (20260526, read once per model after the freeze)

| | window acc | window macro F1 (bg / pickup / putdown) | event F1 @0.3 | event F1 @0.5 | pickup @0.5 | putdown @0.5 |
|---|---|---|---|---|---|---|
| frozen head | 0.792 | 0.480 (0.887 / 0.495 / 0.059) | **0.361** | **0.278** | 0.316 | 0.133 |
| historical frozen probe | | | 0.361 | 0.278 | 0.316 | 0.133 |
| fine-tuned | 0.890 | 0.539 (0.945 / 0.576 / 0.095) | 0.529 | 0.471 | 0.500 | 0.375 |
| historical fine-tuned | | | 0.457 | 0.314 | 0.393 | 0.000 |

Test has no multi-item groups (35 rows = 35 groups), so the two counting policies agree.
The cross-day putdown collapse reproduces: on true-putdown test windows the fine-tuned model's
mean p_putdown is 0.062 (historical 0.062; val 0.393), putdown window recall is 0.05 for both
models, and fine-tuned predictions are 29 pickup / 4 putdown against a GT of 23 / 12.

## Diagnostic: fine-tune without the warm-started head

Tests whether the CVAT code's Gate B behaviour (re-creating the model, so a random head
unless `--skip-tiny-overfit`) explains the fine-tune gap. Same command and settings, no
`--init-head-from` (`$E/finetune_last2_cold`, commit `4956f77`); selection frozen in
`$E/frozen_selection_cold.json` before one test read.

| | warm head (primary) | cold head | historical |
|---|---|---|---|
| epoch 1 val pickup / putdown F1 | 0.617 / 0.484 | 0.565 / 0.263 | 0.635 / 0.057 |
| best epoch / stop | 8 / 13 | 5 / 10 | 4 / — |
| val window macro F1 (bg / pickup / putdown) | 0.7348 (0.962 / 0.711 / 0.532) | 0.7220 (0.951 / 0.684 / 0.531) | 0.743 (— / 0.736 / 0.539) |
| val-tuned decode | 0.50 / 0.20, 3 | 0.40 / 0.35, 5 | 0.40 / 0.45, 5 (config) |
| val event F1 @0.3 / @0.5, per-item (per-event-group) | 0.771 / 0.743 (0.794 / 0.765) | 0.754 / 0.725 (0.776 / 0.746) | 0.806 / 0.746 or 0.783 / 0.638 |
| test event F1 @0.3 / @0.5 | 0.529 / 0.471 | 0.580 / 0.435 | 0.457 / 0.314 |
| test pickup / putdown F1 @0.5 | 0.500 / 0.375 | 0.480 / 0.316 | 0.393 / 0.000 |
| test window macro F1 | 0.539 | 0.583 | — |
| test mean p_putdown on true putdowns | 0.062 | 0.181 | 0.062 |

**The cold-head run does not support loss of the warm start as a sufficient explanation.**
Without the warm start, validation window macro F1 moves further from the historical value
(0.7220 vs 0.7348 warm, 0.743 historical), and neither initialisation reproduces the
historical epoch-1 profile, best epoch, validation event pair or test row (putdown 0.000).
Warm and cold also differ from each other (test F1@0.5 0.471 vs 0.435, putdown window F1
0.095 vs 0.225), but with one run per setting that difference confounds initialisation
with run-to-run variation (batch order, platform numerics); it is not an estimate of
seed-to-seed variance and cannot isolate what caused the historical discrepancy. That
would take several seeds per condition (not run here). Both runs keep the cross-day putdown
weakness: putdown test recall at window level is 0.05 (warm) and 0.18 (cold) against
0.40-0.48 on validation. The cold run's per-event-group val F1@0.5 equals one historical
value (0.746) while the paired F1@0.3 does not (0.776 vs 0.806); one matching number is
not a reproduction of the experiment.

## Differences caused by the integrated code

* **Item-count expansion** (283 rows vs 259): window labels are unchanged; event recall
  denominators grow (val 34 vs 32). It accounts for the entire frozen-head event gap
  (collapsed counting gives the historical numbers exactly).
* **Gate B warm start**: here Gate B runs on a copy and the warm-started head survives
  (verified). On the CVAT code, Gate B re-created the model, so its fine-tune had a warm
  head only if it ran with `--skip-tiny-overfit`, which is not recorded. Its epoch 1
  (putdown F1 0.057) looks more like a re-initialised head than this run's warm-started epoch 1
  (0.484); circumstantial only. The cold-head run (above) does not support this as a
  sufficient explanation; MPS vs CUDA numerics, batch order and other unrecorded
  differences remain candidates, none isolated.
* **Encoder loading**: strict layout conversion on transformers 5.16.1 instead of
  transformers <5; the frozen-head reproduction to the last decimal indicates equivalent
  features.
* Shared by both branches, kept as is: in the pixel trainer, per-sample confidence weights
  replace the class-weighted loss (class weights only enter the validation loss, and the
  balanced sampler is off); the embedding head uses class weights × sample weights. The two
  stages therefore train with different loss weighting. Exact loss computations, their
  history and a separately named follow-up experiment: `docs/TRACK_B1_INTEGRATION.md` §10.

## Runtime and resources

Dataset build 17 s; candidate cache 5 min (10 workers, 7.2 GB total cache); embeddings 2.5 min
(GPU); frozen head 66 s on CPU (Gate B 0.5 s, early stop at epoch 77); fine-tune 21 min
(Gate B 9 s, ~79-89 s train + ~12 s val per epoch, 0.11 s compute vs 0.02 s data wait
per batch, GPU 77-87 %, 6.8 GB VRAM, ~517 W); each val/test inference or diagnostics
pass 1-2 min. RAM ≤ 12 GB of 31 GB; disk 335 GB free after the run (experiment
directory 1.8 GB).

## Commands

```bash
E=.local/track_b1_repro_cvat_20260911; export HF_HUB_OFFLINE=1
python scripts/build_track_b1_dataset.py --input-mode annotation --output-dir $E/dataset
python scripts/build_track_b1_cache.py --dataset-dir $E/dataset --workers 10
python scripts/precompute_track_b1_embeddings.py --dataset-dir $E/dataset --split train val --batch-size 8 --num-workers 4
python scripts/train_track_b1_head.py --dataset-dir $E/dataset --output-dir $E/frozen_head --epochs 300 \
    --patience 40 --batch-size 64 --learning-rate 1e-3 --weight-decay 1e-2 --dropout 0.2 --seed 42 --device cpu
python $E/provenance/preflight_finetune.py
python scripts/train_track_b1.py --dataset-dir $E/dataset --output-dir $E/finetune_last2 --unfreeze-last-n-blocks 2 \
    --backbone-lr 5e-5 --learning-rate 1e-3 --init-head-from $E/frozen_head/checkpoints/head_best.pt --epochs 20 \
    --patience 5 --batch-size 8 --weight-decay 0.01 --dropout 0.1 --num-workers 4 --seed 42 --device cuda
# per model M (frozen_head: C=checkpoints/head_best.pt; finetune_last2: C=checkpoints/best_model.pt)
python scripts/infer_track_b1.py --dataset-dir $E/dataset --checkpoint $E/$M/$C --split val --output-dir $E/$M/eval/val_configured_decode
python scripts/tune_track_b1_thresholds.py --dataset-dir $E/dataset --predictions-dir $E/$M/eval/val_sweep   # scores copied in
python scripts/infer_track_b1.py ... --split val  --pickup-threshold P --putdown-threshold D --smoothing-window S --output-dir $E/$M/eval/val_tuned_decode
python scripts/infer_track_b1.py ... --split test --pickup-threshold P --putdown-threshold D --smoothing-window S --output-dir $E/$M/eval/test_frozen_decode
python scripts/diagnose_track_b1.py --dataset-dir $E/dataset --checkpoint $E/$M/$C --split {val,test} --output-dir $E/$M/diagnostics_{val,test} --workers 4
python scripts/rescore_track_b1_events.py --dataset-dir $E/dataset --predictions-dir $E/$M/eval/<run> --split {val,test}
```

Exact per-run commands with every flag: `docs/results/provenance/track_b1_repro_cvat_20260911/commands.sh`.
`$E/provenance/collapsed_gt_eval.py` was the scratch helper the rescoring script replaces;
its `collapsed_gt_comparison_<split>.json` outputs are kept next to the new
`rescored_by_counting_policy_<split>.json`.

## Paths (under `$E`)

| | frozen head | fine-tuned |
|---|---|---|
| checkpoints | `frozen_head/checkpoints/head_best.pt`, `head_final.pt` | `finetune_last2/checkpoints/best_model.pt`, `final_model.pt`, `checkpoint_epoch_{5,10}.pt` |
| config / history | `frozen_head/head_results.json`, `head_training_history.csv` | `finetune_last2/run_config.json`, `training_results.json` |
| window predictions | `frozen_head/val_window_predictions.csv`, `diagnostics_{val,test}/` | `finetune_last2/diagnostics_{val,test}/` |
| event scores / predictions | `frozen_head/eval/*/{window_scores,predictions,metrics}_*.{parquet,csv,json}` | `finetune_last2/eval/…` |
| threshold sweep | `frozen_head/eval/val_sweep/` | `finetune_last2/eval/val_sweep/` |

Logs, PIDs and commands: `$E/logs/`; provenance, preflight and helper scripts:
`$E/provenance/`. The `diagnostics_{val,test}/` directories predate the split-aware fix and
are kept unchanged: there `diagnose_track_b1.py` named its per-window file
`val_predictions.csv` for every split and warned that metrics "differ from checkpoint" for
test, so `diagnostics_test/val_predictions.csv` holds the 1551 **test** windows and that
warning is spurious. Since the fix, files are named per split (`test_predictions.csv`,
`test_metrics.json`, …), an existing file is never overwritten, and the stored validation F1
is compared only when the run re-evaluates the checkpoint's own validation data
(`frozen_head/diagnostics_splitaware/`: val difference −1e−16, test "not compared").

## Code changes made for this run

Artifact-only, no effect on training or scores: `train_track_b1_head.py` also saves
`head_final.pt` and records `final_epoch`; `track_b1/train.py` saves `final_model.pt` and
logs every optimizer group's learning rate per epoch.

Reproduction cleanup (no effect on training, window scores or per-item event metrics):
counting policies in `track_b1/inference.py` (`ground_truth_for_policy`,
`evaluate_events_by_policy`), recorded by `infer_track_b1.py` and re-scorable with
`scripts/rescore_track_b1_events.py`; split-aware `diagnose_track_b1.py`; the chosen
thresholds record their counting policy; tests `tests/test_track_b1_counting_policy.py`,
`tests/test_diagnose_track_b1.py`.

## Open issues

* Fine-tune reproduction: loss of the warm start was tested and is not supported as a
  sufficient explanation (see the diagnostic above). With one run per initialisation the
  cause of the discrepancy is not isolated; a multi-seed comparison per condition would be
  needed (not run).
* The fine-tuned decode choice sits at the grid edge (putdown 0.20); the historical grid
  was kept.
* The class-weight inconsistency between the two trainers is untouched, to preserve
  reproduction (`docs/TRACK_B1_INTEGRATION.md` §10; follow-up experiment
  `track_b1_consistent_loss_weighting`, not run).
