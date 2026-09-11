# Track B1: what the metrics mean, and why the two routes report different numbers

Two implementations of task_12 trained on the same 42 human-annotated CVAT recordings
report different figures. This document explains each metric, states which comparisons
between the two are valid, and quantifies how much of the difference survives once the
measurement differences are removed.

- **Route A** — `feature/human_annotation_track_b`, results in `TRACK_1B_HUMAN_ANNOTATION.md`.
- **Route B** — `feature/cvat_annotation`, results in `../TRACK_B1_CVAT.md`.

For *how* the two pipelines differ, see
[`TRACK_B1_APPROACH_COMPARISON.md`](TRACK_B1_APPROACH_COMPARISON.md). This document is
only about the numbers.

---

## 1. The problem with reading the numbers side by side

| Window-level validation | Route A | Route B (2.5 s, frozen) |
|---|---|---|
| macro F1 | 0.424 | 0.589 |

Both are "validation macro F1" on "the same annotated corpus", so the difference looks
like a straight 0.165 gap in model quality. It is not, for three reasons that have
nothing to do with the models.

**The validation sets are different sizes with different class balance.**

| | Route A | Route B (2.5 s) | Route B (1.5 s) |
|---|---|---|---|
| validation windows | 1,391 | 428 | 973 |
| background windows | 1,284 | 357 | 814 |
| background share | **92.3%** | **83.4%** | **83.7%** |
| event windows | 107 | 71 | 159 |

**They are not the same validation *days*.** Route A picks validation days by seeded
hash of the day string; Route B ranks days by event count and holds out the sparsest.
Verified from the split registries (`configs/track_b1_splits.yaml`): Route A validates on
20260526, Route B validates on 20260523 and tests on 20260526. They score different days,
so even a perfect reimplementation of one would not reproduce the other's number.

**The window populations differ in kind.** Route A's windows come from Task 5 pose
candidates, so its background class includes long stretches of a person near a shelf
doing nothing. Route B's come from ±3 s around annotated events, so its background is
concentrated in the approach and withdrawal immediately surrounding a real interaction.
Route B's negatives are *harder* per window; Route A's are *more numerous*.

---

## 2. Metric by metric

### 2.1. Accuracy — do not use it here

Fraction of windows whose predicted class is correct.

The class distribution makes this metric actively misleading, a point Route A's document
makes well and which is worth restating because it applies to both routes.

| | Route A | Route B (1.5 s, frozen) |
|---|---|---|
| always-predict-background accuracy | **0.923** | **0.837** |
| trained model accuracy | 0.807 | 0.789 |
| difference | **−0.116** | **−0.048** |

**Both trained models are less accurate than a model that never predicts an event at
all.** They are nonetheless far better at the task. Accuracy rewards the majority class
in proportion to how dominant it is, and background dominates. Any statement of accuracy
in this project is meaningless without its trivial-baseline figure alongside it.

### 2.2. Macro F1 — the right metric, but it moves with class balance

The unweighted mean of per-class F1 across background, pickup and putdown. Unweighted
means the two rare event classes count for two-thirds of the score, which is the correct
emphasis for this task.

The catch: a *fixed* classifier scores differently on validation sets with different
class balance. When negatives are more numerous, the same false-positive *rate* produces
more false positives per true positive, so event-class precision falls, so macro F1
falls. Route A's 92.3% background set is intrinsically harder to score well on than
Route B's 83.7%, independent of model quality.

The trivial baseline shifts too, but subtracting it does not correct for this effect
(see the correction note in §3).

### 2.3. Per-class F1 — where the difference actually lives

| Validation | Route A | Route B (2.5 s, frozen) | Route B (1.5 s, fine-tuned) |
|---|---|---|---|
| background F1 | not reported | 0.856 | 0.952 |
| pickup F1 | 0.261 | 0.471 | 0.736 |
| putdown F1 | 0.113 | 0.441 | 0.539 |

Background F1 is high in every configuration and carries little information — a trivial
predictor reaches 0.91–0.96 on it. The event classes are where the routes separate, and
putdown separates them most.

Both routes agree on the ordering: putdown is harder than pickup everywhere. Putdown is
the minority event class in the corpus (89 events against 170 pickups), and a classifier
uncertain about the direction of transfer minimises expected loss by defaulting to the
more frequent class.

### 2.4. Event-level F1 at tIoU — only Route B reports this

Everything above scores *windows*. The task is to emit *events*, which means turning a
run of window scores into an interval and matching it against ground truth by temporal
intersection-over-union. Only Route B produces these.

| Route B, event level | validation | test |
|---|---|---|
| F1 @ tIoU 0.3, frozen | 0.647 | 0.361 |
| F1 @ tIoU 0.3, fine-tuned | 0.783 | 0.457 |
| F1 @ tIoU 0.5, fine-tuned | 0.638 | 0.314 |

**Unresolved discrepancy:** `../TRACK_B1_CVAT.md` and `configs/track_b1.yaml` report the
fine-tuned validation event F1 as 0.806 / 0.746 (tIoU 0.3 / 0.5), this table as 0.783 /
0.638. The run artefacts that would settle it (`.local/track_b1_finetune/predictions/
metrics_val.json`, `chosen_thresholds_*.json`) are not available on the integration
machine, so neither pair is verified. A plausible, unverified explanation is that 0.806 /
0.746 are after validation-set threshold tuning and 0.783 / 0.638 before it; if so, the
tuned pair is additionally optimistic because the thresholds were selected on the same
validation day.

**Window-level quality does not translate cleanly into event-level quality.** In Route
B's own ablation, a model whose window-level scores were completely unchanged moved from
event F1 0.350 to 0.647 purely by changing how intervals are built from window runs. A
route that reports only window-level metrics has not yet measured the thing the task
asks for.

### 2.5. Boundary MAE and false positives per hour

Mean absolute error between predicted and matched-true event start and end times, and
the false-positive count normalised by footage duration. Route B reports 0.21 s start
MAE and 6.5 FP/hour on validation. Route A reports neither.

FP/hour is the figure an operator cares about: it says how often the system will raise a
spurious event during a shift.

---

## 3. Lift over the trivial baseline (does not remove class-balance effects)

> **Correction (integration branch, 2026-09-11).** An earlier version of this section said
> that subtracting the always-background macro F1 "removes the class-balance effect". It
> does not. The trivial predictor's macro F1 is only its background F1 divided by three
> (its event-class F1 is zero), so subtracting it removes a constant tied to the
> background share and nothing else. The effect described in §2.2 is untouched: for a
> classifier with fixed per-class recall and false-positive rate, event-class precision
> is `recall·P / (recall·P + FPR·N)` and falls as negatives per positive (`N/P`) grow,
> about 12:1 on Route A's set against about 5:1 on Route B's. The lift figures and ratios
> below are therefore **not** balance-corrected comparisons. Prevalence-independent
> quantities (per-class recall and false-positive rate) or scoring both models on the
> same windows would be; see `docs/TRACK_B1_INTEGRATION.md`.

The table compares each model with the trivial baseline *of its own validation set*.

The trivial baseline is a predictor that always outputs background. On Route A's set that
scores macro F1 0.320 — which independently reproduces the ~0.320 figure in their
document, confirming the calculation.

| | val background share | trivial macro F1 | model macro F1 | **lift** |
|---|---|---|---|---|
| Route A | 92.3% | 0.320 | 0.424 | **+0.104** |
| Route B, 2.5 s frozen | 83.4% | 0.303 | 0.589 | **+0.286** |
| Route B, 1.5 s frozen | 83.7% | 0.304 | 0.615 | **+0.311** |
| Route B, 1.5 s fine-tuned | 83.7% | 0.304 | 0.743 | **+0.439** |

Read this way Route B's comparable configuration shows **2.7×** the lift of Route A, and
its best configuration **4.2×**, but per the correction above these ratios still carry
the class-balance effect, as well as the other two differences from §1 (different
validation days and different window populations) and the conditioning advantage in §4.
They are not a fair comparison.

---

## 4. The difference that no normalisation removes

**Route B derives its candidates and crops from the ground-truth annotation boxes, and
uses them at inference as well as training.** Its model is handed the actor's spatial
region and the approximate temporal neighbourhood of every event for free.

Route A's pose-derived candidates are produced from raw video and exist at deployment.
Route B's do not.

So Route B is measuring a materially easier problem, and no rescaling of its metrics
corrects for that. The number that would be comparable — Route B re-run with
pose-derived candidates and crops — has not been produced. Until it is:

- Route A's figures describe a deployable system.
- Route B's figures describe how well a classifier performs *when the conditioning is
  correct*, which is an upper bound on what Route B's route would achieve in production.

Both are legitimate measurements. They are not measurements of the same thing, and the
lift table in §3 should be read with that firmly in mind.

---

## 5. What is safe to conclude

**Comparable, with the §4 caveat:**

- Route B's classifier scores substantially higher per window than Route A's. How much
  of that is the model is not established: the §3 lift does not remove class balance,
  and §4 conditioning is not removed at all.
- Both routes find putdown harder than pickup, by a similar ratio. This is a property of
  the data, not of either implementation.
- Both routes' trained models score below the trivial accuracy baseline while being
  genuinely better at the task (§2.1).

**Not comparable:**

- Raw macro F1 side by side (§1).
- Any accuracy figure without its own baseline (§2.1).
- Event-level performance — Route A has not measured it (§2.4).
- Generalisation to an unseen day — Route A's protocol has no test split, so the
  cross-day collapse Route B found is invisible under it. Route B's own validation-to-test
  drop, event F1 0.783 → 0.457, is the size of effect that a validation-only protocol
  cannot see. (Route A's validation day, 20260526, *is* Route B's test day, so that day
  was used for model selection in Route A and read twice in Route B; it is not an
  untouched test day for combined development.)

**Unresolved:**

- How much of Route B's advantage is the conditioning advantage in §4, and how much is
  the classifier. This is the single most important open question between the two routes,
  and it is answerable: re-run Route B with pose-derived candidates and crops.
- Both routes have five recording days, so every figure in this document rests on a
  one-day validation set. Day-level grouped cross-validation would replace all of them
  with a mean and spread, and should.

---

## 6. Provenance

Route A figures are quoted from `docs/results/TRACK_1B_HUMAN_ANNOTATION.md` on
`feature/human_annotation_track_b`; its validation composition (1,284 of 1,391 windows
background) and accuracy (0.807) are stated there. Its trivial baseline is recomputed
here and matches the ~0.320 that document reports.

Route B figures are computed from `.local/track_b1_run*/head_results.json`,
`.local/track_b1_finetune/predictions/metrics_*.json` and the window manifests in
`.local/track_b1_dataset*/`, all regenerable with `make track-b1-all`, and were verified
against those artefacts rather than transcribed.
