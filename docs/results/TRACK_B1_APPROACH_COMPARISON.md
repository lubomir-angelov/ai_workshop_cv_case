# Track B1: comparing two routes through the same task

> **Integration note (2026-09-11)** — see [`../TRACK_B1_INTEGRATION.md`](../TRACK_B1_INTEGRATION.md).
> Corrections to this document, verified on the merged code and data:
> * Route A's crop was the pose person box + 15% margin only. Shelf regions were never
>   applied (the loader read a key `configs/shelves.yaml` does not have), so
>   "pose box ∪ shelf region" below is inaccurate.
> * Route A's validation day (20260526) is Route B's test day. Route B's test day was
>   therefore used for model selection and manual review by Route A.
> * Route B's actor filter (§4.3) cannot be applied to Route A's candidates as written:
>   CVAT track ids and pose person ids never match, and every window would silently
>   become background. The combined pipeline associates CVAT events to pose actors first.
> * Route A's event-level performance is now measured (§7.2 of the integration doc):
>   the frozen-head checkpoint emits no events at the default 0.5 thresholds on its
>   validation day, and reaches event F1 0.154 / 0.092 (tIoU 0.3 / 0.5) only with
>   thresholds tuned on that same day.
> * The fine-tuned validation event F1 quoted here (0.783 / 0.638) disagrees with
>   `TRACK_B1_CVAT.md` (0.806 / 0.746); unresolved, artefacts unavailable.

Two independent implementations of task_12 were trained on the same 42 human-annotated
CVAT recordings and reached different numbers. This document explains what each did,
where the difference comes from, and — importantly — which parts of the difference are
real and which are artefacts of the two setups measuring different things.

- **Route A — pose-candidate route.** `feature/human_annotation_track_b`, results in
  [`TRACK_1B_HUMAN_ANNOTATION.md`](https://github.com/lubomir-angelov/ai_workshop_cv_case/blob/feature/human_annotation_track_b/docs/results/TRACK_1B_HUMAN_ANNOTATION.md).
  Follows the task specification: Task 5 candidates, pose tracks for crops.
- **Route B — annotation-conditioned route.** `feature/cvat_annotation`, this branch.
  Derives candidates and crops from the CVAT annotation itself.

Route A is the specified design. Route B is a deviation, taken because the pose pipeline
had not been run over these clips; it comes with a significant caveat covered in §4.1.

---

## 1. Headline numbers

Both routes freeze the VideoMAE-Base encoder and train only the classification head, and
Route B's earlier configuration used the same `WindowConfig()` defaults as Route A
(2.5 s window, 0.5 s stride, 16 frames, 0.15 crop margin). That pair is directly
comparable.

| Window-level, validation | Route A | Route B (2.5 s) | Route B (1.5 s, fine-tuned) |
|---|---|---|---|
| macro F1 | 0.424 | **0.589** | **0.743** |
| pickup F1 | 0.261 | **0.471** | **0.736** |
| putdown F1 | 0.113 | **0.441** | **0.539** |
| best epoch | 6 | 105 | 4 |

Route B additionally reports event-level results, which Route A does not produce:

| Event-level (shared Task 8 evaluator) | validation | test |
|---|---|---|
| F1 @ tIoU 0.3, frozen probe | 0.647 | 0.361 |
| F1 @ tIoU 0.3, fine-tuned | 0.783 | 0.457 |
| F1 @ tIoU 0.5, fine-tuned | 0.638 | 0.314 |

**Do not read the first table as "Route B is 40% better."** §4 explains why.

---

## 2. What each route does differently

| | Route A | Route B |
|---|---|---|
| Candidate source | Task 5 pose-based proposals | padded extent of each CVAT annotation track |
| Crop region | pose box ∪ shelf region + 15% margin | annotator's box + 15% margin |
| Crop recomputed | per window | per candidate (fixed within a candidate) |
| Actor identity | Task 5 `actor_id` (person track) | one CVAT track = one actor stream |
| Window labelling | per clip | per actor |
| Events used | those overlapping a pose candidate | all 259 |
| Negatives | pose candidate time not overlapping an event | ±3 s held-box padding + synthesised for cleared clips |
| Splits | recording day, seeded hash, 80/20 | recording day, ranked by event count, train/val/test |
| Held-out test day | none | yes (2026-05-26, read only after config freeze) |
| Frame rate | inherited from registry | probed per file with `ffprobe` |
| Caching | decoded frames (12.5× speedup) | frames **and** frozen-backbone embeddings |
| Encoder | frozen | frozen, then last 2 blocks unfrozen |
| Decoding to events | window-span intervals | configurable; `window_centers` selected |
| Evaluation | window-level metrics | window-level + Task 8 event-level at tIoU 0.3/0.5 |

---

## 3. Route A's analysis is right, and worth preserving

Two points from `TRACK_1B_HUMAN_ANNOTATION.md` are better stated there than anywhere in
this branch's write-ups, and should survive into any combined report.

**Accuracy is actively misleading here.** Route A's best checkpoint reaches 80.7%
validation accuracy against a validation set that is 1,284 background windows out of
1,391. Predicting background for everything scores 92.3% accuracy and ~0.320 macro F1.
The trained model is therefore *worse on accuracy* than the trivial baseline while being
genuinely better at the task. Any report that quotes accuracy without that baseline is
misreporting.

**Train and validation accuracy are not comparable** when training uses class-balanced
sampling and validation keeps the natural imbalance.

Route A also lists four candidate causes for weak event recognition: frozen encoder
features, ambiguous window labels, crops that do not clearly show the interaction, and
few independent annotated events. Route B's experiments bear on the first three
directly, and support all three.

---

## 4. Where the difference comes from

### 4.1. First, the confound that favours Route B

**Route B's candidates and crops are derived from ground-truth annotation boxes, and it
uses them at inference time too.** The model is handed the actor's spatial region and the
approximate temporal neighbourhood of every event for free.

At deployment neither is available. Route A's pose candidates *are* available at
deployment, because a pose pipeline produces them from raw video.

So the two routes are not measuring the same task. Route A measures a deployable system.
Route B measures an oracle-conditioned one, and its numbers are optimistic by an amount
this comparison cannot quantify. **Route B is not deployable as it stands**; making it so
requires a candidate and crop source at inference time — that is, Route A's pose tracks.

The honest reading is that Route B isolates *how well the classifier can do when
conditioning is correct*, and Route A measures *the whole problem*. Both are useful; they
are different questions.

A second, smaller confound runs the same way: Route A's validation set is 92.3%
background against Route B's 83.4%. More negatives means more opportunity for false
positives, which depresses precision and therefore macro F1 independently of model
quality.

### 4.2. Crop tightness

Route A's suspicion about crops looks correct. Its crop is the union of the pose box and
the shelf region — approximately a whole body plus shelving. The annotator's box in
Route B has a median size of 214 × 249 px in a 3840 × 2160 frame, drawn on the hands and
the item.

Resized to 224 × 224, Route B's crop is close to native resolution on the transfer
itself. Route A's downsamples the informative region to a small fraction of the input,
and a pickup differs from a putdown only in fine hand-and-item detail.

This is the most likely single contributor to the gap that is *not* an artefact of §4.1,
and it does not require annotation boxes to fix: tightening the pose-derived crop toward
the wrist/hand keypoints and the shelf edge would test it directly.

### 4.3. A labelling defect still present in Route A

`build_window_manifest` on `feature/human_annotation_track_b` selects ground-truth events
by clip:

```python
clip_events = events_df[events_df["clip_id"] == clip_id]
```

Route A's candidates carry `actor_id` from Task 5, and this corpus contains 273 pairs of
temporally overlapping actor candidates. Wherever two actors interact at overlapping
times, one actor's pickup is stamped onto windows cropped around the other actor, which
injects label noise straight into training.

This branch's fix restricts the lookup when the ground truth is actor-resolved:

```python
clip_events = events_df[events_df["clip_id"] == clip_id]
if "actor_id" in clip_events.columns:
    clip_events = clip_events[clip_events["actor_id"] == actor_id]
```

It is backwards compatible — clip-level ground truth behaves exactly as before.

### 4.4. Lost supervision

Route A's preparation script warns when human events fall outside every pose candidate
and excludes them from training. Route B keeps all 259 events by construction, because
its candidates *are* the events.

The size of this effect is unmeasured and worth measuring: if pose candidates miss a
substantial fraction of annotated events, that is both lost training signal for Route A
and a recall ceiling on the deployed system that no classifier improvement can lift.

### 4.5. Frozen encoder — Route A's hypothesis, tested

Route B ran the ablation. Unfreezing the last two encoder blocks raised window-level
macro F1 from 0.615 to 0.743, with putdown improving from 0.432 to 0.539.

Two implementation details were necessary rather than optional:

- **Discriminative learning rates.** Head at 1e-3, backbone at 5e-5. A single rate for
  both fails in either direction. An early attempt at a uniform 5e-5 left the randomly
  initialised head predicting background exclusively after two epochs, which initially
  read as evidence that unfreezing does not help.
- **Head warm-start** from the trained frozen probe, so the run measures what unfreezing
  contributes rather than how fast a head trains from noise.

Convergence was also non-monotone: epoch 1 reached pickup F1 0.635 with putdown at 0.057,
epoch 2 degraded both, and the optimum arrived at epoch 4. A short-patience schedule
stopped early would have produced the opposite conclusion. Route A's best epoch was 6
with validation declining afterwards, which is consistent with the same instability.

---

## 5. Two findings from Route B that apply to Route A unchanged

### 5.1. Interval construction caps tIoU before the model matters

Route A's `inference.py` builds a predicted interval as the span from the first
above-threshold window's start to the last one's end. A single firing window therefore
produces an interval exactly one window long.

With a 2.5 s window and a 0.90 s median event, temporal IoU cannot exceed
`0.90 / 2.5 ≈ 0.36` — below the standard 0.5 operating point, no matter how good the
classifier is.

Deriving the interval from window *centres* instead, widened by half a stride, raised
Route B's validation event F1 at tIoU 0.3 from 0.350 to **0.647 with no retraining**, and
cut boundary MAE from 1.35 s to 0.21 s. This is the largest single improvement measured
anywhere in either route, and it is a property of the decoder, not the model.

It is available to Route A as a config change once `boundary_mode` is ported.

### 5.2. Caching the embeddings, not just the frames

Route A reports a 12.5× speedup (30.8 → 2.5 min per epoch) from caching decoded frames.

While the encoder is frozen, its output is constant across epochs, so it can be computed
once for the whole manifest. Caching the 768-dimensional embeddings took Route B's epochs
from ~9 minutes to under a second — enough to make a 200-combination threshold sweep and
genuine early stopping affordable, which is why Route B's frozen probe ran to epoch 105
rather than stopping at a coarse optimum.

This is valid only while the backbone is frozen; unfreezing requires the full pixel path.

---

## 6. What Route B measured that Route A's protocol cannot see

Route A splits 80/20 into train and validation with no held-out test day. Route B holds
out a full recording day, and that is where the most consequential result appeared.

**The pickup/putdown direction does not survive a change of recording day.** On the
held-out day, windows whose ground truth is `putdown` receive mean *p*(putdown) = 0.062
and mean *p*(pickup) = 0.588 — the model confidently assigns the opposite class. The
maximum *p*(putdown) across all such windows is 0.467, so no decision threshold recovers
it. Every matched putdown on the test day was classified as a pickup.

Detection and temporal localisation transfer across days with moderate degradation.
Direction does not. The system degrades into an interaction detector with a pickup bias.

Validation F1 of 0.783 against test F1 of 0.457 is the visible symptom. Under a
train/validation-only protocol this failure is invisible, because the model looks strong
on the day it was tuned on.

Two caveats bound the finding. Only five recording days exist, so validation and test are
one day each and both estimates are high-variance. And Route B's test split has been read
twice, once per training regime; neither reading informed any decision, but it is no
longer perfectly naive.

---

## 7. Recommendations

**For Route A, in descending order of expected value:**

1. Port actor-aware window labelling (§4.3). Small diff, removes label noise from training.
2. Port `boundary_mode: window_centers` (§5.1). Largest measured gain, no retraining.
3. Tighten crops toward the hand/shelf region rather than the full body (§4.2).
4. Cache frozen-encoder embeddings rather than frames (§5.2), then unfreeze the last two
   blocks with discriminative learning rates (§4.5).
5. Add a held-out test day, and report per-class recall (§6).

**For Route B:**

1. Replace annotation-derived candidates and crops with pose-derived ones at inference
   time, and re-measure. Until then Route B's numbers are not comparable to a deployable
   system (§4.1). This is the single most important outstanding item on this branch.
2. Quantify how many annotated events fall outside pose candidates (§4.4) — it bounds
   what Route A's route can achieve regardless of classifier quality.

**For both:**

Day-level grouped cross-validation over all five days. At this scale a single-day
validation or test set cannot support the comparisons either route is trying to make,
and every number in this document would be better replaced by a mean and spread over
five folds.

Direction of transfer needs explicit supervision — time-reversal augmentation, a
frame-order pretext task, or fusion with Track A's shelf-state estimate, whose sign is
exactly the quantity that fails to generalise.

---

## 8. Sources

Route A numbers are quoted from `docs/results/TRACK_1B_HUMAN_ANNOTATION.md` on
`feature/human_annotation_track_b`; setup details are read from
`scripts/prepare_track_b1_data.py`, `configs/track_b1.yaml` and
`src/pickup_putdown/layer1/track_b1/{dataset,inference}.py` on that branch.

Route A does not publish per-window score distributions or event-level metrics, so
§5.1 and §6 are demonstrated on Route B's artefacts and argued for Route A from its code
rather than measured on it.

Route B numbers come from `.local/track_b1_run*/`, `.local/track_b1_finetune/` and
`.local/track_b1_dataset_w15/`, regenerable with `make track-b1-all`; the method is
documented in [`../TRACK_B1_CVAT.md`](../TRACK_B1_CVAT.md).
