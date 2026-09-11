# Track B1 on CVAT source-video annotations

How the Track B1 VideoMAE window classifier (task_12) is trained on the 42 clips
annotated in CVAT, what had to be built to get there, and which choices are load-bearing.

Task 12 was written assuming Task 5 candidates plus pose tracks. Neither exists for
these clips: annotation was done directly on source video, and the pose pipeline was
never run over them. Most of the work below is bridging that gap without weakening
what the task is actually testing.

---

## 1. Pipeline

```
CVAT (42 completed jobs)
  │  scripts/export_cvat_annotations.py        read-only w.r.t. CVAT
  ▼
s3://chillnbite-cameras/anon/annotations/cvat/<date>/raw/*.zip   + export_manifest.csv
  │  scripts/build_track_b1_dataset.py
  ▼
.local/track_b1_dataset/
    events.parquet            259 events (170 pickup, 89 putdown)
    ignore_intervals.parquet  3 intervals
    clips.parquet             42 clips, fps probed from video, split by recording day
    actor_tracks/<clip>.parquet   per-frame boxes in pose-track column layout
    candidates.parquet        342 candidates (258 annotated + 84 verified-negative)
    window_manifest.parquet   3420 labelled windows
  │  scripts/build_track_b1_cache.py           decode each candidate once
  ▼
.local/track_b1_cache/        7.2 GB of 224x224 uint8 crops
  │  scripts/precompute_track_b1_embeddings.py  frozen backbone, run once
  ▼
.local/track_b1_embeddings/   [3420, 768] float32
  │  scripts/train_track_b1_head.py            Gate B + head training
  ▼
.local/track_b1_run/checkpoints/head_best.pt
  │  scripts/infer_track_b1.py                 sliding window -> smooth -> peaks -> same-type merge
  ▼
predictions_<split>.csv + Task 8 metrics
```

`make track-b1-all` runs everything from an existing export.

---

## 2. Annotation shape and what it gives us

Each CVAT `<track>` is one annotated interval carrying a box on every frame it spans.
That single structure supplies both things Track B1 needs:

| Need | Source |
|---|---|
| Event type and temporal extent | track label; first to last frame with `outside="0"` |
| Actor-conditioned crop region | the per-frame box |

A track interrupted by `outside="1"` and resumed later becomes **two** events, not one
spanning the gap.

CVAT tracks are drawn per interaction rather than per person, so one track is treated
as one actor stream (`trk007`). Two overlapping tracks in a clip therefore give two
independent actors, which is what the "actor A and actor B in the same clip" acceptance
criterion needs.

### Frame rate

CVAT stores annotations as frame indices, so every timestamp depends on the frame rate.
The rate implied by the clip stem's recording window is off by ~0.2%, which drifts about
a third of a second by the end of a clip — longer than a median event (0.9 s). The rate
is therefore probed from the video file with `ffprobe`; all 42 clips are exactly 20.0 fps.
`clips.parquet` records `fps_source` so an estimated rate is never mistaken for a measured one.

---

## 3. Three decisions that carry the dataset

### 3.1 Held-box padding is what makes negatives usable

Boxes exist only *during* events. A background window would therefore find no box, fall
back to the full 4K frame, and be trivially separable from a tightly-cropped positive —
the model could score well on crop size alone, having never looked at the action.

Each actor track is padded by ±3 s with its boundary box repeated. Negatives drawn from
the run-up to an interaction are then cropped to the same region as the positives, and
only the motion inside the box distinguishes them. Padded rows carry `is_padding=True`.

This is also where hard negatives come from: the padding is the approach and withdrawal
around a real interaction, which is exactly the confusable case.

### 3.2 Verified negatives need a borrowed box

Seven completed jobs contain zero events. A completed-but-empty job is a verified
negative and is as informative as a positive, but it carries no box at all. For these,
a box is borrowed from the pool of real annotated boxes and held over a randomly placed
6 s interval (12 per clip, seeded).

The borrowed box is **not** where a person actually is in that clip. These windows are
honest negatives for *"is a transfer happening in this crop"* — the question Track B1
asks — but not for *"is a person present"*. Worth remembering when reading failure cases.

### 3.3 Splits are by recording day, not by clip

Clips recorded minutes apart share shoppers, lighting and shelf state. A clip-level split
would put near-duplicates on both sides and inflate validation. Days are ranked by event
count and the sparsest are held out, keeping the bulk of supervision in train.

| split | clips | pickup | putdown | windows |
|---|---|---|---|---|
| train | 23 | 127 | 65 | 2314 |
| val | 7 | 20 | 12 | 428 |
| test | 12 | 23 | 12 | 678 |

`build_track_b1_dataset.py` fails if any clip lands in two splits.

---

## 4. Two performance problems that had to be solved

Both were blocking, not cosmetic.

### 4.1 Seeking in 4K H.264 is four orders of magnitude too slow

The sources are 3840×2160. `dataset.decode_window_frames` seeks to each of a window's
16 frames, and every seek decodes forward from the preceding keyframe at full resolution.

Measured on one clip: **16 seeking reads = 20.6 s; 48 sequential reads = 0.4 s.**

At ~27 s per window, one training epoch over 2314 windows would take about 17 hours.

`layer1/track_b1/cache.py` decodes each candidate once, sequentially, crops to the
candidate's actor region, resizes to 224×224 and stores uint8. Windows are then served by
indexing that array. Build cost is ~20 min for all 342 candidates; after that, window
loading is a memory-mapped gather.

The crop is computed **once per candidate** rather than per window. Beyond enabling the
cache, this is better conditioning: within a candidate all windows share a frame of
reference, so a changing crop size cannot encode the label.

### 4.2 The backbone is frozen, so it should not be recomputed every epoch

With `freeze_backbone: true` only 3,843 parameters train, yet each epoch re-ran all 86M
backbone parameters — about 9 minutes per epoch. Since the backbone output never changes,
`precompute_track_b1_embeddings.py` computes it once ([3420, 768]) and
`train_track_b1_head.py` trains the head on those features in under a second per epoch.

This is what makes real early stopping and threshold tuning affordable. It is valid
**only while the backbone is frozen**; unfreezing blocks requires the full pixel path in
`train_track_b1.py`.

---

## 5. Environment

The project conda env (`pickup-putdown-py312`) is **x86_64 under Rosetta**. torch is
capped at 2.2.2 there and `Conv3D is not supported on MPS`, so VideoMAE cannot run on
the GPU at all.

Track B1 uses `.venv-arm64` — native arm64 Python 3.13, torch 2.14 with working Conv3D on
MPS. transformers is pinned `<5`: transformers 5.x loads the VideoMAE checkpoint with
missing q/k/v biases, silently leaving part of the pretrained backbone randomly initialized.

```bash
/opt/homebrew/bin/python3.13 -m venv .venv-arm64
.venv-arm64/bin/python -m pip install torch torchvision "transformers>=4.44,<5" \
    pandas numpy opencv-python-headless pyarrow scikit-learn pyyaml pydantic pytest
.venv-arm64/bin/python -m pip install -e . --no-deps
```

---

## 6. Gates

**Gate A — visual loader inspection** (`make track-b1-inspect`). Renders sampled-frame
grids exactly as the model receives them, index-stamped, so temporal order is checkable
by eye. Verified: pickups show hand in → grasp → lift → item gone; putdowns show the
reverse; backgrounds are framed like positives, confirming the padding removed crop leakage.

**Gate B — tiny overfit.** Runs on a **class-balanced** subset. The original implementation
took the first N manifest rows, which all come from one candidate and almost always one
class; a model can memorise that by collapsing to a constant, which proves nothing about
the pipeline the gate exists to check.

---

## 7. Decoding

Sliding windows are scored, class probabilities smoothed over time, score peaks turned
into regions, and **only same-type regions merged**. A pickup and an adjacent putdown stay
two events. One candidate may emit zero, one or several ordered events.

Evaluation uses the shared Task 8 evaluator (`aggregate_metrics`) with ignore intervals
applied, at tIoU 0.3 and 0.5.

---

## 8. Results

Frozen VideoMAE-base backbone + trained classification head. Event-level scores come from
the shared Task 8 evaluator with ignore intervals applied.

### Validation: what each change bought

| Decode configuration | F1 @ tIoU 0.3 | F1 @ tIoU 0.5 | boundary MAE |
|---|---|---|---|
| 2.5 s window, `window_span`, default thresholds | 0.264 | 0.075 | 1.35 s |
| 2.5 s window, `window_span`, tuned thresholds | 0.350 | — | — |
| 2.5 s window, `window_centers`, tuned | 0.647 | 0.351 | — |
| 1.5 s window, `window_centers`, default | 0.647 | 0.412 | 0.48 s |
| **1.5 s window, `window_centers`, tuned** | 0.585 | **0.523** | — |

Window-level validation macro F1: 0.615 (background 0.883, pickup 0.529, putdown 0.432).

The final row trades a little tIoU 0.3 for a lot of tIoU 0.5; thresholds were selected on
the mean of the two, on validation only.

### The boundary mode dominated everything else

The largest single error source was not the model. With `window_span`, a predicted interval
runs from the first above-threshold window's start to the last one's end, so a single 2.5 s
window yields a 2.5 s interval against a 0.9 s median event — tIoU is capped near
`event_duration / window_duration` ≈ 0.36 however good the classifier is. That is why tIoU
0.5 collapsed to 0.075 while tIoU 0.3 survived.

Deriving boundaries from window centres instead — a window is labelled by what sits at its
centre, so that is where its temporal evidence is — widened by half a stride, raised
validation F1 at tIoU 0.3 from 0.350 to 0.647 and recall from 0.219 to 0.688, with no
retraining at all. Shortening the window from 2.5 s to 1.5 s then cut boundary MAE from
1.35 s to 0.48 s.

`boundary_mode` defaults to `window_span` in code so existing callers are unaffected; the
Track B1 scripts and `configs/track_b1.yaml` select `window_centers`.

### Fine-tuning the last two backbone blocks

Unfreezing the last 2 encoder blocks, with the head warm-started from the frozen probe
and **discriminative learning rates** (head 1e-3, backbone 5e-5), clearly beat the frozen
probe on validation:

| validation | frozen probe | fine-tuned (best, epoch 4) |
|---|---|---|
| window macro F1 | 0.615 | **0.743** |
| event F1 @ tIoU 0.3 | 0.585 | **0.806** |
| event F1 @ tIoU 0.5 | 0.523 | **0.746** |
| boundary MAE | 0.48 s | **0.21 s / 0.33 s** |

A single learning rate for both head and backbone does *not* work: at the backbone's
5e-5 a randomly initialised head is still predicting all background after two epochs.
An earlier run configured that way looked like evidence that unfreezing fails; it was
only evidence that a head cannot train at a backbone's learning rate. Epochs 1-2 of the
corrected run are still unstable (epoch 1 reaches pickup 0.635 while putdown falls to
0.057) before it converges, so a short-patience run can also stop on the wrong conclusion.

### Test — the honest result

Both models, run on test with their own validation-selected configuration:

| test | frozen probe | fine-tuned |
|---|---|---|
| F1 @ tIoU 0.3 | 0.361 | **0.457** |
| F1 @ tIoU 0.5 | 0.278 | **0.314** |
| pickup F1 @ 0.5 | 0.316 | 0.393 |
| putdown F1 @ 0.5 | 0.133 | **0.000** |

Fine-tuning improves the headline number, and the improvement direction agrees with
validation. But validation says 0.806 and test says 0.457, and the gap is not noise —
it has a specific, diagnosable cause.

#### pickup/putdown discrimination does not survive a change of recording day

Mean predicted probability on windows whose ground-truth label is `putdown`:

| | p_putdown | p_pickup | fraction over threshold |
|---|---|---|---|
| validation day | 0.445 | 0.296 | 0.49 |
| test day | **0.062** | **0.588** | **0.02** |

On the test day the model is not merely uncertain about putdowns — it is *confidently
calling them pickups*. The maximum `p_putdown` over all true-putdown test windows is
0.467, so no threshold choice recovers this class; it is not a tuning failure. Predicted
composition on test is 33 pickup / 2 putdown against a ground truth of 23 / 12.

What the model appears to have learned is closer to "a shelf interaction is happening"
than "which direction the item moved". Pickup and putdown are near time-reverses of one
another, so the discriminating evidence is direction of transfer — a cue that is far more
sensitive to camera angle, shelf geometry and item type than mere presence of an
interaction, and that therefore transfers worst across days. Training saw three days
(20260520-22); test is 20260526.

This matters more than the headline F1: on an unseen day the system currently behaves as
an interaction detector with a pickup bias, which for a self-checkout use case is the
failure mode that matters — a putdown scored as a pickup is a charge for an item the
shopper returned.

Nothing was re-tuned on test in response to any of this.

### Independent per-actor predictions

Three actors in one validation clip, overlapping in time, with different event types — the
acceptance criterion observed in real output rather than argued from the design:

```
trk000 putdown 16.35-19.85
trk002 pickup  16.75-19.25
trk001 pickup  17.30-20.30
```

## 8b. Corrections and leave-one-day-out (post-review)

An external review of the manuscript found real errors, all verified against the artefacts:

- **FP/hour was wrong by ~3.7×.** `infer_track_b1.py` divided by the duration of all 42
  clips (2.31 h) instead of the evaluated split's (0.63 h test, 0.33 h val). Fixed; the
  tables above now carry the corrected figures.
- **Two threshold configurations were mixed** for the frozen probe: `metrics_val.json` was
  written at default thresholds (0.647 @ 0.3) while the sweep used tuned ones (0.585). All
  reporting now comes from one registry, `.local/paper/revision/registry.csv`.
- "Drift exceeds the median event" was false (it is ~39% of it); "27 s × 5192 ≈ 17 h" was
  false (that figure was for 2314 windows; 5192 gives ~39 h); "entire putdown distribution
  below threshold" was false (one window of 57 exceeds 0.45).

**Leave-one-day-out, frozen probe** (`scripts/revision_analysis.py`, selection on inner days only):

| held-out day | F1@0.3 | F1@0.5 | putdown F1 | FP/h |
|---|---|---|---|---|
| 2026-05-20 | 0.352 | 0.176 | 0.176 | 110 |
| 2026-05-21 | 0.387 | 0.168 | 0.070 | 145 |
| 2026-05-22 | 0.255 | 0.157 | 0.182 | 91 |
| 2026-05-23 | 0.468 | 0.312 | 0.158 | 99 |
| 2026-05-26 | 0.459 | 0.324 | 0.353 | 43 |
| **mean ± SD** | **0.384 ± 0.087** | 0.227 ± 0.084 | **0.188 ± 0.103** | 98 |

The single-split validation figure (0.585) sits more than two SD above the LODO mean: it
was optimistic. The putdown weakness recurs on every day, which strengthens the finding
while removing any claim that it is specific to one test day.

## 9. Known limitations

- **Actors are per-interaction, not per-person.** Two interactions by the same shopper are
  two actor streams. This does not affect window labelling, which is per-actor, but the
  identity is not a person identity.
- **Verified-negative crops are borrowed** (§3.2).
- **pickup/putdown direction does not generalise across days** (§8). This is the most
  important open problem, not the headline F1. Candidate remedies, roughly in order of
  expected value: day-level grouped cross-validation to measure it properly; explicit
  temporal-order supervision or a time-reversal augmentation that forces the model to
  use direction; more putdown data; and Track A's shelf-state evidence
  (`object_removed` / `object_placed`) fused in as a direction prior.
- **13 of 261 intervals are still `draft`** in CVAT. They are included by default;
  `--accepted-only` drops them.
- **Ignore intervals are sparse** (3 across 42 clips), so ignore handling is exercised but
  barely stressed.
- **Only five recording days exist**, so val and test are one day each and both scores are
  high-variance (§8). Grouped 5-fold cross-validation over days is the right fix and is the
  highest-value next step.
- **putdown is under-represented** — 89 events against 170 pickups, and only 12 in test.
  Its scores are the least reliable numbers in this document.
- **Two test evaluations have now been run** (frozen probe, then fine-tuned). Neither
  informed any choice — model selection and every threshold came from validation alone —
  but the test split is no longer perfectly naive and should be treated accordingly.
