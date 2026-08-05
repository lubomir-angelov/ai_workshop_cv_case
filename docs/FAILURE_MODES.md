# Failure Modes and Error Analysis

This document defines how recurring failures are classified, handled, and
reported. It keeps annotation, training, inference, evaluation, and final
reporting consistent.

Do not change event definitions, labels, thresholds, or test membership to hide
a failure. Fix the responsible stage, measure the result on validation data,
and preserve evidence for audit.

## How to use this document

Use this taxonomy when:

- reviewing annotation disagreements;
- inspecting Stage A/B previews;
- analyzing model false positives and false negatives;
- selecting validation-set mitigations;
- generating evaluation slices and failure galleries;
- documenting limitations in the final report.

Every investigated failure should record:

```text
failure_id
failure_type
clip_id
event_id or prediction_id
affected_stage
observed_symptom
root_cause
action_taken
review_status
```

Do not store personal identity information in failure records.

## Policy summary

| Evidence | Annotation policy | Training policy | Evaluation policy |
|---|---|---|---|
| Visible and likely transfer | Official event, possibly `confidence=low` | Include with configured confidence weight | Include in official matching |
| Difficult but labelable transfer | Official event with `hard_case=true` | Include and retain hard-case metadata | Include and report hard-case slice |
| Decisive evidence unavailable | Internal ignore interval | Zero weight; never sample as background | Exclude ignored interval |
| Visible non-event | No event; hard negative when useful | Include as background | Prediction counts as false positive |
| Corrupt or unusable media | Mark unusable or ignore corrupt section | Exclude affected section | Exclude only through documented policy |

## Data and visibility failures

### FM-DATA-01: Full occlusion

**Scenario:** A body, shelf, object, or another actor fully hides the hand,
item, or decisive transfer.

**Affected stages:** Annotation, Track A/B, Layer 2, Layer 3.

**Expected symptom:** Models infer an event from approach/withdrawal motion
without visible transfer evidence.

**Annotation policy:** Create an ignore interval with
`reason=ACTION_OCCLUDED`. Do not create an official event.

**Training and evaluation:** Assign zero training weight, never sample as
background, and exclude the ignored interval from official matching.

**Required diagnostics:** Ignore reason, interval, preview reference, and
ignored-duration summary.

**Mitigation:** None from the same view when evidence is absent. Additional
camera evidence requires a separately defined multi-view policy.

**Residual risk:** Nearby visible motion may still produce false positives.

### FM-DATA-02: Partial occlusion

**Scenario:** Transfer is observable, but object count, type, or boundaries are
uncertain.

**Affected stages:** Annotation and all event detectors.

**Expected symptom:** Boundary disagreement, pickup/putdown reversal, or
incorrect item count.

**Annotation policy:** Keep the event. Use `hard_case=true` and
`confidence=low` when uncertainty is material.

**Training and evaluation:** Include with configured confidence weight. Include
in official metrics and report low-confidence and hard-case slices separately.

**Required diagnostics:** Visibility note, confidence, hard-case flag, and
boundary disagreement where double annotation exists.

**Mitigation:** Inspect at reduced speed with wider temporal context.

### FM-DATA-03: Out-of-frame transfer

**Scenario:** Approach or withdrawal is visible, but transfer occurs outside
the frame.

**Affected stages:** Annotation and all detectors.

**Expected symptom:** Event type is guessed from actor direction or hand state.

**Annotation policy:** Create `ACTION_OUT_OF_FRAME` ignore interval. Do not
create an event unless the decisive transition is visible.

**Training and evaluation:** Same handling as full occlusion.

**Required diagnostics:** Boundary proximity and ignored-duration summary.

**Mitigation:** None for the current view. Do not infer labels from assumed
inventory state.

### FM-DATA-04: Clip-boundary truncation

**Scenario:** Action starts before the clip or completes after it.

**Affected stages:** Annotation, window generation, boundary estimation.

**Expected symptom:** Incomplete interval, wrong type, or prediction clamped to
the first/last frame.

**Annotation policy:** Use `CLIP_BOUNDARY` ignore interval when decisive
evidence is unavailable. Keep an event only when the complete defining
transition is visible.

**Training and evaluation:** Exclude ignored sections. Report frequency of
boundary-truncated clips.

**Required diagnostics:** Whether event touches clip start/end and which
transition evidence is missing.

**Mitigation:** Preserve context when source footage allows re-extraction.

### FM-DATA-05: Corrupt media or timestamp drift

**Scenario:** Decode fails, frames are missing, or container timestamps do not
match decoded time.

**Affected stages:** Ingestion and every downstream stage.

**Expected symptom:** Probe failure, non-monotonic timestamps, invalid
intervals, or preview/export mismatch.

**Annotation policy:** Mark full clip unusable or create `CORRUPT_SECTION`
ignore intervals.

**Training and evaluation:** Exclude only through documented dataset-version
policy. Never repair labels silently.

**Required diagnostics:** Source key, checksum, probe/decode status, failing
time range, and tool version.

**Mitigation:** Re-encode only into a derived artifact while preserving the
immutable source reference and time mapping.

## Event-semantic failures

### FM-EVENT-01: Touching or browsing predicted as an event

**Scenario:** Hand enters a shelf region but no persistent object transfer
occurs.

**Affected stages:** Stage B, Track A/B, Layer 2.

**Expected symptom:** High candidate volume or false-positive pickup/putdown.

**Annotation policy:** No event. Retain representative cases as hard negatives.

**Training and evaluation:** Train as background. A prediction is a false
positive.

**Required diagnostics:** False positives grouped by touch, inspection, reach,
or browsing.

**Mitigation:** Require persistent shelf-to-hand or hand-to-shelf state change.
Region entry/exit alone is insufficient.

### FM-EVENT-02: Restocking predicted as putdown

**Scenario:** A person introduces a new item and places it on a shelf.

**Affected stages:** Track A/B, Layer 2, Layer 3.

**Expected symptom:** Putdown false positives with no evidence the item was
previously taken in the relevant context.

**Annotation policy:** No event. Visible restocking is a hard negative, not an
ignore interval.

**Training and evaluation:** Include as background. Predicted putdown counts as
a false positive.

**Required diagnostics:** Restocking false-positive count and previews with
pre-action context.

**Mitigation:** Provide enough temporal context to establish prior hand state.
Layer 3 should reject with `RESTOCKING_NOT_PUTDOWN`.

### FM-EVENT-03: Pickup/putdown reversal

**Scenario:** Event timing is correct but temporal direction is classified
incorrectly.

**Affected stages:** Track A/B, Layer 2, Layer 3.

**Expected symptom:** Pickup predicted as putdown or vice versa.

**Annotation policy:** Keep canonical label unchanged.

**Training and evaluation:** Count as FP/FN in class-aware matching and as
explicit type confusion in class-agnostic matching.

**Required diagnostics:** Both confusion directions, chronological frame grid,
and pre/contact/post state evidence.

**Mitigation:** Preserve frame order and improve pre/post object-state evidence.

### FM-EVENT-04: Immediate pickup and putdown merged

**Scenario:** Close opposite-type transitions become one event.

**Affected stages:** Annotation import, temporal decoders, post-processing.

**Expected symptom:** One long interval replaces two ordered events.

**Annotation policy:** Create separate ordered pickup and putdown rows.

**Training and evaluation:** Preserve both rows. One prediction cannot satisfy
both.

**Required diagnostics:** Count of adjacent opposite-type events and merge-rule
audit.

**Mitigation:** Merge only duplicate events of the same type. Use shorter
windows or finer temporal stride where required.

### FM-EVENT-05: Multiple-item undercount

**Scenario:** Two simultaneous transfers produce one event row.

**Affected stages:** Annotation export, item-count estimation, final export.

**Expected symptom:** Correct type and interval but incorrect event count.

**Annotation policy:** Create one official row per item with unique IDs and
optional shared `event_group_id`.

**Training and evaluation:** Preserve duplicate-time rows. One prediction may
match only one ground-truth row.

**Required diagnostics:** Multi-item recall and absolute event-count error.

**Mitigation:** Keep count logic separate from event-type classification.
Uncertain count must not silently create duplicate rows.

### FM-EVENT-06: Simultaneous actors merged

**Scenario:** Evidence or predictions from two people are combined.

**Affected stages:** Tracking, proposal generation, actor crops, decoding.

**Expected symptom:** ID switches, one missing event, or one interval spanning
two actors.

**Annotation policy:** Label every visible actor event. Internal actor IDs stay
clip-local.

**Training and evaluation:** Use actor-conditioned internal processing, while
canonical evaluation remains event-row based.

**Required diagnostics:** Multi-person slice, ID-switch examples, and
actor-conditioned previews.

**Mitigation:** Associate pose and crops to stable person tracks; never infer
personal identity.

## Layer 0 failures

### FM-L0-01: Person-containing clip classified as no-person

**Owner:** Stage A person triage.

**Impact:** Entire downstream pipeline is skipped, creating systematic false
negatives.

**Detection:** Manually review the configured sample of no-person decisions and
all low-confidence or short detections.

**Required metric:** Person-containing clip recall.

**Mitigation:** Compare 1 FPS and 2 FPS, increase image size, then consider a
larger detector. Tune using validation data.

### FM-L0-02: Fragmented or switched person tracks

**Owner:** Stage A tracking and Stage B actor association.

**Impact:** Active spans fragment; pose observations or candidates attach to
the wrong actor.

**Detection:** Track-ID previews, duration/gap statistics, and multi-person
failure gallery.

**Mitigation:** Tune association and gap settings after detector recall is
acceptable. Preserve timestamps and source frame IDs.

### FM-L0-03: Wrist or pose detection failure

**Owner:** Stage B pose tracking.

**Impact:** Valid interactions receive no candidate or wrong hand assignment.

**Detection:** Pose previews and proposal-recall false negatives.

**Required metric:** Stage B proposal recall, separate from detector metrics.

**Mitigation:** Compare 2/4/8 FPS, image size, confidence, and active-span
coverage on validation data.

### FM-L0-04: Shelf-region configuration error

**Owner:** Shelf configuration.

**Impact:** Valid interactions fall outside regions or unrelated motion creates
proposals.

**Detection:** Overlay review at configured source resolution.

**Mitigation:** Version polygon changes, validate bounds, and retain reviewed
overlay evidence. Do not silently scale incompatible coordinates.

### FM-L0-05: Candidate recall failure

**Owner:** Stage B proposal generation.

**Impact:** Candidate-dependent Layer 1 systems cannot recover omitted events.

**Detection:** Measure coverage against complete-active-span ground truth.

**Required metric:** Proposal recall by event type, actor count, and visibility
slice.

**Mitigation:** Favor recall over precision, widen context/regions, adjust
dwell and merge settings on validation data. Candidates must never redefine
ground truth.

## Model and post-processing failures

### FM-MODEL-01: Rare fast action missed

**Scenario:** Sampling rate or temporal stride skips decisive frames.

**Detection:** Short-event slice and 2/4/8 FPS benchmark.

**Mitigation:** Increase sampling rate or use finer windows where measured
recall justifies cost.

**Residual risk:** Motion blur may leave the transfer unobservable even at
higher sampling.

### FM-MODEL-02: Candidate boundary copied as event boundary

**Scenario:** Broad context window is emitted as the event interval.

**Impact:** Poor tIoU and boundary MAE despite correct event presence.

**Detection:** Compare candidate and event boundaries; inspect unusually long
predictions.

**Mitigation:** Estimate transfer onset and stable end separately. Record
explicit fallback method when only wrist-region boundaries are available.

### FM-MODEL-03: Duplicate predictions from overlapping windows

**Scenario:** One event is emitted by several adjacent windows.

**Impact:** Precision loss and event-count inflation.

**Detection:** Same-type overlapping predictions sharing source evidence.

**Mitigation:** Deterministic type-aware merging or temporal NMS. Never merge
pickup and putdown solely because they are close.

### FM-MODEL-04: Threshold overfitting or test leakage

**Scenario:** Thresholds, prompts, merges, or checkpoints are selected after
examining test results.

**Impact:** Reported performance is biased and not reproducible.

**Detection:** Missing threshold provenance, changed split hash, or repeated
test-guided revisions.

**Mitigation:** Freeze test split, tune on validation only, and record
configuration and selecting run.

## VLM failures

### FM-VLM-01: Invalid or non-conforming response

**Scenario:** Response is invalid JSON, uses unsupported values, or contains
invalid timestamps.

**Affected stages:** Layer 2 and Layer 3.

**Handling:** Validate with strict schemas, retry once, preserve raw response,
and record parse failure. Invalid output must never enter canonical
predictions.

**Required metrics:** Invalid-response and retry-success rates.

### FM-VLM-02: Window-relative timestamp conversion error

**Scenario:** Relative VLM times are mapped to wrong source times.

**Impact:** Correct semantic result appears at the wrong interval.

**Detection:** Round-trip tests using window start, source FPS, and overlays.

**Mitigation:** Preserve window metadata and use one deterministic conversion
path.

### FM-VLM-03: Hardware or model-size constraint

**Scenario:** Qwen3.6-27B exceeds available memory or runtime budget.

**Impact:** Reduced coverage, changed quantization, or incomplete results.

**Handling:** Record exact model, quantization, backend, GPU, peak VRAM,
sampling configuration, runtime, and deviation reason.

**Reporting:** Never compare incomplete or materially different configurations
as though they were equivalent.

## Reproducibility and privacy failures

### FM-OPS-01: Incomplete run metadata

**Scenario:** Results lack Git commit, resolved config, dataset/split version,
seed, model ID, or checkpoint hash.

**Impact:** Run cannot be reproduced or audited.

**Mitigation:** Treat missing required metadata as a failed run, not a warning.

### FM-OPS-02: Incompatible artifact overwrite

**Scenario:** A run replaces outputs produced by different inputs or settings.

**Impact:** Evidence and lineage are lost.

**Mitigation:** Use immutable/versioned run directories and compare metadata
before resuming.

### FM-OPS-03: Sensitive media or secrets exposed

**Scenario:** Raw footage, unblurred imagery, credentials, signed URLs, or real
annotator data enters Git or a public report.

**Impact:** Privacy or security incident.

**Mitigation:** Follow `docs/DATA_PRIVACY.md`, stop distribution, rotate
credentials, notify owners, and use approved remediation.

## Required final reporting

Report at minimum:

- precision, recall, and F1 at required temporal criteria;
- pickup-to-putdown and putdown-to-pickup confusion;
- start/end timing error;
- false positives per video hour;
- absolute event-count error and multi-item recall;
- Stage B proposal recall;
- high/medium versus low-confidence slices;
- normal versus hard-case slices;
- single-person versus multiple-person slices;
- short versus long event slices;
- runtime per video minute;
- invalid VLM response rate and hardware details where applicable;
- representative privacy-safe false-positive and false-negative examples.

## Adding a new failure mode

Add a new entry when a failure recurs or materially affects validity. Include:

1. stable failure ID and owner;
2. concrete scenario and observable symptom;
3. affected pipeline stages;
4. annotation, training, and evaluation policy;
5. required diagnostic artifact or metric;
6. mitigation selected on validation data;
7. residual risk after mitigation.

Record fixes in a new run or dataset version. Do not rewrite prior results.
