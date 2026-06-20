# task_8_hard: Shared Two-Pass Temporal Evaluator and Reports

> This task belongs to the pickup/putdown temporal action detection implementation.
> Read `docs/concepts.md`, the copied `manifest/labeling-guidelines.md`, and
> `PICKUP_PUTDOWN_IMPLEMENTATION_PLAN_CONCEPTS_ALIGNED.md` before starting.
> The reference case repository is read-only; implementation artifacts belong in the solution repository.

**Task ID:** `task_8`  
**Difficulty:** `hard`  
**Dependencies:** Task 1 schemas. Synthetic fixtures are sufficient to begin; Task 7 provides real data.  
**Parallel work:** All perception and model tasks.

## Objective

Implement one evaluator used by Track A, Track B1, Track B2, standalone Layer 2, and Layer 3.

## Inputs

- Canonical `events.csv` and `predictions.csv`
- Clip durations
- Optional hard-case/confidence metadata

## Deliverables

- Class-aware one-to-one matcher
- Class-agnostic temporal matcher for type confusion
- tIoU and midpoint metrics
- Multi-item and event-count metrics
- Runtime and false-positive-per-hour metrics
- Markdown/HTML report and failure gallery hooks

## Expected Files or Modules

- `src/pickup_putdown/evaluation/class_aware_matching.py`
- `src/pickup_putdown/evaluation/confusion_matching.py`
- `src/pickup_putdown/evaluation/metrics.py`
- `src/pickup_putdown/evaluation/report.py`
- `tests/test_evaluation.py`

## Implementation Steps

1. Implement interval tIoU and midpoint distance with numerical edge-case tests.
2. Implement maximum-weight one-to-one matching for same-type events. Do not greedily match in a way that depends on input order.
3. Implement a second temporal-only matching pass, then compare matched types to count pickup→putdown and putdown→pickup confusion.
4. Count duplicate ground-truth rows for two-item actions separately; one prediction must not satisfy two ground-truth rows.
5. Report precision, recall, F1 at tIoU 0.3, tIoU 0.5, and midpoint tolerance ±1 second.
6. Report start/end MAE, false positives per video hour, absolute event-count error, and multi-item recall.
7. Provide slices for high/med only, low confidence, hard cases, and multiple-person cases where metadata is available.
8. Exclude ignore intervals from official matching.
9. Add fixtures for no predictions, no ground truth, overlapping events, type flips, immediate pickup/putdown, and two identical-time rows.

## Acceptance Criteria

- [ ] Metrics are invariant to row ordering.
- [ ] A type flip appears as FP/FN in class-aware metrics and as explicit confusion in pass two.
- [ ] Two-item ground truth requires two matched prediction rows.
- [ ] All models can be evaluated without model-specific code.
- [ ] Thresholds are inputs and never optimized against test labels by the evaluator.

## Out of Scope

- Prediction generation
- Model training
- Dataset annotation

## Handoff Contract

The task owner must provide:

- a pull request containing the implementation and tests;
- the resolved configuration used for the acceptance run;
- one machine-readable sample output or fixture;
- a short note listing assumptions, known limitations, and any interface changes;
- confirmation that no source videos, credentials, or model weights were committed.
