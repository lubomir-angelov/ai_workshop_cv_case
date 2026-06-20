# task_12_hard: Track B1 Actor-Conditioned VideoMAE Window Classifier

> This task belongs to the pickup/putdown temporal action detection implementation.
> Read `docs/concepts.md`, the copied `manifest/labeling-guidelines.md`, and
> `PICKUP_PUTDOWN_IMPLEMENTATION_PLAN_CONCEPTS_ALIGNED.md` before starting.
> The reference case repository is read-only; implementation artifacts belong in the solution repository.

**Task ID:** `task_12`  
**Difficulty:** `hard`  
**Dependencies:** Tasks 5, 7, and 8.  
**Parallel work:** Tasks 9–11 and 14.

## Objective

Implement the learned fixed-window baseline using actor-conditioned VideoMAE inputs and temporal score smoothing.

## Inputs

- Actor-specific candidates
- Train/validation/test split
- Events, ignore intervals, and hard negatives
- Source videos

## Deliverables

- Actor-conditioned crop/window dataset
- VideoMAE classifier for background/pickup/putdown
- Visual data-loader inspection
- Tiny-overfit test
- Sliding-window inference and same-type decoding
- Canonical Track B1 predictions and metrics

## Expected Files or Modules

- `src/pickup_putdown/layer1/track_b1/dataset.py`
- `src/pickup_putdown/layer1/track_b1/videomae_classifier.py`
- `src/pickup_putdown/layer1/track_b1/train.py`
- `src/pickup_putdown/layer1/track_b1/inference.py`
- `configs/track_b1.yaml`

## Implementation Steps

1. Create one input per actor and active region using the union of actor boxes and relevant shelf region plus margin.
2. Decode only required intervals and uniformly sample chronologically ordered frames; do not pre-extract all frames as JPEGs.
3. Create event-centered positive windows and hard-negative windows. Exclude ignore intervals and no-person clips.
4. For nearby pickup/putdown events, use shorter windows/finer stride or center-based labels; do not delete the original event rows.
5. Render and inspect sampled-frame grids before training.
6. Pass a tiny-overfit test before a full training run.
7. Start with a frozen pretrained VideoMAE backbone and train the classifier head; optionally unfreeze final blocks after stability.
8. At inference, slide over candidates, smooth class probabilities, and merge only same-type compatible segments.
9. Allow multiple ordered events from one candidate and evaluate with the shared evaluator.

## Acceptance Criteria

- [ ] Temporal order is verified visually.
- [ ] Actor A and actor B can produce independent predictions in the same source clip.
- [ ] Different-type adjacent events are not merged.
- [ ] The tiny-overfit test succeeds.
- [ ] Validation thresholds are recorded in configuration and test data remains untouched.

## Out of Scope

- Cached temporal feature detector
- Qwen verification
- Product counting

## Handoff Contract

The task owner must provide:

- a pull request containing the implementation and tests;
- the resolved configuration used for the acceptance run;
- one machine-readable sample output or fixture;
- a short note listing assumptions, known limitations, and any interface changes;
- confirmation that no source videos, credentials, or model weights were committed.
