# task_13_hard_optional: Track B2 Cached VideoMAE Features and Temporal Convolutional Detector

> This task belongs to the pickup/putdown temporal action detection implementation.
> Read `docs/concepts.md`, the copied `manifest/labeling-guidelines.md`, and
> `PICKUP_PUTDOWN_IMPLEMENTATION_PLAN_CONCEPTS_ALIGNED.md` before starting.
> The reference case repository is read-only; implementation artifacts belong in the solution repository.

**Task ID:** `task_13`  
**Difficulty:** `hard`  
**Status:** `optional`  
**Dependencies:** Tasks 7, 8, and 12. Task 12 supplies validated actor-conditioned loading and encoder setup.  
**Parallel work:** Tasks 14 and 15.

## Objective

Extract reusable timestamped VideoMAE feature sequences and train a lightweight temporal detector that refines event intervals.

## Inputs

- Actor-conditioned active spans/candidates
- Pretrained VideoMAE encoder
- Ground-truth events and ignore intervals

## Deliverables

- Feature extraction command and cache
- Timestamped `[T,D]` feature files
- Per-timestep labels with ignore mask
- Small TCN temporal head
- Type-aware interval decoder
- Canonical Track B2 predictions and comparison with B1

## Expected Files or Modules

- `src/pickup_putdown/layer1/track_b2/feature_extractor.py`
- `src/pickup_putdown/layer1/track_b2/temporal_head.py`
- `src/pickup_putdown/layer1/track_b2/train.py`
- `src/pickup_putdown/layer1/track_b2/inference.py`
- `configs/track_b2.yaml`

## Implementation Steps

1. Extract overlapping micro-clips in chronological order and cache one embedding with its represented timestamp per temporal position.
2. Key caches by dataset version, source checksum, crop definition, encoder checkpoint, FPS, and sampling configuration.
3. Generate background, pickup, putdown, and ignore labels per temporal position.
4. Implement a small residual/dilated Conv1D head with class probabilities over time.
5. Pass a tiny sequence-overfit test before full training.
6. Decode separate class score streams using smoothing, thresholds, gap filling, minimum duration, and temporal NMS.
7. Never merge different event types; permit pickup then putdown inside one candidate.
8. Tune all decoder thresholds on validation data.
9. Compare B2 against B1 using identical canonical evaluation and report whether added complexity improves localization.

## Acceptance Criteria

- [ ] Feature extraction is run once and temporal-head experiments do not decode raw video again.
- [ ] Ignore positions contribute zero loss.
- [ ] Cache metadata prevents incompatible feature reuse.
- [ ] B2 emits valid event intervals and multiple events where appropriate.
- [ ] Comparison against B1 uses the same split and evaluator.

## Out of Scope

- ActionFormer integration
- VLM inference
- Live streaming

## Handoff Contract

The task owner must provide:

- a pull request containing the implementation and tests;
- the resolved configuration used for the acceptance run;
- one machine-readable sample output or fixture;
- a short note listing assumptions, known limitations, and any interface changes;
- confirmation that no source videos, credentials, or model weights were committed.
