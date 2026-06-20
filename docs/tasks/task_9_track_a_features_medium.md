# task_9_medium: Track A Crop Extraction and Appearance Features

> This task belongs to the pickup/putdown temporal action detection implementation.
> Read `docs/concepts.md`, the copied `manifest/labeling-guidelines.md`, and
> `PICKUP_PUTDOWN_IMPLEMENTATION_PLAN_CONCEPTS_ALIGNED.md` before starting.
> The reference case repository is read-only; implementation artifacts belong in the solution repository.

**Task ID:** `task_9`  
**Difficulty:** `medium`  
**Dependencies:** Tasks 5 and 7.  
**Parallel work:** Task 8 and initial Track B preparation.

## Objective

Build the actor/hand/shelf crop dataset and frozen image features required by the interpretable Track A baseline.

## Inputs

- Actor-specific candidates and wrist trajectories
- Ground-truth events and hard negatives
- Shelf regions
- Source videos

## Deliverables

- Pre/contact/post timestamp selection
- Actor-specific hand crops
- Local shelf/contact patches
- Crop manifest with event/negative labels
- Frozen DINOv2/SigLIP/CLIP/MobileNet embeddings
- Visual crop-QA report

## Expected Files or Modules

- `src/pickup_putdown/layer1/track_a/crop_extractor.py`
- `src/pickup_putdown/layer1/track_a/image_features.py`
- `src/pickup_putdown/layer1/common/actor_crops.py`
- `tests/test_crop_extractor.py`

## Implementation Steps

1. For each candidate transition, identify pre-contact, contact, and post-contact sampling points while preserving chronological order.
2. Create hand crops centered on the relevant wrist with scale derived from actor box or limb geometry.
3. Create shelf patches around the estimated contact point and active region.
4. Include positive transfers, touch-only interactions, browsing, reaching, carrying near shelves, and visible restocking negatives.
5. Exclude ignore intervals and fully occluded evidence.
6. Use a frozen lightweight image encoder and cache embeddings keyed by source checksum, timestamp, crop geometry, and encoder version.
7. Visually inspect at least the configured sample count across pickup, putdown, hard negatives, multiple people, and edge cases.
8. Ensure crops from test clips cannot enter training caches used for fitting classifiers.

## Acceptance Criteria

- [ ] Crops identify the intended actor/hand/region in QA samples.
- [ ] Feature extraction is deterministic and cacheable.
- [ ] Temporal order and pre/contact/post labels are preserved.
- [ ] Split leakage checks pass at the source-clip level.
- [ ] The output can be consumed without decoding videos again.

## Out of Scope

- Final state-machine decision logic
- VideoMAE training
- Qwen

## Handoff Contract

The task owner must provide:

- a pull request containing the implementation and tests;
- the resolved configuration used for the acceptance run;
- one machine-readable sample output or fixture;
- a short note listing assumptions, known limitations, and any interface changes;
- confirmation that no source videos, credentials, or model weights were committed.
