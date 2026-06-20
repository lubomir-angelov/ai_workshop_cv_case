# task_5_hard: Layer 0B Pose Tracking and Actor-Specific Interaction Proposals

> This task belongs to the pickup/putdown temporal action detection implementation.
> Read `docs/concepts.md`, the copied `manifest/labeling-guidelines.md`, and
> `PICKUP_PUTDOWN_IMPLEMENTATION_PLAN_CONCEPTS_ALIGNED.md` before starting.
> The reference case repository is read-only; implementation artifacts belong in the solution repository.

**Task ID:** `task_5`  
**Difficulty:** `hard`  
**Dependencies:** Tasks 3 and 4.  
**Parallel work:** Task 6 may proceed with manual candidates; Task 8 can build evaluator fixtures.

## Objective

Use pose keypoints, actor tracks, and fixed shelf regions to generate broad, high-recall interaction candidates without claiming pickup or putdown.

## Inputs

- Person tracks and active spans
- Shelf/surface configuration
- Video files
- Proposal configuration

## Deliverables

- Higher-rate pose inference over active spans
- Actor- and hand-specific wrist trajectories
- Broad interaction candidate intervals
- `tracks_pose.parquet` and `candidates.parquet`
- Candidate preview clips
- Proposal-recall measurement hooks

## Expected Files or Modules

- `src/pickup_putdown/perception/pose_tracker.py`
- `src/pickup_putdown/perception/proposals.py`
- `src/pickup_putdown/perception/candidate_previews.py`
- `configs/proposals.yaml`
- `tests/test_proposals.py`

## Implementation Steps

1. Run a pose model at an initial 8 FPS over person-active spans; compare 4 FPS and 2 FPS later.
2. Associate pose detections to existing actor tracks or maintain a stable actor ID mapping.
3. For each actor, hand side, and region, calculate wrist confidence, distance to region, region entry/exit, dwell time, and velocity reversal.
4. Create a raw interaction when a confident wrist is in or near the expanded region for the configured minimum duration.
5. Merge broad candidates only for the same actor, hand, and region, and only across a short gap.
6. Add pre/post context while preserving the raw interaction timestamps.
7. Do not label candidate type. One candidate may contain zero, one, or multiple ordered events.
8. Render candidate previews with actor ID, hand side, region, raw interval, and padded interval.
9. Provide an API that measures whether each ground-truth event is covered by at least one candidate.

## Acceptance Criteria

- [ ] Candidates are never written to canonical `predictions.csv`.
- [ ] Simultaneous actors produce independent candidates.
- [ ] A pickup followed immediately by a putdown can remain in one candidate without being merged into one event.
- [ ] Candidate timestamps are valid and clamped to source duration.
- [ ] Proposal recall can be computed once labels exist.

## Out of Scope

- Event classification
- Hand-carry state
- Qwen inference

## Handoff Contract

The task owner must provide:

- a pull request containing the implementation and tests;
- the resolved configuration used for the acceptance run;
- one machine-readable sample output or fixture;
- a short note listing assumptions, known limitations, and any interface changes;
- confirmation that no source videos, credentials, or model weights were committed.
