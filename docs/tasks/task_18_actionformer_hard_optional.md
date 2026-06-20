# task_18_hard_optional: Optional ActionFormer Temporal Detector Benchmark

> This task belongs to the pickup/putdown temporal action detection implementation.
> Read `docs/concepts.md`, the copied `manifest/labeling-guidelines.md`, and
> `PICKUP_PUTDOWN_IMPLEMENTATION_PLAN_CONCEPTS_ALIGNED.md` before starting.
> The reference case repository is read-only; implementation artifacts belong in the solution repository.

**Task ID:** `task_18`  
**Difficulty:** `hard`  
**Status:** `optional`  
**Dependencies:** Tasks 7, 8, and 13. Begin only after Track B2 is stable.  
**Parallel work:** Not a critical-path task.

## Objective

Adapt cached actor-conditioned VideoMAE features to ActionFormer and compare it fairly against the simpler TCN.

## Inputs

- Timestamped VideoMAE features
- Ground-truth temporal segments
- Frozen split and evaluator
- ActionFormer repository/version

## Deliverables

- Dataset adapter
- ActionFormer configuration
- Training/inference scripts or wrapper
- Canonical predictions
- Controlled comparison with Track B2
- Integration notes and patch record

## Expected Files or Modules

- `src/pickup_putdown/layer1/actionformer/adapter.py`
- `src/pickup_putdown/layer1/actionformer/inference.py`
- `configs/actionformer.yaml`
- `docs/ACTIONFORMER_INTEGRATION.md`

## Implementation Steps

1. Pin an exact ActionFormer commit and document any local modifications.
2. Convert actor-conditioned timestamped features and event labels to the expected feature/annotation format.
3. Preserve source-time mapping so output segments convert exactly to clip timestamps.
4. Train only on the existing training split and tune on validation data.
5. Export predictions through the same canonical schema and evaluator.
6. Compare accuracy, boundary quality, training complexity, runtime, and engineering effort against B2.
7. Retain B2 as the default unless ActionFormer provides a clear measured benefit.

## Acceptance Criteria

- [ ] ActionFormer and B2 use identical source features and splits.
- [ ] Prediction timestamps round-trip correctly to source clips.
- [ ] No model-specific evaluator is used.
- [ ] The report explicitly states whether the added complexity is justified.

## Out of Scope

- Replacing the required Track A baseline
- Changing the frozen test split
- Streaming deployment

## Handoff Contract

The task owner must provide:

- a pull request containing the implementation and tests;
- the resolved configuration used for the acceptance run;
- one machine-readable sample output or fixture;
- a short note listing assumptions, known limitations, and any interface changes;
- confirmation that no source videos, credentials, or model weights were committed.
