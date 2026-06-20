# task_11_hard_optional: Shared Non-VLM Item-Count Estimator and Row Expansion

> This task belongs to the pickup/putdown temporal action detection implementation.
> Read `docs/concepts.md`, the copied `manifest/labeling-guidelines.md`, and
> `PICKUP_PUTDOWN_IMPLEMENTATION_PLAN_CONCEPTS_ALIGNED.md` before starting.
> The reference case repository is read-only; implementation artifacts belong in the solution repository.

**Task ID:** `task_11`  
**Difficulty:** `hard`  
**Status:** `optional`  
**Dependencies:** Tasks 7, 8, and 9. Can integrate after Task 10 or Track B predictions exist.  
**Parallel work:** Tasks 12–14.

## Objective

Estimate whether an accepted Layer 1 event involves one or two-or-more items and expand multi-item actions into separate canonical rows.

## Inputs

- Event-aligned actor/hand/shelf crops
- Multi-item ground-truth examples
- Layer 1 event predictions

## Deliverables

- Item-count training manifest
- One-item / two-plus / uncertain estimator
- Count confidence and review flag
- Canonical duplicate-row expansion
- Multi-item evaluation report

## Expected Files or Modules

- `src/pickup_putdown/layer1/common/item_count.py`
- `src/pickup_putdown/layer1/common/multi_item_export.py`
- `configs/item_count.yaml`
- `tests/test_multi_item_export.py`

## Implementation Steps

1. Build a count dataset using event-aligned crops and internal event groups.
2. Support one item, two-or-more items, and uncertain. Do not force uncertain cases into two rows.
3. Use a lightweight frozen-feature classifier or deterministic evidence if the dataset is too small.
4. For two-plus predictions, create two canonical event rows with the same interval and score and a shared internal event group.
5. Preserve per-hand evidence so one item in each hand can naturally produce two grouped events.
6. Keep count logic separate from pickup/putdown type classification.
7. Tune the count threshold on validation data and report multi-item recall and event-count error.

## Acceptance Criteria

- [ ] One predicted row cannot satisfy two ground-truth item rows in evaluation.
- [ ] Two-item output contains two unique `pred_id` values.
- [ ] Uncertain count does not silently duplicate events.
- [ ] Track A, B1, and B2 can all call the same post-processing interface.

## Out of Scope

- Product identity
- Inventory counting
- Object segmentation

## Handoff Contract

The task owner must provide:

- a pull request containing the implementation and tests;
- the resolved configuration used for the acceptance run;
- one machine-readable sample output or fixture;
- a short note listing assumptions, known limitations, and any interface changes;
- confirmation that no source videos, credentials, or model weights were committed.
