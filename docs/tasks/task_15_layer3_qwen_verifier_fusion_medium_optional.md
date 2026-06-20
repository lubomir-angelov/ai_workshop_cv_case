# task_15_medium_optional: Layer 3 Qwen Verification and Deterministic Fusion

> This task belongs to the pickup/putdown temporal action detection implementation.
> Read `docs/concepts.md`, the copied `manifest/labeling-guidelines.md`, and
> `PICKUP_PUTDOWN_IMPLEMENTATION_PLAN_CONCEPTS_ALIGNED.md` before starting.
> The reference case repository is read-only; implementation artifacts belong in the solution repository.

**Task ID:** `task_15`  
**Difficulty:** `medium`  
**Status:** `optional`  
**Dependencies:** Tasks 8, 10 or 12/13, and 14 client/schema components.  
**Parallel work:** Task 16 integration can begin with fixtures.

## Objective

Verify Layer 1 proposals with Qwen and apply auditable deterministic fusion rules.

## Inputs

- Layer 1 event predictions
- Source videos
- Qwen client and prompt infrastructure
- Fusion configuration

## Deliverables

- Verification clip renderer with context
- Verifier response schema
- Audit JSONL
- Deterministic accept/reject/relabel/review rules
- Multi-item row expansion using Qwen count
- Canonical final predictions

## Expected Files or Modules

- `src/pickup_putdown/layer3/verifier.py`
- `src/pickup_putdown/layer3/fusion.py`
- `configs/layer3_fusion.yaml`
- `tests/test_fusion.py`

## Implementation Steps

1. For each Layer 1 prediction, render a short clip with enough pre-context to distinguish returning an item from visible restocking.
2. Ask Qwen only to verify event presence, visibility, type, and item count. Layer 1 remains the primary interval source.
3. Preserve original Layer 1 type, interval, and confidence in the audit record.
4. Implement explicit rules for invisible, no event, restocking, confirmed type, changed type, uncertain, and multiple items.
5. Use configured confidence thresholds selected on validation data.
6. Never overwrite raw Layer 1 or Qwen outputs.
7. Export only accepted rows to official predictions; keep rejected and review-required decisions in audit artifacts.
8. Evaluate Layer 3 independently from Layer 1 and standalone Layer 2.

## Acceptance Criteria

- [ ] Every final row can be traced to Layer 1 evidence and one Qwen verification record.
- [ ] Restocking verification rejects the event rather than converting it to putdown.
- [ ] Changed types are explicitly flagged.
- [ ] Uncertain responses do not silently become accepted predictions.
- [ ] Two-item verification creates two unique rows.

## Out of Scope

- Training a learned fusion model
- Changing Layer 1 intervals with free-form VLM timestamps
- Live inference

## Handoff Contract

The task owner must provide:

- a pull request containing the implementation and tests;
- the resolved configuration used for the acceptance run;
- one machine-readable sample output or fixture;
- a short note listing assumptions, known limitations, and any interface changes;
- confirmation that no source videos, credentials, or model weights were committed.
