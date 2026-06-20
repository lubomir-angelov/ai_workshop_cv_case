# task_14_hard: Layer 2 Standalone Qwen3.6-27B Event Detector

> This task belongs to the pickup/putdown temporal action detection implementation.
> Read `docs/concepts.md`, the copied `manifest/labeling-guidelines.md`, and
> `PICKUP_PUTDOWN_IMPLEMENTATION_PLAN_CONCEPTS_ALIGNED.md` before starting.
> The reference case repository is read-only; implementation artifacts belong in the solution repository.

**Task ID:** `task_14`  
**Difficulty:** `hard`  
**Dependencies:** Tasks 3, 7, and 8. It must not depend on Layer 1 predictions.  
**Parallel work:** Tasks 10–13.

## Objective

Implement an independently evaluable VLM detector that scans active-span windows and returns pickup/putdown intervals without using Layer 1 proposals.

## Inputs

- Active spans
- Source videos
- Qwen3.6-27B endpoint/runtime
- Concept definitions and prompt schema

## Deliverables

- Standalone window generator
- Timestamp/frame-overlay renderer
- Strict JSON response schema
- Qwen client with retries and audit logs
- Window-to-source timestamp conversion
- Overlapping-window duplicate merger
- Canonical Layer 2 predictions and performance report

## Expected Files or Modules

- `src/pickup_putdown/layer2/window_generator.py`
- `src/pickup_putdown/layer2/renderer.py`
- `src/pickup_putdown/layer2/prompts.py`
- `src/pickup_putdown/layer2/qwen_client.py`
- `src/pickup_putdown/layer2/merge_predictions.py`
- `configs/layer2_qwen.yaml`

## Implementation Steps

1. Generate fixed overlapping windows only inside Stage A active spans. Do not provide Layer 1 event types, intervals, or scores.
2. Render chronological video sections with clear relative frame numbers or timestamps.
3. Prompt with exact pickup, putdown, negative, restocking, occlusion, multiple-person, immediate-return, and two-item rules.
4. Require zero or more events in strict JSON, including type, relative interval, item count, visibility, and confidence.
5. Validate with Pydantic, retry invalid responses once, and preserve raw responses and prompt/model versions.
6. Convert relative times to source-clip times deterministically.
7. Merge duplicates from overlapping windows using type-aware temporal rules without collapsing two-item rows.
8. Record quantization, backend, sampling rate, frames/window, invalid-response rate, runtime, and peak VRAM.
9. Evaluate independently on the same held-out clips as Layer 1.

## Acceptance Criteria

- [ ] Layer 2 runs without reading any Layer 1 prediction file.
- [ ] Invalid JSON cannot silently enter predictions.
- [ ] Pickup and putdown remain separate when close in time.
- [ ] Source timestamps are reproducible from window metadata.
- [ ] Canonical predictions pass the shared evaluator.

## Out of Scope

- Fine-tuning Qwen
- Layer 1 verification/fusion
- Streaming

## Handoff Contract

The task owner must provide:

- a pull request containing the implementation and tests;
- the resolved configuration used for the acceptance run;
- one machine-readable sample output or fixture;
- a short note listing assumptions, known limitations, and any interface changes;
- confirmation that no source videos, credentials, or model weights were committed.
