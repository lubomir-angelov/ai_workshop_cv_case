# task_16_hard: Batch CLI and End-to-End File Inference

> This task belongs to the pickup/putdown temporal action detection implementation.
> Read `docs/concepts.md`, the copied `manifest/labeling-guidelines.md`, and
> `PICKUP_PUTDOWN_IMPLEMENTATION_PLAN_CONCEPTS_ALIGNED.md` before starting.
> The reference case repository is read-only; implementation artifacts belong in the solution repository.

**Task ID:** `task_16`  
**Difficulty:** `hard`  
**Dependencies:** Task 1 plus completed components selected for the target demo. Full version depends on Tasks 2–15.  
**Parallel work:** Task 17 documentation and packaging.

## Objective

Expose reproducible commands for each stage and one batch inference command that accepts an MP4 file or directory.

## Inputs

- All selected pipeline components
- Resolved configuration
- Model checkpoints/endpoints

## Deliverables

- Typer-based CLI
- Stage-specific commands
- End-to-end `infer` command
- Machine-readable command summaries
- Structured output directory
- Failure handling and resumability
- Integration tests using tiny fixture videos

## Expected Files or Modules

- `src/pickup_putdown/cli.py`
- `src/pickup_putdown/pipeline.py`
- `tests/test_cli.py`
- `tests/test_pipeline_smoke.py`

## Implementation Steps

1. Implement commands: `index`, `triage`, `propose`, `annotate`, `validate-manifest`, Track A/B training and inference, Layer 2, verification, fusion, evaluation, and final `infer`.
2. Every command must resolve configuration, record Git/dataset/model versions, avoid silent overwrite, and exit non-zero on failure.
3. Implement early completion for no-person clips.
4. Allow component selection: Track A, B1, B2, standalone Layer 2, and optional Layer 3.
5. Write stage outputs using stable names and preserve intermediate audit artifacts.
6. Support one video path and recursive/non-recursive directory input according to configuration.
7. Implement resumability based on stage metadata and content hashes, not merely output-file existence.
8. Add a smoke-test fixture that exercises no-person and person-containing paths without requiring large models in CI.

## Acceptance Criteria

- [ ] `pickup-putdown infer --input clip.mp4` produces a valid output directory.
- [ ] Directory mode handles multiple files and reports per-file failures without losing successful outputs.
- [ ] No stage silently overwrites an incompatible prior artifact.
- [ ] Final canonical CSV passes schema validation.
- [ ] The pipeline requires no RTSP, Kafka, or streaming infrastructure.

## Out of Scope

- Production orchestration
- Kubernetes
- Continuous live camera processing

## Handoff Contract

The task owner must provide:

- a pull request containing the implementation and tests;
- the resolved configuration used for the acceptance run;
- one machine-readable sample output or fixture;
- a short note listing assumptions, known limitations, and any interface changes;
- confirmation that no source videos, credentials, or model weights were committed.
