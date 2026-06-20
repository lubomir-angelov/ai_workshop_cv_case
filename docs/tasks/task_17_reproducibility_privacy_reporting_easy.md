# task_17_easy: Reproducibility, Privacy, and Repository Documentation

> This task belongs to the pickup/putdown temporal action detection implementation.
> Read `docs/concepts.md`, the copied `manifest/labeling-guidelines.md`, and
> `PICKUP_PUTDOWN_IMPLEMENTATION_PLAN_CONCEPTS_ALIGNED.md` before starting.
> The reference case repository is read-only; implementation artifacts belong in the solution repository.

**Task ID:** `task_17`  
**Difficulty:** `easy`  
**Dependencies:** Task 1. Final examples require outputs from later tasks.  
**Parallel work:** All implementation tasks.

## Objective

Document how to reproduce, audit, and safely demonstrate the system without exposing source data or identities.

## Inputs

- Implementation plan
- Repository commands and configurations
- Example run metadata and evaluation outputs

## Deliverables

- Top-level README
- Data-handling and privacy policy
- Reproduction guide
- Model/dataset/run naming conventions
- Example configuration files without secrets
- Face-blurring utility or documented report workflow
- Final system limitations and known-failure taxonomy

## Expected Files or Modules

- `README.md`
- `docs/REPRODUCIBILITY.md`
- `docs/DATA_PRIVACY.md`
- `docs/FAILURE_MODES.md`
- `configs/*.example.yaml`
- `tools/blur_faces.py` or equivalent

## Implementation Steps

1. Document that the case repository is read-only and that code/data/artifacts live elsewhere.
2. Document storage endpoint configuration, secrets handling, and the prohibition on committing raw videos or credentials.
3. Document exact commands to reproduce dataset validation, each model, evaluation, and batch inference.
4. Document dataset version, split version, run metadata, checkpoint hashes, and threshold provenance.
5. Prohibit person identification and redistribution of source clips.
6. Ensure published examples have blurred faces or use synthetic illustrations.
7. Document model-size/hardware deviation for Qwen3.6-27B, including quantization and runtime constraints.
8. Document known limitations: occlusion, multiple items, simultaneous actors, restocking, clip boundaries, and rare fast actions.

## Acceptance Criteria

- [ ] A new team member can reproduce a selected run from the documentation.
- [ ] No example configuration contains credentials.
- [ ] No unblurred customer imagery appears in repository documentation.
- [ ] All generated artifacts have a documented owner, location, and retention rule.
- [ ] Limitations are reported honestly rather than hidden.

## Out of Scope

- Model implementation
- Annotation adjudication
- Infrastructure beyond local/Colab use

## Handoff Contract

The task owner must provide:

- a pull request containing the implementation and tests;
- the resolved configuration used for the acceptance run;
- one machine-readable sample output or fixture;
- a short note listing assumptions, known limitations, and any interface changes;
- confirmation that no source videos, credentials, or model weights were committed.
