# task_7_medium: Dataset Validation, Agreement, Splits, and Versioning

> This task belongs to the pickup/putdown temporal action detection implementation.
> Read `docs/concepts.md`, the copied `manifest/labeling-guidelines.md`, and
> `PICKUP_PUTDOWN_IMPLEMENTATION_PLAN_CONCEPTS_ALIGNED.md` before starting.
> The reference case repository is read-only; implementation artifacts belong in the solution repository.

**Task ID:** `task_7`  
**Difficulty:** `medium`  
**Dependencies:** Tasks 2, 3, and 6. Task 5 is required to report proposal recall.  
**Parallel work:** Task 8 evaluator implementation.

## Objective

Turn raw annotations into a trustworthy, immutable dataset version with leakage-safe splits and quality reports.

## Inputs

- `clips.csv` / internal clip manifest
- `events.csv`
- Ignore intervals
- Annotator exports
- Session or recording-group metadata

## Deliverables

- Manifest validator
- Annotation agreement report
- Session-aware train/validation/test split
- Immutable dataset-version directory
- Event and negative preview gallery
- Proposal-recall report
- Dataset card with class and edge-case distributions

## Expected Files or Modules

- `src/pickup_putdown/annotation/validation.py`
- `src/pickup_putdown/annotation/agreement.py`
- `src/pickup_putdown/data/splits.py`
- `src/pickup_putdown/data/versioning.py`
- `tests/test_split_leakage.py`

## Implementation Steps

1. Validate clip references, interval bounds, event types, confidence values, duplicate IDs, and overlaps with ignore intervals.
2. Generate preview clips with context around every event and a representative set of hard negatives.
3. Double-label at least the configured fraction and report event-existence, type, boundary, and item-count disagreement.
4. Resolve disagreements before freezing test labels.
5. Group by session/customer sequence/recording day where possible; whole clip is the minimum grouping unit.
6. Never split derived frames, windows, candidates, or embeddings independently.
7. Freeze and hash the test split before threshold tuning.
8. Create an immutable dataset version including manifests, split definition, labeling-guideline version, and summary statistics.
9. Measure Stage B proposal recall on reviewed clips without allowing proposals to redefine ground truth.

## Acceptance Criteria

- [ ] No source clip or session group appears in more than one split.
- [ ] Every official event has a valid preview and source clip.
- [ ] Dataset version and split hash are reproducible.
- [ ] Test labels are frozen and not used for threshold selection.
- [ ] Proposal recall is reported separately from detector metrics.

## Out of Scope

- Model implementation
- Qwen prompting
- End-to-end CLI

## Handoff Contract

The task owner must provide:

- a pull request containing the implementation and tests;
- the resolved configuration used for the acceptance run;
- one machine-readable sample output or fixture;
- a short note listing assumptions, known limitations, and any interface changes;
- confirmation that no source videos, credentials, or model weights were committed.
