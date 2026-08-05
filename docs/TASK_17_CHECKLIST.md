# Task 17 Completion Checklist

This checklist separates documentation that can be completed with Tasks 1–6
and 8 from evidence that depends on later tasks.

## Available now

- [x] Top-level README links reproducibility, privacy, and failure documentation.
- [x] Reproduction guide covers currently implemented ingestion, Tasks 3–5,
  annotation workflow, and Task 8 evaluator tests.
- [x] Data policy records that AWS source videos are pre-blurred and remain private.
- [x] Publication workflow requires manual review of selected derivatives.
- [x] Artifact ownership, location, Git policy, and retention rules are documented.
- [x] Failure taxonomy covers data, annotation, Layer 0, model, VLM, operations,
  and privacy risks.
- [x] Safe example configurations contain placeholders and no credentials.
- [x] Fictional machine-readable run metadata example is available.
- [x] Execute one selected Tasks 3–5 reproduction run and attach its resolved
  configuration and metadata as private acceptance evidence. Completed with
  run `20260625_task17_acceptance`; evidence remains under ignored
  `.local/task_runs/20260625_task17_acceptance/`.
- [x] Confirm repository documentation contains no customer imagery.
- [x] Confirm no source videos, credentials, or model weights are staged.
- [x] Replace tracked customer-derived annotation previews under
  `resources/annotations/` with generated synthetic fixtures.

## Pending Task 7

- [ ] Record immutable dataset version.
- [ ] Record frozen split version and hash.
- [ ] Document manifest-validation command and output.
- [ ] Document proposal-recall and dataset-quality reports.

## Pending Tasks 9–16

- [ ] Add exact Track A training and inference commands.
- [ ] Add exact Track B training and inference commands.
- [ ] Add standalone Layer 2 command.
- [ ] Record Qwen model, quantization, backend, GPU, peak VRAM, and runtime.
- [ ] Add evaluation report command and canonical output locations.
- [ ] Add final single-file and directory batch-inference commands.
- [ ] Record checkpoint SHA-256 hashes and threshold provenance.
- [ ] Add privacy-safe qualitative examples and final failure analysis.

## Final acceptance

- [ ] A new team member independently reproduces one selected run using
  documentation only.
- [x] Every generated artifact has documented owner, location, and retention rule.
- [x] Example configurations contain no credentials or private infrastructure values.
- [x] Repository documentation contains no published customer-media examples.
- [x] Limitations and partial results known at this stage are reported.
- [x] Checklist identifies resolved configuration, machine-readable sample
  output, assumptions, limitations, interface changes, and confirmation that
  no private media, credentials, or model weights are included.

## Current Task 17 evidence

- Resolved acceptance configuration and metadata:
  `.local/task_runs/20260625_task17_acceptance/` (private, ignored).
- Machine-readable fictional example: `samples/example_run_metadata.json`.
- Assumption: source footage supplied to project is pre-blurred but remains
  private.
- Limitation: final dataset, model, evaluation, and batch evidence depends on
  Tasks 7 and 9–16.
- Interface changes: documentation and safe example configurations only.
- Repository check: tracked annotation videos are generated synthetic fixtures;
  model weights are ignored; no credentials are included in Task 17 files.
