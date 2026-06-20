# Task Files

Copy this directory into the solution repository, for example as `docs/tasks/`.

Start with [`task_index.md`](task_index.md). Each task is intended to be owned by one team member and contains dependencies, exact deliverables, implementation steps, acceptance criteria, scope boundaries, and a handoff contract.

## Filename Convention

- `task_<N>_<name>_easy.md`: mandatory, mostly configuration, documentation, or bounded utility work;
- `task_<N>_<name>_medium.md`: mandatory, multi-module engineering with established libraries;
- `task_<N>_<name>_hard.md`: mandatory, model integration, temporal logic, evaluation algorithms, or cross-stage inference;
- `task_<N>_<name>_<difficulty>_optional.md`: non-critical-path extension; assign only after mandatory dependencies and baseline deliverables are secure.

## Mandatory First

Use the **Mandatory Tasks** section of `task_index.md` for initial assignment. Files ending in `_optional.md` must not delay:

1. a trustworthy Layer 0 dataset;
2. a working Track A baseline;
3. a working Track B1 baseline;
4. a standalone Layer 2 result;
5. common evaluation and batch inference.
