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


## Working on a Task

Before starting, check `docs/tasks/task_index.md` and confirm that the task is not already assigned.

### 1. Announce the task in the team chat

Post a short message:

```text
I am starting task_5_stage_b_interaction_proposals_hard.

Branch:
feature/task-5-stage-b-interaction-proposals
```

When the task is ready for review, post:

```text
task_5 is ready for review.

Pull request: <PR link>
```

When blocked, post:

```text
task_5 is blocked by task_3 because the required track schema is not available yet.
```

### 2. Update the main branch

```bash
git switch main
git pull --ff-only
```

### 3. Create a feature branch

All tasks use feature branches.

Naming convention:

```text
feature/task-<number>-<short-description>
```

Example:

```bash
git switch -c feature/task-5-stage-b-interaction-proposals
```

### 4. Implement the assigned task

Follow the task file's:

* dependencies;
* inputs;
* deliverables;
* acceptance criteria;
* scope boundaries.

Do not change interfaces owned by another task without coordinating in the team chat.

Commit the changes:

```bash
git add .
git commit -m "feat(task-5): generate shelf interaction candidates"
```

### 5. Push the feature branch

```bash
git push -u origin feature/task-5-stage-b-interaction-proposals
```

### 6. Open and merge the pull request in the repository UI

Create the pull request from the feature branch into `main`.

Include:

```text
Task:
task_5_stage_b_interaction_proposals_hard

Implemented:
- ...
- ...

Validation:
- tests executed
- commands executed
- generated artifacts inspected

Dependencies or follow-up work:
- ...

Acceptance criteria:
- [x] ...
- [x] ...
```

Use the repository UI for review and merging. Do not merge the feature branch into `main` locally.

