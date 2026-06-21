# Pickup/Putdown Implementation Task Index

These tasks decompose `PICKUP_PUTDOWN_IMPLEMENTATION_PLAN_CONCEPTS_ALIGNED.md` into work packages that can each be owned end-to-end by one team member. Difficulty reflects technical implementation complexity, not business importance.

Files ending in `_optional.md` are explicitly non-critical-path work. Complete the mandatory tasks first.

## Assignment Rules

- One named owner per task. Other members may review, but the owner is responsible for implementation, tests, acceptance evidence, and handoff.
- Complete mandatory tasks before assigning optional tasks, unless an optional task has a dedicated owner who cannot accelerate the critical path.
- Do not start a task before its listed dependencies are available, except when using explicit fixtures or mocks.
- Interfaces defined by Task 1 are shared contracts. Coordinate schema changes through a dedicated pull request.
- All thresholds are tuned on validation data only.
- The test split remains frozen.
- Stage A/B candidates are never treated as ground-truth events or canonical predictions.

## Mandatory Dependency Overview

```text
task_1 ─┬─ task_2 ─ task_3 ─┬─ task_5 ─┬─ task_9 ─ task_10
        │                   │          └─ task_12
        ├─ task_4 ──────────┘
        ├─ task_6 ─ task_7 ────────────┬─ task_8
        │                              └─ task_14
        ├─ task_8  (starts with fixtures and serves all models)
        ├─ task_17 (runs throughout)
        └─ task_16 (integrates completed mandatory components)
```

## Optional Extensions

```text
task_9 + task_10 ───────────── task_11_optional
task_12 ────────────────────── task_13_optional
task_14 + Layer 1 predictions ─ task_15_optional
task_13_optional ───────────── task_18_optional
```

## Mandatory Tasks

| ID | Difficulty | Task | Primary output | Dependencies |
|---|---|---|---|---|
~~| `task_1` | `easy` | [Repository Bootstrap, Configuration, and Core Schemas](task_1_repository_bootstrap_easy.md) | Installable `pickup_putdown` Python package | None |~~
~~| `task_2` | `medium` | [Cloud Inventory, Video Metadata, and Bounded Cache](task_2_storage_inventory_cache_medium.md) | Bucket/object listing command | Task 1 schemas and configuration. |~~
| `task_3` | `medium` | [Layer 0A Person Triage and Active-Span Extraction](task_3_stage_a_person_triage_medium.md) | Direct video-file person tracking | Tasks 1 and 2. |
| `task_4` | `easy` | [Shelf and Surface Region Configuration](task_4_shelf_region_configuration_easy.md) | `configs/shelves.yaml` | Task 1 only. A representative frame from the camera is required. |
| `task_5` | `hard` | [Layer 0B Pose Tracking and Actor-Specific Interaction Proposals](task_5_stage_b_interaction_proposals_hard.md) | Higher-rate pose inference over active spans | Tasks 3 and 4. |
| `task_6` | `medium` | [Annotation Workflow and Canonical Import/Export](task_6_annotation_workflow_medium.md) | Configured annotation tool or purpose-built minimal UI | Task 1. Task 5 improves candidate-assisted annotation but is not required to start. |
| `task_7` | `medium` | [Dataset Validation, Agreement, Splits, and Versioning](task_7_dataset_quality_splits_medium.md) | Manifest validator | Tasks 2, 3, and 6. Task 5 is required to report proposal recall. |
| `task_8` | `hard` | [Shared Two-Pass Temporal Evaluator and Reports](task_8_shared_evaluator_hard.md) | Class-aware one-to-one matcher | Task 1 schemas. Synthetic fixtures are sufficient to begin; Task 7 provides real data. |
| `task_9` | `medium` | [Track A Crop Extraction and Appearance Features](task_9_track_a_features_medium.md) | Pre/contact/post timestamp selection | Tasks 5 and 7. |
| `task_10` | `hard` | [Track A State Classifiers, Repeating State Machine, and Event Boundaries](task_10_track_a_state_machine_hard.md) | Hand-state classifier | Tasks 8 and 9. |
| `task_12` | `hard` | [Track B1 Actor-Conditioned VideoMAE Window Classifier](task_12_track_b1_videomae_classifier_hard.md) | Actor-conditioned crop/window dataset | Tasks 5, 7, and 8. |
| `task_14` | `hard` | [Layer 2 Standalone Qwen3.6-27B Event Detector](task_14_layer2_qwen_standalone_hard.md) | Standalone window generator | Tasks 3, 7, and 8. It must not depend on Layer 1 predictions. |
| `task_16` | `hard` | [Batch CLI and End-to-End File Inference](task_16_batch_cli_integration_hard.md) | Typer-based CLI | Task 1 plus completed mandatory model components. |
| `task_17` | `easy` | [Reproducibility, Privacy, and Repository Documentation](task_17_reproducibility_privacy_reporting_easy.md) | Top-level README | Task 1. Final examples require outputs from later tasks. |

## Optional Tasks

| ID | Difficulty | Task | Why optional | Dependencies |
|---|---|---|---|---|
| `task_11` | `hard` | [Shared Non-VLM Item-Count Estimator and Row Expansion](task_11_item_count_estimator_hard_optional.md) | Train only when the dataset contains enough two-item examples. Canonical duplicate-row export must still be supported. | Tasks 7, 8, and 9; integrate after Task 10 or Track B predictions exist. |
| `task_13` | `hard` | [Track B2 Cached VideoMAE Features and Temporal Convolutional Detector](task_13_track_b2_videomae_tcn_hard_optional.md) | Stronger interval-localization model after the B1 baseline is stable. | Tasks 7, 8, and 12. |
| `task_15` | `medium` | [Layer 3 Qwen Verification and Deterministic Fusion](task_15_layer3_qwen_verifier_fusion_medium_optional.md) | Layer 3 fusion is a stretch extension after standalone Layer 1 and Layer 2 systems work. | Tasks 8, 10 or 12/13, and Task 14 client/schema components. |
| `task_18` | `hard` | [ActionFormer Temporal Detector Benchmark](task_18_actionformer_hard_optional.md) | Research-grade comparison that must not delay working baselines. | Tasks 7, 8, and optional Task 13. |

## Mandatory Critical Path

1. `task_1` repository and schemas.
2. `task_2` inventory/cache and `task_4` shelf regions in parallel.
3. `task_3` person triage.
4. `task_5` interaction proposals and `task_6` annotation workflow.
5. `task_7` dataset freeze and `task_8` evaluator.
6. `task_9` + `task_10` Track A baseline.
7. `task_12` Track B1 learned baseline.
8. `task_14` standalone Layer 2 detector.
9. `task_16` batch integration.
10. `task_17` final reproducibility, privacy, and reporting documentation.

## Optional Work Order

1. `task_11` when two-item examples justify a learned count estimator.
2. `task_13` when Track B1 is stable and better interval localization is needed.
3. `task_15` when standalone Layer 1 and Layer 2 outputs are reliable.
4. `task_18` only after Task 13 is stable and there is time for a fair benchmark.
