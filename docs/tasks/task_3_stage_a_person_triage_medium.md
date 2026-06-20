# task_3_medium: Layer 0A Person Triage and Active-Span Extraction

> This task belongs to the pickup/putdown temporal action detection implementation.
> Read `docs/concepts.md`, the copied `manifest/labeling-guidelines.md`, and
> `PICKUP_PUTDOWN_IMPLEMENTATION_PLAN_CONCEPTS_ALIGNED.md` before starting.
> The reference case repository is read-only; implementation artifacts belong in the solution repository.

**Task ID:** `task_3`  
**Difficulty:** `medium`  
**Dependencies:** Tasks 1 and 2.  
**Parallel work:** Task 4 can run independently. Task 6 can prepare the annotation tool using manual clips.

## Objective

Run low-rate person detection and tracking directly on video files, determine whether people are present, and derive person-active spans for downstream annotation and inference.

## Inputs

- Cached video path or local MP4
- Clip metadata
- Triage configuration

## Deliverables

- Direct video-file person tracking
- Timestamped person tracklets
- Active-span derivation
- `tracks_person.parquet` and `active_spans.parquet`
- Preview renderer with track IDs and timestamps
- Triage-quality sampling report

## Expected Files or Modules

- `src/pickup_putdown/perception/person_tracker.py`
- `src/pickup_putdown/perception/active_spans.py`
- `src/pickup_putdown/perception/previews.py`
- `configs/triage.yaml`
- `tests/test_active_spans.py`

## Implementation Steps

1. Use a small person detector with ByteTrack or BoT-SORT. The library may receive the MP4 path directly; do not save individual frames.
2. Calculate timestamps from source FPS and decoded frame index, accounting for the configured stride.
3. Start with 1 FPS; compare 2 FPS on short clips. Track acceptance must use timestamps, not frame count alone.
4. Reject one-frame artifacts by requiring a stable track with configurable confidence and visible duration.
5. Derive one or more active spans by merging nearby observations while retaining separate long gaps.
6. Populate `n_person_tracks`, `has_person`, `active_start_s`, `active_end_s`, and internal active spans.
7. Retain no-person clips in the manifest, but exclude them from event annotation and Layer 1 training.
8. Create preview videos for manual QA and sample at least 5–10% of automatic no-person decisions.

## Acceptance Criteria

- [ ] A video path can be passed directly to the command.
- [ ] Short valid person appearances are not lost because of an inconsistent frame-count rule.
- [ ] No-person clips produce zero active spans and remain in the manifest.
- [ ] Active spans remain within source duration.
- [ ] The output is deterministic for fixed model/configuration.

## Out of Scope

- Pose keypoints
- Shelf interaction proposals
- Pickup/putdown classification

## Handoff Contract

The task owner must provide:

- a pull request containing the implementation and tests;
- the resolved configuration used for the acceptance run;
- one machine-readable sample output or fixture;
- a short note listing assumptions, known limitations, and any interface changes;
- confirmation that no source videos, credentials, or model weights were committed.
