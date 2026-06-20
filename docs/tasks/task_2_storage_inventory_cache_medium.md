# task_2_medium: Cloud Inventory, Video Metadata, and Bounded Cache

> This task belongs to the pickup/putdown temporal action detection implementation.
> Read `docs/concepts.md`, the copied `manifest/labeling-guidelines.md`, and
> `PICKUP_PUTDOWN_IMPLEMENTATION_PLAN_CONCEPTS_ALIGNED.md` before starting.
> The reference case repository is read-only; implementation artifacts belong in the solution repository.

**Task ID:** `task_2`  
**Difficulty:** `medium`  
**Dependencies:** Task 1 schemas and configuration.  
**Parallel work:** Tasks 3 and 4 can proceed using local fixture videos.

## Objective

Index the read-only video source without downloading the full dataset, probe media metadata, detect duplicates or decode failures, and provide a bounded local cache.

## Inputs

- Storage endpoint, region, bucket/prefix, and authentication mode
- Canonical clip schema
- A small set of test videos including one corrupt fixture

## Deliverables

- Bucket/object listing command
- `ffprobe` metadata extraction
- Stable `clip_id` generation
- Duplicate and decode-status fields
- Bounded cache with deterministic paths and eviction policy
- Initial `clips.parquet` plus canonical `clips.csv` exporter

## Expected Files or Modules

- `src/pickup_putdown/ingestion/index_bucket.py`
- `src/pickup_putdown/ingestion/video_probe.py`
- `src/pickup_putdown/ingestion/cache.py`
- `tests/test_video_probe.py`
- `configs/storage.yaml`

## Implementation Steps

1. Implement S3/S3-compatible listing with configurable `endpoint_url`, `region`, anonymous mode, and credentials from the environment.
2. Generate stable clip identifiers from immutable source attributes. Do not use local download paths as IDs.
3. Probe duration, average FPS, width, height, codec, and basic decode validity using `ffprobe`.
4. Record object size and ETag/checksum where available, and flag likely duplicates without deleting them.
5. Implement on-demand download to a private cache. Add size- or count-based eviction and a lock to avoid concurrent duplicate downloads.
6. Never download the complete bucket as part of indexing.
7. Emit a machine-readable summary: indexed count, failures, duplicate candidates, total source bytes, and local cache usage.

## Acceptance Criteria

- [ ] Indexing can run without materializing all source videos.
- [ ] A corrupt video is marked failed rather than crashing the entire run.
- [ ] Repeated indexing produces the same `clip_id` values.
- [ ] Cache retrieval is idempotent and concurrency-safe.
- [ ] Canonical `clips.csv` contains the case-required columns.

## Out of Scope

- Person detection
- Annotation
- Dataset splitting
- Training-window generation

## Handoff Contract

The task owner must provide:

- a pull request containing the implementation and tests;
- the resolved configuration used for the acceptance run;
- one machine-readable sample output or fixture;
- a short note listing assumptions, known limitations, and any interface changes;
- confirmation that no source videos, credentials, or model weights were committed.
