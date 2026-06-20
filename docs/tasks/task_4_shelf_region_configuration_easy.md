# task_4_easy: Shelf and Surface Region Configuration

> This task belongs to the pickup/putdown temporal action detection implementation.
> Read `docs/concepts.md`, the copied `manifest/labeling-guidelines.md`, and
> `PICKUP_PUTDOWN_IMPLEMENTATION_PLAN_CONCEPTS_ALIGNED.md` before starting.
> The reference case repository is read-only; implementation artifacts belong in the solution repository.

**Task ID:** `task_4`  
**Difficulty:** `easy`  
**Dependencies:** Task 1 only. A representative frame from the camera is required.  
**Parallel work:** Tasks 2, 3, and 6.

## Objective

Create a version-controlled configuration for fixed shelf and placement surfaces and provide visual validation of the polygons.

## Inputs

- Representative full-resolution frame for each camera view
- Camera identifier and source resolution

## Deliverables

- `configs/shelves.yaml`
- Polygon schema and loader
- Expanded interaction-region generation
- Overlay image/video utility
- Validation for coordinates and camera resolution

## Expected Files or Modules

- `src/pickup_putdown/perception/shelf_regions.py`
- `configs/shelves.yaml`
- `tests/test_shelf_regions.py`
- `tools/draw_shelf_regions.py`

## Implementation Steps

1. Mark every shelf or surface where a valid pickup or putdown can occur.
2. Assign stable `region_id` values and region types (`shelf`, `surface`, or project-approved equivalent).
3. Store polygons in source-frame pixel coordinates together with expected width and height.
4. Generate expanded interaction polygons using configurable pixel or normalized margins.
5. Implement scaling or fail-fast behavior when inference resolution differs from configuration resolution.
6. Produce an overlay artifact that clearly distinguishes exact regions from expanded interaction regions.
7. Review polygons with at least one annotator before freezing version 1.

## Acceptance Criteria

- [ ] Every configured polygon is inside the frame and has at least three non-collinear points.
- [ ] The visual overlay matches the actual shelf/surface boundaries.
- [ ] Expanded polygons do not silently exceed image bounds.
- [ ] Configuration changes are version-controlled and traceable.

## Out of Scope

- Pose inference
- Event labels
- Object segmentation

## Handoff Contract

The task owner must provide:

- a pull request containing the implementation and tests;
- the resolved configuration used for the acceptance run;
- one machine-readable sample output or fixture;
- a short note listing assumptions, known limitations, and any interface changes;
- confirmation that no source videos, credentials, or model weights were committed.
