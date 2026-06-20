# task_10_hard: Track A State Classifiers, Repeating State Machine, and Event Boundaries

> This task belongs to the pickup/putdown temporal action detection implementation.
> Read `docs/concepts.md`, the copied `manifest/labeling-guidelines.md`, and
> `PICKUP_PUTDOWN_IMPLEMENTATION_PLAN_CONCEPTS_ALIGNED.md` before starting.
> The reference case repository is read-only; implementation artifacts belong in the solution repository.

**Task ID:** `task_10`  
**Difficulty:** `hard`  
**Dependencies:** Tasks 8 and 9.  
**Parallel work:** Task 12 Track B1.

## Objective

Implement the first complete non-VLM detector: repeated actor-specific state transitions that output zero, one, or multiple pickup/putdown events.

## Inputs

- Cached Track A embeddings and crop labels
- Actor/hand trajectories
- Candidate intervals
- Validation evaluator

## Deliverables

- Hand-state classifier
- Shelf-transition classifier
- Repeating temporal state machine
- Transfer/stabilization event-boundary estimation
- Confidence scoring
- Canonical Track A predictions
- Validation report and failure previews

## Expected Files or Modules

- `src/pickup_putdown/layer1/track_a/hand_state.py`
- `src/pickup_putdown/layer1/track_a/shelf_state.py`
- `src/pickup_putdown/layer1/track_a/state_machine.py`
- `src/pickup_putdown/layer1/track_a/inference.py`
- `configs/track_a.yaml`

## Implementation Steps

1. Train simple, interpretable classifiers such as logistic regression, small MLP, or gradient-boosted trees for hand carrying state and shelf transition.
2. Operate per `actor_id + hand_side + region_id` and preserve time order.
3. Implement a repeated state machine that may emit no event, one event, or multiple ordered events inside one candidate.
4. Detect pickup from a persistent shelf→hand transition and putdown from a persistent hand→shelf transition.
5. Treat touch-only, browsing, reaching, and visible restocking as background.
6. Estimate event start from the final purposeful transfer action and event end from stable held/resting state. Wrist region entry/exit is only a documented fallback.
7. Never merge adjacent predictions of different types. Same-type merge rules must be validation-configured.
8. Compute a reproducible score from classifier/state evidence and store diagnostic evidence internally.
9. Tune thresholds only on validation data and export exact canonical predictions.

## Acceptance Criteria

- [ ] Immediate pickup followed by putdown can produce two rows.
- [ ] Multiple actors are processed independently.
- [ ] Candidate boundaries are not blindly copied as event boundaries.
- [ ] Visible restocking does not produce putdown.
- [ ] Track A runs end-to-end on untrimmed clips after Stage A/B and is evaluated with Task 8.

## Out of Scope

- Multi-item expansion, which is Task 11
- VideoMAE models
- VLM verification

## Handoff Contract

The task owner must provide:

- a pull request containing the implementation and tests;
- the resolved configuration used for the acceptance run;
- one machine-readable sample output or fixture;
- a short note listing assumptions, known limitations, and any interface changes;
- confirmation that no source videos, credentials, or model weights were committed.
