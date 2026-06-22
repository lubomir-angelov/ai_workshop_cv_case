"""Perception modules for person detection, tracking, pose, and interaction proposals."""

from pickup_putdown.perception.active_spans import compute_clip_summary, derive_active_spans
from pickup_putdown.perception.candidate_previews import (
    CandidateOverlayConfig,
    render_candidate_preview,
)
from pickup_putdown.perception.person_tracker import PersonObservation, PersonTracker, TrackSummary
from pickup_putdown.perception.pose_tracker import PoseTracker
from pickup_putdown.perception.previews import OverlayConfig, draw_overlay, render_triage_preview
from pickup_putdown.perception.proposals import (
    RawInteraction,
    associate_poses_with_actors,
    compute_proposal_recall,
    detect_raw_interactions,
    generate_candidates,
)

__all__ = [
    "CandidateOverlayConfig",
    "RawInteraction",
    "compute_clip_summary",
    "compute_proposal_recall",
    "derive_active_spans",
    "associate_poses_with_actors",
    "detect_raw_interactions",
    "draw_overlay",
    "generate_candidates",
    "OverlayConfig",
    "PersonTracker",
    "PersonObservation",
    "PoseTracker",
    "TrackSummary",
    "render_candidate_preview",
    "render_triage_preview",
]
