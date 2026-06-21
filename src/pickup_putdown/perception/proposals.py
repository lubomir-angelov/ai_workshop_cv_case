"""Actor association, region measurements, raw interactions, and candidate generation.

This module ties together:
- pose observations from :mod:`pose_tracker`
- actor tracks from Layer 0A (person observations)
- shelf/surface regions from :mod:`shelf_regions`

It produces:
- actor-assigned pose observations
- raw interactions (wrist inside expanded region for min duration)
- merged candidate intervals
- a proposal-recall API for measuring ground-truth event coverage
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass

from pickup_putdown.common.exceptions import ConfigError, ValidationError
from pickup_putdown.common.schemas import (
    Candidate,
    PersonObservation,
    PoseObservation,
    ProposalRecallResult,
)
from pickup_putdown.config import (
    ActorAssociationConfig,
    ProposalsConfig,
    RegionMeasurementConfig,
)
from pickup_putdown.perception.shelf_regions import (
    CameraShelfConfig,
    Polygon,
    get_expanded_regions,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RawInteraction:
    """A single wrist-inside-region observation meeting minimum duration."""

    clip_id: str
    actor_id: str
    hand_side: str
    region_id: str
    start_s: float
    end_s: float
    n_observations: int = 0
    mean_wrist_confidence: float = 0.0
    mean_distance: float = 0.0


# ---------------------------------------------------------------------------
# Actor association
# ---------------------------------------------------------------------------


def associate_poses_with_actors(
    pose_observations: list[PoseObservation],
    person_observations: list[PersonObservation],
    actor_cfg: ActorAssociationConfig,
) -> list[PoseObservation]:
    """Associate each pose detection with the best-matching actor track.

    Uses IoU-based spatial matching between the person bounding box in the
    pose observation and the closest person observation for the same clip.
    Unmatched detections are left with their original actor_id (untracked).

    Parameters
    ----------
    pose_observations : list[PoseObservation]
        Raw pose observations from PoseTracker.run().
    person_observations : list[PersonObservation]
        Person track observations from Layer 0A.
    actor_cfg : ActorAssociationConfig
        Matching configuration.

    Returns
    -------
    list[PoseObservation]
        Pose observations with updated actor_id and pose_association_confidence.
    """
    # Build a lookup: clip_id -> list of person observations sorted by timestamp
    person_by_clip: dict[str, list[PersonObservation]] = defaultdict(list)
    for po in person_observations:
        person_by_clip[po.clip_id].append(po)

    for clip_id in person_by_clip:
        person_by_clip[clip_id].sort(key=lambda o: o.timestamp_s)

    # Group pose observations by clip
    poses_by_clip: dict[str, list[PoseObservation]] = defaultdict(list)
    for po in pose_observations:
        poses_by_clip[po.clip_id].append(po)

    for clip_id, poses in poses_by_clip.items():
        person_obs = person_by_clip.get(clip_id, [])
        for pose in poses:
            best_match = _find_best_actor_match(pose, person_obs, actor_cfg)
            if best_match is not None:
                pose.actor_id = best_match.person_track_id
                pose.pose_association_confidence = best_match.confidence
                pose.person_bbox_x1 = best_match.bbox_x1
                pose.person_bbox_y1 = best_match.bbox_y1
                pose.person_bbox_x2 = best_match.bbox_x2
                pose.person_bbox_y2 = best_match.bbox_y2

    return pose_observations


def _find_best_actor_match(
    pose: PoseObservation,
    person_obs: list[PersonObservation],
    actor_cfg: ActorAssociationConfig,
) -> PersonObservation | None:
    """Find the best-matching person observation for a pose detection.

    Uses bounding-box IoU. Returns None when no match exceeds the threshold.
    """
    if not person_obs:
        return None

    px1 = pose.person_bbox_x1 or 0.0
    py1 = pose.person_bbox_y1 or 0.0
    px2 = pose.person_bbox_x2 or 0.0
    py2 = pose.person_bbox_y2 or 0.0

    best_obs: PersonObservation | None = None
    best_iou = 0.0

    for po in person_obs:
        # Temporal window: only consider person observations within max_gap_s
        if abs(po.timestamp_s - pose.timestamp_s) > actor_cfg.max_gap_s:
            continue

        iou = _compute_bbox_iou(px1, py1, px2, py2, po.bbox_x1, po.bbox_y1, po.bbox_x2, po.bbox_y2)
        if iou > best_iou:
            best_iou = iou
            best_obs = po

    if best_obs is not None and best_iou >= actor_cfg.match_iou_threshold:
        return best_obs

    return None


def _compute_bbox_iou(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
    x4: float,
    y4: float,
) -> float:
    """Compute IoU between two axis-aligned bounding boxes."""
    # Intersection
    ix1 = max(x1, x3)
    iy1 = max(y1, y3)
    ix2 = min(x2, x4)
    iy2 = min(y2, y4)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    area1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area2 = max(0.0, x4 - x3) * max(0.0, y4 - y3)
    union = area1 + area2 - inter_area

    return inter_area / max(union, 1e-9)


# ---------------------------------------------------------------------------
# Region-based measurements
# ---------------------------------------------------------------------------


@dataclass
class _RegionMeasurement:
    """Measurements for one wrist observation against one region."""

    wrist_x: float
    wrist_y: float
    wrist_confidence: float
    distance: float
    inside_region: bool
    inside_expanded: bool
    entry_event: bool = False
    exit_event: bool = False
    dwell_duration_s: float = 0.0
    speed: float = 0.0
    velocity_reversal: bool = False


def compute_region_measurements(
    pose_observations: list[PoseObservation],
    camera_config: CameraShelfConfig,
    region_cfg: RegionMeasurementConfig,
) -> dict[str, list[_RegionMeasurement]]:
    """Compute region measurements for each (actor, hand_side) group.

    Parameters
    ----------
    pose_observations : list[PoseObservation]
        Actor-assigned pose observations.
    camera_config : CameraShelfConfig
        Shelf configuration for the camera.
    region_cfg : RegionMeasurementConfig
        Region measurement thresholds.

    Returns
    -------
    dict[str, list[_RegionMeasurement]]
        Keyed by ``"{clip_id}:{actor_id}:{hand_side}:{region_id}"``.
    """
    expanded = get_expanded_regions(camera_config)

    # Override margin if configured
    if region_cfg.expanded_margin_override is not None:
        original_expansion = camera_config.expansion
        camera_config.expansion.value = region_cfg.expanded_margin_override
        expanded = get_expanded_regions(camera_config)
        camera_config.expansion = original_expansion

    # Group pose observations
    groups: dict[str, list[PoseObservation]] = defaultdict(list)
    for po in pose_observations:
        key = f"{po.clip_id}:{po.actor_id}:{po.hand_side}"
        groups[key].append(po)

    result: dict[str, list[_RegionMeasurement]] = {}

    for _group_key, poses in groups.items():
        poses.sort(key=lambda p: p.timestamp_s)

        for region_id, orig_pts in expanded.items():
            expanded_pts = expanded[region_id]
            measurements: list[_RegionMeasurement] = []

            prev_wrist_x: float | None = None
            prev_wrist_y: float | None = None
            prev_inside: bool | None = None
            entry_time: float | None = None
            dwell = 0.0

            for i, pose in enumerate(poses):
                dist = _point_to_polygon_distance(pose.wrist_x, pose.wrist_y, orig_pts)
                inside = _point_in_polygon(pose.wrist_x, pose.wrist_y, orig_pts)
                inside_exp = _point_in_polygon(pose.wrist_x, pose.wrist_y, expanded_pts)

                entry = False
                exit_ev = False
                if prev_inside is not None:
                    if not prev_inside and inside:
                        entry = True
                        entry_time = pose.timestamp_s
                    elif prev_inside and not inside:
                        exit_ev = True
                        if entry_time is not None:
                            dwell += pose.timestamp_s - entry_time
                            entry_time = None
                elif inside:
                    entry = True
                    entry_time = pose.timestamp_s

                # Speed
                speed = 0.0
                if prev_wrist_x is not None:
                    dt = max(pose.timestamp_s - poses[i - 1].timestamp_s, 1e-9)
                    dx = pose.wrist_x - prev_wrist_x
                    dy = pose.wrist_y - prev_wrist_y
                    speed = (dx**2 + dy**2) ** 0.5 / dt

                # Velocity reversal
                reversal = False
                if i >= 2:
                    prev_dx = poses[i - 1].wrist_x - poses[i - 2].wrist_x
                    prev_dy = poses[i - 1].wrist_y - poses[i - 2].wrist_y
                    curr_dx = pose.wrist_x - poses[i - 1].wrist_x
                    curr_dy = pose.wrist_y - poses[i - 1].wrist_y
                    prev_dot = prev_dx * curr_dx + prev_dy * curr_dy
                    prev_mag = max((prev_dx**2 + prev_dy**2) ** 0.5, 1e-9)
                    curr_mag = max((curr_dx**2 + curr_dy**2) ** 0.5, 1e-9)
                    cos_angle = prev_dot / (prev_mag * curr_mag)
                    reversal = cos_angle < -region_cfg.reversal_threshold

                    measurements.append(
                        _RegionMeasurement(
                            wrist_x=pose.wrist_x,
                            wrist_y=pose.wrist_y,
                            wrist_confidence=pose.wrist_confidence,
                            distance=dist,
                            inside_region=inside,
                            inside_expanded=inside_exp,
                            entry_event=entry,
                            exit_event=exit_ev,
                            dwell_duration_s=dwell,
                            speed=speed,
                            velocity_reversal=reversal,
                        )
                    )

                prev_wrist_x = pose.wrist_x
                prev_wrist_y = pose.wrist_y
                prev_inside = inside

            # Close any open dwell
            if entry_time is not None:
                last_ts = poses[-1].timestamp_s if poses else entry_time
                dwell += last_ts - entry_time

            mkey = f"{_group_key}:{region_id}"
            result[mkey] = measurements

    return result


# ---------------------------------------------------------------------------
# Raw interaction detection
# ---------------------------------------------------------------------------


def detect_raw_interactions(
    pose_observations: list[PoseObservation],
    camera_config: CameraShelfConfig,
    proposals_cfg: ProposalsConfig,
    region_cfg: RegionMeasurementConfig,
) -> list[RawInteraction]:
    """Detect raw interactions where a confident wrist stays inside an expanded region.

    A raw interaction starts when a wrist with confidence >= minimum_wrist_confidence
    is inside the expanded region and ends when it leaves or confidence drops below
    the threshold for longer than gap_tolerance_s.

    Parameters
    ----------
    pose_observations : list[PoseObservation]
        Actor-assigned pose observations.
    camera_config : CameraShelfConfig
        Shelf configuration for the camera.
    proposals_cfg : ProposalsConfig
        Proposal generation thresholds.
    region_cfg : RegionMeasurementConfig
        Region measurement thresholds.

    Returns
    -------
    list[RawInteraction]
        Raw interactions meeting the minimum duration requirement.
    """
    expanded = get_expanded_regions(camera_config)
    min_conf = proposals_cfg.minimum_wrist_confidence
    min_dur = proposals_cfg.minimum_interaction_duration_s

    # Group by (clip_id, actor_id, hand_side, region_id) — only include
    # poses whose wrist falls inside the expanded polygon for that region.
    groups: dict[str, list[PoseObservation]] = defaultdict(list)
    for po in pose_observations:
        for region_id, exp_pts in expanded.items():
            if _point_in_polygon(po.wrist_x, po.wrist_y, exp_pts):
                key = f"{po.clip_id}:{po.actor_id}:{po.hand_side}:{region_id}"
                groups[key].append(po)

    interactions: list[RawInteraction] = []

    for group_key, poses in groups.items():
        in_interaction = False
        start_s: float | None = None
        obs_count = 0
        conf_sum = 0.0
        dist_sum = 0.0
        clip_id = ""
        actor_id = ""
        hand_side = ""
        region_id = ""

        poses.sort(key=lambda p: p.timestamp_s)
        parts = group_key.split(":")
        if len(parts) != 4:
            continue
        clip_id, actor_id, hand_side, region_id = parts

        for pose in poses:
            inside_exp = _point_in_polygon(pose.wrist_x, pose.wrist_y, expanded[region_id])
            is_confident = pose.wrist_confidence >= min_conf

            if inside_exp and is_confident:
                if not in_interaction:
                    in_interaction = True
                    start_s = pose.timestamp_s
                    obs_count = 0
                    conf_sum = 0.0
                    dist_sum = 0.0
                obs_count += 1
                conf_sum += pose.wrist_confidence
                dist_sum += _point_to_polygon_distance(
                    pose.wrist_x,
                    pose.wrist_y,
                    _original_polygon_from_expanded(
                        pose.wrist_x,
                        pose.wrist_y,
                        expanded[region_id],
                        camera_config.source_width,
                        camera_config.source_height,
                    ),
                )
            else:
                if in_interaction and start_s is not None:
                    end_s = pose.timestamp_s
                    dur = end_s - start_s
                    if dur >= min_dur:
                        interactions.append(
                            RawInteraction(
                                clip_id=clip_id,
                                actor_id=actor_id,
                                hand_side=hand_side,
                                region_id=region_id,
                                start_s=start_s,
                                end_s=end_s,
                                n_observations=obs_count,
                                mean_wrist_confidence=conf_sum / max(obs_count, 1),
                                mean_distance=dist_sum / max(obs_count, 1),
                            )
                        )
                in_interaction = False
                start_s = None
                obs_count = 0
                conf_sum = 0.0
                dist_sum = 0.0

        # Flush any interaction still open at end of poses for this group
        if in_interaction and start_s is not None:
            last_ts = poses[-1].timestamp_s
            end_s = last_ts
            dur = end_s - start_s
            if dur >= min_dur:
                interactions.append(
                    RawInteraction(
                        clip_id=clip_id,
                        actor_id=actor_id,
                        hand_side=hand_side,
                        region_id=region_id,
                        start_s=start_s,
                        end_s=end_s,
                        n_observations=obs_count,
                        mean_wrist_confidence=conf_sum / max(obs_count, 1),
                        mean_distance=dist_sum / max(obs_count, 1),
                    )
                )

    return interactions


def _original_polygon_from_expanded(
    wx: float,
    wy: float,
    expanded_pts: Polygon,
    img_w: int,
    img_h: int,
) -> Polygon:
    """Approximate the original polygon from an expanded one (reverse expansion).

    This is a best-effort reconstruction used only for distance calculation.
    """
    return [(p[0], p[1]) for p in expanded_pts]  # Simplified: use expanded as proxy


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def generate_candidates(
    raw_interactions: list[RawInteraction],
    clip_durations: dict[str, float],
    proposals_cfg: ProposalsConfig,
) -> list[Candidate]:
    """Merge raw interactions and add context to produce candidate intervals.

    Merging rules:
    - Only merge raw interactions with the same clip_id, actor_id, hand_side, region_id.
    - Merge when the gap between consecutive raw interactions <= merge_gap_s.
    - Add pre/post context to create padded candidate intervals.
    - Clamp to [0, clip_duration].
    - Never merge across actors, hands, regions, or clips.

    Parameters
    ----------
    raw_interactions : list[RawInteraction]
        Raw interactions from detect_raw_interactions().
    clip_durations : dict[str, float]
        Mapping of clip_id -> source duration in seconds.
    proposals_cfg : ProposalsConfig
        Proposal generation configuration.

    Returns
    -------
    list[Candidate]
        Deterministically ordered candidate intervals.
    """
    # Group by (clip_id, actor_id, hand_side, region_id)
    groups: dict[str, list[RawInteraction]] = defaultdict(list)
    for ri in raw_interactions:
        key = f"{ri.clip_id}:{ri.actor_id}:{ri.hand_side}:{ri.region_id}"
        groups[key].append(ri)

    candidates: list[Candidate] = []

    for _group_key, interactions in groups.items():
        interactions.sort(key=lambda r: r.start_s)

        # Merge interactions within merge_gap_s
        merged_groups: list[list[RawInteraction]] = []
        current_group: list[RawInteraction] = [interactions[0]]

        for ri in interactions[1:]:
            prev = current_group[-1]
            gap = ri.start_s - prev.end_s
            if gap <= proposals_cfg.merge_gap_s:
                current_group.append(ri)
            else:
                merged_groups.append(current_group)
                current_group = [ri]
        merged_groups.append(current_group)

        for merged in merged_groups:
            clip_id = merged[0].clip_id
            actor_id = merged[0].actor_id
            hand_side = merged[0].hand_side
            region_id = merged[0].region_id

            raw_start = min(r.start_s for r in merged)
            raw_end = max(r.end_s for r in merged)

            clip_dur = clip_durations.get(clip_id, raw_end + 1.0)

            # Add context
            padded_start = max(0.0, raw_start - proposals_cfg.context_before_s)
            padded_end = min(clip_dur, raw_end + proposals_cfg.context_after_s)

            # Clamp
            padded_start = max(0.0, min(padded_start, clip_dur))
            padded_end = max(0.0, min(padded_end, clip_dur))

            # Skip degenerate candidates
            if padded_end <= padded_start:
                continue

            # Cap maximum duration
            if (padded_end - padded_start) > proposals_cfg.maximum_candidate_duration_s:
                # Trim from center
                mid = (padded_start + padded_end) / 2
                half = proposals_cfg.maximum_candidate_duration_s / 2
                padded_start = max(0.0, mid - half)
                padded_end = min(clip_dur, mid + half)

            # Compute signal summaries
            min_dist = min(r.mean_distance for r in merged) if merged else 0.0
            max_conf = max(r.mean_wrist_confidence for r in merged) if merged else 0.0
            total_dwell = sum(r.end_s - r.start_s for r in merged)

            # Deterministic candidate ID
            id_parts = [
                clip_id,
                actor_id,
                hand_side,
                region_id,
                f"{raw_start:.3f}",
                f"{raw_end:.3f}",
            ]
            id_hash = hashlib.sha256(":".join(id_parts).encode()).hexdigest()[:12]
            candidate_id = f"cand_{id_hash}"

            # Config fingerprint
            cfg_str = (
                f"fps={proposals_cfg.target_fps};"
                f"min_conf={proposals_cfg.minimum_wrist_confidence};"
                f"min_dur={proposals_cfg.minimum_interaction_duration_s};"
                f"merge_gap={proposals_cfg.merge_gap_s};"
                f"ctx_before={proposals_cfg.context_before_s};"
                f"ctx_after={proposals_cfg.context_after_s};"
                f"max_dur={proposals_cfg.maximum_candidate_duration_s}"
            )
            cfg_fp = hashlib.sha256(cfg_str.encode()).hexdigest()[:8]

            candidate = Candidate(
                candidate_id=candidate_id,
                clip_id=clip_id,
                actor_id=actor_id,
                hand_side=hand_side,
                region_id=region_id,
                raw_start_s=round(raw_start, 4),
                raw_end_s=round(raw_end, 4),
                window_start_s=round(padded_start, 4),
                window_end_s=round(padded_end, 4),
                n_raw_interactions=len(merged),
                min_region_distance=round(min_dist, 4),
                max_wrist_confidence=round(max_conf, 4),
                total_dwell_duration_s=round(total_dwell, 4),
                config_fingerprint=cfg_fp,
                proposal_reason="wrist_in_expanded_region",
                proposal_score=max_conf,
                review_status="pending",
            )
            candidates.append(candidate)

    # Deterministic ordering
    candidates.sort(
        key=lambda c: (c.clip_id, c.actor_id, c.hand_side or "", c.region_id or "", c.raw_start_s)
    )
    return candidates


# ---------------------------------------------------------------------------
# Proposal recall API
# ---------------------------------------------------------------------------


def compute_proposal_recall(
    candidates: list[Candidate],
    ground_truth_events: list[dict],
    *,
    actor_aware: bool = False,
    region_aware: bool = False,
) -> tuple[list[ProposalRecallResult], dict]:
    """Compute proposal recall: fraction of ground-truth events covered by candidates.

    Coverage methods:
    - "raw": event timestamp within [raw_start_s, raw_end_s]
    - "padded": event timestamp within [window_start_s, window_end_s]

    Parameters
    ----------
    candidates : list[Candidate]
        Candidates from generate_candidates().
    ground_truth_events : list[dict]
        Ground-truth events with keys: event_id, clip_id, type, t_start, t_end,
        and optionally actor_id, region_id.
    actor_aware : bool
        When True, require actor_id match for coverage.
    region_aware : bool
        When True, require region_id match for coverage.

    Returns
    -------
    per_event_results : list[ProposalRecallResult]
        Per-event coverage results.
    aggregate : dict
        Aggregate recall metrics.
    """
    if not candidates:
        results = [
            ProposalRecallResult(
                event_id=ev.get("event_id"),
                clip_id=ev["clip_id"],
                gt_type=ev["type"],
                gt_t_start=ev["t_start"],
                gt_t_end=ev["t_end"],
                covered=False,
            )
            for ev in ground_truth_events
        ]
        return results, {
            "total_events": len(ground_truth_events),
            "covered_events": 0,
            "uncovered_events": len(ground_truth_events),
            "proposal_recall": 0.0,
        }

    covered_ids: set[str] = set()
    results: list[ProposalRecallResult] = []

    for ev in ground_truth_events:
        event_id = ev.get("event_id", "unknown")
        clip_id = ev["clip_id"]
        gt_start = ev["t_start"]
        gt_end = ev["t_end"]
        gt_type = ev["type"]
        gt_actor = ev.get("actor_id")
        gt_region = ev.get("region_id")

        covered = False
        coverage_method: str | None = None
        matching_cand: str | None = None
        actor_match: bool | None = None
        region_match: bool | None = None

        for cand in candidates:
            if cand.clip_id != clip_id:
                continue

            if actor_aware and gt_actor is not None and cand.actor_id != gt_actor:
                actor_match = False
                continue
            else:
                actor_match = True

            if region_aware and gt_region is not None and cand.region_id != gt_region:
                region_match = False
                continue
            else:
                region_match = True

            # Check raw coverage
            if cand.raw_start_s <= gt_start <= cand.raw_end_s:
                covered = True
                coverage_method = "raw"
                matching_cand = cand.candidate_id
                break

            # Check padded coverage
            if cand.window_start_s <= gt_start <= cand.window_end_s:
                covered = True
                coverage_method = "padded"
                matching_cand = cand.candidate_id
                break

        if covered:
            covered_ids.add(event_id)

        results.append(
            ProposalRecallResult(
                event_id=event_id,
                clip_id=clip_id,
                gt_type=gt_type,
                gt_t_start=gt_start,
                gt_t_end=gt_end,
                covered=covered,
                coverage_method=coverage_method,
                matching_candidate_id=matching_cand,
                actor_match=actor_match,
                region_match=region_match,
            )
        )

    total = len(ground_truth_events)
    n_covered = len(covered_ids)
    recall = n_covered / total if total > 0 else 0.0

    aggregate = {
        "total_events": total,
        "covered_events": n_covered,
        "uncovered_events": total - n_covered,
        "proposal_recall": round(recall, 4),
        "actor_aware": actor_aware,
        "region_aware": region_aware,
    }

    return results, aggregate


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _point_in_polygon(px: float, py: float, polygon: Polygon) -> bool:
    """Ray-casting algorithm for point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _point_to_polygon_distance(px: float, py: float, polygon: Polygon) -> float:
    """Compute minimum distance from point to polygon edges."""
    min_dist = float("inf")
    n = len(polygon)
    for i in range(n):
        j = (i + 1) % n
        dist = _point_to_segment_distance(
            px, py, polygon[i][0], polygon[i][1], polygon[j][0], polygon[j][1]
        )
        min_dist = min(min_dist, dist)
    return min_dist


def _point_to_segment_distance(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """Minimum distance from point to line segment."""
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5

    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_proposals_config(
    proposals_cfg: ProposalsConfig, region_cfg: RegionMeasurementConfig
) -> None:
    """Validate proposal configuration and raise ConfigError for invalid settings."""
    if proposals_cfg.target_fps <= 0:
        raise ConfigError(f"target_fps must be positive, got {proposals_cfg.target_fps}")
    if proposals_cfg.target_fps > 120:
        raise ConfigError(f"target_fps too high: {proposals_cfg.target_fps}")

    if not (0.0 <= proposals_cfg.minimum_wrist_confidence <= 1.0):
        raise ConfigError(
            f"minimum_wrist_confidence must be in [0, 1], got {proposals_cfg.minimum_wrist_confidence}"
        )

    if proposals_cfg.minimum_interaction_duration_s < 0:
        raise ConfigError(
            f"minimum_interaction_duration_s must be non-negative, got {proposals_cfg.minimum_interaction_duration_s}"
        )

    if proposals_cfg.merge_gap_s < 0:
        raise ConfigError(f"merge_gap_s must be non-negative, got {proposals_cfg.merge_gap_s}")

    if proposals_cfg.context_before_s < 0:
        raise ConfigError(
            f"context_before_s must be non-negative, got {proposals_cfg.context_before_s}"
        )

    if proposals_cfg.context_after_s < 0:
        raise ConfigError(
            f"context_after_s must be non-negative, got {proposals_cfg.context_after_s}"
        )

    if proposals_cfg.maximum_candidate_duration_s <= 0:
        raise ConfigError(
            f"maximum_candidate_duration_s must be positive, got {proposals_cfg.maximum_candidate_duration_s}"
        )

    if region_cfg.velocity_window_frames < 1:
        raise ConfigError(
            f"velocity_window_frames must be >= 1, got {region_cfg.velocity_window_frames}"
        )

    if not (0.0 <= region_cfg.reversal_threshold <= 1.0):
        raise ConfigError(
            f"reversal_threshold must be in [0, 1], got {region_cfg.reversal_threshold}"
        )


def validate_candidate(candidate: Candidate, clip_duration: float) -> None:
    """Validate a single candidate's timestamps against clip duration."""
    if not (0.0 <= candidate.raw_start_s <= candidate.raw_end_s <= clip_duration + 1e-6):
        raise ValidationError(
            f"Candidate {candidate.candidate_id}: raw timestamps out of range "
            f"[0, {clip_duration}]: [{candidate.raw_start_s}, {candidate.raw_end_s}]"
        )
    if not (0.0 <= candidate.window_start_s <= candidate.window_end_s <= clip_duration + 1e-6):
        raise ValidationError(
            f"Candidate {candidate.candidate_id}: window timestamps out of range "
            f"[0, {clip_duration}]: [{candidate.window_start_s}, {candidate.window_end_s}]"
        )
    if not (candidate.raw_start_s < candidate.raw_end_s):
        raise ValidationError(f"Candidate {candidate.candidate_id}: raw_start must be < raw_end")
    if not (candidate.window_start_s < candidate.window_end_s):
        raise ValidationError(
            f"Candidate {candidate.candidate_id}: window_start must be < window_end"
        )
