"""Merge duplicates caused by overlapping windows.

Deterministic, event-type-aware merging:
- Never merge pickup with putdown
- Do not over-merge repeated actions
- Preserve item counts and two-item semantics
- Retain contributing window IDs
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pickup_putdown.layer2.schemas import Layer2Prediction, Layer2WindowResponse


def _make_pred_id(clip_id: str, event_type: str, t_start: float, t_end: float) -> str:
    """Deterministic prediction ID for dedup grouping."""
    raw = f"{clip_id}|{event_type}|{t_start:.3f}|{t_end:.3f}"
    return f"l2_{hashlib.md5(raw.encode()).hexdigest()[:12]}"


@dataclass(frozen=True)
class _GroupKey:
    """Key for grouping overlapping predictions of same event type."""

    clip_id: str
    event_type: str


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    """True if intervals have any positive temporal overlap."""
    return a_start < b_end and b_start < a_end


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping intervals, return sorted list."""
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [sorted_iv[0]]
    for start, end in sorted_iv[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def merge_predictions(
    responses: list[Layer2WindowResponse],
    *,
    merge_threshold_s: float = 1.0,
) -> list[Layer2Prediction]:
    """Merge duplicate predictions from overlapping windows.

    Algorithm:
    1. Collect all events from all windows with their window IDs.
    2. Group by (clip_id, event_type).
    3. Within each group, sort by start time and merge overlapping
       predictions that are within merge_threshold_s of each other.
    4. Preserve highest confidence, max item_count, and all contributing window IDs.
    5. Never merge different event types.
    """
    # Collect all events with provenance
    all_events: list[dict] = []
    for resp in responses:
        for ev in resp.events:
            if ev.event_type in ("no_event",):
                continue
            all_events.append(
                {
                    "event_type": ev.event_type,
                    "t_start": resp.window_start_s + ev.relative_start_s,
                    "t_end": resp.window_start_s + ev.relative_end_s,
                    "item_count": ev.item_count,
                    "visibility": ev.visibility,
                    "confidence": ev.confidence,
                    "window_id": resp.window_id,
                    "clip_id": resp.clip_id,
                }
            )

    if not all_events:
        return []

    # Group by (clip_id, event_type)
    groups: dict[_GroupKey, list[dict]] = {}
    for ev in all_events:
        key = _GroupKey(
            clip_id=ev["clip_id"],
            event_type=ev["event_type"],
        )
        groups.setdefault(key, []).append(ev)

    predictions: list[Layer2Prediction] = []

    for key, events in sorted(groups.items(), key=lambda x: (x[0].clip_id, x[0].event_type)):
        # Sort by start time for deterministic merging
        events.sort(key=lambda e: (e["t_start"], e["t_end"]))

        # Merge overlapping predictions within threshold
        merged_groups: list[list[dict]] = []
        current_group: list[dict] = [events[0]]

        for ev in events[1:]:
            prev = current_group[-1]
            # Check if this event overlaps with any in current group
            overlaps_any = any(
                _overlaps(ev["t_start"], ev["t_end"], g["t_start"], g["t_end"])
                for g in current_group
            )
            # Also check proximity to handle gap-filling
            gap = ev["t_start"] - prev["t_end"]
            if overlaps_any or 0 < gap <= merge_threshold_s:
                current_group.append(ev)
            else:
                merged_groups.append(current_group)
                current_group = [ev]

        if current_group:
            merged_groups.append(current_group)

        # Build predictions from merged groups
        for group in merged_groups:
            # Merge intervals
            intervals = [(e["t_start"], e["t_end"]) for e in group]
            merged_intervals = _merge_intervals(intervals)

            # Preserve max item_count and highest confidence
            max_items = max(e["item_count"] for e in group)
            max_conf = max(e["confidence"] for e in group)

            for m_start, m_end in merged_intervals:
                pred_id = _make_pred_id(key.clip_id, key.event_type, m_start, m_end)
                contributing = list({e["window_id"] for e in group})
                # Two-item convention: shared group_id across all items
                group_id = pred_id if max_items > 1 else ""

                predictions.append(
                    Layer2Prediction(
                        clip_id=key.clip_id,
                        pred_id=pred_id,
                        event_type=key.event_type,
                        t_start_s=m_start,
                        t_end_s=m_end,
                        item_count=max_items,
                        confidence=max_conf,
                        contributing_window_ids=sorted(contributing),
                        event_group_id=group_id,
                    )
                )

    return predictions
