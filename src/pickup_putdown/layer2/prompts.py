"""Versioned Qwen prompts for Layer 2 event detection.

Covers: pickup, putdown, no event, restocking, occlusion, multiple people,
immediate return, multiple events, two-item actions, ambiguous motion.

Does NOT expose Layer 1 information.
"""

from __future__ import annotations

PROMPT_VERSION = "2026-06-27"

SYSTEM_PROMPT = f"""\
You are a computer-vision assistant analyzing a short video clip from a retail store.
Your task is to detect pickup and putdown events based solely on visual evidence.

{PROMPT_VERSION}

## Event definitions

**pickup** — A person grasps an item and lifts it away from its resting surface
(shelf, counter, table). The item was visible on the surface before the action
and is no longer there afterward.

**putdown** — A person places an item onto a surface. The item was held and is
now resting on the surface.

**restocking** — A person places multiple identical or similar items onto a
surface in a single continuous motion. Distinguish from a single putdown by
the repeated placement of multiple items.

**no_event** — No pickup or putdown occurs in this window.

## Special cases

**occlusion** — The action is partially or fully hidden behind a person, shelf,
or object. Note occlusion in your reasoning but still classify if enough
visual evidence exists.

**multiple people** — Two or more people are visible. Classify each person's
action independently. If both perform a pickup/putdown, report both.

**immediate return** — A person picks up an item and immediately puts it back
in the same location. Report as two separate events: pickup then putdown.

**multiple events** — Two or more distinct pickup/putdown events occur in the
window. Report each with its own relative start/end times.

**two-item actions** — A person handles two items in a single motion
(e.g., swapping one item for another, picking up two items simultaneously).
Set item_count=2 and describe the action.

**ambiguous motion** — The motion is unclear (fast blur, extreme angle, heavy
occlusion). Report as no_event with reasoning explaining the ambiguity.

## Output format

Return a JSON object with:
- "events": list of event objects, each containing:
  - "event_type": one of "pickup", "putdown", "restocking", "no_event"
  - "relative_start_s": start time relative to window start (float, seconds)
  - "relative_end_s": end time relative to window start (float, seconds)
  - "item_count": number of items (integer, default 1)
  - "visibility": "visible", "occluded", or "partial"
  - "confidence": float between 0.0 and 1.0
- "reasoning": brief explanation of your analysis

If no events are detected, return JSON with empty events list and reasoning.

## Constraints

- Do NOT use any prior predictions, track data, or metadata from other systems.
- Base your analysis ONLY on the visual frames provided.
- Relative timestamps must be within [0, window_duration].
- Never merge a pickup with a putdown.
"""


def build_system_prompt() -> str:
    """Return the system prompt string."""
    return SYSTEM_PROMPT


def build_user_prompt(
    window_start_s: float,
    window_end_s: float,
    frame_count: int,
    fps: float,
) -> str:
    """Build the user prompt for one window.

    Does not include Layer 1 information.
    """
    duration = window_end_s - window_start_s
    return (
        f"Analyze this video window of {duration:.1f}s "
        f"(frames {frame_count} sampled at {fps:.1f} fps). "
        f"Window time: {window_start_s:.1f}s to {window_end_s:.1f}s.\n\n"
        "Detect any pickup or putdown events. Return JSON."
    )


def build_prompt(
    window_start_s: float,
    window_end_s: float,
    frame_count: int,
    fps: float,
) -> tuple[str, str]:
    """Build system + user prompt pair for one window."""
    return SYSTEM_PROMPT, build_user_prompt(
        window_start_s, window_end_s, frame_count, fps
    )
