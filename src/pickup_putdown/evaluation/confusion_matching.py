"""Pass 2: type-agnostic temporal matching -> JSON-serializable confusion."""

from __future__ import annotations

from .class_aware_matching import by_clip, drop_ignored, match_one_to_one
from .contracts import VALID_TYPES, type_name


def evaluate_confusion(events, preds, criterion, ignores=()):
    """Nested string-keyed confusion: {gt_type: {pred_type: count}} (JSON-safe)."""
    events = drop_ignored(events, ignores)
    preds = drop_ignored(preds, ignores)
    ge, gp = by_clip(events), by_clip(preds)
    conf = {a: dict.fromkeys(VALID_TYPES, 0) for a in VALID_TYPES}
    for clip in set(ge) | set(gp):
        r = match_one_to_one(ge.get(clip, []), gp.get(clip, []), criterion)
        for g, p in r.matched:
            conf[type_name(g.type)][type_name(p.type)] += 1
    return conf
