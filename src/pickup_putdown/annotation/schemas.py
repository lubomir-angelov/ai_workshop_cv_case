"""Typed models shared by the annotation and VLM-annotation modules.

Annotation interchange is handled by CVAT (see
``scripts/convert_cvat_source_export.py``); these enums remain as the shared
vocabulary for event labels and confidence levels.
"""

from __future__ import annotations

from enum import StrEnum


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MED = "med"
    LOW = "low"


class EventLabel(StrEnum):
    PICKUP = "pickup"
    PUTDOWN = "putdown"
    IGNORE = "ignore"
