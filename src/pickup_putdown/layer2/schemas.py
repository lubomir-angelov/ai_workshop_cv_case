"""Strict Pydantic schemas for Layer 2 VLM event predictions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

VALID_EVENT_TYPES = ("pickup", "putdown", "restocking", "no_event", "ambiguous")


class Layer2Event(BaseModel):
    """One predicted event from a Layer 2 window.

    Zero or more of these may appear in a single window response.
    """

    model_config = {"extra": "forbid"}

    event_type: str = Field(..., description="One of pickup, putdown, restocking, no_event, ambiguous")
    relative_start_s: float = Field(..., description="Start time relative to window start (seconds)")
    relative_end_s: float = Field(..., description="End time relative to window start (seconds)")
    item_count: int = Field(default=1, ge=1, le=10, description="Number of items involved")
    visibility: str = Field(default="visible", description="visible, occluded, partial")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")

    @field_validator("event_type")
    @classmethod
    def valid_event_type(cls, v: str) -> str:
        if v not in VALID_EVENT_TYPES:
            raise ValueError(
                f"event_type must be one of {VALID_EVENT_TYPES}, got {v!r}"
            )
        return v

    @field_validator("relative_start_s", "relative_end_s")
    @classmethod
    def non_negative_timestamp(cls, v: float) -> float:
        if v < 0:
            raise ValueError("timestamp must be non-negative")
        return v

    @field_validator("relative_end_s")
    @classmethod
    def end_after_start(cls, v: float, info) -> float:
        data = info.data or {}
        start = data.get("relative_start_s")
        if start is not None and v <= start:
            raise ValueError("relative_end_s must be greater than relative_start_s")
        return v


class Layer2WindowResponse(BaseModel):
    """Validated response from a single Layer 2 window.

    Contains zero or more Layer2Event predictions plus metadata.
    """

    window_id: str
    clip_id: str
    window_start_s: float
    window_end_s: float
    events: list[Layer2Event] = Field(default_factory=list)
    reasoning: str = ""
    validation_errors: list[str] = Field(default_factory=list)

    @field_validator("window_end_s")
    @classmethod
    def end_after_start(cls, v: float, info) -> float:
        data = info.data or {}
        start = data.get("window_start_s")
        if start is not None and v <= start:
            raise ValueError("window_end_s must be greater than window_start_s")
        return v


class Layer2Prediction(BaseModel):
    """Canonical Layer 2 prediction with absolute timestamps and provenance.

    This is the output format used by merge_predictions and evaluation.
    """

    clip_id: str
    pred_id: str
    event_type: str
    t_start_s: float
    t_end_s: float
    item_count: int = 1
    visibility: str = "visible"
    confidence: float = 1.0
    contributing_window_ids: list[str] = Field(default_factory=list)
    model: str = "layer2_qwen"

    @field_validator("t_end_s")
    @classmethod
    def end_after_start(cls, v: float, info) -> float:
        data = info.data or {}
        start = data.get("t_start_s")
        if start is not None and v <= start:
            raise ValueError("t_end_s must be greater than t_start_s")
        return v

    @field_validator("confidence")
    @classmethod
    def valid_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be in [0.0, 1.0]")
        return v

    def to_canonical(self) -> dict[str, Any]:
        """Convert to Task 8 canonical prediction dict."""
        return {
            "clip_id": self.clip_id,
            "pred_id": self.pred_id,
            "type": self.event_type,
            "t_start": self.t_start_s,
            "t_end": self.t_end_s,
            "score": self.confidence,
            "model": self.model,
        }
