"""Layer 2 Qwen client wrapping the shared VLM client.

Wraps call_vlm from pickup_putdown.vlm with Layer 2-specific prompt
construction, JSON parsing, and schema validation.

One retry after invalid JSON or schema validation. Failed responses
never enter predictions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from pickup_putdown.layer2.prompts import build_prompt
from pickup_putdown.layer2.schemas import Layer2Event, Layer2WindowResponse
from pickup_putdown.vlm import (
    VlmClientConfig,
    VlmRequest,
    call_vlm,
)

logger = logging.getLogger(__name__)


@dataclass
class QwenClientConfig:
    """Layer 2-specific client configuration."""

    model_id: str = "llamacpp/Qwen3.6-35B-A3B-UD-Q4_K_XL"
    base_url: str = "http://localhost:8080"
    max_tokens: int = 2048
    retry_max_tokens: int = 4096
    max_attempts: int = 2
    retry_delay_s: float = 0.0
    timeout_s: int = 180
    temperature: float = 0.0


@dataclass
class QwenAttempt:
    """One attempt (request + raw response) for audit."""

    attempt_number: int
    request: VlmRequest
    raw_response: str
    is_success: bool
    validation_errors: list[str] = field(default_factory=list)


@dataclass
class QwenResult:
    """Result of a Layer 2 inference call on one window."""

    window_id: str
    clip_id: str
    window_start_s: float
    window_end_s: float
    validated_response: Layer2WindowResponse | None
    attempts: list[QwenAttempt] = field(default_factory=list)
    error: str | None = None

    @property
    def predictions(self) -> list[Layer2Event]:
        if self.validated_response is not None:
            return self.validated_response.events
        return []


def _build_request(
    window_start_s: float,
    window_end_s: float,
    frame_count: int,
    fps: float,
    config: QwenClientConfig,
    max_tokens: int,
    attempt: int,
) -> VlmRequest:
    """Build a VlmRequest for one window."""
    system_prompt, user_prompt = build_prompt(window_start_s, window_end_s, frame_count, fps)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return VlmRequest(
        model=config.model_id,
        messages=messages,
        max_tokens=max_tokens,
        temperature=config.temperature,
        stream=False,
    )


def _parse_and_validate(
    content: str,
    window_id: str,
    clip_id: str,
    window_start_s: float,
    window_end_s: float,
) -> tuple[Layer2WindowResponse | None, list[str]]:
    """Parse JSON content and validate against Layer2WindowResponse schema.

    Returns (validated_response, validation_errors).
    """
    errors: list[str] = []

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        errors.append(f"JSON parse error: {exc}")
        return None, errors

    if not isinstance(parsed, dict):
        errors.append(f"Expected JSON object, got {type(parsed).__name__}")
        return None, errors

    # Validate events list
    events_raw = parsed.get("events", [])
    if not isinstance(events_raw, list):
        errors.append("'events' must be a list")
        events_raw = []

    validated_events: list[Layer2Event] = []
    for i, ev in enumerate(events_raw):
        if not isinstance(ev, dict):
            errors.append(f"events[{i}] is not an object")
            continue
        try:
            event = Layer2Event.model_validate(ev)
            validated_events.append(event)
        except Exception as exc:
            errors.append(f"events[{i}] validation failed: {exc}")

    # Check for extra top-level keys (warn but don't reject)
    allowed_keys = {"events", "reasoning"}
    extra_keys = set(parsed.keys()) - allowed_keys
    if extra_keys:
        errors.append(f"Extra top-level keys: {extra_keys}")

    response = Layer2WindowResponse(
        window_id=window_id,
        clip_id=clip_id,
        window_start_s=window_start_s,
        window_end_s=window_end_s,
        events=validated_events,
        reasoning=str(parsed.get("reasoning", "")),
        validation_errors=errors,
    )

    return response, errors


def call_qwen(
    window_id: str,
    clip_id: str,
    window_start_s: float,
    window_end_s: float,
    frame_count: int,
    fps: float,
    qwen_config: QwenClientConfig | None = None,
    vlm_config: VlmClientConfig | None = None,
) -> QwenResult:
    """Run Layer 2 inference on one window.

    Uses the shared VLM client. One retry after invalid JSON or schema
    validation failure. Failed responses never enter predictions.

    Preserves both raw attempts for audit.
    """
    qc = qwen_config or QwenClientConfig()
    vc = vlm_config or VlmClientConfig(
        base_url=qc.base_url,
        model=qc.model_id,
        max_tokens=qc.max_tokens,
        retry_max_tokens=qc.retry_max_tokens,
        max_attempts=qc.max_attempts,
        retry_delay_s=qc.retry_delay_s,
        timeout_s=qc.timeout_s,
        temperature=qc.temperature,
    )

    attempts: list[QwenAttempt] = []
    validated: Layer2WindowResponse | None = None
    last_error: str | None = None

    for attempt_num in range(1, qc.max_attempts + 1):
        max_tokens = qc.max_tokens if attempt_num == 1 else qc.retry_max_tokens

        request = _build_request(
            window_start_s,
            window_end_s,
            frame_count,
            fps,
            qc,
            max_tokens,
            attempt_num,
        )

        response = call_vlm(request, vc)

        attempt = QwenAttempt(
            attempt_number=attempt_num,
            request=request,
            raw_response=response.content,
            is_success=response.is_success,
        )

        if not response.is_success:
            attempt.validation_errors.append(
                f"VLM call failed: {response.error.message if response.error else 'unknown'}"
            )
            attempts.append(attempt)
            last_error = response.error.message if response.error else "unknown"
            continue

        validated_response, validation_errors = _parse_and_validate(
            response.content,
            window_id,
            clip_id,
            window_start_s,
            window_end_s,
        )
        attempt.validation_errors = validation_errors
        attempts.append(attempt)

        # Retry on any validation error: structural (bad JSON, missing events)
        # or per-event (invalid type, bad timestamps, wrong item count, etc.).
        # Both attempts and both validation results are preserved for audit.
        if validated_response is None or validation_errors:
            if attempt_num < qc.max_attempts:
                logger.warning(
                    "Window %s: attempt %d validation errors, retrying: %s",
                    window_id,
                    attempt_num,
                    validation_errors,
                )
                continue
            # Last attempt still recorded for audit even with errors
            last_error = "; ".join(validation_errors)
        else:
            validated = validated_response
            break

    return QwenResult(
        window_id=window_id,
        clip_id=clip_id,
        window_start_s=window_start_s,
        window_end_s=window_end_s,
        validated_response=validated,
        attempts=attempts,
        error=last_error,
    )


def call_qwen_batch(
    windows_info: list[dict],
    video_path: str,
    clip_fps: float,
    clip_duration_s: float,
    *,
    qwen_config: QwenClientConfig | None = None,
    vlm_config: VlmClientConfig | None = None,
) -> list[QwenResult]:
    """Run Layer 2 inference on multiple windows.

    windows_info: list of dicts with keys:
        window_id, clip_id, window_start_s, window_end_s,
        frame_count, fps
    """
    results: list[QwenResult] = []
    for info in windows_info:
        result = call_qwen(
            window_id=info["window_id"],
            clip_id=info["clip_id"],
            window_start_s=info["window_start_s"],
            window_end_s=info["window_end_s"],
            frame_count=info.get("frame_count", 10),
            fps=info.get("fps", clip_fps),
            qwen_config=qwen_config,
            vlm_config=vlm_config,
        )
        results.append(result)
    return results
