"""Layer 2 Qwen client wrapping the shared VLM client.

Wraps call_vlm from pickup_putdown.vlm with Layer 2-specific prompt
construction, JSON parsing, and schema validation.

Multimodal: sends rendered timestamped frames as base64 data URLs.
Two semantic retries max per window; the shared client is called with
max_attempts=1 so Layer 2 controls the retry loop.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path

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

    model_id: str = "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
    base_url: str = "http://127.0.0.1:8000"
    max_tokens: int = 2048
    retry_max_tokens: int = 4096
    max_attempts: int = 2
    retry_delay_s: float = 0.0
    timeout_s: int = 300
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


def _encode_image_file(frame_path: str) -> dict:
    """Read an image file, detect MIME type, base64-encode, return OpenAI-compatible content part.

    Raises FileNotFoundError with a clear message when the frame does not exist.
    """
    p = Path(frame_path)
    if not p.is_file():
        raise FileNotFoundError(f"Frame file not found: {frame_path}")

    mime_type, _ = mimetypes.guess_type(frame_path)
    if mime_type is None:
        mime_type = "image/jpeg"

    encoded = base64.b64encode(p.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{encoded}",
        },
    }


def _build_request(
    window_start_s: float,
    window_end_s: float,
    frame_paths: list[str],
    fps: float,
    config: QwenClientConfig,
    max_tokens: int,
    attempt: int,
    previous_validation_errors: list[str] | None = None,
) -> VlmRequest:
    """Build a multimodal VlmRequest for one window.

    The user message interleaves chronological frame labels with encoded
    image content parts.  On retry (attempt > 1) the previous validation
    errors are appended so the model can self-correct.
    """
    system_prompt, _ = build_prompt(window_start_s, window_end_s, len(frame_paths), fps)

    # Build user content as a list: text + images interleaved.
    content_parts: list[dict] = []

    # Text preamble
    duration = window_end_s - window_start_s
    preamble = (
        f"Analyze this video window of {duration:.1f}s "
        f"({len(frame_paths)} frames sampled at {fps:.1f} fps). "
        f"Window time: {window_start_s:.1f}s to {window_end_s:.1f}s.\n\n"
    )
    content_parts.append({"type": "text", "text": preamble})

    # Interleave frame labels and images in chronological order.
    for i, frame_path in enumerate(frame_paths):
        label = f"Frame {i + 1}/{len(frame_paths)} (relative time ~{i * (duration / max(len(frame_paths), 1)):.1f}s):\n"
        content_parts.append({"type": "text", "text": label})
        content_parts.append(_encode_image_file(frame_path))

    # On retry, append validation errors so the model self-corrects.
    if attempt > 1 and previous_validation_errors:
        retry_text = (
            "\n\nPrevious attempt failed validation. "
            "Errors to fix:\n"
            + "\n".join(f"- {e}" for e in previous_validation_errors)
            + "\n\nReturn only corrected JSON with no extra prose."
        )
        content_parts.append({"type": "text", "text": retry_text})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_parts},
    ]

    return VlmRequest(
        model=config.model_id,
        messages=messages,
        max_tokens=max_tokens,
        temperature=config.temperature,
        stream=False,
        response_format={"type": "json_object"},
        chat_template_kwargs={"enable_thinking": False},
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

    Missing top-level 'events' key is a hard failure.  Extra top-level
    keys are rejected (strict mode).
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

    # Hard-fail: 'events' key must be present.
    if "events" not in parsed:
        errors.append("Missing required top-level key: 'events'")
        events_raw = []
    else:
        events_raw = parsed["events"]

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

    # Strict rejection of extra top-level keys.
    allowed_keys = {"events", "reasoning"}
    extra_keys = set(parsed.keys()) - allowed_keys
    if extra_keys:
        errors.append(f"Extra top-level keys not allowed: {extra_keys}")

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
    frame_paths: list[str],
    fps: float,
    qwen_config: QwenClientConfig | None = None,
    vlm_config: VlmClientConfig | None = None,
) -> QwenResult:
    """Run Layer 2 inference on one window.

    Uses the shared VLM client with max_attempts=1 so Layer 2 controls
    the retry loop.  Two semantic attempts max: first pass, then a
    self-correction pass with validation errors appended.

    Failed or partially invalid responses never enter predictions.
    """
    qc = qwen_config or QwenClientConfig()
    vc = vlm_config or VlmClientConfig(
        base_url=qc.base_url,
        model=qc.model_id,
        max_tokens=qc.max_tokens,
        retry_max_tokens=qc.retry_max_tokens,
        max_attempts=1,
        retry_delay_s=qc.retry_delay_s,
        timeout_s=qc.timeout_s,
        temperature=qc.temperature,
    )

    attempts: list[QwenAttempt] = []
    validated: Layer2WindowResponse | None = None
    last_error: str | None = None
    previous_errors: list[str] = []

    for attempt_num in range(1, qc.max_attempts + 1):
        max_tokens = qc.max_tokens if attempt_num == 1 else qc.retry_max_tokens

        request = _build_request(
            window_start_s,
            window_end_s,
            frame_paths,
            fps,
            qc,
            max_tokens,
            attempt_num,
            previous_errors if attempt_num > 1 else None,
        )

        response = call_vlm(request, vc, max_attempts=1)

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

        if validated_response is not None and not validation_errors:
            validated = validated_response
            break

        # Retry on any validation error.
        previous_errors = list(validation_errors)
        if attempt_num < qc.max_attempts:
            logger.warning(
                "Window %s: attempt %d validation errors, retrying: %s",
                window_id,
                attempt_num,
                validation_errors,
            )
            continue

        last_error = "; ".join(validation_errors)

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
    *,
    qwen_config: QwenClientConfig | None = None,
    vlm_config: VlmClientConfig | None = None,
) -> list[QwenResult]:
    """Run Layer 2 inference on multiple windows.

    windows_info: list of dicts with keys:
        window_id, clip_id, window_start_s, window_end_s,
        frame_paths, fps
    """
    results: list[QwenResult] = []
    for info in windows_info:
        result = call_qwen(
            window_id=info["window_id"],
            clip_id=info["clip_id"],
            window_start_s=info["window_start_s"],
            window_end_s=info["window_end_s"],
            frame_paths=info.get("frame_paths", []),
            fps=info.get("fps", 1.0),
            qwen_config=qwen_config,
            vlm_config=vlm_config,
        )
        results.append(result)
    return results
