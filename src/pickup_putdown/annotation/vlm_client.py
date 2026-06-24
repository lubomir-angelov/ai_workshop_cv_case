"""llama.cpp OpenAI-compatible vision client for VLM annotation.

Sends contact sheet images to a local llama.cpp server running a vision model
and parses the structured JSON response into event annotations.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from pickup_putdown.annotation.schemas import ConfidenceLevel, EventLabel

logger = logging.getLogger(__name__)


class VlmClientConfig(BaseModel):
    """Configuration for the llama.cpp VLM client."""

    base_url: str = "http://localhost:8080"
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_s: int = 120


SYSTEM_PROMPT = """\
You are a video event annotator for a retail pickup/putdown detection task.

Analyze the contact sheet image showing sequential frames from a short candidate \
video clip. Each frame is labeled with its frame number.

Identify pickup and putdown events:

pickup: A person removes an item from a shelf or surface and takes it into their \
hand so that the item leaves its resting place.

putdown: A person places an item they were holding onto a shelf or surface and \
releases it so that it remains resting there.

Do NOT annotate:
- touching or inspecting without removing
- reaching past an item
- browsing or standing near shelves
- hand motion without object transfer
- empty or no-person clips

For each event, determine:
- label: "pickup" or "putdown"
- start_frame: first frame where the purposeful transfer action begins
- end_frame: last frame where the transfer is complete
- item_count: number of items involved (default 1)
- confidence: "high", "med", or "low"
- hard_case: true if the event is ambiguous or difficult to determine
- notes: brief explanation

If no valid pickup or putdown events are present, return an empty events array.

Respond ONLY with a JSON object matching this schema:
{
  "events": [
    {
      "label": "pickup",
      "start_frame": 5,
      "end_frame": 12,
      "item_count": 1,
      "confidence": "high",
      "hard_case": false,
      "notes": "clear pickup from shelf"
    }
  ],
  "reasoning": "brief explanation of what was observed"
}
"""


def _image_to_base64(image_path: Path) -> str:
    """Read image file and return base64-encoded JPEG string."""
    data = image_path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def call_vlm(
    contact_sheet_path: Path,
    frame_count: int,
    fps: float,
    duration_s: float,
    config: VlmClientConfig,
) -> dict[str, Any]:
    """Call the VLM via llama.cpp OpenAI-compatible API.

    Returns dict with keys "events" (list of event dicts) and "reasoning" (str).
    On failure, returns {"events": [], "reasoning": "<error message>"}.
    """
    import urllib.error
    import urllib.request

    if not contact_sheet_path.exists():
        return {
            "events": [],
            "reasoning": f"Contact sheet not found: {contact_sheet_path}",
        }

    image_b64 = _image_to_base64(contact_sheet_path)

    payload = {
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Contact sheet: {frame_count} frames at {fps:.1f} review-fps "
                            f"over {duration_s:.1f}s candidate window. "
                            f"Frames are numbered sequentially. Identify any pickup or putdown events."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    },
                ],
            },
        ],
    }

    url = f"{config.base_url.rstrip('/')}/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=config.timeout_s) as resp:
            response = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        logger.error("VLM request failed: %s", exc)
        return {"events": [], "reasoning": f"VLM request failed: {exc}"}
    except json.JSONDecodeError as exc:
        logger.error("VLM returned invalid JSON: %s", exc)
        return {"events": [], "reasoning": f"Invalid JSON from VLM: {exc}"}
    except Exception as exc:
        logger.error("Unexpected VLM error: %s", exc)
        return {"events": [], "reasoning": f"Unexpected error: {exc}"}

    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        logger.error("Unexpected VLM response structure: %s", exc)
        return {"events": [], "reasoning": f"Unexpected response structure: {exc}"}

    return _parse_vlm_response(content)


def _parse_vlm_response(content: str) -> dict[str, Any]:
    """Parse VLM text response into structured dict.

    Handles JSON wrapped in markdown code blocks or bare JSON.
    """
    text = content.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [
            line
            for line in lines
            if not (line.strip().startswith("```") or line.strip().startswith("json"))
        ]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse VLM response as JSON")
        return {"events": [], "reasoning": f"Parse error. Raw: {text[:200]}"}

    events_raw = parsed.get("events", [])
    reasoning = parsed.get("reasoning", "")

    normalized: list[dict[str, Any]] = []
    for evt in events_raw:
        label_str = str(evt.get("label", "")).lower()
        if label_str not in ("pickup", "putdown"):
            continue

        normalized.append(
            {
                "label": label_str,
                "start_frame": int(evt.get("start_frame", 0)),
                "end_frame": int(evt.get("end_frame", 0)),
                "item_count": max(1, int(evt.get("item_count", 1))),
                "confidence": evt.get("confidence", "med"),
                "hard_case": bool(evt.get("hard_case", False)),
                "notes": str(evt.get("notes", "")),
            }
        )

    return {"events": normalized, "reasoning": reasoning}


def vlm_result_to_annotations(
    vlm_response: dict[str, Any],
    fps: float,
) -> list[dict[str, Any]]:
    """Convert VLM frame-based events to time-based annotations.

    Returns list of dicts compatible with VlMEventAnnotation constructor:
        label, start_s, end_s, item_count, confidence, hard_case, notes
    """
    if fps <= 0:
        fps = 5.0

    annotations: list[dict[str, Any]] = []
    for evt in vlm_response.get("events", []):
        start_frame = evt.get("start_frame", 0)
        end_frame = evt.get("end_frame", 0)

        start_s = start_frame / fps
        end_s = (end_frame + 1) / fps

        confidence_str = evt.get("confidence", "med")
        confidence_map = {
            "high": ConfidenceLevel.HIGH,
            "med": ConfidenceLevel.MED,
            "low": ConfidenceLevel.LOW,
        }

        annotations.append(
            {
                "label": EventLabel(evt["label"]),
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "item_count": evt.get("item_count", 1),
                "confidence": confidence_map.get(confidence_str, ConfidenceLevel.MED),
                "hard_case": evt.get("hard_case", False),
                "notes": evt.get("notes", ""),
            }
        )

    return annotations
