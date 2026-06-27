"""Deterministic overlapping window generation inside Stage A active spans.

Reads only active-span metadata — never Layer 1 predictions.
Produces stable, deterministic window IDs and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Window:
    """One analysis window with absolute timestamps and metadata."""

    clip_id: str
    active_span_id: str
    window_id: str
    window_start_s: float
    window_end_s: float
    duration_s: float
    overlap_s: float
    source_timestamp_s: float  # deterministic anchor (active_span t_start)

    @property
    def n_samples(self) -> int:
        return max(1, int(round(self.duration_s)))


@dataclass
class WindowConfig:
    """Parameters controlling window generation."""

    window_duration_s: float = 10.0
    step_duration_s: float = 5.0
    min_window_duration_s: float = 2.0
    max_windows_per_span: int = 50
    clip_duration_s: float | None = None  # optional clip-boundary clamp


def generate_windows(
    clip_id: str,
    active_span_id: str,
    active_start_s: float,
    active_end_s: float,
    *,
    config: WindowConfig | None = None,
    clip_duration_s: float | None = None,
) -> list[Window]:
    """Generate deterministic overlapping windows inside one active span.

    Handles:
    - short spans shorter than window_duration_s
    - final partial windows
    - clip-boundary clamping via clip_duration_s
    - stable source timestamps
    - deterministic window IDs
    """
    cfg = config or WindowConfig()

    # Clamp to clip boundary if provided
    if clip_duration_s is not None:
        active_end_s = min(active_end_s, clip_duration_s)

    # Skip if span is invalid
    if active_end_s <= active_start_s:
        return []

    # Clamp span to minimum window duration
    effective_start = active_start_s
    effective_end = active_end_s
    if (effective_end - effective_start) < cfg.min_window_duration_s:
        return []

    windows: list[Window] = []
    t = effective_start
    step = cfg.step_duration_s
    window_dur = cfg.window_duration_s

    while True:
        w_end = min(t + window_dur, effective_end)
        w_dur = w_end - t

        # Accept final partial window if above minimum
        if w_dur < cfg.min_window_duration_s:
            # Try shrinking previous window to make room
            if windows:
                prev = windows[-1]
                shrink = prev.window_end_s - (prev.window_start_s + cfg.min_window_duration_s)
                if shrink > 0:
                    # Remove previous, add two windows
                    windows.pop()
                    t = prev.window_start_s
                    windows.append(
                        Window(
                            clip_id=clip_id,
                            active_span_id=active_span_id,
                            window_id=_make_id(clip_id, active_span_id, len(windows)),
                            window_start_s=t,
                            window_end_s=t + cfg.min_window_duration_s,
                            duration_s=cfg.min_window_duration_s,
                            overlap_s=window_dur - cfg.min_window_duration_s,
                            source_timestamp_s=effective_start,
                        )
                    )
                    t = t + cfg.min_window_duration_s
                    continue
            break

        # Stop if we exceed max windows
        if len(windows) >= cfg.max_windows_per_span:
            break

        overlap = max(0.0, window_dur - step) if step < window_dur else 0.0

        windows.append(
            Window(
                clip_id=clip_id,
                active_span_id=active_span_id,
                window_id=_make_id(clip_id, active_span_id, len(windows)),
                window_start_s=t,
                window_end_s=w_end,
                duration_s=w_dur,
                overlap_s=overlap,
                source_timestamp_s=effective_start,
            )
        )

        next_t = t + step
        if next_t >= effective_end:
            break
        t = next_t

    return windows


def _make_id(clip_id: str, active_span_id: str, index: int) -> str:
    """Deterministic window ID from clip, span, and index."""
    return f"{clip_id}__{active_span_id}__w{index:04d}"


def generate_all_windows(
    active_spans: list[dict],
    *,
    config: WindowConfig | None = None,
    clip_durations: dict[str, float] | None = None,
) -> list[Window]:
    """Generate windows for all active spans across all clips.

    Parameters
    ----------
    active_spans:
        List of dicts with keys: clip_id, active_span_id, t_start, t_end.
    clip_durations:
        Optional mapping of clip_id -> clip_duration_s for boundary clamping.
    """
    cfg = config or WindowConfig()
    clip_durations = clip_durations or {}
    all_windows: list[Window] = []

    # Sort by clip_id then t_start for deterministic ordering
    sorted_spans = sorted(
        active_spans,
        key=lambda s: (s["clip_id"], s["t_start"], s.get("active_span_id", "")),
    )

    for span in sorted_spans:
        clip_id = span["clip_id"]
        clip_dur = clip_durations.get(clip_id)
        windows = generate_windows(
            clip_id=clip_id,
            active_span_id=span["active_span_id"],
            active_start_s=span["t_start"],
            active_end_s=span["t_end"],
            config=cfg,
            clip_duration_s=clip_dur,
        )
        all_windows.extend(windows)

    return all_windows
