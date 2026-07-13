from __future__ import annotations

import math
from collections.abc import Sequence

from ..models import Cue

DEFAULT_COLOR = "ResolveColorBlue"
DEFAULT_MARKER_PREFIX = "Sentence"
DEFAULT_FPS = 25.0
DEFAULT_TIMELINE_START_HOURS = 1.0


def _single_line(value: object, *, strip_pipes: bool = False) -> str:
    text = str(value or "")
    if strip_pipes:
        text = text.replace("|", " ")
    return " ".join(text.split())


def _nominal_timecode_base(fps: float) -> int:
    try:
        value = float(fps)
    except (TypeError, ValueError) as exc:
        raise ValueError("fps must be a finite number > 0") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("fps must be a finite number > 0")
    # Timecode frame fields use the nominal integer rate (23.976 -> 24,
    # 29.97 -> 30). Keep very small but positive rates representable.
    return max(1, math.floor(value + 0.5))


def _timecode(frame_index: int, base: int) -> str:
    frames_per_hour = base * 60 * 60
    hours, frame_index = divmod(frame_index, frames_per_hour)
    minutes, frame_index = divmod(frame_index, base * 60)
    seconds, frames = divmod(frame_index, base)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"


def render_resolve_edl(
    cues: Sequence[Cue],
    title: str,
    fps: float = DEFAULT_FPS,
    color: str = DEFAULT_COLOR,
    marker_prefix: str = DEFAULT_MARKER_PREFIX,
    timeline_start_hours: float = DEFAULT_TIMELINE_START_HOURS,
) -> str:
    """Render one one-frame Resolve marker per cue as a non-drop-frame EDL."""
    base = _nominal_timecode_base(fps)
    rate = float(fps)
    try:
        timeline_hours = float(timeline_start_hours)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeline_start_hours must be a finite number >= 0") from exc
    if not math.isfinite(timeline_hours) or timeline_hours < 0:
        raise ValueError("timeline_start_hours must be a finite number >= 0")

    timeline_start_frame = math.floor(timeline_hours * 60 * 60 * base + 0.5)
    safe_title = _single_line(title)
    safe_color = _single_line(color, strip_pipes=True)
    safe_prefix = _single_line(marker_prefix, strip_pipes=True)
    lines = [f"TITLE: {safe_title}", "FCM: NON-DROP FRAME", ""]

    for index, cue in enumerate(cues, start=1):
        cue_frame = math.floor(max(cue.start, 0.0) * rate + 0.5)
        record_in = _timecode(timeline_start_frame + cue_frame, base)
        record_out = _timecode(timeline_start_frame + cue_frame + 1, base)
        marker_text = f"{safe_prefix} {index}" if safe_prefix else str(index)
        lines.append(f"{index:03d}  001      V     C        {record_in} {record_out} {record_in} {record_out}")
        lines.append(f"|C:{safe_color} |M:{marker_text} |D:1")
        lines.append("")

    return "\n".join(lines)
