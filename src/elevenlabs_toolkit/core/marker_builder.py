from pathlib import Path
from typing import List, Tuple


DEFAULT_MARKER_COLOR = "ResolveColorBlue"
DEFAULT_MARKER_PREFIX = "Sentence"
DEFAULT_MARKER_FPS = 25
DEFAULT_TIMELINE_START_HOURS = 1


def _sanitize_marker_text(text: str) -> str:
    return " ".join((text or "").replace("|", " ").split()).strip()


def _timeline_start_frame(fps: int) -> int:
    return DEFAULT_TIMELINE_START_HOURS * 60 * 60 * fps


def seconds_to_edl_timecode(seconds: float, fps: int, start_frame: int | None = None) -> str:
    total_frames = int(round(max(seconds, 0.0) * fps))
    frame_index = total_frames + (start_frame if start_frame is not None else _timeline_start_frame(fps))

    frames_per_hour = fps * 60 * 60
    hh = frame_index // frames_per_hour
    frame_index %= frames_per_hour
    mm = frame_index // (fps * 60)
    frame_index %= fps * 60
    ss = frame_index // fps
    ff = frame_index % fps
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def cues_to_marker_edl(
    cues: List[Tuple[float, float, str]],
    *,
    title: str,
    fps: int = DEFAULT_MARKER_FPS,
    color: str = DEFAULT_MARKER_COLOR,
    marker_prefix: str = DEFAULT_MARKER_PREFIX,
) -> str:
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]

    for i, (start, _end, _text) in enumerate(cues, start=1):
        rec_in = seconds_to_edl_timecode(start, fps)
        rec_out = seconds_to_edl_timecode(start + (1.0 / fps), fps)
        marker_text = _sanitize_marker_text(f"{marker_prefix} {i}") if marker_prefix else str(i)
        lines.append(f"{i:03d}  001      V     C        {rec_in} {rec_out} {rec_in} {rec_out}")
        lines.append(f"|C:{color} |M:{marker_text} |D:1")
        lines.append("")

    return "\n".join(lines)


def write_marker_edl(
    cues: List[Tuple[float, float, str]],
    out_path: Path,
    *,
    title: str,
    fps: int = DEFAULT_MARKER_FPS,
    color: str = DEFAULT_MARKER_COLOR,
    marker_prefix: str = DEFAULT_MARKER_PREFIX,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        cues_to_marker_edl(
            cues,
            title=title,
            fps=fps,
            color=color,
            marker_prefix=marker_prefix,
        ),
        encoding="utf-8",
    )
