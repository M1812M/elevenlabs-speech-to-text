"""Timecode serialization helpers for local renderers."""


def srt_timestamp(seconds: float) -> str:
    """Convert seconds to the SubRip ``HH:MM:SS,mmm`` representation."""

    milliseconds = round(max(seconds, 0.0) * 1000)
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    seconds_part = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d},{milliseconds:03d}"
