def srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format."""
    ms = int(round(max(seconds, 0.0) * 1000))
    hh = ms // 3_600_000
    ms %= 3_600_000
    mm = ms // 60_000
    ms %= 60_000
    ss = ms // 1000
    ms %= 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

