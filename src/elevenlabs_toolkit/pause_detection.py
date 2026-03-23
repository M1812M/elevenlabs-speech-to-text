from statistics import median
from typing import Dict, List, Optional, Tuple


STRETCHED_CHAR_MIN_DURATION = 0.35
STRETCHED_CHAR_MIN_SHARE = 0.45
STRETCHED_CHAR_MIN_EXCESS = 0.18
STRETCHED_CHAR_RATIO_TO_MAX = 2.5
STRETCHED_CHAR_RATIO_TO_MEDIAN = 4.0
STRETCHED_CHAR_KEEP_FACTOR_MAX = 1.25
STRETCHED_CHAR_KEEP_FACTOR_MEDIAN = 2.0
STRETCHED_CHAR_KEEP_MIN = 0.16
STRETCHED_CHAR_KEEP_MAX = 0.30


def _safe_float(value, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _is_spoken_character(text: str) -> bool:
    return any(char.isalnum() for char in str(text or "").strip())


def detect_stretched_character_pause_end(word: Dict) -> Optional[float]:
    characters = word.get("characters")
    if not isinstance(characters, list) or len(characters) < 3:
        return None

    parsed: List[Tuple[str, float, float, float]] = []
    for item in characters:
        text = str(item.get("text") or "")
        start = _safe_float(item.get("start"), 0.0)
        end = _safe_float(item.get("end"), start)
        duration = max(0.0, end - start)
        parsed.append((text, start, end, duration))

    if len(parsed) < 3:
        return None

    anchor_idx = None
    for idx in range(len(parsed) - 1, -1, -1):
        text, _start, _end, duration = parsed[idx]
        if duration > 0.0 and _is_spoken_character(text):
            anchor_idx = idx
            break

    if anchor_idx is None or anchor_idx < 2:
        return None

    anchor_text, anchor_start, anchor_end, anchor_duration = parsed[anchor_idx]
    del anchor_text

    spoken_before = [
        duration
        for text, _start, _end, duration in parsed[:anchor_idx]
        if duration > 0.0 and _is_spoken_character(text)
    ]
    if len(spoken_before) < 2:
        return None

    word_start = _safe_float(word.get("start"), parsed[0][1])
    total_duration = max(0.0, anchor_end - word_start)
    if total_duration <= 0.0:
        return None

    prev_max = max(spoken_before)
    prev_median = median(spoken_before)
    if prev_max <= 0.0 or prev_median <= 0.0:
        return None

    if anchor_duration < STRETCHED_CHAR_MIN_DURATION:
        return None
    if (anchor_duration / total_duration) < STRETCHED_CHAR_MIN_SHARE:
        return None
    if (anchor_duration - prev_max) < STRETCHED_CHAR_MIN_EXCESS:
        return None
    if not (
        anchor_duration >= (prev_max * STRETCHED_CHAR_RATIO_TO_MAX)
        or anchor_duration >= (prev_median * STRETCHED_CHAR_RATIO_TO_MEDIAN)
    ):
        return None

    kept_duration = max(
        STRETCHED_CHAR_KEEP_MIN,
        prev_max * STRETCHED_CHAR_KEEP_FACTOR_MAX,
        prev_median * STRETCHED_CHAR_KEEP_FACTOR_MEDIAN,
    )
    kept_duration = min(STRETCHED_CHAR_KEEP_MAX, kept_duration)
    effective_end = anchor_start + min(anchor_duration, kept_duration)

    if effective_end >= anchor_end:
        return None
    if effective_end <= word_start:
        return None
    return effective_end


def effective_word_end(word: Dict, pause_detection: bool = False) -> float:
    start = _safe_float(word.get("start"), 0.0)
    end = _safe_float(word.get("end"), start)
    if not pause_detection:
        return end

    adjusted_end = detect_stretched_character_pause_end(word)
    if adjusted_end is None:
        return end
    return max(start, min(end, adjusted_end))


def detected_pause_gap(word: Dict, pause_detection: bool = False) -> float:
    if not pause_detection:
        return 0.0
    start = _safe_float(word.get("start"), 0.0)
    end = _safe_float(word.get("end"), start)
    adjusted_end = detect_stretched_character_pause_end(word)
    if adjusted_end is None:
        return 0.0
    return max(0.0, end - adjusted_end)
