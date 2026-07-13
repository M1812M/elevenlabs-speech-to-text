"""Character-timing heuristics used by subtitle and sentence segmentation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from statistics import median
from typing import Any

STRETCHED_CHAR_MIN_DURATION = 0.35
STRETCHED_CHAR_MIN_SHARE = 0.45
STRETCHED_CHAR_MIN_EXCESS = 0.18
STRETCHED_CHAR_RATIO_TO_MAX = 2.5
STRETCHED_CHAR_RATIO_TO_MEDIAN = 4.0
STRETCHED_CHAR_KEEP_FACTOR_MAX = 1.25
STRETCHED_CHAR_KEEP_FACTOR_MEDIAN = 2.0
STRETCHED_CHAR_KEEP_MIN = 0.16
STRETCHED_CHAR_KEEP_MAX = 0.30


def _safe_float(value: object, fallback: float) -> float:
    if isinstance(value, bool):
        return fallback
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    return parsed if math.isfinite(parsed) else fallback


def _is_spoken_character(text: str) -> bool:
    return any(character.isalnum() for character in str(text or "").strip())


def detect_stretched_character_pause_end(word: Mapping[str, Any]) -> float | None:
    """Return the likely spoken end when the last character contains a long hold."""

    characters = word.get("characters")
    if not isinstance(characters, list) or len(characters) < 3:
        return None

    parsed: list[tuple[str, float, float, float]] = []
    for item in characters:
        if not isinstance(item, Mapping):
            return None
        text = str(item.get("text") or "")
        start = _safe_float(item.get("start"), 0.0)
        end = _safe_float(item.get("end"), start)
        parsed.append((text, start, end, max(0.0, end - start)))

    anchor_index: int | None = None
    for index in range(len(parsed) - 1, -1, -1):
        text, _start, _end, duration = parsed[index]
        if duration > 0.0 and _is_spoken_character(text):
            anchor_index = index
            break

    if anchor_index is None or anchor_index < 2:
        return None

    _anchor_text, anchor_start, anchor_end, anchor_duration = parsed[anchor_index]
    spoken_before = [
        duration
        for text, _start, _end, duration in parsed[:anchor_index]
        if duration > 0.0 and _is_spoken_character(text)
    ]
    if len(spoken_before) < 2:
        return None

    word_start = _safe_float(word.get("start"), parsed[0][1])
    total_duration = max(0.0, anchor_end - word_start)
    previous_max = max(spoken_before)
    previous_median = median(spoken_before)
    if total_duration <= 0.0 or previous_max <= 0.0 or previous_median <= 0.0:
        return None

    if anchor_duration < STRETCHED_CHAR_MIN_DURATION:
        return None
    if (anchor_duration / total_duration) < STRETCHED_CHAR_MIN_SHARE:
        return None
    if (anchor_duration - previous_max) < STRETCHED_CHAR_MIN_EXCESS:
        return None
    if not (
        anchor_duration >= previous_max * STRETCHED_CHAR_RATIO_TO_MAX
        or anchor_duration >= previous_median * STRETCHED_CHAR_RATIO_TO_MEDIAN
    ):
        return None

    kept_duration = max(
        STRETCHED_CHAR_KEEP_MIN,
        previous_max * STRETCHED_CHAR_KEEP_FACTOR_MAX,
        previous_median * STRETCHED_CHAR_KEEP_FACTOR_MEDIAN,
    )
    kept_duration = min(STRETCHED_CHAR_KEEP_MAX, kept_duration)
    effective_end = anchor_start + min(anchor_duration, kept_duration)
    if effective_end >= anchor_end or effective_end <= word_start:
        return None
    return effective_end


def effective_word_end(word: Mapping[str, Any], pause_detection: bool = False) -> float:
    start = _safe_float(word.get("start"), 0.0)
    end = _safe_float(word.get("end"), start)
    if not pause_detection:
        return end
    adjusted_end = detect_stretched_character_pause_end(word)
    return end if adjusted_end is None else max(start, min(end, adjusted_end))


def detected_pause_gap(word: Mapping[str, Any], pause_detection: bool = False) -> float:
    if not pause_detection:
        return 0.0
    start = _safe_float(word.get("start"), 0.0)
    end = _safe_float(word.get("end"), start)
    adjusted_end = detect_stretched_character_pause_end(word)
    return 0.0 if adjusted_end is None else max(0.0, end - adjusted_end)
