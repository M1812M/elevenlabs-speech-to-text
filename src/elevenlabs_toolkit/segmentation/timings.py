from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from ..models import Cue, Word

DEFAULT_SRT_FPS = 30.0
DEFAULT_PADDING_FRAMES = 2
MIN_GAP_MILLISECONDS = 80
MIN_CUE_MILLISECONDS = 1


def _is_spoken_character(text: str) -> bool:
    return any(character.isalnum() for character in text.strip())


def spoken_start(word: Word) -> float:
    characters = tuple(character for character in word.characters if _is_spoken_character(character.text))
    if not characters:
        return word.start
    return min(word.end, max(word.start, min(character.start for character in characters)))


def spoken_end(word: Word) -> float:
    characters = tuple(character for character in word.characters if _is_spoken_character(character.text))
    if not characters:
        return word.end
    return max(word.start, min(word.end, max(character.end for character in characters)))


def _natural_milliseconds(group: tuple[Word, ...]) -> tuple[int, int]:
    start = min(spoken_start(word) for word in group)
    end = max(spoken_end(word) for word in group)
    return round(start * 1000), round(end * 1000)


def cues_with_precise_gaps(
    groups: Iterable[Sequence[Word]],
    *,
    frames_per_second: float = DEFAULT_SRT_FPS,
    padding_frames: int = DEFAULT_PADDING_FRAMES,
    minimum_gap_milliseconds: int = MIN_GAP_MILLISECONDS,
) -> tuple[Cue, ...]:
    """Pad precise character edges by frames while retaining a minimum cue gap."""

    if isinstance(frames_per_second, bool) or not math.isfinite(frames_per_second) or frames_per_second <= 0:
        raise ValueError("frames_per_second must be a finite number > 0")
    if isinstance(padding_frames, bool) or not isinstance(padding_frames, int) or padding_frames < 0:
        raise ValueError("padding_frames must be an integer >= 0")
    if (
        isinstance(minimum_gap_milliseconds, bool)
        or not isinstance(minimum_gap_milliseconds, int)
        or minimum_gap_milliseconds < 0
    ):
        raise ValueError("minimum_gap_milliseconds must be an integer >= 0")

    padding_milliseconds = round(padding_frames * 1000 / frames_per_second)

    pending = [tuple(group) for group in groups if group]
    while pending:
        timings = [_natural_milliseconds(group) for group in pending]
        natural_starts = [start for start, _end in timings]
        natural_ends = [max(end, start + MIN_CUE_MILLISECONDS) for start, end in timings]
        starts = [max(0, start - padding_milliseconds) for start in natural_starts]
        ends = [end + padding_milliseconds for end in natural_ends]
        merge_at: int | None = None

        for index in range(len(pending) - 1):
            if starts[index + 1] - ends[index] >= minimum_gap_milliseconds:
                continue
            preferred_end = round((natural_ends[index] + natural_starts[index + 1] - minimum_gap_milliseconds) / 2)
            lower_end = max(starts[index], natural_starts[index]) + MIN_CUE_MILLISECONDS
            upper_end = natural_ends[index + 1] - minimum_gap_milliseconds - MIN_CUE_MILLISECONDS
            if lower_end > upper_end:
                merge_at = index
                break
            ends[index] = min(max(preferred_end, lower_end), upper_end)
            starts[index + 1] = ends[index] + minimum_gap_milliseconds

        if merge_at is None:
            invalid_index = next(
                (index for index, (start, end) in enumerate(zip(starts, ends, strict=True)) if end <= start),
                None,
            )
            if invalid_index is None:
                return tuple(
                    Cue(group, start_override=start / 1000, end_override=end / 1000)
                    for group, start, end in zip(pending, starts, ends, strict=True)
                )
            merge_at = max(0, min(invalid_index, len(pending) - 2))

        pending[merge_at : merge_at + 2] = [(*pending[merge_at], *pending[merge_at + 1])]
    return ()


__all__ = [
    "DEFAULT_PADDING_FRAMES",
    "DEFAULT_SRT_FPS",
    "MIN_GAP_MILLISECONDS",
    "cues_with_precise_gaps",
    "spoken_end",
    "spoken_start",
]
