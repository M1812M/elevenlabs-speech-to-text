from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..models import Cue, Word

MIN_GAP_MILLISECONDS = 100
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
    minimum_gap_milliseconds: int = MIN_GAP_MILLISECONDS,
) -> tuple[Cue, ...]:
    """Build cues from character edges, falling back to words, with a fixed minimum gap."""

    pending = [tuple(group) for group in groups if group]
    while pending:
        timings = [_natural_milliseconds(group) for group in pending]
        starts = [start for start, _end in timings]
        ends = [max(end, start + MIN_CUE_MILLISECONDS) for start, end in timings]
        merge_at: int | None = None

        for index in range(len(pending) - 1):
            if starts[index + 1] - ends[index] >= minimum_gap_milliseconds:
                continue
            preferred_end = round((ends[index] + starts[index + 1] - minimum_gap_milliseconds) / 2)
            lower_end = starts[index] + MIN_CUE_MILLISECONDS
            upper_end = ends[index + 1] - minimum_gap_milliseconds - MIN_CUE_MILLISECONDS
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


__all__ = ["MIN_GAP_MILLISECONDS", "cues_with_precise_gaps", "spoken_end", "spoken_start"]
