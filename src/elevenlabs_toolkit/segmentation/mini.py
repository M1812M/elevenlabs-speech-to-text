from __future__ import annotations

import re
from functools import cache

from ..models import Cue, Transcript, Word

SENTENCE_END_RE = re.compile(r"[.!?\u2026]+(?:[\"'\u2019\u00bb)]*)$")
COMMA_END_RE = re.compile(r",(?:[\"'\u2019\u00bb)]*)$")
LEXICAL_WORD_RE = re.compile(r"\w+(?:['\u2019\u02bb\u02bc-]\w+)*", re.UNICODE)

MIN_COMMA_WORDS = 3
MIN_COMMA_DURATION_SECONDS = 0.8
MIN_GAP_MILLISECONDS = 100
MIN_CUE_MILLISECONDS = 1


class MiniSrtError(ValueError):
    """Raised when deterministic mini subtitle segmentation is impossible."""


def _split_sentences(words: tuple[Word, ...]) -> tuple[tuple[Word, ...], ...]:
    groups: list[tuple[Word, ...]] = []
    start = 0
    for index, word in enumerate(words[:-1], start=1):
        if SENTENCE_END_RE.search(word.text.strip()):
            groups.append(words[start:index])
            start = index
    groups.append(words[start:])
    return tuple(group for group in groups if group)


def _spoken_word_count(words: tuple[Word, ...]) -> int:
    return sum(len(LEXICAL_WORD_RE.findall(word.text)) for word in words if word.kind == "word")


def _duration(words: tuple[Word, ...]) -> float:
    return max(word.end for word in words) - words[0].start


def _readable_comma_clause(words: tuple[Word, ...]) -> bool:
    return _spoken_word_count(words) >= MIN_COMMA_WORDS and _duration(words) >= MIN_COMMA_DURATION_SECONDS


def _is_yoki(word: Word) -> bool:
    return tuple(token.casefold() for token in LEXICAL_WORD_RE.findall(word.text)) == ("yoki",)


def _clause_groups(sentence: tuple[Word, ...]) -> tuple[tuple[Word, ...], ...]:
    """Choose safe comma and ``yoki`` splits without producing tiny fragments."""
    comma_boundaries = (
        index for index, word in enumerate(sentence[:-1], start=1) if COMMA_END_RE.search(word.text.strip())
    )
    yoki_boundaries = (index for index, word in enumerate(sentence[1:-1], start=1) if _is_yoki(word))
    boundaries = tuple(sorted({*comma_boundaries, *yoki_boundaries}))
    if not boundaries:
        return (sentence,)
    endpoints = (*boundaries, len(sentence))

    @cache
    def choose(start: int) -> tuple[tuple[Word, ...], ...] | None:
        best: tuple[tuple[Word, ...], ...] | None = None
        for end in endpoints:
            if end <= start:
                continue
            clause = sentence[start:end]
            is_whole_sentence = start == 0 and end == len(sentence)
            if not is_whole_sentence and not _readable_comma_clause(clause):
                continue
            candidate: tuple[tuple[Word, ...], ...]
            if end == len(sentence):
                candidate = (clause,)
            else:
                suffix = choose(end)
                if suffix is None:
                    continue
                candidate = (clause, *suffix)
            if best is None or (len(candidate), -max(len(Cue(group).text) for group in candidate)) > (
                len(best),
                -max(len(Cue(group).text) for group in best),
            ):
                best = candidate
        return best

    return choose(0) or (sentence,)


def _natural_milliseconds(group: tuple[Word, ...]) -> tuple[int, int]:
    return round(group[0].start * 1000), round(max(word.end for word in group) * 1000)


def _apply_fixed_gaps(groups: list[tuple[Word, ...]]) -> tuple[Cue, ...]:
    """Apply at least 100 ms between cues, merging only if no valid gap fits."""
    pending = list(groups)
    while True:
        timings = [_natural_milliseconds(group) for group in pending]
        starts = [start for start, _end in timings]
        ends = [max(end, start + MIN_CUE_MILLISECONDS) for start, end in timings]
        merge_at: int | None = None

        for index in range(len(pending) - 1):
            if starts[index + 1] - ends[index] >= MIN_GAP_MILLISECONDS:
                continue
            preferred_end = round((ends[index] + starts[index + 1] - MIN_GAP_MILLISECONDS) / 2)
            lower_end = starts[index] + MIN_CUE_MILLISECONDS
            upper_end = ends[index + 1] - MIN_GAP_MILLISECONDS - MIN_CUE_MILLISECONDS
            if lower_end > upper_end:
                merge_at = index
                break
            ends[index] = min(max(preferred_end, lower_end), upper_end)
            starts[index + 1] = ends[index] + MIN_GAP_MILLISECONDS

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


def segment_mini(transcript: Transcript) -> tuple[Cue, ...]:
    """Create sentence cues with safe comma splits from JSON timings alone."""
    words = transcript.timed_words
    if not words:
        raise MiniSrtError("srt-mini requires word or segment timestamps")

    groups: list[tuple[Word, ...]] = []
    for sentence in _split_sentences(words):
        groups.extend(_clause_groups(sentence))
    return _apply_fixed_gaps(groups)
