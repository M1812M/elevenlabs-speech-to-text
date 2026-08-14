from __future__ import annotations

import re
from functools import cache

from ..languages import connector_boundaries, lexical_tokens
from ..models import Cue, Transcript, Word
from .timings import cues_with_precise_gaps

SENTENCE_END_RE = re.compile(r"[.!?\u2026]+(?:[\"'\u2019\u00bb)]*)$")
SEMICOLON_END_RE = re.compile(r";(?:[\"'\u2019\u00bb)]*)$")
COMMA_END_RE = re.compile(r",(?:[\"'\u2019\u00bb)]*)$")

MIN_CLAUSE_WORDS = 3
MIN_CLAUSE_DURATION_SECONDS = 0.8


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
    return sum(len(lexical_tokens(word.text)) for word in words if word.kind == "word")


def _duration(words: tuple[Word, ...]) -> float:
    return max(word.end for word in words) - words[0].start


def _readable_clause(words: tuple[Word, ...]) -> bool:
    return _spoken_word_count(words) >= MIN_CLAUSE_WORDS and _duration(words) >= MIN_CLAUSE_DURATION_SECONDS


def _clause_groups(sentence: tuple[Word, ...], language_code: str | None) -> tuple[tuple[Word, ...], ...]:
    """Choose safe punctuation and language-aware splits without tiny fragments."""
    semicolon_boundaries = (
        index for index, word in enumerate(sentence[:-1], start=1) if SEMICOLON_END_RE.search(word.text.strip())
    )
    comma_boundaries = (
        index for index, word in enumerate(sentence[:-1], start=1) if COMMA_END_RE.search(word.text.strip())
    )
    language_boundaries = connector_boundaries(tuple(word.text for word in sentence), language_code)
    preferred_boundaries = frozenset(semicolon_boundaries)
    boundaries = tuple(sorted({*preferred_boundaries, *comma_boundaries, *language_boundaries}))
    if not boundaries:
        return (sentence,)
    endpoints = (*boundaries, len(sentence))

    def score(groups: tuple[tuple[Word, ...], ...]) -> tuple[int, int, int]:
        position = 0
        semicolon_splits = 0
        for group in groups[:-1]:
            position += len(group)
            semicolon_splits += position in preferred_boundaries
        return semicolon_splits, len(groups), -max(len(Cue(group).text) for group in groups)

    @cache
    def choose(start: int) -> tuple[tuple[Word, ...], ...] | None:
        best: tuple[tuple[Word, ...], ...] | None = None
        for end in endpoints:
            if end <= start:
                continue
            clause = sentence[start:end]
            is_whole_sentence = start == 0 and end == len(sentence)
            if not is_whole_sentence and not _readable_clause(clause):
                continue
            candidate: tuple[tuple[Word, ...], ...]
            if end == len(sentence):
                candidate = (clause,)
            else:
                suffix = choose(end)
                if suffix is None:
                    continue
                candidate = (clause, *suffix)
            if best is None or score(candidate) > score(best):
                best = candidate
        return best

    return choose(0) or (sentence,)


def segment_mini(transcript: Transcript) -> tuple[Cue, ...]:
    """Create sentence cues with safe semicolon, comma, and connector splits."""
    words = transcript.timed_words
    if not words:
        raise MiniSrtError("srt-mini requires word or segment timestamps")

    groups: list[tuple[Word, ...]] = []
    for sentence in _split_sentences(words):
        groups.extend(_clause_groups(sentence, transcript.language_code))
    return cues_with_precise_gaps(groups)
