from __future__ import annotations

import re
from collections.abc import Callable
from functools import cache

from ..languages import connector_boundaries, lexical_tokens
from ..models import Cue, Transcript, Word
from .timings import (
    DEFAULT_PADDING_FRAMES,
    DEFAULT_SRT_FPS,
    MIN_GAP_MILLISECONDS,
    cues_with_precise_gaps,
    spoken_end,
    spoken_start,
)

SENTENCE_END_RE = re.compile(r"[.!?\u2026]+(?:[\"'\u2019\u00bb)]*)$")
SEMICOLON_END_RE = re.compile(r";(?:[\"'\u2019\u00bb)]*)$")
COMMA_END_RE = re.compile(r",(?:[\"'\u2019\u00bb)]*)$")

MAX_CUE_CHARACTERS = 80
PAUSE_SPLIT_SECONDS = 1.0
MIN_SPLIT_WORDS = 2


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


def _rendered_text(
    words: tuple[Word, ...],
    text_transform: Callable[[str], str] | None,
) -> str:
    text = Cue(words).text
    if text_transform is not None:
        text = text_transform(text)
    return text


def _split_on_pauses(words: tuple[Word, ...]) -> tuple[tuple[Word, ...], ...]:
    boundaries = tuple(
        index
        for index in range(1, len(words))
        if spoken_start(words[index]) - spoken_end(words[index - 1]) > PAUSE_SPLIT_SECONDS
    )
    if not boundaries:
        return (words,)
    endpoints = (*boundaries, len(words))

    @cache
    def choose(start: int) -> tuple[tuple[Word, ...], ...] | None:
        best: tuple[tuple[Word, ...], ...] | None = None
        for end in endpoints:
            if end <= start:
                continue
            group = words[start:end]
            is_whole_sentence = start == 0 and end == len(words)
            if not is_whole_sentence and _spoken_word_count(group) < MIN_SPLIT_WORDS:
                continue
            candidate: tuple[tuple[Word, ...], ...]
            if end == len(words):
                candidate = (group,)
            else:
                suffix = choose(end)
                if suffix is None:
                    continue
                candidate = (group, *suffix)
            if best is None or (len(candidate), -max(len(Cue(item).text) for item in candidate)) > (
                len(best),
                -max(len(Cue(item).text) for item in best),
            ):
                best = candidate
        return best

    return choose(0) or (words,)


def _length_groups(
    words: tuple[Word, ...],
    language_code: str | None,
    text_transform: Callable[[str], str] | None,
) -> tuple[tuple[Word, ...], ...]:
    """Recursively split overlong cues at the highest-priority readable boundary."""
    if len(_rendered_text(words, text_transform)) <= MAX_CUE_CHARACTERS:
        return (words,)

    semicolon_boundaries = frozenset(
        index for index, word in enumerate(words[:-1], start=1) if SEMICOLON_END_RE.search(word.text.strip())
    )
    comma_boundaries = frozenset(
        index for index, word in enumerate(words[:-1], start=1) if COMMA_END_RE.search(word.text.strip())
    )
    language_boundaries = connector_boundaries(tuple(word.text for word in words), language_code)

    for boundaries in (semicolon_boundaries, comma_boundaries, language_boundaries):
        readable = tuple(
            boundary
            for boundary in boundaries
            if _spoken_word_count(words[:boundary]) >= MIN_SPLIT_WORDS
            and _spoken_word_count(words[boundary:]) >= MIN_SPLIT_WORDS
        )
        if not readable:
            continue
        boundary = min(
            readable,
            key=lambda item: (
                max(
                    len(_rendered_text(words[:item], text_transform)),
                    len(_rendered_text(words[item:], text_transform)),
                ),
                abs(
                    len(_rendered_text(words[:item], text_transform))
                    - len(_rendered_text(words[item:], text_transform))
                ),
                item,
            ),
        )
        return (
            *_length_groups(words[:boundary], language_code, text_transform),
            *_length_groups(words[boundary:], language_code, text_transform),
        )
    return (words,)


def segment_mini(
    transcript: Transcript,
    text_transform: Callable[[str], str] | None = None,
    *,
    srt_fps: float = DEFAULT_SRT_FPS,
    srt_padding_frames: int = DEFAULT_PADDING_FRAMES,
    srt_gap_milliseconds: int = MIN_GAP_MILLISECONDS,
) -> tuple[Cue, ...]:
    """Create sentence cues with safe semicolon, comma, and connector splits."""
    words = transcript.timed_words
    if not words:
        raise MiniSrtError("srt-mini requires word or segment timestamps")

    groups: list[tuple[Word, ...]] = []
    for sentence in _split_sentences(words):
        for pause_group in _split_on_pauses(sentence):
            groups.extend(_length_groups(pause_group, transcript.language_code, text_transform))
    return cues_with_precise_gaps(
        groups,
        frames_per_second=srt_fps,
        padding_frames=srt_padding_frames,
        minimum_gap_milliseconds=srt_gap_milliseconds,
    )
