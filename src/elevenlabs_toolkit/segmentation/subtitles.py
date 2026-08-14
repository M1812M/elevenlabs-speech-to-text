from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace

from ..models import Cue, SegmentationOptions, Transcript, Word
from .pauses import detected_pause_gap, effective_word_end

HARD_END_RE = re.compile(r"[.!?\u2026]+$")
SOFT_END_RE = re.compile(r"[,;:]+$")
DETECTED_PAUSE_GAP = 0.6


def _word_payload(word: Word) -> dict:
    return {
        "type": "word",
        "text": word.text,
        "start": word.start,
        "end": word.end,
        "characters": [
            {"text": character.text, "start": character.start, "end": character.end} for character in word.characters
        ],
    }


def _effective_word(word: Word, pause_detection: bool) -> tuple[Word, bool]:
    payload = _word_payload(word)
    effective_end = effective_word_end(payload, pause_detection=pause_detection)
    pause_after = detected_pause_gap(payload, pause_detection=pause_detection) >= DETECTED_PAUSE_GAP
    return replace(word, end=effective_end), pause_after


def _text(
    words: list[Word] | tuple[Word, ...],
    text_transform: Callable[[str], str] | None = None,
    text_prefix: Callable[[tuple[Word, ...]], str] | None = None,
) -> str:
    word_tuple = tuple(words)
    text = Cue(word_tuple).text if word_tuple else ""
    if text and text_transform is not None:
        text = text_transform(text)
    return f"{text_prefix(word_tuple) if text and text_prefix is not None else ''}{text}"


def _wrapped_line_count(text: str, width: int) -> int:
    words = tuple(text.split())
    if not words:
        return 0
    lines = 1
    current_width = 0
    for word in words:
        candidate_width = len(word) if not current_width else current_width + 1 + len(word)
        if current_width and candidate_width > width:
            lines += 1
            current_width = len(word)
        else:
            current_width = candidate_width
    return lines


def _group_end(words: list[Word] | tuple[Word, ...]) -> float:
    return max(word.end for word in words)


def _fits(
    words: list[Word],
    options: SegmentationOptions,
    text_transform: Callable[[str], str] | None = None,
    text_prefix: Callable[[tuple[Word, ...]], str] | None = None,
) -> bool:
    if not words:
        return True
    if _wrapped_line_count(_text(words, text_transform, text_prefix), options.max_chars_per_line) > options.max_lines:
        return False
    if _group_end(words) - words[0].start > options.max_duration:
        return False
    if options.max_words is not None and len(words) > options.max_words:
        return False
    return True


def _speaker_changed(previous: Word, current: Word, options: SegmentationOptions) -> bool:
    return bool(
        options.split_on_speaker_change and previous.speaker and current.speaker and previous.speaker != current.speaker
    )


def _merge_and_rebalance(
    groups: list[list[Word]],
    options: SegmentationOptions,
    text_transform: Callable[[str], str] | None = None,
    text_prefix: Callable[[tuple[Word, ...]], str] | None = None,
) -> tuple[Cue, ...]:
    merged: list[list[Word]] = []
    for group in groups:
        if merged and (_group_end(group) - group[0].start) < options.min_duration:
            candidate = [*merged[-1], *group]
            speaker_safe = not _speaker_changed(merged[-1][-1], group[0], options)
            boundary_gap = max(0.0, group[0].start - _group_end(merged[-1]))
            if (
                boundary_gap <= options.gap_seconds
                and speaker_safe
                and _fits(candidate, options, text_transform, text_prefix)
            ):
                merged[-1] = candidate
                continue
        merged.append(group)

    rebalanced: list[list[Word]] = []
    for group in merged:
        if rebalanced and len(group) == 1 and len(rebalanced[-1]) >= 3:
            previous = rebalanced[-1]
            candidate = [previous[-1], *group]
            boundary_gap = max(0.0, group[0].start - _group_end(previous))
            if (
                boundary_gap <= options.gap_seconds
                and not _speaker_changed(previous[-1], group[0], options)
                and _fits(candidate, options, text_transform, text_prefix)
            ):
                rebalanced[-1] = previous[:-1]
                rebalanced.append(candidate)
                continue
        rebalanced.append(group)
    return tuple(Cue(tuple(group)) for group in rebalanced if group)


def segment_standard(
    transcript: Transcript,
    options: SegmentationOptions,
    text_transform: Callable[[str], str] | None = None,
    text_prefix: Callable[[tuple[Word, ...]], str] | None = None,
) -> tuple[Cue, ...]:
    prepared = [_effective_word(word, options.pause_detection) for word in transcript.timed_words]
    groups: list[list[Word]] = []
    current: list[Word] = []
    previous_pause = False

    for word, pause_after in prepared:
        if current:
            gap = max(0.0, word.start - _group_end(current))
            if (
                gap > options.gap_seconds
                or (previous_pause and gap >= min(options.gap_seconds, DETECTED_PAUSE_GAP))
                or _speaker_changed(current[-1], word, options)
            ):
                groups.append(current)
                current = []

        if current:
            candidate = [*current, word]
            punctuation_break = len(_text(current, text_transform, text_prefix)) >= 28 and HARD_END_RE.search(
                _text(current, text_transform, text_prefix)
            )
            if not _fits(candidate, options, text_transform, text_prefix) or punctuation_break:
                groups.append(current)
                current = []

        current.append(word)
        previous_pause = pause_after

    if current:
        groups.append(current)
    return _merge_and_rebalance(groups, options, text_transform, text_prefix)


def segment_social(
    transcript: Transcript,
    options: SegmentationOptions,
    text_transform: Callable[[str], str] | None = None,
    text_prefix: Callable[[tuple[Word, ...]], str] | None = None,
) -> tuple[Cue, ...]:
    prepared = [_effective_word(word, options.pause_detection) for word in transcript.timed_words]
    groups: list[list[Word]] = []
    current: list[Word] = []
    previous_pause = False

    for word, pause_after in prepared:
        if current:
            gap = max(0.0, word.start - _group_end(current))
            if (
                gap > options.gap_seconds
                or (previous_pause and gap >= min(options.gap_seconds, DETECTED_PAUSE_GAP))
                or _speaker_changed(current[-1], word, options)
            ):
                groups.append(current)
                current = []

        candidate = [*current, word]
        if current and not _fits(candidate, options, text_transform, text_prefix):
            groups.append(current)
            current = [word]
        else:
            current = candidate

        current_text = _text(current, text_transform, text_prefix)
        if HARD_END_RE.search(current_text) or (
            SOFT_END_RE.search(current_text)
            and (
                len(current_text) >= 20
                or len(current) >= 5
                or current[-1].end - current[0].start >= options.min_duration
            )
        ):
            groups.append(current)
            current = []
        previous_pause = pause_after

    if current:
        groups.append(current)
    return _merge_and_rebalance(groups, options, text_transform, text_prefix)


def segment_transcript(
    transcript: Transcript,
    options: SegmentationOptions,
    text_transform: Callable[[str], str] | None = None,
    text_prefix: Callable[[tuple[Word, ...]], str] | None = None,
) -> tuple[Cue, ...]:
    if options.preset == "social":
        return segment_social(transcript, options, text_transform, text_prefix)
    return segment_standard(transcript, options, text_transform, text_prefix)
