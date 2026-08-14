from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable

from ..languages import connector_boundaries
from ..models import SegmentationOptions, Sentence, Transcript, Word
from .subtitles import _effective_word

SENTENCE_END_RE = re.compile(r"[.!?\u2026]+$")


def _text_to_sentences(text: str) -> tuple[str, ...]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return ()
    return tuple(part.strip() for part in re.split(r"(?<=[.!?\u2026])\s+", normalized) if part.strip())


def _dominant_speaker(words: list[Word]) -> str | None:
    durations: dict[str, float] = defaultdict(float)
    order: list[str] = []
    for word in words:
        if not word.speaker:
            continue
        if word.speaker not in durations:
            order.append(word.speaker)
        durations[word.speaker] += word.duration
    if not durations:
        return None
    return max(order, key=lambda speaker: durations[speaker])


def sentences_from_transcript(
    transcript: Transcript,
    options: SegmentationOptions,
    *,
    text_transform: Callable[[str], str] | None = None,
) -> tuple[Sentence, ...]:
    if not transcript.timed_words:
        text = text_transform(transcript.text) if text_transform is not None else transcript.text
        return tuple(Sentence(sentence) for sentence in _text_to_sentences(text))

    result: list[Sentence] = []
    current: list[Word] = []
    previous_pause = False
    connector_breaks = connector_boundaries(
        tuple(word.text for word in transcript.timed_words),
        transcript.language_code,
    )

    def flush() -> None:
        nonlocal current
        if current:
            text = " ".join(word.text for word in current)
            text = re.sub(r"\s+([,.;:!?\u2026])", r"\1", re.sub(r"\s+", " ", text)).strip()
            if text_transform is not None:
                text = text_transform(text)
            if text:
                result.append(Sentence(text, _dominant_speaker(current)))
        current = []

    for index, raw_word in enumerate(transcript.timed_words):
        word, pause_after = _effective_word(raw_word, options.pause_detection)
        if current:
            gap = max(0.0, word.start - max(item.end for item in current))
            if (
                gap >= options.hard_gap_seconds
                or (gap >= options.gap_seconds and index in connector_breaks)
                or (previous_pause and gap >= min(options.gap_seconds, 0.6))
                or (
                    options.split_on_speaker_change
                    and current[-1].speaker
                    and word.speaker
                    and current[-1].speaker != word.speaker
                )
            ):
                flush()
        current.append(word)
        previous_pause = pause_after
        boundary_text = text_transform(word.text) if text_transform is not None else word.text
        if SENTENCE_END_RE.search(boundary_text):
            flush()
    flush()
    return tuple(result)
