from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

PUNCTUATION_SPACING_RE = re.compile(r"\s+([,.;:!?\u2026])")


class TranscriptValidationError(ValueError):
    """Raised when a provider transcript cannot be normalized safely."""


def _number(value: Any, field_path: str) -> float:
    if isinstance(value, bool):
        raise TranscriptValidationError(f"{field_path} must be a number, not a boolean")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TranscriptValidationError(f"{field_path} must be a number; got {value!r}") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise TranscriptValidationError(f"{field_path} must be a finite number >= 0; got {parsed}")
    return parsed


def normalize_text(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    return PUNCTUATION_SPACING_RE.sub(r"\1", value)


@dataclass(frozen=True, slots=True)
class CharacterTiming:
    text: str
    start: float
    end: float

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TranscriptValidationError("character text must be a string")
        start = _number(self.start, "character start")
        end = _number(self.end, "character end")
        if end < start:
            raise TranscriptValidationError("character end must be >= start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


def _parse_character_timings(raw_characters: Any, field_path: str) -> tuple[CharacterTiming, ...]:
    if raw_characters is None:
        return ()
    if not isinstance(raw_characters, list):
        raise TranscriptValidationError(f"{field_path} must be an array")
    characters: list[CharacterTiming] = []
    for char_index, character in enumerate(raw_characters):
        char_path = f"{field_path}[{char_index}]"
        if not isinstance(character, Mapping):
            raise TranscriptValidationError(f"{char_path} must be an object")
        char_text = character.get("text", "")
        if char_text is None:
            char_text = ""
        if not isinstance(char_text, str):
            raise TranscriptValidationError(f"{char_path}.text must be a string")
        char_start = _number(character.get("start"), f"{char_path}.start")
        char_end = _number(character.get("end"), f"{char_path}.end")
        if char_end < char_start:
            raise TranscriptValidationError(f"{char_path}.end must be >= {char_path}.start")
        characters.append(CharacterTiming(char_text, char_start, char_end))
    return tuple(characters)


@dataclass(frozen=True, slots=True)
class Word:
    text: str
    start: float
    end: float
    speaker: str | None = None
    characters: tuple[CharacterTiming, ...] = ()
    kind: str = "word"
    source_text: str | None = None
    source_characters: tuple[CharacterTiming, ...] = ()
    characters_from_source: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise TranscriptValidationError("word text must not be empty")
        if self.speaker is not None and not isinstance(self.speaker, str):
            raise TranscriptValidationError("word speaker must be a string or null")
        if isinstance(self.characters, (str, bytes)) or not isinstance(self.characters, (tuple, list)):
            raise TranscriptValidationError("word characters must be a sequence")
        if not all(isinstance(character, CharacterTiming) for character in self.characters):
            raise TranscriptValidationError("word characters must contain CharacterTiming values")
        if not isinstance(self.kind, str) or self.kind.casefold() not in {"word", "audio_event"}:
            raise TranscriptValidationError("word kind must be 'word' or 'audio_event'")
        if self.source_text is not None and not isinstance(self.source_text, str):
            raise TranscriptValidationError("word source_text must be a string or null")
        if isinstance(self.source_characters, (str, bytes)) or not isinstance(self.source_characters, (tuple, list)):
            raise TranscriptValidationError("word source_characters must be a sequence")
        if not all(isinstance(character, CharacterTiming) for character in self.source_characters):
            raise TranscriptValidationError("word source_characters must contain CharacterTiming values")
        if not isinstance(self.characters_from_source, bool):
            raise TranscriptValidationError("word characters_from_source must be a boolean")
        start = _number(self.start, "word start")
        end = _number(self.end, "word end")
        if end < start:
            raise TranscriptValidationError("word end must be >= start")
        characters = tuple(self.characters)
        source_characters = tuple(self.source_characters)
        if self.characters_from_source and (not source_characters or characters != source_characters):
            raise TranscriptValidationError(
                "word characters_from_source requires characters to match source_characters"
            )
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "characters", characters)
        object.__setattr__(self, "kind", self.kind.casefold())
        object.__setattr__(self, "source_characters", source_characters)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def with_text(self, text: str) -> Word:
        return replace(self, text=text)


@dataclass(frozen=True, slots=True)
class Segment:
    text: str
    start: float
    end: float
    speaker: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise TranscriptValidationError("segment text must not be empty")
        if self.speaker is not None and not isinstance(self.speaker, str):
            raise TranscriptValidationError("segment speaker must be a string or null")
        start = _number(self.start, "segment start")
        end = _number(self.end, "segment end")
        if end < start:
            raise TranscriptValidationError("segment end must be >= start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def as_word(self) -> Word:
        return Word(text=self.text, start=self.start, end=self.end, speaker=self.speaker)


@dataclass(frozen=True, slots=True)
class Sentence:
    text: str
    speaker: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise TranscriptValidationError("sentence text must not be empty")
        if self.speaker is not None and not isinstance(self.speaker, str):
            raise TranscriptValidationError("sentence speaker must be a string or null")


def join_word_text(words: Sequence[Word]) -> str:
    return normalize_text(" ".join(word.text for word in words))


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    words: tuple[Word, ...] = ()
    segments: tuple[Segment, ...] = ()
    language_code: str | None = None
    provider: str = "elevenlabs"
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    raw_payload: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TranscriptValidationError("transcript text must be a string")
        if isinstance(self.words, (str, bytes)) or not isinstance(self.words, (tuple, list)):
            raise TranscriptValidationError("transcript words must be a sequence")
        if isinstance(self.segments, (str, bytes)) or not isinstance(self.segments, (tuple, list)):
            raise TranscriptValidationError("transcript segments must be a sequence")
        if not all(isinstance(word, Word) for word in self.words):
            raise TranscriptValidationError("transcript words must contain Word values")
        if not all(isinstance(segment, Segment) for segment in self.segments):
            raise TranscriptValidationError("transcript segments must contain Segment values")
        if any(current.start < previous.start for previous, current in zip(self.words, self.words[1:], strict=False)):
            raise TranscriptValidationError("transcript words must be ordered by start time")
        if any(
            current.start < previous.start for previous, current in zip(self.segments, self.segments[1:], strict=False)
        ):
            raise TranscriptValidationError("transcript segments must be ordered by start time")
        if self.language_code is not None and not isinstance(self.language_code, str):
            raise TranscriptValidationError("language_code must be a string or null")
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise TranscriptValidationError("provider must be a non-empty string")
        if not isinstance(self.metadata, Mapping) or not isinstance(self.raw_payload, Mapping):
            raise TranscriptValidationError("transcript metadata and raw_payload must be mappings")
        object.__setattr__(self, "text", normalize_text(self.text))
        object.__setattr__(self, "words", tuple(self.words))
        object.__setattr__(self, "segments", tuple(self.segments))
        object.__setattr__(self, "provider", self.provider.strip())

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], *, provider: str = "elevenlabs") -> Transcript:
        if not isinstance(payload, Mapping):
            raise TranscriptValidationError(f"transcript payload must be an object; got {type(payload).__name__}")
        if not any(key in payload for key in ("text", "words", "segments")):
            raise TranscriptValidationError("transcript must contain text, words, or segments")

        words, word_text = cls._parse_words(payload.get("words"))
        segments = cls._parse_segments(payload.get("segments"))
        raw_text = payload.get("text")
        if raw_text is not None and not isinstance(raw_text, str):
            raise TranscriptValidationError("text must be a string")
        normalized_raw_text = normalize_text(raw_text or "")
        segment_text = normalize_text(" ".join(segment.text for segment in segments))
        if raw_text and not normalized_raw_text and not word_text and not segment_text:
            raise TranscriptValidationError("text must not contain only whitespace")
        text = normalized_raw_text or word_text or segment_text
        language_code = payload.get("language_code")
        if language_code is not None and not isinstance(language_code, str):
            raise TranscriptValidationError("language_code must be a string or null")

        excluded = {"text", "words", "segments", "language_code"}
        metadata = {key: copy.deepcopy(value) for key, value in payload.items() if key not in excluded}
        return cls(
            text=text,
            words=words,
            segments=segments,
            language_code=language_code,
            provider=provider,
            metadata=metadata,
            raw_payload=copy.deepcopy(dict(payload)),
        )

    @staticmethod
    def _parse_words(raw_words: Any) -> tuple[tuple[Word, ...], str]:
        if raw_words is None:
            return (), ""
        if not isinstance(raw_words, list):
            raise TranscriptValidationError("words must be an array")

        parsed: list[Word] = []
        display_tokens: list[str] = []
        pending_prefix = ""
        last_timed_index: int | None = None
        saw_timed = False
        saw_untimed = False
        for index, item in enumerate(raw_words):
            path = f"words[{index}]"
            if not isinstance(item, Mapping):
                raise TranscriptValidationError(f"{path} must be an object")
            raw_type = item.get("type", "word")
            token_text = item.get("text", "")
            if raw_type is None:
                raw_type = "word"
            if token_text is None:
                token_text = ""
            if not isinstance(raw_type, str) or not raw_type.strip():
                raise TranscriptValidationError(f"{path}.type must be a string")
            if not isinstance(token_text, str):
                raise TranscriptValidationError(f"{path}.text must be a string")
            token_type = raw_type.lower()
            if token_type in {"spacing", "punctuation"}:
                token_text = token_text.strip()
                if not token_text:
                    continue
                if display_tokens:
                    display_tokens[-1] += token_text
                else:
                    pending_prefix += token_text
                if last_timed_index is not None:
                    parsed[last_timed_index] = parsed[last_timed_index].with_text(
                        parsed[last_timed_index].text + token_text
                    )
                continue
            if token_type not in {"word", "audio_event"}:
                continue
            if not token_text.strip():
                raise TranscriptValidationError(f"{path}.text must not be empty")

            display_text = pending_prefix + token_text
            display_tokens.append(display_text)
            pending_prefix = ""

            raw_start = item.get("start")
            raw_end = item.get("end")
            if raw_start is None and raw_end is None:
                if token_type == "word":
                    saw_untimed = True
                last_timed_index = None
            elif raw_start is None or raw_end is None:
                missing = "start" if raw_start is None else "end"
                raise TranscriptValidationError(f"{path}.{missing} must be present when the word is timed")
            else:
                saw_timed = True

            active_characters = _parse_character_timings(item.get("characters"), f"{path}.characters")
            source_characters = _parse_character_timings(item.get("source_characters"), f"{path}.source_characters")
            characters_from_source = not active_characters and bool(source_characters)
            timing_characters = active_characters or source_characters

            source_text = item.get("source_text")
            if source_text is not None and not isinstance(source_text, str):
                raise TranscriptValidationError(f"{path}.source_text must be a string or null")

            if raw_start is None and raw_end is None:
                continue

            start = _number(raw_start, f"{path}.start")
            end = _number(raw_end, f"{path}.end")
            if end < start:
                raise TranscriptValidationError(f"{path}.end must be >= {path}.start")

            speaker = item.get("speaker_id", item.get("speaker"))
            parsed.append(
                Word(
                    text=display_text,
                    start=start,
                    end=end,
                    speaker=str(speaker) if speaker is not None else None,
                    characters=timing_characters,
                    kind=token_type,
                    source_text=source_text,
                    source_characters=source_characters,
                    characters_from_source=characters_from_source,
                )
            )
            last_timed_index = len(parsed) - 1

        if saw_timed and saw_untimed:
            raise TranscriptValidationError("word timestamps must be present for either every word or no words")
        return tuple(parsed), normalize_text(" ".join(display_tokens))

    @staticmethod
    def _parse_segments(raw_segments: Any) -> tuple[Segment, ...]:
        if raw_segments is None:
            return ()
        if not isinstance(raw_segments, list):
            raise TranscriptValidationError("segments must be an array")
        parsed: list[Segment] = []
        for index, item in enumerate(raw_segments):
            path = f"segments[{index}]"
            if not isinstance(item, Mapping):
                raise TranscriptValidationError(f"{path} must be an object")
            raw_text = item.get("text", "")
            if raw_text is None:
                raw_text = ""
            if not isinstance(raw_text, str):
                raise TranscriptValidationError(f"{path}.text must be a string")
            text = raw_text.strip()
            if not text:
                continue
            if item.get("start") is None or item.get("end") is None:
                raise TranscriptValidationError(f"{path} must contain start and end timestamps")
            start = _number(item.get("start"), f"{path}.start")
            end = _number(item.get("end"), f"{path}.end")
            if end < start:
                raise TranscriptValidationError(f"{path}.end must be >= {path}.start")
            speaker = item.get("speaker_id", item.get("speaker"))
            parsed.append(Segment(text, start, end, str(speaker) if speaker is not None else None))
        return tuple(parsed)

    @property
    def timed_words(self) -> tuple[Word, ...]:
        if self.words:
            return self.words
        return tuple(segment.as_word() for segment in self.segments)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = dict(self.metadata)
        payload.update(
            {
                "text": self.text,
                "language_code": self.language_code,
                "words": [
                    {
                        "type": word.kind,
                        "text": word.text,
                        "start": word.start,
                        "end": word.end,
                        **({"speaker_id": word.speaker} if word.speaker else {}),
                        **({"source_text": word.source_text} if word.source_text is not None else {}),
                        **(
                            {
                                "characters": [
                                    {"text": char.text, "start": char.start, "end": char.end}
                                    for char in word.characters
                                ]
                            }
                            if word.characters and not word.characters_from_source
                            else {}
                        ),
                        **(
                            {
                                "source_characters": [
                                    {"text": char.text, "start": char.start, "end": char.end}
                                    for char in word.source_characters
                                ]
                            }
                            if word.source_characters
                            else {}
                        ),
                    }
                    for word in self.words
                ],
                "segments": [
                    {
                        "text": segment.text,
                        "start": segment.start,
                        "end": segment.end,
                        **({"speaker_id": segment.speaker} if segment.speaker else {}),
                    }
                    for segment in self.segments
                ],
            }
        )
        return {key: value for key, value in payload.items() if value is not None}
