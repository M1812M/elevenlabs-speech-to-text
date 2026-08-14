from __future__ import annotations

import math
from dataclasses import dataclass

from .transcript import TranscriptValidationError, Word, join_word_text


@dataclass(frozen=True, slots=True)
class Cue:
    words: tuple[Word, ...]
    start_override: float | None = None
    end_override: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.words, (str, bytes)) or not isinstance(self.words, (tuple, list)) or not self.words:
            raise TranscriptValidationError("cue must contain at least one timed word")
        if not all(isinstance(word, Word) for word in self.words):
            raise TranscriptValidationError("cue words must contain Word values")
        for previous, current in zip(self.words, self.words[1:], strict=False):
            if current.start < previous.start:
                raise TranscriptValidationError("cue words must be ordered by start time")
        object.__setattr__(self, "words", tuple(self.words))
        natural_start = self.words[0].start
        natural_end = max(word.end for word in self.words)
        start = natural_start if self.start_override is None else float(self.start_override)
        end = natural_end if self.end_override is None else float(self.end_override)
        if not math.isfinite(start) or start < 0 or not math.isfinite(end) or end < start:
            raise TranscriptValidationError("cue override timings must be finite and end must be >= start")
        object.__setattr__(self, "start_override", start if self.start_override is not None else None)
        object.__setattr__(self, "end_override", end if self.end_override is not None else None)

    @property
    def start(self) -> float:
        return self.words[0].start if self.start_override is None else self.start_override

    @property
    def end(self) -> float:
        return max(word.end for word in self.words) if self.end_override is None else self.end_override

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def text(self) -> str:
        return join_word_text(self.words)

    @property
    def speaker(self) -> str | None:
        durations: dict[str, float] = {}
        order: list[str] = []
        for word in self.words:
            if not word.speaker:
                continue
            if word.speaker not in durations:
                durations[word.speaker] = 0.0
                order.append(word.speaker)
            durations[word.speaker] += word.duration
        if not durations:
            return None
        return max(order, key=lambda speaker: durations[speaker])
