from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from ..models import Transcript


class TranscriptSummary(TypedDict, total=False):
    path: str
    valid: bool
    language_code: str | None
    words: int
    segments: int
    speakers: list[str]
    duration: float
    characters: int
    error: str


def inspect_transcript(source: Path) -> TranscriptSummary:
    """Validate one transcript without writing or requiring provider credentials."""
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        transcript = Transcript.from_payload(payload)
        words = transcript.timed_words
        return {
            "path": str(source),
            "valid": True,
            "language_code": transcript.language_code,
            "words": len(transcript.words),
            "segments": len(transcript.segments),
            "speakers": sorted({word.speaker for word in words if word.speaker}),
            "duration": max((word.end for word in words), default=0.0),
            "characters": sum(len(word.characters) for word in words),
        }
    except Exception as exc:
        return {"path": str(source), "valid": False, "error": f"{type(exc).__name__}: {exc}"}
