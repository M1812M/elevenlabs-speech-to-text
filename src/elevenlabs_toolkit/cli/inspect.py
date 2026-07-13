from __future__ import annotations

import argparse
import json
from typing import TypedDict

from ..files import discover_inputs
from ..models import Transcript
from .common import add_input_arguments, input_spec
from .context import CliContext


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


def configure_parser(parser: argparse.ArgumentParser) -> None:
    add_input_arguments(parser, label="TRANSCRIPT")
    parser.add_argument("--include-generated", action="store_true")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace, context: CliContext) -> int:
    sources = discover_inputs(
        input_spec(args), {".json"}, default_glob="*.json", exclude_generated=not args.include_generated
    )
    summaries: list[TranscriptSummary] = []
    failed = 0
    for source in sources:
        try:
            payload = json.loads(source.read_text(encoding="utf-8-sig"))
            transcript = Transcript.from_payload(payload)
            words = transcript.timed_words
            speakers = sorted({word.speaker for word in words if word.speaker})
            summaries.append(
                {
                    "path": str(source),
                    "valid": True,
                    "language_code": transcript.language_code,
                    "words": len(transcript.words),
                    "segments": len(transcript.segments),
                    "speakers": speakers,
                    "duration": max((word.end for word in words), default=0.0),
                    "characters": sum(len(word.characters) for word in words),
                }
            )
        except Exception as exc:
            failed += 1
            summaries.append({"path": str(source), "valid": False, "error": f"{type(exc).__name__}: {exc}"})
    if context.json_output:
        context.emit({"status": "failed" if failed else "ok", "transcripts": summaries})
    else:
        for item in summaries:
            if item["valid"]:
                context.emit(
                    f"{item['path']}: {item['words']} words, {item['segments']} segments, "
                    f"{item['duration']:.2f}s, {len(item['speakers'])} speaker(s), language={item['language_code'] or 'unknown'}"
                )
            else:
                context.emit(f"{item['path']}: INVALID - {item['error']}")
    return 1 if failed else 0
