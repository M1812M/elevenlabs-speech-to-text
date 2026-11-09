from __future__ import annotations

import argparse

from ..application.inspection import inspect_transcript
from ..files import discover_inputs
from .common import add_input_arguments, input_spec
from .context import CliContext


def configure_parser(parser: argparse.ArgumentParser) -> None:
    add_input_arguments(parser, label="TRANSCRIPT")
    parser.add_argument("--include-generated", action="store_true")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace, context: CliContext) -> int:
    sources = discover_inputs(
        input_spec(args), {".json"}, default_glob="*.json", exclude_generated=not args.include_generated
    )
    summaries = [inspect_transcript(source) for source in sources]
    failed = sum(not item["valid"] for item in summaries)
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
