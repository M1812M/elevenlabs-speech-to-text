from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from .context import CliContext


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(handler=run)


def run(
    _args: argparse.Namespace,
    context: CliContext,
    dispatch: Callable[[list[str]], int] | None = None,
) -> int:
    if not sys.stdin.isatty():
        context.error("wizard requires an interactive terminal")
        return 2
    command = input("Workflow [transcribe/export]: ").strip().casefold()
    if command not in {"transcribe", "export"}:
        context.error("workflow must be transcribe or export")
        return 2
    source = input("Input file or directory: ").strip().strip('"')
    if not source:
        context.error("input is required")
        return 2
    default_output = "artifacts" if command == "transcribe" else "exports"
    output = input(f"Output directory [{default_output}]: ").strip().strip('"') or default_output
    default_format = "json" if command == "transcribe" else "srt"
    formats = input(f"Formats, comma-separated [{default_format}]: ").strip() or default_format
    argv = [command, source, "--output-dir", output]
    for item in formats.split(","):
        if item.strip():
            argv.extend(("--format", item.strip()))
    if dispatch is None:
        from .main import main

        dispatch = main
    preview_code = dispatch([*argv, "--dry-run"])
    if preview_code:
        return preview_code
    confirm = input("Run this plan? [y/N]: ").strip().casefold()
    if confirm not in {"y", "yes"}:
        context.log("Cancelled; no work was performed.")
        return 0
    return dispatch(argv)
