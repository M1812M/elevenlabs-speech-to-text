from __future__ import annotations

import argparse
import traceback
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version

from ..application import ExportError, PlanningError, TranscriptionJobError
from ..config import ConfigurationError
from ..files import DiscoveryError, OutputConflictError
from ..models import TranscriptValidationError
from ..providers import ProviderError
from . import clean, config_command, export, inspect, transcribe, transliterate, wizard
from .context import CliContext


def _version() -> str:
    try:
        return version("elevenlabs-toolkit")
    except PackageNotFoundError:
        return "0.0.0+local"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="elevenlabs-toolkit",
        description="Transcribe media and produce safe, reproducible post-production artifacts.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    parser.add_argument(
        "--json", action="store_true", dest="json_output", help="Emit one machine-readable JSON result."
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-q", "--quiet", action="store_true")
    verbosity.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    commands = (
        ("transcribe", "Transcribe local audio/video and cache the provider response.", transcribe.configure_parser),
        ("export", "Render transcript JSON into local formats.", export.configure_parser),
        ("transliterate", "Convert SRT subtitle text between Uzbek scripts.", transliterate.configure_parser),
        ("clean", "Create an explicitly cleaned transcript derivative.", clean.configure_parser),
        ("inspect", "Validate and summarize transcript JSON without writing.", inspect.configure_parser),
        ("config", "Inspect effective configuration and profiles.", config_command.configure_parser),
        ("wizard", "Start an explicit interactive workflow wizard.", wizard.configure_parser),
    )
    for name, help_text, configure in commands:
        command_parser = subparsers.add_parser(name, help=help_text, description=help_text)
        configure(command_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    context = CliContext(
        json_output=args.json_output,
        quiet=args.quiet,
        verbose=args.verbose,
    )
    try:
        if args.command == "wizard":
            return args.handler(args, context, dispatch=lambda nested: main(nested))
        return int(args.handler(args, context))
    except KeyboardInterrupt:
        context.error("interrupted")
        return 130
    except (ConfigurationError, DiscoveryError, PlanningError, TranscriptValidationError, ValueError) as exc:
        context.error(str(exc))
        return 2
    except (ExportError, TranscriptionJobError, OutputConflictError, ProviderError, OSError) as exc:
        context.error(str(exc))
        return 1
    except Exception as exc:  # Defensive command boundary: details require -v.
        if context.verbose:
            traceback.print_exc(file=context.stderr)
        context.error(f"unexpected {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
