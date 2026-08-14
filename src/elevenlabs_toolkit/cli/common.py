from __future__ import annotations

import argparse
from pathlib import Path

from ..models import ConflictPolicy, InputSpec

DEFAULT_MEDIA_DIR = Path("media")
_GLOB_CHARACTERS = frozenset("*?[")


def add_input_arguments(parser: argparse.ArgumentParser, *, label: str = "INPUT") -> None:
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        metavar=label,
        help="Input file(s), directories, or one glob path (default: ./media).",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--glob", help="Explicit glob applied inside each input directory.")
    selection.add_argument("--regex", help="Explicit regular expression matched against relative paths.")
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search input directories recursively (default: enabled).",
    )


def input_spec(args: argparse.Namespace) -> InputSpec:
    inputs = list(args.inputs)
    glob = args.glob
    regex = args.regex

    if glob is None and regex is None:
        patterns = [path for path in inputs if any(character in str(path) for character in _GLOB_CHARACTERS)]
        if len(patterns) > 1:
            raise ValueError("pass only one positional glob pattern")
        if patterns:
            pattern = patterns[0]
            inputs.remove(pattern)
            parent = pattern.parent
            if parent != Path("."):
                if any(character in str(parent) for character in _GLOB_CHARACTERS):
                    raise ValueError("glob path directories must not contain wildcard characters")
                if inputs:
                    raise ValueError("use either FOLDER PATTERN or a single FOLDER/PATTERN glob path")
                inputs.append(parent)
                glob = pattern.name
            else:
                glob = str(pattern)

    if not inputs:
        inputs.append(DEFAULT_MEDIA_DIR)
    return InputSpec(tuple(inputs), glob=glob, regex=regex, recursive=args.recursive)


def add_execution_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_policy: str = "error",
    allowed_policies: tuple[ConflictPolicy, ...] = tuple(ConflictPolicy),
) -> None:
    parser.add_argument(
        "--on-conflict",
        choices=[item.value for item in allowed_policies],
        default=default_policy,
        help="How to handle outputs that already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the complete path/conflict/API plan without writing files or calling APIs.",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop the batch after the first failed source.")


def add_srt_timing_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--srt-fps",
        type=float,
        help="Frame rate used for SRT padding (default: 30).",
    )
    parser.add_argument(
        "--srt-padding-frames",
        type=int,
        help="Frames added before and after each SRT cue (default: 2).",
    )
    parser.add_argument(
        "--srt-gap-ms",
        type=int,
        help="Minimum gap between adjacent SRT cues in milliseconds (default: 80).",
    )


def srt_timing_overrides(args: argparse.Namespace) -> dict[str, object]:
    mapping = {
        "srt_fps": args.srt_fps,
        "srt_padding_frames": args.srt_padding_frames,
        "srt_gap_milliseconds": args.srt_gap_ms,
    }
    return {f"segmentation.{name}": value for name, value in mapping.items() if value is not None}
