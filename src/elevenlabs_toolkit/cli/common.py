from __future__ import annotations

import argparse
from pathlib import Path

from ..models import ConflictPolicy, InputSpec


def add_input_arguments(parser: argparse.ArgumentParser, *, label: str = "INPUT") -> None:
    parser.add_argument("inputs", nargs="+", type=Path, metavar=label, help="Existing input file(s) or directories.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--glob", help="Explicit glob applied inside each input directory.")
    selection.add_argument("--regex", help="Explicit regular expression matched against relative paths.")
    parser.add_argument("--recursive", action="store_true", help="Search input directories recursively.")


def input_spec(args: argparse.Namespace) -> InputSpec:
    return InputSpec(tuple(args.inputs), glob=args.glob, regex=args.regex, recursive=args.recursive)


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
