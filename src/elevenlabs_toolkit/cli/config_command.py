from __future__ import annotations

import argparse
from pathlib import Path

from ..config import available_profiles, effective_config
from .context import CliContext


def configure_parser(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="config_action")
    show = subparsers.add_parser("show", help="Show effective configuration.")
    show.add_argument("--profile")
    show.set_defaults(handler=run_show)


def run_show(args: argparse.Namespace, context: CliContext) -> int:
    config = effective_config(args.profile, cwd=Path.cwd())
    payload = {"effective": config, "available_profiles": list(available_profiles(cwd=Path.cwd()))}
    context.emit(payload)
    return 0
