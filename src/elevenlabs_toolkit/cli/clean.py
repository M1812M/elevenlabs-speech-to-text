from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ..application import execute_export, plan_exports
from ..config import effective_config, profile_options
from ..files import discover_inputs
from ..models import ArtifactFormat, ConflictPolicy, ExportOptions, ScriptMode
from .common import add_execution_arguments, add_input_arguments, input_spec
from .context import CliContext


def configure_parser(parser: argparse.ArgumentParser) -> None:
    add_input_arguments(parser, label="TRANSCRIPT")
    parser.add_argument("--language", choices=["uzbek"], default="uzbek")
    parser.add_argument("--profile", help="Named built-in or configured workflow profile.")
    parser.add_argument("--script", choices=[item.value for item in ScriptMode])
    parser.add_argument("--replace", action="append", default=None, metavar="TOKEN=TOKEN")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("exports"))
    parser.add_argument("--include-generated", action="store_true")
    add_execution_arguments(parser)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace, context: CliContext) -> int:
    sources = discover_inputs(
        input_spec(args), {".json"}, default_glob="*.json", exclude_generated=not args.include_generated
    )
    overrides: dict[str, object] = {}
    if args.script is not None:
        overrides["text.script"] = args.script
    if args.replace is not None:
        overrides["text.replacements"] = args.replace
    config = effective_config(args.profile, overrides=overrides, cwd=Path.cwd())
    segmentation, text = profile_options(config["profile"], config)
    text = replace(text, cleanup=args.language)
    options = ExportOptions((ArtifactFormat.CLEAN_JSON,), args.output_dir, segmentation, text)
    policy = ConflictPolicy(args.on_conflict)
    plan = plan_exports(sources, options, policy=policy, dry_run=args.dry_run)
    if args.dry_run or not plan.valid:
        context.emit_plan(plan)
        return 0 if plan.valid else 1
    result = execute_export(plan, options, policy=policy, fail_fast=args.fail_fast)
    context.emit_result(result)
    return result.exit_code
