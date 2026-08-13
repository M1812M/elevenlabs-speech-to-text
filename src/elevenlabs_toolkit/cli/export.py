from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ..application import execute_export, plan_exports
from ..config import effective_config, profile_options
from ..files import discover_inputs
from ..models import (
    ArtifactFormat,
    ConflictPolicy,
    ExportOptions,
    ScriptMode,
    SpeakerLabels,
)
from .common import add_execution_arguments, add_input_arguments, input_spec
from .context import CliContext

EXPORT_FORMATS = (
    ArtifactFormat.SRT,
    ArtifactFormat.TXT,
    ArtifactFormat.COMBINED_TXT,
    ArtifactFormat.SOCIAL_SRT,
    ArtifactFormat.RESOLVE_EDL,
    ArtifactFormat.CUE_INDEX_SRT,
    ArtifactFormat.CLEAN_JSON,
)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    add_input_arguments(parser, label="TRANSCRIPT")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("media"), help="Output root (default: ./media).")
    parser.add_argument(
        "--format",
        action="append",
        choices=[item.value for item in EXPORT_FORMATS],
        dest="formats",
        help="Repeatable output format (default: srt).",
    )
    parser.add_argument("--profile", help="Named built-in or configured workflow profile.")
    parser.add_argument(
        "--script", choices=[item.value for item in ScriptMode], help="Output script; source preserves input."
    )
    parser.add_argument("--clean", choices=["none", "uzbek"], help="Explicit editorial cleanup profile.")
    parser.add_argument("--speaker-labels", choices=[item.value for item in SpeakerLabels])
    parser.add_argument(
        "--replace",
        action="append",
        default=None,
        metavar="TOKEN=TOKEN",
        help="Repeatable literal single-token text replacement.",
    )
    parser.add_argument("--pause-detection", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-chars-per-line", type=int)
    parser.add_argument("--max-lines", type=int)
    parser.add_argument("--max-duration", type=float)
    parser.add_argument("--marker-fps", type=float, default=25.0)
    parser.add_argument("--marker-color", default="ResolveColorBlue")
    parser.add_argument("--marker-prefix", default="Sentence")
    parser.add_argument("--combined-name", default="combined.txt")
    parser.add_argument("--include-generated", action="store_true", help="Allow generated JSON artifacts as inputs.")
    add_execution_arguments(parser)
    parser.set_defaults(handler=run)


def _options(args: argparse.Namespace) -> ExportOptions:
    overrides: dict[str, object] = {}
    for name in ("max_chars_per_line", "max_lines", "max_duration", "pause_detection"):
        value = getattr(args, name)
        if value is not None:
            overrides[f"segmentation.{name}"] = value
    if args.script is not None:
        overrides["text.script"] = args.script
    if args.clean is not None:
        overrides["text.cleanup"] = None if args.clean == "none" else args.clean
    if args.speaker_labels is not None:
        overrides["text.speaker_labels"] = args.speaker_labels
    if args.replace is not None:
        overrides["text.replacements"] = args.replace

    config = effective_config(args.profile, overrides=overrides, cwd=Path.cwd())
    segmentation, text = profile_options(config["profile"], config)
    formats = tuple(ArtifactFormat(item) for item in (args.formats or [ArtifactFormat.SRT.value]))
    if ArtifactFormat.CLEAN_JSON in formats and args.clean == "none":
        raise ValueError("clean-json cannot be combined with --clean none")
    if ArtifactFormat.CLEAN_JSON in formats and text.cleanup is None:
        text = replace(text, cleanup="uzbek")
    return ExportOptions(
        formats=formats,
        output_dir=args.output_dir,
        segmentation=segmentation,
        text=text,
        marker_fps=args.marker_fps,
        marker_color=args.marker_color,
        marker_prefix=args.marker_prefix,
    )


def run(args: argparse.Namespace, context: CliContext) -> int:
    sources = discover_inputs(
        input_spec(args),
        {".json"},
        default_glob="*.json",
        exclude_generated=not args.include_generated,
    )
    options = _options(args)
    policy = ConflictPolicy(args.on_conflict)
    plan = plan_exports(
        sources,
        options,
        policy=policy,
        dry_run=args.dry_run,
        combined_name=args.combined_name,
    )
    if args.dry_run or not plan.valid:
        context.emit_plan(plan)
        return 0 if plan.valid else 1
    result = execute_export(plan, options, policy=policy, fail_fast=args.fail_fast)
    context.emit_result(result)
    return result.exit_code
