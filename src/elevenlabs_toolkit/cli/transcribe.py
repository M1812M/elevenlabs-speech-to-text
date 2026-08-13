from __future__ import annotations

import argparse
import math
from pathlib import Path

from ..application import execute_transcription, plan_transcription
from ..config import effective_config, profile_options
from ..files import AUDIO_VIDEO_SUFFIXES, discover_inputs
from ..models import (
    ArtifactFormat,
    ConflictPolicy,
    ExportOptions,
    ScriptMode,
    SpeakerLabels,
    TranscriptionOptions,
)
from ..providers import ElevenLabsProvider
from .common import add_execution_arguments, add_input_arguments, input_spec
from .context import CliContext

LOCAL_FORMATS = (
    ArtifactFormat.SRT,
    ArtifactFormat.TXT,
    ArtifactFormat.SOCIAL_SRT,
    ArtifactFormat.RESOLVE_EDL,
    ArtifactFormat.CUE_INDEX_SRT,
)
REMOTE_FORMATS = ("pdf", "docx", "html", "segmented-json")


def configure_parser(parser: argparse.ArgumentParser) -> None:
    add_input_arguments(parser, label="MEDIA")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("media"), help="Output root (default: ./media).")
    parser.add_argument(
        "--format",
        action="append",
        choices=[item.value for item in LOCAL_FORMATS],
        dest="formats",
        help="Additional local output format; JSON is always written.",
    )
    parser.add_argument("--remote-format", action="append", choices=REMOTE_FORMATS, default=[])
    parser.add_argument("--profile")
    parser.add_argument("--script", choices=[item.value for item in ScriptMode])
    parser.add_argument("--clean", choices=["none", "uzbek"])
    parser.add_argument("--speaker-labels", choices=[item.value for item in SpeakerLabels])
    parser.add_argument("--replace", action="append", default=None, metavar="TOKEN=TOKEN")

    parser.add_argument("--model", default="scribe_v2")
    parser.add_argument("--language-code")
    parser.add_argument("--timestamps", choices=["none", "word", "character"])
    parser.add_argument("--diarize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--audio-events", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--keyterm", action="append", default=[])
    parser.add_argument(
        "--no-verbatim", action="store_true", help="Ask the provider to remove fillers and false starts."
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Dotenv file (default: ./.env).")

    parser.add_argument("--pause-detection", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Additional transient-error attempts; each may incur another provider charge (default: 0).",
    )
    parser.add_argument("--retry-backoff", type=float, default=1.0)
    parser.add_argument("--request-delay", type=float, default=0.0)
    add_execution_arguments(
        parser,
        default_policy="replace",
        allowed_policies=(ConflictPolicy.ERROR, ConflictPolicy.SKIP, ConflictPolicy.REPLACE),
    )
    parser.set_defaults(handler=run)


def _build_options(args: argparse.Namespace) -> tuple[TranscriptionOptions, ExportOptions, tuple[ArtifactFormat, ...]]:
    requested_formats = (ArtifactFormat(item) for item in (args.formats or ()))
    formats = tuple(dict.fromkeys((ArtifactFormat.JSON, *requested_formats)))
    timed_formats = {
        ArtifactFormat.SRT,
        ArtifactFormat.SOCIAL_SRT,
        ArtifactFormat.RESOLVE_EDL,
        ArtifactFormat.CUE_INDEX_SRT,
    }
    overrides: dict[str, object] = {}
    if args.script is not None:
        overrides["text.script"] = args.script
    if args.clean is not None:
        overrides["text.cleanup"] = None if args.clean == "none" else args.clean
    if args.speaker_labels is not None:
        overrides["text.speaker_labels"] = args.speaker_labels
    if args.replace is not None:
        overrides["text.replacements"] = args.replace
    if args.pause_detection is not None:
        overrides["segmentation.pause_detection"] = args.pause_detection
    config = effective_config(args.profile, overrides=overrides, cwd=Path.cwd())
    segmentation, text = profile_options(config["profile"], config)
    timestamps = args.timestamps or ("character" if segmentation.pause_detection else "word")
    if timestamps == "none" and any(item in timed_formats for item in formats):
        raise ValueError("timed local formats require --timestamps word or character")
    if segmentation.pause_detection and timestamps != "character":
        raise ValueError("pause detection requires --timestamps character")
    pacing = {
        "retry backoff": args.retry_backoff,
        "request delay": args.request_delay,
    }
    if args.retries < 0:
        raise ValueError("retries must be >= 0")
    for label, value in pacing.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{label} must be a finite number >= 0")

    transcription = TranscriptionOptions(
        model_id=args.model,
        language_code=args.language_code,
        timestamps_granularity=timestamps,
        diarize=args.diarize,
        tag_audio_events=args.audio_events,
        num_speakers=args.num_speakers,
        keyterms=tuple(args.keyterm),
        no_verbatim=args.no_verbatim,
        seed=args.seed,
        temperature=args.temperature,
        remote_formats=tuple(args.remote_format),
    )
    export = ExportOptions(formats, args.output_dir, segmentation, text)
    return transcription, export, formats


def run(args: argparse.Namespace, context: CliContext) -> int:
    sources = discover_inputs(input_spec(args), set(AUDIO_VIDEO_SUFFIXES), exclude_generated=False)
    transcription, export, formats = _build_options(args)
    policy = ConflictPolicy(args.on_conflict)
    plan = plan_transcription(
        sources,
        args.output_dir,
        formats,
        policy=policy,
        dry_run=args.dry_run,
        transcription_options=transcription,
    )
    if args.dry_run or not plan.valid:
        context.emit_plan(plan, max_api_attempts=plan.api_requests * (args.retries + 1))
        return 0 if plan.valid else 1

    provider = ElevenLabsProvider(env_file=args.env_file) if plan.api_requests else None
    result = execute_transcription(
        plan,
        transcription,
        export,
        provider=provider,
        policy=policy,
        retries=args.retries,
        backoff_seconds=args.retry_backoff,
        request_delay=args.request_delay,
        fail_fast=args.fail_fast,
        progress=context.log,
    )
    context.emit_result(result)
    return result.exit_code
