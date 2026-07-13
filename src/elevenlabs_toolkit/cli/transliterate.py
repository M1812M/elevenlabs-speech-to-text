from __future__ import annotations

import argparse
import re
from dataclasses import replace
from pathlib import Path

from ..application import plan_transliteration
from ..files import atomic_write_text, discover_inputs
from ..languages import get_language_processor
from ..models import ArtifactResult, ArtifactStatus, ConflictPolicy, JobResult, ScriptMode
from .common import add_execution_arguments, add_input_arguments, input_spec
from .context import CliContext

MARKUP_RE = re.compile(r"(<[^>]+>|&(?:#[0-9]+|#x[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]+);)")
TIMECODE_RE = re.compile(r"^\s*\d{2}:\d{2}:\d{2}[,.]\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}[,.]\d{3}(?:\s+.*)?$")


def _is_srt_meta_line(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.isdigit() or TIMECODE_RE.match(stripped) is not None


def configure_parser(parser: argparse.ArgumentParser) -> None:
    add_input_arguments(parser, label="SRT")
    parser.add_argument("--to", required=True, choices=[ScriptMode.LATIN.value, ScriptMode.CYRILLIC.value])
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("exports"))
    add_execution_arguments(parser)
    parser.set_defaults(handler=run)


def _convert_srt(source: str, mode: ScriptMode) -> str:
    processor = get_language_processor("uzbek")

    def transform(value: str) -> str:
        return processor.transform_text(value, target=mode)

    lines = []
    for line in source.splitlines():
        if _is_srt_meta_line(line):
            lines.append(line)
            continue
        parts = MARKUP_RE.split(line)
        lines.append("".join(part if MARKUP_RE.fullmatch(part) else transform(part) for part in parts))
    return "\n".join(lines) + ("\n" if source.endswith(("\n", "\r")) else "")


def run(args: argparse.Namespace, context: CliContext) -> int:
    sources = discover_inputs(input_spec(args), {".srt"}, default_glob="*.srt", exclude_generated=False)
    policy = ConflictPolicy(args.on_conflict)
    mode = ScriptMode(args.to)
    plan = plan_transliteration(sources, args.output_dir, mode, policy=policy, dry_run=args.dry_run)
    if args.dry_run or not plan.valid:
        context.emit_plan(plan)
        return 0 if plan.valid else 1
    write_policy = ConflictPolicy.ERROR if policy is ConflictPolicy.RENAME else policy
    results = []
    for artifact in plan.artifacts:
        try:
            content = _convert_srt(artifact.source.read_text(encoding="utf-8-sig"), mode)
            target, status = atomic_write_text(artifact.target, content, write_policy)
            results.append(ArtifactResult(replace(artifact, target=target), status))
        except Exception as exc:
            results.append(ArtifactResult(artifact, ArtifactStatus.FAILED, f"{type(exc).__name__}: {exc}"))
            if args.fail_fast:
                break
    result = JobResult(tuple(results))
    context.emit_result(result)
    return result.exit_code
