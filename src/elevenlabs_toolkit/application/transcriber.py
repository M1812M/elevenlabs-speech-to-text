from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from ..files import (
    atomic_write_bytes,
    atomic_write_text,
    ensure_atomic_no_clobber_supported,
)
from ..models import (
    ArtifactFormat,
    ArtifactResult,
    ArtifactStatus,
    ConflictPolicy,
    ExportOptions,
    JobPlan,
    JobResult,
    PlannedArtifact,
    Transcript,
    TranscriptionOptions,
)
from ..providers import ProviderTransientError, SpeechToTextProvider, decode_additional_formats
from .exporter import render_artifact
from .planner import PlanningError, _safe_leaf_name

REMOTE_FORMATS = {
    ArtifactFormat.PDF,
    ArtifactFormat.DOCX,
    ArtifactFormat.HTML,
    ArtifactFormat.SEGMENTED_JSON,
}


class TranscriptionJobError(RuntimeError):
    """Raised when a planned transcription cannot be executed safely."""


def _finite_nonnegative(value: float, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number >= 0")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number >= 0") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{label} must be a finite number >= 0")
    return parsed


def _validate_planned_names(plan: JobPlan) -> None:
    for source in plan.sources:
        if not _safe_leaf_name(source.stem):
            raise PlanningError(f"transcription source stem is not a portable output name: {source.name!r}")
    for output in plan.artifacts:
        if not _safe_leaf_name(output.target.name):
            raise PlanningError(f"generated transcription output name is unsafe: {output.target.name!r}")


def _artifact_map(plan: JobPlan, source: Path) -> dict[ArtifactFormat, PlannedArtifact]:
    return {output.format: output for output in plan.artifacts if output.source == source}


def _path_occupied(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _source_snapshot(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    if not path.is_file():
        raise TranscriptionJobError(f"transcription input must be a regular file: {path}")
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _file_size_text(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise RuntimeError("unreachable")


def _elapsed_text(seconds: float) -> str:
    total = max(0, round(seconds))
    minutes, seconds = divmod(total, 60)
    return f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"


def _provider_label(value: str | None) -> str:
    if value and value.casefold() == "elevenlabs":
        return "ElevenLabs"
    return value or "speech-to-text provider"


def _prepare_payload(
    payload: dict,
    source: Path,
    options: TranscriptionOptions,
) -> tuple[Transcript, dict[str, bytes | str]]:
    if not any(key in payload for key in ("text", "words", "segments")):
        raise TranscriptionJobError("provider response is not a synchronous transcript payload")
    transcript = Transcript.from_payload(payload)
    if options.timestamps_granularity != "none" and transcript.text and not transcript.timed_words:
        raise TranscriptionJobError(
            f"provider response contains text but no {options.timestamps_granularity} timestamps"
        )
    if options.timestamps_granularity == "character" and any(
        word.kind == "word" and not word.characters for word in transcript.timed_words
    ):
        raise TranscriptionJobError("provider response does not contain character timestamps for every spoken word")
    decoded = {name.casefold(): content for name, content in decode_additional_formats(payload, source.stem)}
    return transcript, decoded


def _prepare_outputs(
    outputs: dict[ArtifactFormat, PlannedArtifact],
    transcript: Transcript,
    payload: dict,
    decoded: dict[str, bytes | str],
    export_options: ExportOptions,
    policy: ConflictPolicy,
) -> dict[ArtifactFormat, bytes | str | None]:
    prepared: dict[ArtifactFormat, bytes | str | None] = {}
    for output_format, output in outputs.items():
        if output_format is ArtifactFormat.JSON:
            prepared[output_format] = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        elif output_format in REMOTE_FORMATS:
            content = decoded.get(output.target.name.casefold())
            if content is None:
                raise TranscriptionJobError(
                    f"ElevenLabs response did not contain requested remote format {output_format.value}"
                )
            prepared[output_format] = content
        elif policy is ConflictPolicy.SKIP and _path_occupied(output.target):
            prepared[output_format] = None
        else:
            prepared[output_format] = render_artifact(output_format, transcript, payload, export_options)
    return prepared


def _request_with_retry(
    provider: SpeechToTextProvider,
    source: Path,
    options: TranscriptionOptions,
    *,
    retries: int,
    backoff_seconds: float,
    progress: Callable[[str], None] | None,
    status_label: str,
    provider_label: str,
) -> dict:
    for attempt in range(retries + 1):
        try:
            if progress:
                attempt_text = f" (attempt {attempt + 1}/{retries + 1})" if retries else ""
                progress(f"{status_label} - uploading + transcribing with {provider_label}{attempt_text}")
            return provider.transcribe(source, options)
        except ProviderTransientError as exc:
            if attempt >= retries:
                raise
            delay = (
                exc.retry_after_seconds
                if exc.retry_after_seconds is not None
                else max(0.0, backoff_seconds) * (2**attempt)
            )
            if progress:
                progress(
                    f"{status_label} - temporary provider error; "
                    f"retry {attempt + 2}/{retries + 1} in {delay:.1f}s: {exc}"
                )
            if delay:
                time.sleep(delay)
    raise RuntimeError("unreachable")


def execute_transcription(
    plan: JobPlan,
    transcription_options: TranscriptionOptions,
    export_options: ExportOptions,
    *,
    provider: SpeechToTextProvider | None,
    policy: ConflictPolicy = ConflictPolicy.ERROR,
    retries: int = 0,
    backoff_seconds: float = 1.0,
    request_delay: float = 0.0,
    fail_fast: bool = False,
    progress: Callable[[str], None] | None = None,
) -> JobResult:
    if not plan.valid:
        details = "; ".join(f"{item.target}: {item.reason}" for item in plan.conflicts)
        raise PlanningError(f"transcription plan has conflicts: {details}")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("retries must be an integer >= 0")
    backoff_seconds = _finite_nonnegative(backoff_seconds, "backoff_seconds")
    request_delay = _finite_nonnegative(request_delay, "request_delay")
    _validate_planned_names(plan)
    if plan.dry_run:
        return JobResult(tuple(ArtifactResult(output, ArtifactStatus.SKIPPED, "dry-run") for output in plan.artifacts))

    effective_policy = ConflictPolicy.ERROR if policy is ConflictPolicy.RENAME else policy
    results: list[ArtifactResult] = []
    sources_attempted = 0

    source_count = len(plan.sources)
    provider_label = _provider_label(plan.provider)
    for source_number, source in enumerate(plan.sources, start=1):
        outputs = _artifact_map(plan, source)
        completed_formats: set[ArtifactFormat] = set()

        if policy is ConflictPolicy.SKIP and all(_path_occupied(output.target) for output in outputs.values()):
            results.extend(
                ArtifactResult(output, ArtifactStatus.SKIPPED, "output already exists") for output in outputs.values()
            )
            continue

        try:
            if provider is None:
                raise TranscriptionJobError("a speech-to-text provider is required")
            if effective_policy is not ConflictPolicy.REPLACE:
                for parent in {output.target.parent for output in outputs.values()}:
                    ensure_atomic_no_clobber_supported(parent)
            if sources_attempted and request_delay:
                if progress:
                    progress(f"Waiting {request_delay:.1f}s before the next provider request")
                time.sleep(request_delay)
            sources_attempted += 1
            source_before = _source_snapshot(source)
            status_label = f"[{source_number}/{source_count}] {source.name} ({_file_size_text(source_before[2])})"
            request_started = time.monotonic()
            payload = _request_with_retry(
                provider,
                source,
                transcription_options,
                retries=retries,
                backoff_seconds=backoff_seconds,
                progress=progress,
                status_label=status_label,
                provider_label=provider_label,
            )
            if progress:
                progress(
                    f"{status_label} - response received after "
                    f"{_elapsed_text(time.monotonic() - request_started)}; preparing outputs"
                )
            if _source_snapshot(source) != source_before:
                raise TranscriptionJobError(
                    f"source changed while it was being transcribed; response was discarded: {source}"
                )

            transcript, decoded = _prepare_payload(payload, source, transcription_options)
            transcript = replace(
                transcript,
                metadata={**dict(transcript.metadata), "source_name": source.stem},
            )
            prepared = _prepare_outputs(
                outputs,
                transcript,
                payload,
                decoded,
                export_options,
                effective_policy,
            )
            if progress:
                progress(
                    f"{status_label} - transcript ready "
                    f"({len(transcript.text):,} characters, {len(transcript.timed_words):,} timed items); "
                    f"writing {len(prepared)} output(s)"
                )

            for output_format, content in prepared.items():
                output = outputs[output_format]
                if content is None:
                    results.append(ArtifactResult(output, ArtifactStatus.SKIPPED, "output already exists"))
                elif isinstance(content, bytes):
                    target, status = atomic_write_bytes(output.target, content, effective_policy)
                    results.append(ArtifactResult(replace(output, target=target), status))
                else:
                    target, status = atomic_write_text(output.target, content, effective_policy)
                    results.append(ArtifactResult(replace(output, target=target), status))
                completed_formats.add(output_format)
            if progress:
                progress(f"{status_label} - complete")
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            for output_format, output in outputs.items():
                if output_format not in completed_formats:
                    results.append(ArtifactResult(output, ArtifactStatus.FAILED, message))
            if fail_fast:
                break

    return JobResult(tuple(results))
