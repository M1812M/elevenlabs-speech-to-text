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
    exclusive_file_lock,
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
from .cache import CACHE_PROVIDER, build_manifest, cache_matches, source_fingerprint
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
    for artifact in plan.artifacts:
        if not _safe_leaf_name(artifact.target.name):
            raise PlanningError(f"generated transcription output name is unsafe: {artifact.target.name!r}")


def _artifact_map(plan: JobPlan, source: Path) -> dict[ArtifactFormat, PlannedArtifact]:
    return {artifact.format: artifact for artifact in plan.artifacts if artifact.source == source}


def _read_cached_payload(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TranscriptionJobError(f"could not load cached transcript '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise TranscriptionJobError(f"cached transcript '{path}' is not a JSON object")
    return payload


def _path_occupied(path: Path) -> bool:
    return path.exists() or path.is_symlink()


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


def _cache_lock_path(transcript_path: Path) -> Path:
    return transcript_path.with_name(f".{transcript_path.stem}.transcription.lock")


def _cache_snapshot(path: Path) -> bytes | None:
    if path.is_symlink():
        raise TranscriptionJobError(f"cache target must not be a symlink: {path}")
    if not path.exists():
        return None
    if not path.is_file():
        raise TranscriptionJobError(f"cache target must be a regular file: {path}")
    return path.read_bytes()


def _restore_cache_file(path: Path, previous: bytes | None, published: bytes) -> None:
    if previous is not None:
        atomic_write_bytes(path, previous, ConflictPolicy.REPLACE)
        return
    if path.is_file() and path.read_bytes() == published:
        path.unlink()


def _write_cache_pair(
    transcript_path: Path,
    transcript_content: str,
    manifest_path: Path,
    manifest_content: str,
    policy: ConflictPolicy,
) -> tuple[ArtifactStatus, ArtifactStatus]:
    previous_transcript = _cache_snapshot(transcript_path) if policy is ConflictPolicy.REPLACE else None
    previous_manifest = _cache_snapshot(manifest_path) if policy is ConflictPolicy.REPLACE else None
    transcript_bytes = transcript_content.encode("utf-8")
    manifest_bytes = manifest_content.encode("utf-8")
    transcript_published = False
    try:
        _path, transcript_status = atomic_write_text(transcript_path, transcript_content, policy)
        if transcript_status is ArtifactStatus.SKIPPED:
            raise TranscriptionJobError("cache target appeared during transcription; response was not cached")
        transcript_published = True
        _path, manifest_status = atomic_write_text(manifest_path, manifest_content, policy)
        if manifest_status is ArtifactStatus.SKIPPED:
            raise TranscriptionJobError("manifest target appeared during transcription; cache is incomplete")
        return transcript_status, manifest_status
    except Exception as exc:
        rollback_errors: list[str] = []
        if transcript_published:
            for path, previous, published in (
                (transcript_path, previous_transcript, transcript_bytes),
                (manifest_path, previous_manifest, manifest_bytes),
            ):
                try:
                    _restore_cache_file(path, previous, published)
                except OSError as rollback_exc:
                    rollback_errors.append(f"{path}: {rollback_exc}")
        if rollback_errors:
            details = "; ".join(rollback_errors)
            raise TranscriptionJobError(f"cache publish failed ({exc}); rollback also failed: {details}") from exc
        raise


def _prepare_outputs(
    artifacts: dict[ArtifactFormat, PlannedArtifact],
    transcript: Transcript,
    payload: dict,
    decoded: dict[str, bytes | str],
    export_options: ExportOptions,
    policy: ConflictPolicy,
) -> dict[ArtifactFormat, bytes | str | None]:
    prepared: dict[ArtifactFormat, bytes | str | None] = {}
    for artifact_format, artifact in artifacts.items():
        if artifact_format in {ArtifactFormat.JSON, ArtifactFormat.MANIFEST}:
            continue
        if artifact_format in REMOTE_FORMATS:
            content = decoded.get(artifact.target.name.casefold())
            if content is None:
                raise TranscriptionJobError(
                    f"ElevenLabs response did not contain requested remote format {artifact_format.value}"
                )
            prepared[artifact_format] = content
            continue
        if policy is ConflictPolicy.SKIP and _path_occupied(artifact.target):
            prepared[artifact_format] = None
            continue
        prepared[artifact_format] = render_artifact(artifact_format, transcript, payload, export_options)
    return prepared


def _request_with_retry(
    provider: SpeechToTextProvider,
    source: Path,
    options: TranscriptionOptions,
    *,
    retries: int,
    backoff_seconds: float,
    progress: Callable[[str], None] | None,
) -> dict:
    for attempt in range(retries + 1):
        try:
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
                progress(f"Transient error for {source.name}; retry {attempt + 2}/{retries + 1} in {delay:.1f}s: {exc}")
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
    resume: bool = True,
    retries: int = 0,
    backoff_seconds: float = 1.0,
    request_delay: float = 0.0,
    lock_timeout_seconds: float = 300.0,
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
    lock_timeout_seconds = _finite_nonnegative(lock_timeout_seconds, "lock_timeout_seconds")
    _validate_planned_names(plan)
    if plan.dry_run:
        return JobResult(
            tuple(ArtifactResult(artifact, ArtifactStatus.SKIPPED, "dry-run") for artifact in plan.artifacts)
        )

    effective_policy = ConflictPolicy.ERROR if policy is ConflictPolicy.RENAME else policy
    cache_provider = plan.cache_key or CACHE_PROVIDER
    if provider is not None:
        provider_cache_key = getattr(provider, "cache_key", None)
        if not isinstance(provider_cache_key, str) or not provider_cache_key.strip():
            raise ValueError("speech-to-text providers must expose a non-empty cache_key")
        if provider_cache_key.strip() != cache_provider:
            raise ValueError(f"provider cache key {provider_cache_key!r} does not match planned key {cache_provider!r}")
    results: list[ArtifactResult] = []
    sources_attempted = 0

    for source in plan.sources:
        artifacts = _artifact_map(plan, source)
        json_artifact = artifacts.get(ArtifactFormat.JSON)
        manifest_artifact = artifacts.get(ArtifactFormat.MANIFEST)
        if json_artifact is None or manifest_artifact is None:
            raise TranscriptionJobError(f"plan for '{source}' has no JSON/manifest cache artifacts")
        completed_formats: set[ArtifactFormat] = set()

        try:
            with exclusive_file_lock(
                _cache_lock_path(json_artifact.target),
                timeout_seconds=lock_timeout_seconds,
            ):
                # Planning is deliberately repeated while holding the cache
                # lock. Another process may have completed the same paid work
                # between preflight and execution.
                cache_valid = bool(
                    resume
                    and cache_matches(
                        source,
                        json_artifact.target,
                        manifest_artifact.target,
                        transcription_options,
                        provider=cache_provider,
                    )
                )
                if cache_valid:
                    payload = _read_cached_payload(json_artifact.target)
                    transcript, decoded = _prepare_payload(payload, source, transcription_options)
                    transcript = replace(
                        transcript,
                        metadata={**dict(transcript.metadata), "source_name": source.stem},
                    )
                    prepared = _prepare_outputs(
                        artifacts,
                        transcript,
                        payload,
                        decoded,
                        export_options,
                        effective_policy,
                    )
                    results.extend(
                        (
                            ArtifactResult(json_artifact, ArtifactStatus.SKIPPED, "valid cache"),
                            ArtifactResult(manifest_artifact, ArtifactStatus.SKIPPED, "valid cache"),
                        )
                    )
                    completed_formats.update({ArtifactFormat.JSON, ArtifactFormat.MANIFEST})
                    if progress:
                        progress(f"CACHE {source.name}")
                else:
                    for cache_target in (json_artifact.target, manifest_artifact.target):
                        if cache_target.is_symlink() or (cache_target.exists() and not cache_target.is_file()):
                            raise TranscriptionJobError(f"cache target must be a regular file: {cache_target}")
                    if effective_policy is not ConflictPolicy.REPLACE and any(
                        _path_occupied(target) for target in (json_artifact.target, manifest_artifact.target)
                    ):
                        raise TranscriptionJobError(
                            "a stale or incomplete cache appeared after planning; no provider request was made"
                        )
                    if provider is None:
                        raise TranscriptionJobError(
                            "a speech-to-text provider is required because the cache is missing or stale"
                        )
                    if effective_policy is not ConflictPolicy.REPLACE:
                        ensure_atomic_no_clobber_supported(json_artifact.target.parent)
                    if sources_attempted and request_delay:
                        time.sleep(request_delay)
                    sources_attempted += 1
                    source_before = source_fingerprint(source)
                    if progress:
                        progress(f"TRANSCRIBE {source.name}")
                    payload = _request_with_retry(
                        provider,
                        source,
                        transcription_options,
                        retries=retries,
                        backoff_seconds=backoff_seconds,
                        progress=progress,
                    )
                    if source_fingerprint(source) != source_before:
                        raise TranscriptionJobError(
                            f"source changed while it was being transcribed; response was not cached: {source}"
                        )

                    transcript, decoded = _prepare_payload(payload, source, transcription_options)
                    transcript = replace(
                        transcript,
                        metadata={**dict(transcript.metadata), "source_name": source.stem},
                    )
                    # Render and validate every requested artifact before the
                    # response is made resumable through its manifest.
                    prepared = _prepare_outputs(
                        artifacts,
                        transcript,
                        payload,
                        decoded,
                        export_options,
                        effective_policy,
                    )
                    json_content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
                    manifest = build_manifest(
                        source,
                        transcription_options,
                        transcript_name=json_artifact.target.name,
                        transcript_content=json_content,
                        source_data=source_before,
                        provider=cache_provider,
                    )
                    manifest_content = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
                    json_status, manifest_status = _write_cache_pair(
                        json_artifact.target,
                        json_content,
                        manifest_artifact.target,
                        manifest_content,
                        effective_policy,
                    )
                    results.extend(
                        (
                            ArtifactResult(json_artifact, json_status),
                            ArtifactResult(manifest_artifact, manifest_status),
                        )
                    )
                    completed_formats.update({ArtifactFormat.JSON, ArtifactFormat.MANIFEST})

                for artifact_format, content in prepared.items():
                    artifact = artifacts[artifact_format]
                    if content is None:
                        results.append(ArtifactResult(artifact, ArtifactStatus.SKIPPED, "output already exists"))
                        completed_formats.add(artifact_format)
                        continue
                    if isinstance(content, bytes):
                        target, status = atomic_write_bytes(artifact.target, content, effective_policy)
                    else:
                        target, status = atomic_write_text(artifact.target, content, effective_policy)
                    results.append(ArtifactResult(replace(artifact, target=target), status))
                    completed_formats.add(artifact_format)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            for artifact_format, artifact in artifacts.items():
                if artifact_format not in completed_formats:
                    results.append(ArtifactResult(artifact, ArtifactStatus.FAILED, message))
            if fail_fast:
                break

    return JobResult(tuple(results))
