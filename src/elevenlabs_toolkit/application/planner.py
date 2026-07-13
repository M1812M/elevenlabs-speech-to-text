from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from ..models import (
    ArtifactFormat,
    ConflictPolicy,
    ExportOptions,
    JobPlan,
    PlanConflict,
    PlannedArtifact,
    ScriptMode,
    TranscriptionOptions,
)
from .cache import cache_matches


class PlanningError(ValueError):
    """Raised when a job cannot be planned without ambiguous output."""


_WINDOWS_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{number}" for number in range(1, 10)}
    | {f"LPT{number}" for number in range(1, 10)}
)


def _safe_leaf_name(value: str) -> bool:
    if not value or value in {".", ".."} or value[-1:] in {" ", "."}:
        return False
    if any(ord(character) < 32 or character in '<>:"/\\|?*' for character in value):
        return False
    return value.split(".", 1)[0].upper() not in _WINDOWS_DEVICE_NAMES


def _validate_transcription_names(
    sources: tuple[Path, ...],
    formats: tuple[ArtifactFormat, ...],
) -> None:
    """Reject names the provider-output boundary cannot publish portably."""

    for source in sources:
        if not _safe_leaf_name(source.stem):
            raise PlanningError(f"transcription source stem is not a portable output name: {source.name!r}")
        for artifact_format in formats:
            leaf = artifact_name(source, artifact_format)
            if not _safe_leaf_name(leaf):
                raise PlanningError(f"generated transcription output name is unsafe: {leaf!r}")


def artifact_name(source: Path, artifact_format: ArtifactFormat, script: ScriptMode = ScriptMode.SOURCE) -> str:
    stem = source.stem
    names = {
        ArtifactFormat.JSON: f"{stem}.json",
        ArtifactFormat.SRT: f"{stem}.srt",
        ArtifactFormat.TXT: f"{stem}.txt",
        ArtifactFormat.SOCIAL_SRT: f"{stem}.social.{script.value}.srt",
        ArtifactFormat.RESOLVE_EDL: f"{stem}.resolve.edl",
        ArtifactFormat.CUE_INDEX_SRT: f"{stem}.cue-index.srt",
        ArtifactFormat.CLEAN_JSON: f"{stem}.clean.json",
        ArtifactFormat.MANIFEST: f"{stem}.manifest.json",
        ArtifactFormat.PDF: f"{stem}.pdf",
        ArtifactFormat.DOCX: f"{stem}.docx",
        ArtifactFormat.HTML: f"{stem}.html",
        ArtifactFormat.SEGMENTED_JSON: f"{stem}.segmented.json",
    }
    if artifact_format is ArtifactFormat.COMBINED_TXT:
        return "combined.txt"
    return names[artifact_format]


def _common_parent(sources: tuple[Path, ...]) -> Path:
    parents = [str(source.resolve().parent) for source in sources]
    try:
        return Path(os.path.commonpath(parents))
    except ValueError:
        # Windows paths on different drives have no common parent. Flattening
        # them is deterministic, and normal target-collision checks still
        # protect equal stems.
        return Path()


def _relative_parent(source: Path, common_parent: Path) -> Path:
    try:
        return source.resolve().parent.relative_to(common_parent)
    except ValueError:
        return Path()


def _renamed_target(target: Path, unavailable: set[Path]) -> Path:
    counter = 2
    candidate = target
    while _path_occupied(candidate) or candidate in unavailable:
        candidate = target.with_name(f"{target.stem} ({counter}){target.suffix}")
        counter += 1
    return candidate


def _absolute_lexical(path: Path) -> Path:
    """Make an output path absolute without following its final symlink."""

    return Path(os.path.abspath(path))


def _path_occupied(path: Path) -> bool:
    """Treat broken symlinks as occupied output names."""

    return path.exists() or path.is_symlink()


def _same_file(source: Path, target: Path) -> bool:
    if source == target:
        return True
    if not _path_occupied(target):
        return False
    try:
        return os.path.samefile(source, target)
    except OSError:
        return False


def _finalize_plan(
    sources: tuple[Path, ...],
    requested: Iterable[PlannedArtifact],
    *,
    policy: ConflictPolicy,
    api_requests: int = 0,
    dry_run: bool = False,
    existing_ok: set[Path] | None = None,
    cache_key: str | None = None,
) -> JobPlan:
    artifacts: list[PlannedArtifact] = []
    conflicts: list[PlanConflict] = []
    by_target: dict[Path, list[Path]] = defaultdict(list)
    unavailable: set[Path] = set()

    for artifact in requested:
        target = _absolute_lexical(artifact.target)
        if policy is ConflictPolicy.RENAME:
            target = _renamed_target(target, unavailable)
        normalized = PlannedArtifact(artifact.source.resolve(), target, artifact.format)
        artifacts.append(normalized)
        by_target[target].append(normalized.source)
        unavailable.add(target)

    for target, target_sources in by_target.items():
        if len(target_sources) > 1:
            conflicts.append(PlanConflict(target, tuple(target_sources), "multiple inputs map to the same output"))
        elif _same_file(target_sources[0], target):
            conflicts.append(PlanConflict(target, tuple(target_sources), "output would overwrite its input"))
        elif _path_occupied(target) and policy is ConflictPolicy.ERROR and target not in (existing_ok or set()):
            conflicts.append(PlanConflict(target, tuple(target_sources), "output already exists"))

    return JobPlan(
        sources=tuple(source.resolve() for source in sources),
        artifacts=tuple(artifacts),
        conflicts=tuple(conflicts),
        api_requests=api_requests,
        dry_run=dry_run,
        cache_key=cache_key,
    )


def plan_exports(
    sources: Iterable[Path],
    options: ExportOptions,
    *,
    policy: ConflictPolicy = ConflictPolicy.ERROR,
    dry_run: bool = False,
    combined_name: str = "combined.txt",
) -> JobPlan:
    source_tuple = tuple(Path(source) for source in sources)
    if not source_tuple:
        raise PlanningError("no transcript sources were selected")
    common_parent = _common_parent(source_tuple)
    requested: list[PlannedArtifact] = []

    for source in source_tuple:
        relative_parent = _relative_parent(source, common_parent)
        for artifact_format in options.formats:
            if artifact_format is ArtifactFormat.COMBINED_TXT:
                continue
            target = options.output_dir / relative_parent / artifact_name(source, artifact_format, options.text.script)
            requested.append(PlannedArtifact(source, target, artifact_format))

    if ArtifactFormat.COMBINED_TXT in options.formats:
        combined_path = Path(combined_name)
        if not _safe_leaf_name(combined_name) or combined_path.is_absolute() or combined_path.name != combined_name:
            raise PlanningError("combined_name must be a non-empty filename, not a path")
        target = options.output_dir / combined_path
        requested.append(PlannedArtifact(source_tuple[0], target, ArtifactFormat.COMBINED_TXT))

    return _finalize_plan(source_tuple, requested, policy=policy, dry_run=dry_run)


def plan_transcription(
    sources: Iterable[Path],
    output_dir: Path,
    formats: Iterable[ArtifactFormat],
    *,
    policy: ConflictPolicy = ConflictPolicy.ERROR,
    resume: bool = True,
    dry_run: bool = False,
    transcription_options: TranscriptionOptions,
    provider_key: str = "elevenlabs",
) -> JobPlan:
    source_tuple = tuple(Path(source) for source in sources)
    if not source_tuple:
        raise PlanningError("no audio/video sources were selected")
    if policy is ConflictPolicy.RENAME:
        raise PlanningError("rename is not supported for transcription because cache names must remain stable")
    if not isinstance(provider_key, str) or not provider_key.strip():
        raise PlanningError("provider_key must be a non-empty string")
    provider_key = provider_key.strip()
    remote_map = {
        "pdf": ArtifactFormat.PDF,
        "docx": ArtifactFormat.DOCX,
        "html": ArtifactFormat.HTML,
        "segmented_json": ArtifactFormat.SEGMENTED_JSON,
        "segmented-json": ArtifactFormat.SEGMENTED_JSON,
    }
    remote_formats = tuple(remote_map[value] for value in transcription_options.remote_formats if value in remote_map)
    format_tuple = tuple(dict.fromkeys((ArtifactFormat.JSON, *formats, *remote_formats, ArtifactFormat.MANIFEST)))
    _validate_transcription_names(source_tuple, format_tuple)
    common_parent = _common_parent(source_tuple)
    requested: list[PlannedArtifact] = []
    api_requests = 0
    existing_ok: set[Path] = set()
    cache_conflicts: list[PlanConflict] = []

    for source in source_tuple:
        relative_parent = _relative_parent(source, common_parent)
        json_target = output_dir / relative_parent / artifact_name(source, ArtifactFormat.JSON)
        manifest_target = output_dir / relative_parent / artifact_name(source, ArtifactFormat.MANIFEST)
        valid_cache = bool(
            resume
            and json_target.is_file()
            and cache_matches(source, json_target, manifest_target, transcription_options, provider=provider_key)
        )
        if not valid_cache:
            api_requests += 1
            if policy is ConflictPolicy.SKIP and (_path_occupied(json_target) or _path_occupied(manifest_target)):
                cache_conflicts.append(
                    PlanConflict(
                        _absolute_lexical(json_target),
                        (source.resolve(),),
                        "stale or incomplete cache cannot be refreshed with the skip policy",
                    )
                )
        else:
            existing_ok.update({_absolute_lexical(json_target), _absolute_lexical(manifest_target)})
        for artifact_format in format_tuple:
            target = output_dir / relative_parent / artifact_name(source, artifact_format)
            requested.append(PlannedArtifact(source, target, artifact_format))

    plan = _finalize_plan(
        source_tuple,
        requested,
        policy=policy,
        api_requests=api_requests,
        dry_run=dry_run,
        existing_ok=existing_ok,
        cache_key=provider_key,
    )
    if not cache_conflicts:
        return plan
    return JobPlan(
        sources=plan.sources,
        artifacts=plan.artifacts,
        conflicts=(*plan.conflicts, *cache_conflicts),
        api_requests=0,
        dry_run=plan.dry_run,
        cache_key=plan.cache_key,
    )


def plan_transliteration(
    sources: Iterable[Path],
    output_dir: Path,
    to_script: ScriptMode,
    *,
    policy: ConflictPolicy = ConflictPolicy.ERROR,
    dry_run: bool = False,
) -> JobPlan:
    source_tuple = tuple(Path(source) for source in sources)
    if not source_tuple:
        raise PlanningError("no SRT sources were selected")
    common_parent = _common_parent(source_tuple)
    requested: list[PlannedArtifact] = []
    for source in source_tuple:
        relative_parent = _relative_parent(source, common_parent)
        stem = source.stem
        for suffix in ("_latin", "_cyrillic", ".latin", ".cyrillic"):
            if stem.casefold().endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        target = output_dir / relative_parent / f"{stem}.{to_script.value}.srt"
        requested.append(PlannedArtifact(source, target, ArtifactFormat.SRT))
    return _finalize_plan(source_tuple, requested, policy=policy, dry_run=dry_run)
