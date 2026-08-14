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
        ArtifactFormat.SRT_MINI: f"{stem}.mini.srt",
        ArtifactFormat.TXT: f"{stem}.txt",
        ArtifactFormat.RESOLVE_EDL: f"{stem}.resolve.edl",
        ArtifactFormat.CUE_INDEX_SRT: f"{stem}.cue-index.srt",
        ArtifactFormat.CLEAN_JSON: f"{stem}.clean.json",
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
    provider: str | None = None,
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
        elif _path_occupied(target) and policy is ConflictPolicy.ERROR:
            conflicts.append(PlanConflict(target, tuple(target_sources), "output already exists"))

    return JobPlan(
        sources=tuple(source.resolve() for source in sources),
        artifacts=tuple(artifacts),
        conflicts=tuple(conflicts),
        api_requests=api_requests,
        dry_run=dry_run,
        provider=provider,
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
    dry_run: bool = False,
    transcription_options: TranscriptionOptions,
    provider: str = "elevenlabs",
) -> JobPlan:
    source_tuple = tuple(Path(source) for source in sources)
    if not source_tuple:
        raise PlanningError("no audio/video sources were selected")
    if policy is ConflictPolicy.RENAME:
        raise PlanningError("rename is not supported for transcription outputs")
    if not isinstance(provider, str) or not provider.strip():
        raise PlanningError("provider must be a non-empty string")
    provider = provider.strip()
    remote_map = {
        "pdf": ArtifactFormat.PDF,
        "docx": ArtifactFormat.DOCX,
        "html": ArtifactFormat.HTML,
        "segmented_json": ArtifactFormat.SEGMENTED_JSON,
        "segmented-json": ArtifactFormat.SEGMENTED_JSON,
    }
    local_formats = tuple(formats)
    allowed_local_formats = {
        ArtifactFormat.JSON,
        ArtifactFormat.SRT,
        ArtifactFormat.SRT_MINI,
        ArtifactFormat.TXT,
        ArtifactFormat.RESOLVE_EDL,
        ArtifactFormat.CUE_INDEX_SRT,
    }
    invalid_formats = [item.value for item in local_formats if item not in allowed_local_formats]
    if invalid_formats:
        raise PlanningError("unsupported direct transcription output format(s): " + ", ".join(invalid_formats))
    remote_formats = tuple(remote_map[value] for value in transcription_options.remote_formats if value in remote_map)
    format_tuple = tuple(dict.fromkeys((*local_formats, *remote_formats)))
    if not format_tuple:
        raise PlanningError("at least one transcription output format is required")
    _validate_transcription_names(source_tuple, format_tuple)
    common_parent = _common_parent(source_tuple)
    requested: list[PlannedArtifact] = []
    api_requests = 0

    for source in source_tuple:
        relative_parent = _relative_parent(source, common_parent)
        targets = [output_dir / relative_parent / artifact_name(source, item) for item in format_tuple]
        if policy is not ConflictPolicy.SKIP or not all(_path_occupied(target) for target in targets):
            api_requests += 1
        for artifact_format, target in zip(format_tuple, targets, strict=True):
            requested.append(PlannedArtifact(source, target, artifact_format))

    plan = _finalize_plan(
        source_tuple,
        requested,
        policy=policy,
        api_requests=api_requests,
        dry_run=dry_run,
        provider=provider,
    )
    if plan.valid:
        return plan
    return JobPlan(
        sources=plan.sources,
        artifacts=plan.artifacts,
        conflicts=plan.conflicts,
        api_requests=0,
        dry_run=plan.dry_run,
        provider=plan.provider,
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
