"""Shared preparation for desktop previews and execution.

The desktop defaults to source-adjacent outputs. Existing CLI planners retain
their explicit output-directory behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..config import effective_config, profile_options
from ..models import (
    ArtifactFormat,
    ConflictPolicy,
    ExportOptions,
    JobPlan,
    ScriptMode,
    TranscriptionOptions,
)
from .planner import (
    PlanningError,
    _common_parent,
    _relative_parent,
    _renamed_target,
    plan_exports,
    plan_transcription,
    plan_transliteration,
)

PROJECT_DIRECTORY = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DesktopSettings:
    project_directory: Path = PROJECT_DIRECTORY
    profile: str | None = None
    output_script: ScriptMode | None = None
    replacements: tuple[str, ...] | None = None
    cleanup: str | None = None
    segmentation: tuple[tuple[str, object], ...] = ()
    smart_line_breaks: bool = False
    marker_fps: float = 25.0
    marker_color: str = "ResolveColorBlue"
    marker_prefix: str = "Sentence"
    combined_name: str = "combined.txt"
    transcription: TranscriptionOptions = field(default_factory=TranscriptionOptions)
    additional_formats: tuple[ArtifactFormat, ...] = ()
    env_file: Path | None = None
    retries: int = 0
    retry_backoff: float = 1.0
    request_delay: float = 0.0
    output_directory: Path | None = None
    fail_fast: bool = False

    def configuration(self, *, clean: bool = False) -> dict:
        overrides: dict[str, object] = {f"segmentation.{name}": value for name, value in self.segmentation}
        if self.output_script is not None:
            overrides["text.script"] = self.output_script.value
        if self.replacements is not None:
            overrides["text.replacements"] = self.replacements
        if self.cleanup is not None:
            overrides["text.cleanup"] = None if self.cleanup == "none" else self.cleanup
        if clean:
            overrides["text.cleanup"] = "uzbek"
        return effective_config(self.profile, overrides=overrides, cwd=self.project_directory)


@dataclass(frozen=True)
class PreparedFile:
    plan: JobPlan
    export: ExportOptions | None = None
    transcription: TranscriptionOptions | None = None


def prepare_files(
    action: str,
    sources: tuple[Path, ...],
    mode: ScriptMode,
    policy: ConflictPolicy,
    formats: tuple[ArtifactFormat, ...],
    settings: DesktopSettings,
) -> tuple[PreparedFile, ...]:
    """Plan the complete selection before any processing or paid request."""
    if action not in {"Transcribe", "Convert Script", "Export", "Clean", "Inspect"}:
        raise ValueError(f"Unsupported workflow: {action}")
    if not sources:
        raise ValueError("Select at least one supported file")
    sources = tuple(dict.fromkeys(source.resolve() for source in sources))
    if action == "Export" and not formats:
        raise ValueError("Select at least one export format")
    if action == "Transcribe":
        if settings.retries < 0:
            raise ValueError("Retries must be non-negative")
        for delay in (settings.retry_backoff, settings.request_delay):
            if not math.isfinite(delay) or delay < 0:
                raise ValueError("Request delay and retry backoff must be finite and non-negative")
    output_directory = settings.output_directory
    common_parent = _common_parent(sources) if output_directory is not None else None
    prepared = []
    for source in sources:
        directory = (
            output_directory / _relative_parent(source, common_parent)
            if output_directory is not None and common_parent is not None
            else source.parent
        )
        if action == "Export" and formats == (ArtifactFormat.COMBINED_TXT,):
            break
        if action == "Inspect":
            prepared.append(PreparedFile(JobPlan((source,), ())))
        elif action == "Convert Script":
            plan = plan_transliteration((source,), directory, mode, policy=policy)
            prepared.append(PreparedFile(plan))
        else:
            if action == "Export" and not any(item is not ArtifactFormat.COMBINED_TXT for item in formats):
                continue
            config = settings.configuration(clean=action == "Clean")
            segmentation, text = profile_options(config["profile"], config)
            selected: tuple[ArtifactFormat, ...] = tuple(
                item for item in formats if item is not ArtifactFormat.COMBINED_TXT
            )
            if action == "Clean":
                selected = (ArtifactFormat.CLEAN_JSON,)
            if action == "Transcribe":
                selected = (ArtifactFormat.JSON, *settings.additional_formats)
            export = ExportOptions(
                selected,
                directory,
                segmentation,
                text,
                marker_fps=settings.marker_fps,
                marker_color=settings.marker_color,
                marker_prefix=settings.marker_prefix,
                srt_smart_line_breaks=settings.smart_line_breaks,
            )
            if action == "Transcribe":
                transcription = settings.transcription
                timestamps = transcription.timestamps_granularity
                if timestamps == "none" and any(
                    item in selected
                    for item in (
                        ArtifactFormat.SRT,
                        ArtifactFormat.SRT_MINI,
                        ArtifactFormat.RESOLVE_EDL,
                        ArtifactFormat.CUE_INDEX_SRT,
                    )
                ):
                    raise ValueError("Timed outputs require word or character timestamps")
                if segmentation.pause_detection and timestamps != "character":
                    raise ValueError("Pause detection requires character timestamps")
                plan = plan_transcription(
                    (source,),
                    directory,
                    selected,
                    policy=policy,
                    transcription_options=transcription,
                    output_stems={source: f"{source.name}.transcript"},
                )
                prepared.append(PreparedFile(plan, export, transcription))
            else:
                if action == "Export" and ArtifactFormat.CLEAN_JSON in selected and settings.cleanup == "none":
                    raise ValueError("Clean JSON cannot be combined with cleanup set to None")
                if ArtifactFormat.CLEAN_JSON in selected and export.text.cleanup is None:
                    export = replace(export, text=replace(export.text, cleanup="uzbek"))
                plan = plan_exports((source,), export, policy=policy)
                prepared.append(PreparedFile(plan, export))
    if action == "Export" and ArtifactFormat.COMBINED_TXT in formats:
        config = settings.configuration()
        segmentation, text = profile_options(config["profile"], config)
        directories = (
            (output_directory,)
            if output_directory is not None
            else tuple(dict.fromkeys(source.parent for source in sources))
        )
        for directory in directories:
            group = tuple(
                sorted(source for source in sources if output_directory is not None or source.parent == directory)
            )
            options = ExportOptions((ArtifactFormat.COMBINED_TXT,), directory, segmentation, text)
            plan = plan_exports(group, options, policy=policy, combined_name=settings.combined_name)
            prepared.append(PreparedFile(plan, options))
    source_paths = set(sources)
    source_ids = {(info.st_dev, info.st_ino) for info in (source.stat() for source in sources)}
    targets: dict[Path, Path] = {}
    for index, item in enumerate(prepared):
        artifacts = []
        for conflict in item.plan.conflicts:
            raise PlanningError(f"{conflict.target}: {conflict.reason}")
        for artifact in item.plan.artifacts:
            target = artifact.target
            if policy is ConflictPolicy.RENAME and target in targets:
                target = _renamed_target(target, set(targets))
                artifact = replace(artifact, target=target)
            if target in targets:
                raise PlanningError(f"Multiple selected files would write {target}")
            try:
                info = target.stat()
            except FileNotFoundError:
                info = None
            if target.resolve() in source_paths or (info is not None and (info.st_dev, info.st_ino) in source_ids):
                raise PlanningError(f"Output would overwrite a selected input: {target}")
            targets[target] = artifact.source
            artifacts.append(artifact)
        if policy is ConflictPolicy.RENAME:
            prepared[index] = replace(item, plan=replace(item.plan, artifacts=tuple(artifacts)))
    return tuple(prepared)
