from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from ..files import atomic_write_text
from ..languages import get_language_processor
from ..models import (
    ArtifactFormat,
    ArtifactResult,
    ArtifactStatus,
    ConflictPolicy,
    Cue,
    ExportOptions,
    JobPlan,
    JobResult,
    ScriptMode,
    SpeakerLabels,
    Transcript,
    Word,
)
from ..renderers import render_cue_index_srt, render_resolve_edl, render_srt, render_txt
from ..segmentation import segment_standard, segment_transcript, sentences_from_transcript
from .planner import PlanningError

UZBEK_SENTENCE_MARKERS = {"keyin", "shunda", "lekin", "ammo", "biroq", "xullas", "mana", "hozir", "umuman", "demak"}


class ExportError(RuntimeError):
    """Raised when an export artifact cannot be produced."""


def _validate_pause_timings(transcript: Transcript, options: ExportOptions) -> None:
    if not options.segmentation.pause_detection:
        return
    missing = [
        index for index, word in enumerate(transcript.timed_words) if word.kind == "word" and not word.characters
    ]
    if missing:
        indices = ", ".join(str(index) for index in missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ExportError(
            f"pause detection requires character timestamps for every spoken word; missing indices: {indices}{suffix}"
        )


def _sentence_marker_options(
    options: ExportOptions,
) -> tuple[frozenset[str] | set[str], Callable[[str], str] | None]:
    if options.text.cleanup != "uzbek":
        return frozenset(), None
    processor = get_language_processor("uzbek")

    def normalize(value: str) -> str:
        return processor.transform_text(
            value,
            target=ScriptMode.LATIN,
            cleanup=True,
            token_safe=True,
        )

    return UZBEK_SENTENCE_MARKERS, normalize


def _read_payload(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise ExportError(f"could not read transcript '{path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ExportError(f"invalid JSON in '{path}' at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ExportError(f"transcript '{path}' must contain a JSON object")
    return payload


def _text_transform(options: ExportOptions) -> Callable[[str], str]:
    if options.text.cleanup not in {None, "uzbek"}:
        raise ExportError(f"unsupported cleanup processor: {options.text.cleanup}")
    processor = get_language_processor("uzbek")

    def transform(text: str) -> str:
        return processor.transform_text(
            text,
            target=options.text.script,
            cleanup=options.text.cleanup == "uzbek",
            replacements=options.text.replacements,
        )

    return transform


def _subtitle_measurement(
    transcript: Transcript,
    options: ExportOptions,
) -> tuple[Callable[[tuple[Word, ...]], str] | None, str | None]:
    labels = options.text.speaker_labels
    timed_words = transcript.timed_words
    main_speaker = Cue(timed_words).speaker if timed_words else None
    if labels is SpeakerLabels.NONE or main_speaker is None:
        return None, main_speaker

    def prefix(words: tuple[Word, ...]) -> str:
        speaker = Cue(words).speaker
        should_label = bool(speaker) and (
            labels is SpeakerLabels.ALL or (labels is SpeakerLabels.SECONDARY and speaker != main_speaker)
        )
        return "x" * len(f"[{speaker}] ") if should_label else ""

    return prefix, main_speaker


def render_artifact(
    artifact_format: ArtifactFormat,
    transcript: Transcript,
    payload: dict,
    options: ExportOptions,
) -> str:
    transform = _text_transform(options)
    segmentation = options.segmentation
    timed_formats = {
        ArtifactFormat.SRT,
        ArtifactFormat.SOCIAL_SRT,
        ArtifactFormat.CUE_INDEX_SRT,
        ArtifactFormat.RESOLVE_EDL,
    }
    if artifact_format in timed_formats and not transcript.timed_words:
        raise ExportError(f"{artifact_format.value} requires timed words or segments")
    if artifact_format in {*timed_formats, ArtifactFormat.TXT}:
        _validate_pause_timings(transcript, options)

    if artifact_format is ArtifactFormat.SRT:
        text_prefix, main_speaker = _subtitle_measurement(transcript, options)
        cues = segment_standard(transcript, segmentation, transform, text_prefix)
        return render_srt(
            cues,
            text_transform=transform,
            max_chars_per_line=segmentation.max_chars_per_line,
            max_lines=segmentation.max_lines,
            speaker_labels=options.text.speaker_labels,
            main_speaker=main_speaker,
        )
    if artifact_format is ArtifactFormat.SOCIAL_SRT:
        if segmentation.preset not in {"social", "social-uzbek"}:
            segmentation = replace(
                segmentation,
                preset="social",
                max_chars_per_line=min(segmentation.max_chars_per_line, 30),
                max_duration=min(segmentation.max_duration, 2.6),
                min_duration=min(segmentation.min_duration, 0.9),
                gap_seconds=min(segmentation.gap_seconds, 0.75),
                max_words=segmentation.max_words or 9,
            )
        text_prefix, main_speaker = _subtitle_measurement(transcript, options)
        cues = segment_transcript(transcript, segmentation, transform, text_prefix)
        return render_srt(
            cues,
            text_transform=transform,
            max_chars_per_line=segmentation.max_chars_per_line,
            max_lines=segmentation.max_lines,
            speaker_labels=options.text.speaker_labels,
            main_speaker=main_speaker,
        )
    if artifact_format is ArtifactFormat.CUE_INDEX_SRT:
        return render_cue_index_srt(segment_standard(transcript, segmentation))
    if artifact_format is ArtifactFormat.RESOLVE_EDL:
        return render_resolve_edl(
            segment_standard(transcript, segmentation),
            title=str(transcript.metadata.get("source_name") or "Transcript"),
            fps=options.marker_fps,
            color=options.marker_color,
            marker_prefix=options.marker_prefix,
        )
    if artifact_format is ArtifactFormat.TXT:
        markers, marker_normalizer = _sentence_marker_options(options)
        sentences = sentences_from_transcript(
            transcript,
            segmentation,
            marker_breaks=markers,
            marker_normalizer=marker_normalizer,
            text_transform=transform,
        )
        return render_txt(
            [(sentence.text, sentence.speaker) for sentence in sentences],
            speaker_labels=options.text.speaker_labels,
        )
    if artifact_format is ArtifactFormat.CLEAN_JSON:
        if options.text.cleanup != "uzbek":
            raise ExportError("clean-json requires an explicit cleanup profile such as --clean uzbek")
        processor = get_language_processor("uzbek")
        cleaned_payload = processor.transform_payload(
            payload,
            target=options.text.script,
            cleanup=True,
            replacements=options.text.replacements,
        )
        return json.dumps(cleaned_payload, ensure_ascii=False, indent=2) + "\n"
    if artifact_format is ArtifactFormat.JSON:
        return json.dumps(transcript.to_payload(), ensure_ascii=False, indent=2) + "\n"
    raise ExportError(f"unsupported local export format: {artifact_format.value}")


def execute_export(
    plan: JobPlan,
    options: ExportOptions,
    *,
    policy: ConflictPolicy = ConflictPolicy.ERROR,
    fail_fast: bool = False,
) -> JobResult:
    if not plan.valid:
        details = "; ".join(f"{item.target}: {item.reason}" for item in plan.conflicts)
        raise PlanningError(f"export plan has conflicts: {details}")

    effective_policy = ConflictPolicy.ERROR if policy is ConflictPolicy.RENAME else policy
    results: list[ArtifactResult] = []
    loaded: dict[Path, tuple[dict, Transcript]] = {}
    combined_artifacts = [item for item in plan.artifacts if item.format is ArtifactFormat.COMBINED_TXT]

    def load(source: Path) -> tuple[dict, Transcript]:
        if source not in loaded:
            payload = _read_payload(source)
            transcript = Transcript.from_payload(payload)
            transcript = replace(
                transcript,
                metadata={**dict(transcript.metadata), "source_name": source.stem},
            )
            loaded[source] = (payload, transcript)
        return loaded[source]

    for artifact in plan.artifacts:
        if artifact.format is ArtifactFormat.COMBINED_TXT:
            continue
        if plan.dry_run:
            results.append(ArtifactResult(artifact, ArtifactStatus.SKIPPED, "dry-run"))
            continue
        if policy is ConflictPolicy.SKIP and (artifact.target.exists() or artifact.target.is_symlink()):
            results.append(ArtifactResult(artifact, ArtifactStatus.SKIPPED, "output already exists"))
            continue
        try:
            payload, transcript = load(artifact.source)
            content = render_artifact(artifact.format, transcript, payload, options)
            written_path, status = atomic_write_text(artifact.target, content, effective_policy)
            actual = replace(artifact, target=written_path)
            results.append(ArtifactResult(actual, status))
        except Exception as exc:
            results.append(ArtifactResult(artifact, ArtifactStatus.FAILED, f"{type(exc).__name__}: {exc}"))
            if fail_fast:
                return JobResult(tuple(results))

    for artifact in combined_artifacts:
        if plan.dry_run:
            results.append(ArtifactResult(artifact, ArtifactStatus.SKIPPED, "dry-run"))
            continue
        if policy is ConflictPolicy.SKIP and (artifact.target.exists() or artifact.target.is_symlink()):
            results.append(ArtifactResult(artifact, ArtifactStatus.SKIPPED, "output already exists"))
            continue
        try:
            sections: list[str] = []
            for source in plan.sources:
                _payload, transcript = load(source)
                _validate_pause_timings(transcript, options)
                transform = _text_transform(options)
                markers, marker_normalizer = _sentence_marker_options(options)
                sentences = sentences_from_transcript(
                    transcript,
                    options.segmentation,
                    marker_breaks=markers,
                    marker_normalizer=marker_normalizer,
                    text_transform=transform,
                )
                sections.append(
                    render_txt(
                        [(sentence.text, sentence.speaker) for sentence in sentences],
                        speaker_labels=options.text.speaker_labels,
                        include_source=source.name,
                    ).rstrip()
                )
            content = "\n\n".join(section for section in sections if section) + "\n"
            written_path, status = atomic_write_text(artifact.target, content, effective_policy)
            results.append(ArtifactResult(replace(artifact, target=written_path), status))
        except Exception as exc:
            results.append(ArtifactResult(artifact, ArtifactStatus.FAILED, f"{type(exc).__name__}: {exc}"))
            if fail_fast:
                break

    return JobResult(tuple(results))
