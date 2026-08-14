from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ArtifactFormat(str, Enum):
    JSON = "json"
    SRT = "srt"
    SRT_MINI = "srt-mini"
    TXT = "txt"
    COMBINED_TXT = "combined-txt"
    RESOLVE_EDL = "resolve-edl"
    CUE_INDEX_SRT = "cue-index-srt"
    CLEAN_JSON = "clean-json"
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    SEGMENTED_JSON = "segmented-json"


class ScriptMode(str, Enum):
    SOURCE = "source"
    LATIN = "latin"
    CYRILLIC = "cyrillic"


class ConflictPolicy(str, Enum):
    ERROR = "error"
    SKIP = "skip"
    REPLACE = "replace"
    RENAME = "rename"


class ArtifactStatus(str, Enum):
    WRITTEN = "written"
    SKIPPED = "skipped"
    FAILED = "failed"


class SpeakerLabels(str, Enum):
    NONE = "none"
    SECONDARY = "secondary"
    ALL = "all"


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not a boolean")
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number")
    return parsed


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


@dataclass(frozen=True, slots=True)
class InputSpec:
    paths: tuple[Path, ...]
    glob: str | None = None
    regex: str | None = None
    recursive: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.paths, (str, bytes)) or not isinstance(self.paths, (tuple, list)) or not self.paths:
            raise ValueError("at least one input path is required")
        for name, value in (("glob", self.glob), ("regex", self.regex)):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string")
        if self.glob and self.regex:
            raise ValueError("glob and regex selection are mutually exclusive")
        _require_bool(self.recursive, "recursive")
        try:
            paths = tuple(Path(path) for path in self.paths)
        except TypeError as exc:
            raise ValueError("input paths must be path-like values") from exc
        object.__setattr__(self, "paths", paths)


@dataclass(frozen=True, slots=True)
class SegmentationOptions:
    preset: str = "standard"
    max_chars_per_line: int = 42
    max_lines: int = 2
    max_duration: float = 5.5
    min_duration: float = 1.0
    gap_seconds: float = 0.9
    hard_gap_seconds: float = 1.8
    pause_detection: bool = False
    max_words: int | None = None
    split_on_speaker_change: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.preset, str) or not self.preset.strip():
            raise ValueError("preset must not be empty")
        max_chars = _require_int(self.max_chars_per_line, "max_chars_per_line")
        max_lines = _require_int(self.max_lines, "max_lines")
        max_duration = _finite_number(self.max_duration, "max_duration")
        min_duration = _finite_number(self.min_duration, "min_duration")
        gap_seconds = _finite_number(self.gap_seconds, "gap_seconds")
        hard_gap_seconds = _finite_number(self.hard_gap_seconds, "hard_gap_seconds")
        _require_bool(self.pause_detection, "pause_detection")
        _require_bool(self.split_on_speaker_change, "split_on_speaker_change")
        if max_chars <= 0 or max_lines <= 0:
            raise ValueError("line limits must be > 0")
        if max_duration <= 0 or min_duration < 0:
            raise ValueError("duration limits must be positive")
        if min_duration > max_duration:
            raise ValueError("min_duration must be <= max_duration")
        if gap_seconds <= 0 or hard_gap_seconds <= 0:
            raise ValueError("gap thresholds must be > 0")
        if hard_gap_seconds < gap_seconds:
            raise ValueError("hard_gap_seconds must be >= gap_seconds")
        if self.max_words is not None and _require_int(self.max_words, "max_words") <= 0:
            raise ValueError("max_words must be > 0")
        object.__setattr__(self, "preset", self.preset.strip())
        object.__setattr__(self, "max_duration", max_duration)
        object.__setattr__(self, "min_duration", min_duration)
        object.__setattr__(self, "gap_seconds", gap_seconds)
        object.__setattr__(self, "hard_gap_seconds", hard_gap_seconds)


@dataclass(frozen=True, slots=True)
class TextOptions:
    script: ScriptMode = ScriptMode.SOURCE
    cleanup: str | None = None
    speaker_labels: SpeakerLabels = SpeakerLabels.NONE
    replacements: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            script = ScriptMode(self.script)
            speaker_labels = SpeakerLabels(self.speaker_labels)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid text option: {exc}") from exc
        cleanup = self.cleanup.strip() if isinstance(self.cleanup, str) else self.cleanup
        if cleanup not in {None, "uzbek"}:
            raise ValueError("cleanup must be 'uzbek' or null")
        if not isinstance(self.replacements, (tuple, list)) or not all(
            isinstance(item, str) for item in self.replacements
        ):
            raise ValueError("text replacements must be strings")
        replacements = tuple(self.replacements)
        invalid = []
        for item in replacements:
            if "=" not in item:
                invalid.append(item)
                continue
            source, target = (part.strip() for part in item.split("=", 1))
            if not source or not target or any(character.isspace() for character in source + target):
                invalid.append(item)
        if invalid:
            raise ValueError("text replacements must use non-empty TOKEN=TOKEN entries")
        object.__setattr__(self, "script", script)
        object.__setattr__(self, "speaker_labels", speaker_labels)
        object.__setattr__(self, "cleanup", cleanup)
        object.__setattr__(self, "replacements", replacements)


@dataclass(frozen=True, slots=True)
class TranscriptionOptions:
    model_id: str = "scribe_v2"
    language_code: str | None = None
    timestamps_granularity: str = "character"
    diarize: bool = True
    tag_audio_events: bool = True
    num_speakers: int | None = None
    keyterms: tuple[str, ...] = ()
    no_verbatim: bool = False
    seed: int | None = None
    temperature: float | None = None
    remote_formats: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if self.timestamps_granularity not in {"none", "word", "character"}:
            raise ValueError("timestamps_granularity must be none, word, or character")
        for name in ("diarize", "tag_audio_events", "no_verbatim"):
            _require_bool(getattr(self, name), name)
        if self.num_speakers is not None and not 1 <= _require_int(self.num_speakers, "num_speakers") <= 32:
            raise ValueError("num_speakers must be between 1 and 32")
        if self.seed is not None and not 0 <= _require_int(self.seed, "seed") <= 2_147_483_647:
            raise ValueError("seed must be between 0 and 2147483647")
        temperature = None if self.temperature is None else _finite_number(self.temperature, "temperature")
        if temperature is not None and not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.language_code is not None and (
            not isinstance(self.language_code, str) or not self.language_code.strip()
        ):
            raise ValueError("language_code must be a non-empty string")
        if isinstance(self.keyterms, (str, bytes)) or not isinstance(self.keyterms, (tuple, list)):
            raise ValueError("keyterms must be a sequence of non-empty strings")
        if not all(isinstance(item, str) and item.strip() for item in self.keyterms):
            raise ValueError("keyterms must be non-empty strings")
        keyterms = tuple(dict.fromkeys(item.strip() for item in self.keyterms))
        if len(keyterms) > 1000:
            raise ValueError("keyterms cannot contain more than 1000 entries")
        forbidden_keyterm_characters = set("<>{}[]\\")
        for keyterm in keyterms:
            if len(keyterm) >= 50:
                raise ValueError("each keyterm must contain fewer than 50 characters")
            if len(keyterm.split()) > 5:
                raise ValueError("each keyterm can contain at most 5 words")
            if any(character in forbidden_keyterm_characters for character in keyterm):
                raise ValueError("keyterms cannot contain <, >, {, }, [, ], or \\")
        if self.no_verbatim and self.model_id.strip() != "scribe_v2":
            raise ValueError("no_verbatim is supported only by the scribe_v2 model")
        if isinstance(self.remote_formats, (str, bytes)) or not isinstance(self.remote_formats, (tuple, list)):
            raise ValueError("remote_formats must be a sequence of format names")
        allowed_remote = {"pdf", "docx", "html", "segmented_json"}
        normalized_remote: list[str] = []
        for item in self.remote_formats:
            if not isinstance(item, str):
                raise ValueError("remote format names must be strings")
            normalized = item.strip().lower().replace("-", "_")
            if normalized not in allowed_remote:
                choices = ", ".join(sorted(allowed_remote))
                raise ValueError(f"unsupported remote format {item!r}; choose {choices}")
            if normalized not in normalized_remote:
                normalized_remote.append(normalized)
        object.__setattr__(self, "model_id", self.model_id.strip())
        object.__setattr__(self, "language_code", self.language_code.strip() if self.language_code else None)
        object.__setattr__(self, "keyterms", keyterms)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "remote_formats", tuple(normalized_remote))


@dataclass(frozen=True, slots=True)
class ExportOptions:
    formats: tuple[ArtifactFormat, ...]
    output_dir: Path
    segmentation: SegmentationOptions = field(default_factory=SegmentationOptions)
    text: TextOptions = field(default_factory=TextOptions)
    marker_fps: float = 25.0
    marker_color: str = "ResolveColorBlue"
    marker_prefix: str = "Sentence"
    srt_smart_line_breaks: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.formats, (str, bytes)) or not isinstance(self.formats, (tuple, list)) or not self.formats:
            raise ValueError("at least one export format is required")
        try:
            formats = tuple(dict.fromkeys(ArtifactFormat(item) for item in self.formats))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid export format: {exc}") from exc
        if not isinstance(self.segmentation, SegmentationOptions):
            raise ValueError("segmentation must be SegmentationOptions")
        if not isinstance(self.text, TextOptions):
            raise ValueError("text must be TextOptions")
        _require_bool(self.srt_smart_line_breaks, "srt_smart_line_breaks")
        if not isinstance(self.marker_color, str) or not self.marker_color.strip():
            raise ValueError("marker_color must be a non-empty string")
        if not isinstance(self.marker_prefix, str):
            raise ValueError("marker_prefix must be a string")
        marker_fps = _finite_number(self.marker_fps, "marker_fps")
        if marker_fps <= 0:
            raise ValueError("marker_fps must be > 0")
        try:
            output_dir = Path(self.output_dir)
        except TypeError as exc:
            raise ValueError("output_dir must be path-like") from exc
        object.__setattr__(self, "formats", formats)
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "marker_fps", marker_fps)


@dataclass(frozen=True, slots=True)
class PlannedArtifact:
    source: Path
    target: Path
    format: ArtifactFormat


@dataclass(frozen=True, slots=True)
class PlanConflict:
    target: Path
    sources: tuple[Path, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class JobPlan:
    sources: tuple[Path, ...]
    artifacts: tuple[PlannedArtifact, ...]
    conflicts: tuple[PlanConflict, ...] = ()
    api_requests: int = 0
    dry_run: bool = False
    provider: str | None = None

    @property
    def valid(self) -> bool:
        return not self.conflicts


@dataclass(frozen=True, slots=True)
class ArtifactResult:
    artifact: PlannedArtifact
    status: ArtifactStatus
    message: str = ""


@dataclass(frozen=True, slots=True)
class JobResult:
    artifacts: tuple[ArtifactResult, ...] = ()

    @property
    def written(self) -> int:
        return sum(item.status is ArtifactStatus.WRITTEN for item in self.artifacts)

    @property
    def skipped(self) -> int:
        return sum(item.status is ArtifactStatus.SKIPPED for item in self.artifacts)

    @property
    def failed(self) -> int:
        return sum(item.status is ArtifactStatus.FAILED for item in self.artifacts)

    @property
    def succeeded(self) -> int:
        return self.written + self.skipped

    @property
    def exit_code(self) -> int:
        return 1 if self.failed else 0
