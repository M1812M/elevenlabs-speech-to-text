from __future__ import annotations

import json
import re
from collections.abc import Iterable
from fnmatch import fnmatchcase
from pathlib import Path
from re import Pattern

from elevenlabs_toolkit.models.jobs import InputSpec


class DiscoveryError(ValueError):
    """Raised when an input specification cannot produce usable files."""


AUDIO_VIDEO_SUFFIXES = frozenset(
    {
        ".3gp",
        ".3gpp",
        ".aac",
        ".amr",
        ".aiff",
        ".au",
        ".avi",
        ".caf",
        ".flac",
        ".flv",
        ".m2ts",
        ".m4a",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp3",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".mts",
        ".mxf",
        ".oga",
        ".ogg",
        ".opus",
        ".ts",
        ".wav",
        ".webm",
        ".wma",
        ".wmv",
    }
)


_NATURAL_PARTS = re.compile(r"(\d+)")
_GENERATED_TRANSCRIPT_SUFFIXES = (".segmented.json",)
_METADATA_SUFFIXES = (".manifest.json",)
_TRANSCRIPT_PAYLOAD_KEYS = frozenset({"text", "words", "segments"})
_CACHE_MANIFEST_KEYS = frozenset({"schema_version", "provider", "source", "transcript", "transcription"})


def discover_inputs(
    spec: InputSpec,
    allowed_suffixes: set[str],
    default_glob: str | None = None,
    exclude_generated: bool = True,
) -> tuple[Path, ...]:
    """Resolve an explicit input specification to a stable collection of files.

    Positional files are selected exactly. Positional directories are scanned with
    ``spec.glob``, ``spec.regex``, or ``default_glob`` (in that order). A selector
    is never inferred from a nonexistent positional path.
    """

    if spec.glob is not None and spec.regex is not None:
        raise DiscoveryError("Glob and regex selectors are mutually exclusive.")
    suffixes = _normalise_suffixes(allowed_suffixes)
    regex = _compile_regex(spec.regex)
    glob_pattern = spec.glob if spec.glob is not None else (default_glob if regex is None else None)
    case_insensitive_glob = spec.glob is None and glob_pattern is not None
    if glob_pattern is not None:
        _validate_glob(glob_pattern)

    discovered: dict[Path, Path] = {}
    for raw_path in spec.paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise DiscoveryError(
                f"Input path does not exist: '{path}'. Pass an existing file or "
                "directory; use the explicit glob or regex option to select files "
                "inside a directory."
            )

        if path.is_file():
            selected = _validate_explicit_file(path, suffixes, exclude_generated)
            discovered.setdefault(selected, selected)
            continue

        if not path.is_dir():
            raise DiscoveryError(f"Input path is neither a regular file nor a directory: '{path}'.")

        matches = _discover_in_directory(
            path,
            suffixes=suffixes,
            glob_pattern=glob_pattern,
            regex=regex,
            recursive=spec.recursive,
            exclude_generated=exclude_generated,
            case_insensitive_glob=case_insensitive_glob,
        )
        if not matches:
            selector = _selector_description(glob_pattern, spec.regex)
            recursion = "recursively" if spec.recursive else "at its top level"
            generated_hint = (
                " Generated transcript artifacts are excluded by default; pass exclude_generated=False to include them."
                if exclude_generated and ".json" in suffixes
                else ""
            )
            raise DiscoveryError(
                f"No supported input files found {recursion} in '{path}' using "
                f"{selector}. Allowed suffixes: {_format_suffixes(suffixes)}."
                f"{generated_hint}"
            )
        for match in matches:
            discovered.setdefault(match, match)

    if not discovered:
        raise DiscoveryError(
            "The input specification did not contain any files. Pass at least one existing file or directory."
        )

    return tuple(sorted(discovered.values(), key=_natural_path_key))


def _normalise_suffixes(values: Iterable[str]) -> frozenset[str]:
    suffixes: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise DiscoveryError("Allowed suffixes must be non-empty strings.")
        suffix = value.strip().casefold()
        suffixes.add(suffix if suffix.startswith(".") else f".{suffix}")
    if not suffixes:
        raise DiscoveryError("At least one allowed file suffix is required.")
    return frozenset(suffixes)


def _compile_regex(pattern: str | None) -> Pattern[str] | None:
    if pattern is None:
        return None
    if not pattern:
        raise DiscoveryError("The regex selector must not be empty.")
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise DiscoveryError(f"Invalid regex selector '{pattern}': {exc}.") from exc


def _validate_glob(pattern: str) -> None:
    if not pattern:
        raise DiscoveryError("The glob selector must not be empty.")
    pattern_path = Path(pattern)
    if pattern_path.is_absolute() or ".." in pattern_path.parts:
        raise DiscoveryError(
            f"Glob selector '{pattern}' must be relative to each input directory and must not contain '..'."
        )


def _validate_explicit_file(
    path: Path,
    suffixes: frozenset[str],
    exclude_generated: bool,
) -> Path:
    if path.suffix.casefold() not in suffixes:
        raise DiscoveryError(f"Unsupported input file '{path}'. Allowed suffixes: {_format_suffixes(suffixes)}.")
    if _is_metadata_artifact(path):
        raise DiscoveryError(f"Cache metadata is not a transcript input: '{path}'.")
    if exclude_generated and _is_generated_transcript(path):
        raise DiscoveryError(
            f"Generated transcript artifact is excluded by default: '{path}'. "
            "Pass exclude_generated=False to select it explicitly."
        )
    return _resolve(path)


def _discover_in_directory(
    directory: Path,
    *,
    suffixes: frozenset[str],
    glob_pattern: str | None,
    regex: Pattern[str] | None,
    recursive: bool,
    exclude_generated: bool,
    case_insensitive_glob: bool,
) -> tuple[Path, ...]:
    try:
        if glob_pattern is not None:
            if case_insensitive_glob:
                candidates = _case_insensitive_glob(directory, glob_pattern, recursive)
            else:
                candidates = directory.rglob(glob_pattern) if recursive else directory.glob(glob_pattern)
        else:
            candidates = directory.rglob("*") if recursive else directory.iterdir()

        matches: dict[Path, Path] = {}
        for candidate in candidates:
            if not candidate.is_file():
                continue
            if candidate.suffix.casefold() not in suffixes:
                continue
            if _is_metadata_artifact(candidate):
                continue
            if regex is not None:
                relative_name = candidate.relative_to(directory).as_posix()
                if regex.search(relative_name) is None:
                    continue
            if exclude_generated and _is_generated_transcript(candidate):
                continue
            resolved = _resolve(candidate)
            matches.setdefault(resolved, resolved)
    except (OSError, ValueError) as exc:
        raise DiscoveryError(f"Could not scan input directory '{directory}': {exc}") from exc

    return tuple(sorted(matches.values(), key=_natural_path_key))


def _case_insensitive_glob(directory: Path, pattern: str, recursive: bool) -> Iterable[Path]:
    """Apply an application-owned default glob consistently across platforms."""

    pattern_parts = Path(pattern).parts
    if recursive:
        pattern_parts = ("**", *pattern_parts)
    needs_tree_scan = recursive or len(pattern_parts) > 1 or "**" in pattern_parts
    candidates = directory.rglob("*") if needs_tree_scan else directory.iterdir()
    folded_pattern = tuple(part.casefold() for part in pattern_parts)
    for candidate in candidates:
        relative_parts = tuple(part.casefold() for part in candidate.relative_to(directory).parts)
        if _glob_parts_match(relative_parts, folded_pattern):
            yield candidate


def _glob_parts_match(path_parts: tuple[str, ...], pattern_parts: tuple[str, ...]) -> bool:
    """Match case-folded path parts, treating a complete ``**`` part recursively."""

    memo: dict[tuple[int, int], bool] = {}

    def match(path_index: int, pattern_index: int) -> bool:
        state = (path_index, pattern_index)
        if state in memo:
            return memo[state]
        if pattern_index == len(pattern_parts):
            result = path_index == len(path_parts)
        elif pattern_parts[pattern_index] == "**":
            result = match(path_index, pattern_index + 1) or (
                path_index < len(path_parts) and match(path_index + 1, pattern_index)
            )
        else:
            result = (
                path_index < len(path_parts)
                and fnmatchcase(path_parts[path_index], pattern_parts[pattern_index])
                and match(path_index + 1, pattern_index + 1)
            )
        memo[state] = result
        return result

    return match(0, 0)


def _resolve(path: Path) -> Path:
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise DiscoveryError(f"Could not resolve input file '{path}': {exc}") from exc


def _is_generated_transcript(path: Path) -> bool:
    name = path.name.casefold()
    if any(name.endswith(suffix) for suffix in _GENERATED_TRANSCRIPT_SUFFIXES):
        return True
    if not name.endswith(".clean.json"):
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return False
    return isinstance(payload, dict) and isinstance(payload.get("toolkit_processing"), dict)


def _is_metadata_artifact(path: Path) -> bool:
    name = path.name.casefold()
    if not any(name.endswith(suffix) for suffix in _METADATA_SUFFIXES):
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return True
    if not isinstance(payload, dict):
        return True
    if _CACHE_MANIFEST_KEYS.issubset(payload):
        return True
    return _TRANSCRIPT_PAYLOAD_KEYS.isdisjoint(payload)


def _selector_description(glob_pattern: str | None, regex: str | None) -> str:
    if regex is not None:
        return f"regex '{regex}'"
    if glob_pattern is not None:
        return f"glob '{glob_pattern}'"
    return "no filename selector"


def _format_suffixes(suffixes: Iterable[str]) -> str:
    return ", ".join(sorted(suffixes))


def _natural_path_key(
    path: Path,
) -> tuple[tuple[tuple[int, object], ...], str]:
    normalised = path.as_posix().casefold()
    natural_parts = tuple((1, int(part)) if part.isdigit() else (0, part) for part in _NATURAL_PARTS.split(normalised))
    return natural_parts, path.as_posix()
