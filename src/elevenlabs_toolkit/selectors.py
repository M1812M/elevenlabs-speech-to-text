import re
from pathlib import Path
from typing import Iterable, List


REGEX_META_RE = re.compile(r"[.^$*+?{}\[\]|()]")

ALLOWED_AUDIO_EXTS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".mp4",
    ".wma",
    ".webm",
}


def _collect_regex_matches(path_expression: Path, predicate, label: str) -> List[Path]:
    parent = path_expression.parent
    pattern = path_expression.name

    if not parent.exists() or not parent.is_dir():
        raise FileNotFoundError(f"{label} path not found: {path_expression}")
    if not REGEX_META_RE.search(pattern):
        raise FileNotFoundError(f"{label} path not found: {path_expression}")

    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"Invalid regex in --path expression '{pattern}': {exc}") from exc

    matches = sorted(path for path in parent.iterdir() if path.is_file() and predicate(path, regex))
    if not matches:
        raise FileNotFoundError(f"No {label} files matched regex '{pattern}' in {parent}")
    return matches


def _collect_by_glob(directory: Path, glob_pattern: str, suffixes: Iterable[str], label: str) -> List[Path]:
    suffix_set = {value.lower() for value in suffixes}
    files = sorted(
        path
        for path in directory.glob(glob_pattern)
        if path.is_file() and path.suffix.lower() in suffix_set
    )
    if not files:
        raise FileNotFoundError(f"No {label} files found in {directory} matching {glob_pattern}")
    return files


def collect_audio_files(path: Path) -> List[Path]:
    if not path.exists():
        return _collect_regex_matches(
            path,
            lambda p, regex: p.suffix.lower() in ALLOWED_AUDIO_EXTS and regex.search(p.name),
            "audio/video",
        )

    if path.is_file():
        if path.suffix.lower() not in ALLOWED_AUDIO_EXTS:
            raise ValueError(
                f"Unsupported input file extension: {path.suffix}. Supported: {sorted(ALLOWED_AUDIO_EXTS)}"
            )
        return [path]

    if path.is_dir():
        files = sorted(
            path_obj
            for path_obj in path.iterdir()
            if path_obj.is_file() and path_obj.suffix.lower() in ALLOWED_AUDIO_EXTS
        )
        if not files:
            raise FileNotFoundError(
                f"No supported audio/video files found in {path}. Supported extensions: {sorted(ALLOWED_AUDIO_EXTS)}"
            )
        return files

    raise ValueError(f"Input path is neither file nor directory: {path}")


def collect_json_sources(path: Path, glob_pattern: str = "*.json") -> List[Path]:
    if not path.exists():
        return _collect_regex_matches(
            path,
            lambda p, regex: p.suffix.lower() == ".json" and regex.search(p.name),
            "JSON",
        )

    if path.is_file():
        if path.suffix.lower() != ".json":
            raise ValueError(f"--path points to a file but not JSON: {path}")
        return [path]

    if path.is_dir():
        return _collect_by_glob(path, glob_pattern, [".json"], "JSON")

    raise ValueError(f"Input path is neither file nor directory: {path}")


def collect_latin_srt_sources(path: Path, glob_pattern: str = "*_latin.srt") -> List[Path]:
    if not path.exists():
        return _collect_regex_matches(
            path,
            lambda p, regex: p.suffix.lower() == ".srt" and regex.search(p.name),
            "Latin SRT",
        )

    if path.is_file():
        if path.suffix.lower() != ".srt":
            raise ValueError(f"--path points to a file but not SRT: {path}")
        return [path]

    if path.is_dir():
        return _collect_by_glob(path, glob_pattern, [".srt"], "Latin SRT")

    raise ValueError(f"Input path is neither file nor directory: {path}")
