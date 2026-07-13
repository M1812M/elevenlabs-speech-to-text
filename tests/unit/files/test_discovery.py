from pathlib import Path

import pytest

from elevenlabs_toolkit.files import DiscoveryError, discover_inputs
from elevenlabs_toolkit.models import InputSpec


def _write(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def test_discovers_files_and_directories_in_natural_order_without_duplicates(
    tmp_path: Path,
) -> None:
    file_10 = _write(tmp_path / "clip10.JSON")
    file_2 = _write(tmp_path / "clip2.json")
    _write(tmp_path / "ignore.srt")

    result = discover_inputs(
        InputSpec((file_10, tmp_path)),
        {"json"},
    )

    assert result == (file_2.resolve(), file_10.resolve())


def test_glob_is_explicit_and_recursion_is_opt_in(tmp_path: Path) -> None:
    top_level = _write(tmp_path / "take-1.json")
    nested = _write(tmp_path / "nested" / "take-2.json")
    _write(tmp_path / "other.json")

    shallow = discover_inputs(
        InputSpec((tmp_path,), glob="take-*.json"),
        {".json"},
    )
    recursive = discover_inputs(
        InputSpec((tmp_path,), glob="take-*.json", recursive=True),
        {".json"},
    )

    assert shallow == (top_level.resolve(),)
    assert recursive == (nested.resolve(), top_level.resolve())


def test_regex_matches_relative_names_when_recursive(tmp_path: Path) -> None:
    selected_2 = _write(tmp_path / "day2" / "take-12.json")
    selected_10 = _write(tmp_path / "day10" / "take-3.json")
    _write(tmp_path / "day2" / "draft.json")

    result = discover_inputs(
        InputSpec(
            (tmp_path,),
            regex=r"^day\d+/take-\d+[.]json$",
            recursive=True,
        ),
        {".json"},
    )

    assert result == (selected_2.resolve(), selected_10.resolve())


def test_default_glob_applies_only_when_no_explicit_selector(tmp_path: Path) -> None:
    default_match = _write(tmp_path / "episode.transcript.json")
    uppercase_default_match = _write(tmp_path / "BONUS.TRANSCRIPT.JSON")
    explicit_match = _write(tmp_path / "episode.raw.json")

    default_result = discover_inputs(
        InputSpec((tmp_path,)),
        {".json"},
        default_glob="*.transcript.json",
    )
    explicit_result = discover_inputs(
        InputSpec((tmp_path,), glob="*.raw.json"),
        {".json"},
        default_glob="*.transcript.json",
    )

    assert default_result == (uppercase_default_match.resolve(), default_match.resolve())
    assert explicit_result == (explicit_match.resolve(),)


def test_explicit_glob_keeps_native_case_semantics(tmp_path: Path) -> None:
    _write(tmp_path / "lower.json")
    _write(tmp_path / "UPPER.JSON")

    result = discover_inputs(InputSpec((tmp_path,), glob="*.json"), {".json"})
    expected = {path.resolve() for path in tmp_path.glob("*.json")}

    assert set(result) == expected


def test_generated_transcript_artifacts_are_excluded_by_default(tmp_path: Path) -> None:
    source = _write(tmp_path / "episode.json")
    clean = tmp_path / "episode.clean.json"
    clean.write_text('{"text":"clean","toolkit_processing":{}}', encoding="utf-8")
    generated = [
        clean,
        _write(tmp_path / "episode.segmented.json"),
    ]

    result = discover_inputs(InputSpec((tmp_path,)), {".json"})
    result_with_generated = discover_inputs(
        InputSpec((tmp_path,)),
        {".json"},
        exclude_generated=False,
    )

    assert result == (source.resolve(),)
    assert set(result_with_generated) == {
        source.resolve(),
        clean.resolve(),
        generated[1].resolve(),
    }


def test_cache_manifest_is_not_treated_as_a_transcript(tmp_path: Path) -> None:
    manifest = tmp_path / "episode.manifest.json"
    manifest.write_text(
        """{
            "schema_version": 2,
            "provider": "elevenlabs",
            "source": {"name": "episode.mp3", "size": 5, "sha256": "source"},
            "transcript": {"name": "episode.json", "sha256": "transcript"},
            "transcription": {}
        }""",
        encoding="utf-8",
    )

    with pytest.raises(DiscoveryError, match="metadata is not a transcript"):
        discover_inputs(InputSpec((manifest,)), {".json"}, exclude_generated=False)


def test_canonical_transcript_whose_stem_ends_in_manifest_is_selectable(tmp_path: Path) -> None:
    transcript = tmp_path / "episode.manifest.json"
    transcript.write_text('{"text": "hello"}', encoding="utf-8")
    metadata = tmp_path / "episode.manifest.manifest.json"
    metadata.write_text(
        """{
            "schema_version": 2,
            "provider": "elevenlabs",
            "source": {"name": "episode.manifest.mp3", "size": 5, "sha256": "source"},
            "transcript": {"name": "episode.manifest.json", "sha256": "transcript"},
            "transcription": {}
        }""",
        encoding="utf-8",
    )

    explicit = discover_inputs(InputSpec((transcript,)), {".json"})
    directory = discover_inputs(InputSpec((tmp_path,)), {".json"}, default_glob="*.json")

    assert explicit == (transcript.resolve(),)
    assert directory == (transcript.resolve(),)


def test_malformed_manifest_metadata_remains_excluded(tmp_path: Path) -> None:
    manifest = _write(tmp_path / "episode.manifest.json")

    with pytest.raises(DiscoveryError, match="metadata is not a transcript"):
        discover_inputs(InputSpec((manifest,)), {".json"})


def test_nonexistent_path_is_not_reinterpreted_as_a_pattern(tmp_path: Path) -> None:
    _write(tmp_path / "episode.json")
    expression = tmp_path / ".*[.]json"

    with pytest.raises(DiscoveryError, match="Input path does not exist"):
        discover_inputs(InputSpec((expression,)), {".json"})


def test_explicit_file_reports_suffix_and_generated_file_errors(tmp_path: Path) -> None:
    unsupported = _write(tmp_path / "episode.txt")
    generated = tmp_path / "episode.clean.json"
    generated.write_text('{"text":"clean","toolkit_processing":{}}', encoding="utf-8")

    with pytest.raises(DiscoveryError, match=r"Allowed suffixes: [.]json"):
        discover_inputs(InputSpec((unsupported,)), {".json"})
    with pytest.raises(DiscoveryError, match="exclude_generated=False"):
        discover_inputs(InputSpec((generated,)), {".json"})


def test_empty_result_and_invalid_selectors_have_actionable_errors(tmp_path: Path) -> None:
    _write(tmp_path / "episode.txt")

    with pytest.raises(DiscoveryError, match=r"glob '[*][.]json'.*[.]json"):
        discover_inputs(InputSpec((tmp_path,), glob="*.json"), {".json"})
    with pytest.raises(DiscoveryError, match="Invalid regex selector"):
        discover_inputs(InputSpec((tmp_path,), regex="["), {".json"})
    with pytest.raises(DiscoveryError, match="must be relative"):
        discover_inputs(InputSpec((tmp_path,), glob="../*.json"), {".json"})


def test_suffix_configuration_must_not_be_empty(tmp_path: Path) -> None:
    _write(tmp_path / "episode.json")

    with pytest.raises(DiscoveryError, match="At least one allowed"):
        discover_inputs(InputSpec((tmp_path,)), set())
