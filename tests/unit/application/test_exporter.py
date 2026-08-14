import json
from pathlib import Path

import pytest

from elevenlabs_toolkit.application import ExportError, execute_export, plan_exports, render_artifact
from elevenlabs_toolkit.models import (
    ArtifactFormat,
    ConflictPolicy,
    ExportOptions,
    ScriptMode,
    SegmentationOptions,
    SpeakerLabels,
    TextOptions,
    Transcript,
)


def _write_transcript(path: Path, text: str = "Salom dunyo.") -> Path:
    payload = {
        "text": text,
        "words": [
            {"type": "word", "text": "Salom", "start": 0, "end": 0.3, "speaker_id": "speaker_0"},
            {"type": "word", "text": "dunyo.", "start": 0.4, "end": 0.8, "speaker_id": "speaker_1"},
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_export_writes_multiple_formats_through_atomic_store(tmp_path: Path) -> None:
    source = _write_transcript(tmp_path / "sample.json")
    options = ExportOptions(
        (ArtifactFormat.SRT, ArtifactFormat.TXT, ArtifactFormat.RESOLVE_EDL),
        tmp_path / "out",
        text=TextOptions(speaker_labels=SpeakerLabels.ALL),
    )
    plan = plan_exports((source,), options)

    result = execute_export(plan, options)

    assert result.failed == 0
    assert result.written == 3
    srt = (tmp_path / "out" / "sample.srt").read_text(encoding="utf-8")
    assert "Salom" in srt and "dunyo." in srt
    assert "[speaker_0] Salom" in srt
    assert "[speaker_0] Salom" in (tmp_path / "out" / "sample.txt").read_text(encoding="utf-8")
    assert "FCM: NON-DROP FRAME" in (tmp_path / "out" / "sample.resolve.edl").read_text(encoding="utf-8")


def test_srt_mini_export_uses_json_sentence_timings(tmp_path: Path) -> None:
    source = tmp_path / "sample.json"
    payload = {
        "text": "First. Second sentence.",
        "words": [
            {"type": "word", "text": "First.", "start": 0.1, "end": 0.4},
            {"type": "word", "text": "Second", "start": 0.41, "end": 0.7},
            {"type": "word", "text": "sentence.", "start": 0.71, "end": 1.0},
        ],
    }
    source.write_text(json.dumps(payload), encoding="utf-8")
    options = ExportOptions((ArtifactFormat.SRT_MINI,), tmp_path / "out")

    result = execute_export(plan_exports((source,), options), options)

    assert result.failed == 0
    rendered = (tmp_path / "out" / "sample.mini.srt").read_text(encoding="utf-8")
    assert rendered.count(" --> ") == 2
    assert "00:00:00,100 --> 00:00:00,355" in rendered
    assert "00:00:00,455 --> 00:00:01,000" in rendered
    assert "First." in rendered
    assert "Second sentence." in rendered


def test_combined_text_includes_source_provenance(tmp_path: Path) -> None:
    first = _write_transcript(tmp_path / "first.json")
    second = _write_transcript(tmp_path / "second.json")
    options = ExportOptions((ArtifactFormat.COMBINED_TXT,), tmp_path / "out")

    result = execute_export(plan_exports((first, second), options), options)

    assert result.failed == 0
    content = (tmp_path / "out" / "combined.txt").read_text(encoding="utf-8")
    assert "# first.json" in content
    assert "# second.json" in content


def test_dry_run_does_not_create_output_directory(tmp_path: Path) -> None:
    source = _write_transcript(tmp_path / "sample.json")
    options = ExportOptions((ArtifactFormat.SRT,), tmp_path / "out")
    plan = plan_exports((source,), options, dry_run=True)

    result = execute_export(plan, options)

    assert result.skipped == 1
    assert not options.output_dir.exists()


def test_skip_policy_preserves_manually_edited_output(tmp_path: Path) -> None:
    source = _write_transcript(tmp_path / "sample.json")
    output = tmp_path / "out"
    output.mkdir()
    existing = output / "sample.srt"
    existing.write_text("manual edit", encoding="utf-8")
    options = ExportOptions((ArtifactFormat.SRT,), output)

    result = execute_export(
        plan_exports((source,), options, policy=ConflictPolicy.SKIP),
        options,
        policy=ConflictPolicy.SKIP,
    )

    assert result.skipped == 1
    assert existing.read_text(encoding="utf-8") == "manual edit"


def test_skip_policy_does_not_parse_an_input_for_an_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "sample.json"
    source.write_text("not json", encoding="utf-8")
    output = tmp_path / "out"
    output.mkdir()
    existing = output / "sample.srt"
    existing.write_text("manual edit", encoding="utf-8")
    options = ExportOptions((ArtifactFormat.SRT,), output)

    result = execute_export(
        plan_exports((source,), options, policy=ConflictPolicy.SKIP),
        options,
        policy=ConflictPolicy.SKIP,
    )

    assert result.failed == 0
    assert result.skipped == 1
    assert existing.read_text(encoding="utf-8") == "manual edit"


def test_transformed_text_length_guides_subtitle_segmentation(tmp_path: Path) -> None:
    payload = {
        "text": "щ щ щ щ",
        "words": [{"type": "word", "text": "щ", "start": index * 0.2, "end": index * 0.2 + 0.1} for index in range(4)],
    }
    transcript = Transcript.from_payload(payload)
    options = ExportOptions(
        (ArtifactFormat.SRT,),
        tmp_path,
        segmentation=SegmentationOptions(
            max_chars_per_line=5,
            max_lines=1,
            max_duration=10,
            min_duration=0,
        ),
        text=TextOptions(script=ScriptMode.LATIN),
    )

    rendered = render_artifact(ArtifactFormat.SRT, transcript, payload, options)
    text_lines = [line for line in rendered.splitlines() if line and not line.isdigit() and " --> " not in line]

    assert " ".join(text_lines) == "sh sh sh sh"
    assert all(len(line) <= 5 for line in text_lines)


def test_speaker_label_width_guides_subtitle_segmentation(tmp_path: Path) -> None:
    payload = {
        "text": "one two",
        "words": [
            {"type": "word", "text": "one", "start": 0, "end": 0.2, "speaker_id": "spk"},
            {"type": "word", "text": "two", "start": 0.3, "end": 0.5, "speaker_id": "spk"},
        ],
    }
    transcript = Transcript.from_payload(payload)
    options = ExportOptions(
        (ArtifactFormat.SRT,),
        tmp_path,
        segmentation=SegmentationOptions(
            max_chars_per_line=12,
            max_lines=1,
            max_duration=10,
            min_duration=0,
        ),
        text=TextOptions(speaker_labels=SpeakerLabels.ALL),
    )

    rendered = render_artifact(ArtifactFormat.SRT, transcript, payload, options)
    text_lines = [line for line in rendered.splitlines() if line and not line.isdigit() and " --> " not in line]

    assert text_lines == ["[spk] one", "[spk] two"]


def test_secondary_labels_do_not_split_the_only_speaker(tmp_path: Path) -> None:
    payload = {
        "text": "one two",
        "words": [
            {"type": "word", "text": "one", "start": 0, "end": 0.2, "speaker_id": "spk"},
            {"type": "word", "text": "two", "start": 0.3, "end": 0.5, "speaker_id": "spk"},
        ],
    }
    transcript = Transcript.from_payload(payload)
    options = ExportOptions(
        (ArtifactFormat.SRT,),
        tmp_path,
        segmentation=SegmentationOptions(
            max_chars_per_line=10,
            max_lines=1,
            max_duration=10,
            min_duration=0,
        ),
        text=TextOptions(speaker_labels=SpeakerLabels.SECONDARY),
    )

    rendered = render_artifact(ArtifactFormat.SRT, transcript, payload, options)

    assert "one two" in rendered
    assert "[spk]" not in rendered
    assert rendered.count(" --> ") == 1


def test_pause_detection_requires_character_timestamps(tmp_path: Path) -> None:
    payload = {
        "text": "hello",
        "words": [{"type": "word", "text": "hello", "start": 0, "end": 1}],
    }
    options = ExportOptions(
        (ArtifactFormat.SRT,),
        tmp_path,
        segmentation=SegmentationOptions(pause_detection=True),
    )

    with pytest.raises(ExportError, match="character timestamps"):
        render_artifact(ArtifactFormat.SRT, Transcript.from_payload(payload), payload, options)


def test_pause_detection_rejects_partial_character_timestamp_coverage(tmp_path: Path) -> None:
    payload = {
        "text": "hello world",
        "words": [
            {
                "type": "word",
                "text": "hello",
                "start": 0,
                "end": 0.4,
                "characters": [{"text": "h", "start": 0, "end": 0.1}],
            },
            {"type": "word", "text": "world", "start": 0.5, "end": 1},
        ],
    }
    options = ExportOptions(
        (ArtifactFormat.SRT,),
        tmp_path,
        segmentation=SegmentationOptions(pause_detection=True),
    )

    with pytest.raises(ExportError, match="missing indices: 1"):
        render_artifact(ArtifactFormat.SRT, Transcript.from_payload(payload), payload, options)


@pytest.mark.parametrize(
    ("replacement", "source_text", "expected"),
    [
        ("foo=done.", "foo bar", ["done.", "bar"]),
        ("foo.=done", "foo. bar", ["done bar"]),
    ],
)
def test_txt_sentence_boundaries_follow_final_replacements(
    tmp_path: Path,
    replacement: str,
    source_text: str,
    expected: list[str],
) -> None:
    first, second = source_text.split()
    payload = {
        "text": source_text,
        "words": [
            {"type": "word", "text": first, "start": 0, "end": 0.2},
            {"type": "word", "text": second, "start": 0.3, "end": 0.5},
        ],
    }
    options = ExportOptions(
        (ArtifactFormat.TXT,),
        tmp_path,
        text=TextOptions(replacements=(replacement,)),
    )

    rendered = render_artifact(ArtifactFormat.TXT, Transcript.from_payload(payload), payload, options)

    assert rendered.splitlines() == expected


@pytest.mark.parametrize(
    ("language_code", "first", "connector", "second"),
    [
        ("eng", "First", ("and",), "second"),
        ("uzb", "Avval", ("shuning", "uchun"), "davom"),
        ("kir", "Биринчи", ("андан", "кийин"), "экинчи"),
        ("rus", "Первый", ("потому", "что"), "второй"),
    ],
)
def test_language_connector_and_gap_break_txt_sentence(
    tmp_path: Path,
    language_code: str,
    first: str,
    connector: tuple[str, ...],
    second: str,
) -> None:
    connector_words = [
        {"type": "word", "text": word, "start": 1.2 + index * 0.3, "end": 1.5 + index * 0.3}
        for index, word in enumerate(connector)
    ]
    payload = {
        "language_code": language_code,
        "text": " ".join((first, *connector, second)),
        "words": [
            {"type": "word", "text": first, "start": 0, "end": 0.2},
            *connector_words,
            {"type": "word", "text": second, "start": 2.0, "end": 2.3},
        ],
    }
    options = ExportOptions((ArtifactFormat.TXT,), tmp_path)

    rendered = render_artifact(ArtifactFormat.TXT, Transcript.from_payload(payload), payload, options)

    assert rendered.splitlines() == [first, " ".join((*connector, second))]


def test_neutral_russian_replacement_does_not_use_uzbek_transliteration(tmp_path: Path) -> None:
    payload = {
        "language_code": "rus",
        "text": "Привет мир",
        "words": [
            {"type": "word", "text": "Привет", "start": 0, "end": 0.3},
            {"type": "word", "text": "мир", "start": 0.4, "end": 0.7},
        ],
    }
    options = ExportOptions(
        (ArtifactFormat.TXT,),
        tmp_path,
        text=TextOptions(replacements=("мир=МИР",)),
    )

    rendered = render_artifact(ArtifactFormat.TXT, Transcript.from_payload(payload), payload, options)

    assert rendered.strip() == "Привет МИР"


def test_uzbek_script_conversion_rejects_known_non_uzbek_transcript(tmp_path: Path) -> None:
    payload = {
        "language_code": "rus",
        "text": "Привет.",
        "words": [{"type": "word", "text": "Привет.", "start": 0, "end": 0.5}],
    }
    options = ExportOptions(
        (ArtifactFormat.SRT,),
        tmp_path,
        text=TextOptions(script=ScriptMode.LATIN),
    )

    with pytest.raises(ExportError, match="Uzbek-only"):
        render_artifact(ArtifactFormat.SRT, Transcript.from_payload(payload), payload, options)


def test_srt_length_split_uses_spoken_character_edges(tmp_path: Path) -> None:
    payload = {
        "text": "abcdefgh ijklmnop",
        "words": [
            {
                "type": "word",
                "text": "abcdefgh",
                "start": 0.0,
                "end": 1.0,
                "characters": [
                    {"text": "a", "start": 0.1, "end": 0.2},
                    {"text": "h", "start": 0.7, "end": 0.8},
                ],
            },
            {
                "type": "word",
                "text": "ijklmnop",
                "start": 1.0,
                "end": 2.0,
                "characters": [
                    {"text": "i", "start": 1.1, "end": 1.2},
                    {"text": "p", "start": 1.7, "end": 1.8},
                ],
            },
        ],
    }
    options = ExportOptions(
        (ArtifactFormat.SRT,),
        tmp_path,
        segmentation=SegmentationOptions(
            max_chars_per_line=8,
            max_lines=1,
            max_duration=10,
            min_duration=0,
        ),
    )

    rendered = render_artifact(ArtifactFormat.SRT, Transcript.from_payload(payload), payload, options)

    assert "00:00:00,100 --> 00:00:00,800" in rendered
    assert "00:00:01,100 --> 00:00:01,800" in rendered


def test_srt_length_split_enforces_100ms_gap(tmp_path: Path) -> None:
    payload = {
        "text": "abcdefgh ijklmnop",
        "words": [
            {"type": "word", "text": "abcdefgh", "start": 0.0, "end": 1.0},
            {"type": "word", "text": "ijklmnop", "start": 1.04, "end": 2.0},
        ],
    }
    options = ExportOptions(
        (ArtifactFormat.SRT,),
        tmp_path,
        segmentation=SegmentationOptions(
            max_chars_per_line=8,
            max_lines=1,
            max_duration=10,
            min_duration=0,
        ),
    )

    rendered = render_artifact(ArtifactFormat.SRT, Transcript.from_payload(payload), payload, options)

    assert "00:00:00,000 --> 00:00:00,970" in rendered
    assert "00:00:01,070 --> 00:00:02,000" in rendered
