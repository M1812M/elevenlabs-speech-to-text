from pathlib import Path

import pytest

from elevenlabs_toolkit.application.desktop import DesktopSettings, prepare_files
from elevenlabs_toolkit.application.planner import PlanningError
from elevenlabs_toolkit.models import ArtifactFormat, ConflictPolicy, ScriptMode, TranscriptionOptions


def test_transcription_same_stem_keeps_distinct_json_outputs(tmp_path: Path) -> None:
    sources = (tmp_path / "audio.wav", tmp_path / "audio.flac")
    for source in sources:
        source.write_bytes(b"audio")
    planned = prepare_files(
        "Transcribe",
        sources,
        ScriptMode.SOURCE,
        ConflictPolicy.SKIP,
        (),
        DesktopSettings(project_directory=tmp_path),
    )
    assert [artifact.target for item in planned for artifact in item.plan.artifacts] == [
        tmp_path / "audio.wav.transcript.json",
        tmp_path / "audio.flac.transcript.json",
    ]
    assert all(item.plan.api_requests == 1 for item in planned)


def test_preflight_rejects_output_overwriting_another_selected_source(tmp_path: Path) -> None:
    source = tmp_path / "text.txt"
    source.write_text("Salom")
    generated = tmp_path / "text.cyrillic.txt"
    generated.write_text("Keep me")
    with pytest.raises(PlanningError, match="overwrite"):
        prepare_files(
            "Convert Script",
            (source, generated),
            ScriptMode.CYRILLIC,
            ConflictPolicy.REPLACE,
            (),
            DesktopSettings(project_directory=tmp_path),
        )
    assert generated.read_text() == "Keep me"


def test_export_configuration_uses_selected_context_and_explicit_overrides(tmp_path: Path) -> None:
    (tmp_path / "elevenlabs-toolkit.toml").write_text(
        '[profiles.custom.text]\nscript = "latin"\nreplacements = ["Salom=Assalom"]\n'
    )
    source = tmp_path / "transcript.json"
    source.write_text("{}")
    settings = DesktopSettings(
        project_directory=tmp_path,
        profile="custom",
        output_script=ScriptMode.CYRILLIC,
        replacements=(),
    )
    (item,) = prepare_files(
        "Export",
        (source,),
        ScriptMode.SOURCE,
        ConflictPolicy.SKIP,
        (ArtifactFormat.SRT, ArtifactFormat.TXT),
        settings,
    )
    assert item.export is not None
    assert item.export.text.script is ScriptMode.CYRILLIC
    assert item.export.text.replacements == ()
    assert {artifact.target for artifact in item.plan.artifacts} == {
        source.with_suffix(".srt"),
        source.with_suffix(".txt"),
    }


def test_clean_forces_uzbek_cleanup_and_clean_derivative(tmp_path: Path) -> None:
    source = tmp_path / "transcript.json"
    source.write_text("{}")
    (item,) = prepare_files(
        "Clean",
        (source,),
        ScriptMode.SOURCE,
        ConflictPolicy.SKIP,
        (),
        DesktopSettings(project_directory=tmp_path, cleanup="none"),
    )
    assert item.export is not None
    assert item.export.text.cleanup == "uzbek"
    assert item.plan.artifacts[0].target == tmp_path / "transcript.clean.json"


def test_txt_conversion_has_txt_plan_and_strips_previous_script_suffix(tmp_path: Path) -> None:
    source = tmp_path / "text.latin.txt"
    source.write_text("Salom")
    (item,) = prepare_files(
        "Convert Script",
        (source,),
        ScriptMode.CYRILLIC,
        ConflictPolicy.SKIP,
        (),
        DesktopSettings(project_directory=tmp_path),
    )
    assert item.plan.artifacts[0].format is ArtifactFormat.TXT
    assert item.plan.artifacts[0].target == tmp_path / "text.cyrillic.txt"


def test_combined_export_groups_sources_by_directory(tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    sources = (first / "z.json", first / "a.json", second / "one.json")
    for path in sources:
        path.write_text("{}")
    prepared = prepare_files(
        "Export",
        sources,
        ScriptMode.SOURCE,
        ConflictPolicy.SKIP,
        (ArtifactFormat.SRT, ArtifactFormat.COMBINED_TXT),
        DesktopSettings(project_directory=tmp_path, combined_name="all.txt"),
    )
    combined = [item for item in prepared if item.export and item.export.formats == (ArtifactFormat.COMBINED_TXT,)]
    assert len(combined) == 2
    assert combined[0].plan.sources == (first / "a.json", first / "z.json")
    assert combined[0].plan.artifacts[0].target == first / "all.txt"
    assert combined[1].plan.artifacts[0].target == second / "all.txt"


def test_subtitle_overrides_are_validated_in_preflight(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    source.write_text("{}")
    with pytest.raises(ValueError, match="srt_fps"):
        prepare_files(
            "Export",
            (source,),
            ScriptMode.SOURCE,
            ConflictPolicy.SKIP,
            (ArtifactFormat.SRT,),
            DesktopSettings(project_directory=tmp_path, segmentation=(("srt_fps", -1.0),)),
        )


def test_transcription_plans_selected_local_and_remote_formats(tmp_path: Path) -> None:
    source = tmp_path / "audio.wav"
    source.write_bytes(b"audio")
    (item,) = prepare_files(
        "Transcribe",
        (source,),
        ScriptMode.SOURCE,
        ConflictPolicy.SKIP,
        (),
        DesktopSettings(
            project_directory=tmp_path,
            transcription=TranscriptionOptions(language_code="uz", remote_formats=("pdf",)),
            additional_formats=(ArtifactFormat.SRT,),
        ),
    )
    assert item.transcription.language_code == "uz"
    assert {artifact.target.name for artifact in item.plan.artifacts} == {
        "audio.wav.transcript.json",
        "audio.wav.transcript.srt",
        "audio.wav.transcript.pdf",
    }
