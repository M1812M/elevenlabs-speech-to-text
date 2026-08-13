from pathlib import Path

import pytest

from elevenlabs_toolkit.application import PlanningError, plan_exports, plan_transcription
from elevenlabs_toolkit.models import ArtifactFormat, ConflictPolicy, ExportOptions, TranscriptionOptions


def test_existing_output_is_a_preflight_conflict(tmp_path: Path) -> None:
    source = tmp_path / "input" / "sample.json"
    source.parent.mkdir()
    source.write_text("{}", encoding="utf-8")
    output = tmp_path / "out"
    output.mkdir()
    (output / "sample.srt").write_text("edited", encoding="utf-8")

    plan = plan_exports((source,), ExportOptions((ArtifactFormat.SRT,), output))

    assert not plan.valid
    assert plan.conflicts[0].reason == "output already exists"


def test_rename_policy_allocates_before_execution(tmp_path: Path) -> None:
    source = tmp_path / "sample.json"
    source.write_text("{}", encoding="utf-8")
    output = tmp_path / "out"
    output.mkdir()
    (output / "sample.srt").write_text("edited", encoding="utf-8")

    plan = plan_exports(
        (source,),
        ExportOptions((ArtifactFormat.SRT,), output),
        policy=ConflictPolicy.RENAME,
    )

    assert plan.valid
    assert plan.artifacts[0].target.name == "sample (2).srt"


def test_transcription_plans_requested_json_and_derived_outputs(tmp_path: Path) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"wav")

    plan = plan_transcription(
        (source,),
        tmp_path / "out",
        (ArtifactFormat.JSON, ArtifactFormat.SRT, ArtifactFormat.TXT),
        transcription_options=TranscriptionOptions(),
    )

    assert plan.valid
    assert plan.api_requests == 1
    assert plan.provider == "elevenlabs"
    assert {output.target.name for output in plan.artifacts} == {"clip.json", "clip.srt", "clip.txt"}


def test_transcription_detects_same_stem_collisions_before_api_work(tmp_path: Path) -> None:
    first = tmp_path / "clip.mp3"
    second = tmp_path / "clip.wav"
    first.write_bytes(b"mp3")
    second.write_bytes(b"wav")

    plan = plan_transcription(
        (first, second),
        tmp_path / "out",
        (ArtifactFormat.SRT,),
        transcription_options=TranscriptionOptions(),
    )

    assert not plan.valid
    assert plan.api_requests == 0
    assert any(conflict.reason == "multiple inputs map to the same output" for conflict in plan.conflicts)


def test_skip_avoids_api_only_when_every_output_for_source_exists(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"audio")
    output = tmp_path / "out"
    output.mkdir()
    (output / "clip.srt").write_text("edited", encoding="utf-8")
    formats = (ArtifactFormat.SRT, ArtifactFormat.TXT)

    partial = plan_transcription(
        (source,),
        output,
        formats,
        policy=ConflictPolicy.SKIP,
        transcription_options=TranscriptionOptions(),
    )
    (output / "clip.txt").write_text("edited", encoding="utf-8")
    complete = plan_transcription(
        (source,),
        output,
        formats,
        policy=ConflictPolicy.SKIP,
        transcription_options=TranscriptionOptions(),
    )

    assert partial.api_requests == 1
    assert complete.api_requests == 0


def test_recursive_sources_preserve_relative_directories(tmp_path: Path) -> None:
    first = tmp_path / "inputs" / "a" / "sample.json"
    second = tmp_path / "inputs" / "b" / "sample.json"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")

    plan = plan_exports(
        (first, second),
        ExportOptions((ArtifactFormat.SRT,), tmp_path / "out"),
    )

    assert plan.valid
    assert {item.target.relative_to(tmp_path / "out") for item in plan.artifacts} == {
        Path("a/sample.srt"),
        Path("b/sample.srt"),
    }


def test_transcription_rejects_rename_before_planning(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"audio")

    with pytest.raises(PlanningError, match="rename is not supported"):
        plan_transcription(
            (source,),
            tmp_path / "out",
            (ArtifactFormat.SRT,),
            policy=ConflictPolicy.RENAME,
            transcription_options=TranscriptionOptions(),
        )


def test_transcription_accepts_provider_json_as_an_output(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"audio")

    plan = plan_transcription(
        (source,),
        tmp_path / "out",
        (ArtifactFormat.JSON,),
        transcription_options=TranscriptionOptions(),
    )

    assert plan.valid
    assert [output.target.name for output in plan.artifacts] == ["clip.json"]


@pytest.mark.parametrize("name", ["bad:name.mp3", "CON.mp3", "trailing..mp3"])
def test_transcription_rejects_unsafe_output_stems_before_api_work(tmp_path: Path, name: str) -> None:
    source = tmp_path / name

    with pytest.raises(PlanningError, match="portable output name"):
        plan_transcription(
            (source,),
            tmp_path / "out",
            (ArtifactFormat.TXT,),
            transcription_options=TranscriptionOptions(),
        )


@pytest.mark.parametrize(
    "name",
    ["", "../combined.txt", "folder/combined.txt", "..", "CON.txt", "bad:name.txt", "trailing."],
)
def test_combined_output_name_must_be_a_leaf(tmp_path: Path, name: str) -> None:
    source = tmp_path / "sample.json"
    source.write_text('{"text":"hello"}', encoding="utf-8")
    options = ExportOptions((ArtifactFormat.COMBINED_TXT,), tmp_path / "out")

    with pytest.raises(PlanningError, match="combined_name"):
        plan_exports((source,), options, combined_name=name)


def test_output_cannot_alias_its_input_through_a_parent_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    source = real / "sample.json"
    source.write_text('{"text":"hello"}', encoding="utf-8")
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    plan = plan_exports(
        (source,),
        ExportOptions((ArtifactFormat.JSON,), alias),
        policy=ConflictPolicy.REPLACE,
    )

    assert not plan.valid
    assert plan.conflicts[0].reason == "output would overwrite its input"
