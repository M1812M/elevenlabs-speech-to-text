import json
from pathlib import Path

import pytest

from elevenlabs_toolkit.application import PlanningError, build_manifest, plan_exports, plan_transcription
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
    assert any(conflict.reason == "multiple inputs map to the same output" for conflict in plan.conflicts)


def test_resume_reuses_json_but_still_plans_missing_local_outputs(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"audio")
    output = tmp_path / "out"
    output.mkdir()
    options = TranscriptionOptions()
    transcript = output / "clip.json"
    transcript_content = '{"text": "hello"}\n'
    transcript.write_text(transcript_content, encoding="utf-8")
    (output / "clip.manifest.json").write_text(
        json.dumps(
            build_manifest(
                source,
                options,
                transcript_name=transcript.name,
                transcript_content=transcript_content,
            )
        ),
        encoding="utf-8",
    )

    plan = plan_transcription(
        (source,),
        output,
        (ArtifactFormat.SRT,),
        policy=ConflictPolicy.SKIP,
        resume=True,
        transcription_options=options,
    )

    assert plan.api_requests == 0
    assert any(item.format is ArtifactFormat.SRT for item in plan.artifacts)


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


def test_stale_cache_with_skip_blocks_before_api_work(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"audio")
    output = tmp_path / "out"
    output.mkdir()
    (output / "clip.json").write_text('{"text":"stale"}', encoding="utf-8")

    plan = plan_transcription(
        (source,),
        output,
        (ArtifactFormat.SRT,),
        policy=ConflictPolicy.SKIP,
        transcription_options=TranscriptionOptions(),
    )

    assert not plan.valid
    assert plan.api_requests == 0
    assert any("stale or incomplete cache" in conflict.reason for conflict in plan.conflicts)


def test_transcription_rejects_rename_before_planning(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"audio")

    with pytest.raises(PlanningError, match="rename is not supported"):
        plan_transcription(
            (source,),
            tmp_path / "out",
            (),
            policy=ConflictPolicy.RENAME,
            transcription_options=TranscriptionOptions(),
        )


@pytest.mark.parametrize("name", ["bad:name.mp3", "CON.mp3", "trailing..mp3"])
def test_transcription_rejects_unsafe_output_stems_before_api_work(tmp_path: Path, name: str) -> None:
    source = tmp_path / name

    with pytest.raises(PlanningError, match="portable output name"):
        plan_transcription(
            (source,),
            tmp_path / "out",
            (),
            transcription_options=TranscriptionOptions(),
        )


def test_manifest_suffix_in_media_stem_has_distinct_cache_names(tmp_path: Path) -> None:
    source = tmp_path / "episode.manifest.mp3"
    source.write_bytes(b"audio")

    plan = plan_transcription(
        (source,),
        tmp_path / "out",
        (),
        transcription_options=TranscriptionOptions(),
    )

    assert plan.valid
    assert {artifact.target.name for artifact in plan.artifacts} == {
        "episode.manifest.json",
        "episode.manifest.manifest.json",
    }


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
