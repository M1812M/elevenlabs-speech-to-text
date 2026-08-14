from pathlib import Path

import pytest

from elevenlabs_toolkit.models import (
    ArtifactFormat,
    ArtifactResult,
    ArtifactStatus,
    InputSpec,
    JobResult,
    PlannedArtifact,
    SegmentationOptions,
    TextOptions,
    TranscriptionOptions,
)


def test_input_selection_rejects_ambiguous_patterns() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        InputSpec((Path("input"),), glob="*.json", regex=".*[.]json")


def test_option_constraints_fail_early() -> None:
    with pytest.raises(ValueError, match="min_duration"):
        SegmentationOptions(min_duration=6, max_duration=5)
    with pytest.raises(ValueError, match="num_speakers"):
        TranscriptionOptions(num_speakers=33)


def test_character_timestamps_are_the_transcription_default() -> None:
    assert TranscriptionOptions().timestamps_granularity == "character"


def test_srt_timing_defaults_use_two_frames_at_30fps_and_80ms_gap() -> None:
    options = SegmentationOptions()

    assert options.srt_fps == 30.0
    assert options.srt_padding_frames == 2
    assert options.srt_gap_milliseconds == 80


def test_job_result_exposes_stable_counts_and_exit_code() -> None:
    artifact = PlannedArtifact(Path("in.json"), Path("out.srt"), ArtifactFormat.SRT)
    result = JobResult(
        (
            ArtifactResult(artifact, ArtifactStatus.WRITTEN),
            ArtifactResult(artifact, ArtifactStatus.SKIPPED),
            ArtifactResult(artifact, ArtifactStatus.FAILED, "boom"),
        )
    )

    assert result.written == 1
    assert result.skipped == 1
    assert result.failed == 1
    assert result.succeeded == 2
    assert result.exit_code == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_lines": True},
        {"gap_seconds": float("nan")},
        {"max_duration": float("inf")},
        {"preset": None},
        {"srt_fps": 0},
        {"srt_padding_frames": -1},
        {"srt_gap_milliseconds": -1},
    ],
)
def test_segmentation_rejects_wrongly_typed_or_nonfinite_values(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SegmentationOptions(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"language_code": 1},
        {"remote_formats": (1,)},
        {"temperature": float("nan")},
        {"no_verbatim": True, "model_id": "scribe_v1"},
        {"keyterms": ("six words are not allowed here",)},
        {"keyterms": ("bad<term",)},
    ],
)
def test_transcription_options_reject_invalid_provider_constraints(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        TranscriptionOptions(**kwargs)


@pytest.mark.parametrize("replacement", ["missing-equals", "TOKEN=", "two words=target", "a=two words"])
def test_text_replacements_are_unambiguous_single_tokens(replacement: str) -> None:
    with pytest.raises(ValueError, match="TOKEN=TOKEN"):
        TextOptions(replacements=(replacement,))
