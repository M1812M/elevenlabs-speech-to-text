import pytest

from elevenlabs_toolkit.models import Word
from elevenlabs_toolkit.segmentation.timings import cues_with_precise_gaps


def test_two_frame_padding_scales_with_frame_rate() -> None:
    group = ((Word("hello", 1.0, 2.0),),)

    at_30_fps = cues_with_precise_gaps(group, frames_per_second=30)
    at_60_fps = cues_with_precise_gaps(group, frames_per_second=60)

    assert (at_30_fps[0].start, at_30_fps[0].end) == (0.933, 2.067)
    assert (at_60_fps[0].start, at_60_fps[0].end) == (0.967, 2.033)


def test_padding_is_shared_around_an_80ms_minimum_gap() -> None:
    groups = (
        (Word("first", 1.0, 2.0),),
        (Word("second", 2.2, 3.2),),
    )

    cues = cues_with_precise_gaps(groups)

    assert (cues[0].start, cues[0].end) == (0.933, 2.06)
    assert (cues[1].start, cues[1].end) == (2.14, 3.267)
    assert cues[1].start - cues[0].end == pytest.approx(0.08)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frames_per_second": 0},
        {"frames_per_second": float("nan")},
        {"padding_frames": -1},
        {"padding_frames": 1.5},
        {"minimum_gap_milliseconds": -1},
        {"minimum_gap_milliseconds": 1.5},
    ],
)
def test_srt_timing_values_are_validated(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        cues_with_precise_gaps(((Word("hello", 1.0, 2.0),),), **kwargs)
