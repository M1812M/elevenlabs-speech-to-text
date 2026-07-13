import pytest

from elevenlabs_toolkit.models import Cue, Word
from elevenlabs_toolkit.renderers import render_resolve_edl


def _cue(start: float, end: float, text: str = "text") -> Cue:
    return Cue((Word(text, start, end),))


def test_render_resolve_edl_uses_rounded_nominal_base_for_fractional_fps() -> None:
    rendered = render_resolve_edl((_cue(1.0, 2.0),), title="Episode", fps=23.976)

    assert rendered == (
        "TITLE: Episode\n"
        "FCM: NON-DROP FRAME\n\n"
        "001  001      V     C        "
        "01:00:01:00 01:00:01:01 01:00:01:00 01:00:01:01\n"
        "|C:ResolveColorBlue |M:Sentence 1 |D:1\n"
    )


def test_fractional_fps_uses_actual_rate_for_elapsed_marker_time() -> None:
    rendered = render_resolve_edl((_cue(3600.0, 3601.0),), title="Long episode", fps=23.976)

    assert "01:59:56:10 01:59:56:11" in rendered


def test_render_resolve_edl_supports_fractional_timeline_start_and_bare_indices() -> None:
    rendered = render_resolve_edl(
        (_cue(0.0, 0.5),),
        title="Markers",
        fps=25.0,
        marker_prefix="",
        timeline_start_hours=0.5,
    )

    assert "00:30:00:00 00:30:00:01" in rendered
    assert "|M:1 |D:1" in rendered


def test_render_resolve_edl_sanitizes_structural_marker_characters() -> None:
    rendered = render_resolve_edl(
        (_cue(0.0, 0.5),),
        title="My\nTitle",
        color="Blue|bad",
        marker_prefix="Shot | primary",
    )

    assert rendered.startswith("TITLE: My Title\n")
    assert "|C:Blue bad |M:Shot primary 1 |D:1" in rendered


@pytest.mark.parametrize("fps", [0, -25, float("nan"), float("inf")])
def test_render_resolve_edl_rejects_invalid_fps(fps: float) -> None:
    with pytest.raises(ValueError, match="fps"):
        render_resolve_edl((), title="Invalid", fps=fps)


@pytest.mark.parametrize("hours", [-1, float("nan"), float("inf")])
def test_render_resolve_edl_rejects_invalid_timeline_start(hours: float) -> None:
    with pytest.raises(ValueError, match="timeline_start_hours"):
        render_resolve_edl((), title="Invalid", timeline_start_hours=hours)
