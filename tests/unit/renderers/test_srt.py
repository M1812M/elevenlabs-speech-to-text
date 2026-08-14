import pytest

from elevenlabs_toolkit.models import Cue, Word
from elevenlabs_toolkit.renderers import render_cue_index_srt, render_srt, wrap_text_lossless


def _cue(*words: Word) -> Cue:
    return Cue(tuple(words))


def test_wrap_text_lossless_balances_two_lines() -> None:
    wrapped = wrap_text_lossless("one two three four", max_chars_per_line=9, max_lines=2)

    assert wrapped == "one two\nthree four"
    assert wrapped.replace("\n", " ") == "one two three four"


def test_wrap_text_lossless_preserves_every_word_when_width_is_impossible() -> None:
    source = "alpha beta gamma delta epsilon zeta eta theta iota"

    wrapped = wrap_text_lossless(source, max_chars_per_line=8, max_lines=2)

    assert wrapped.count("\n") == 1
    assert wrapped.replace("\n", " ") == source
    assert any(len(line) > 8 for line in wrapped.splitlines())


@pytest.mark.parametrize(
    ("name", "value", "exception"),
    [
        ("max_chars_per_line", 0, ValueError),
        ("max_lines", -1, ValueError),
        ("max_lines", 2.0, TypeError),
    ],
)
def test_wrap_text_lossless_validates_limits(name: str, value: object, exception: type[Exception]) -> None:
    kwargs = {name: value}
    with pytest.raises(exception):
        wrap_text_lossless("some text", **kwargs)  # type: ignore[arg-type]


def test_render_srt_keeps_cue_text_on_one_line_by_default() -> None:
    cues = (
        _cue(
            Word("hello", 0.0, 0.5),
            Word("world", 0.6, 1.25),
        ),
    )

    rendered = render_srt(
        cues,
        text_transform=str.upper,
        max_chars_per_line=6,
        max_lines=2,
    )

    assert rendered == "1\n00:00:00,000 --> 00:00:01,250\nHELLO WORLD\n"


def test_render_srt_uses_requested_smart_line_breaks() -> None:
    cues = (_cue(Word("hello", 0.0, 0.5), Word("world", 0.6, 1.25)),)

    rendered = render_srt(
        cues,
        text_transform=str.upper,
        max_chars_per_line=6,
        max_lines=2,
        smart_line_breaks=True,
    )

    assert rendered == "1\n00:00:00,000 --> 00:00:01,250\nHELLO\nWORLD\n"


def test_render_srt_returns_empty_string_for_no_cues() -> None:
    assert render_srt(()) == ""


def test_render_cue_index_srt_uses_visible_one_based_indices() -> None:
    cues = (
        _cue(Word("first", 1.0, 1.5)),
        _cue(Word("second", 2.0, 2.75)),
    )

    assert render_cue_index_srt(cues) == (
        "1\n00:00:01,000 --> 00:00:01,500\n1\n\n2\n00:00:02,000 --> 00:00:02,750\n2\n"
    )
