import pytest

from elevenlabs_toolkit.models import SegmentationOptions, Transcript
from elevenlabs_toolkit.renderers import wrap_text_lossless
from elevenlabs_toolkit.segmentation import segment_standard, segment_transcript


def _transcript(words: list[dict]) -> Transcript:
    return Transcript.from_payload({"words": [{"type": "word", **word} for word in words]})


def test_orphan_rebalance_preserves_words_and_moves_timing() -> None:
    texts = [
        "shunda",
        "shu",
        "ovozni",
        "eshitganman,",
        "hushimdan",
        "ketganman-da,",
        "uje,",
        "uje",
        "skoriy",
        "kelyapti",
        "olgani.",
    ]
    words = [
        {"text": text, "start": 26 + index * 0.55, "end": 26 + index * 0.55 + 0.25} for index, text in enumerate(texts)
    ]

    cues = segment_standard(_transcript(words), SegmentationOptions())

    assert " ".join(cue.text for cue in cues) == " ".join(texts)
    assert cues[-1].text == "kelyapti olgani."
    assert cues[-1].start == words[-2]["start"]
    assert cues[-2].end == words[-3]["end"]


def test_speaker_transition_starts_a_new_cue() -> None:
    transcript = _transcript(
        [
            {"text": "one", "start": 0, "end": 0.3, "speaker_id": "speaker_0"},
            {"text": "two", "start": 0.4, "end": 0.7, "speaker_id": "speaker_1"},
        ]
    )

    cues = segment_standard(transcript, SegmentationOptions(min_duration=0))

    assert [cue.text for cue in cues] == ["one", "two"]


def test_segments_only_payload_still_renders_as_a_timed_cue() -> None:
    transcript = Transcript.from_payload({"segments": [{"text": "segment fallback", "start": 2, "end": 3}]})

    cues = segment_transcript(transcript, SegmentationOptions())

    assert [(cue.start, cue.end, cue.text) for cue in cues] == [(2.0, 3.0, "segment fallback")]


def test_social_preset_obeys_word_limit_without_losing_words() -> None:
    transcript = _transcript(
        [{"text": f"word{index}", "start": index * 0.2, "end": index * 0.2 + 0.1} for index in range(8)]
    )
    options = SegmentationOptions(
        preset="social",
        max_chars_per_line=100,
        max_duration=10,
        min_duration=0,
        max_words=3,
    )

    cues = segment_transcript(transcript, options)

    assert all(len(cue.words) <= 3 for cue in cues)
    assert [word.text for cue in cues for word in cue.words] == [f"word{index}" for index in range(8)]


def test_social_punctuation_split_keeps_provider_timing() -> None:
    transcript = _transcript(
        [
            {"text": "ibodat", "start": 410.88, "end": 411.38},
            {"text": "qiladigan,", "start": 411.42, "end": 413.00},
            {"text": "kalom", "start": 413.04, "end": 413.47},
            {"text": "o'qiydikan,", "start": 413.56, "end": 414.16},
        ]
    )
    options = SegmentationOptions(
        preset="social",
        max_chars_per_line=100,
        max_duration=10,
        min_duration=0,
    )

    cues = segment_transcript(transcript, options)

    assert [(cue.start, cue.end, cue.text) for cue in cues] == [
        (410.88, 413.00, "ibodat qiladigan,"),
        (413.04, 414.16, "kalom o'qiydikan,"),
    ]


def test_orphan_rebalance_never_crosses_a_hard_gap() -> None:
    transcript = _transcript(
        [
            {"text": "one", "start": 0.0, "end": 0.1},
            {"text": "two", "start": 0.2, "end": 0.3},
            {"text": "three", "start": 0.4, "end": 0.5},
            {"text": "four", "start": 4.0, "end": 4.2},
        ]
    )

    cues = segment_standard(transcript, SegmentationOptions())

    assert [cue.text for cue in cues] == ["one two three", "four"]
    assert cues[0].end == 0.5
    assert cues[1].start == 4.0


@pytest.mark.parametrize("preset", ["standard", "social"])
def test_every_input_word_occurs_exactly_once(preset: str) -> None:
    transcript = _transcript(
        [
            {
                "text": f"token-{index}",
                "start": index * 0.35 + (2.0 if index >= 7 else 0.0),
                "end": index * 0.35 + (2.0 if index >= 7 else 0.0) + 0.2,
                "speaker_id": f"speaker_{index // 5}",
            }
            for index in range(15)
        ]
    )
    options = SegmentationOptions(
        preset=preset,
        max_chars_per_line=12,
        max_lines=2,
        max_duration=2.0,
        min_duration=0.4,
        max_words=4 if preset == "social" else None,
    )

    cues = segment_transcript(transcript, options)

    assert [word.text for cue in cues for word in cue.words] == [f"token-{index}" for index in range(15)]


def test_multiline_fit_uses_real_word_wrap_after_text_transformation() -> None:
    transcript = _transcript(
        [{"text": "\u0449", "start": index * 0.2, "end": index * 0.2 + 0.1} for index in range(10)]
    )
    options = SegmentationOptions(
        max_chars_per_line=10,
        max_lines=2,
        max_duration=10,
        min_duration=0,
    )

    cues = segment_standard(transcript, options, lambda text: text.replace("\u0449", "sh"))

    assert len(cues) > 1
    assert all(
        len(line) <= options.max_chars_per_line
        for cue in cues
        for line in wrap_text_lossless(
            cue.text.replace("\u0449", "sh"),
            options.max_chars_per_line,
            options.max_lines,
        ).splitlines()
    )
