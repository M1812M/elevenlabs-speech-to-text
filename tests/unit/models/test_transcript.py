import pytest

from elevenlabs_toolkit.models import Cue, Segment, Transcript, TranscriptValidationError, Word


def test_payload_normalizes_punctuation_and_speaker_fields() -> None:
    payload = {
        "language_code": "uzb",
        "text": "Salom, dunyo!",
        "words": [
            {"type": "word", "text": "Salom", "start": 0, "end": 0.4, "speaker_id": "speaker_1"},
            {"type": "punctuation", "text": ","},
            {"type": "spacing", "text": " "},
            {"type": "word", "text": "dunyo", "start": 0.5, "end": 0.9, "speaker": "speaker_1"},
            {"type": "punctuation", "text": "!"},
        ],
    }

    transcript = Transcript.from_payload(payload)

    assert [word.text for word in transcript.words] == ["Salom,", "dunyo!"]
    assert transcript.language_code == "uzb"
    assert transcript.words[0].speaker == "speaker_1"
    assert transcript.raw_payload == payload


def test_sdk_spacing_punctuation_and_audio_events_are_not_lost() -> None:
    transcript = Transcript.from_payload(
        {
            "text": "Hello, [music] world!",
            "words": [
                {"type": "word", "text": "Hello", "start": 0, "end": 0.3},
                {"type": "spacing", "text": ", "},
                {"type": "audio_event", "text": "[music]", "start": 0.4, "end": 0.8},
                {"type": "spacing", "text": " "},
                {"type": "word", "text": "world", "start": 0.9, "end": 1.2},
                {"type": "spacing", "text": "!"},
            ],
        }
    )

    assert [word.text for word in transcript.words] == ["Hello,", "[music]", "world!"]
    assert [word.kind for word in transcript.words] == ["word", "audio_event", "word"]
    assert [item["type"] for item in transcript.to_payload()["words"]] == ["word", "audio_event", "word"]


def test_cue_derives_timing_after_moving_a_word() -> None:
    first = Word("first", 1.0, 1.2)
    moved = Word("moved", 1.4, 1.6)
    last = Word("last", 1.8, 2.0)

    cue = Cue((moved, last))

    assert cue.start == 1.4
    assert cue.end == 2.0
    assert cue.text == "moved last"
    assert Cue((first,)).end == 1.2


def test_payload_errors_include_the_invalid_field_path() -> None:
    with pytest.raises(TranscriptValidationError, match=r"words\[0\]\.start"):
        Transcript.from_payload({"words": [{"type": "word", "text": "bad", "start": "later", "end": 1}]})


def test_segments_are_available_as_timed_words_when_words_are_absent() -> None:
    transcript = Transcript.from_payload({"segments": [{"text": "whole segment", "start": 1, "end": 2}]})

    assert transcript.timed_words == (Word("whole segment", 1.0, 2.0),)


def test_cue_end_uses_latest_word_end_for_overlapping_words() -> None:
    cue = Cue((Word("long", 0.0, 10.0), Word("short", 1.0, 2.0)))

    assert cue.end == 10.0


@pytest.mark.parametrize(
    "payload",
    [
        {"request_id": "metadata-only"},
        {"words": [{"type": "word", "text": " ", "start": 0, "end": 1}]},
        {"segments": [{"text": "missing timing"}]},
        {"words": [{"type": "word", "text": "bad", "start": float("nan"), "end": 1}]},
        {"words": [{"type": "word", "text": "bad", "start": False, "end": 1}]},
        {"words": [{"type": 0, "text": "bad", "start": 0, "end": 1}]},
        {"segments": [{"text": 0, "start": 0, "end": 1}]},
        {
            "words": [
                {
                    "type": "word",
                    "text": "bad",
                    "start": 0,
                    "end": 1,
                    "characters": [{"text": 0, "start": 0, "end": 1}],
                }
            ]
        },
    ],
)
def test_malformed_or_nonfinite_transcript_payloads_are_rejected(payload: dict) -> None:
    with pytest.raises(TranscriptValidationError):
        Transcript.from_payload(payload)


def test_nullable_words_form_an_untimed_text_transcript() -> None:
    transcript = Transcript.from_payload(
        {
            "words": [
                {"type": "word", "text": "hello", "start": None, "end": None},
                {"type": "punctuation", "text": ","},
                {"type": "word", "text": "world", "start": None, "end": None},
            ]
        }
    )

    assert transcript.text == "hello, world"
    assert transcript.timed_words == ()


def test_mixed_timed_and_untimed_words_are_rejected() -> None:
    with pytest.raises(TranscriptValidationError, match="every word or no words"):
        Transcript.from_payload(
            {
                "words": [
                    {"type": "word", "text": "timed", "start": 0, "end": 1},
                    {"type": "word", "text": "untimed", "start": None, "end": None},
                ]
            }
        )


def test_source_character_timings_remain_available_on_clean_derivatives() -> None:
    transcript = Transcript.from_payload(
        {
            "words": [
                {
                    "type": "word",
                    "text": "salom",
                    "start": 0,
                    "end": 1,
                    "source_text": "салом",
                    "source_characters": [{"text": "s", "start": 0, "end": 0.2}],
                }
            ]
        }
    )

    assert transcript.words[0].characters[0].start == 0.0
    serialized = transcript.to_payload()["words"][0]
    assert serialized["source_text"] == "салом"
    assert serialized["source_characters"] == [{"text": "s", "start": 0.0, "end": 0.2}]
    assert "characters" not in serialized


def test_direct_segment_rejects_whitespace_text() -> None:
    with pytest.raises(TranscriptValidationError, match="segment text"):
        Segment(" ", 0, 1)
