from elevenlabs_toolkit.models import SegmentationOptions, Transcript
from elevenlabs_toolkit.segmentation import sentences_from_transcript


def test_sentence_segmentation_uses_punctuation_and_dominant_speaker() -> None:
    transcript = Transcript.from_payload(
        {
            "words": [
                {"type": "word", "text": "Hello", "start": 0, "end": 0.2, "speaker_id": "a"},
                {"type": "word", "text": "world.", "start": 0.3, "end": 0.6, "speaker_id": "a"},
                {"type": "word", "text": "Next", "start": 0.7, "end": 0.9, "speaker_id": "b"},
            ]
        }
    )

    sentences = sentences_from_transcript(transcript, SegmentationOptions())

    assert [(item.text, item.speaker) for item in sentences] == [("Hello world.", "a"), ("Next", "b")]


def test_marker_and_gap_can_split_unpunctuated_speech() -> None:
    transcript = Transcript.from_payload(
        {
            "language_code": "uzb",
            "words": [
                {"type": "word", "text": "first", "start": 0, "end": 0.2},
                {"type": "word", "text": "keyin", "start": 1.3, "end": 1.5},
                {"type": "word", "text": "second", "start": 1.6, "end": 1.9},
            ],
        }
    )

    sentences = sentences_from_transcript(
        transcript,
        SegmentationOptions(),
    )

    assert [item.text for item in sentences] == ["first", "keyin second"]
