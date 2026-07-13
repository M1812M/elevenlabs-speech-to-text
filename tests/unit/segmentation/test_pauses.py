from elevenlabs_toolkit.models import SegmentationOptions, Transcript
from elevenlabs_toolkit.segmentation import segment_standard, sentences_from_transcript
from elevenlabs_toolkit.segmentation.pauses import (
    detect_stretched_character_pause_end,
    effective_word_end,
)


def _stretched_word(text: str = "narsa", start: float = 460.19) -> dict:
    return {
        "type": "word",
        "text": text,
        "start": start,
        "end": start + 1.61,
        "characters": [
            {"text": "n", "start": start, "end": start + 0.13},
            {"text": "a", "start": start + 0.13, "end": start + 0.15},
            {"text": "r", "start": start + 0.15, "end": start + 0.31},
            {"text": "s", "start": start + 0.31, "end": start + 0.35},
            {"text": "a", "start": start + 0.35, "end": start + 1.61},
        ],
    }


def test_stretched_final_character_produces_an_earlier_effective_end() -> None:
    word = _stretched_word()

    detected = detect_stretched_character_pause_end(word)

    assert detected is not None
    assert 460.65 < detected < 461.0
    assert effective_word_end(word, pause_detection=True) == detected


def test_normal_character_timing_is_not_adjusted() -> None:
    word = {
        "type": "word",
        "text": "ovozi",
        "start": 462.28,
        "end": 462.82,
        "characters": [
            {"text": "o", "start": 462.28, "end": 462.36},
            {"text": "v", "start": 462.36, "end": 462.38},
            {"text": "o", "start": 462.38, "end": 462.60},
            {"text": "z", "start": 462.60, "end": 462.61},
            {"text": "i", "start": 462.61, "end": 462.82},
        ],
    }

    assert detect_stretched_character_pause_end(word) is None
    assert effective_word_end(word, pause_detection=True) == 462.82


def test_detected_pause_splits_subtitles_and_sentences_without_losing_words() -> None:
    stretched = _stretched_word(start=1.0)
    payload = {
        "words": [
            {"type": "word", "text": "before", "start": 0.5, "end": 0.9},
            stretched,
            {"type": "word", "text": "after", "start": 2.67, "end": 2.9},
        ]
    }
    transcript = Transcript.from_payload(payload)
    plain_options = SegmentationOptions(min_duration=0, pause_detection=False)
    pause_options = SegmentationOptions(min_duration=0, pause_detection=True)

    plain_cues = segment_standard(transcript, plain_options)
    pause_cues = segment_standard(transcript, pause_options)
    plain_sentences = sentences_from_transcript(transcript, plain_options)
    pause_sentences = sentences_from_transcript(transcript, pause_options)

    assert [cue.text for cue in plain_cues] == ["before narsa after"]
    assert [cue.text for cue in pause_cues] == ["before narsa", "after"]
    assert pause_cues[0].end < plain_cues[0].end
    assert [sentence.text for sentence in plain_sentences] == ["before narsa after"]
    assert [sentence.text for sentence in pause_sentences] == ["before narsa", "after"]
