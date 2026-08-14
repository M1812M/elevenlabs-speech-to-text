import pytest

from elevenlabs_toolkit.models import Transcript
from elevenlabs_toolkit.segmentation import MiniSrtError, segment_mini


def _transcript(items: list[tuple[str, float, float]]) -> Transcript:
    return Transcript.from_payload(
        {
            "text": " ".join(text for text, _start, _end in items),
            "words": [{"type": "word", "text": text, "start": start, "end": end} for text, start, end in items],
        }
    )


def test_complete_sentences_are_separate_and_keep_all_words() -> None:
    transcript = _transcript(
        [
            ("Tug'ruq", 10.66, 11.1),
            ("paytida", 11.11, 11.4),
            ("chaqaloqning", 11.5, 12.0),
            ("bo'yniga", 12.1, 12.5),
            ("kindik", 12.6, 13.0),
            ("o'ralishi", 13.1, 13.6),
            ("ko'p", 13.7, 14.0),
            ("uchraydigan", 14.1, 14.6),
            ("holatdir.", 14.7, 15.12),
            ("Bu", 15.16, 15.5),
            ("deyarli", 15.6, 16.0),
            ("har", 16.1, 16.3),
            ("uchta", 16.4, 16.8),
            ("tug'ruqdan", 16.9, 17.5),
            ("birida", 17.6, 18.0),
            ("kuzatiladi.", 18.1, 19.46),
        ]
    )

    cues = segment_mini(transcript)

    assert [cue.text for cue in cues] == [
        "Tug'ruq paytida chaqaloqning bo'yniga kindik o'ralishi ko'p uchraydigan holatdir.",
        "Bu deyarli har uchta tug'ruqdan birida kuzatiladi.",
    ]
    assert cues[1].start - cues[0].end == pytest.approx(0.1)


def test_readable_comma_clauses_are_split() -> None:
    transcript = _transcript(
        [
            ("Ushbu", 0.0, 0.4),
            ("videoda", 0.4, 0.9),
            ("birinchi", 0.9, 1.4),
            ("qism,", 1.4, 1.9),
            ("keyingi", 1.9, 2.3),
            ("muhim", 2.3, 2.8),
            ("qism", 2.8, 3.2),
            ("ko'rsatiladi.", 3.2, 3.8),
        ]
    )

    cues = segment_mini(transcript)

    assert [cue.text for cue in cues] == ["Ushbu videoda birinchi qism,", "keyingi muhim qism ko'rsatiladi."]
    assert cues[1].start - cues[0].end == pytest.approx(0.1)


def test_readable_yoki_clause_starts_a_new_cue() -> None:
    transcript = _transcript(
        [
            ("Kindik", 0.0, 0.4),
            ("bo'sh", 0.4, 0.8),
            ("bo'lishi", 0.8, 1.3),
            ("mumkin", 1.3, 1.8),
            ("yoki", 1.8, 2.2),
            ("shifokor", 2.2, 2.7),
            ("yordam", 2.7, 3.2),
            ("beradi.", 3.2, 3.8),
        ]
    )

    cues = segment_mini(transcript)

    assert [cue.text for cue in cues] == ["Kindik bo'sh bo'lishi mumkin", "yoki shifokor yordam beradi."]
    assert cues[1].start - cues[0].end == pytest.approx(0.1)


def test_short_yoki_fragment_stays_with_sentence() -> None:
    transcript = _transcript(
        [
            ("Tanlang", 0.0, 0.4),
            ("yoki", 0.4, 0.8),
            ("davom", 0.8, 1.2),
            ("eting.", 1.2, 1.8),
        ]
    )

    assert [cue.text for cue in segment_mini(transcript)] == ["Tanlang yoki davom eting."]


def test_short_intro_before_comma_stays_with_sentence() -> None:
    transcript = _transcript(
        [
            ("Tug'ruq", 0.0, 0.4),
            ("paytida,", 0.4, 0.9),
            ("chaqaloqning", 0.9, 1.4),
            ("bo'yniga", 1.4, 1.8),
            ("kindik", 1.8, 2.2),
            ("o'ralishi", 2.2, 2.6),
            ("mumkin.", 2.6, 3.1),
        ]
    )

    cues = segment_mini(transcript)

    assert [cue.text for cue in cues] == ["Tug'ruq paytida, chaqaloqning bo'yniga kindik o'ralishi mumkin."]


def test_short_timed_clause_before_comma_stays_with_sentence() -> None:
    transcript = _transcript(
        [
            ("Agar", 0.0, 0.2),
            ("kerak", 0.2, 0.4),
            ("bo'lsa,", 0.4, 0.6),
            ("shifokor", 0.6, 1.1),
            ("yordam", 1.1, 1.6),
            ("beradi.", 1.6, 2.1),
        ]
    )

    cues = segment_mini(transcript)

    assert [cue.text for cue in cues] == ["Agar kerak bo'lsa, shifokor yordam beradi."]


def test_enumeration_does_not_create_single_word_cues() -> None:
    transcript = _transcript(
        [
            ("Kerakli", 0.0, 0.4),
            ("buyumlar:", 0.4, 0.9),
            ("suv,", 0.9, 1.2),
            ("sovun,", 1.2, 1.5),
            ("sochiq,", 1.5, 1.9),
            ("va", 1.9, 2.1),
            ("qo'lqopdir.", 2.1, 2.8),
        ]
    )

    cues = segment_mini(transcript)

    assert all(len(cue.words) >= 3 for cue in cues)
    assert " ".join(cue.text for cue in cues) == "Kerakli buyumlar: suv, sovun, sochiq, va qo'lqopdir."


def test_existing_silence_longer_than_100ms_is_preserved() -> None:
    transcript = _transcript([("First.", 0.0, 0.8), ("Second.", 1.2, 2.0)])

    cues = segment_mini(transcript)

    assert cues[0].end == 0.8
    assert cues[1].start == 1.2


def test_impossibly_short_neighboring_cues_are_merged() -> None:
    transcript = _transcript([("Yes.", 0.0, 0.03), ("No.", 0.03, 0.06)])

    cues = segment_mini(transcript)

    assert [cue.text for cue in cues] == ["Yes. No."]


def test_zero_duration_word_gets_a_valid_millisecond_cue() -> None:
    transcript = _transcript([("Yes.", 1.0, 1.0)])

    cues = segment_mini(transcript)

    assert cues[0].start == 1.0
    assert cues[0].end == 1.001


def test_timestamps_are_required() -> None:
    transcript = Transcript.from_payload({"text": "Untimed text."})

    with pytest.raises(MiniSrtError, match="requires word or segment timestamps"):
        segment_mini(transcript)
