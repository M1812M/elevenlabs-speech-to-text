import pytest

from elevenlabs_toolkit.models import Transcript
from elevenlabs_toolkit.segmentation import MiniSrtError, segment_mini


def _transcript(items: list[tuple[str, float, float]], language_code: str | None = None) -> Transcript:
    return Transcript.from_payload(
        {
            "text": " ".join(text for text, _start, _end in items),
            "language_code": language_code,
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
    assert cues[1].start - cues[0].end == pytest.approx(0.08)


def test_sentence_under_80_characters_is_not_split_at_comma() -> None:
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

    assert [cue.text for cue in cues] == ["Ushbu videoda birinchi qism, keyingi muhim qism ko'rsatiladi."]


def test_speaker_identifier_is_not_part_of_80_character_limit() -> None:
    payload = {
        "text": "First complete clause and second complete clause.",
        "language_code": "eng",
        "words": [
            {
                "type": "word",
                "text": text,
                "start": index * 0.4,
                "end": index * 0.4 + 0.3,
                "speaker_id": "speaker_identifier_that_must_never_be_rendered_or_measured",
            }
            for index, text in enumerate("First complete clause and second complete clause.".split())
        ],
    }

    cues = segment_mini(Transcript.from_payload(payload))

    assert [cue.text for cue in cues] == ["First complete clause and second complete clause."]


def test_sentence_over_80_characters_is_split_at_semicolon() -> None:
    transcript = _transcript(
        [
            ("First", 0.0, 0.3),
            ("extraordinarilylong", 0.3, 0.7),
            ("complete", 0.7, 1.0),
            ("clause;", 1.0, 1.3),
            ("second", 1.3, 1.6),
            ("exceptionallylong", 1.6, 2.0),
            ("complete", 2.0, 2.3),
            ("clause.", 2.3, 2.7),
        ]
    )

    cues = segment_mini(transcript)

    assert [cue.text for cue in cues] == [
        "First extraordinarilylong complete clause;",
        "second exceptionallylong complete clause.",
    ]
    assert cues[1].start - cues[0].end == pytest.approx(0.08)


def test_split_uses_spoken_character_edges_when_available() -> None:
    transcript = Transcript.from_payload(
        {
            "text": "First extraordinarilylongcomplete clause; second exceptionallylongcomplete clause.",
            "words": [
                {
                    "type": "word",
                    "text": "First",
                    "start": 0.0,
                    "end": 0.4,
                    "characters": [
                        {"text": "F", "start": 0.08, "end": 0.14},
                        {"text": "t", "start": 0.30, "end": 0.34},
                    ],
                },
                {"type": "word", "text": "extraordinarilylongcomplete", "start": 0.4, "end": 0.8},
                {
                    "type": "word",
                    "text": "clause;",
                    "start": 0.8,
                    "end": 1.3,
                    "characters": [
                        {"text": "c", "start": 0.84, "end": 0.90},
                        {"text": "e", "start": 1.08, "end": 1.14},
                        {"text": ";", "start": 1.14, "end": 1.20},
                    ],
                },
                {
                    "type": "word",
                    "text": "second",
                    "start": 1.3,
                    "end": 1.8,
                    "characters": [
                        {"text": "s", "start": 1.42, "end": 1.48},
                        {"text": "d", "start": 1.70, "end": 1.76},
                    ],
                },
                {"type": "word", "text": "exceptionallylongcomplete", "start": 1.8, "end": 2.2},
                {
                    "type": "word",
                    "text": "clause.",
                    "start": 2.2,
                    "end": 2.8,
                    "characters": [
                        {"text": "c", "start": 2.24, "end": 2.30},
                        {"text": "e", "start": 2.56, "end": 2.62},
                        {"text": ".", "start": 2.62, "end": 2.68},
                    ],
                },
            ],
        }
    )

    cues = segment_mini(transcript)

    assert [(cue.start, cue.end) for cue in cues] == [(0.013, 1.207), (1.353, 2.687)]


def test_semicolon_is_preferred_over_a_competing_comma_split() -> None:
    transcript = _transcript(
        [
            ("oneextremelylongwordindeed", 0.0, 0.3),
            ("two", 0.3, 0.6),
            ("three;", 0.6, 0.9),
            ("verylongword", 0.9, 1.2),
            ("anotherverylongword,", 1.2, 1.5),
            ("six", 1.5, 1.8),
            ("seven", 1.8, 2.1),
            ("eight.", 2.1, 2.4),
        ]
    )

    cues = segment_mini(transcript)

    assert [cue.text for cue in cues] == [
        "oneextremelylongwordindeed two three;",
        "verylongword anotherverylongword, six seven eight.",
    ]


def test_overlong_yoki_clause_starts_a_new_cue() -> None:
    transcript = _transcript(
        [
            ("Kindik", 0.0, 0.4),
            ("bo'sh", 0.4, 0.8),
            ("bo'lishi", 0.8, 1.3),
            ("mumkin", 1.3, 1.8),
            ("favquloddauzun", 1.8, 2.2),
            ("tushuntirishbilan", 2.2, 2.7),
            ("yoki", 2.7, 3.1),
            ("shifokor", 3.1, 3.5),
            ("yordam", 3.5, 3.9),
            ("beradi.", 3.9, 4.4),
        ]
    )

    cues = segment_mini(transcript)

    assert [cue.text for cue in cues] == [
        "Kindik bo'sh bo'lishi mumkin favquloddauzun tushuntirishbilan",
        "yoki shifokor yordam beradi.",
    ]
    assert cues[1].start - cues[0].end == pytest.approx(0.08)


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


def test_pause_over_one_second_splits_a_short_sentence() -> None:
    transcript = _transcript(
        [
            ("Short", 0.0, 0.3),
            ("opening", 0.3, 0.7),
            ("continues", 1.71, 2.1),
            ("here.", 2.1, 2.5),
        ]
    )

    cues = segment_mini(transcript)

    assert [cue.text for cue in cues] == ["Short opening", "continues here."]
    assert cues[1].start - cues[0].end == pytest.approx(0.876)


def test_uzbek_comma_pause_wins_before_the_80_character_rule() -> None:
    transcript = _transcript(
        [
            ("Bir", 227.180, 227.329),
            ("uchi", 227.460, 227.570),
            ("mana", 227.680, 227.880),
            ("bu", 227.960, 228.060),
            ("yerda,", 228.140, 228.500),
            ("ikkinchi", 230.400, 230.780),
            ("uchi", 230.860, 230.970),
            ("esa", 231.020, 231.300),
            ("ichkariga", 231.320, 231.730),
            ("tortilib", 231.940, 232.450),
            ("ketgan.", 232.480, 232.880),
        ],
        "uzb",
    )

    cues = segment_mini(transcript)

    assert [cue.text for cue in cues] == [
        "Bir uchi mana bu yerda,",
        "ikkinchi uchi esa ichkariga tortilib ketgan.",
    ]
    assert all(len(cue.text) < 80 for cue in cues)
    assert cues[1].start - cues[0].end == pytest.approx(1.766)


def test_pause_of_exactly_one_second_does_not_split_a_short_sentence() -> None:
    transcript = _transcript(
        [
            ("Short", 0.0, 0.3),
            ("opening", 0.3, 0.7),
            ("continues", 1.7, 2.1),
            ("here.", 2.1, 2.5),
        ]
    )

    assert [cue.text for cue in segment_mini(transcript)] == ["Short opening continues here."]


def test_pause_does_not_create_a_one_word_fragment() -> None:
    transcript = _transcript(
        [
            ("Opening", 0.0, 0.4),
            ("continues", 1.5, 1.9),
            ("safely", 1.9, 2.3),
            ("here.", 2.3, 2.7),
        ]
    )

    assert [cue.text for cue in segment_mini(transcript)] == ["Opening continues safely here."]


def test_overlong_connector_does_not_create_a_one_word_fragment() -> None:
    transcript = _transcript(
        [
            ("This", 0.0, 0.3),
            ("extraordinarilylongexplanation", 0.3, 0.7),
            ("contains", 0.7, 1.0),
            ("several", 1.0, 1.3),
            ("carefullychosen", 1.3, 1.6),
            ("descriptive", 1.6, 1.9),
            ("words", 1.9, 2.2),
            ("and.", 2.2, 2.5),
        ],
        "eng",
    )

    cues = segment_mini(transcript)

    assert len(cues) == 1
    assert len(cues[0].text) > 80


def test_80_character_limit_uses_transformed_output_text() -> None:
    transcript = _transcript(
        [
            ("first", 0.0, 0.4),
            ("safe", 0.4, 0.8),
            ("clause", 0.8, 1.2),
            ("and", 1.2, 1.6),
            ("second", 1.6, 2.0),
            ("safe", 2.0, 2.4),
            ("clause.", 2.4, 2.8),
        ],
        "eng",
    )

    source_cues = segment_mini(transcript)
    transformed_cues = segment_mini(
        transcript,
        lambda text: text.replace("first", "f" * 35).replace("second", "s" * 35),
    )

    assert len(source_cues) == 1
    assert [cue.text for cue in transformed_cues] == ["first safe clause", "and second safe clause."]


@pytest.mark.parametrize(
    ("language_code", "connector"),
    [
        ("eng", "and"),
        ("uzb", "va"),
        ("kir", "жана"),
        ("rus", "и"),
    ],
)
def test_four_languages_use_the_same_safe_connector_rule(language_code: str, connector: str) -> None:
    transcript = _transcript(
        [
            ("extraordinarilylongfirst", 0.0, 0.4),
            ("complete", 0.4, 0.9),
            ("safe", 0.9, 1.3),
            ("clause", 1.3, 1.8),
            (connector, 1.8, 2.2),
            ("exceptionallylongsecond", 2.2, 2.6),
            ("complete", 2.6, 3.1),
            ("clause.", 3.1, 3.7),
        ],
        language_code,
    )

    cues = segment_mini(transcript)

    assert [cue.text for cue in cues] == [
        "extraordinarilylongfirst complete safe clause",
        f"{connector} exceptionallylongsecond complete clause.",
    ]
    assert cues[1].start - cues[0].end == pytest.approx(0.08)


@pytest.mark.parametrize(
    ("language_code", "connector"),
    [
        ("eng", ("because",)),
        ("uzb", ("shuning", "uchun")),
        ("kir", ("андан", "кийин")),
        ("rus", ("потому", "что")),
    ],
)
def test_four_languages_support_multiword_structure_phrases(
    language_code: str,
    connector: tuple[str, ...],
) -> None:
    items = [
        ("extraordinarilylongfirst", 0.0, 0.4),
        ("complete", 0.4, 0.9),
        ("safe", 0.9, 1.3),
        ("clause", 1.3, 1.8),
        *((word, 1.8 + index * 0.3, 2.1 + index * 0.3) for index, word in enumerate(connector)),
        ("exceptionallylongsecond", 2.5, 2.9),
        ("complete", 2.9, 3.4),
        ("clause.", 3.4, 4.0),
    ]
    transcript = _transcript(items, language_code)

    cues = segment_mini(transcript)

    assert cues[0].text == "extraordinarilylongfirst complete safe clause"
    assert cues[1].text.startswith(" ".join(connector))


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


def test_existing_silence_allows_full_two_frame_padding() -> None:
    transcript = _transcript([("First.", 0.0, 0.8), ("Second.", 1.2, 2.0)])

    cues = segment_mini(transcript)

    assert (cues[0].start, cues[0].end) == (0.0, 0.867)
    assert (cues[1].start, cues[1].end) == (1.133, 2.067)


def test_impossibly_short_neighboring_cues_are_merged() -> None:
    transcript = _transcript([("Yes.", 0.0, 0.03), ("No.", 0.03, 0.06)])

    cues = segment_mini(transcript)

    assert [cue.text for cue in cues] == ["Yes. No."]


def test_zero_duration_word_gets_a_valid_padded_cue() -> None:
    transcript = _transcript([("Yes.", 1.0, 1.0)])

    cues = segment_mini(transcript)

    assert cues[0].start == 0.933
    assert cues[0].end == 1.068


def test_timestamps_are_required() -> None:
    transcript = Transcript.from_payload({"text": "Untimed text."})

    with pytest.raises(MiniSrtError, match="requires word or segment timestamps"):
        segment_mini(transcript)
