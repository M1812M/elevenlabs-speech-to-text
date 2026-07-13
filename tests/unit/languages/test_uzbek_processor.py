import pytest

from elevenlabs_toolkit.languages import get_language_processor
from elevenlabs_toolkit.languages.uzbek import clean_text
from elevenlabs_toolkit.models import ScriptMode


def test_cleanup_canonicalizes_cyrillic_then_restores_source_script() -> None:
    processor = get_language_processor("uzbek")

    result = processor.transform_text("ман манга бордим", cleanup=True, target=ScriptMode.SOURCE)

    assert result == "Мен менга бордим"


def test_explicit_latin_output_and_custom_replacement() -> None:
    processor = get_language_processor("uzbek")

    result = processor.transform_text(
        "Салом дунё",
        target=ScriptMode.LATIN,
        replacements=("dunyo=jahon",),
    )

    assert result == "Salom jahon"


def test_payload_processing_preserves_timing_and_character_provenance() -> None:
    payload = {
        "text": "ман",
        "words": [
            {
                "type": "word",
                "text": "ман",
                "start": 1.0,
                "end": 1.4,
                "characters": [{"text": "м", "start": 1.0, "end": 1.1}],
            }
        ],
    }

    result = get_language_processor("uzbek").transform_payload(payload, cleanup=True)

    assert result["words"][0]["text"] == "мен"
    assert result["words"][0]["start"] == 1.0
    assert "characters" not in result["words"][0]
    assert result["words"][0]["source_characters"] == payload["words"][0]["characters"]
    assert result["words"][0]["source_text"] == "ман"
    assert result["source_text"] == "ман"
    assert payload["words"][0]["text"] == "ман"


def test_cleanup_normalizes_common_words_names_and_apostrophes() -> None:
    processor = get_language_processor("uzbek")

    result = processor.transform_text(
        "man manga misofir bo'lib keldim, iso masih va xudo o'zbekiston haqida gapirdi",
        cleanup=True,
        target=ScriptMode.LATIN,
    )

    for expected in ("Men", "menga", "musofir", "Iso Masih", "Xudo", "O‘zbekiston"):
        assert expected in result


def test_russian_cyrillic_letters_are_normalized_to_uzbek_latin() -> None:
    processor = get_language_processor("uzbek")

    result = processor.transform_text(
        "Скорый келяпти, виу-виу деб нимаси уж эшитилганда.",
        target=ScriptMode.LATIN,
    )

    assert result == "Skoriy kelyapti, viu-viu deb nimasi uj eshitilganda."


def test_custom_replacement_target_is_literal_in_every_source_case() -> None:
    processor = get_language_processor("uzbek")

    assert processor.transform_text("acme Acme ACME", replacements=("Acme=ACME",)) == "ACME ACME ACME"


def test_custom_replacement_target_bypasses_script_conversion() -> None:
    processor = get_language_processor("uzbek")

    assert (
        processor.transform_text(
            "Acme dunyo",
            target=ScriptMode.CYRILLIC,
            replacements=("Acme=ACME",),
        )
        == "ACME дунё"
    )


def test_custom_replacement_source_accepts_cyrillic_script() -> None:
    processor = get_language_processor("uzbek")

    assert (
        processor.transform_text(
            "дунё",
            target=ScriptMode.SOURCE,
            replacements=("дунё=жаҳон",),
        )
        == "жаҳон"
    )


def test_cleanup_preserves_decimals_times_and_urls() -> None:
    result = get_language_processor("uzbek").transform_text(
        "qiymat 3.14, vaqt 12:30, manzil example.com",
        cleanup=True,
        target=ScriptMode.LATIN,
    )

    assert "3.14" in result
    assert "12:30" in result
    assert "example.com" in result


def test_payload_source_script_falls_back_to_words_and_segments() -> None:
    processor = get_language_processor("uzbek")
    from_words = processor.transform_payload(
        {
            "text": "   ",
            "words": [{"type": "WORD", "text": "кейин", "start": 0, "end": 1}],
        },
        target=ScriptMode.SOURCE,
        cleanup=True,
    )
    from_segments = processor.transform_payload(
        {"segments": [{"text": "кейин", "start": 0, "end": 1}]},
        target=ScriptMode.SOURCE,
        cleanup=True,
    )

    assert from_words["toolkit_processing"]["resolved_script"] == "cyrillic"
    assert from_words["words"][0]["text"] == "кейин"
    assert from_segments["toolkit_processing"]["resolved_script"] == "cyrillic"
    assert from_segments["segments"][0]["text"] == "Кейин"
    assert from_segments["segments"][0]["source_text"] == "кейин"


def test_payload_transforms_audio_event_text_and_preserves_provenance() -> None:
    event = "[мусиқа]"
    payload = {
        "text": event,
        "words": [
            {
                "type": "audio_event",
                "text": event,
                "start": 0,
                "end": 1,
                "characters": [{"text": "м", "start": 0, "end": 0.1}],
            }
        ],
    }

    result = get_language_processor("uzbek").transform_payload(payload, target=ScriptMode.LATIN)

    assert result["text"] == "[musiqa]"
    assert result["words"][0]["type"] == "audio_event"
    assert result["words"][0]["text"] == "[musiqa]"
    assert result["words"][0]["source_text"] == event
    assert result["words"][0]["source_characters"] == payload["words"][0]["characters"]
    assert "characters" not in result["words"][0]


def test_marker_breaks_are_case_insensitive() -> None:
    assert clean_text("avval keyin davom", add_marker_breaks=True) == "Avval. Keyin davom"


def test_mixed_payload_uses_one_resolved_script() -> None:
    result = get_language_processor("uzbek").transform_payload(
        {
            "text": "Бу NASA ҳақида.",
            "words": [
                {"type": "word", "text": "Бу", "start": 0, "end": 0.2},
                {"type": "word", "text": "NASA", "start": 0.3, "end": 0.5},
                {"type": "word", "text": "ҳақида.", "start": 0.6, "end": 1.0},
            ],
        },
        target=ScriptMode.SOURCE,
    )

    assert result["toolkit_processing"]["resolved_script"] == "cyrillic"
    assert result["words"][1]["text"] == "НАСА"


@pytest.mark.parametrize(
    ("latin", "cyrillic"),
    [
        ("a", "а"),
        ("b", "б"),
        ("d", "д"),
        ("e", "э"),
        ("f", "ф"),
        ("g", "г"),
        ("g'", "ғ"),
        ("h", "ҳ"),
        ("i", "и"),
        ("j", "ж"),
        ("k", "к"),
        ("l", "л"),
        ("m", "м"),
        ("n", "н"),
        ("o", "о"),
        ("o'", "ў"),
        ("p", "п"),
        ("q", "қ"),
        ("r", "р"),
        ("s", "с"),
        ("t", "т"),
        ("u", "у"),
        ("v", "в"),
        ("x", "х"),
        ("y", "й"),
        ("z", "з"),
        ("sh", "ш"),
        ("ch", "ч"),
        ("ng", "нг"),
        ("yo", "ё"),
        ("yu", "ю"),
        ("ya", "я"),
        ("ye", "е"),
    ],
)
def test_uzbek_alphabet_and_digraph_round_trip(latin: str, cyrillic: str) -> None:
    processor = get_language_processor("uzbek")

    assert processor.transform_text(latin, target=ScriptMode.CYRILLIC) == cyrillic
    assert processor.transform_text(cyrillic, target=ScriptMode.LATIN) == latin.replace("'", "‘")
