import pytest

from elevenlabs_toolkit.languages import (
    connector_boundaries,
    connector_phrases,
    language_structure,
)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("en-US", "english"),
        ("eng", "english"),
        ("uzb", "uzbek"),
        ("ky", "kyrgyz"),
        ("kir", "kyrgyz"),
        ("ru-RU", "russian"),
        ("rus", "russian"),
    ],
)
def test_language_aliases_are_normalized(code: str, expected: str) -> None:
    rules = language_structure(code)

    assert rules is not None
    assert rules.name == expected


@pytest.mark.parametrize(
    ("code", "words", "boundary"),
    [
        ("eng", ("first", "part", "because", "second", "part"), 2),
        ("uzb", ("birinchi", "qism", "shuning", "uchun", "ikkinchi"), 2),
        ("kir", ("биринчи", "бөлүк", "андан", "кийин", "экинчи"), 2),
        ("rus", ("первая", "часть", "потому", "что", "вторая"), 2),
    ],
)
def test_single_and_multiword_connectors_share_one_matcher(
    code: str,
    words: tuple[str, ...],
    boundary: int,
) -> None:
    assert connector_boundaries(words, code) == {boundary}


def test_unknown_language_does_not_receive_foreign_connectors() -> None:
    assert connector_phrases("de") == ()
    assert connector_boundaries(("eins", "und", "zwei"), "de") == set()


def test_missing_language_code_uses_supported_language_union() -> None:
    starters = {phrase[0] for phrase in connector_phrases(None)}

    assert {"and", "va", "жана", "и"} <= starters


@pytest.mark.parametrize(
    ("role", "phrases"),
    [
        ("addition", {"eng": "and", "uzb": "va", "kir": "жана", "rus": "и"}),
        ("alternative", {"eng": "or", "uzb": "yoki", "kir": "же", "rus": "или"}),
        ("cause", {"eng": "because", "uzb": "chunki", "kir": "анткени", "rus": "потому что"}),
        (
            "consequence",
            {"eng": "therefore", "uzb": "shuning uchun", "kir": "ошондуктан", "rus": "поэтому"},
        ),
        ("condition", {"eng": "if", "uzb": "agar", "kir": "эгерде", "rus": "если"}),
        ("summary", {"eng": "in short", "uzb": "xullas", "kir": "кыскасы", "rus": "короче"}),
    ],
)
def test_four_languages_cover_the_same_structural_roles(role: str, phrases: dict[str, str]) -> None:
    del role
    for code, phrase in phrases.items():
        assert tuple(phrase.casefold().split()) in connector_phrases(code)
