from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

LEXICAL_WORD_RE = re.compile(r"\w+(?:['\u2018\u2019\u02bb\u02bc-]\w+)*", re.UNICODE)


@dataclass(frozen=True, slots=True)
class LanguageStructure:
    name: str
    codes: frozenset[str]
    connectors: tuple[tuple[str, ...], ...]


def _phrases(*values: str) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(value.casefold().split()) for value in values)


LANGUAGE_STRUCTURES = (
    LanguageStructure(
        "english",
        frozenset({"english", "en", "eng"}),
        _phrases(
            "and",
            "or",
            "but",
            "because",
            "therefore",
            "however",
            "otherwise",
            "then",
            "while",
            "although",
            "whereas",
            "so",
            "yet",
            "if",
            "now",
            "in short",
            "in general",
            "finally",
        ),
    ),
    LanguageStructure(
        "uzbek",
        frozenset({"uzbek", "uz", "uzb"}),
        _phrases(
            "va",
            "yoki",
            "hamda",
            "lekin",
            "ammo",
            "biroq",
            "chunki",
            "shuning uchun",
            "aks holda",
            "keyin",
            "shunda",
            "demak",
            "garchi",
            "esa",
            "agar",
            "hozir",
            "xullas",
            "umuman",
            "nihoyat",
            "mana",
            "ва",
            "ёки",
            "ҳамда",
            "лекин",
            "аммо",
            "бироқ",
            "чунки",
            "шунинг учун",
            "акс ҳолда",
            "кейин",
            "шунда",
            "демак",
            "гарчи",
            "эса",
            "агар",  # noqa: RUF001 - intentional Uzbek Cyrillic spelling
            "ҳозир",
            "хуллас",
            "умуман",
            "ниҳоят",
            "мана",
        ),
    ),
    LanguageStructure(
        "kyrgyz",
        frozenset({"kyrgyz", "kirghiz", "ky", "kir"}),
        _phrases(
            "жана",
            "же",
            "же болбосо",
            "бирок",
            "анткени",
            "ошондуктан",
            "болбосо",
            "андан кийин",
            "анда",
            "демек",
            "ошентсе да",
            "ал эми",
            "эгерде",
            "азыр",
            "кыскасы",
            "жалпысынан",
            "акыры",
        ),
    ),
    LanguageStructure(
        "russian",
        frozenset({"russian", "ru", "rus"}),
        _phrases(
            "и",
            "или",
            "либо",
            "но",
            "а",  # noqa: RUF001 - intentional Cyrillic conjunction
            "потому что",
            "поэтому",
            "однако",
            "иначе",
            "затем",
            "тогда",
            "хотя",
            "так что",
            "при этом",
            "если",
            "теперь",
            "короче",
            "в общем",
            "вообще",
            "наконец",
        ),
    ),
)


def normalize_language_code(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    normalized = value.strip().casefold().replace("_", "-")
    return normalized.split("-", 1)[0]


def language_structure(value: str | None) -> LanguageStructure | None:
    normalized = normalize_language_code(value)
    if normalized is None:
        return None
    return next((rules for rules in LANGUAGE_STRUCTURES if normalized in rules.codes), None)


def connector_phrases(value: str | None) -> tuple[tuple[str, ...], ...]:
    rules = language_structure(value)
    if rules is not None:
        return rules.connectors
    if normalize_language_code(value) is not None:
        return ()
    return tuple(dict.fromkeys(phrase for item in LANGUAGE_STRUCTURES for phrase in item.connectors))


def lexical_tokens(value: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in LEXICAL_WORD_RE.findall(value))


def connector_boundaries(words: Sequence[str], language_code: str | None) -> frozenset[int]:
    normalized = tuple(lexical_tokens(word) for word in words)
    boundaries: set[int] = set()
    for index in range(1, len(words)):
        for phrase in connector_phrases(language_code):
            candidate = normalized[index : index + len(phrase)]
            if len(candidate) == len(phrase) and all(
                tokens == (expected,) for tokens, expected in zip(candidate, phrase, strict=True)
            ):
                boundaries.add(index)
                break
    return frozenset(boundaries)


__all__ = [
    "LANGUAGE_STRUCTURES",
    "LanguageStructure",
    "connector_boundaries",
    "connector_phrases",
    "language_structure",
    "lexical_tokens",
    "normalize_language_code",
]
