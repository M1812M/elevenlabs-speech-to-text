from __future__ import annotations

import re

UZ_APOS = "\u2018"
HAMZA_APOS = "\u2019"
ELLIPSIS = "\u2026"
SPACING_PUNCT_RE = re.compile(rf"\s+([,.;:!?{ELLIPSIS}])")
SENTENCE_START_RE = re.compile(rf"(^|\n|[.!?{ELLIPSIS}]\s+)([a-z\u0430-\u044f\u0451])")
DEFAULT_SENTENCE_MARKERS = ("Keyin", "Shunda", "Lekin", "Ammo", "Biroq", "Hozir", "Umuman", "Xullas", "Mana", "Demak")


def normalize_spaces(text: str) -> str:
    value = re.sub(r"[ \t\u00A0]+", " ", text)
    value = SPACING_PUNCT_RE.sub(r"\1", value)
    value = re.sub(r"\s*\n\s*", "\n", value)
    return re.sub(r" {2,}", " ", value).strip()


def fix_apostrophes(text: str) -> str:
    value = re.sub(r"([OoGg])[\u02bb\u02bc\u2018\u2019'`\u00b4](?=[A-Za-z])", r"\1" + UZ_APOS, text)
    return re.sub(r"(?<=[A-Za-z])[\u02bb\u02bc\u2018\u2019'`\u00b4](?=[A-Za-z])", HAMZA_APOS, value)


def fix_common_words(text: str) -> str:
    replacements = (
        (r"\bmanga\b", "menga"),
        (r"\bman\b", "men"),
        (r"\bmisofir\b", "musofir"),
    )
    value = text
    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    return value


def capitalize_proper_nouns(text: str) -> str:
    value = re.sub(r"\biso\s+masih\b", "Iso Masih", text, flags=re.IGNORECASE)
    replacements = (
        (r"\binjil\b", "Injil"),
        (r"\btavrot\b", "Tavrot"),
        (r"\bzabur\b", "Zabur"),
        (r"\biso\b", "Iso"),
        (r"\bmasih\b", "Masih"),
        (r"\brossiya([a-z" + UZ_APOS + HAMZA_APOS + r"]*)\b", r"Rossiya\1"),
        (r"\bnovosibirsk([a-z" + UZ_APOS + HAMZA_APOS + r"]*)\b", r"Novosibirsk\1"),
        (
            r"\bfarg[\u02bb\u02bc\u2018\u2019'`\u00b4]ona([a-z" + UZ_APOS + HAMZA_APOS + r"]*)\b",
            r"Farg" + UZ_APOS + r"ona\1",
        ),
        (
            r"\bo[\u02bb\u02bc\u2018\u2019'`\u00b4]zbekiston([a-z" + UZ_APOS + HAMZA_APOS + r"]*)\b",
            r"O" + UZ_APOS + r"zbekiston\1",
        ),
        (r"\bqur[\u02bb\u02bc\u2018\u2019'`\u00b4]on\b", "Qur" + HAMZA_APOS + "on"),
        (r"\bxudo([a-z" + UZ_APOS + HAMZA_APOS + r"]*)\b", r"Xudo\1"),
    )
    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    return value


def capitalize_sentence_starts(text: str) -> str:
    return SENTENCE_START_RE.sub(lambda match: match.group(1) + match.group(2).upper(), text)


def clean_text(text: str, *, add_marker_breaks: bool = False) -> str:
    value = capitalize_proper_nouns(fix_common_words(fix_apostrophes(normalize_spaces(text))))
    if add_marker_breaks:
        for marker in DEFAULT_SENTENCE_MARKERS:
            value = re.sub(
                rf"(?<![.!?{ELLIPSIS}])\s+({re.escape(marker)})\b",
                r". \1",
                value,
                flags=re.IGNORECASE,
            )
        value = re.sub(r"\.\s*\.", ".", value)
    return capitalize_sentence_starts(normalize_spaces(value))


def clean_token(text: str) -> str:
    return capitalize_proper_nouns(fix_common_words(fix_apostrophes(text)))


def _replacement_parts(entry: str) -> tuple[str, str]:
    if "=" not in entry:
        raise ValueError(f"invalid replacement {entry!r}; expected FROM=TO")
    source, target = (part.strip() for part in entry.split("=", 1))
    if not source or not target or any(character.isspace() for character in source + target):
        raise ValueError("replacements must use non-empty TOKEN=TOKEN entries")
    return source, target


def apply_replacements(text: str, replacements: tuple[str, ...]) -> str:
    value = text
    for entry in replacements:
        source, target = _replacement_parts(entry)
        pattern = re.compile(rf"(?<!\w){re.escape(source)}(?!\w)", flags=re.IGNORECASE)

        def replace_match(_match: re.Match[str], replacement: str = target) -> str:
            return replacement

        value = pattern.sub(replace_match, value)
    return value
