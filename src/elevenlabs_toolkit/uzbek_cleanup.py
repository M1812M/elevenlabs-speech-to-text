import copy
import re
from typing import Dict, List, Optional


UZ_APOS = "\u2018"  # o‘ / g‘
HAMZA_APOS = "\u2019"  # ma’no
ELLIPSIS = "\u2026"

SPACING_PUNCT_RE = re.compile(rf"\s+([,.;:!?{ELLIPSIS}])")
SENTENCE_START_RE = re.compile(rf"(^|\n|[.!?{ELLIPSIS}]\s+)([a-z\u0430-\u044f\u0451])")

# Conservative discourse markers for readability-oriented sentence starts.
DEFAULT_SENTENCE_MARKERS = (
    "Keyin",
    "Shunda",
    "Lekin",
    "Ammo",
    "Biroq",
    "Hozir",
    "Umuman",
    "Xullas",
    "Mana",
    "Demak",
)


def normalize_spaces(text: str) -> str:
    text = re.sub(r"[ \t\u00A0]+", " ", text)
    text = SPACING_PUNCT_RE.sub(r"\1", text)
    text = re.sub(rf"([,.;:!?{ELLIPSIS}])(?=[^\s,.;:!?{ELLIPSIS}])", r"\1 ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _apply_case(sample: str, replacement: str) -> str:
    if sample.isupper():
        return replacement.upper()
    if sample.islower():
        return replacement.lower()
    if len(sample) > 1 and sample[0].isupper() and sample[1:].islower():
        return replacement[0].upper() + replacement[1:]
    return replacement


def _case_word_replace(pattern: str, replacement: str, text: str) -> str:
    regex = re.compile(pattern, flags=re.IGNORECASE)
    return regex.sub(lambda m: _apply_case(m.group(0), replacement), text)


def fix_apostrophes(text: str) -> str:
    # o' / g' family => o‘ / g‘
    text = re.sub(r"([OoGg])[\u02bb\u02bc\u2018\u2019'`´](?=[A-Za-z])", r"\1" + UZ_APOS, text)
    # Remaining apostrophes between letters => hamza form.
    text = re.sub(r"(?<=[A-Za-z])[\u02bb\u02bc\u2018\u2019'`´](?=[A-Za-z])", HAMZA_APOS, text)
    return text


def fix_common_words(text: str) -> str:
    replacements = [
        (r"\bmanga\b", "menga"),
        (r"\bman\b", "men"),
        (r"\bmisofir\b", "musofir"),
        (r"\bbir birimiz\b", "bir-birimiz"),
        (r"\bbir biriga\b", "bir-biriga"),
        (r"\bbir birini\b", "bir-birini"),
        (r"\bbir bir\b", "bir-bir"),
        (r"\bota onam([a-z" + UZ_APOS + HAMZA_APOS + r"]*)\b", r"ota-onam\1"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def capitalize_proper_nouns(text: str) -> str:
    # Multiword first.
    text = re.sub(r"\biso\s+masih\b", "Iso Masih", text, flags=re.IGNORECASE)

    simple_caps = [
        (r"\binjil\b", "Injil"),
        (r"\btavrot\b", "Tavrot"),
        (r"\bzabur\b", "Zabur"),
        (r"\biso\b", "Iso"),
        (r"\bmasih\b", "Masih"),
        (r"\brossiya([a-z" + UZ_APOS + HAMZA_APOS + r"]*)\b", r"Rossiya\1"),
        (r"\bnovosibirsk([a-z" + UZ_APOS + HAMZA_APOS + r"]*)\b", r"Novosibirsk\1"),
        (r"\bfarg[\u02bb\u02bc\u2018\u2019'`´]ona([a-z" + UZ_APOS + HAMZA_APOS + r"]*)\b", r"Farg" + UZ_APOS + r"ona\1"),
        (r"\bo[\u02bb\u02bc\u2018\u2019'`´]zbekiston([a-z" + UZ_APOS + HAMZA_APOS + r"]*)\b", r"O" + UZ_APOS + r"zbekiston\1"),
        (r"\bqur[\u02bb\u02bc\u2018\u2019'`´]on\b", "Qur" + HAMZA_APOS + "on"),
    ]
    for pattern, replacement in simple_caps:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Xudo (+suffix).
    text = re.sub(
        r"\bxudo([a-z" + UZ_APOS + HAMZA_APOS + r"]*)\b",
        r"Xudo\1",
        text,
        flags=re.IGNORECASE,
    )
    return text


def light_sentence_breaks(text: str, markers: Optional[List[str]] = None) -> str:
    if markers is None:
        markers = list(DEFAULT_SENTENCE_MARKERS)
    for marker in markers:
        text = re.sub(
            rf"(?<![.!?{ELLIPSIS}])\s+({re.escape(marker)})\b",
            r". \1",
            text,
        )
    text = re.sub(r"\.\s*\.", ".", text)
    return text


def capitalize_sentence_starts(text: str) -> str:
    return SENTENCE_START_RE.sub(lambda m: m.group(1) + m.group(2).upper(), text)


def clean_uzbek_text(text: str, add_marker_breaks: bool = False) -> str:
    text = normalize_spaces(text)
    text = fix_apostrophes(text)
    text = fix_common_words(text)
    text = capitalize_proper_nouns(text)
    if add_marker_breaks:
        text = light_sentence_breaks(text)
    text = normalize_spaces(text)
    text = capitalize_sentence_starts(text)
    return text


def clean_uzbek_token(token: str) -> str:
    if not token:
        return token
    # Token-safe: no sentence punctuation insertion, only lexical normalization.
    token = fix_apostrophes(token)
    token = fix_common_words(token)
    token = capitalize_proper_nouns(token)
    return token


def clean_uzbek_payload(payload: Dict, clean_word_tokens: bool = True) -> Dict:
    cleaned = copy.deepcopy(payload)

    text = cleaned.get("text")
    if isinstance(text, str):
        cleaned["text"] = clean_uzbek_text(text, add_marker_breaks=True)

    segments = cleaned.get("segments")
    if isinstance(segments, list):
        for segment in segments:
            if isinstance(segment, dict) and isinstance(segment.get("text"), str):
                segment["text"] = clean_uzbek_text(segment["text"])

    if clean_word_tokens:
        words = cleaned.get("words")
        if isinstance(words, list):
            for word in words:
                if isinstance(word, dict) and isinstance(word.get("text"), str):
                    word["text"] = clean_uzbek_token(word["text"])

    return cleaned
