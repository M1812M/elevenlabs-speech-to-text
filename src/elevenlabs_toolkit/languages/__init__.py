from __future__ import annotations

from .protocol import LanguageProcessor, UnsupportedLanguageError
from .replacements import apply_replacements
from .structure import (
    LANGUAGE_STRUCTURES,
    LanguageStructure,
    connector_boundaries,
    connector_phrases,
    language_structure,
    lexical_tokens,
    normalize_language_code,
)
from .uzbek import UzbekProcessor


def get_language_processor(name: str) -> LanguageProcessor:
    normalized = name.strip().casefold()
    if normalized in {"uzbek", "uz", "uzb"}:
        return UzbekProcessor()
    raise UnsupportedLanguageError(f"unsupported language processor {name!r}; available: uzbek")


__all__ = [
    "LANGUAGE_STRUCTURES",
    "LanguageProcessor",
    "LanguageStructure",
    "UnsupportedLanguageError",
    "UzbekProcessor",
    "apply_replacements",
    "connector_boundaries",
    "connector_phrases",
    "get_language_processor",
    "language_structure",
    "lexical_tokens",
    "normalize_language_code",
]
