from __future__ import annotations

from .protocol import LanguageProcessor, UnsupportedLanguageError
from .uzbek import UzbekProcessor


def get_language_processor(name: str) -> LanguageProcessor:
    normalized = name.strip().casefold()
    if normalized in {"uzbek", "uz", "uzb"}:
        return UzbekProcessor()
    raise UnsupportedLanguageError(f"unsupported language processor {name!r}; available: uzbek")


__all__ = ["LanguageProcessor", "UnsupportedLanguageError", "UzbekProcessor", "get_language_processor"]
