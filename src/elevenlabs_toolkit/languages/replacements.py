from __future__ import annotations

import re


def replacement_parts(entry: str) -> tuple[str, str]:
    if "=" not in entry:
        raise ValueError(f"invalid replacement {entry!r}; expected FROM=TO")
    source, target = (part.strip() for part in entry.split("=", 1))
    if not source or not target or any(character.isspace() for character in source + target):
        raise ValueError("replacements must use non-empty TOKEN=TOKEN entries")
    return source, target


def apply_replacements(text: str, replacements: tuple[str, ...]) -> str:
    value = text
    for entry in replacements:
        source, target = replacement_parts(entry)
        pattern = re.compile(rf"(?<!\w){re.escape(source)}(?!\w)", flags=re.IGNORECASE)

        def replace_match(_match: re.Match[str], replacement: str = target) -> str:
            return replacement

        value = pattern.sub(replace_match, value)
    return value


__all__ = ["apply_replacements", "replacement_parts"]
