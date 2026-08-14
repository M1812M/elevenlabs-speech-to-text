from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any

from ...models import ScriptMode
from ..replacements import apply_replacements, replacement_parts
from .cleanup import clean_text, clean_token
from .transliteration import to_cyrillic, to_latin

CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
LATIN_RE = re.compile(r"[A-Za-z]")


def _protect_literal_replacements(
    text: str,
    replacements: tuple[str, ...],
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Replace canonical sources with transliteration-safe target placeholders."""

    if not replacements:
        return text, ()

    parts = tuple(replacement_parts(entry) for entry in replacements)
    occupied = text + "".join(source + target for source, target in parts)
    protected: list[tuple[str, str]] = []
    value = text
    placeholder_index = 0
    for source, target in parts:
        while True:
            placeholder = f"\ue000{placeholder_index}\ue001"
            placeholder_index += 1
            if placeholder not in occupied and placeholder not in value:
                break
        canonical_source = to_latin(source)
        value = apply_replacements(value, (f"{canonical_source}={placeholder}",))
        protected.append((placeholder, target))
    return value, tuple(protected)


def _restore_literal_replacements(text: str, protected: tuple[tuple[str, str], ...]) -> str:
    value = text
    for placeholder, target in protected:
        value = value.replace(placeholder, target)
    return value


class UzbekProcessor:
    name = "uzbek"

    @staticmethod
    def detect_script(text: str) -> ScriptMode:
        cyrillic = len(CYRILLIC_RE.findall(text))
        latin = len(LATIN_RE.findall(text))
        return ScriptMode.CYRILLIC if cyrillic > latin else ScriptMode.LATIN

    def transform_text(
        self,
        text: str,
        *,
        target: ScriptMode = ScriptMode.SOURCE,
        cleanup: bool = False,
        token_safe: bool = False,
        replacements: tuple[str, ...] = (),
    ) -> str:
        target = ScriptMode(target)
        if target is ScriptMode.SOURCE and not cleanup and not replacements:
            return text
        source_script = self.detect_script(text)
        canonical = to_latin(text)
        if cleanup:
            canonical = clean_token(canonical) if token_safe else clean_text(canonical)
        canonical, protected = _protect_literal_replacements(canonical, replacements)
        output_script = source_script if target is ScriptMode.SOURCE else target
        converted = to_cyrillic(canonical) if output_script is ScriptMode.CYRILLIC else canonical
        return _restore_literal_replacements(converted, protected)

    def transform_payload(
        self,
        payload: Mapping[str, Any],
        *,
        target: ScriptMode = ScriptMode.SOURCE,
        cleanup: bool = False,
        replacements: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        transformed = copy.deepcopy(dict(payload))
        requested_target = ScriptMode(target)
        source_parts: list[str] = []
        if isinstance(transformed.get("text"), str) and transformed["text"].strip():
            source_parts.append(transformed["text"])
        if not source_parts:
            source_parts.extend(
                str(item.get("text") or "")
                for collection in (transformed.get("words") or [], transformed.get("segments") or [])
                for item in collection
                if isinstance(item, dict) and str(item.get("text") or "").strip()
            )
        resolved_target = (
            self.detect_script(" ".join(source_parts)) if requested_target is ScriptMode.SOURCE else requested_target
        )

        if isinstance(transformed.get("text"), str):
            source_text = transformed["text"]
            converted = self.transform_text(
                source_text, target=resolved_target, cleanup=cleanup, replacements=replacements
            )
            if converted != source_text:
                transformed.setdefault("source_text", source_text)
            transformed["text"] = converted
        for segment in transformed.get("segments") or []:
            if isinstance(segment, dict) and isinstance(segment.get("text"), str):
                source_text = segment["text"]
                converted = self.transform_text(
                    source_text, target=resolved_target, cleanup=cleanup, replacements=replacements
                )
                if converted != source_text:
                    segment.setdefault("source_text", source_text)
                segment["text"] = converted
        for word in transformed.get("words") or []:
            if (
                isinstance(word, dict)
                and str(word.get("type", "word")).casefold() in {"word", "audio_event"}
                and isinstance(word.get("text"), str)
            ):
                source_text = word["text"]
                converted = self.transform_text(
                    source_text,
                    target=resolved_target,
                    cleanup=cleanup,
                    token_safe=True,
                    replacements=replacements,
                )
                if converted != source_text:
                    word.setdefault("source_text", source_text)
                    characters = word.pop("characters", None)
                    if characters is not None:
                        word.setdefault("source_characters", characters)
                word["text"] = converted

        processing: dict[str, Any] = {
            "language": self.name,
            "cleanup": cleanup,
            "script": requested_target.value,
            "resolved_script": resolved_target.value,
            "replacements": list(replacements),
        }
        previous_processing = transformed.get("toolkit_processing")
        if previous_processing is not None:
            processing["previous"] = previous_processing
        transformed["toolkit_processing"] = processing
        return transformed
