from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from ..models import ScriptMode


class LanguageProcessor(Protocol):
    name: str

    def detect_script(self, text: str) -> ScriptMode: ...

    def transform_text(
        self,
        text: str,
        *,
        target: ScriptMode = ScriptMode.SOURCE,
        cleanup: bool = False,
        token_safe: bool = False,
        replacements: tuple[str, ...] = (),
    ) -> str: ...

    def transform_payload(
        self,
        payload: Mapping[str, Any],
        *,
        target: ScriptMode = ScriptMode.SOURCE,
        cleanup: bool = False,
        replacements: tuple[str, ...] = (),
    ) -> dict[str, Any]: ...


class UnsupportedLanguageError(ValueError):
    pass
