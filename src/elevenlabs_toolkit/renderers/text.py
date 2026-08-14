from __future__ import annotations

from collections.abc import Sequence

TextItem = tuple[str, str | None]


def _inline_text(value: str) -> str:
    return " ".join(str(value or "").split())


def render_txt(
    items: Sequence[TextItem],
    include_source: str | None = None,
) -> str:
    """Render sentence-like text items without speaker prefixes."""
    lines: list[str] = []
    if include_source is not None:
        lines.extend((f"# {_inline_text(include_source)}", ""))

    for text, _speaker in items:
        line = _inline_text(text)
        lines.append(line)

    return "\n".join(lines) + ("\n" if lines else "")
