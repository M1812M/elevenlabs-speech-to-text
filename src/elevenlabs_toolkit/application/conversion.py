from __future__ import annotations

import re

from ..languages import get_language_processor
from ..models import ScriptMode

MARKUP_RE = re.compile(r"(<[^>]+>|&(?:#[0-9]+|#x[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]+);)")
TIMECODE_RE = re.compile(r"^\s*\d{2}:\d{2}:\d{2}[,.]\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}[,.]\d{3}(?:\s+.*)?$")


def _is_srt_meta_line(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.isdigit() or TIMECODE_RE.match(stripped) is not None


def convert_srt(source: str, mode: ScriptMode) -> str:
    processor = get_language_processor("uzbek")

    def transform(value: str) -> str:
        return processor.transform_text(value, target=mode)

    lines = []
    for line in source.splitlines():
        if _is_srt_meta_line(line):
            lines.append(line)
            continue
        parts = MARKUP_RE.split(line)
        lines.append("".join(part if MARKUP_RE.fullmatch(part) else transform(part) for part in parts))
    return "\n".join(lines) + ("\n" if source.endswith(("\n", "\r")) else "")
