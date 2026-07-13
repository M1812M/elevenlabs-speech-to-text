from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from ..models import SpeakerLabels

TextItem = tuple[str, str | None]


def _inline_text(value: str) -> str:
    return " ".join(str(value or "").split())


def _infer_main_speaker(items: Sequence[TextItem]) -> str | None:
    counts = Counter(speaker for _text, speaker in items if speaker)
    if not counts:
        return None
    # Counter preserves first-seen order, which makes ties deterministic.
    return max(counts, key=counts.__getitem__)


def render_txt(
    items: Sequence[TextItem],
    speaker_labels: SpeakerLabels = SpeakerLabels.NONE,
    main_speaker: str | None = None,
    include_source: str | None = None,
) -> str:
    """Render sentence-like text items with optional speaker labels and heading."""
    labels = SpeakerLabels(speaker_labels)
    if labels is SpeakerLabels.SECONDARY and main_speaker is None:
        main_speaker = _infer_main_speaker(items)

    lines: list[str] = []
    if include_source is not None:
        lines.extend((f"# {_inline_text(include_source)}", ""))

    for text, speaker in items:
        line = _inline_text(text)
        should_label = bool(speaker) and (
            labels is SpeakerLabels.ALL or (labels is SpeakerLabels.SECONDARY and speaker != main_speaker)
        )
        if should_label:
            line = f"[{speaker}] {line}"
        lines.append(line)

    return "\n".join(lines) + ("\n" if lines else "")
