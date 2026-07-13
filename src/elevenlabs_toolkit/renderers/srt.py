from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from functools import cache

from ..models import Cue, SpeakerLabels
from .timecode import srt_timestamp

DEFAULT_MAX_CHARS_PER_LINE = 42
DEFAULT_MAX_LINES = 2


def _validate_line_limit(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _inline_text(text: str) -> str:
    return " ".join(str(text or "").split())


def _greedy_line_count(words: tuple[str, ...], width: int) -> int:
    lines = 1
    current_length = 0
    for word in words:
        candidate_length = len(word) if current_length == 0 else current_length + 1 + len(word)
        if current_length and candidate_length > width:
            lines += 1
            current_length = len(word)
        else:
            current_length = candidate_length
    return lines


def _balanced_lines(words: tuple[str, ...], line_count: int, width: int) -> tuple[str, ...]:
    """Partition words without loss, preferring width compliance and balance."""
    word_lengths = tuple(len(word) for word in words)
    prefix_lengths = [0]
    for length in word_lengths:
        prefix_lengths.append(prefix_lengths[-1] + length)

    def line_length(start: int, end: int) -> int:
        return prefix_lengths[end] - prefix_lengths[start] + (end - start - 1)

    # Newlines replace the spaces at line boundaries, so this is the average
    # rendered line length for an exactly ``line_count``-line partition.
    target = (prefix_lengths[-1] + len(words) - line_count) / line_count

    @cache
    def choose(start: int, remaining_lines: int) -> tuple[int, int, float, tuple[int, ...]]:
        if remaining_lines == 1:
            length = line_length(start, len(words))
            overflow = max(0, length - width)
            return overflow, overflow * overflow, (length - target) ** 2, (len(words),)

        best: tuple[int, int, float, tuple[int, ...]] | None = None
        last_end = len(words) - remaining_lines + 1
        for end in range(start + 1, last_end + 1):
            length = line_length(start, end)
            overflow = max(0, length - width)
            suffix_overflow, suffix_squared, suffix_balance, suffix_ends = choose(end, remaining_lines - 1)
            candidate = (
                overflow + suffix_overflow,
                overflow * overflow + suffix_squared,
                (length - target) ** 2 + suffix_balance,
                (end, *suffix_ends),
            )
            if best is None or candidate < best:
                best = candidate

        if best is None:  # pragma: no cover - guarded by the caller's line count
            raise RuntimeError("could not partition subtitle text")
        return best

    *_, ends = choose(0, line_count)
    lines: list[str] = []
    start = 0
    for end in ends:
        lines.append(" ".join(words[start:end]))
        start = end
    return tuple(lines)


def wrap_text_lossless(
    text: str,
    max_chars_per_line: int = DEFAULT_MAX_CHARS_PER_LINE,
    max_lines: int = DEFAULT_MAX_LINES,
) -> str:
    """Wrap normalized inline text without truncating or splitting words.

    Width is a preference, not a data-loss boundary. If the words cannot fit
    within the requested number of lines, one or more lines are allowed to
    exceed ``max_chars_per_line``.
    """
    _validate_line_limit(max_chars_per_line, "max_chars_per_line")
    _validate_line_limit(max_lines, "max_lines")

    normalized = _inline_text(text)
    if not normalized or max_lines == 1 or len(normalized) <= max_chars_per_line:
        return normalized

    words = tuple(normalized.split(" "))
    line_count = min(_greedy_line_count(words, max_chars_per_line), max_lines, len(words))
    if line_count <= 1:
        return normalized
    return "\n".join(_balanced_lines(words, line_count, max_chars_per_line))


def _render_block(index: int, cue: Cue, body: str) -> str:
    return "\n".join(
        (
            str(index),
            f"{srt_timestamp(cue.start)} --> {srt_timestamp(cue.end)}",
            body,
        )
    )


def render_srt(
    cues: Sequence[Cue],
    text_transform: Callable[[str], str] | None = None,
    max_chars_per_line: int = DEFAULT_MAX_CHARS_PER_LINE,
    max_lines: int = DEFAULT_MAX_LINES,
    speaker_labels: SpeakerLabels = SpeakerLabels.NONE,
    main_speaker: str | None = None,
) -> str:
    """Render timed cues as SubRip text without performing filesystem I/O."""
    _validate_line_limit(max_chars_per_line, "max_chars_per_line")
    _validate_line_limit(max_lines, "max_lines")

    labels = SpeakerLabels(speaker_labels)
    if main_speaker is None:
        speaker_counts = Counter(cue.speaker for cue in cues if cue.speaker)
        main_speaker = max(speaker_counts, key=speaker_counts.__getitem__) if speaker_counts else None
    blocks: list[str] = []
    for index, cue in enumerate(cues, start=1):
        text = cue.text
        if text_transform is not None:
            text = text_transform(text)
        should_label = bool(cue.speaker) and (
            labels is SpeakerLabels.ALL or (labels is SpeakerLabels.SECONDARY and cue.speaker != main_speaker)
        )
        if should_label:
            text = f"[{cue.speaker}] {text}"
        body = wrap_text_lossless(text, max_chars_per_line, max_lines)
        blocks.append(_render_block(index, cue, body))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_cue_index_srt(cues: Sequence[Cue]) -> str:
    """Render cues with their ordinal number as the visible subtitle text."""
    blocks = [_render_block(index, cue, str(index)) for index, cue in enumerate(cues, start=1)]
    return "\n\n".join(blocks) + ("\n" if blocks else "")
