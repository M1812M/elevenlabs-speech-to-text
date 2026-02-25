import re
from pathlib import Path
from typing import Dict, List, Optional


ELLIPSIS = "\u2026"
SENTENCE_END_RE = re.compile(rf"[.!?{ELLIPSIS}]+$")
PUNCT_SPACING_RE = re.compile(rf"\s+([,.;:!?{ELLIPSIS}])")

SentenceItem = Dict[str, Optional[str]]


def get_speaker_id(word: Dict) -> Optional[str]:
    """Normalize speaker ID across possible payload shapes."""
    speaker_id = word.get("speaker_id")
    if speaker_id is not None:
        return speaker_id
    return word.get("speaker")


def normalize_spaces(text: str) -> str:
    """Normalize whitespace and punctuation spacing."""
    text = re.sub(r"\s+", " ", text).strip()
    return PUNCT_SPACING_RE.sub(r"\1", text)


def text_to_sentences(text: str) -> List[str]:
    """Split plain text into sentences using punctuation boundaries."""
    text = normalize_spaces(text)
    if not text:
        return []
    parts = re.split(rf"(?<=[.!?{ELLIPSIS}])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def sentence_items_from_text(text: str) -> List[SentenceItem]:
    """Wrap plain sentences with empty speaker metadata."""
    return [{"text": sentence, "speaker": None} for sentence in text_to_sentences(text)]


def compute_main_speaker(sentence_items: List[SentenceItem]) -> Optional[str]:
    """Find the speaker who owns the most sentences."""
    counts: Dict[str, int] = {}
    for item in sentence_items:
        speaker_id = item.get("speaker")
        if not speaker_id:
            continue
        counts[speaker_id] = counts.get(speaker_id, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def format_sentence_line(
    item: SentenceItem,
    main_speaker: Optional[str],
    tag_all_speakers: bool = False,
) -> str:
    """Format sentence line, optionally tagging every speaker."""
    speaker_id = item.get("speaker")
    text = item.get("text") or ""
    if tag_all_speakers and speaker_id:
        return f"[{speaker_id}] {text}"
    if main_speaker and speaker_id and speaker_id != main_speaker:
        return f"[{speaker_id}] {text}"
    return text


def words_to_sentence_items(words: List[Dict]) -> List[SentenceItem]:
    """Build sentence items (text + dominant speaker) from word tokens."""
    parts: List[str] = []
    sentences: List[SentenceItem] = []
    speaker_counts: Dict[str, int] = {}
    speaker_order: Dict[str, int] = {}
    order_idx = 0

    def note_speaker(speaker_id: Optional[str]) -> None:
        nonlocal order_idx
        if not speaker_id:
            return
        speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
        if speaker_id not in speaker_order:
            speaker_order[speaker_id] = order_idx
            order_idx += 1

    def pick_sentence_speaker() -> Optional[str]:
        if not speaker_counts:
            return None
        return max(speaker_counts.items(), key=lambda kv: (kv[1], -speaker_order[kv[0]]))[0]

    def flush() -> None:
        nonlocal parts, speaker_counts, speaker_order, order_idx
        if parts:
            sentence = normalize_spaces("".join(parts))
            if sentence:
                sentences.append({"text": sentence, "speaker": pick_sentence_speaker()})
        parts = []
        speaker_counts = {}
        speaker_order = {}
        order_idx = 0

    for word in words:
        txt = (word.get("text") or "").strip()
        if not txt:
            continue
        token_type = word.get("type")
        if token_type == "punctuation":
            parts.append(txt)
            if SENTENCE_END_RE.search(txt):
                flush()
            continue
        if token_type != "word":
            continue
        if parts and not parts[-1].endswith((" ", "\n")):
            parts.append(" ")
        parts.append(txt)
        note_speaker(get_speaker_id(word))
        if SENTENCE_END_RE.search(txt):
            flush()

    flush()
    return sentences


def payload_to_sentence_items(payload: Dict) -> List[SentenceItem]:
    """Extract sentence items from payload (prefers word-level timing)."""
    words = payload.get("words")
    if isinstance(words, list) and words:
        return words_to_sentence_items(words)

    segments = payload.get("segments") or []
    if segments:
        text = " ".join(str(segment.get("text", "")).strip() for segment in segments)
        return sentence_items_from_text(text)

    return []


def write_sentences_txt(
    sentence_items: List[SentenceItem],
    out_path: Path,
    main_speaker: Optional[str] = None,
    tag_all_speakers: bool = False,
) -> None:
    """Write sentences to TXT, optionally tagging all speakers."""
    if main_speaker is None and not tag_all_speakers:
        main_speaker = compute_main_speaker(sentence_items)
    lines = [format_sentence_line(item, main_speaker, tag_all_speakers) for item in sentence_items]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_speaker_remap(words: List[Dict]) -> Dict[str, str]:
    """Build a per-file mapping so the dominant speaker becomes speaker_0."""
    stats: Dict[str, Dict[str, float]] = {}
    for idx, word in enumerate(words):
        if word.get("type") != "word":
            continue
        speaker_id = get_speaker_id(word)
        if not speaker_id:
            continue
        try:
            start = float(word.get("start", 0.0) or 0.0)
            end = float(word.get("end", start) or start)
        except (TypeError, ValueError):
            start = 0.0
            end = 0.0
        duration = max(0.0, end - start)
        if speaker_id not in stats:
            stats[speaker_id] = {"dur": 0.0, "count": 0.0, "first": float(idx)}
        stats[speaker_id]["dur"] += duration
        stats[speaker_id]["count"] += 1.0
    if not stats:
        return {}
    ordered = sorted(
        stats.items(),
        key=lambda kv: (-kv[1]["dur"], -kv[1]["count"], kv[1]["first"]),
    )
    return {speaker_id: f"speaker_{i}" for i, (speaker_id, _) in enumerate(ordered)}


def remap_sentence_items(sentence_items: List[SentenceItem], remap: Dict[str, str]) -> List[SentenceItem]:
    """Apply a speaker-id remap to sentence items."""
    if not remap:
        return sentence_items
    remapped: List[SentenceItem] = []
    for item in sentence_items:
        speaker_id = item.get("speaker")
        if speaker_id and speaker_id in remap:
            remapped.append({"text": item.get("text"), "speaker": remap[speaker_id]})
            continue
        remapped.append(item)
    return remapped

