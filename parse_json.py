import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

BASE_DIR = Path(__file__).resolve().parent
MEDIA_DIR = BASE_DIR / "media"
JSON_DIR = MEDIA_DIR / "JSON"
SRT_DIR = MEDIA_DIR / "SRT"
TXT_DIR = MEDIA_DIR / "TXT"
IN_JSON = JSON_DIR / "transcript.json"
OUT_SRT = SRT_DIR / "transcript.srt"

# Tuning knobs (good defaults for social + general subtitles)
MAX_CHARS_PER_LINE = 42
MAX_LINES = 2
MAX_CHARS = MAX_CHARS_PER_LINE * MAX_LINES  # ~84
MAX_DURATION = 5.5   # seconds per subtitle
MIN_DURATION = 1.0   # seconds; shorter gets merged if possible
GAP_SPLIT = 0.9      # split if there's a silence gap bigger than this (seconds)

PUNCT_END_RE = re.compile(r"[.!?…]+$")
SENTENCE_END_RE = re.compile(r"[.!?…]+$")
SentenceItem = Dict[str, Optional[str]]


def get_speaker_id(word: Dict) -> Optional[str]:
    # Normalize speaker ID across possible payload shapes.
    return word.get("speaker_id") or word.get("speaker")


def srt_timestamp(t: float) -> str:
    # Convert seconds to SRT timestamp format.
    if t < 0:
        t = 0.0
    ms = int(round(t * 1000))
    hh = ms // 3_600_000
    ms -= hh * 3_600_000
    mm = ms // 60_000
    ms -= mm * 60_000
    ss = ms // 1000
    ms -= ss * 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def normalize_spaces(text: str) -> str:
    # collapse whitespace, keep punctuation tight
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?…])", r"\1", text)
    return text


def text_to_sentences(text: str) -> List[str]:
    # Split plain text into sentences using punctuation boundaries.
    text = normalize_spaces(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?…])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def sentence_items_from_text(text: str) -> List[SentenceItem]:
    # Wrap plain sentences with empty speaker metadata.
    return [{"text": s, "speaker": None} for s in text_to_sentences(text)]


def compute_main_speaker(sentence_items: List[SentenceItem]) -> Optional[str]:
    # Find the speaker who owns the most sentences.
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
    # Format sentence line, optionally tagging every speaker.
    speaker_id = item.get("speaker")
    text = item.get("text") or ""
    if tag_all_speakers and speaker_id:
        return f"[{speaker_id}] {text}"
    if main_speaker and speaker_id and speaker_id != main_speaker:
        return f"[{speaker_id}] {text}"
    return text


def wrap_two_lines(text: str, max_chars_per_line: int = 42) -> str:
    # Simple greedy wrap into up to 2 lines
    words = text.split()
    lines: List[str] = []
    cur: List[str] = []

    for w in words:
        candidate = (" ".join(cur + [w])).strip()
        if len(candidate) <= max_chars_per_line or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
            if len(lines) == MAX_LINES - 1:
                # last line: dump rest
                break

    if cur:
        if len(lines) < MAX_LINES:
            remaining = words[len(" ".join(lines).split()):]
            # rebuild from remaining
            lines2 = []
            cur2 = []
            for w in remaining:
                cand = (" ".join(cur2 + [w])).strip()
                if len(cand) <= max_chars_per_line or not cur2:
                    cur2.append(w)
                else:
                    lines2.append(" ".join(cur2))
                    cur2 = [w]
                    if len(lines) + len(lines2) == MAX_LINES - 1:
                        break
            if cur2:
                lines2.append(" ".join(cur2))
            lines.extend(lines2)

    # hard cap 2 lines; if overflow, truncate with ellipsis
    lines = lines[:MAX_LINES]
    joined = "\n".join(lines)
    if len(joined.replace("\n", " ")) > MAX_CHARS:
        joined = (joined.replace("\n", " ")[: MAX_CHARS - 1].rstrip() + "…")
        joined = "\n".join([joined[:max_chars_per_line].rstrip(),
                            joined[max_chars_per_line:].lstrip()])
    return joined.strip()


def build_tokens(words: List[Dict]) -> List[Dict]:
    """
    Keep only word tokens; spacing tokens are ignored.
    Expected word shape: {text, start, end, type, speaker_id}
    """
    # Build a compact list of timecoded word tokens for SRT cueing.
    toks = []
    for w in words:
        if w.get("type") != "word":
            continue
        txt = w.get("text", "")
        if not txt:
            continue
        toks.append({
            "text": txt,
            "start": float(w["start"]),
            "end": float(w["end"]),
            "speaker": get_speaker_id(w)
        })
    return toks


def words_to_sentence_items(words: List[Dict]) -> List[SentenceItem]:
    # Build sentence items (text + dominant speaker) from word tokens.
    parts: List[str] = []
    sentences: List[SentenceItem] = []
    speaker_counts: Dict[str, int] = {}
    speaker_order: Dict[str, int] = {}
    order_idx = 0

    def note_speaker(speaker_id: Optional[str]) -> None:
        # Track speakers within the current sentence window.
        nonlocal order_idx
        if not speaker_id:
            return
        speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
        if speaker_id not in speaker_order:
            speaker_order[speaker_id] = order_idx
            order_idx += 1

    def pick_sentence_speaker() -> Optional[str]:
        # Choose the most frequent speaker in the sentence (ties by first seen).
        if not speaker_counts:
            return None
        return max(speaker_counts.items(), key=lambda kv: (kv[1], -speaker_order[kv[0]]))[0]

    def flush() -> None:
        # Emit the current sentence and reset buffers.
        nonlocal parts, speaker_counts, speaker_order, order_idx
        if parts:
            sentence = normalize_spaces("".join(parts))
            if sentence:
                sentences.append({"text": sentence, "speaker": pick_sentence_speaker()})
        parts = []
        speaker_counts = {}
        speaker_order = {}
        order_idx = 0

    for w in words:
        txt = (w.get("text") or "").strip()
        if not txt:
            continue
        t = w.get("type")
        if t == "punctuation":
            parts.append(txt)
            if SENTENCE_END_RE.search(txt):
                flush()
            continue
        if t != "word":
            continue
        if parts and not parts[-1].endswith((" ", "\n")):
            parts.append(" ")
        parts.append(txt)
        note_speaker(get_speaker_id(w))
        if SENTENCE_END_RE.search(txt):
            flush()

    flush()
    return sentences


def tokens_to_cues(tokens: List[Dict]) -> List[Tuple[float, float, str]]:
    # Convert word tokens into time-based subtitle cues.
    cues: List[Tuple[float, float, str]] = []
    if not tokens:
        return cues

    cur_text_parts: List[str] = []
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None
    last_end: Optional[float] = None

    def flush():
        # Finalize the current cue, if any.
        nonlocal cur_text_parts, cur_start, cur_end
        if not cur_text_parts or cur_start is None or cur_end is None:
            cur_text_parts = []
            cur_start = None
            cur_end = None
            return
        text = normalize_spaces(" ".join(cur_text_parts))
        if text:
            cues.append((cur_start, cur_end, text))
        cur_text_parts = []
        cur_start = None
        cur_end = None

    for tok in tokens:
        txt = tok["text"]
        st = tok["start"]
        en = tok["end"]

        if cur_start is None:
            cur_start = st
            cur_end = en
            cur_text_parts = [txt]
            last_end = en
            continue

        # split on big gap (silence)
        if last_end is not None and (st - last_end) > GAP_SPLIT:
            flush()
            cur_start = st
            cur_end = en
            cur_text_parts = [txt]
            last_end = en
            continue

        candidate = normalize_spaces(" ".join(cur_text_parts + [txt]))
        candidate_dur = en - cur_start

        # Decide whether to split BEFORE adding this token
        should_split = False

        if len(candidate) > MAX_CHARS:
            should_split = True

        if candidate_dur > MAX_DURATION:
            should_split = True

        # Prefer splitting at punctuation boundaries (after we've got a reasonable cue)
        if not should_split:
            # If current text already ends with punctuation and is long enough, start new cue
            cur_text_norm = normalize_spaces(" ".join(cur_text_parts))
            if (len(cur_text_norm) >= 28) and PUNCT_END_RE.search(cur_text_norm):
                should_split = True

        if should_split:
            flush()
            cur_start = st
            cur_end = en
            cur_text_parts = [txt]
        else:
            cur_text_parts.append(txt)
            cur_end = en

        last_end = en

    flush()

    # Merge too-short cues forward when possible
    merged: List[Tuple[float, float, str]] = []
    for st, en, tx in cues:
        if merged:
            pst, pen, ptx = merged[-1]
            if (en - st) < MIN_DURATION:
                candidate = normalize_spaces(ptx + " " + tx)
                if len(candidate) <= MAX_CHARS and (en - pst) <= MAX_DURATION:
                    merged[-1] = (pst, en, candidate)
                    continue
        merged.append((st, en, tx))

    return merged


def write_srt(cues: List[Tuple[float, float, str]], out_path: Path) -> None:
    # Serialize cues into an SRT file.
    lines: List[str] = []
    for i, (st, en, text) in enumerate(cues, start=1):
        wrapped = wrap_two_lines(text, MAX_CHARS_PER_LINE)
        lines.append(str(i))
        lines.append(f"{srt_timestamp(st)} --> {srt_timestamp(en)}")
        lines.append(wrapped)
        lines.append("")  # blank line
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def sentence_srt_path(out_path: Path) -> Path:
    # Build a sibling path like "file.sentence.srt".
    return out_path.with_name(f"{out_path.stem}.sentence{out_path.suffix}")


def write_sentence_srt(cues: List[Tuple[float, float, str]], out_path: Path) -> None:
    # Serialize cues into an SRT file with cue number as visible subtitle text.
    lines: List[str] = []
    for i, (st, en, _text) in enumerate(cues, start=1):
        lines.append(str(i))
        lines.append(f"{srt_timestamp(st)} --> {srt_timestamp(en)}")
        lines.append(str(i))
        lines.append("")  # blank line
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_txt_sentences(
    sentence_items: List[SentenceItem],
    out_path: Path,
    main_speaker: Optional[str] = None,
    tag_all_speakers: bool = False,
) -> None:
    # Write sentences to TXT, optionally tagging all speakers.
    if main_speaker is None and not tag_all_speakers:
        main_speaker = compute_main_speaker(sentence_items)
    lines = [format_sentence_line(item, main_speaker, tag_all_speakers) for item in sentence_items]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_speaker_remap(words: List[Dict]) -> Dict[str, str]:
    # Build a per-file mapping so the dominant speaker becomes speaker_0.
    stats: Dict[str, Dict[str, float]] = {}
    for idx, w in enumerate(words):
        if w.get("type") != "word":
            continue
        speaker_id = get_speaker_id(w)
        if not speaker_id:
            continue
        try:
            st = float(w.get("start", 0.0) or 0.0)
            en = float(w.get("end", st) or st)
        except (TypeError, ValueError):
            st = 0.0
            en = 0.0
        dur = max(0.0, en - st)
        if speaker_id not in stats:
            stats[speaker_id] = {"dur": 0.0, "count": 0.0, "first": float(idx)}
        stats[speaker_id]["dur"] += dur
        stats[speaker_id]["count"] += 1.0
    if not stats:
        return {}
    ordered = sorted(
        stats.items(),
        key=lambda kv: (-kv[1]["dur"], -kv[1]["count"], kv[1]["first"]),
    )
    return {speaker_id: f"speaker_{i}" for i, (speaker_id, _) in enumerate(ordered)}


def remap_sentence_items(sentence_items: List[SentenceItem], remap: Dict[str, str]) -> List[SentenceItem]:
    # Apply a speaker-id remap to sentence items.
    if not remap:
        return sentence_items
    remapped: List[SentenceItem] = []
    for item in sentence_items:
        speaker_id = item.get("speaker")
        if speaker_id and speaker_id in remap:
            remapped.append({"text": item.get("text"), "speaker": remap[speaker_id]})
        else:
            remapped.append(item)
    return remapped


def build_sentences_from_payload(payload: Dict) -> List[SentenceItem]:
    # Extract sentence items from payload (prefers word-level timing).
    words = payload.get("words")
    if isinstance(words, list) and words:
        return words_to_sentence_items(words)

    segments = payload.get("segments") or []
    if segments:
        text = " ".join([str(s.get("text", "")).strip() for s in segments])
        return sentence_items_from_text(text)

    return []


def combine_dir_to_txt(in_dir: Path, out_txt: Path) -> int:
    # Combine all JSON transcripts in a folder into one TXT.
    json_files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"])
    all_sentences: List[SentenceItem] = []
    for p in json_files:
        payload = json.loads(p.read_text(encoding="utf-8"))
        sentences = build_sentences_from_payload(payload)
        remap = build_speaker_remap(payload.get("words") or [])
        sentences = remap_sentence_items(sentences, remap)
        all_sentences.extend(sentences)
    write_txt_sentences(all_sentences, out_txt, tag_all_speakers=True)
    return len(all_sentences)


def build_srt_for_dir(
    in_dir: Path,
    out_srt_dir: Path,
    out_txt_dir: Path,
    write_sentence_copy: bool = False,
) -> int:
    # Build SRT (and TXT) for every JSON file in a folder.
    json_files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"])
    out_srt_dir.mkdir(parents=True, exist_ok=True)
    out_txt_dir.mkdir(parents=True, exist_ok=True)
    total_cues = 0
    for p in json_files:
        payload = json.loads(p.read_text(encoding="utf-8"))
        out_srt = out_srt_dir / f"{p.stem}.srt"
        out_txt = out_txt_dir / f"{p.stem}.txt"

        words = payload.get("words", [])
        tokens = build_tokens(words)
        cues = tokens_to_cues(tokens)
        write_srt(cues, out_srt)
        if write_sentence_copy:
            write_sentence_srt(cues, sentence_srt_path(out_srt))
        total_cues += len(cues)

        sentences = build_sentences_from_payload(payload)
        if sentences:
            remap = build_speaker_remap(payload.get("words") or [])
            sentences = remap_sentence_items(sentences, remap)
            write_txt_sentences(sentences, out_txt, tag_all_speakers=True)

    return total_cues


def main():
    # CLI entry point for single-file and batch combination.
    parser = argparse.ArgumentParser(description="Convert ElevenLabs JSON to SRT and/or combined TXT.")
    parser.add_argument("--input", type=Path, default=IN_JSON, help="Input JSON file.")
    parser.add_argument("--out-srt", type=Path, default=OUT_SRT, help="Output SRT file.")
    parser.add_argument("--out-txt", type=Path, default=TXT_DIR / "combined.txt", help="Output TXT file.")
    parser.add_argument("--combine-dir", type=Path, default=None, help="Directory with JSON files to combine.")
    parser.add_argument("--srt-dir", type=Path, default=None, help="Directory with JSON files to convert to per-file SRT/TXT.")
    parser.add_argument("--srt-out-dir", type=Path, default=SRT_DIR, help="Output directory for SRT files from --srt-dir.")
    parser.add_argument("--txt-out-dir", type=Path, default=TXT_DIR, help="Output directory for TXT files from --srt-dir.")
    parser.add_argument("--sentence-srt", action="store_true", help="Write a copy of each SRT with only cue numbers as visible text.")
    parser.add_argument("--no-srt", action="store_true", help="Skip SRT generation for --input.")
    parser.add_argument("--only-combine", action="store_true", help="Only build combined TXT from --combine-dir.")
    args = parser.parse_args()

    if args.only_combine and not args.combine_dir:
        raise SystemExit("--only-combine requires --combine-dir")

    if args.srt_dir:
        count = build_srt_for_dir(
            args.srt_dir,
            out_srt_dir=args.srt_out_dir,
            out_txt_dir=args.txt_out_dir,
            write_sentence_copy=args.sentence_srt,
        )
        print(f"Wrote SRT to {args.srt_out_dir} and TXT to {args.txt_out_dir} ({count} subtitles across JSON files)")
        return

    if args.combine_dir:
        count = combine_dir_to_txt(args.combine_dir, args.out_txt)
        print(f"Wrote: {args.out_txt} ({count} sentences)")
        if args.only_combine:
            return

    payload = json.loads(args.input.read_text(encoding="utf-8"))

    if not args.no_srt:
        words = payload.get("words", [])
        tokens = build_tokens(words)
        cues = tokens_to_cues(tokens)
        write_srt(cues, args.out_srt)
        print(f"Wrote: {args.out_srt} ({len(cues)} subtitles)")
        if args.sentence_srt:
            out_sentence_srt = sentence_srt_path(args.out_srt)
            write_sentence_srt(cues, out_sentence_srt)
            print(f"Wrote: {out_sentence_srt} ({len(cues)} numbered subtitles)")

    sentences = build_sentences_from_payload(payload)
    if sentences:
        remap = build_speaker_remap(payload.get("words") or [])
        sentences = remap_sentence_items(sentences, remap)
        write_txt_sentences(sentences, args.out_txt, tag_all_speakers=True)
        print(f"Wrote: {args.out_txt} ({len(sentences)} sentences)")

    print(f"Language detected: {payload.get('language_code')} (p={payload.get('language_probability')})")


if __name__ == "__main__":
    main()
