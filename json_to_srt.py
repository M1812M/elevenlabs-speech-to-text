import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

BASE_DIR = Path(__file__).resolve().parent
IN_JSON = BASE_DIR / "transcript.json"
OUT_SRT = BASE_DIR / "transcript.srt"

# Tuning knobs (good defaults for social + general subtitles)
MAX_CHARS_PER_LINE = 42
MAX_LINES = 2
MAX_CHARS = MAX_CHARS_PER_LINE * MAX_LINES  # ~84
MAX_DURATION = 5.5   # seconds per subtitle
MIN_DURATION = 1.0   # seconds; shorter gets merged if possible
GAP_SPLIT = 0.9      # split if there's a silence gap bigger than this (seconds)

PUNCT_END_RE = re.compile(r"[.!?…]+$")
SENTENCE_END_RE = re.compile(r"[.!?…]+$")


def srt_timestamp(t: float) -> str:
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
    text = normalize_spaces(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?…])\s+", text)
    return [p.strip() for p in parts if p.strip()]


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
            "speaker": w.get("speaker_id")
        })
    return toks


def words_to_sentences(words: List[Dict]) -> List[str]:
    parts: List[str] = []
    sentences: List[str] = []

    def flush():
        nonlocal parts
        if parts:
            sentence = normalize_spaces("".join(parts))
            if sentence:
                sentences.append(sentence)
        parts = []

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
        if SENTENCE_END_RE.search(txt):
            flush()

    flush()
    return sentences


def tokens_to_cues(tokens: List[Dict]) -> List[Tuple[float, float, str]]:
    cues: List[Tuple[float, float, str]] = []
    if not tokens:
        return cues

    cur_text_parts: List[str] = []
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None
    last_end: Optional[float] = None

    def flush():
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
    lines: List[str] = []
    for i, (st, en, text) in enumerate(cues, start=1):
        wrapped = wrap_two_lines(text, MAX_CHARS_PER_LINE)
        lines.append(str(i))
        lines.append(f"{srt_timestamp(st)} --> {srt_timestamp(en)}")
        lines.append(wrapped)
        lines.append("")  # blank line
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_txt_sentences(sentences: List[str], out_path: Path) -> None:
    out_path.write_text("\n".join(sentences) + ("\n" if sentences else ""), encoding="utf-8")


def build_sentences_from_payload(payload: Dict) -> List[str]:
    words = payload.get("words")
    if isinstance(words, list) and words:
        return words_to_sentences(words)

    segments = payload.get("segments") or []
    if segments:
        text = " ".join([str(s.get("text", "")).strip() for s in segments])
        return text_to_sentences(text)

    return []


def combine_dir_to_txt(in_dir: Path, out_txt: Path) -> int:
    json_files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"])
    all_sentences: List[str] = []
    for p in json_files:
        payload = json.loads(p.read_text(encoding="utf-8"))
        sentences = build_sentences_from_payload(payload)
        all_sentences.extend(sentences)
    write_txt_sentences(all_sentences, out_txt)
    return len(all_sentences)


def main():
    parser = argparse.ArgumentParser(description="Convert ElevenLabs JSON to SRT and/or combined TXT.")
    parser.add_argument("--input", type=Path, default=IN_JSON, help="Input JSON file.")
    parser.add_argument("--out-srt", type=Path, default=OUT_SRT, help="Output SRT file.")
    parser.add_argument("--out-txt", type=Path, default=BASE_DIR / "combined.txt", help="Output TXT file.")
    parser.add_argument("--combine-dir", type=Path, default=None, help="Directory with JSON files to combine.")
    parser.add_argument("--no-srt", action="store_true", help="Skip SRT generation for --input.")
    parser.add_argument("--only-combine", action="store_true", help="Only build combined TXT from --combine-dir.")
    args = parser.parse_args()

    if args.only_combine and not args.combine_dir:
        raise SystemExit("--only-combine requires --combine-dir")

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

    sentences = build_sentences_from_payload(payload)
    if sentences:
        write_txt_sentences(sentences, args.out_txt)
        print(f"Wrote: {args.out_txt} ({len(sentences)} sentences)")

    print(f"Language detected: {payload.get('language_code')} (p={payload.get('language_probability')})")


if __name__ == "__main__":
    main()
