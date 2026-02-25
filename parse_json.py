import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from elevenlabs_toolkit.paths import JSON_DIR, SRT_DIR, TXT_DIR
from elevenlabs_toolkit.timecode import srt_timestamp
from elevenlabs_toolkit.transcript_utils import (
    SentenceItem,
    build_speaker_remap,
    get_speaker_id,
    normalize_spaces,
    payload_to_sentence_items,
    remap_sentence_items,
    write_sentences_txt,
)


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


IN_JSON = JSON_DIR / "transcript.json"
OUT_SRT = SRT_DIR / "transcript.srt"

# Tuning knobs (good defaults for social + general subtitles)
MAX_CHARS_PER_LINE = 42
MAX_LINES = 2
MAX_CHARS = MAX_CHARS_PER_LINE * MAX_LINES  # ~84
MAX_DURATION = 5.5   # seconds per subtitle
MIN_DURATION = 1.0   # seconds; shorter gets merged if possible
GAP_SPLIT = 0.9      # split if there's a silence gap bigger than this (seconds)

PUNCT_END_RE = re.compile(r"[.!?\u2026]+$")


def wrap_two_lines(text: str, max_chars_per_line: int = 42) -> str:
    # Simple greedy wrap into up to 2 lines.
    words = text.split()
    lines: List[str] = []
    cur: List[str] = []

    for word in words:
        candidate = " ".join(cur + [word]).strip()
        if len(candidate) <= max_chars_per_line or not cur:
            cur.append(word)
        else:
            lines.append(" ".join(cur))
            cur = [word]
            if len(lines) == MAX_LINES - 1:
                break

    if cur:
        if len(lines) < MAX_LINES:
            remaining = words[len(" ".join(lines).split()):]
            lines2: List[str] = []
            cur2: List[str] = []
            for word in remaining:
                candidate = " ".join(cur2 + [word]).strip()
                if len(candidate) <= max_chars_per_line or not cur2:
                    cur2.append(word)
                else:
                    lines2.append(" ".join(cur2))
                    cur2 = [word]
                    if len(lines) + len(lines2) == MAX_LINES - 1:
                        break
            if cur2:
                lines2.append(" ".join(cur2))
            lines.extend(lines2)

    # Hard cap 2 lines; if overflow, truncate with ellipsis.
    lines = lines[:MAX_LINES]
    joined = "\n".join(lines)
    if len(joined.replace("\n", " ")) > MAX_CHARS:
        joined = joined.replace("\n", " ")[: MAX_CHARS - 1].rstrip() + "\u2026"
        joined = "\n".join(
            [
                joined[:max_chars_per_line].rstrip(),
                joined[max_chars_per_line:].lstrip(),
            ]
        )
    return joined.strip()


def build_tokens(words: List[Dict]) -> List[Dict]:
    """
    Keep only word tokens; spacing tokens are ignored.
    Expected word shape: {text, start, end, type, speaker_id}
    """
    toks = []
    for word in words:
        if word.get("type") != "word":
            continue
        txt = word.get("text", "")
        if not txt:
            continue
        toks.append(
            {
                "text": txt,
                "start": float(word["start"]),
                "end": float(word["end"]),
                "speaker": get_speaker_id(word),
            }
        )
    return toks


def tokens_to_cues(tokens: List[Dict]) -> List[Tuple[float, float, str]]:
    # Convert word tokens into time-based subtitle cues.
    cues: List[Tuple[float, float, str]] = []
    if not tokens:
        return cues

    cur_text_parts: List[str] = []
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None
    last_end: Optional[float] = None

    def flush() -> None:
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

        # Split on big gap (silence).
        if last_end is not None and (st - last_end) > GAP_SPLIT:
            flush()
            cur_start = st
            cur_end = en
            cur_text_parts = [txt]
            last_end = en
            continue

        candidate = normalize_spaces(" ".join(cur_text_parts + [txt]))
        candidate_dur = en - cur_start

        should_split = False
        if len(candidate) > MAX_CHARS:
            should_split = True
        if candidate_dur > MAX_DURATION:
            should_split = True

        # Prefer splitting at punctuation boundaries (after we've got a reasonable cue).
        if not should_split:
            cur_text_norm = normalize_spaces(" ".join(cur_text_parts))
            if len(cur_text_norm) >= 28 and PUNCT_END_RE.search(cur_text_norm):
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

    # Merge too-short cues forward when possible.
    merged: List[Tuple[float, float, str]] = []
    for st, en, tx in cues:
        if merged:
            prev_start, prev_end, prev_text = merged[-1]
            if (en - st) < MIN_DURATION:
                candidate = normalize_spaces(prev_text + " " + tx)
                if len(candidate) <= MAX_CHARS and (en - prev_start) <= MAX_DURATION:
                    merged[-1] = (prev_start, en, candidate)
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
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def sentence_srt_path(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}.sentence{out_path.suffix}")


def write_sentence_srt(cues: List[Tuple[float, float, str]], out_path: Path) -> None:
    # Serialize cues into an SRT file with cue number as visible subtitle text.
    lines: List[str] = []
    for i, (st, en, _text) in enumerate(cues, start=1):
        lines.append(str(i))
        lines.append(f"{srt_timestamp(st)} --> {srt_timestamp(en)}")
        lines.append(str(i))
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def combine_dir_to_txt(in_dir: Path, out_txt: Path) -> int:
    # Combine all JSON transcripts in a folder into one TXT.
    json_files = sorted(p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json")
    all_sentences: List[SentenceItem] = []
    for path in json_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        sentences = payload_to_sentence_items(payload)
        remap = build_speaker_remap(payload.get("words") or [])
        sentences = remap_sentence_items(sentences, remap)
        all_sentences.extend(sentences)
    write_sentences_txt(all_sentences, out_txt, tag_all_speakers=True)
    return len(all_sentences)


def build_srt_for_dir(
    in_dir: Path,
    out_srt_dir: Path,
    out_txt_dir: Path,
    write_sentence_copy: bool = False,
) -> int:
    # Build SRT (and TXT) for every JSON file in a folder.
    json_files = sorted(p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json")
    out_srt_dir.mkdir(parents=True, exist_ok=True)
    out_txt_dir.mkdir(parents=True, exist_ok=True)
    total_cues = 0

    for path in json_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        out_srt = out_srt_dir / f"{path.stem}.srt"
        out_txt = out_txt_dir / f"{path.stem}.txt"

        words = payload.get("words", [])
        tokens = build_tokens(words)
        cues = tokens_to_cues(tokens)
        write_srt(cues, out_srt)
        if write_sentence_copy:
            write_sentence_srt(cues, sentence_srt_path(out_srt))
        total_cues += len(cues)

        sentences = payload_to_sentence_items(payload)
        if sentences:
            remap = build_speaker_remap(payload.get("words") or [])
            sentences = remap_sentence_items(sentences, remap)
            write_sentences_txt(sentences, out_txt, tag_all_speakers=True)

    return total_cues


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ElevenLabs JSON transcripts into subtitle and text outputs.\n"
            "Works on a single JSON file or on entire directories."
        ),
        formatter_class=HelpFormatter,
        epilog=(
            "Examples:\n"
            "  python parse_json.py --input media/JSON/file.json --out-srt media/SRT/file.srt --out-txt media/TXT/file.txt\n"
            "  python parse_json.py --srt-dir media/JSON\n"
            "  python parse_json.py --srt-dir media/JSON --sentence-srt\n"
            "  python parse_json.py --combine-dir media/JSON --out-txt media/TXT/combined.txt --only-combine"
        ),
    )
    parser.add_argument("--input", type=Path, default=IN_JSON, help="Input JSON for single-file mode.")
    parser.add_argument("--out-srt", type=Path, default=OUT_SRT, help="Output SRT path for single-file mode.")
    parser.add_argument(
        "--out-txt",
        type=Path,
        default=TXT_DIR / "combined.txt",
        help="Output TXT path for single-file mode or combined output.",
    )
    parser.add_argument("--combine-dir", type=Path, default=None, help="Combine all JSON files in this directory into one TXT.")
    parser.add_argument("--srt-dir", type=Path, default=None, help="Batch-convert all JSON files in this directory to per-file SRT/TXT.")
    parser.add_argument("--srt-out-dir", type=Path, default=SRT_DIR, help="Output directory for SRT files when --srt-dir is used.")
    parser.add_argument("--txt-out-dir", type=Path, default=TXT_DIR, help="Output directory for TXT files when --srt-dir is used.")
    parser.add_argument("--sentence-srt", action="store_true", help="Also write *.sentence.srt with cue numbers as visible subtitle text.")
    parser.add_argument("--no-srt", action="store_true", help="Skip SRT generation in single-file mode.")
    parser.add_argument("--only-combine", action="store_true", help="With --combine-dir: write only combined TXT and exit.")
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

    sentences = payload_to_sentence_items(payload)
    if sentences:
        remap = build_speaker_remap(payload.get("words") or [])
        sentences = remap_sentence_items(sentences, remap)
        write_sentences_txt(sentences, args.out_txt, tag_all_speakers=True)
        print(f"Wrote: {args.out_txt} ({len(sentences)} sentences)")

    print(f"Language detected: {payload.get('language_code')} (p={payload.get('language_probability')})")


if __name__ == "__main__":
    main()

