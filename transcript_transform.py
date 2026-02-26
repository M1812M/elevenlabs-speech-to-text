import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from elevenlabs_toolkit.paths import JSON_DIR, SOCIAL_SRT_DIR, SRT_DIR, TXT_DIR
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


MAX_CHARS_PER_LINE = 42
MAX_LINES = 2
MAX_CHARS = MAX_CHARS_PER_LINE * MAX_LINES
MAX_DURATION = 5.5
MIN_DURATION = 1.0
GAP_SPLIT = 0.9
PUNCT_END_RE = re.compile(r"[.!?\u2026]+$")

SOCIAL_MAX_CHARS_PER_LINE = 30
SOCIAL_MAX_LINES = 2
SOCIAL_MAX_CHARS = SOCIAL_MAX_CHARS_PER_LINE * SOCIAL_MAX_LINES
SOCIAL_MAX_WORDS = 9
SOCIAL_MAX_DURATION = 2.6
SOCIAL_MIN_DURATION = 0.9
SOCIAL_GAP_SPLIT = 0.75
SOCIAL_HARD_END_RE = re.compile(r"[.!?\u2026]+$")
SOCIAL_SOFT_END_RE = re.compile(r"[,;:]+$")

UZ_APOS = "\u2018"
APOS_RE = re.compile(r"[\u02bb\u02bc\u2018\u2019`\u00b4']")
CYRILLIC_LETTER_RE = re.compile(r"[\u0400-\u04FF]")
CYRILLIC_VOWELS_AND_SIGNS = set(
    "\u0410\u0430\u0415\u0435\u0401\u0451\u0418\u0438\u041e\u043e\u0423\u0443"
    "\u040e\u045e\u042d\u044d\u042e\u044e\u042f\u044f\u042a\u044a\u042c\u044c"
)

LATIN_DIGRAPHS_TO_CYRILLIC = [
    (re.compile(r"o'", flags=re.IGNORECASE), "\u045e"),
    (re.compile(r"g'", flags=re.IGNORECASE), "\u0493"),
    (re.compile(r"sh", flags=re.IGNORECASE), "\u0448"),
    (re.compile(r"ch", flags=re.IGNORECASE), "\u0447"),
    (re.compile(r"ng", flags=re.IGNORECASE), "\u043d\u0433"),
    (re.compile(r"yo", flags=re.IGNORECASE), "\u0451"),
    (re.compile(r"yu", flags=re.IGNORECASE), "\u044e"),
    (re.compile(r"ya", flags=re.IGNORECASE), "\u044f"),
    (re.compile(r"ye", flags=re.IGNORECASE), "\u0435"),
]

LATIN_TO_CYRILLIC_CHARS = {
    "a": "\u0430",
    "b": "\u0431",
    "c": "\u0446",
    "d": "\u0434",
    "e": "\u0435",
    "f": "\u0444",
    "g": "\u0433",
    "h": "\u04b3",
    "i": "\u0438",
    "j": "\u0436",
    "k": "\u043a",
    "l": "\u043b",
    "m": "\u043c",
    "n": "\u043d",
    "o": "\u043e",
    "p": "\u043f",
    "q": "\u049b",
    "r": "\u0440",
    "s": "\u0441",
    "t": "\u0442",
    "u": "\u0443",
    "v": "\u0432",
    "w": "\u0432",
    "x": "\u0445",
    "y": "\u0439",
    "z": "\u0437",
}

CYRILLIC_TO_LATIN_CHARS = {
    "\u0410": "A",
    "\u0430": "a",
    "\u0411": "B",
    "\u0431": "b",
    "\u0412": "V",
    "\u0432": "v",
    "\u0413": "G",
    "\u0433": "g",
    "\u0492": f"G{UZ_APOS}",
    "\u0493": f"g{UZ_APOS}",
    "\u0414": "D",
    "\u0434": "d",
    "\u0401": "Yo",
    "\u0451": "yo",
    "\u0416": "J",
    "\u0436": "j",
    "\u0417": "Z",
    "\u0437": "z",
    "\u0418": "I",
    "\u0438": "i",
    "\u0419": "Y",
    "\u0439": "y",
    "\u041a": "K",
    "\u043a": "k",
    "\u049a": "Q",
    "\u049b": "q",
    "\u041b": "L",
    "\u043b": "l",
    "\u041c": "M",
    "\u043c": "m",
    "\u041d": "N",
    "\u043d": "n",
    "\u041e": "O",
    "\u043e": "o",
    "\u041f": "P",
    "\u043f": "p",
    "\u0420": "R",
    "\u0440": "r",
    "\u0421": "S",
    "\u0441": "s",
    "\u0422": "T",
    "\u0442": "t",
    "\u0423": "U",
    "\u0443": "u",
    "\u0424": "F",
    "\u0444": "f",
    "\u0425": "X",
    "\u0445": "x",
    "\u04b2": "H",
    "\u04b3": "h",
    "\u0426": "S",
    "\u0446": "s",
    "\u0427": "Ch",
    "\u0447": "ch",
    "\u0428": "Sh",
    "\u0448": "sh",
    "\u0429": "Sh",
    "\u0449": "sh",
    "\u042a": "'",
    "\u044a": "'",
    "\u042c": "'",
    "\u044c": "'",
    "\u042d": "E",
    "\u044d": "e",
    "\u042e": "Yu",
    "\u044e": "yu",
    "\u042f": "Ya",
    "\u044f": "ya",
    "\u040e": f"O{UZ_APOS}",
    "\u045e": f"o{UZ_APOS}",
}

TIMECODE_RE = re.compile(r"^\s*\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}\s*$")
HTML_TAG_RE = re.compile(r"(<[^>]+>)")


def wrap_two_lines(text: str, max_chars_per_line: int = MAX_CHARS_PER_LINE) -> str:
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

    if cur and len(lines) < MAX_LINES:
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


def build_standard_tokens(words: List[Dict]) -> List[Dict]:
    tokens = []
    for word in words:
        if word.get("type") != "word":
            continue
        txt = word.get("text", "")
        if not txt:
            continue
        tokens.append(
            {
                "text": txt,
                "start": float(word.get("start", 0.0)),
                "end": float(word.get("end", 0.0)),
                "speaker": get_speaker_id(word),
            }
        )
    return tokens


def tokens_to_standard_cues(tokens: List[Dict]) -> List[Tuple[float, float, str]]:
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

    for token in tokens:
        txt = token["text"]
        st = token["start"]
        en = token["end"]

        if cur_start is None:
            cur_start = st
            cur_end = en
            cur_text_parts = [txt]
            last_end = en
            continue

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
        if len(candidate) > MAX_CHARS or candidate_dur > MAX_DURATION:
            should_split = True

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
    lines: List[str] = []
    for i, (st, en, _text) in enumerate(cues, start=1):
        lines.append(str(i))
        lines.append(f"{srt_timestamp(st)} --> {srt_timestamp(en)}")
        lines.append(str(i))
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def social_normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?\u2026])", r"\1", text)
    return text


def apply_case(sample: str, replacement: str) -> str:
    if sample.isupper():
        return replacement.upper()
    if sample.islower():
        return replacement.lower()
    if len(sample) > 1 and sample[0].isupper() and sample[1:].islower():
        return replacement[0].upper() + replacement[1:].lower()
    return replacement


def is_cyrillic_letter(ch: Optional[str]) -> bool:
    return bool(ch) and CYRILLIC_LETTER_RE.match(ch) is not None


def to_cyrillic(text: str) -> str:
    value = APOS_RE.sub("'", text)
    for pattern, replacement in LATIN_DIGRAPHS_TO_CYRILLIC:
        value = pattern.sub(lambda m: apply_case(m.group(0), replacement), value)

    out_chars: List[str] = []
    for idx, ch in enumerate(value):
        lower = ch.lower()
        mapped = LATIN_TO_CYRILLIC_CHARS.get(lower)
        if mapped is None:
            if ch == "'":
                out_chars.append("\u044a")
            else:
                out_chars.append(ch)
            continue

        if lower == "e":
            prev = value[idx - 1] if idx > 0 else None
            if prev is None or not prev.isalpha():
                mapped = "\u044d"

        out_chars.append(mapped.upper() if ch.isupper() else mapped)

    return "".join(out_chars)


def to_latin(text: str) -> str:
    out_parts: List[str] = []
    for idx, ch in enumerate(text):
        if ch in ("\u0415", "\u0435"):
            prev = text[idx - 1] if idx > 0 else None
            if not is_cyrillic_letter(prev) or (prev in CYRILLIC_VOWELS_AND_SIGNS):
                out_parts.append("Ye" if ch == "\u0415" else "ye")
            else:
                out_parts.append("E" if ch == "\u0415" else "e")
            continue

        out_parts.append(CYRILLIC_TO_LATIN_CHARS.get(ch, ch))

    return "".join(out_parts)


def build_social_word_tokens(payload: Dict) -> List[Dict]:
    out = []
    for word in payload.get("words", []):
        if word.get("type") != "word":
            continue
        txt = word.get("text", "")
        if not txt:
            continue
        out.append({"text": txt, "start": float(word.get("start", 0.0)), "end": float(word.get("end", 0.0))})
    return out


def tokens_to_social_cues(tokens: List[Dict]) -> List[Tuple[float, float, str]]:
    cues: List[Tuple[float, float, str]] = []
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None
    cur_parts: List[str] = []
    cur_words = 0
    last_end: Optional[float] = None

    def flush() -> None:
        nonlocal cur_start, cur_end, cur_parts, cur_words
        if cur_start is not None and cur_end is not None and cur_parts:
            text = social_normalize(" ".join(cur_parts))
            if text:
                cues.append((cur_start, cur_end, text))
        cur_start = None
        cur_end = None
        cur_parts = []
        cur_words = 0

    for token in tokens:
        txt = token["text"]
        st = token["start"]
        en = token["end"]

        if cur_start is None:
            cur_start = st

        if last_end is not None and (st - last_end) > SOCIAL_GAP_SPLIT and cur_parts:
            flush()
            cur_start = st

        tentative_parts = cur_parts + [txt]
        tentative_text = social_normalize(" ".join(tentative_parts))
        tentative_words = cur_words + 1
        tentative_dur = en - cur_start

        if SOCIAL_HARD_END_RE.search(tentative_text):
            cur_parts = tentative_parts
            cur_words = tentative_words
            cur_end = en
            flush()
            last_end = en
            continue

        hard_limit_hit = (
            len(tentative_text) > SOCIAL_MAX_CHARS
            or tentative_words > SOCIAL_MAX_WORDS
            or tentative_dur > SOCIAL_MAX_DURATION
        )
        if hard_limit_hit and cur_parts:
            flush()
            cur_start = st
            cur_parts = [txt]
            cur_words = 1
            cur_end = en
            last_end = en
            continue

        if SOCIAL_SOFT_END_RE.search(tentative_text):
            if (len(tentative_text) >= 20) or (tentative_words >= 5) or (tentative_dur >= SOCIAL_MIN_DURATION):
                cur_parts = tentative_parts
                cur_words = tentative_words
                cur_end = en
                flush()
                last_end = en
                continue

        cur_parts = tentative_parts
        cur_words = tentative_words
        cur_end = en
        last_end = en

    flush()

    merged: List[Tuple[float, float, str]] = []
    for st, en, tx in cues:
        if merged:
            prev_start, prev_end, prev_text = merged[-1]
            if (en - st) < SOCIAL_MIN_DURATION:
                candidate = social_normalize(prev_text + " " + tx)
                if len(candidate) <= SOCIAL_MAX_CHARS and (en - prev_start) <= SOCIAL_MAX_DURATION:
                    merged[-1] = (prev_start, en, candidate)
                    continue
        merged.append((st, en, tx))

    return merged


def cues_to_social_srt(cues: List[Tuple[float, float, str]], transform: Optional[Callable[[str], str]] = None) -> str:
    out = []
    for i, (st, en, tx) in enumerate(cues, 1):
        single_line = social_normalize(tx)
        if transform is not None:
            single_line = social_normalize(transform(single_line))
        out.append(str(i))
        out.append(f"{srt_timestamp(st)} --> {srt_timestamp(en)}")
        out.append(single_line)
        out.append("")
    return "\n".join(out)


def is_srt_meta_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.isdigit():
        return True
    return TIMECODE_RE.match(stripped) is not None


def latin_srt_to_cyrillic_text(srt_text: str) -> str:
    out_lines: List[str] = []
    for line in srt_text.splitlines():
        if is_srt_meta_line(line):
            out_lines.append(line)
            continue
        parts = HTML_TAG_RE.split(line)
        transformed_parts = [part if HTML_TAG_RE.fullmatch(part) else to_cyrillic(part) for part in parts]
        out_lines.append("".join(transformed_parts))
    return "\n".join(out_lines) + ("\n" if srt_text.endswith(("\n", "\r")) else "")


def cyrillic_output_path_for_latin(latin_srt_path: Path, out_dir: Optional[Path] = None) -> Path:
    name = latin_srt_path.name
    if name.endswith("_latin.srt"):
        out_name = name[:-10] + "_cyrillic.srt"
    elif name.endswith(".srt"):
        out_name = name[:-4] + "_cyrillic.srt"
    else:
        out_name = name + "_cyrillic.srt"
    if out_dir is not None:
        return out_dir / out_name
    return latin_srt_path.with_name(out_name)


def ensure_dir(path: Path, arg_name: str) -> Path:
    if path.exists() and not path.is_dir():
        raise ValueError(f"{arg_name} must be a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def collect_json_sources(path: Path, glob_pattern: str) -> List[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if path.is_file():
        if path.suffix.lower() != ".json":
            raise ValueError(f"--path points to a file but not JSON: {path}")
        return [path]

    json_files = sorted(p for p in path.glob(glob_pattern) if p.is_file() and p.suffix.lower() == ".json")
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {path} matching {glob_pattern}")
    return json_files


def collect_latin_srt_sources(path: Path, glob_pattern: str) -> List[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if path.is_file():
        if path.suffix.lower() != ".srt":
            raise ValueError(f"--path points to a file but not SRT: {path}")
        return [path]

    srt_files = sorted(p for p in path.glob(glob_pattern) if p.is_file() and p.suffix.lower() == ".srt")
    if not srt_files:
        raise FileNotFoundError(f"No Latin SRT files found in {path} matching {glob_pattern}")
    return srt_files


def infer_combined_txt_name(json_files: List[Path]) -> str:
    if not json_files:
        return "combined.txt"

    stems = [path.stem for path in json_files]
    if len(stems) == 1:
        base = stems[0].strip(" ._-")
        return f"{base}_comb.txt" if base else "combined.txt"

    common_prefix = os.path.commonprefix(stems).strip(" ._-")
    if common_prefix:
        return f"{common_prefix}_comb.txt"
    return "combined.txt"


def parse_args() -> Optional[argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description=(
            "Transform existing transcript files (JSON/SRT) without calling ElevenLabs.\n"
            "Use --path with a file or directory and select one or more create/convert actions."
        ),
        formatter_class=HelpFormatter,
        epilog=(
            "Examples:\n"
            "  python transcript_transform.py --path media/JSON --create-srt --create-txt\n"
            "  python transcript_transform.py --path media/JSON --create-txt-combined\n"
            "  python transcript_transform.py --path media/JSON --create-social-srt-latin --create-social-srt-cyrillic\n"
            "  python transcript_transform.py --path media/SRT-social --convert-latin-srt-to-cyrillic"
        ),
    )
    parser.add_argument("--path", type=Path, default=None, help="Input file or directory.")
    parser.add_argument("--json-glob", type=str, default="*.json", help="Glob for JSON files when --path is a directory.")
    parser.add_argument(
        "--latin-srt-glob",
        type=str,
        default="*_latin.srt",
        help="Glob for Latin SRT files when --path is a directory and converting to Cyrillic.",
    )

    parser.add_argument("--create-srt", action="store_true", help="Create standard SRT files from JSON inputs.")
    parser.add_argument(
        "--create-sentence-srt",
        action="store_true",
        help="Also create *.sentence.srt with cue numbers as visible text.",
    )
    parser.add_argument("--create-txt", action="store_true", help="Create per-file TXT sentence outputs from JSON inputs.")
    parser.add_argument("--create-txt-combined", action="store_true", help="Create one combined TXT from all JSON inputs.")
    parser.add_argument(
        "--create-social-srt-latin",
        action="store_true",
        help="Create *_social_latin.srt from JSON inputs.",
    )
    parser.add_argument(
        "--create-social-srt-cyrillic",
        action="store_true",
        help="Create *_social_cyrillic.srt from JSON inputs.",
    )
    parser.add_argument(
        "--convert-latin-srt-to-cyrillic",
        action="store_true",
        help="Convert Latin SRT input(s) to Cyrillic while preserving timing and HTML tags.",
    )

    parser.add_argument("--srt-out-dir", type=Path, default=SRT_DIR, help="Output directory for --create-srt.")
    parser.add_argument("--txt-out-dir", type=Path, default=TXT_DIR, help="Output directory for TXT outputs.")
    parser.add_argument(
        "--social-out-dir",
        type=Path,
        default=SOCIAL_SRT_DIR,
        help="Output directory for social SRT outputs.",
    )
    parser.add_argument(
        "--latin-cyr-out-dir",
        type=Path,
        default=None,
        help="Optional output directory for --convert-latin-srt-to-cyrillic. Default: next to source file.",
    )
    parser.add_argument(
        "--combined-txt-path",
        type=Path,
        default=None,
        help="Explicit output path for --create-txt-combined. Default is inferred from source name(s).",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        return None

    args = parser.parse_args()

    selected_actions = [
        args.create_srt,
        args.create_sentence_srt,
        args.create_txt,
        args.create_txt_combined,
        args.create_social_srt_latin,
        args.create_social_srt_cyrillic,
        args.convert_latin_srt_to_cyrillic,
    ]
    if not any(selected_actions):
        parser.error(
            "Select at least one action: --create-srt, --create-txt, --create-txt-combined, "
            "--create-social-srt-latin, --create-social-srt-cyrillic, or --convert-latin-srt-to-cyrillic."
        )

    if args.path is None:
        parser.error("--path is required when selecting actions.")

    if args.create_sentence_srt:
        args.create_srt = True

    if args.path.is_file() and args.convert_latin_srt_to_cyrillic and (
        args.create_srt
        or args.create_txt
        or args.create_txt_combined
        or args.create_social_srt_latin
        or args.create_social_srt_cyrillic
    ):
        parser.error(
            "For mixed JSON + Latin-SRT actions, --path must be a directory containing both input types."
        )

    return args


def main() -> None:
    args = parse_args()
    if args is None:
        return

    base_path = args.path.resolve()

    json_actions_enabled = (
        args.create_srt
        or args.create_txt
        or args.create_txt_combined
        or args.create_social_srt_latin
        or args.create_social_srt_cyrillic
    )

    json_files: List[Path] = []
    payloads: Dict[Path, Dict] = {}

    if json_actions_enabled:
        json_files = collect_json_sources(base_path, args.json_glob)
        payloads = {path: json.loads(path.read_text(encoding="utf-8")) for path in json_files}

    if args.create_srt:
        srt_out_dir = ensure_dir(args.srt_out_dir.resolve(), "--srt-out-dir")
        total_cues = 0
        for path in json_files:
            payload = payloads[path]
            words = payload.get("words") or []
            tokens = build_standard_tokens(words)
            cues = tokens_to_standard_cues(tokens)

            out_srt = srt_out_dir / f"{path.stem}.srt"
            write_srt(cues, out_srt)
            total_cues += len(cues)
            print(f"Wrote {out_srt}")

            if args.create_sentence_srt:
                out_sentence_srt = sentence_srt_path(out_srt)
                write_sentence_srt(cues, out_sentence_srt)
                print(f"Wrote {out_sentence_srt}")

        print(f"Standard SRT complete ({total_cues} subtitle cues)")

    if args.create_txt:
        txt_out_dir = ensure_dir(args.txt_out_dir.resolve(), "--txt-out-dir")
        total_sentences = 0
        for path in json_files:
            payload = payloads[path]
            out_txt = txt_out_dir / f"{path.stem}.txt"

            sentences = payload_to_sentence_items(payload)
            if sentences:
                remap = build_speaker_remap(payload.get("words") or [])
                sentences = remap_sentence_items(sentences, remap)
                write_sentences_txt(sentences, out_txt, tag_all_speakers=True)
                total_sentences += len(sentences)
                print(f"Wrote {out_txt}")

        print(f"TXT complete ({total_sentences} sentences)")

    if args.create_txt_combined:
        txt_out_dir = ensure_dir(args.txt_out_dir.resolve(), "--txt-out-dir")
        if args.combined_txt_path is not None:
            combined_out = args.combined_txt_path.resolve()
            if combined_out.exists() and combined_out.is_dir():
                raise ValueError("--combined-txt-path must be a file path, not a directory.")
        else:
            combined_out = txt_out_dir / infer_combined_txt_name(json_files)

        combined_sentences: List[SentenceItem] = []
        for path in json_files:
            payload = payloads[path]
            sentences = payload_to_sentence_items(payload)
            remap = build_speaker_remap(payload.get("words") or [])
            sentences = remap_sentence_items(sentences, remap)
            combined_sentences.extend(sentences)

        write_sentences_txt(combined_sentences, combined_out, tag_all_speakers=True)
        print(f"Wrote {combined_out} ({len(combined_sentences)} sentences)")

    if args.create_social_srt_latin or args.create_social_srt_cyrillic:
        social_out_dir = ensure_dir(args.social_out_dir.resolve(), "--social-out-dir")
        total_social = 0
        for path in json_files:
            payload = payloads[path]
            tokens = build_social_word_tokens(payload)
            cues = tokens_to_social_cues(tokens)

            if args.create_social_srt_cyrillic:
                out_cyr = social_out_dir / f"{path.stem}_social_cyrillic.srt"
                out_cyr.write_text(cues_to_social_srt(cues, transform=to_cyrillic), encoding="utf-8")
                print(f"Wrote {out_cyr}")

            if args.create_social_srt_latin:
                out_lat = social_out_dir / f"{path.stem}_social_latin.srt"
                out_lat.write_text(cues_to_social_srt(cues, transform=to_latin), encoding="utf-8")
                print(f"Wrote {out_lat}")

            total_social += len(cues)

        print(f"Social SRT complete ({total_social} subtitle cues)")

    if args.convert_latin_srt_to_cyrillic:
        latin_sources = collect_latin_srt_sources(base_path, args.latin_srt_glob)
        latin_out_dir = ensure_dir(args.latin_cyr_out_dir.resolve(), "--latin-cyr-out-dir") if args.latin_cyr_out_dir else None

        for latin_path in latin_sources:
            target = cyrillic_output_path_for_latin(latin_path, out_dir=latin_out_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            source_text = latin_path.read_text(encoding="utf-8")
            target.write_text(latin_srt_to_cyrillic_text(source_text), encoding="utf-8")
            print(f"Wrote {target} from {latin_path}")


if __name__ == "__main__":
    main()
