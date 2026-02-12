import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# =========================
# TUNING (Social Defaults)
# =========================
MAX_CHARS_PER_LINE = 30
MAX_LINES = 2
MAX_CHARS = MAX_CHARS_PER_LINE * MAX_LINES  # ~60
MAX_WORDS = 9
MAX_DURATION = 2.6          # seconds
MIN_DURATION = 0.9          # seconds
GAP_SPLIT = 0.75            # split if silence gap bigger than this (seconds)

# Hard boundaries
HARD_END_RE = re.compile(r"[.!?\u2026]+$")       # always end cue
# Soft boundaries (prefer ending here if cue is "long enough")
SOFT_END_RE = re.compile(r"[,;:]+$")

# Optional: only keep one speaker (set to None to keep all)
ONLY_SPEAKER_ID = None  # e.g. 0

BASE_DIR = Path(__file__).resolve().parent
MEDIA_DIR = BASE_DIR / "media"
JSON_DIR = MEDIA_DIR / "JSON"
SOCIAL_SRT_DIR = MEDIA_DIR / "SRT-social"

UZ_APOS = "\u2018"
APOS_RE = re.compile(r"[\u02bb\u02bc\u2018\u2019`´']")
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
    "a": "\u0430", "b": "\u0431", "c": "\u0446", "d": "\u0434", "e": "\u0435", "f": "\u0444", "g": "\u0433",
    "h": "\u04b3", "i": "\u0438", "j": "\u0436", "k": "\u043a", "l": "\u043b", "m": "\u043c", "n": "\u043d",
    "o": "\u043e", "p": "\u043f", "q": "\u049b", "r": "\u0440", "s": "\u0441", "t": "\u0442", "u": "\u0443",
    "v": "\u0432", "w": "\u0432", "x": "\u0445", "y": "\u0439", "z": "\u0437",
}

CYRILLIC_TO_LATIN_CHARS = {
    "\u0410": "A", "\u0430": "a", "\u0411": "B", "\u0431": "b", "\u0412": "V", "\u0432": "v",
    "\u0413": "G", "\u0433": "g", "\u0492": f"G{UZ_APOS}", "\u0493": f"g{UZ_APOS}", "\u0414": "D", "\u0434": "d",
    "\u0401": "Yo", "\u0451": "yo", "\u0416": "J", "\u0436": "j", "\u0417": "Z", "\u0437": "z",
    "\u0418": "I", "\u0438": "i", "\u0419": "Y", "\u0439": "y", "\u041a": "K", "\u043a": "k",
    "\u049a": "Q", "\u049b": "q", "\u041b": "L", "\u043b": "l", "\u041c": "M", "\u043c": "m",
    "\u041d": "N", "\u043d": "n", "\u041e": "O", "\u043e": "o", "\u041f": "P", "\u043f": "p",
    "\u0420": "R", "\u0440": "r", "\u0421": "S", "\u0441": "s", "\u0422": "T", "\u0442": "t",
    "\u0423": "U", "\u0443": "u", "\u0424": "F", "\u0444": "f", "\u0425": "X", "\u0445": "x",
    "\u04b2": "H", "\u04b3": "h", "\u0426": "S", "\u0446": "s", "\u0427": "Ch", "\u0447": "ch",
    "\u0428": "Sh", "\u0448": "sh", "\u0429": "Sh", "\u0449": "sh", "\u042a": "'", "\u044a": "'",
    "\u042c": "'", "\u044c": "'", "\u042d": "E", "\u044d": "e", "\u042e": "Yu", "\u044e": "yu",
    "\u042f": "Ya", "\u044f": "ya", "\u040e": f"O{UZ_APOS}", "\u045e": f"o{UZ_APOS}",
}


def srt_timestamp(t: float) -> str:
    ms = int(round(max(t, 0.0) * 1000))
    hh = ms // 3_600_000
    ms %= 3_600_000
    mm = ms // 60_000
    ms %= 60_000
    ss = ms // 1000
    ms %= 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?\u2026])", r"\1", text)  # no space before punctuation
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
    # Transliterate Uzbek Latin text to Cyrillic (non-matching chars pass through).
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

        # Match uzlatin.com: word-initial 'e' maps to 'э', otherwise 'е'.
        if lower == "e":
            prev = value[idx - 1] if idx > 0 else None
            if prev is None or not prev.isalpha():
                mapped = "\u044d"

        out_chars.append(mapped.upper() if ch.isupper() else mapped)

    return "".join(out_chars)


def to_latin(text: str) -> str:
    # Transliterate Uzbek Cyrillic text to Latin (non-matching chars pass through).
    out_parts: List[str] = []
    for idx, ch in enumerate(text):
        # Match uzlatin.com: 'Е/е' -> 'Ye/ye' at word start or after vowels/signs.
        if ch in ("\u0415", "\u0435"):
            prev = text[idx - 1] if idx > 0 else None
            if not is_cyrillic_letter(prev) or (prev in CYRILLIC_VOWELS_AND_SIGNS):
                out_parts.append("Ye" if ch == "\u0415" else "ye")
            else:
                out_parts.append("E" if ch == "\u0415" else "e")
            continue

        out_parts.append(CYRILLIC_TO_LATIN_CHARS.get(ch, ch))

    return "".join(out_parts)


def build_word_tokens(payload: Dict) -> List[Dict]:
    out = []
    for w in payload.get("words", []):
        if w.get("type") != "word":
            continue
        if ONLY_SPEAKER_ID is not None and w.get("speaker_id") != ONLY_SPEAKER_ID:
            continue
        txt = w.get("text", "")
        if not txt:
            continue
        out.append({
            "text": txt,
            "start": float(w.get("start", 0.0)),
            "end": float(w.get("end", 0.0)),
        })
    return out


def tokens_to_cues(tokens: List[Dict]) -> List[Tuple[float, float, str]]:
    cues: List[Tuple[float, float, str]] = []

    cur_start: Optional[float] = None
    cur_end: Optional[float] = None
    cur_parts: List[str] = []
    cur_words = 0
    last_end: Optional[float] = None

    def flush() -> None:
        nonlocal cur_start, cur_end, cur_parts, cur_words
        if cur_start is not None and cur_end is not None and cur_parts:
            text = normalize(" ".join(cur_parts))
            if text:
                cues.append((cur_start, cur_end, text))
        cur_start = None
        cur_end = None
        cur_parts = []
        cur_words = 0

    for tok in tokens:
        txt = tok["text"]
        st = tok["start"]
        en = tok["end"]

        if cur_start is None:
            cur_start = st

        # hard split on silence gap
        if last_end is not None and (st - last_end) > GAP_SPLIT and cur_parts:
            flush()
            cur_start = st

        # add token tentatively
        tentative_parts = cur_parts + [txt]
        tentative_text = normalize(" ".join(tentative_parts))
        tentative_words = cur_words + 1
        tentative_dur = en - cur_start

        # 1) HARD stop at sentence end punctuation (after adding)
        if HARD_END_RE.search(tentative_text):
            cur_parts = tentative_parts
            cur_words = tentative_words
            cur_end = en
            flush()
            last_end = en
            continue

        # 2) Enforce hard limits: chars/words/duration
        hard_limit_hit = (
            len(tentative_text) > MAX_CHARS or
            tentative_words > MAX_WORDS or
            tentative_dur > MAX_DURATION
        )
        if hard_limit_hit and cur_parts:
            # flush current cue BEFORE adding this token
            flush()
            cur_start = st
            cur_parts = [txt]
            cur_words = 1
            cur_end = en
            last_end = en
            continue

        # 3) SOFT split: prefer comma/semicolon/colon if cue is long enough
        if SOFT_END_RE.search(tentative_text):
            if (len(tentative_text) >= 20) or (tentative_words >= 5) or (tentative_dur >= MIN_DURATION):
                cur_parts = tentative_parts
                cur_words = tentative_words
                cur_end = en
                flush()
                last_end = en
                continue

        # 4) Otherwise just keep building
        cur_parts = tentative_parts
        cur_words = tentative_words
        cur_end = en
        last_end = en

    flush()

    # Merge too-short cues forward when possible (keeps readability)
    merged: List[Tuple[float, float, str]] = []
    for st, en, tx in cues:
        if merged:
            pst, pen, ptx = merged[-1]
            if (en - st) < MIN_DURATION:
                candidate = normalize(ptx + " " + tx)
                if len(candidate) <= MAX_CHARS and (en - pst) <= MAX_DURATION:
                    merged[-1] = (pst, en, candidate)
                    continue
        merged.append((st, en, tx))

    return merged


def cues_to_srt(cues: List[Tuple[float, float, str]], transform: Optional[Callable[[str], str]] = None) -> str:
    out = []
    for i, (st, en, tx) in enumerate(cues, 1):
        single_line = normalize(tx)
        if transform is not None:
            single_line = normalize(transform(single_line))
        out.append(str(i))
        out.append(f"{srt_timestamp(st)} --> {srt_timestamp(en)}")
        out.append(single_line)
        out.append("")
    return "\n".join(out)


def convert_one(json_path: Path, cyrillic_srt_path: Path, latin_srt_path: Path) -> int:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    tokens = build_word_tokens(payload)
    cues = tokens_to_cues(tokens)
    cyrillic_srt_path.write_text(cues_to_srt(cues, transform=to_cyrillic), encoding="utf-8")
    latin_srt_path.write_text(cues_to_srt(cues, transform=to_latin), encoding="utf-8")
    return len(cues)


def main() -> None:
    # Convert all JSON files in media/JSON to social-style SRT in media/SRT-social.
    json_files = sorted(JSON_DIR.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {JSON_DIR}")
        return

    SOCIAL_SRT_DIR.mkdir(parents=True, exist_ok=True)

    for in_json in json_files:
        out_cyr = SOCIAL_SRT_DIR / f"{in_json.stem}_social_cyrillic.srt"
        out_lat = SOCIAL_SRT_DIR / f"{in_json.stem}_social_latin.srt"
        n = convert_one(in_json, out_cyr, out_lat)
        print(f"Wrote {out_cyr.name} and {out_lat.name} from {in_json.name} ({n} cues)")


if __name__ == "__main__":
    main()
