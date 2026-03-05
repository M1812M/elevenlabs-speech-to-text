import re
from pathlib import Path
from typing import List, Optional


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
    "\u042b": "I",
    "\u044b": "i",
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


def apply_case(sample: str, replacement: str) -> str:
    if sample.isupper():
        return replacement.upper()
    if sample.islower():
        return replacement.lower()
    if len(sample) > 1 and sample[0].isupper() and sample[1:].islower():
        return replacement[0].upper() + replacement[1:].lower()
    return replacement


def _is_cyrillic_letter(ch: Optional[str]) -> bool:
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
            if not _is_cyrillic_letter(prev) or (prev in CYRILLIC_VOWELS_AND_SIGNS):
                out_parts.append("Ye" if ch == "\u0415" else "ye")
            else:
                out_parts.append("E" if ch == "\u0415" else "e")
            continue

        out_parts.append(CYRILLIC_TO_LATIN_CHARS.get(ch, ch))

    return "".join(out_parts)


def normalize_script_text(text: str, script_mode: str) -> str:
    if script_mode == "latin":
        return to_latin(text)
    if script_mode == "cyrillic":
        return to_cyrillic(text)
    return text


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
