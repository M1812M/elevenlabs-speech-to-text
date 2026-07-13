from __future__ import annotations

import re

UZ_APOS = "\u2018"
APOS_RE = re.compile(r"[\u02bb\u02bc\u2018\u2019`\u00b4']")
CYRILLIC_LETTER_RE = re.compile(r"[\u0400-\u04FF]")
CYRILLIC_VOWELS_AND_SIGNS = set(
    "\u0410\u0430\u0415\u0435\u0401\u0451\u0418\u0438\u041e\u043e\u0423\u0443\u040e\u045e\u042d\u044d\u042e\u044e\u042f\u044f\u042a\u044a\u042c\u044c"
)
LATIN_DIGRAPHS = (
    (re.compile(r"o'", re.I), "\u045e"),
    (re.compile(r"g'", re.I), "\u0493"),
    (re.compile(r"sh", re.I), "\u0448"),
    (re.compile(r"ch", re.I), "\u0447"),
    (re.compile(r"ng", re.I), "\u043d\u0433"),
    (re.compile(r"yo", re.I), "\u0451"),
    (re.compile(r"yu", re.I), "\u044e"),
    (re.compile(r"ya", re.I), "\u044f"),
    (re.compile(r"ye", re.I), "\u0435"),
)
LATIN_TO_CYRILLIC = dict(
    zip(
        "abcdefghijklmnopqrstuvwyxz",
        "\u0430\u0431\u0446\u0434\u0435\u0444\u0433\u04b3\u0438\u0436\u043a\u043b\u043c\u043d\u043e\u043f\u049b\u0440\u0441\u0442\u0443\u0432\u0432\u0439\u0445\u0437",
        strict=True,
    )
)
CYRILLIC_TO_LATIN = {
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


def _apply_case(sample: str, replacement: str) -> str:
    if sample.isupper():
        return replacement.upper()
    if sample.islower():
        return replacement.lower()
    if len(sample) > 1 and sample[0].isupper() and sample[1:].islower():
        return replacement[0].upper() + replacement[1:].lower()
    return replacement


def to_cyrillic(text: str) -> str:
    value = APOS_RE.sub("'", text)
    for pattern, replacement in LATIN_DIGRAPHS:

        def replace_digraph(match: re.Match[str], digraph: str = replacement) -> str:
            return _apply_case(match.group(0), digraph)

        value = pattern.sub(replace_digraph, value)
    output: list[str] = []
    for index, character in enumerate(value):
        lower = character.lower()
        mapped = LATIN_TO_CYRILLIC.get(lower)
        if mapped is None:
            output.append("\u044a" if character == "'" else character)
            continue
        if lower == "e":
            previous = value[index - 1] if index else None
            if previous is None or not previous.isalpha():
                mapped = "\u044d"
        output.append(mapped.upper() if character.isupper() else mapped)
    return "".join(output)


def to_latin(text: str) -> str:
    output: list[str] = []
    for index, character in enumerate(text):
        if character in ("\u0415", "\u0435"):
            previous = text[index - 1] if index else None
            previous_is_cyrillic = bool(previous and CYRILLIC_LETTER_RE.match(previous))
            if not previous_is_cyrillic or previous in CYRILLIC_VOWELS_AND_SIGNS:
                output.append("Ye" if character == "\u0415" else "ye")
            else:
                output.append("E" if character == "\u0415" else "e")
            continue
        output.append(CYRILLIC_TO_LATIN.get(character, character))
    return "".join(output)
