import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ..pause_detection import detected_pause_gap, effective_word_end
from ..timecode import srt_timestamp
from ..transcript_utils import get_speaker_id, normalize_spaces


MAX_CHARS_PER_LINE = 42
MAX_LINES = 2
MAX_CHARS = MAX_CHARS_PER_LINE * MAX_LINES
MAX_DURATION = 5.5
MIN_DURATION = 1.0
GAP_SPLIT = 0.9
DETECTED_PAUSE_GAP_SPLIT = 0.6
PUNCT_END_RE = re.compile(r"[.!?\u2026]+$")

SOCIAL_MAX_CHARS_PER_LINE = 30
SOCIAL_MAX_LINES = 2
SOCIAL_MAX_CHARS = SOCIAL_MAX_CHARS_PER_LINE * SOCIAL_MAX_LINES
SOCIAL_MAX_WORDS = 9
SOCIAL_MAX_DURATION = 2.6
SOCIAL_MIN_DURATION = 0.9
SOCIAL_GAP_SPLIT = 0.75
SOCIAL_DURATION_EPSILON = 1e-6
SOCIAL_HARD_END_RE = re.compile(r"[.!?\u2026]+$")
SOCIAL_SOFT_END_RE = re.compile(r"[,;:]+$")


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


def build_standard_tokens(words: List[Dict], pause_detection: bool = False) -> List[Dict]:
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
                "end": effective_word_end(word, pause_detection=pause_detection),
                "pause_after": detected_pause_gap(word, pause_detection=pause_detection) >= DETECTED_PAUSE_GAP_SPLIT,
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
    last_pause_after = False

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

        gap_after_last = (st - last_end) if last_end is not None else 0.0
        if last_end is not None and (gap_after_last > GAP_SPLIT or (last_pause_after and gap_after_last >= DETECTED_PAUSE_GAP_SPLIT)):
            flush()
            cur_start = st
            cur_end = en
            cur_text_parts = [txt]
            last_end = en
            last_pause_after = bool(token.get("pause_after"))
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
        last_pause_after = bool(token.get("pause_after"))

    flush()

    merged: List[Tuple[float, float, str]] = []
    for st, en, tx in cues:
        if merged:
            prev_start, _prev_end, prev_text = merged[-1]
            if (en - st) < MIN_DURATION:
                candidate = normalize_spaces(prev_text + " " + tx)
                if len(candidate) <= MAX_CHARS and (en - prev_start) <= MAX_DURATION:
                    merged[-1] = (prev_start, en, candidate)
                    continue
        merged.append((st, en, tx))

    # Prevent one-word orphan cues by borrowing one trailing word from the previous cue.
    rebalanced: List[Tuple[float, float, str]] = []
    for st, en, tx in merged:
        if rebalanced and len(tx.split()) == 1:
            prev_start, prev_end, prev_text = rebalanced[-1]
            prev_words = prev_text.split()
            if len(prev_words) >= 3:
                moved = prev_words.pop()
                new_prev = normalize_spaces(" ".join(prev_words))
                new_cur = normalize_spaces(f"{moved} {tx}")
                if new_prev and len(new_prev) <= MAX_CHARS and len(new_cur) <= MAX_CHARS:
                    rebalanced[-1] = (prev_start, prev_end, new_prev)
                    rebalanced.append((st, en, new_cur))
                    continue
        rebalanced.append((st, en, tx))

    return rebalanced


def write_srt(
    cues: List[Tuple[float, float, str]],
    out_path: Path,
    text_transform: Optional[Callable[[str], str]] = None,
) -> None:
    lines: List[str] = []
    for i, (st, en, text) in enumerate(cues, start=1):
        if text_transform is not None:
            text = normalize_spaces(text_transform(text))
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


def build_social_word_tokens(payload: Dict, pause_detection: bool = False) -> List[Dict]:
    out = []
    for word in payload.get("words", []):
        if word.get("type") != "word":
            continue
        txt = word.get("text", "")
        if not txt:
            continue
        out.append(
            {
                "text": txt,
                "start": float(word.get("start", 0.0)),
                "end": effective_word_end(word, pause_detection=pause_detection),
                "pause_after": detected_pause_gap(word, pause_detection=pause_detection) >= DETECTED_PAUSE_GAP_SPLIT,
            }
        )
    return out


def tokens_to_social_cues(tokens: List[Dict]) -> List[Tuple[float, float, str]]:
    cues: List[Tuple[float, float, str]] = []
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None
    cur_parts: List[str] = []
    cur_words = 0
    last_end: Optional[float] = None
    last_pause_after = False

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

        gap_after_last = (st - last_end) if last_end is not None else 0.0
        if last_end is not None and (
            gap_after_last > SOCIAL_GAP_SPLIT
            or (last_pause_after and gap_after_last >= DETECTED_PAUSE_GAP_SPLIT)
        ) and cur_parts:
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
            or tentative_dur > (SOCIAL_MAX_DURATION + SOCIAL_DURATION_EPSILON)
        )
        if hard_limit_hit and cur_parts:
            flush()
            cur_start = st
            cur_parts = [txt]
            cur_words = 1
            cur_end = en
            last_end = en
            last_pause_after = bool(token.get("pause_after"))
            continue

        if SOCIAL_SOFT_END_RE.search(tentative_text):
            if (len(tentative_text) >= 20) or (tentative_words >= 5) or (tentative_dur >= SOCIAL_MIN_DURATION):
                cur_parts = tentative_parts
                cur_words = tentative_words
                cur_end = en
                flush()
                last_end = en
                last_pause_after = bool(token.get("pause_after"))
                continue

        cur_parts = tentative_parts
        cur_words = tentative_words
        cur_end = en
        last_end = en
        last_pause_after = bool(token.get("pause_after"))

    flush()

    merged: List[Tuple[float, float, str]] = []
    for st, en, tx in cues:
        if merged:
            prev_start, _prev_end, prev_text = merged[-1]
            if (en - st) < SOCIAL_MIN_DURATION:
                candidate = social_normalize(prev_text + " " + tx)
                if len(candidate) <= SOCIAL_MAX_CHARS and (en - prev_start) <= (SOCIAL_MAX_DURATION + SOCIAL_DURATION_EPSILON):
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


def words_to_basic_srt(
    words: List[Dict],
    max_chars: int = MAX_CHARS,
    max_dur: float = MAX_DURATION,
    pause_detection: bool = False,
) -> str:
    cues = []
    cur_text = ""
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None
    last_end: Optional[float] = None
    last_pause_after = False

    def flush() -> None:
        nonlocal cur_text, cur_start, cur_end
        if cur_text.strip() and cur_start is not None and cur_end is not None:
            cues.append((cur_start, cur_end, cur_text.strip()))
        cur_text = ""
        cur_start = None
        cur_end = None

    for word in words or []:
        if word.get("type") != "word":
            continue
        txt = word.get("text", "")
        if not txt:
            continue
        st = float(word.get("start", 0.0))
        en = effective_word_end(word, pause_detection=pause_detection)

        if cur_start is None:
            cur_start = st

        gap_after_last = (st - last_end) if last_end is not None else 0.0
        if last_end is not None and (gap_after_last > GAP_SPLIT or (last_pause_after and gap_after_last >= DETECTED_PAUSE_GAP_SPLIT)):
            flush()
            cur_start = st

        candidate = f"{cur_text} {txt}" if cur_text else txt
        duration = en - cur_start
        if len(candidate) > max_chars or duration > max_dur:
            flush()
            cur_start = st
            candidate = txt

        cur_text = candidate
        cur_end = en
        last_end = en
        last_pause_after = detected_pause_gap(word, pause_detection=pause_detection) >= DETECTED_PAUSE_GAP_SPLIT

    flush()

    out_lines = []
    for i, (st, en, tx) in enumerate(cues, 1):
        out_lines.append(str(i))
        out_lines.append(f"{srt_timestamp(st)} --> {srt_timestamp(en)}")
        out_lines.append(tx)
        out_lines.append("")
    return "\n".join(out_lines)
