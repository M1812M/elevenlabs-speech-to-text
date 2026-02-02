import os
import sys
import json
import time
import mimetypes
from pathlib import Path
from elevenlabs.client import ElevenLabs
from tqdm import tqdm

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent
IN_DIR = BASE_DIR / "transcribe"
OUT_DIR = BASE_DIR / "transcript"

LANGUAGE_CODE = "uzb"
MODEL_ID = "scribe_v2"

SLEEP_BETWEEN_REQUESTS = 1.5  # seconds
OVERWRITE = False             # True: re-generate even if output exists
# Show per-file byte-level upload progress. Set to True to display a small progress bar during file upload.
PER_FILE_PROGRESS = True

# Extensions we will accept in the input folder
ALLOWED_EXTS = {
    ".wav", ".mp3", ".flac", ".m4a", ".aac",
    ".ogg", ".opus", ".mp4", ".wma", ".webm"
}

# Some MIME types may be missing on Windows; add common fallbacks
MIME_FALLBACK = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".mp4": "audio/mp4",
    ".wma": "audio/x-ms-wma",
    ".webm": "audio/webm",
}

# =========================
# SRT HELPERS
# =========================
def srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h = ms // 3_60_000
    ms %= 3_60_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def words_to_srt(words, max_chars=84, max_dur=5.5):
    cues = []
    cur_text = ""
    cur_start = None
    cur_end = None

    def flush():
        nonlocal cur_text, cur_start, cur_end
        if cur_text.strip() and cur_start is not None and cur_end is not None:
            cues.append((cur_start, cur_end, cur_text.strip()))
        cur_text = ""
        cur_start = None
        cur_end = None

    for w in words:
        if w.get("type") != "word":
            continue
        txt = w.get("text", "")
        if not txt:
            continue
        st = float(w.get("start", 0.0))
        en = float(w.get("end", st))

        if cur_start is None:
            cur_start = st

        candidate = f"{cur_text} {txt}" if cur_text else txt
        dur = en - cur_start

        if len(candidate) > max_chars or dur > max_dur:
            flush()
            cur_start = st
            candidate = txt

        cur_text = candidate
        cur_end = en

    flush()

    out_lines = []
    for i, (st, en, tx) in enumerate(cues, 1):
        out_lines.append(str(i))
        out_lines.append(f"{srt_timestamp(st)} --> {srt_timestamp(en)}")
        out_lines.append(tx)
        out_lines.append("")
    return "\n".join(out_lines)

# =========================
# ENV KEY LOADING (same approach you used)
# =========================
def load_api_key() -> str:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if api_key:
        return api_key

    # Try python-dotenv first
    try:
        from dotenv import load_dotenv
        load_dotenv(str(BASE_DIR / ".env"))
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if api_key:
            print("Loaded ELEVENLABS_API_KEY from .env (python-dotenv)")
            return api_key
    except Exception:
        pass

    # Manual .env parse fallback
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as ef:
            for line in ef:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("\"'")
                if k == "ELEVENLABS_API_KEY" and v:
                    os.environ[k] = v
                    print("Loaded ELEVENLABS_API_KEY from .env (manual)")
                    return v

    raise RuntimeError(
        "Missing ELEVENLABS_API_KEY. Set it as env var or add it to .env."
    )

# =========================
# CORE
# =========================
def guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        return mime
    return MIME_FALLBACK.get(path.suffix.lower(), "application/octet-stream")

def to_payload(result):
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return result


class TqdmFileWrapper:
    """Wrap a file object so reads update a tqdm progress bar."""
    def __init__(self, f, pbar):
        self._f = f
        self._pbar = pbar
    def read(self, n=-1):
        data = self._f.read(n)
        if data:
            self._pbar.update(len(data))
        return data
    def __getattr__(self, name):
        return getattr(self._f, name)

def main():
    api_key = load_api_key()

    if not IN_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {IN_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    client = ElevenLabs(api_key=api_key)

    # Gather files with allowed extensions
    audio_files = sorted([p for p in IN_DIR.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTS])

    if not audio_files:
        print(f"No audio files found in {IN_DIR} with extensions: {sorted(ALLOWED_EXTS)}")
        return

    print(f"Python: {sys.executable}")
    print(f"Found {len(audio_files)} audio files in: {IN_DIR}")

    for idx, audio_path in enumerate(tqdm(audio_files, desc="Transcribing files", unit="file"), start=1):
        stem = audio_path.stem
        out_json = OUT_DIR / f"{stem}.json"
        out_srt = OUT_DIR / f"{stem}.srt"

        if not OVERWRITE and out_json.exists() and out_srt.exists():
            tqdm.write(f"[{idx}/{len(audio_files)}] SKIP (exists): {audio_path.name}")
            continue

        mime = guess_mime(audio_path)
        tqdm.write(f"[{idx}/{len(audio_files)}] Transcribing: {audio_path.name} (mime={mime})")

        try:
            filesize = audio_path.stat().st_size
            if PER_FILE_PROGRESS:
                with open(audio_path, "rb") as f, tqdm(total=filesize, unit="B", unit_scale=True, desc=audio_path.name, leave=False) as fb:
                    wrapped = TqdmFileWrapper(f, fb)
                    result = client.speech_to_text.convert(
                        file=wrapped,
                        model_id=MODEL_ID,
                        diarize=True,
                        tag_audio_events=True,
                        language_code=LANGUAGE_CODE,
                    )
            else:
                with open(audio_path, "rb") as f:
                    result = client.speech_to_text.convert(
                        file=f,
                        model_id=MODEL_ID,
                        diarize=True,
                        tag_audio_events=True,
                        language_code=LANGUAGE_CODE,
                    )

            payload = to_payload(result)

            # Save JSON
            with open(out_json, "w", encoding="utf-8") as out:
                json.dump(payload, out, ensure_ascii=False, indent=2)

            # Build SRT (expects payload["words"])
            words = payload.get("words") or [
                {"type":"word","start":s["start"], "end":s["end"], "text": s["text"]}
                for s in payload.get("segments", [])
            ]
            srt_text = words_to_srt(words)

            with open(out_srt, "w", encoding="utf-8") as out:
                out.write(srt_text)

            tqdm.write(f"    OK → {out_json.name}, {out_srt.name}")

        except Exception as e:
            tqdm.write(f"    ERROR on {audio_path.name}: {type(e).__name__}: {e}")

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    print("Done.")

if __name__ == "__main__":
    main()
