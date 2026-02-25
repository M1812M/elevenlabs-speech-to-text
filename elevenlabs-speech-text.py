import argparse
import base64
import json
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

from elevenlabs.client import ElevenLabs
from tqdm import tqdm

from elevenlabs_toolkit.paths import BASE_DIR, JSON_DIR, REC_DIR, SRT_DIR, TXT_DIR
from elevenlabs_toolkit.timecode import srt_timestamp
from elevenlabs_toolkit.transcript_utils import (
    build_speaker_remap,
    payload_to_sentence_items,
    remap_sentence_items,
    write_sentences_txt,
)


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


MODEL_ID = "scribe_v2"

SLEEP_BETWEEN_REQUESTS = 1.5  # seconds
OVERWRITE = False             # True: re-generate even if output exists
PER_FILE_PROGRESS = True      # Show per-file upload progress.

# Combine all transcripts into one TXT (one sentence per line).
COMBINE_TXT = False
COMBINED_TXT_NAME = "combined.txt"

# CLI defaults for STT API options.
DEFAULT_TIMESTAMPS_GRANULARITY = "word"
DEFAULT_ADDITIONAL_FORMATS = ["srt", "txt"]
DEFAULT_INCLUDE_SPEAKERS = True
DEFAULT_INCLUDE_TIMESTAMPS = True
SUPPORTED_ADDITIONAL_FORMATS = {"docx", "html", "pdf", "segmented_json", "srt", "txt"}
SUPPORTED_LANGUAGE_CODES_HELP = (
    "Examples from ElevenLabs docs: "
    "English=eng, German=deu, Russian=rus, Kyrgyz=kir, Uzbek=uzb"
)

# Extensions accepted in the input folder.
ALLOWED_EXTS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".mp4",
    ".wma",
    ".webm",
}

# MIME fallbacks for Windows.
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


def words_to_srt(words: List[Dict], max_chars: int = 84, max_dur: float = 5.5) -> str:
    # Convert word tokens into simple SRT cues.
    cues = []
    cur_text = ""
    cur_start = None
    cur_end = None

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
        en = float(word.get("end", st))

        if cur_start is None:
            cur_start = st

        candidate = f"{cur_text} {txt}" if cur_text else txt
        duration = en - cur_start
        if len(candidate) > max_chars or duration > max_dur:
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


def combine_dir_to_txt(out_dir: Path, ordered_stems: List[str], out_path: Path) -> int:
    # Combine all JSON transcripts into one TXT.
    all_sentences = []
    for stem in ordered_stems:
        json_path = out_dir / f"{stem}.json"
        if not json_path.exists():
            continue
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        sentences = payload_to_sentence_items(payload)
        remap = build_speaker_remap(payload.get("words") or [])
        sentences = remap_sentence_items(sentences, remap)
        all_sentences.extend(sentences)
    write_sentences_txt(all_sentences, out_path, tag_all_speakers=True)
    return len(all_sentences)


def normalize_additional_formats(raw_values: List[str]) -> List[str]:
    # Accept both spaced and comma-separated input, remove duplicates, keep order.
    normalized: List[str] = []
    seen = set()
    for raw in raw_values:
        for part in str(raw).split(","):
            fmt = part.strip().lower()
            if not fmt or fmt in seen:
                continue
            normalized.append(fmt)
            seen.add(fmt)
    return normalized


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch-transcribe audio files from media/REC with ElevenLabs.\n"
            "Outputs are written to media/JSON, media/SRT, and media/TXT."
        ),
        formatter_class=HelpFormatter,
        epilog=(
            "Examples:\n"
            "  python elevenlabs-speech-text.py\n"
            "  python elevenlabs-speech-text.py --timestamps-granularity character\n"
            "  python elevenlabs-speech-text.py --additional-formats srt txt segmented_json\n"
            "  python elevenlabs-speech-text.py --additional-formats --no-include-speakers\n"
            "  python elevenlabs-speech-text.py --language-code deu"
        ),
    )
    parser.add_argument(
        "--timestamps-granularity",
        choices=["none", "word", "character"],
        default=DEFAULT_TIMESTAMPS_GRANULARITY,
        help="Timestamp detail in ElevenLabs JSON output.",
    )
    parser.add_argument(
        "--additional-formats",
        nargs="*",
        default=list(DEFAULT_ADDITIONAL_FORMATS),
        metavar="FORMAT",
        help=(
            "Request server-generated additional formats. "
            "Use an empty list to disable extra API exports."
        ),
    )
    parser.add_argument(
        "--include-speakers",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_INCLUDE_SPEAKERS,
        help="Include speaker labels in requested additional formats.",
    )
    parser.add_argument(
        "--include-timestamps",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_INCLUDE_TIMESTAMPS,
        help="Include timestamps in requested additional formats.",
    )
    parser.add_argument(
        "--language-code",
        default=None,
        metavar="CODE",
        help=(
            "Optional ISO 639-3 language code for forced transcription language. "
            "If omitted, ElevenLabs auto-detects language. "
            f"{SUPPORTED_LANGUAGE_CODES_HELP}"
        ),
    )

    args = parser.parse_args()
    args.additional_formats = normalize_additional_formats(args.additional_formats)
    invalid_formats = [fmt for fmt in args.additional_formats if fmt not in SUPPORTED_ADDITIONAL_FORMATS]
    if invalid_formats:
        parser.error(
            f"Unsupported additional format(s): {invalid_formats}. "
            f"Supported: {sorted(SUPPORTED_ADDITIONAL_FORMATS)}"
        )
    return args


def build_additional_format_options(formats: List[str], include_speakers: bool, include_timestamps: bool):
    # Build API export options from CLI preferences.
    return [
        {
            "format": fmt,
            "include_speakers": include_speakers,
            "include_timestamps": include_timestamps,
        }
        for fmt in formats
    ]


def write_api_additional_formats(
    payload,
    stem: str,
    json_dir: Path,
    srt_dir: Path,
    txt_dir: Path,
):
    # Write API-returned additional formats to files and return written file map.
    written = {}
    for item in payload.get("additional_formats") or []:
        requested_format = str(item.get("requested_format") or "").strip().lower()
        file_extension = str(item.get("file_extension") or "").strip().lower().lstrip(".")
        ext = file_extension or requested_format
        content = item.get("content")
        if not ext or content is None:
            continue

        if requested_format == "srt" or ext == "srt":
            out_path = srt_dir / f"{stem}.srt"
        elif requested_format == "txt" or ext == "txt":
            out_path = txt_dir / f"{stem}.txt"
        elif requested_format == "segmented_json":
            out_path = json_dir / f"{stem}.segmented.json"
        elif ext == "json" and requested_format and requested_format != "json":
            out_path = json_dir / f"{stem}.{requested_format}.json"
        else:
            out_path = json_dir / f"{stem}.{ext}"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if item.get("is_base64_encoded"):
            out_path.write_bytes(base64.b64decode(content))
        else:
            out_path.write_text(str(content), encoding="utf-8")

        written[ext] = out_path
        if requested_format:
            written[requested_format] = out_path
    return written


def load_api_key() -> str:
    # Resolve API key from env or .env file.
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if api_key:
        return api_key

    # Try python-dotenv first.
    try:
        from dotenv import load_dotenv

        load_dotenv(str(BASE_DIR / ".env"))
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if api_key:
            print("Loaded ELEVENLABS_API_KEY from .env (python-dotenv)")
            return api_key
    except Exception:
        pass

    # Manual .env parse fallback.
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        with env_path.open("r", encoding="utf-8") as env_file:
            for line in env_file:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"'")
                if key == "ELEVENLABS_API_KEY" and value:
                    os.environ[key] = value
                    print("Loaded ELEVENLABS_API_KEY from .env (manual)")
                    return value

    raise RuntimeError("Missing ELEVENLABS_API_KEY. Set it as env var or add it to .env.")


def guess_mime(path: Path) -> str:
    # Guess MIME type with fallbacks for Windows.
    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        return mime
    return MIME_FALLBACK.get(path.suffix.lower(), "application/octet-stream")


def to_payload(result):
    # Normalize SDK response objects into plain dicts.
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return result


class TqdmFileWrapper:
    """Wrap a file object so reads update a tqdm progress bar."""

    def __init__(self, file_obj, pbar):
        self._file_obj = file_obj
        self._pbar = pbar

    def read(self, n: int = -1):
        data = self._file_obj.read(n)
        if data:
            self._pbar.update(len(data))
        return data

    def __getattr__(self, name):
        return getattr(self._file_obj, name)


def main() -> None:
    args = parse_args()
    api_key = load_api_key()

    if not REC_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {REC_DIR}")

    JSON_DIR.mkdir(parents=True, exist_ok=True)
    SRT_DIR.mkdir(parents=True, exist_ok=True)
    TXT_DIR.mkdir(parents=True, exist_ok=True)

    client = ElevenLabs(api_key=api_key)
    audio_files = sorted(path for path in REC_DIR.iterdir() if path.is_file() and path.suffix.lower() in ALLOWED_EXTS)

    if not audio_files:
        print(f"No audio files found in {REC_DIR} with extensions: {sorted(ALLOWED_EXTS)}")
        return

    print(f"Python: {sys.executable}")
    print(f"Found {len(audio_files)} audio files in: {REC_DIR}")
    print(
        "STT options: "
        f"language_code={args.language_code or 'auto'}, "
        f"timestamps_granularity={args.timestamps_granularity}, "
        f"additional_formats={args.additional_formats or []}, "
        f"include_speakers={args.include_speakers}, "
        f"include_timestamps={args.include_timestamps}"
    )

    additional_format_options = build_additional_format_options(
        args.additional_formats,
        args.include_speakers,
        args.include_timestamps,
    )

    for idx, audio_path in enumerate(tqdm(audio_files, desc="Transcribing files", unit="file"), start=1):
        stem = audio_path.stem
        out_json = JSON_DIR / f"{stem}.json"
        out_srt = SRT_DIR / f"{stem}.srt"
        out_txt = TXT_DIR / f"{stem}.txt"

        if not OVERWRITE and out_json.exists() and out_srt.exists() and out_txt.exists():
            tqdm.write(f"[{idx}/{len(audio_files)}] SKIP (exists): {audio_path.name}")
            continue

        mime = guess_mime(audio_path)
        tqdm.write(f"[{idx}/{len(audio_files)}] Transcribing: {audio_path.name} (mime={mime})")

        try:
            filesize = audio_path.stat().st_size
            convert_kwargs = {
                "model_id": MODEL_ID,
                "diarize": True,
                "tag_audio_events": True,
                "timestamps_granularity": args.timestamps_granularity,
            }
            if args.language_code:
                convert_kwargs["language_code"] = args.language_code
            if additional_format_options:
                convert_kwargs["additional_formats"] = additional_format_options

            if PER_FILE_PROGRESS:
                with audio_path.open("rb") as file_obj, tqdm(
                    total=filesize,
                    unit="B",
                    unit_scale=True,
                    desc=audio_path.name,
                    leave=False,
                ) as pbar:
                    wrapped = TqdmFileWrapper(file_obj, pbar)
                    result = client.speech_to_text.convert(file=wrapped, **convert_kwargs)
            else:
                with audio_path.open("rb") as file_obj:
                    result = client.speech_to_text.convert(file=file_obj, **convert_kwargs)

            payload = to_payload(result)
            out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            api_written_formats = write_api_additional_formats(
                payload,
                stem,
                JSON_DIR,
                SRT_DIR,
                TXT_DIR,
            )

            if "srt" not in api_written_formats:
                # Build local SRT fallback.
                words = payload.get("words") or [
                    {"type": "word", "start": segment["start"], "end": segment["end"], "text": segment["text"]}
                    for segment in payload.get("segments", [])
                ]
                out_srt.write_text(words_to_srt(words), encoding="utf-8")

            if "txt" not in api_written_formats:
                # Build local TXT fallback with speaker labels.
                sentence_items = payload_to_sentence_items(payload)
                if sentence_items:
                    remap = build_speaker_remap(payload.get("words") or [])
                    sentence_items = remap_sentence_items(sentence_items, remap)
                    write_sentences_txt(sentence_items, out_txt, tag_all_speakers=True)

            written_files = [out_json.name]
            if out_srt.exists():
                written_files.append(out_srt.name)
            if out_txt.exists():
                written_files.append(out_txt.name)
            for fmt, path in api_written_formats.items():
                if fmt in SUPPORTED_ADDITIONAL_FORMATS and path.name not in written_files:
                    written_files.append(path.name)
            tqdm.write(f"    OK -> {', '.join(written_files)}")

        except Exception as exc:
            tqdm.write(f"    ERROR on {audio_path.name}: {type(exc).__name__}: {exc}")

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    if COMBINE_TXT:
        ordered_stems = [path.stem for path in audio_files]
        out_txt = TXT_DIR / COMBINED_TXT_NAME
        count = combine_dir_to_txt(JSON_DIR, ordered_stems, out_txt)
        print(f"Wrote: {out_txt} ({count} sentences)")

    print("Done.")


if __name__ == "__main__":
    main()
