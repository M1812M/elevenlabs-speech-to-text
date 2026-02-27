import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from elevenlabs.client import ElevenLabs
from tqdm import tqdm

from elevenlabs_toolkit.paths import BASE_DIR
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
DEFAULT_TIMESTAMPS_GRANULARITY = "word"
DEFAULT_INCLUDE_SPEAKERS = True
DEFAULT_INCLUDE_TIMESTAMPS = True
DEFAULT_SLEEP_SECONDS = 1.5
SUPPORTED_ADDITIONAL_FORMATS = {"docx", "html", "pdf", "segmented_json", "srt", "txt"}
SUPPORTED_LANGUAGE_CODES_HELP = (
    "Examples from ElevenLabs docs: "
    "English=eng, German=deu, Russian=rus, Kyrgyz=kir, Uzbek=uzb"
)

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

REGEX_META_RE = re.compile(r"[.^$*+?{}\[\]|()]")


def normalize_additional_formats(raw_values: List[str]) -> List[str]:
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


def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{prompt} {suffix}: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def interactive_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    print("No arguments provided. Starting interactive mode.")
    print("This tool uploads a file/folder to ElevenLabs and writes JSON outputs.")

    while True:
        input_path_raw = input("Input path (audio file or folder): ").strip().strip('"')
        if not input_path_raw:
            print("Input path is required.")
            continue
        input_path = Path(input_path_raw)
        if input_path.exists():
            break
        print(f"Path not found: {input_path}")

    while True:
        json_out_raw = input("JSON output folder: ").strip().strip('"')
        if not json_out_raw:
            print("Output folder is required.")
            continue
        json_out_dir = Path(json_out_raw)
        if json_out_dir.exists() and not json_out_dir.is_dir():
            print(f"Output path exists but is not a directory: {json_out_dir}")
            continue
        break

    create_srt = prompt_yes_no("Create local SRT files", default=True)
    create_txt = prompt_yes_no("Create local TXT files", default=True)
    language_code = input("Forced language code (empty = auto-detect): ").strip() or None

    return parser.parse_args(
        [
            "--path",
            str(input_path),
            "--json-out-dir",
            str(json_out_dir),
            *(["--create-srt"] if create_srt else []),
            *(["--create-txt"] if create_txt else []),
            *(["--language-code", language_code] if language_code else []),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe recordings with ElevenLabs.\n"
            "Input can be a single audio/video file or a directory with supported files."
        ),
        formatter_class=HelpFormatter,
        epilog=(
            "Examples:\n"
            "  python elevenlabs_transcribe.py --path media/REC --json-out-dir media/JSON --create-srt --create-txt\n"
            "  python elevenlabs_transcribe.py --path media/REC/sample.mp3 --json-out-dir media/JSON --language-code deu\n"
            "  python elevenlabs_transcribe.py --path media/REC --json-out-dir media/JSON --api-formats srt txt\n"
            "  python elevenlabs_transcribe.py --path \"media/REC/^PTT-.*[.]mp3$\" --json-out-dir media/JSON"
        ),
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help=(
            "Input path. Accepts one audio/video file, a directory, or a path expression. "
            "If the exact path does not exist but its parent directory exists, the last segment is treated as regex."
        ),
    )
    parser.add_argument(
        "--json-out-dir",
        type=Path,
        default=None,
        help="Directory where JSON transcripts are written.",
    )
    parser.add_argument(
        "--create-srt",
        action="store_true",
        help="Create local SRT files from transcript words.",
    )
    parser.add_argument(
        "--create-txt",
        action="store_true",
        help="Create local TXT files with sentence lines.",
    )
    parser.add_argument(
        "--srt-out-dir",
        type=Path,
        default=None,
        help="Output directory for local SRT files. Defaults to --json-out-dir.",
    )
    parser.add_argument(
        "--txt-out-dir",
        type=Path,
        default=None,
        help="Output directory for local TXT files. Defaults to --json-out-dir.",
    )
    parser.add_argument(
        "--timestamps-granularity",
        choices=["none", "word", "character"],
        default=DEFAULT_TIMESTAMPS_GRANULARITY,
        help="Timestamp detail in ElevenLabs JSON output.",
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
    parser.add_argument(
        "--api-formats",
        nargs="*",
        default=[],
        metavar="FORMAT",
        help="Optional ElevenLabs server-generated formats (srt, txt, segmented_json, pdf, html, docx).",
    )
    parser.add_argument(
        "--include-speakers",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_INCLUDE_SPEAKERS,
        help="Include speaker labels in API additional formats.",
    )
    parser.add_argument(
        "--include-timestamps",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_INCLUDE_TIMESTAMPS,
        help="Include timestamps in API additional formats.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Delay between requests to ElevenLabs API.",
    )

    if len(sys.argv) == 1:
        return interactive_args(parser)

    args = parser.parse_args()
    if args.path is None or args.json_out_dir is None:
        parser.error("--path and --json-out-dir are required unless you run with no arguments (interactive mode).")

    args.api_formats = normalize_additional_formats(args.api_formats)
    invalid_formats = [fmt for fmt in args.api_formats if fmt not in SUPPORTED_ADDITIONAL_FORMATS]
    if invalid_formats:
        parser.error(
            f"Unsupported --api-formats value(s): {invalid_formats}. "
            f"Supported: {sorted(SUPPORTED_ADDITIONAL_FORMATS)}"
        )
    return args


def words_to_srt(words: List[Dict], max_chars: int = 84, max_dur: float = 5.5) -> str:
    cues = []
    cur_text = ""
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None

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


def load_api_key() -> str:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if api_key:
        return api_key

    try:
        from dotenv import load_dotenv

        load_dotenv(str(BASE_DIR / ".env"))
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if api_key:
            print("Loaded ELEVENLABS_API_KEY from .env (python-dotenv)")
            return api_key
    except Exception:
        pass

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


def collect_regex_matches(path_expression: Path, allowed_exts: set[str], label: str) -> List[Path]:
    parent = path_expression.parent
    pattern = path_expression.name

    if not parent.exists() or not parent.is_dir():
        raise FileNotFoundError(f"{label} path not found: {path_expression}")
    if not REGEX_META_RE.search(pattern):
        raise FileNotFoundError(f"{label} path not found: {path_expression}")

    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"Invalid regex in --path expression '{pattern}': {exc}") from exc

    matches = sorted(
        p for p in parent.iterdir()
        if p.is_file() and p.suffix.lower() in allowed_exts and regex.search(p.name)
    )
    if not matches:
        raise FileNotFoundError(f"No {label} files matched regex '{pattern}' in {parent}")
    return matches


def collect_audio_files(path: Path) -> List[Path]:
    if not path.exists():
        return collect_regex_matches(path, ALLOWED_EXTS, "audio/video")

    if path.is_file():
        if path.suffix.lower() not in ALLOWED_EXTS:
            raise ValueError(
                f"Unsupported input file extension: {path.suffix}. Supported: {sorted(ALLOWED_EXTS)}"
            )
        return [path]

    if path.is_dir():
        audio_files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTS)
        if not audio_files:
            raise FileNotFoundError(
                f"No supported audio/video files found in {path}. Supported extensions: {sorted(ALLOWED_EXTS)}"
            )
        return audio_files

    raise ValueError(f"Input path is neither file nor directory: {path}")


def ensure_dir(path: Path, arg_name: str) -> Path:
    if path.exists() and not path.is_dir():
        raise ValueError(f"{arg_name} must be a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_additional_format_options(formats: List[str], include_speakers: bool, include_timestamps: bool):
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
    srt_dir: Optional[Path],
    txt_dir: Optional[Path],
):
    written = {}
    for item in payload.get("additional_formats") or []:
        requested_format = str(item.get("requested_format") or "").strip().lower()
        file_extension = str(item.get("file_extension") or "").strip().lower().lstrip(".")
        ext = file_extension or requested_format
        content = item.get("content")
        if not ext or content is None:
            continue

        if requested_format == "srt" or ext == "srt":
            out_dir = srt_dir or json_dir
            out_path = out_dir / f"{stem}.srt"
        elif requested_format == "txt" or ext == "txt":
            out_dir = txt_dir or json_dir
            out_path = out_dir / f"{stem}.txt"
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


class TqdmFileWrapper:
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

    input_path = args.path.resolve()
    json_out_dir = ensure_dir(args.json_out_dir.resolve(), "--json-out-dir")
    srt_out_dir = ensure_dir((args.srt_out_dir or json_out_dir).resolve(), "--srt-out-dir") if args.create_srt else None
    txt_out_dir = ensure_dir((args.txt_out_dir or json_out_dir).resolve(), "--txt-out-dir") if args.create_txt else None

    audio_files = collect_audio_files(input_path)
    additional_format_options = build_additional_format_options(
        args.api_formats,
        args.include_speakers,
        args.include_timestamps,
    )

    client = ElevenLabs(api_key=api_key)

    print(f"Python: {sys.executable}")
    print(f"Found {len(audio_files)} file(s) from: {input_path}")
    print(
        "Options: "
        f"language_code={args.language_code or 'auto'}, "
        f"timestamps_granularity={args.timestamps_granularity}, "
        f"create_srt={args.create_srt}, "
        f"create_txt={args.create_txt}, "
        f"api_formats={args.api_formats or []}"
    )

    for idx, audio_path in enumerate(tqdm(audio_files, desc="Transcribing", unit="file"), start=1):
        stem = audio_path.stem
        out_json = json_out_dir / f"{stem}.json"
        out_srt = (srt_out_dir / f"{stem}.srt") if srt_out_dir else None
        out_txt = (txt_out_dir / f"{stem}.txt") if txt_out_dir else None

        expected = [out_json]
        if out_srt is not None:
            expected.append(out_srt)
        if out_txt is not None:
            expected.append(out_txt)

        if (not args.overwrite) and expected and all(path.exists() for path in expected):
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

            with audio_path.open("rb") as file_obj, tqdm(
                total=filesize,
                unit="B",
                unit_scale=True,
                desc=audio_path.name,
                leave=False,
            ) as pbar:
                wrapped = TqdmFileWrapper(file_obj, pbar)
                result = client.speech_to_text.convert(file=wrapped, **convert_kwargs)

            payload = to_payload(result)
            out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            api_written_formats = write_api_additional_formats(
                payload,
                stem,
                json_out_dir,
                srt_out_dir,
                txt_out_dir,
            )

            if out_srt is not None and "srt" not in api_written_formats:
                words = payload.get("words") or [
                    {"type": "word", "start": segment["start"], "end": segment["end"], "text": segment["text"]}
                    for segment in payload.get("segments", [])
                ]
                out_srt.write_text(words_to_srt(words), encoding="utf-8")

            if out_txt is not None and "txt" not in api_written_formats:
                sentence_items = payload_to_sentence_items(payload)
                if sentence_items:
                    remap = build_speaker_remap(payload.get("words") or [])
                    sentence_items = remap_sentence_items(sentence_items, remap)
                    write_sentences_txt(sentence_items, out_txt, main_speaker="", tag_all_speakers=False)

            written_files = [out_json.name]
            if out_srt is not None and out_srt.exists():
                written_files.append(out_srt.name)
            if out_txt is not None and out_txt.exists():
                written_files.append(out_txt.name)
            for fmt, path in api_written_formats.items():
                if fmt in SUPPORTED_ADDITIONAL_FORMATS and path.name not in written_files:
                    written_files.append(path.name)
            tqdm.write(f"    OK -> {', '.join(written_files)}")

        except Exception as exc:
            tqdm.write(f"    ERROR on {audio_path.name}: {type(exc).__name__}: {exc}")

        time.sleep(max(args.sleep_seconds, 0.0))

    print("Done.")


if __name__ == "__main__":
    main()
