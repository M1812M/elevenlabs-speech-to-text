import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from elevenlabs.client import ElevenLabs
from tqdm import tqdm

from ..core.srt_builder import words_to_basic_srt
from ..core.stt_client import (
    DEFAULT_INCLUDE_SPEAKERS,
    DEFAULT_INCLUDE_TIMESTAMPS,
    DEFAULT_SLEEP_SECONDS,
    DEFAULT_TIMESTAMPS_GRANULARITY,
    MODEL_ID,
    SUPPORTED_ADDITIONAL_FORMATS,
    SUPPORTED_LANGUAGE_CODES_HELP,
    TqdmFileWrapper,
    build_additional_format_options,
    guess_mime,
    load_api_key,
    normalize_additional_formats,
    to_payload,
    write_api_additional_formats,
)
from ..io_paths import JSON_DIR
from ..selectors import collect_audio_files
from ..transcript_utils import (
    build_speaker_remap,
    payload_to_sentence_items,
    remap_sentence_items,
    write_sentences_txt,
)


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


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

    json_default = JSON_DIR.resolve()
    while True:
        json_out_raw = input(f"JSON output folder [{json_default}]: ").strip().strip('"')
        json_out_dir = Path(json_out_raw) if json_out_raw else json_default
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
            "  python scripts/transcribe.py --path media/REC --create-srt --create-txt\n"
            "  python scripts/transcribe.py --path media/REC/sample.mp3 --language-code deu\n"
            "  python scripts/transcribe.py --path media/REC --api-formats srt txt\n"
            "  python scripts/transcribe.py --path \"media/REC/^PTT-.*[.]mp3$\" --json-out-dir media/JSON"
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
        default=JSON_DIR,
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
    if args.path is None:
        parser.error("--path is required unless you run with no arguments (interactive mode).")

    args.api_formats = normalize_additional_formats(args.api_formats)
    invalid_formats = [fmt for fmt in args.api_formats if fmt not in SUPPORTED_ADDITIONAL_FORMATS]
    if invalid_formats:
        parser.error(
            f"Unsupported --api-formats value(s): {invalid_formats}. "
            f"Supported: {sorted(SUPPORTED_ADDITIONAL_FORMATS)}"
        )
    return args


def ensure_dir(path: Path, arg_name: str) -> Path:
    if path.exists() and not path.is_dir():
        raise ValueError(f"{arg_name} must be a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


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
                out_srt.write_text(words_to_basic_srt(words), encoding="utf-8")

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
