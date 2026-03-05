import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from ..core.srt_builder import (
    build_social_word_tokens,
    build_standard_tokens,
    cues_to_social_srt,
    sentence_srt_path,
    tokens_to_social_cues,
    tokens_to_standard_cues,
    write_sentence_srt,
    write_srt,
)
from ..io_paths import SOCIAL_SRT_DIR, SRT_DIR, TXT_DIR
from ..selectors import collect_json_sources, collect_latin_srt_sources
from ..transcript_utils import (
    SentenceItem,
    build_speaker_remap,
    payload_to_sentence_items,
    remap_sentence_items,
    write_sentences_txt,
)
from ..translit import (
    cyrillic_output_path_for_latin,
    latin_srt_to_cyrillic_text,
    normalize_script_text,
    to_cyrillic,
    to_latin,
)
from ..uzbek_cleanup import clean_uzbek_payload, clean_uzbek_text


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def ensure_dir(path: Path, arg_name: str) -> Path:
    if path.exists() and not path.is_dir():
        raise ValueError(f"{arg_name} must be a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


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
            "  python scripts/transform.py --path media/JSON --create-srt --create-txt\n"
            "  python scripts/transform.py --path media/JSON --create-txt-combined\n"
            "  python scripts/transform.py --path media/JSON --create-clean-json --uzbek-clean\n"
            "  python scripts/transform.py --path media/JSON --create-social-srt-latin --create-social-srt-cyrillic\n"
            "  python scripts/transform.py --path media/SRT-social --convert-latin-srt-to-cyrillic\n"
            "  python scripts/transform.py --path \"media/JSON/^2025-06.*Shock[.]json$\" --create-srt"
        ),
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help=(
            "Input file, directory, or path expression. "
            "If the exact path does not exist but its parent directory exists, the last segment is treated as regex."
        ),
    )
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
        "--create-clean-json",
        action="store_true",
        help="Create cleaned Uzbek JSON copies without modifying original JSON files.",
    )
    parser.add_argument(
        "--uzbek-clean",
        action="store_true",
        help=(
            "Apply Uzbek readability cleanup to generated outputs "
            "(orthography, punctuation spacing, proper noun casing)."
        ),
    )
    parser.add_argument(
        "--sentence-gap-seconds",
        type=float,
        default=0.9,
        help="Pause threshold to split sentences when --uzbek-clean is used for TXT outputs.",
    )
    parser.add_argument(
        "--sentence-hard-gap-seconds",
        type=float,
        default=1.8,
        help="Hard pause threshold to always split sentences when --uzbek-clean is used for TXT outputs.",
    )
    parser.add_argument(
        "--script",
        choices=["latin", "cyrillic", "source"],
        default="latin",
        help=(
            "Script normalization for standard create outputs "
            "(--create-srt, --create-txt, --create-txt-combined)."
        ),
    )
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
        "--create-social-srt-raw",
        action="store_true",
        help="Create *_social_raw.srt without script normalization.",
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
        "--clean-json-out-dir",
        type=Path,
        default=None,
        help="Optional output directory for --create-clean-json. Default: next to each source JSON file.",
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
        args.create_clean_json,
        args.create_social_srt_latin,
        args.create_social_srt_cyrillic,
        args.create_social_srt_raw,
        args.convert_latin_srt_to_cyrillic,
    ]
    if not any(selected_actions):
        parser.error(
            "Select at least one action: --create-srt, --create-txt, --create-txt-combined, --create-clean-json, "
            "--create-social-srt-latin, --create-social-srt-cyrillic, --create-social-srt-raw, "
            "or --convert-latin-srt-to-cyrillic."
        )

    if args.path is None:
        parser.error("--path is required when selecting actions.")

    if args.create_sentence_srt:
        args.create_srt = True

    if args.path.is_file() and args.convert_latin_srt_to_cyrillic and (
        args.create_srt
        or args.create_txt
        or args.create_txt_combined
        or args.create_clean_json
        or args.create_social_srt_latin
        or args.create_social_srt_cyrillic
        or args.create_social_srt_raw
    ):
        parser.error(
            "For mixed JSON + Latin-SRT actions, --path must be a directory containing both input types."
        )

    if args.sentence_gap_seconds <= 0:
        parser.error("--sentence-gap-seconds must be > 0.")
    if args.sentence_hard_gap_seconds <= 0:
        parser.error("--sentence-hard-gap-seconds must be > 0.")
    if args.sentence_hard_gap_seconds < args.sentence_gap_seconds:
        parser.error("--sentence-hard-gap-seconds must be >= --sentence-gap-seconds.")

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
        or args.create_clean_json
        or args.create_social_srt_latin
        or args.create_social_srt_cyrillic
        or args.create_social_srt_raw
    )

    json_files: List[Path] = []
    payloads: Dict[Path, Dict] = {}
    cleaned_payloads: Dict[Path, Dict] = {}

    if json_actions_enabled:
        json_files = collect_json_sources(base_path, args.json_glob)
        payloads = {path: json.loads(path.read_text(encoding="utf-8")) for path in json_files}
        if args.uzbek_clean or args.create_clean_json:
            cleaned_payloads = {path: clean_uzbek_payload(payload) for path, payload in payloads.items()}

    if args.create_clean_json:
        clean_json_out_dir = ensure_dir(args.clean_json_out_dir.resolve(), "--clean-json-out-dir") if args.clean_json_out_dir else None
        for path in json_files:
            cleaned_payload = cleaned_payloads.get(path) or clean_uzbek_payload(payloads[path])
            target_dir = clean_json_out_dir or path.parent
            out_json = target_dir / f"{path.stem}_uz_clean.json"
            out_json.write_text(json.dumps(cleaned_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote {out_json}")

    def normalize_output_text(text: str) -> str:
        value = text
        if args.uzbek_clean:
            value = clean_uzbek_text(value)
        if args.script != "source":
            value = normalize_script_text(value, args.script)
        return value

    if args.create_srt:
        srt_out_dir = ensure_dir(args.srt_out_dir.resolve(), "--srt-out-dir")
        srt_transform = normalize_output_text if (args.uzbek_clean or args.script != "source") else None
        total_cues = 0
        for path in json_files:
            payload = cleaned_payloads[path] if args.uzbek_clean else payloads[path]
            words = payload.get("words") or []
            tokens = build_standard_tokens(words)
            cues = tokens_to_standard_cues(tokens)

            out_srt = srt_out_dir / f"{path.stem}.srt"
            write_srt(cues, out_srt, text_transform=srt_transform)
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
            payload = cleaned_payloads[path] if args.uzbek_clean else payloads[path]
            out_txt = txt_out_dir / f"{path.stem}.txt"

            sentences = payload_to_sentence_items(
                payload,
                use_timing_split=args.uzbek_clean,
                gap_split_seconds=args.sentence_gap_seconds,
                hard_gap_split_seconds=args.sentence_hard_gap_seconds,
            )
            if sentences:
                remap = build_speaker_remap(payload.get("words") or [])
                sentences = remap_sentence_items(sentences, remap)
                if args.uzbek_clean or args.script != "source":
                    sentences = [
                        {"text": normalize_output_text(item.get("text") or ""), "speaker": item.get("speaker")}
                        for item in sentences
                    ]
                write_sentences_txt(sentences, out_txt, main_speaker="", tag_all_speakers=False)
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
            payload = cleaned_payloads[path] if args.uzbek_clean else payloads[path]
            sentences = payload_to_sentence_items(
                payload,
                use_timing_split=args.uzbek_clean,
                gap_split_seconds=args.sentence_gap_seconds,
                hard_gap_split_seconds=args.sentence_hard_gap_seconds,
            )
            remap = build_speaker_remap(payload.get("words") or [])
            sentences = remap_sentence_items(sentences, remap)
            if args.uzbek_clean or args.script != "source":
                sentences = [
                    {"text": normalize_output_text(item.get("text") or ""), "speaker": item.get("speaker")}
                    for item in sentences
                ]
            combined_sentences.extend(sentences)

        write_sentences_txt(combined_sentences, combined_out, main_speaker="", tag_all_speakers=False)
        print(f"Wrote {combined_out} ({len(combined_sentences)} sentences)")

    if args.create_social_srt_latin or args.create_social_srt_cyrillic or args.create_social_srt_raw:
        social_out_dir = ensure_dir(args.social_out_dir.resolve(), "--social-out-dir")
        total_social = 0

        def compose_social_transform(script_fn):
            if not args.uzbek_clean and script_fn is None:
                return None

            def _transform(value: str) -> str:
                text = clean_uzbek_text(value) if args.uzbek_clean else value
                if script_fn is not None:
                    text = script_fn(text)
                return text

            return _transform

        for path in json_files:
            payload = cleaned_payloads[path] if args.uzbek_clean else payloads[path]
            tokens = build_social_word_tokens(payload)
            cues = tokens_to_social_cues(tokens)

            if args.create_social_srt_cyrillic:
                out_cyr = social_out_dir / f"{path.stem}_social_cyrillic.srt"
                out_cyr.write_text(cues_to_social_srt(cues, transform=compose_social_transform(to_cyrillic)), encoding="utf-8")
                print(f"Wrote {out_cyr}")

            if args.create_social_srt_latin:
                out_lat = social_out_dir / f"{path.stem}_social_latin.srt"
                out_lat.write_text(cues_to_social_srt(cues, transform=compose_social_transform(to_latin)), encoding="utf-8")
                print(f"Wrote {out_lat}")

            if args.create_social_srt_raw:
                out_raw = social_out_dir / f"{path.stem}_social_raw.srt"
                out_raw.write_text(cues_to_social_srt(cues, transform=compose_social_transform(None)), encoding="utf-8")
                print(f"Wrote {out_raw}")

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
