import base64
import mimetypes
import os
from pathlib import Path
from typing import Dict, List, Optional

from ..io_paths import BASE_DIR


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
) -> Dict[str, Path]:
    written: Dict[str, Path] = {}
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
