from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..models import TranscriptionOptions

CACHE_SCHEMA_VERSION = 2
CACHE_PROVIDER = "elevenlabs"


def source_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(content: str | bytes) -> str:
    data = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(data).hexdigest()


def source_fingerprint(path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = source_sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise OSError(f"source changed while it was being fingerprinted: {path}")
    return {"name": path.name, "size": after.st_size, "sha256": digest}


def transcription_options_data(options: TranscriptionOptions) -> dict[str, Any]:
    """Return the canonical provider-request fingerprint used for cache keys."""

    data = asdict(options)
    if not options.diarize:
        data["num_speakers"] = None
    data["remote_formats"] = sorted({str(item).strip().lower().replace("-", "_") for item in options.remote_formats})
    return json.loads(json.dumps(data, ensure_ascii=False))


def build_manifest(
    source: Path,
    options: TranscriptionOptions,
    *,
    transcript_name: str,
    transcript_content: str | bytes,
    source_data: dict[str, Any] | None = None,
    provider: str = CACHE_PROVIDER,
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "provider": provider,
        "source": source_fingerprint(source) if source_data is None else dict(source_data),
        "transcript": {
            "name": transcript_name,
            "sha256": content_sha256(transcript_content),
        },
        "transcription": transcription_options_data(options),
    }


def cache_matches(
    source: Path,
    transcript_path: Path,
    manifest_path: Path,
    options: TranscriptionOptions,
    *,
    provider: str = CACHE_PROVIDER,
) -> bool:
    if not transcript_path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_data = manifest["source"]
        transcript_data = manifest["transcript"]
        if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
            return False
        if manifest.get("provider") != provider:
            return False
        if source_data != source_fingerprint(source):
            return False
        if manifest.get("transcription") != transcription_options_data(options):
            return False
        if transcript_data.get("name") != transcript_path.name:
            return False
        return transcript_data.get("sha256") == content_sha256(transcript_path.read_bytes())
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
