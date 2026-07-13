"""ElevenLabs speech-to-text adapter.

The SDK is deliberately imported only when a real client is first needed.
This keeps local exports, CLI help, and tests usable without the optional SDK.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import os
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from ..models import TranscriptionOptions
from .protocol import (
    ProviderCredentialError,
    ProviderDependencyError,
    ProviderError,
    ProviderResponseError,
    ProviderTransientError,
)

_API_KEY_NAME = "ELEVENLABS_API_KEY"
_REMOTE_FORMATS = frozenset({"docx", "html", "pdf", "segmented_json"})
_DECODED_FORMATS = frozenset({"docx", "html", "pdf", "segmented_json"})
_FORMAT_EXTENSIONS = {
    "docx": frozenset({"docx"}),
    "html": frozenset({"html", "htm"}),
    "pdf": frozenset({"pdf"}),
    "segmented_json": frozenset({"json"}),
    "srt": frozenset({"srt"}),
    "txt": frozenset({"txt"}),
}
_SAFE_EXTENSION = re.compile(r"^[a-z0-9_]+$")
_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_WINDOWS_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{number}" for number in range(1, 10)}
    | {f"LPT{number}" for number in range(1, 10)}
)


class ElevenLabsProvider:
    """Lazy ElevenLabs SDK adapter.

    ``client`` is an intentional injection point for applications and tests.
    When supplied, neither the SDK nor credentials are required.
    """

    cache_key = "elevenlabs"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        env_file: str | Path | None = None,
        client: Any | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._api_key = api_key
        self._env_file = Path(env_file).expanduser() if env_file is not None else None
        self._client = client
        self._environ = os.environ if environ is None else environ

    @property
    def client(self) -> Any:
        """Return the client, importing and constructing it on first access."""

        if self._client is None:
            self._client = self._create_client()
        return self._client

    def transcribe(
        self,
        path: str | Path,
        options: TranscriptionOptions,
    ) -> dict[str, Any]:
        source = Path(path).expanduser()
        kwargs = build_request_kwargs(options)

        try:
            stream = source.open("rb")
        except OSError as exc:
            raise ProviderError(f"Cannot open transcription input {source}: {exc}") from exc

        try:
            with stream:
                client = self.client
                result = client.speech_to_text.convert(file=stream, **kwargs)
        except ProviderError:
            raise
        except Exception as exc:  # SDK exception types must not leak into this boundary.
            raise _classified_error(exc) from exc
        return normalize_response(result)

    def _create_client(self) -> Any:
        try:
            from elevenlabs.client import ElevenLabs
        except (ImportError, ModuleNotFoundError) as exc:
            raise ProviderDependencyError(
                "ElevenLabs transcription requires the optional 'elevenlabs' package"
            ) from exc

        api_key = resolve_api_key(
            self._api_key,
            env_file=self._env_file,
            environ=self._environ,
        )
        try:
            return ElevenLabs(api_key=api_key)
        except Exception as exc:
            classified = _classified_error(exc)
            if type(classified) is ProviderError:
                classified = ProviderDependencyError(f"Could not initialize the ElevenLabs SDK: {exc}")
            raise classified from exc


def resolve_api_key(
    api_key: str | None = None,
    *,
    env_file: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve a key without searching package or working directories."""

    if api_key is not None and api_key.strip():
        return api_key.strip()

    values = os.environ if environ is None else environ
    environment_key = values.get(_API_KEY_NAME)
    if environment_key and environment_key.strip():
        return environment_key.strip()

    if env_file is not None:
        path = Path(env_file).expanduser()
        file_values = _dotenv_values(path)
        file_key = file_values.get(_API_KEY_NAME)
        if file_key and str(file_key).strip():
            return str(file_key).strip()

    location = f" or in the explicit env file {env_file}" if env_file is not None else ""
    raise ProviderCredentialError(f"Missing {_API_KEY_NAME}; set it in the environment{location}")


def build_request_kwargs(options: TranscriptionOptions) -> dict[str, Any]:
    """Translate provider-neutral options into ElevenLabs SDK arguments."""

    kwargs: dict[str, Any] = {
        "model_id": options.model_id,
        "timestamps_granularity": options.timestamps_granularity,
        "diarize": options.diarize,
        "tag_audio_events": options.tag_audio_events,
        "no_verbatim": options.no_verbatim,
    }

    if options.language_code is not None:
        kwargs["language_code"] = options.language_code
    if options.num_speakers is not None and options.diarize:
        kwargs["num_speakers"] = options.num_speakers
    if options.keyterms:
        kwargs["keyterms"] = list(options.keyterms)
    if options.seed is not None:
        kwargs["seed"] = options.seed
    if options.temperature is not None:
        kwargs["temperature"] = options.temperature
    if options.remote_formats:
        formats = _normalise_remote_formats(options.remote_formats)
        kwargs["additional_formats"] = [
            {
                "format": item,
                "include_speakers": options.diarize,
                "include_timestamps": options.timestamps_granularity != "none",
            }
            for item in formats
        ]
    return kwargs


def normalize_response(result: Any) -> dict[str, Any]:
    """Convert supported SDK response representations to a plain dict."""

    if isinstance(result, Mapping):
        return dict(result)

    for method_name in ("model_dump", "dict"):
        method = getattr(result, method_name, None)
        if not callable(method):
            continue
        try:
            payload = method()
        except Exception as exc:
            raise ProviderResponseError(f"Could not decode ElevenLabs response via {method_name}(): {exc}") from exc
        if isinstance(payload, Mapping):
            return dict(payload)
        raise ProviderResponseError(f"ElevenLabs {method_name}() returned {type(payload).__name__}, not a mapping")

    raise ProviderResponseError(f"Unsupported ElevenLabs response type: {type(result).__name__}")


def decode_additional_formats(
    payload: Mapping[str, Any],
    stem: str,
) -> tuple[tuple[str, bytes | str], ...]:
    """Decode non-native remote artifacts without writing them to disk."""

    safe_stem = _validate_stem(stem)
    raw_items = payload.get("additional_formats") or ()
    if isinstance(raw_items, (str, bytes, bytearray)) or not hasattr(raw_items, "__iter__"):
        raise ProviderResponseError("additional_formats must be a sequence")

    decoded: list[tuple[str, bytes | str]] = []
    filenames: set[str] = set()
    for raw_item in raw_items:
        if not isinstance(raw_item, Mapping):
            raise ProviderResponseError("each additional format must be a mapping")

        requested = str(raw_item.get("requested_format") or raw_item.get("format") or "")
        requested = requested.strip().lower().replace("-", "_")
        raw_extension = str(raw_item.get("file_extension") or "").strip().lower()
        extension = raw_extension.removeprefix(".")
        if extension and not _SAFE_EXTENSION.fullmatch(extension):
            raise ProviderResponseError(f"Unsafe additional-format extension: {raw_extension!r}")

        for key in ("filename", "file_name"):
            supplied_name = raw_item.get(key)
            if supplied_name is not None:
                _validate_leaf_name(str(supplied_name), label="additional-format filename")

        # SRT and text are rendered locally. Unknown but path-safe future
        # formats remain present in the canonical payload and are ignored here.
        if requested not in _DECODED_FORMATS:
            continue

        allowed_extensions = _FORMAT_EXTENSIONS[requested]
        default_extension = "json" if requested == "segmented_json" else requested
        extension = extension or default_extension
        if extension not in allowed_extensions:
            raise ProviderResponseError(f"Unexpected extension {extension!r} for additional format {requested!r}")

        content = raw_item.get("content")
        if content is None:
            raise ProviderResponseError(f"Additional format {requested!r} has no content")
        encoded = raw_item.get("is_base_64_encoded", raw_item.get("is_base64_encoded", False))
        if not isinstance(encoded, bool):
            raise ProviderResponseError(f"Additional format {requested!r} has a non-boolean base64 flag")
        value = _decode_content(content, encoded, requested)
        if requested == "segmented_json":
            filename = f"{safe_stem}.segmented.json"
        elif requested == "html":
            filename = f"{safe_stem}.html"
        else:
            filename = f"{safe_stem}.{extension}"
        if filename.casefold() in filenames:
            raise ProviderResponseError(f"Duplicate additional-format filename: {filename}")
        filenames.add(filename.casefold())
        decoded.append((filename, value))

    return tuple(decoded)


def _normalise_remote_formats(formats: tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for raw_format in formats:
        item = str(raw_format).strip().lower().replace("-", "_")
        if item not in _REMOTE_FORMATS:
            choices = ", ".join(sorted(_REMOTE_FORMATS))
            raise ProviderError(f"Unsupported ElevenLabs remote format {raw_format!r}; choose {choices}")
        if item not in seen:
            result.append(item)
            seen.add(item)
    return tuple(result)


def _dotenv_values(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ProviderCredentialError(f"Environment file does not exist: {path}")
    try:
        from dotenv import dotenv_values
    except (ImportError, ModuleNotFoundError):
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            raise ProviderCredentialError(f"Cannot read environment file {path}: {exc}") from exc
        return _parse_dotenv(text)

    try:
        return dotenv_values(path)
    except (OSError, ValueError) as exc:
        raise ProviderCredentialError(f"Cannot read environment file {path}: {exc}") from exc


def _parse_dotenv(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        if "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if not _ENV_KEY.fullmatch(key):
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def _classified_error(exc: Exception) -> ProviderError:
    status = _status_code(exc)
    message = str(exc) or type(exc).__name__
    if status in {401, 403}:
        return ProviderCredentialError(f"ElevenLabs rejected the credentials ({status}): {message}")
    if status == 429 or (status is not None and 500 <= status <= 599):
        return ProviderTransientError(
            f"Temporary ElevenLabs failure ({status}): {message}",
            retry_after_seconds=_retry_after_seconds(exc),
        )
    return ProviderError(f"ElevenLabs transcription failed: {message}")


def _status_code(exc: Exception) -> int | None:
    candidates: list[object] = [getattr(exc, "status_code", None), getattr(exc, "status", None)]
    response = getattr(exc, "response", None)
    if response is not None:
        candidates.extend((getattr(response, "status_code", None), getattr(response, "status", None)))
    for candidate in candidates:
        if isinstance(candidate, bool) or not isinstance(candidate, (int, str)):
            continue
        try:
            status = int(candidate)
        except (TypeError, ValueError):
            continue
        if 100 <= status <= 599:
            return status
    return None


def _retry_after_seconds(exc: Exception) -> float | None:
    direct_headers = getattr(exc, "headers", None)
    response = getattr(exc, "response", None)
    response_headers = getattr(response, "headers", None) if response is not None else None
    for headers in (direct_headers, response_headers):
        if not isinstance(headers, Mapping):
            continue
        normalized = {str(key).casefold(): value for key, value in headers.items()}
        milliseconds = _nonnegative_number(normalized.get("retry-after-ms"))
        if milliseconds is not None:
            return milliseconds / 1000
        value = normalized.get("retry-after")
        seconds = _nonnegative_number(value)
        if seconds is not None:
            return seconds
        if isinstance(value, str):
            try:
                retry_at = parsedate_to_datetime(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
    return None


def _nonnegative_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _decode_content(content: Any, encoded: bool, requested: str) -> bytes | str:
    if encoded:
        if not isinstance(content, (str, bytes, bytearray)):
            raise ProviderResponseError(f"Base64 {requested!r} content must be text or bytes")
        try:
            decoded = base64.b64decode(content, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ProviderResponseError(f"Invalid base64 content for {requested!r}") from exc
        return _normalise_segmented_json(decoded) if requested == "segmented_json" else decoded
    if requested in {"pdf", "docx"}:
        if isinstance(content, bytes):
            return content
        raise ProviderResponseError(f"Binary {requested!r} content must be base64 encoded or bytes")
    if requested == "segmented_json":
        return _normalise_segmented_json(content)
    if isinstance(content, str):
        return content
    if isinstance(content, bytes):
        return content
    raise ProviderResponseError(f"Unsupported content type for {requested!r}: {type(content).__name__}")


def _normalise_segmented_json(content: Any) -> str:
    if isinstance(content, (bytes, bytearray)):
        try:
            content = bytes(content).decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ProviderResponseError("Segmented JSON content must be UTF-8") from exc
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(f"Invalid segmented JSON content: {exc.msg}") from exc
    if not isinstance(content, (Mapping, list, tuple)):
        raise ProviderResponseError(f"Unsupported segmented JSON content type: {type(content).__name__}")
    return json.dumps(content, ensure_ascii=False, indent=2)


def _validate_stem(stem: str) -> str:
    value = str(stem)
    _validate_leaf_name(value, label="output stem")
    return value


def _validate_leaf_name(value: str, *, label: str) -> None:
    if not value or value in {".", ".."}:
        raise ProviderResponseError(f"Unsafe {label}: {value!r}")
    if value[-1:] in {" ", "."} or any(ord(char) < 32 for char in value):
        raise ProviderResponseError(f"Unsafe {label}: {value!r}")
    if any(char in value for char in '<>:"/\\|?*'):
        raise ProviderResponseError(f"Unsafe {label}: {value!r}")
    device_part = value.split(".", 1)[0].upper()
    if device_part in _WINDOWS_DEVICE_NAMES:
        raise ProviderResponseError(f"Unsafe {label}: {value!r}")


__all__ = [
    "ElevenLabsProvider",
    "build_request_kwargs",
    "decode_additional_formats",
    "normalize_response",
    "resolve_api_key",
]
