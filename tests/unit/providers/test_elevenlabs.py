from __future__ import annotations

import base64
import builtins
from pathlib import Path

import pytest

from elevenlabs_toolkit.models import TranscriptionOptions
from elevenlabs_toolkit.providers import (
    ElevenLabsProvider,
    ProviderCredentialError,
    ProviderDependencyError,
    ProviderError,
    ProviderResponseError,
    ProviderTransientError,
    SpeechToTextProvider,
    build_request_kwargs,
    decode_additional_formats,
    normalize_response,
    resolve_api_key,
)


class _Response:
    def model_dump(self):
        return {"text": "hello", "words": []}


class _SpeechToText:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = _Response() if response is None else response
        self.error = error
        self.file = None
        self.content = b""
        self.kwargs = {}

    def convert(self, *, file, **kwargs):
        self.file = file
        self.content = file.read()
        self.kwargs = kwargs
        assert not file.closed
        if self.error is not None:
            raise self.error
        return self.response


class _Client:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.speech_to_text = _SpeechToText(response, error)


def test_injected_client_transcribes_and_closes_input(tmp_path: Path) -> None:
    source = tmp_path / "sample.wav"
    source.write_bytes(b"media")
    client = _Client()
    provider = ElevenLabsProvider(client=client, environ={})

    assert isinstance(provider, SpeechToTextProvider)
    assert provider.transcribe(source, TranscriptionOptions()) == {"text": "hello", "words": []}
    assert client.speech_to_text.content == b"media"
    assert client.speech_to_text.file.closed
    assert client.speech_to_text.kwargs["model_id"] == "scribe_v2"


def test_request_kwargs_cover_all_options_and_omit_inapplicable_values() -> None:
    options = TranscriptionOptions(
        model_id="scribe_v2",
        language_code="uzb",
        timestamps_granularity="character",
        diarize=True,
        tag_audio_events=False,
        num_speakers=3,
        keyterms=("Codex", "Toshkent"),
        no_verbatim=True,
        seed=7,
        temperature=0.25,
        remote_formats=("PDF", "segmented-json", "pdf"),
    )

    assert build_request_kwargs(options) == {
        "model_id": "scribe_v2",
        "language_code": "uzb",
        "timestamps_granularity": "character",
        "diarize": True,
        "tag_audio_events": False,
        "num_speakers": 3,
        "keyterms": ["Codex", "Toshkent"],
        "no_verbatim": True,
        "seed": 7,
        "temperature": 0.25,
        "additional_formats": [
            {"format": "pdf", "include_speakers": True, "include_timestamps": True},
            {
                "format": "segmented_json",
                "include_speakers": True,
                "include_timestamps": True,
            },
        ],
    }

    inapplicable = build_request_kwargs(
        TranscriptionOptions(
            language_code=None,
            diarize=False,
            num_speakers=2,
            seed=None,
            temperature=None,
        )
    )
    assert "language_code" not in inapplicable
    assert "num_speakers" not in inapplicable
    assert "seed" not in inapplicable
    assert "temperature" not in inapplicable
    assert "additional_formats" not in inapplicable


def test_credentials_use_environment_then_only_an_explicit_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nELEVENLABS_API_KEY='from file'\n", encoding="utf-8")

    assert resolve_api_key(environ={"ELEVENLABS_API_KEY": " from env "}, env_file=env_file) == "from env"
    assert resolve_api_key(environ={}, env_file=env_file) == "from file"
    with pytest.raises(ProviderCredentialError, match="Missing ELEVENLABS_API_KEY"):
        resolve_api_key(environ={})


def test_provider_construction_and_import_do_not_require_sdk(monkeypatch) -> None:
    provider = ElevenLabsProvider(api_key="test")
    real_import = builtins.__import__

    def reject_sdk(name, *args, **kwargs):
        if name == "elevenlabs" or name.startswith("elevenlabs."):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_sdk)
    with pytest.raises(ProviderDependencyError, match="optional 'elevenlabs'"):
        _ = provider.client


@pytest.mark.parametrize("status", [429, 500, 503])
def test_transient_http_statuses_are_classified_without_sdk_types(tmp_path: Path, status: int) -> None:
    class HttpFailure(Exception):
        status_code = status

    source = tmp_path / "sample.wav"
    source.write_bytes(b"media")
    with pytest.raises(ProviderTransientError):
        ElevenLabsProvider(client=_Client(error=HttpFailure("try later"))).transcribe(source, TranscriptionOptions())


def test_auth_http_status_is_classified(tmp_path: Path) -> None:
    class Response:
        status_code = 401

    class HttpFailure(Exception):
        response = Response()

    source = tmp_path / "sample.wav"
    source.write_bytes(b"media")
    with pytest.raises(ProviderCredentialError):
        ElevenLabsProvider(client=_Client(error=HttpFailure("bad key"))).transcribe(source, TranscriptionOptions())


def test_retry_after_seconds_are_preserved_on_transient_error(tmp_path: Path) -> None:
    class Response:
        def __init__(self) -> None:
            self.status_code = 429
            self.headers = {"Retry-After": "2.5"}

    class HttpFailure(Exception):
        response = Response()

    source = tmp_path / "sample.wav"
    source.write_bytes(b"media")
    with pytest.raises(ProviderTransientError) as caught:
        ElevenLabsProvider(client=_Client(error=HttpFailure("rate limited"))).transcribe(source, TranscriptionOptions())

    assert caught.value.retry_after_seconds == 2.5


def test_direct_retry_after_milliseconds_take_precedence(tmp_path: Path) -> None:
    class HttpFailure(Exception):
        status_code = 429

        def __init__(self) -> None:
            self.headers = {"Retry-After": "9", "Retry-After-Ms": "1250"}

    source = tmp_path / "sample.wav"
    source.write_bytes(b"media")
    with pytest.raises(ProviderTransientError) as caught:
        ElevenLabsProvider(client=_Client(error=HttpFailure())).transcribe(source, TranscriptionOptions())

    assert caught.value.retry_after_seconds == 1.25


def test_normalize_response_supports_mapping_and_legacy_dict_method() -> None:
    class LegacyResponse:
        def dict(self):
            return {"text": "legacy"}

    assert normalize_response({"text": "mapping"}) == {"text": "mapping"}
    assert normalize_response(LegacyResponse()) == {"text": "legacy"}
    with pytest.raises(ProviderResponseError, match="Unsupported"):
        normalize_response(object())


def test_decode_supported_additional_formats_without_writing() -> None:
    binary_pdf = b"%PDF-test"
    payload = {
        "additional_formats": [
            {
                "requested_format": "pdf",
                "file_extension": ".pdf",
                "content": base64.b64encode(binary_pdf).decode("ascii"),
                "is_base_64_encoded": True,
            },
            {
                "requested_format": "docx",
                "file_extension": "docx",
                "content": base64.b64encode(b"docx").decode("ascii"),
                "is_base64_encoded": True,
            },
            {
                "requested_format": "html",
                "file_extension": "htm",
                "content": "<p>Hello</p>",
            },
            {
                "requested_format": "segmented_json",
                "file_extension": "json",
                "content": {"segments": []},
            },
            # Native outputs are intentionally rendered elsewhere.
            {"requested_format": "srt", "file_extension": "srt", "content": "1"},
        ]
    }

    assert decode_additional_formats(payload, "episode 1") == (
        ("episode 1.pdf", binary_pdf),
        ("episode 1.docx", b"docx"),
        ("episode 1.html", "<p>Hello</p>"),
        ("episode 1.segmented.json", '{\n  "segments": []\n}'),
    )


@pytest.mark.parametrize("stem", ["../escape", "folder/name", "CON", "bad:name", "trailing."])
def test_decode_rejects_unsafe_output_stems(stem: str) -> None:
    with pytest.raises(ProviderResponseError, match="Unsafe output stem"):
        decode_additional_formats({}, stem)


def test_decode_rejects_unsafe_or_mismatched_extensions_and_names() -> None:
    with pytest.raises(ProviderResponseError, match="Unsafe additional-format extension"):
        decode_additional_formats(
            {"additional_formats": [{"requested_format": "pdf", "file_extension": "../pdf", "content": "x"}]},
            "safe",
        )
    with pytest.raises(ProviderResponseError, match="Unexpected extension"):
        decode_additional_formats(
            {"additional_formats": [{"requested_format": "pdf", "file_extension": "html", "content": "x"}]},
            "safe",
        )
    with pytest.raises(ProviderResponseError, match="Unsafe additional-format filename"):
        decode_additional_formats(
            {
                "additional_formats": [
                    {
                        "requested_format": "html",
                        "file_extension": "html",
                        "filename": "../escape.html",
                        "content": "x",
                    }
                ]
            },
            "safe",
        )


@pytest.mark.parametrize(
    "item",
    [
        {"requested_format": "pdf", "file_extension": "pdf", "content": "not base64"},
        {
            "requested_format": "pdf",
            "file_extension": "pdf",
            "content": "eA==",
            "is_base_64_encoded": "true",
        },
        {"requested_format": "segmented_json", "file_extension": "json", "content": "not json"},
    ],
)
def test_decode_rejects_corrupt_or_ambiguous_remote_content(item: dict) -> None:
    with pytest.raises(ProviderResponseError):
        decode_additional_formats({"additional_formats": [item]}, "safe")


def test_missing_input_is_provider_error_and_unknown_format_is_model_error(tmp_path: Path) -> None:
    with pytest.raises(ProviderError, match="Cannot open"):
        ElevenLabsProvider(client=_Client()).transcribe(tmp_path / "missing.wav", TranscriptionOptions())
    with pytest.raises(ValueError, match="unsupported remote format"):
        TranscriptionOptions(remote_formats=("exe",))
