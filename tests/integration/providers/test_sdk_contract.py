"""Contract checks against the installed optional ElevenLabs SDK."""

import inspect
from pathlib import Path

import pytest

from elevenlabs_toolkit.models import Transcript, TranscriptionOptions
from elevenlabs_toolkit.providers import (
    ElevenLabsProvider,
    ProviderTransientError,
    build_request_kwargs,
    normalize_response,
)


def test_request_mapping_uses_only_supported_sdk_keywords() -> None:
    pytest.importorskip("elevenlabs")
    from elevenlabs.client import ElevenLabs

    convert = ElevenLabs(api_key="contract-check-only").speech_to_text.convert
    supported = set(inspect.signature(convert).parameters)
    mapped = build_request_kwargs(
        TranscriptionOptions(
            timestamps_granularity="character",
            remote_formats=("pdf", "segmented-json"),
        )
    )

    assert set(mapped) <= supported - {"file"}
    from elevenlabs.types.export_options import ExportOptions_Pdf, ExportOptions_SegmentedJson

    assert ExportOptions_Pdf(**mapped["additional_formats"][0]).format == "pdf"
    assert ExportOptions_SegmentedJson(**mapped["additional_formats"][1]).format == "segmented_json"


def test_locked_sdk_response_schema_matches_adapter_assumptions() -> None:
    pytest.importorskip("elevenlabs")
    from elevenlabs.types import (
        AdditionalFormatResponseModel,
        SpeechToTextChunkResponseModel,
        SpeechToTextWordResponseModel,
    )

    assert "is_base_64_encoded" in AdditionalFormatResponseModel.model_fields
    assert SpeechToTextWordResponseModel.model_fields["start"].default is None
    assert SpeechToTextWordResponseModel.model_fields["end"].default is None

    response = SpeechToTextChunkResponseModel(
        language_code="en",
        language_probability=0.99,
        text="hello",
        words=[SpeechToTextWordResponseModel(text="hello", type="word", logprob=-0.1)],
    )
    transcript = Transcript.from_payload(normalize_response(response))

    assert transcript.text == "hello"
    assert transcript.timed_words == ()


def test_locked_sdk_api_error_exposes_retry_headers_directly(tmp_path: Path) -> None:
    pytest.importorskip("elevenlabs")
    from elevenlabs.core.api_error import ApiError

    class Converter:
        def convert(self, **_kwargs):
            raise ApiError(headers={"retry-after": "2.5"}, status_code=429, body="rate limited")

    class Client:
        class SpeechToText:
            convert = Converter().convert

        speech_to_text = SpeechToText()

    source = tmp_path / "sample.wav"
    source.write_bytes(b"media")
    with pytest.raises(ProviderTransientError) as caught:
        ElevenLabsProvider(client=Client()).transcribe(source, TranscriptionOptions())

    assert caught.value.retry_after_seconds == 2.5
