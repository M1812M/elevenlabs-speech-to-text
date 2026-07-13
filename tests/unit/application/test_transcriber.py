import base64
import json
from pathlib import Path

import pytest

import elevenlabs_toolkit.application.transcriber as transcriber_module
from elevenlabs_toolkit.application import execute_transcription, plan_transcription
from elevenlabs_toolkit.files import exclusive_file_lock
from elevenlabs_toolkit.models import ArtifactFormat, ExportOptions, TranscriptionOptions
from elevenlabs_toolkit.providers import ProviderTransientError

PAYLOAD = {
    "text": "hello world",
    "words": [
        {"type": "word", "text": "hello", "start": 0, "end": 0.3},
        {"type": "word", "text": "world", "start": 0.4, "end": 0.8},
    ],
}


class FakeProvider:
    cache_key = "elevenlabs"

    def __init__(self, payload: dict, transient_failures: int = 0) -> None:
        self.payload = payload
        self.transient_failures = transient_failures
        self.calls = 0

    def transcribe(self, path: Path, options: TranscriptionOptions) -> dict:
        self.calls += 1
        if self.calls <= self.transient_failures:
            raise ProviderTransientError("rate limited")
        return self.payload


def test_transcription_writes_cache_manifest_and_local_render(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"audio")
    output = tmp_path / "out"
    stt = TranscriptionOptions()
    export = ExportOptions((ArtifactFormat.SRT,), output)
    plan = plan_transcription((source,), output, (ArtifactFormat.SRT,), transcription_options=stt)
    provider = FakeProvider(PAYLOAD)

    result = execute_transcription(plan, stt, export, provider=provider)

    assert result.failed == 0
    assert provider.calls == 1
    assert (output / "clip.json").is_file()
    assert (output / "clip.manifest.json").is_file()
    assert "hello world" in (output / "clip.srt").read_text(encoding="utf-8")


def test_valid_cache_renders_new_format_without_provider_call(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"audio")
    output = tmp_path / "out"
    stt = TranscriptionOptions()
    initial_options = ExportOptions((ArtifactFormat.JSON,), output)
    initial = plan_transcription((source,), output, (), transcription_options=stt)
    execute_transcription(initial, stt, initial_options, provider=FakeProvider(PAYLOAD))

    export = ExportOptions((ArtifactFormat.SRT,), output)
    resumed = plan_transcription((source,), output, (ArtifactFormat.SRT,), transcription_options=stt)

    assert resumed.api_requests == 0
    result = execute_transcription(resumed, stt, export, provider=None)
    assert result.failed == 0
    assert (output / "clip.srt").is_file()


def test_transient_provider_error_is_retried(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"audio")
    stt = TranscriptionOptions()
    export = ExportOptions((ArtifactFormat.JSON,), tmp_path / "out")
    plan = plan_transcription((source,), export.output_dir, (), transcription_options=stt)
    provider = FakeProvider(PAYLOAD, transient_failures=1)

    result = execute_transcription(plan, stt, export, provider=provider, retries=1, backoff_seconds=0)

    assert result.failed == 0
    assert provider.calls == 2


def test_provider_retry_after_overrides_local_backoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RetryAfterProvider(FakeProvider):
        def transcribe(self, path: Path, options: TranscriptionOptions) -> dict:
            self.calls += 1
            if self.calls == 1:
                raise ProviderTransientError("rate limited", retry_after_seconds=2.5)
            return self.payload

    source = tmp_path / "clip.mp3"
    source.write_bytes(b"audio")
    stt = TranscriptionOptions()
    export = ExportOptions((ArtifactFormat.JSON,), tmp_path / "out")
    plan = plan_transcription((source,), export.output_dir, (), transcription_options=stt)
    delays: list[float] = []
    monkeypatch.setattr(transcriber_module.time, "sleep", delays.append)

    result = execute_transcription(
        plan,
        stt,
        export,
        provider=RetryAfterProvider(PAYLOAD),
        retries=1,
        backoff_seconds=99,
    )

    assert result.failed == 0
    assert delays == [2.5]


def test_remote_additional_format_is_decoded_and_written(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"audio")
    output = tmp_path / "out"
    stt = TranscriptionOptions(remote_formats=("pdf",))
    payload = {
        **PAYLOAD,
        "additional_formats": [
            {
                "requested_format": "pdf",
                "file_extension": "pdf",
                "is_base64_encoded": True,
                "content": base64.b64encode(b"pdf bytes").decode("ascii"),
            }
        ],
    }
    export = ExportOptions((ArtifactFormat.JSON,), output)
    plan = plan_transcription((source,), output, (), transcription_options=stt)

    result = execute_transcription(plan, stt, export, provider=FakeProvider(payload))

    assert result.failed == 0
    assert (output / "clip.pdf").read_bytes() == b"pdf bytes"


def test_untimed_provider_words_are_valid_when_timestamps_are_disabled(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"audio")
    output = tmp_path / "out"
    stt = TranscriptionOptions(timestamps_granularity="none")
    payload = {
        "text": "hello world",
        "words": [
            {"type": "word", "text": "hello", "start": None, "end": None},
            {"type": "spacing", "text": " ", "start": None, "end": None},
            {"type": "word", "text": "world", "start": None, "end": None},
        ],
    }
    export = ExportOptions((ArtifactFormat.JSON,), output)
    plan = plan_transcription((source,), output, (), transcription_options=stt)

    result = execute_transcription(plan, stt, export, provider=FakeProvider(payload))

    assert result.failed == 0
    assert json.loads((output / "clip.json").read_text(encoding="utf-8"))["text"] == "hello world"


@pytest.mark.parametrize(
    ("stt", "formats", "payload"),
    [
        (TranscriptionOptions(), (), {"message": "queued", "request_id": "request-1"}),
        (TranscriptionOptions(), (), {"text": "hello", "words": []}),
        (
            TranscriptionOptions(timestamps_granularity="none"),
            (ArtifactFormat.SRT,),
            {"text": "hello", "words": []},
        ),
        (TranscriptionOptions(timestamps_granularity="character"), (), PAYLOAD),
        (TranscriptionOptions(remote_formats=("pdf",)), (), PAYLOAD),
    ],
)
def test_invalid_or_incomplete_response_is_not_cached(
    tmp_path: Path,
    stt: TranscriptionOptions,
    formats: tuple[ArtifactFormat, ...],
    payload: dict,
) -> None:
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"audio")
    output = tmp_path / "out"
    export = ExportOptions(formats or (ArtifactFormat.JSON,), output)
    plan = plan_transcription((source,), output, formats, transcription_options=stt)

    result = execute_transcription(plan, stt, export, provider=FakeProvider(payload))

    assert result.failed > 0
    assert not (output / "clip.json").exists()
    assert not (output / "clip.manifest.json").exists()


def test_source_change_during_provider_call_discards_response(tmp_path: Path) -> None:
    class MutatingProvider(FakeProvider):
        def transcribe(self, path: Path, options: TranscriptionOptions) -> dict:
            payload = super().transcribe(path, options)
            path.write_bytes(b"changed while uploading")
            return payload

    source = tmp_path / "clip.mp3"
    source.write_bytes(b"audio")
    output = tmp_path / "out"
    stt = TranscriptionOptions()
    export = ExportOptions((ArtifactFormat.JSON,), output)
    plan = plan_transcription((source,), output, (), transcription_options=stt)

    result = execute_transcription(plan, stt, export, provider=MutatingProvider(PAYLOAD))

    assert result.failed > 0
    assert not (output / "clip.json").exists()


def test_cache_lock_prevents_duplicate_paid_request(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"audio")
    output = tmp_path / "out"
    stt = TranscriptionOptions()
    export = ExportOptions((ArtifactFormat.JSON,), output)
    plan = plan_transcription((source,), output, (), transcription_options=stt)
    provider = FakeProvider(PAYLOAD)

    with exclusive_file_lock(output / ".clip.transcription.lock"):
        result = execute_transcription(plan, stt, export, provider=provider, lock_timeout_seconds=0)

    assert result.failed > 0
    assert provider.calls == 0


def test_stale_plan_rechecks_cache_after_lock_acquisition(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"audio")
    output = tmp_path / "out"
    stt = TranscriptionOptions()
    export = ExportOptions((ArtifactFormat.JSON,), output)
    first_plan = plan_transcription((source,), output, (), transcription_options=stt)
    waiting_plan = plan_transcription((source,), output, (), transcription_options=stt)

    first = execute_transcription(first_plan, stt, export, provider=FakeProvider(PAYLOAD))
    waiting_provider = FakeProvider(PAYLOAD)
    second = execute_transcription(waiting_plan, stt, export, provider=waiting_provider)

    assert first.failed == 0
    assert second.failed == 0
    assert waiting_provider.calls == 0


def test_no_clobber_capability_is_checked_before_provider_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"audio")
    output = tmp_path / "out"
    stt = TranscriptionOptions()
    export = ExportOptions((ArtifactFormat.JSON,), output)
    plan = plan_transcription((source,), output, (), transcription_options=stt)
    provider = FakeProvider(PAYLOAD)

    def unsupported(_directory: Path) -> None:
        raise OSError("hard links unavailable")

    monkeypatch.setattr(transcriber_module, "ensure_atomic_no_clobber_supported", unsupported)

    result = execute_transcription(plan, stt, export, provider=provider)

    assert result.failed > 0
    assert provider.calls == 0
    assert not (output / "clip.json").exists()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"backoff_seconds": float("nan")}, "backoff_seconds"),
        ({"request_delay": float("inf")}, "request_delay"),
        ({"lock_timeout_seconds": float("-inf")}, "lock_timeout_seconds"),
    ],
)
def test_non_finite_execution_pacing_is_rejected_before_provider_request(
    tmp_path: Path,
    kwargs: dict[str, float],
    message: str,
) -> None:
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"audio")
    output = tmp_path / "out"
    stt = TranscriptionOptions()
    export = ExportOptions((ArtifactFormat.JSON,), output)
    plan = plan_transcription((source,), output, (), transcription_options=stt)
    provider = FakeProvider(PAYLOAD)

    with pytest.raises(ValueError, match=message):
        execute_transcription(plan, stt, export, provider=provider, **kwargs)

    assert provider.calls == 0
    assert not output.exists()


def test_failed_manifest_publish_rolls_back_new_transcript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"audio")
    output = tmp_path / "out"
    stt = TranscriptionOptions()
    export = ExportOptions((ArtifactFormat.JSON,), output)
    plan = plan_transcription((source,), output, (), transcription_options=stt)
    real_atomic_write = transcriber_module.atomic_write_text

    def fail_manifest(path: Path, content: str, policy):
        if Path(path).name.endswith(".manifest.json"):
            raise OSError("manifest disk failure")
        return real_atomic_write(path, content, policy)

    monkeypatch.setattr(transcriber_module, "atomic_write_text", fail_manifest)

    result = execute_transcription(plan, stt, export, provider=FakeProvider(PAYLOAD))

    assert result.failed > 0
    assert not (output / "clip.json").exists()
    assert not (output / "clip.manifest.json").exists()
