import base64
import json
from pathlib import Path

import pytest

import elevenlabs_toolkit.application.transcriber as transcriber_module
from elevenlabs_toolkit.application import execute_transcription, plan_transcription
from elevenlabs_toolkit.models import ArtifactFormat, ConflictPolicy, ExportOptions, TranscriptionOptions
from elevenlabs_toolkit.providers import ProviderTransientError

WORD_PAYLOAD = {
    "text": "hello world",
    "words": [
        {"type": "word", "text": "hello", "start": 0, "end": 0.3},
        {"type": "word", "text": "world", "start": 0.4, "end": 0.8},
    ],
}
PAYLOAD = {
    "text": "hello world",
    "words": [
        {
            "type": "word",
            "text": "hello",
            "start": 0,
            "end": 0.3,
            "characters": [
                {"text": "h", "start": 0, "end": 0.06},
                {"text": "o", "start": 0.24, "end": 0.3},
            ],
        },
        {
            "type": "word",
            "text": "world",
            "start": 0.4,
            "end": 0.8,
            "characters": [
                {"text": "w", "start": 0.4, "end": 0.48},
                {"text": "d", "start": 0.72, "end": 0.8},
            ],
        },
    ],
}


class FakeProvider:
    def __init__(self, payload: dict, transient_failures: int = 0) -> None:
        self.payload = payload
        self.transient_failures = transient_failures
        self.calls = 0

    def transcribe(self, path: Path, options: TranscriptionOptions) -> dict:
        self.calls += 1
        if self.calls <= self.transient_failures:
            raise ProviderTransientError("rate limited")
        return self.payload


def _job(
    tmp_path: Path,
    formats: tuple[ArtifactFormat, ...] = (ArtifactFormat.JSON,),
    *,
    stt: TranscriptionOptions | None = None,
    policy: ConflictPolicy = ConflictPolicy.ERROR,
):
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"audio")
    output = tmp_path / "out"
    options = stt or TranscriptionOptions()
    export = ExportOptions(formats, output)
    plan = plan_transcription(
        source.parent.glob("*.mp3"), output, formats, policy=policy, transcription_options=options
    )
    return source, output, options, export, plan


def test_transcription_writes_provider_json_without_sidecar_files(tmp_path: Path) -> None:
    _source, output, stt, export, plan = _job(tmp_path)

    result = execute_transcription(plan, stt, export, provider=FakeProvider(PAYLOAD))

    assert result.failed == 0
    assert {path.name for path in output.iterdir()} == {"clip.json"}
    assert json.loads((output / "clip.json").read_text(encoding="utf-8")) == PAYLOAD


def test_transcription_progress_reports_file_phase_and_completion(tmp_path: Path) -> None:
    _source, _output, stt, export, plan = _job(tmp_path)
    messages: list[str] = []

    result = execute_transcription(
        plan,
        stt,
        export,
        provider=FakeProvider(PAYLOAD),
        progress=messages.append,
    )

    assert result.failed == 0
    assert messages[0] == "[1/1] clip.mp3 (5 B) - uploading + transcribing with ElevenLabs"
    assert messages[1].startswith("[1/1] clip.mp3 (5 B) - response received after ")
    assert "transcript ready (11 characters, 2 timed items); writing 1 output(s)" in messages[2]
    assert messages[3] == "[1/1] clip.mp3 (5 B) - complete"


def test_replace_policy_always_requests_a_fresh_transcription(tmp_path: Path) -> None:
    _source, output, stt, export, first_plan = _job(tmp_path, policy=ConflictPolicy.REPLACE)
    first_provider = FakeProvider(PAYLOAD)
    execute_transcription(first_plan, stt, export, provider=first_provider, policy=ConflictPolicy.REPLACE)

    second_plan = plan_transcription(
        first_plan.sources,
        output,
        export.formats,
        policy=ConflictPolicy.REPLACE,
        transcription_options=stt,
    )
    second_provider = FakeProvider(
        {
            "text": "fresh text",
            "words": [
                {
                    "type": "word",
                    "text": "fresh",
                    "start": 0,
                    "end": 0.3,
                    "characters": [{"text": "f", "start": 0, "end": 0.1}],
                },
                {
                    "type": "word",
                    "text": "text",
                    "start": 0.4,
                    "end": 0.8,
                    "characters": [{"text": "t", "start": 0.4, "end": 0.5}],
                },
            ],
        }
    )
    result = execute_transcription(
        second_plan,
        stt,
        export,
        provider=second_provider,
        policy=ConflictPolicy.REPLACE,
    )

    assert result.failed == 0
    assert first_provider.calls == second_provider.calls == 1
    saved = json.loads((output / "clip.json").read_text(encoding="utf-8"))
    assert saved["text"] == "fresh text"


def test_skip_policy_avoids_provider_when_all_outputs_exist(tmp_path: Path) -> None:
    source, output, stt, export, _plan = _job(tmp_path, policy=ConflictPolicy.SKIP)
    output.mkdir()
    (output / "clip.json").write_text('{"text":"edited"}', encoding="utf-8")
    plan = plan_transcription((source,), output, export.formats, policy=ConflictPolicy.SKIP, transcription_options=stt)
    provider = FakeProvider(PAYLOAD)

    result = execute_transcription(plan, stt, export, provider=provider, policy=ConflictPolicy.SKIP)

    assert plan.api_requests == 0
    assert result.skipped == 1
    assert provider.calls == 0


def test_transient_provider_error_is_retried(tmp_path: Path) -> None:
    _source, _output, stt, export, plan = _job(tmp_path, (ArtifactFormat.TXT,))
    provider = FakeProvider(PAYLOAD, transient_failures=1)

    result = execute_transcription(plan, stt, export, provider=provider, retries=1, backoff_seconds=0)

    assert result.failed == 0
    assert provider.calls == 2


def test_provider_retry_after_overrides_local_backoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class RetryAfterProvider(FakeProvider):
        def transcribe(self, path: Path, options: TranscriptionOptions) -> dict:
            self.calls += 1
            if self.calls == 1:
                raise ProviderTransientError("rate limited", retry_after_seconds=2.5)
            return self.payload

    _source, _output, stt, export, plan = _job(tmp_path, (ArtifactFormat.TXT,))
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
    stt = TranscriptionOptions(remote_formats=("pdf",))
    source, output, _stt, _export, _plan = _job(tmp_path, (ArtifactFormat.TXT,), stt=stt)
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
    formats = (ArtifactFormat.TXT, ArtifactFormat.PDF)
    export = ExportOptions(formats, output)
    plan = plan_transcription((source,), output, (ArtifactFormat.TXT,), transcription_options=stt)

    result = execute_transcription(plan, stt, export, provider=FakeProvider(payload))

    assert result.failed == 0
    assert {path.name for path in output.iterdir()} == {"clip.txt", "clip.pdf"}
    assert (output / "clip.pdf").read_bytes() == b"pdf bytes"


def test_untimed_provider_words_can_render_plain_text(tmp_path: Path) -> None:
    stt = TranscriptionOptions(timestamps_granularity="none")
    _source, output, _stt, export, plan = _job(tmp_path, (ArtifactFormat.TXT,), stt=stt)
    payload = {
        "text": "hello world",
        "words": [
            {"type": "word", "text": "hello", "start": None, "end": None},
            {"type": "spacing", "text": " ", "start": None, "end": None},
            {"type": "word", "text": "world", "start": None, "end": None},
        ],
    }

    result = execute_transcription(plan, stt, export, provider=FakeProvider(payload))

    assert result.failed == 0
    assert (output / "clip.txt").read_text(encoding="utf-8").strip() == "hello world"


@pytest.mark.parametrize(
    ("stt", "payload"),
    [
        (TranscriptionOptions(), {"message": "queued", "request_id": "request-1"}),
        (TranscriptionOptions(), {"text": "hello", "words": []}),
        (TranscriptionOptions(timestamps_granularity="character"), WORD_PAYLOAD),
        (TranscriptionOptions(remote_formats=("pdf",)), PAYLOAD),
    ],
)
def test_invalid_or_incomplete_response_leaves_no_outputs(
    tmp_path: Path,
    stt: TranscriptionOptions,
    payload: dict,
) -> None:
    source, output, _stt, _export, _plan = _job(tmp_path, (ArtifactFormat.SRT,), stt=stt)
    export_formats = (ArtifactFormat.SRT,)
    if stt.remote_formats:
        export_formats = (*export_formats, ArtifactFormat.PDF)
    export = ExportOptions(export_formats, output)
    plan = plan_transcription((source,), output, (ArtifactFormat.SRT,), transcription_options=stt)

    result = execute_transcription(plan, stt, export, provider=FakeProvider(payload))

    assert result.failed > 0
    assert not output.exists() or not any(output.iterdir())


def test_source_change_during_provider_call_discards_response(tmp_path: Path) -> None:
    class MutatingProvider(FakeProvider):
        def transcribe(self, path: Path, options: TranscriptionOptions) -> dict:
            payload = super().transcribe(path, options)
            path.write_bytes(b"changed while uploading")
            return payload

    _source, output, stt, export, plan = _job(tmp_path, (ArtifactFormat.TXT,))

    result = execute_transcription(plan, stt, export, provider=MutatingProvider(PAYLOAD))

    assert result.failed > 0
    assert not output.exists() or not any(output.iterdir())


def test_no_clobber_capability_is_checked_before_provider_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, output, stt, export, plan = _job(tmp_path, (ArtifactFormat.TXT,))
    provider = FakeProvider(PAYLOAD)

    def unsupported(_directory: Path) -> None:
        raise OSError("hard links unavailable")

    monkeypatch.setattr(transcriber_module, "ensure_atomic_no_clobber_supported", unsupported)

    result = execute_transcription(plan, stt, export, provider=provider)

    assert result.failed > 0
    assert provider.calls == 0
    assert not output.exists()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"backoff_seconds": float("nan")}, "backoff_seconds"),
        ({"request_delay": float("inf")}, "request_delay"),
    ],
)
def test_non_finite_execution_pacing_is_rejected_before_provider_request(
    tmp_path: Path,
    kwargs: dict[str, float],
    message: str,
) -> None:
    _source, output, stt, export, plan = _job(tmp_path, (ArtifactFormat.TXT,))
    provider = FakeProvider(PAYLOAD)

    with pytest.raises(ValueError, match=message):
        execute_transcription(plan, stt, export, provider=provider, **kwargs)

    assert provider.calls == 0
    assert not output.exists()
