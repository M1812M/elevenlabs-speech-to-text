import json
from pathlib import Path

from elevenlabs_toolkit.application import build_manifest, cache_matches
from elevenlabs_toolkit.models import TranscriptionOptions


def test_cache_manifest_matches_source_content_and_options(tmp_path: Path) -> None:
    source = tmp_path / "audio.mp3"
    source.write_bytes(b"audio")
    options = TranscriptionOptions(language_code="uzb")
    transcript = tmp_path / "audio.json"
    transcript_content = '{"text": "hello"}\n'
    transcript.write_text(transcript_content, encoding="utf-8", newline="")
    manifest = tmp_path / "audio.manifest.json"
    manifest.write_text(
        json.dumps(
            build_manifest(
                source,
                options,
                transcript_name=transcript.name,
                transcript_content=transcript_content,
            )
        ),
        encoding="utf-8",
    )

    assert cache_matches(source, transcript, manifest, options)
    assert not cache_matches(source, transcript, manifest, TranscriptionOptions(language_code="eng"))

    transcript.write_text('{"text": "edited"}\n', encoding="utf-8")
    assert not cache_matches(source, transcript, manifest, options)
    transcript.write_text(transcript_content, encoding="utf-8")

    source.write_bytes(b"changed")
    assert not cache_matches(source, transcript, manifest, options)


def test_cache_is_bound_to_provider_and_canonical_request_options(tmp_path: Path) -> None:
    source = tmp_path / "audio.mp3"
    source.write_bytes(b"audio")
    transcript = tmp_path / "audio.json"
    content = '{"text":"hello"}'
    transcript.write_text(content, encoding="utf-8")
    manifest = tmp_path / "audio.manifest.json"
    original = TranscriptionOptions(
        diarize=False,
        num_speakers=3,
        remote_formats=("segmented-json", "PDF", "pdf"),
    )
    manifest.write_text(
        json.dumps(
            build_manifest(
                source,
                original,
                transcript_name=transcript.name,
                transcript_content=content,
                provider="provider-a",
            )
        ),
        encoding="utf-8",
    )

    equivalent = TranscriptionOptions(
        diarize=False,
        num_speakers=None,
        remote_formats=("segmented_json", "pdf"),
    )
    assert cache_matches(source, transcript, manifest, equivalent, provider="provider-a")
    assert not cache_matches(source, transcript, manifest, equivalent, provider="provider-b")
