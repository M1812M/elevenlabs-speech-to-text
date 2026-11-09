from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt

from elevenlabs_toolkit.gui import Worker
from elevenlabs_toolkit.models import ScriptMode


def test_conversion_continues_after_bad_input_and_preserves_source(tmp_path: Path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xff")
    good = tmp_path / "good.txt"
    good.write_text("Salom", encoding="utf-8")
    worker = Worker("Convert Script", ScriptMode.CYRILLIC, [bad, good], False)
    messages = []
    worker.finished_result.connect(messages.append, Qt.ConnectionType.DirectConnection)
    worker.run()
    assert good.read_text() == "Salom"
    assert (tmp_path / "good.cyrillic.txt").read_text() == "Салом"
    assert "FAILED" in messages[0]


def test_skipped_transcription_needs_no_provider(tmp_path: Path, monkeypatch) -> None:
    import elevenlabs_toolkit.providers as providers

    def forbidden(*args, **kwargs):
        pytest.fail("Provider must not be constructed for a skipped job")

    monkeypatch.setattr(providers, "ElevenLabsProvider", forbidden)
    source = tmp_path / "audio.flac"
    source.write_bytes(b"placeholder")
    target = tmp_path / "audio.flac.transcript.json"
    target.write_text("{}")
    worker = Worker("Transcribe", ScriptMode.SOURCE, [source], False)
    messages = []
    worker.finished_result.connect(messages.append, Qt.ConnectionType.DirectConnection)
    worker.run()
    assert "skipped" in messages[0]
    assert "FAILED" not in messages[0]


def test_stop_between_files(tmp_path: Path) -> None:
    sources = [tmp_path / f"{number}.txt" for number in range(3)]
    for source in sources:
        source.write_text("Salom")
    worker = Worker("Convert Script", ScriptMode.CYRILLIC, sources, False)
    worker.progress_changed.connect(
        lambda completed, total: worker.requestInterruption(),
        Qt.ConnectionType.DirectConnection,
    )
    worker.start()
    assert worker.wait(5000)
    assert sources[0].with_suffix(".cyrillic.txt").exists()
    assert not sources[1].with_suffix(".cyrillic.txt").exists()


def test_export_multiple_formats_is_offline(tmp_path: Path, monkeypatch) -> None:
    import json

    import elevenlabs_toolkit.providers as providers
    from elevenlabs_toolkit.models import ArtifactFormat

    def forbidden(*args, **kwargs):
        pytest.fail("Offline export must not construct a provider")

    monkeypatch.setattr(providers, "ElevenLabsProvider", forbidden)
    source = tmp_path / "transcript.json"
    source.write_text(
        json.dumps(
            {
                "text": "Hello world.",
                "words": [
                    {"text": "Hello", "start": 0.0, "end": 0.4, "type": "word"},
                    {"text": "world.", "start": 0.5, "end": 1.0, "type": "word"},
                ],
            }
        )
    )
    worker = Worker(
        "Export",
        ScriptMode.SOURCE,
        [source],
        False,
        (ArtifactFormat.SRT, ArtifactFormat.SRT_MINI, ArtifactFormat.TXT),
    )
    messages = []
    worker.finished_result.connect(messages.append, Qt.ConnectionType.DirectConnection)
    worker.run()
    assert "failed" not in messages[0].lower()
    assert "Hello" in source.with_suffix(".txt").read_text()
    assert "-->" in source.with_suffix(".srt").read_text()
    assert "-->" in source.with_suffix(".mini.srt").read_text()
