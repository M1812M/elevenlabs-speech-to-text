import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from elevenlabs_toolkit.application.desktop import DesktopSettings
from elevenlabs_toolkit.application.inspection import inspect_transcript
from elevenlabs_toolkit.cli.main import main
from elevenlabs_toolkit.gui import MainWindow, Worker
from elevenlabs_toolkit.models import ScriptMode


def transcript(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "language_code": "uz",
                "text": "Salom dunyo.",
                "words": [
                    {"text": "Salom", "start": 0.0, "end": 0.5, "type": "word"},
                    {"text": "dunyo.", "start": 0.6, "end": 1.1, "type": "word"},
                ],
            }
        )
    )
    return path


def test_clean_matches_cli_and_inspection_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = transcript(tmp_path / "source.json")
    settings = DesktopSettings(project_directory=tmp_path, output_script=ScriptMode.CYRILLIC)
    worker = Worker("Clean", ScriptMode.SOURCE, [source], False, settings=settings)
    messages = []
    worker.finished_result.connect(messages.append, Qt.ConnectionType.DirectConnection)
    worker.run()
    assert "failed" not in messages[0].lower()
    assert main(["clean", str(source), "--script", "cyrillic", "-o", str(tmp_path / "cli")]) == 0
    assert json.loads((tmp_path / "source.clean.json").read_text()) == json.loads(
        (tmp_path / "cli/source.clean.json").read_text()
    )
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    inspect_worker = Worker("Inspect", ScriptMode.SOURCE, [source], False, settings=settings)
    results = []
    inspect_worker.finished_result.connect(results.append, Qt.ConnectionType.DirectConnection)
    inspect_worker.run()
    assert json.loads(results[0]) == inspect_transcript(source)
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_inspection_continues_after_malformed_input(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("broken")
    good = transcript(tmp_path / "good.json")
    worker = Worker("Inspect", ScriptMode.SOURCE, [bad, good], False)
    messages = []
    worker.finished_result.connect(messages.append, Qt.ConnectionType.DirectConnection)
    worker.run()
    assert '"valid": false' in messages[0]
    assert '"valid": true' in messages[0]


def test_window_workflow_selection_and_configuration(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    source = transcript(tmp_path / "transcript.json")
    window.navigation.setCurrentRow(2)
    window.add_paths([source])
    window.navigation.setCurrentRow(3)
    assert window.start.text() == "Clean"
    assert window.paths == []
    window.add_paths([source])
    window.show_preview()
    assert window.worker.wait(5000)
    app.processEvents()
    assert "transcript.clean.json" in window.details.toPlainText()
    window.navigation.setCurrentRow(2)
    assert window.paths == [source]
    window.navigation.setCurrentRow(4)
    assert window.start.text() == "Inspect"
    window.add_paths([source])
    window.show_preview()
    assert window.worker.wait(5000)
    app.processEvents()
    assert "read-only" in window.details.toPlainText()
    window.navigation.setCurrentRow(5)
    assert '"available_profiles"' in window.details.toPlainText()
    assert not window.file_list.isVisible()
    window.close()
    app.processEvents()


def test_combined_export_cancellation_does_not_publish_partial_file(tmp_path: Path) -> None:
    from elevenlabs_toolkit.application import execute_export
    from elevenlabs_toolkit.application.desktop import prepare_files
    from elevenlabs_toolkit.models import ArtifactFormat, ArtifactStatus, ConflictPolicy

    sources = (transcript(tmp_path / "one.json"), transcript(tmp_path / "two.json"))
    (item,) = prepare_files(
        "Export",
        sources,
        ScriptMode.SOURCE,
        ConflictPolicy.SKIP,
        (ArtifactFormat.COMBINED_TXT,),
        DesktopSettings(project_directory=tmp_path),
    )
    finished = []
    result = execute_export(
        item.plan,
        item.export,
        policy=ConflictPolicy.SKIP,
        should_stop=lambda: bool(finished),
        source_finished=finished.append,
    )
    assert result.artifacts[-1].status is ArtifactStatus.SKIPPED
    assert len(finished) == 1
    assert not (tmp_path / "combined.txt").exists()


def test_folder_filter_and_custom_output_preview(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.navigation.setCurrentRow(2)
    transcript(tmp_path / "include.json")
    transcript(tmp_path / "exclude.json")
    nested = tmp_path / "nested"
    nested.mkdir()
    transcript(nested / "include-nested.json")
    window.add_paths([tmp_path])
    window.filter_kind.setCurrentIndex(1)
    window.pattern.setText("include*.json")
    window.recursive.setChecked(False)
    output = tmp_path / "out"
    output.mkdir()
    (output / "include.srt").write_text("existing")
    window.output_directory.setText(str(output))
    window.conflict.setCurrentIndex(3)
    window.show_preview()
    assert window.worker.wait(5000)
    app.processEvents()
    preview = window.details.toPlainText()
    assert str(output / "include (2).srt") in preview
    assert "exclude.json" not in preview
    assert "include-nested.json" not in preview
    assert not (output / "include (2).srt").exists()
    window.close()
    app.processEvents()


def test_stop_on_error_does_not_process_later_sources(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("broken")
    good = transcript(tmp_path / "good.json")
    worker = Worker(
        "Export",
        ScriptMode.SOURCE,
        [bad, good],
        False,
        settings=DesktopSettings(project_directory=tmp_path, fail_fast=True),
    )
    messages = []
    worker.finished_result.connect(messages.append, Qt.ConnectionType.DirectConnection)
    worker.run()
    assert "Stopped after the first failure" in messages[0]
    assert not good.with_suffix(".srt").exists()


def test_cancel_background_discovery_before_processing(tmp_path: Path, monkeypatch) -> None:
    from threading import Event

    from PySide6.QtCore import QThread

    from elevenlabs_toolkit import gui
    from elevenlabs_toolkit.files.discovery import DiscoveryCancelled

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.navigation.setCurrentRow(2)
    source = transcript(tmp_path / "input.json")
    window.add_paths([tmp_path])
    scanning = Event()
    release = Event()
    threads = []

    def scan(*args, should_stop, **kwargs):
        threads.append(QThread.currentThread())
        scanning.set()
        assert release.wait(5)
        if should_stop():
            raise DiscoveryCancelled("Input scanning cancelled")
        return (source,)

    monkeypatch.setattr(gui, "discover_inputs", scan)
    window.start_job()
    try:
        assert scanning.wait(5)
        assert threads == [window.worker]
        assert not window.start.isEnabled()
        window.stop_job()
    finally:
        release.set()
        assert window.worker.wait(5000)
        app.processEvents()
    assert "Input scanning cancelled" in window.details.toPlainText()
    assert window.start.isEnabled()
    assert not source.with_suffix(".srt").exists()
    window.close()


def test_json_preview_and_results_are_machine_readable(tmp_path: Path) -> None:
    source = transcript(tmp_path / "input.json")
    for preview_only in (True, False):
        worker = Worker(
            "Export",
            ScriptMode.SOURCE,
            [source],
            False,
            settings=DesktopSettings(project_directory=tmp_path),
            preview_only=preview_only,
            report_mode="JSON",
        )
        messages = []
        worker.finished_result.connect(messages.append, Qt.ConnectionType.DirectConnection)
        worker.run()
        report = json.loads(messages[0])
        target = source.with_suffix(".srt")
        if preview_only:
            assert report["artifacts"][0]["target"] == str(target)
            assert not target.exists()
        else:
            assert report["results"][0]["artifacts"][0]["status"] == "written"
            assert target.exists()


@pytest.mark.parametrize("approved", [False, True])
def test_review_uses_prepared_plan_and_requires_confirmation(tmp_path: Path, approved: bool) -> None:
    source = transcript(tmp_path / "input.json")
    worker = Worker(
        "Export",
        ScriptMode.SOURCE,
        [source],
        False,
        settings=DesktopSettings(project_directory=tmp_path),
        review_before_running=True,
    )
    reviews = []
    target = source.with_suffix(".srt")

    def answer(preview: str) -> None:
        reviews.append((preview, target.exists()))
        worker.answer_review(approved)

    worker.review_requested.connect(answer, Qt.ConnectionType.DirectConnection)
    worker.start()
    assert worker.wait(5000)
    assert len(reviews) == 1
    assert str(target) in reviews[0][0]
    assert reviews[0][1] is False
    assert target.exists() is approved


def test_desktop_transcribe_then_export_keeps_json_only_default(tmp_path: Path, monkeypatch) -> None:
    from PySide6.QtCore import QMimeData, QPointF, QUrl
    from PySide6.QtGui import QDropEvent

    from elevenlabs_toolkit import providers

    calls = []

    class Provider:
        def __init__(self, *, env_file: Path) -> None:
            self.env_file = env_file

        def transcribe(self, path, options):
            calls.append((path, options.language_code, self.env_file))
            return {
                "text": "Salom",
                "language_code": "uz",
                "words": [
                    {
                        "type": "word",
                        "text": "Salom",
                        "start": 0,
                        "end": 1,
                        "characters": [{"text": "S", "start": 0, "end": 0.2}],
                    }
                ],
            }

    monkeypatch.setattr(providers, "ElevenLabsProvider", Provider)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.project_directory = tmp_path
    source = tmp_path / "audio.wav"
    source.write_bytes(b"audio")
    data = QMimeData()
    data.setUrls([QUrl.fromLocalFile(str(source))])
    event = QDropEvent(
        QPointF(),
        Qt.DropAction.CopyAction,
        data,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    window.file_list.dropEvent(event)
    assert window.paths == [source]
    window.transcription_options.language.setText("uz")
    window.show_preview()
    assert window.worker.wait(5000)
    app.processEvents()
    assert "Planned API requests: 1" in window.details.toPlainText()
    assert not calls

    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    saved = tmp_path / "audio.wav.transcript.json"
    assert calls == [(source, "uz", tmp_path / ".env")]
    assert set(tmp_path.iterdir()) == {source, saved}
    window.navigation.setCurrentRow(2)
    window.add_paths([saved])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert "Salom" in saved.with_suffix(".srt").read_text()
    assert len(calls) == 1
    window.close()


def test_workflow_controls_do_not_leak_between_workflows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.navigation.setCurrentRow(2)
    window.output_directory.setText(str(tmp_path))
    window.conflict.setCurrentIndex(1)
    window.filter_kind.setCurrentIndex(2)
    window.pattern.setText("clip")
    window.navigation.setCurrentRow(0)
    assert not window.output_directory.text()
    assert window.conflict.currentData() == "skip"
    assert window.filter_kind.currentIndex() == 0
    window.navigation.setCurrentRow(2)
    assert window.output_directory.text() == str(tmp_path)
    assert window.conflict.currentData() == "replace"
    assert window.pattern.text() == "clip"
    window.close()
    app.processEvents()
