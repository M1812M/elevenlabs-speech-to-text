import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from elevenlabs_toolkit.application.desktop import DesktopSettings
from elevenlabs_toolkit.application.inspection import inspect_transcript
from elevenlabs_toolkit.cli.main import main
from elevenlabs_toolkit.gui import MainWindow, Worker
from elevenlabs_toolkit.models import ArtifactFormat, ConflictPolicy, InputSpec, ScriptMode


@pytest.mark.parametrize("change", ["input", "script", "output"])
def test_preview_is_marked_outdated_after_changes(tmp_path: Path, monkeypatch, change: str) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    source = tmp_path / "first.txt"
    source.write_text("Salom", encoding="utf-8")
    window.navigation.setCurrentRow(1)
    window.output_subfolder.setChecked(False)
    window.add_paths([source])
    window.show_preview()
    assert window.worker.wait(5000)
    app.processEvents()
    if change == "input":
        other = tmp_path / "other.txt"
        other.write_text("Salom", encoding="utf-8")
        window.add_paths([other])
    elif change == "script":
        window.mode.setCurrentText("Latin")
    else:
        window.output_subfolder.setChecked(True)
    assert "out of date" in window.batch_table.item(0, 2).text().lower()
    assert "preview again" in window.details.toPlainText().lower()
    window.close()


def test_editing_profile_name_marks_export_preview_outdated(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.navigation.setCurrentRow(2)
    window.add_paths([transcript(tmp_path / "input.json")])
    window.show_preview()
    assert window.worker.wait(5000)
    app.processEvents()
    assert window.batch_table.item(0, 2).text() == "Pending"
    window.export_text.profile.setEditText("another-profile")
    assert "out of date" in window.batch_table.item(0, 2).text().lower()
    assert "preview again" in window.details.toPlainText().lower()
    window.close()


@pytest.mark.parametrize("change", ["project", "preset", "reset"])
def test_other_workflow_previews_become_stale_after_global_settings_change(
    tmp_path: Path, monkeypatch, change: str
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings)
    window.save_preset("Defaults")
    source = transcript(tmp_path / "input.json")
    window.navigation.setCurrentRow(2)
    window.add_paths([source])
    window.show_preview()
    assert window.worker.wait(5000)
    app.processEvents()
    window.navigation.setCurrentRow(5)
    if change == "project":
        monkeypatch.setattr("elevenlabs_toolkit.gui.QFileDialog.getExistingDirectory", lambda *args: str(tmp_path))
        window.select_project()
    elif change == "preset":
        window.load_preset("Defaults")
    else:
        window.reset_preferences()
    window.navigation.setCurrentRow(2)
    assert "out of date" in window.batch_table.item(0, 2).text().lower()
    assert "preview again" in window.details.toPlainText().lower()
    window.close()


@pytest.mark.parametrize(("kind", "invalid", "valid"), [(1, "../*.srt", "*.srt"), (2, "[", ".*srt")])
def test_invalid_folder_filter_has_inline_error_before_start(tmp_path: Path, monkeypatch, kind, invalid, valid) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.navigation.setCurrentRow(1)
    window.add_paths([tmp_path])
    window.filter_kind.setCurrentIndex(kind)
    window.pattern.setText(invalid)
    assert window.pattern_error.text()
    window.start_job()
    assert "Preflight failed" in window.details.toPlainText()
    assert window.start.isEnabled()
    window.pattern.setText(valid)
    assert not window.pattern_error.text()
    window.close()
    app.processEvents()


@pytest.mark.parametrize("preview", [False, True])
def test_json_mode_reports_inline_preflight_errors_as_json(tmp_path: Path, monkeypatch, preview: bool) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "input.txt"
    source.write_text("Salom", encoding="utf-8")
    window = MainWindow()
    window.navigation.setCurrentRow(1)
    window.add_paths([source])
    window.report_mode.setCurrentText("JSON")
    window.filter_kind.setCurrentIndex(2)
    window.pattern.setText("[")
    window.show_preview() if preview else window.start_job()
    app.processEvents()
    result = json.loads(window.details.toPlainText())
    assert result["status"] == "error"
    assert "regex" in result["message"].lower()
    assert window.pattern_error.text()
    assert source.read_text(encoding="utf-8") == "Salom"
    assert not (tmp_path / "Output").exists()
    window.close()


def test_about_dialog_displays_same_version_as_cli(monkeypatch, capsys) -> None:
    from PySide6.QtWidgets import QMessageBox

    from elevenlabs_toolkit.cli.main import build_parser

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    with pytest.raises(SystemExit) as result:
        build_parser().parse_args(["--version"])
    assert result.value.code == 0
    cli_version = capsys.readouterr().out.strip().rsplit(" ", 1)[-1]
    shown = []
    monkeypatch.setattr(QMessageBox, "about", lambda parent, title, message: shown.append((title, message)))
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show_about()
    assert shown == [("About ElevenLabs Toolkit", f"ElevenLabs Toolkit {cli_version}")]
    window.close()
    app.processEvents()


def test_batch_table_accepts_dropped_files_and_folders(tmp_path: Path, monkeypatch) -> None:
    from PySide6.QtCore import QMimeData, QPointF, QUrl
    from PySide6.QtGui import QDropEvent

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    source = tmp_path / "input.txt"
    source.write_text("Salom", encoding="utf-8")
    folder = tmp_path / "folder"
    folder.mkdir()
    data = QMimeData()
    data.setUrls([QUrl.fromLocalFile(str(source)), QUrl.fromLocalFile(str(folder))])
    event = QDropEvent(
        QPointF(), Qt.DropAction.CopyAction, data, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    )
    window.batch_table.dropEvent(event)
    assert window.paths == [source, folder]
    assert event.isAccepted()
    window.close()
    app.processEvents()


@pytest.mark.parametrize("target", ["file_list", "batch_table"])
def test_dropping_audio_and_its_folder_transcribes_only_once(tmp_path: Path, monkeypatch, target: str) -> None:
    from PySide6.QtCore import QMimeData, QPointF, QUrl
    from PySide6.QtGui import QDropEvent

    from elevenlabs_toolkit import providers

    calls = []

    class Provider:
        def __init__(self, *, env_file: Path) -> None:
            pass

        def transcribe(self, path, options):
            calls.append(path)
            return {
                "text": "Salom",
                "words": [
                    {
                        "text": "Salom",
                        "type": "word",
                        "start": 0,
                        "end": 1,
                        "characters": [{"text": "S", "start": 0, "end": 0.2}],
                    }
                ],
            }

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(providers, "ElevenLabsProvider", Provider)
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "input.flac"
    source.write_bytes(b"audio")
    window = MainWindow()
    window.project_directory = tmp_path
    data = QMimeData()
    data.setUrls([QUrl.fromLocalFile(str(source)), QUrl.fromLocalFile(str(tmp_path))])
    event = QDropEvent(
        QPointF(), Qt.DropAction.CopyAction, data, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    )
    getattr(window, target).dropEvent(event)
    assert event.isAccepted()
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert calls == [source]
    assert (tmp_path / "Output" / "input.flac.transcript.json").is_file()
    assert source.read_bytes() == b"audio"
    window.close()


def test_batch_table_rejects_drop_while_job_is_running(tmp_path: Path, monkeypatch) -> None:
    from PySide6.QtCore import QMimeData, QPointF, QUrl
    from PySide6.QtGui import QDropEvent

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    source = tmp_path / "input.txt"
    source.write_text("Salom", encoding="utf-8")
    data = QMimeData()
    data.setUrls([QUrl.fromLocalFile(str(source))])
    event = QDropEvent(
        QPointF(), Qt.DropAction.CopyAction, data, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    )
    window.set_running(True)
    assert not window.batch_table.isEnabled()
    window.batch_table.dropEvent(event)
    assert window.paths == []
    assert not event.isAccepted()
    window.set_running(False)
    window.close()
    app.processEvents()


def test_batch_areas_are_labelled_and_hidden_on_configuration(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    assert "drop" in window.input_label.text().lower()
    assert "batch" in window.batch_label.text().lower()
    assert window.batch_table.isVisible()
    window.navigation.setCurrentRow(5)
    app.processEvents()
    assert not window.file_list.isVisible()
    assert not window.batch_table.isVisible()
    assert not window.input_label.isVisible()
    assert not window.batch_label.isVisible()
    assert not window.retry_button.isVisible()
    window.close()


def test_transcription_charge_notice_is_visible_on_initial_screen(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    assert window.formats.isVisible()
    assert "charges" in window.formats.text().lower()
    initial_notice = window.formats.text()
    window.navigation.setCurrentRow(1)
    window.navigation.setCurrentRow(0)
    assert window.formats.text() == initial_notice
    window.close()


def test_multiple_inputs_can_be_selected_and_removed_with_keyboard(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    paths = [tmp_path / f"{number}.txt" for number in range(3)]
    for path in paths:
        path.write_text("Salom", encoding="utf-8")
    window.navigation.setCurrentRow(1)
    window.add_paths(paths)
    window.show()
    window.file_list.setFocus()
    window.file_list.setCurrentRow(0)
    QTest.keyClick(window.file_list, Qt.Key.Key_Down, Qt.KeyboardModifier.ShiftModifier)
    app.processEvents()
    assert len(window.file_list.selectedItems()) == 2
    window.remove_selected()
    assert window.paths == [paths[2]]
    assert window.file_list.count() == 1
    window.close()


@pytest.mark.parametrize("recursive", [False, True])
@pytest.mark.parametrize(("kind", "pattern"), [("glob", "*.flac"), ("regex", r"\.flac$")])
def test_transcription_folder_filter_matches_cli_without_paid_calls(
    tmp_path: Path, monkeypatch, capsys, recursive: bool, kind: str, pattern: str
) -> None:
    from elevenlabs_toolkit import providers
    from elevenlabs_toolkit.cli import transcribe

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    calls = []

    class Provider:
        def __init__(self, *, env_file: Path) -> None:
            pass

        def transcribe(self, path, options):
            calls.append(path)
            return {
                "text": "Salom",
                "words": [
                    {
                        "text": "Salom",
                        "start": 0,
                        "end": 1,
                        "type": "word",
                        "characters": [{"text": "S", "start": 0, "end": 0.2}],
                    }
                ],
            }

    monkeypatch.setattr(providers, "ElevenLabsProvider", Provider)
    monkeypatch.setattr(transcribe, "ElevenLabsProvider", Provider)
    inputs = tmp_path / "inputs"
    nested = inputs / "nested"
    nested.mkdir(parents=True)
    for path in (inputs / "first.flac", inputs / "ignored.wav", nested / "second.flac"):
        path.write_bytes(b"audio")
    cli_dir, gui_dir = tmp_path / "cli", tmp_path / "gui"
    args = ["transcribe", str(inputs), f"--{kind}", pattern, "-o", str(cli_dir)]
    if not recursive:
        args.append("--no-recursive")
    assert main(args) == 0
    capsys.readouterr()
    cli_calls = list(calls)
    calls.clear()
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.project_directory = tmp_path
    window.output_directory.setText(str(gui_dir))
    window.filter_kind.setCurrentIndex(1 if kind == "glob" else 2)
    window.pattern.setText(pattern)
    window.recursive.setChecked(recursive)
    window.add_paths([inputs])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    expected = [inputs / "first.flac"] + ([nested / "second.flac"] if recursive else [])
    assert calls == cli_calls == expected
    for source in expected:
        relative = source.relative_to(inputs)
        assert (gui_dir / relative.parent / f"{source.name}.transcript.json").read_bytes() == (
            cli_dir / relative.parent / f"{source.stem}.json"
        ).read_bytes()
    assert all(
        path.read_bytes() == b"audio"
        for path in (inputs / "first.flac", inputs / "ignored.wav", nested / "second.flac")
    )
    window.close()


@pytest.mark.parametrize("fail_fast", [False, True])
@pytest.mark.parametrize("combined", [False, True])
def test_gui_and_cli_batch_error_continuation_match(
    tmp_path: Path, monkeypatch, capsys, fail_fast: bool, combined: bool
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    app = QApplication.instance() or QApplication([])
    sources = tmp_path / "sources"
    sources.mkdir()
    bad = sources / "a.json"
    bad.write_text("broken", encoding="utf-8")
    good = transcript(sources / "b.json")
    before = {path: path.read_bytes() for path in sources.iterdir()}
    cli_dir, gui_dir = tmp_path / "cli", tmp_path / "gui"
    args = ["export", str(sources), "-o", str(cli_dir)]
    if combined:
        args.extend(["--format", "srt", "--format", "combined-txt"])
    if fail_fast:
        args.append("--fail-fast")
    assert main(args) != 0
    capsys.readouterr()
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(2)
    window.output_subfolder.setChecked(False)
    window.output_directory.setText(str(gui_dir))
    window.fail_fast.setChecked(fail_fast)
    window.export_formats[ArtifactFormat.COMBINED_TXT].setChecked(combined)
    window.add_paths([sources])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert window.batch_table.item(0, 2).text() == "failed"
    assert (gui_dir / "b.srt").exists() == (cli_dir / "b.srt").exists() == (not fail_fast)
    if not fail_fast:
        assert (gui_dir / "b.srt").read_bytes() == (cli_dir / "b.srt").read_bytes()
    if combined:
        assert not (gui_dir / "combined.txt").exists()
        assert not (cli_dir / "combined.txt").exists()
    assert {path: path.read_bytes() for path in sources.iterdir()} == before
    assert good.exists()
    window.close()


def test_retry_combined_export_after_repairing_invalid_source(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    bad = tmp_path / "bad.json"
    bad.write_text("broken", encoding="utf-8")
    good = transcript(tmp_path / "good.json")
    good_original = good.read_bytes()
    window = MainWindow()
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(2)
    window.output_subfolder.setChecked(False)
    window.export_formats[ArtifactFormat.SRT].setChecked(False)
    window.export_formats[ArtifactFormat.COMBINED_TXT].setChecked(True)
    window.add_paths([bad, good])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert window.retry_button.isEnabled()
    assert all(window.batch_table.item(row, 2).text() == "failed" for row in range(2))
    assert not (tmp_path / "combined.txt").exists()
    transcript(bad)
    window.retry_button.click()
    assert window.worker.wait(5000)
    app.processEvents()
    output = (tmp_path / "combined.txt").read_text(encoding="utf-8")
    assert "bad.json" in output and "good.json" in output
    assert all(window.batch_table.item(row, 2).text() == "ok" for row in range(2))
    assert not window.retry_button.isEnabled()
    assert good.read_bytes() == good_original
    window.close()


def test_window_retries_cancelled_combined_export(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    sources = [transcript(tmp_path / "first.json"), transcript(tmp_path / "second.json")]
    original_execute = Worker.execute_file

    def stop_after_first(worker, item):
        worker.status.connect(
            lambda text: worker.requestInterruption() if text.startswith("Combined text: processed") else None,
            Qt.ConnectionType.DirectConnection,
        )
        return original_execute(worker, item)

    monkeypatch.setattr(Worker, "execute_file", stop_after_first)
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(2)
    window.output_subfolder.setChecked(False)
    window.export_formats[ArtifactFormat.SRT].setChecked(False)
    window.export_formats[ArtifactFormat.COMBINED_TXT].setChecked(True)
    window.add_paths(sources)
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert not (tmp_path / "combined.txt").exists()
    assert window.retry_button.isEnabled()
    assert all(window.batch_table.item(row, 2).text() == "cancelled" for row in range(window.batch_table.rowCount()))
    monkeypatch.setattr(Worker, "execute_file", original_execute)
    window.retry_button.click()
    assert window.worker.wait(5000)
    app.processEvents()
    output = (tmp_path / "combined.txt").read_text(encoding="utf-8")
    assert "first.json" in output and "second.json" in output
    assert not window.retry_button.isEnabled()
    window.close()


def test_retry_preserves_successful_outputs_after_partial_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    source = transcript(tmp_path / "input.json")
    original_execute = Worker.execute_file

    def partial_failure(worker, item):
        (tmp_path / "input.srt").write_text("already written", encoding="utf-8")
        raise OSError("simulated second-output failure")

    monkeypatch.setattr(Worker, "execute_file", partial_failure)
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(2)
    window.output_subfolder.setChecked(False)
    window.export_formats[ArtifactFormat.TXT].setChecked(True)
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert window.retry_button.isEnabled()
    monkeypatch.setattr(Worker, "execute_file", original_execute)
    window.retry_button.click()
    assert window.worker.wait(5000)
    app.processEvents()
    assert (tmp_path / "input.srt").read_text(encoding="utf-8") == "already written"
    assert "Salom" in (tmp_path / "input.txt").read_text(encoding="utf-8")
    assert window.batch_table.item(0, 2).text() == "partial"
    window.close()


@pytest.mark.parametrize("change_target", [False, True])
def test_retry_export_replaces_failed_target_but_preserves_successful_output(
    tmp_path: Path, monkeypatch, change_target: bool
) -> None:
    from elevenlabs_toolkit.application import exporter

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    source = transcript(tmp_path / "input.json")
    output = tmp_path / "input.txt"
    output.write_text("old output", encoding="utf-8")
    original_render = exporter.render_artifact

    def fail_text(format_, *args, **kwargs):
        if format_ is ArtifactFormat.TXT:
            raise OSError("simulated text rendering failure")
        return original_render(format_, *args, **kwargs)

    monkeypatch.setattr(exporter, "render_artifact", fail_text)
    window = MainWindow()
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(2)
    window.output_subfolder.setChecked(False)
    window.conflict.setCurrentIndex(1)
    window.export_formats[ArtifactFormat.TXT].setChecked(True)
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert window.retry_button.isEnabled()
    srt = tmp_path / "input.srt"
    successful_output = srt.read_bytes()
    assert output.read_text(encoding="utf-8") == "old output"
    monkeypatch.setattr(exporter, "render_artifact", original_render)
    if change_target:
        output.write_text("changed by user", encoding="utf-8")
    window.retry_button.click()
    assert window.worker.wait(5000)
    app.processEvents()
    assert (
        output.read_text(encoding="utf-8") == "changed by user"
        if change_target
        else "Salom" in output.read_text(encoding="utf-8")
    )
    assert srt.read_bytes() == successful_output
    window.close()


@pytest.mark.parametrize("stop_reason", ["failure", "cancel"])
def test_retry_includes_files_not_run_in_original_batch(tmp_path: Path, monkeypatch, stop_reason: str) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    sources = [tmp_path / "first.txt", tmp_path / "second.txt"]
    for source in sources:
        source.write_text("Salom", encoding="utf-8")
    original_execute = Worker.execute_file

    def execute_first(worker, item):
        if stop_reason == "failure":
            raise OSError("simulated failure")
        result = original_execute(worker, item)
        worker.requestInterruption()
        return result

    monkeypatch.setattr(Worker, "execute_file", execute_first)
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.navigation.setCurrentRow(1)
    window.output_subfolder.setChecked(False)
    window.fail_fast.setChecked(stop_reason == "failure")
    window.add_paths(sources)
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert window.batch_table.item(1, 2).text().startswith("Not run")
    assert window.retry_button.isEnabled()
    monkeypatch.setattr(Worker, "execute_file", original_execute)
    window.retry_button.click()
    assert window.worker.wait(5000)
    app.processEvents()
    assert all((tmp_path / f"{source.stem} CYR.txt").is_file() for source in sources)
    assert not window.retry_button.isEnabled()
    assert window.batch_table.rowCount() == 2
    assert {window.batch_table.item(row, 0).text() for row in range(2)} == {str(source) for source in sources}
    assert all(window.batch_table.item(row, 2).text() == "written" for row in range(2))
    report = tmp_path / "batch-report.json"
    monkeypatch.setattr("elevenlabs_toolkit.gui.QFileDialog.getSaveFileName", lambda *args: (str(report), ""))
    window.save_batch_report()
    assert {row["input"] for row in json.loads(report.read_text(encoding="utf-8"))["files"]} == {
        str(source) for source in sources
    }
    window.close()


@pytest.mark.parametrize("change_target", [False, True])
def test_retry_unstarted_replace_respects_original_target(tmp_path: Path, monkeypatch, change_target: bool) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    first, second = tmp_path / "a.txt", tmp_path / "b.txt"
    first.write_bytes(b"\xff")
    second.write_text("Salom", encoding="utf-8")
    target = tmp_path / "b CYR.txt"
    target.write_text("old output", encoding="utf-8")
    window = MainWindow()
    window.navigation.setCurrentRow(1)
    window.output_subfolder.setChecked(False)
    window.conflict.setCurrentIndex(1)
    window.fail_fast.setChecked(True)
    window.add_paths([first, second])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert window.batch_table.item(1, 2).text().startswith("Not run")
    first.write_text("Salom", encoding="utf-8")
    if change_target:
        target.write_text("changed after stop", encoding="utf-8")
    window.retry_failed()
    assert window.worker.wait(5000)
    app.processEvents()
    assert target.read_text(encoding="utf-8") == ("changed after stop" if change_target else "Салом")
    window.close()


@pytest.mark.parametrize("change_target", [False, True])
def test_retry_unstarted_export_respects_original_target(tmp_path: Path, monkeypatch, change_target: bool) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    first = tmp_path / "a.json"
    first.write_text("broken", encoding="utf-8")
    second = transcript(tmp_path / "b.json")
    target = tmp_path / "b.srt"
    target.write_text("old output", encoding="utf-8")
    window = MainWindow()
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(2)
    window.output_subfolder.setChecked(False)
    window.conflict.setCurrentIndex(1)
    window.fail_fast.setChecked(True)
    window.add_paths([first, second])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert window.batch_table.item(1, 2).text().startswith("Not run")
    transcript(first)
    if change_target:
        target.write_text("changed after stop", encoding="utf-8")
    window.retry_failed()
    assert window.worker.wait(5000)
    app.processEvents()
    if change_target:
        assert target.read_text(encoding="utf-8") == "changed after stop"
    else:
        assert "Salom" in target.read_text(encoding="utf-8")
    window.close()


def test_format_controls_explain_outputs_and_provider_requests(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    resolve = window.export_formats[ArtifactFormat.RESOLVE_EDL]
    assert "Resolve" in resolve.text()
    assert "marker" in resolve.toolTip().lower()
    mini = window.export_formats[ArtifactFormat.SRT_MINI]
    assert "subtitle" in mini.toolTip().lower()
    provider = window.transcription_options.remote_formats["pdf"]
    assert "ElevenLabs" in provider.toolTip()
    assert "request" in provider.toolTip().lower()
    window.close()
    app.processEvents()


@pytest.mark.parametrize("output_appears", [True, False])
def test_transcription_rechecks_outputs_after_planning_without_unexpected_provider_calls(
    tmp_path: Path, monkeypatch, output_appears: bool
) -> None:
    from elevenlabs_toolkit.application.desktop import prepare_files

    source = tmp_path / "audio.flac"
    source.write_bytes(b"audio")
    target = tmp_path / "audio.flac.transcript.json"
    if not output_appears:
        target.write_text("existing", encoding="utf-8")
    settings = DesktopSettings(project_directory=tmp_path)
    prepared = prepare_files("Transcribe", (source,), ScriptMode.SOURCE, ConflictPolicy.SKIP, (), settings)
    assert prepared[0].plan.api_requests == (1 if output_appears else 0)
    if output_appears:
        target.write_text("existing", encoding="utf-8")
    else:
        target.unlink()

    def unexpected_provider(*args, **kwargs):
        raise AssertionError("Provider should not be constructed")

    monkeypatch.setattr("elevenlabs_toolkit.providers.ElevenLabsProvider", unexpected_provider)
    worker = Worker("Transcribe", ScriptMode.SOURCE, [], False, settings=settings, retry_plan=prepared)
    results = []
    worker.file_result.connect(results.append, Qt.ConnectionType.DirectConnection)
    worker.run()
    if output_appears:
        assert results[0]["status"] == "skipped"
        assert target.read_text(encoding="utf-8") == "existing"
    else:
        assert results[0]["status"] == "failed"
        assert "plan" in results[0]["error"].lower()
        assert not target.exists()
    assert source.read_bytes() == b"audio"


def test_reviewed_transcription_retry_replans_missing_skipped_output(tmp_path: Path, monkeypatch) -> None:
    from elevenlabs_toolkit import providers
    from elevenlabs_toolkit.application.desktop import prepare_files

    calls = []

    class Provider:
        def __init__(self, *, env_file: Path) -> None:
            pass

        def transcribe(self, path, options):
            calls.append(path)
            return {
                "text": "Salom",
                "words": [
                    {
                        "text": "Salom",
                        "type": "word",
                        "start": 0,
                        "end": 1,
                        "characters": [{"text": "S", "start": 0, "end": 0.2}],
                    }
                ],
            }

    source = tmp_path / "audio.flac"
    source.write_bytes(b"audio")
    target = tmp_path / "audio.flac.transcript.json"
    target.write_text("existing", encoding="utf-8")
    settings = DesktopSettings(project_directory=tmp_path)
    prepared = prepare_files("Transcribe", (source,), ScriptMode.SOURCE, ConflictPolicy.SKIP, (), settings)
    assert prepared[0].plan.api_requests == 0
    target.unlink()
    monkeypatch.setattr(providers, "ElevenLabsProvider", Provider)
    worker = Worker(
        "Transcribe", ScriptMode.SOURCE, [], False, settings=settings, retry_plan=prepared, review_before_running=True
    )
    reviews = []

    def approve(preview):
        reviews.append(preview)
        worker.answer_review(True)

    worker.review_requested.connect(approve, Qt.ConnectionType.DirectConnection)
    worker.run()
    assert "Planned API requests: 1" in reviews[0]
    assert calls == [source]
    assert json.loads(target.read_text(encoding="utf-8"))["text"] == "Salom"
    assert source.read_bytes() == b"audio"


@pytest.mark.parametrize("fail_fast", [False, True])
def test_transcription_failure_continuation_matches_cli_without_paid_calls(
    tmp_path: Path, monkeypatch, capsys, fail_fast: bool
) -> None:
    from elevenlabs_toolkit import providers
    from elevenlabs_toolkit.cli import transcribe
    from elevenlabs_toolkit.providers import ProviderError

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    calls = []

    class Provider:
        def __init__(self, *, env_file: Path) -> None:
            pass

        def transcribe(self, path, options):
            calls.append(path.name)
            if path.name == "a.wav":
                raise ProviderError("simulated provider failure")
            return {
                "text": "Salom",
                "words": [
                    {
                        "text": "Salom",
                        "type": "word",
                        "start": 0,
                        "end": 1,
                        "characters": [{"text": "S", "start": 0, "end": 0.2}],
                    }
                ],
            }

    monkeypatch.setattr(providers, "ElevenLabsProvider", Provider)
    monkeypatch.setattr(transcribe, "ElevenLabsProvider", Provider)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in ("a.wav", "b.wav"):
        (inputs / name).write_bytes(b"audio")
    cli_dir, gui_dir = tmp_path / "cli", tmp_path / "gui"
    args = ["transcribe", str(inputs), "-o", str(cli_dir)]
    if fail_fast:
        args.append("--fail-fast")
    assert main(args) != 0
    capsys.readouterr()
    cli_calls = list(calls)
    calls.clear()
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.project_directory = tmp_path
    window.output_directory.setText(str(gui_dir))
    window.fail_fast.setChecked(fail_fast)
    window.add_paths([inputs])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert calls == cli_calls == (["a.wav"] if fail_fast else ["a.wav", "b.wav"])
    assert (gui_dir / "b.wav.transcript.json").exists() == (cli_dir / "b.json").exists() == (not fail_fast)
    assert window.batch_table.item(0, 2).text() == "failed"
    if fail_fast:
        assert "Not run" in window.batch_table.item(1, 2).text()
        assert window.retry_button.isEnabled()
    assert all((inputs / name).read_bytes() == b"audio" for name in ("a.wav", "b.wav"))
    window.close()


@pytest.mark.parametrize("change_target", [False, True])
def test_transcription_retry_replaces_failed_existing_output(tmp_path: Path, monkeypatch, change_target: bool) -> None:
    from elevenlabs_toolkit import providers

    calls = []

    class Provider:
        def __init__(self, *, env_file: Path) -> None:
            pass

        def transcribe(self, path, options):
            calls.append(path)
            if len(calls) == 1:
                raise OSError("temporary provider failure")
            return {
                "text": "Salom",
                "words": [
                    {
                        "text": "Salom",
                        "type": "word",
                        "start": 0,
                        "end": 1,
                        "characters": [{"text": "S", "start": 0, "end": 0.2}],
                    }
                ],
            }

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(providers, "ElevenLabsProvider", Provider)
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "audio.flac"
    source.write_bytes(b"audio")
    output = tmp_path / "audio.flac.transcript.json"
    output.write_text("old output", encoding="utf-8")
    window = MainWindow()
    window.project_directory = tmp_path
    window.output_subfolder.setChecked(False)
    window.conflict.setCurrentIndex(1)
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert window.retry_button.isEnabled()
    if change_target:
        output.write_text("changed by user", encoding="utf-8")
    reviews = []

    def approve_retry(preview):
        reviews.append(preview)
        window.worker.answer_review(True)

    monkeypatch.setattr(window, "review_plan", approve_retry)
    window.retry_button.click()
    for _ in range(500):
        app.processEvents()
        if not window.worker.isRunning():
            break
        QTest.qWait(10)
    assert window.worker.wait(1000)
    app.processEvents()
    assert len(calls) == (1 if change_target else 2)
    assert f"Planned API requests: {0 if change_target else 1}" in reviews[0]
    assert ("Skip existing" if change_target else "Replace") in reviews[0]
    if change_target:
        assert output.read_text(encoding="utf-8") == "changed by user"
    else:
        assert window.batch_table.item(0, 2).text() == "ok", window.details.toPlainText()
        assert json.loads(output.read_text(encoding="utf-8"))["text"] == "Salom"
    window.close()


def test_transcription_retry_stops_if_outputs_change_during_review(tmp_path: Path, monkeypatch) -> None:
    from elevenlabs_toolkit import providers

    calls = []

    class Provider:
        def __init__(self, *, env_file: Path) -> None:
            pass

        def transcribe(self, path, options):
            calls.append(path)
            raise OSError("temporary provider failure")

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(providers, "ElevenLabsProvider", Provider)
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "audio.flac"
    source.write_bytes(b"audio")
    target = tmp_path / "audio.flac.transcript.json"
    target.write_text("old output", encoding="utf-8")
    window = MainWindow()
    window.project_directory = tmp_path
    window.output_subfolder.setChecked(False)
    window.conflict.setCurrentIndex(1)
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    target.write_text("changed output", encoding="utf-8")

    def change_during_review(preview):
        assert "Planned API requests: 0" in preview
        target.unlink()
        window.worker.answer_review(True)

    monkeypatch.setattr(window, "review_plan", change_during_review)
    window.retry_failed()
    for _ in range(500):
        app.processEvents()
        if not window.worker.isRunning():
            break
        QTest.qWait(10)
    assert window.worker.wait(1000)
    app.processEvents()
    assert calls == [source]
    assert window.retry_button.isEnabled()
    assert "changed during review" in window.details.toPlainText()
    window.close()


@pytest.mark.parametrize("remove_when", ["before_preview", "after_review"])
def test_transcription_retry_does_not_add_request_after_review(tmp_path: Path, monkeypatch, remove_when: str) -> None:
    from elevenlabs_toolkit import providers

    calls = []

    class Provider:
        def __init__(self, *, env_file: Path) -> None:
            pass

        def transcribe(self, path, options):
            calls.append(path)
            raise OSError("temporary provider failure")

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(providers, "ElevenLabsProvider", Provider)
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "audio.flac"
    source.write_bytes(b"audio")
    target = tmp_path / "audio.flac.transcript.json"
    target.write_text("old output", encoding="utf-8")
    window = MainWindow()
    window.project_directory = tmp_path
    window.output_subfolder.setChecked(False)
    window.conflict.setCurrentIndex(1)
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    target.write_text("changed output", encoding="utf-8")
    if remove_when == "after_review":
        original_execute = Worker.execute_file

        def remove_before_execution(worker, item):
            target.unlink()
            return original_execute(worker, item)

        monkeypatch.setattr(Worker, "execute_file", remove_before_execution)
    else:
        original_needs_request = Worker.transcription_needs_request

        def remove_after_snapshot(worker, item):
            result = original_needs_request(worker, item)
            if target.exists():
                target.unlink()
            return result

        monkeypatch.setattr(Worker, "transcription_needs_request", remove_after_snapshot)
    reviews = []

    def approve(preview):
        reviews.append(preview)
        window.worker.answer_review(True)

    monkeypatch.setattr(window, "review_plan", approve)
    window.retry_failed()
    for _ in range(500):
        app.processEvents()
        if not window.worker.isRunning():
            break
        QTest.qWait(10)
    assert window.worker.wait(1000)
    app.processEvents()
    assert calls == [source]
    if remove_when == "before_preview":
        assert not reviews
        assert "changed while preparing review" in window.details.toPlainText()
    else:
        assert "Planned API requests: 0" in reviews[0]
    assert window.retry_button.isEnabled()
    window.close()


@pytest.mark.parametrize("format_name", ["srt", "pdf"])
def test_partial_transcription_retry_preserves_successful_json(tmp_path: Path, monkeypatch, format_name: str) -> None:
    import base64

    from elevenlabs_toolkit import providers
    from elevenlabs_toolkit.application import transcriber

    calls = []

    class Provider:
        def __init__(self, *, env_file: Path) -> None:
            pass

        def transcribe(self, path, options):
            calls.append(path)
            return {
                "text": f"Salom {len(calls)}",
                "words": [
                    {
                        "text": "Salom",
                        "type": "word",
                        "start": 0,
                        "end": 1,
                        "characters": [{"text": "S", "start": 0, "end": 0.2}],
                    }
                ],
                "additional_formats": [
                    {"requested_format": "pdf", "is_base64_encoded": True, "content": base64.b64encode(b"pdf").decode()}
                ]
                if format_name == "pdf"
                else [],
            }

    write_name = "atomic_write_text" if format_name == "srt" else "atomic_write_bytes"
    original_write = getattr(transcriber, write_name)

    def fail_output(path, content, policy):
        if path.suffix == f".{format_name}":
            raise OSError("simulated output write failure")
        return original_write(path, content, policy)

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(providers, "ElevenLabsProvider", Provider)
    monkeypatch.setattr(transcriber, write_name, fail_output)
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "audio.flac"
    source.write_bytes(b"audio")
    target = tmp_path / f"audio.flac.transcript.{format_name}"
    target.write_bytes(b"old output")
    window = MainWindow()
    window.project_directory = tmp_path
    window.output_subfolder.setChecked(False)
    window.conflict.setCurrentIndex(1)
    if format_name == "srt":
        window.transcription_options.local_formats[ArtifactFormat.SRT].setChecked(True)
    else:
        window.transcription_options.remote_formats["pdf"].setChecked(True)
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert window.retry_button.isEnabled()
    output = tmp_path / "audio.flac.transcript.json"
    successful_json = output.read_bytes()
    assert target.read_bytes() == b"old output"
    monkeypatch.setattr(transcriber, write_name, original_write)
    monkeypatch.setattr(window, "review_plan", lambda preview: window.worker.answer_review(True))
    window.retry_button.click()
    for _ in range(500):
        app.processEvents()
        if not window.worker.isRunning():
            break
        QTest.qWait(10)
    assert window.worker.wait(1000)
    app.processEvents()
    assert len(calls) == 2
    assert output.read_bytes() == successful_json
    assert b"Salom" in target.read_bytes() if format_name == "srt" else target.read_bytes() == b"pdf"
    window.close()


@pytest.mark.parametrize("change_target", [False, True])
def test_retry_unstarted_transcription_respects_original_target(
    tmp_path: Path, monkeypatch, change_target: bool
) -> None:
    from elevenlabs_toolkit import providers

    calls = []

    class Provider:
        def __init__(self, *, env_file: Path) -> None:
            pass

        def transcribe(self, path, options):
            calls.append(path)
            if len(calls) == 1:
                raise OSError("temporary provider failure")
            return {
                "text": "Salom",
                "words": [
                    {
                        "text": "Salom",
                        "type": "word",
                        "start": 0,
                        "end": 1,
                        "characters": [{"text": "S", "start": 0, "end": 0.2}],
                    }
                ],
            }

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(providers, "ElevenLabsProvider", Provider)
    app = QApplication.instance() or QApplication([])
    sources = [tmp_path / "a.flac", tmp_path / "b.flac"]
    for source in sources:
        source.write_bytes(b"audio")
    target = tmp_path / "b.flac.transcript.json"
    target.write_text("old output", encoding="utf-8")
    window = MainWindow()
    window.project_directory = tmp_path
    window.output_subfolder.setChecked(False)
    window.conflict.setCurrentIndex(1)
    window.fail_fast.setChecked(True)
    window.add_paths(sources)
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert calls == [sources[0]]
    if change_target:
        target.write_text("changed after stop", encoding="utf-8")
    monkeypatch.setattr(window, "review_plan", lambda preview: window.worker.answer_review(True))
    window.retry_failed()
    for _ in range(500):
        app.processEvents()
        if not window.worker.isRunning():
            break
        QTest.qWait(10)
    assert window.worker.wait(1000)
    app.processEvents()
    assert calls == ([sources[0], sources[0]] if change_target else [sources[0], sources[0], sources[1]])
    if change_target:
        assert target.read_text(encoding="utf-8") == "changed after stop"
    else:
        assert json.loads(target.read_text(encoding="utf-8"))["text"] == "Salom"
    window.close()


def test_window_review_counts_removed_skipped_output_before_retry(tmp_path: Path, monkeypatch) -> None:
    from elevenlabs_toolkit import providers

    calls = []

    class Provider:
        def __init__(self, *, env_file: Path) -> None:
            pass

        def transcribe(self, path, options):
            calls.append(path)
            if len(calls) == 1:
                raise OSError("temporary provider failure")
            return {
                "text": "Salom",
                "words": [
                    {
                        "text": "Salom",
                        "type": "word",
                        "start": 0,
                        "end": 1,
                        "characters": [{"text": "S", "start": 0, "end": 0.2}],
                    }
                ],
            }

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(providers, "ElevenLabsProvider", Provider)
    app = QApplication.instance() or QApplication([])
    sources = [tmp_path / "a.flac", tmp_path / "b.flac"]
    for source in sources:
        source.write_bytes(b"audio")
    target = tmp_path / "b.flac.transcript.json"
    target.write_text("existing", encoding="utf-8")
    window = MainWindow()
    window.project_directory = tmp_path
    window.output_subfolder.setChecked(False)
    window.conflict.setCurrentIndex(window.conflict.findData("skip"))
    window.fail_fast.setChecked(True)
    window.add_paths(sources)
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert calls == [sources[0]]
    assert window.retry_button.isEnabled()
    target.unlink()
    reviews = []

    def approve(preview):
        reviews.append(preview)
        window.worker.answer_review(True)

    monkeypatch.setattr(window, "review_plan", approve)
    window.retry_failed()
    for _ in range(500):
        app.processEvents()
        if not window.worker.isRunning():
            break
        QTest.qWait(10)
    assert window.worker.wait(1000)
    app.processEvents()
    assert "Planned API requests: 2" in reviews[0]
    assert calls == [sources[0], sources[0], sources[1]]
    assert json.loads(target.read_text(encoding="utf-8"))["text"] == "Salom"
    assert all(source.read_bytes() == b"audio" for source in sources)
    window.close()


def test_stop_during_active_transcription_finishes_current_file_and_preserves_retry(
    tmp_path: Path, monkeypatch
) -> None:
    from threading import Event

    from elevenlabs_toolkit import providers

    started, release = Event(), Event()
    calls = []

    class Provider:
        def __init__(self, *, env_file: Path) -> None:
            pass

        def transcribe(self, path, options):
            calls.append(path)
            started.set()
            assert release.wait(5)
            return {
                "text": "Salom",
                "words": [
                    {
                        "text": "Salom",
                        "type": "word",
                        "start": 0,
                        "end": 1,
                        "characters": [{"text": "S", "start": 0, "end": 0.2}],
                    }
                ],
            }

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(providers, "ElevenLabsProvider", Provider)
    app = QApplication.instance() or QApplication([])
    sources = [tmp_path / "a.flac", tmp_path / "b.flac"]
    for source in sources:
        source.write_bytes(b"audio")
    window = MainWindow()
    window.project_directory = tmp_path
    window.add_paths(sources)
    window.start_job()
    try:
        assert started.wait(5)
        window.stop_job()
    finally:
        release.set()
        assert window.worker.wait(5000)
        app.processEvents()
    assert calls == [sources[0]]
    assert (tmp_path / "Output" / "a.flac.transcript.json").is_file()
    assert not (tmp_path / "Output" / "b.flac.transcript.json").exists()
    assert "not run" in window.batch_table.item(1, 2).text().lower()
    assert window.retry_button.isEnabled()
    stopped_report = tmp_path / "stopped-report.json"
    monkeypatch.setattr("elevenlabs_toolkit.gui.QFileDialog.getSaveFileName", lambda *args: (str(stopped_report), ""))
    window.save_batch_report()
    stopped_rows = json.loads(stopped_report.read_text(encoding="utf-8"))["files"]
    assert stopped_rows[0]["input"] == str(sources[0])
    assert stopped_rows[0]["status"] == "ok"
    assert stopped_rows[1]["input"] == str(sources[1])
    assert "not run" in stopped_rows[1]["status"].lower()
    reviewed = []

    def approve(preview):
        reviewed.append(preview)
        window.worker.answer_review(True)

    monkeypatch.setattr(window, "review_plan", approve)
    window.retry_button.click()
    for _ in range(500):
        app.processEvents()
        if not window.worker.isRunning():
            break
        QTest.qWait(10)
    assert window.worker.wait(1000)
    app.processEvents()
    assert len(reviewed) == 1
    assert "Planned API requests: 1" in reviewed[0]
    assert calls == sources
    assert (tmp_path / "Output" / "b.flac.transcript.json").is_file()
    assert not window.retry_button.isEnabled()
    final_report = tmp_path / "final-report.json"
    monkeypatch.setattr("elevenlabs_toolkit.gui.QFileDialog.getSaveFileName", lambda *args: (str(final_report), ""))
    window.save_batch_report()
    final_rows = json.loads(final_report.read_text(encoding="utf-8"))["files"]
    assert [row["input"] for row in final_rows] == [str(source) for source in sources]
    assert [row["status"] for row in final_rows] == ["ok", "ok"]
    assert all(source.read_bytes() == b"audio" for source in sources)
    window.close()


def test_stop_during_request_delay_keeps_remaining_transcription_retryable(tmp_path: Path, monkeypatch) -> None:
    from elevenlabs_toolkit import providers

    calls = []

    class Provider:
        def __init__(self, *, env_file: Path) -> None:
            pass

        def transcribe(self, path, options):
            calls.append(path.name)
            return {
                "text": "Salom",
                "words": [
                    {
                        "text": "Salom",
                        "type": "word",
                        "start": 0,
                        "end": 1,
                        "characters": [{"text": "S", "start": 0, "end": 0.2}],
                    }
                ],
            }

    monkeypatch.setattr(providers, "ElevenLabsProvider", Provider)
    sources = [tmp_path / "a.wav", tmp_path / "b.wav"]
    for source in sources:
        source.write_bytes(b"audio")
    worker = Worker(
        "Transcribe",
        ScriptMode.SOURCE,
        sources,
        False,
        settings=DesktopSettings(project_directory=tmp_path, request_delay=30),
    )
    retry_items = []
    worker.failed_item.connect(retry_items.append, Qt.ConnectionType.DirectConnection)
    worker.status.connect(
        lambda message: worker.requestInterruption() if message.startswith("Waiting ") else None,
        Qt.ConnectionType.DirectConnection,
    )
    worker.start()
    assert worker.wait(5000)
    assert calls == ["a.wav"]
    assert [item.plan.sources[0] for item in retry_items] == [sources[1]]
    assert (tmp_path / "a.wav.transcript.json").is_file()
    assert not (tmp_path / "b.wav.transcript.json").exists()
    assert all(source.read_bytes() == b"audio" for source in sources)


def test_transcription_retry_settings_match_cli_without_paid_calls(tmp_path: Path, monkeypatch) -> None:
    from elevenlabs_toolkit import providers
    from elevenlabs_toolkit.cli import transcribe
    from elevenlabs_toolkit.providers import ProviderTransientError

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    app = QApplication.instance() or QApplication([])
    attempts = []

    class Provider:
        def __init__(self, *, env_file: Path) -> None:
            self.attempts = 0

        def transcribe(self, path, options):
            self.attempts += 1
            attempts.append((path, self.attempts))
            if self.attempts == 1:
                raise ProviderTransientError("temporary", retry_after_seconds=0)
            return {
                "text": "Salom",
                "words": [
                    {
                        "text": "Salom",
                        "type": "word",
                        "start": 0,
                        "end": 1,
                        "characters": [{"text": "S", "start": 0, "end": 0.2}],
                    }
                ],
            }

    monkeypatch.setattr(providers, "ElevenLabsProvider", Provider)
    monkeypatch.setattr(transcribe, "ElevenLabsProvider", Provider)
    source = tmp_path / "audio.wav"
    source.write_bytes(b"audio")
    cli_dir, gui_dir = tmp_path / "cli", tmp_path / "gui"
    assert main(["transcribe", str(source), "-o", str(cli_dir), "--retries", "1", "--retry-backoff", "0"]) == 0
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.project_directory = tmp_path
    window.transcription_options.retries.setText("1")
    window.transcription_options.backoff.setText("0")
    window.output_directory.setText(str(gui_dir))
    window.add_paths([source])
    window.show_preview()
    assert window.worker.wait(5000)
    app.processEvents()
    assert "up to 2 attempts" in window.details.toPlainText()
    assert len(attempts) == 2
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert attempts == [(source, 1), (source, 2), (source, 1), (source, 2)]
    assert (gui_dir / "audio.wav.transcript.json").read_bytes() == (cli_dir / "audio.json").read_bytes()
    window.close()


def test_transcription_partial_outputs_match_cli_and_preserve_existing_json(tmp_path: Path, monkeypatch) -> None:
    import base64

    from elevenlabs_toolkit import providers
    from elevenlabs_toolkit.cli import transcribe

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    app = QApplication.instance() or QApplication([])
    calls = []

    class Provider:
        def __init__(self, *, env_file: Path) -> None:
            pass

        def transcribe(self, path, options):
            calls.append(options.remote_formats)
            return {
                "text": "Salom",
                "words": [
                    {
                        "text": "Salom",
                        "type": "word",
                        "start": 0,
                        "end": 1,
                        "characters": [{"text": "S", "start": 0, "end": 0.2}],
                    }
                ],
                "additional_formats": [
                    {"requested_format": "pdf", "is_base64_encoded": True, "content": base64.b64encode(b"pdf").decode()}
                ],
            }

    monkeypatch.setattr(providers, "ElevenLabsProvider", Provider)
    monkeypatch.setattr(transcribe, "ElevenLabsProvider", Provider)
    source = tmp_path / "audio.wav"
    source.write_bytes(b"audio")
    cli_dir, gui_dir = tmp_path / "cli", tmp_path / "gui"
    cli_dir.mkdir()
    gui_dir.mkdir()
    (cli_dir / "audio.json").write_text("keep", encoding="utf-8")
    (gui_dir / "audio.wav.transcript.json").write_text("keep", encoding="utf-8")
    assert main(["transcribe", str(source), "-o", str(cli_dir), "--on-conflict", "skip", "--remote-format", "pdf"]) == 0
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.project_directory = tmp_path
    window.output_directory.setText(str(gui_dir))
    window.transcription_options.remote_formats["pdf"].setChecked(True)
    window.add_paths([source])
    window.show_preview()
    assert window.worker.wait(5000)
    app.processEvents()
    assert "Planned API requests: 1" in window.details.toPlainText()
    assert calls == [("pdf",)]
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert calls == [("pdf",), ("pdf",)]
    assert (gui_dir / "audio.wav.transcript.json").read_text() == "keep"
    assert (cli_dir / "audio.json").read_text() == "keep"
    assert (gui_dir / "audio.wav.transcript.pdf").read_bytes() == (cli_dir / "audio.pdf").read_bytes() == b"pdf"
    assert window.batch_table.item(0, 2).text() == "partial"
    window.close()


def test_workflows_fit_a_900_pixel_window(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    for row in range(6):
        window.navigation.setCurrentRow(row)
        window.resize(900, 700)
        app.processEvents()
        assert window.width() <= 900, f"workflow {row} forces width {window.width()}"
        assert window.start.isVisible()
    window.close()


def test_workflows_are_navigable_at_800_pixels(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    for row in range(6):
        window.navigation.setCurrentRow(row)
        app.processEvents()
        scrollbar = window.main_scroll.horizontalScrollBar()
        if sys.platform != "win32":
            assert scrollbar.maximum() == 0, f"workflow {row}"
        else:
            scrollbar.setValue(scrollbar.maximum())
            assert scrollbar.value() == scrollbar.maximum()
    window.close()


@pytest.mark.parametrize(("width", "height"), [(800, 400), (600, 300)])
def test_small_window_keeps_processing_controls_visible_while_scrolling(monkeypatch, width: int, height: int) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(width, height)
    window.show()
    app.processEvents()
    assert window.height() <= height
    assert window.main_scroll.verticalScrollBar().maximum() > 0
    for row in range(6):
        window.navigation.setCurrentRow(row)
        app.processEvents()
        scroll = window.main_scroll.verticalScrollBar()
        for position in (scroll.minimum(), scroll.maximum()):
            scroll.setValue(position)
            app.processEvents()
            for button in (window.start, window.stop, window.preview, window.preview_content_button):
                if button.isVisible():
                    point = button.mapTo(window, button.rect().center())
                    assert window.rect().contains(point), f"workflow {row}: {button.text()}"
                    for corner in (button.rect().topLeft(), button.rect().bottomRight()):
                        assert window.rect().contains(button.mapTo(window, corner)), f"workflow {row}: {button.text()}"
                    assert not window.main_scroll.isAncestorOf(button)
    window.close()


def test_extremely_small_window_keeps_bottom_buttons_inside_window(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(320, 200)
    window.show()
    app.processEvents()
    for row in range(6):
        window.navigation.setCurrentRow(row)
        app.processEvents()
        for button in (window.start, window.stop, window.preview, window.preview_content_button):
            if button.isVisible():
                for corner in (button.rect().topLeft(), button.rect().bottomRight()):
                    assert window.rect().contains(button.mapTo(window, corner)), f"workflow {row}: {button.text()}"
    window.close()


def test_focus_reveals_center_of_wide_control(monkeypatch) -> None:
    from PySide6.QtCore import QRect

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(800, 400)
    window.show()
    window.navigation.setCurrentRow(0)
    window.job_settings_button.setChecked(True)
    app.processEvents()
    field = window.transcription_options.model
    field.setMinimumWidth(1400)
    for _ in range(20):
        app.processEvents()
        if window.options_scroll.horizontalScrollBar().maximum() > 0:
            break
    assert window.options_scroll.horizontalScrollBar().maximum() > 0
    field.setFocus()
    app.processEvents()
    window.reveal_focused_control()
    app.processEvents()
    viewport = window.main_scroll.viewport()
    visible = QRect(viewport.mapToGlobal(viewport.rect().topLeft()), viewport.size())
    assert visible.contains(field.mapToGlobal(field.rect().center()))
    window.close()


@pytest.mark.parametrize("workflow", range(6))
def test_keyboard_tab_reaches_start_button_without_text_field_trap(monkeypatch, workflow: int) -> None:
    from PySide6.QtCore import QRect
    from PySide6.QtWidgets import QScrollArea

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(600, 300)
    window.show()
    window.navigation.setCurrentRow(workflow)
    window.navigation.setFocus()
    app.processEvents()
    for _ in range(150):
        focused = app.focusWidget()
        if focused is window.start:
            break
        assert focused is not None
        if window.main_scroll.isAncestorOf(focused):
            parent = focused.parentWidget()
            while parent is not None:
                if isinstance(parent, QScrollArea):
                    viewport = parent.viewport()
                    visible = QRect(viewport.mapToGlobal(viewport.rect().topLeft()), viewport.size())
                    assert visible.contains(focused.mapToGlobal(focused.rect().center())), (
                        f"Focused {type(focused).__name__} is outside the visible panel in workflow {workflow}"
                    )
                parent = parent.parentWidget()
        QTest.keyClick(focused, Qt.Key.Key_Tab)
        app.processEvents()
    assert app.focusWidget() is window.start, f"Tab did not reach Start in workflow {workflow}"
    window.close()


@pytest.mark.parametrize("workflow", [0, 2])
def test_keyboard_can_reach_visible_expanded_advanced_options(monkeypatch, workflow: int) -> None:
    from PySide6.QtCore import QRect
    from PySide6.QtWidgets import QScrollArea

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(600, 300)
    window.show()
    window.navigation.setCurrentRow(workflow)
    window.job_settings_button.setChecked(True)
    app.processEvents()
    panel = window.transcription_options if workflow == 0 else window.export_text
    panel.advanced_section.toggle.setChecked(True)
    window.navigation.setFocus()
    app.processEvents()
    visited_duration = False
    for _ in range(250):
        focused = app.focusWidget()
        assert focused is not None
        if focused is panel.numeric_inputs["max_duration"]:
            visited_duration = True
        if focused is window.start:
            break
        parent = focused.parentWidget()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                viewport = parent.viewport()
                visible = QRect(viewport.mapToGlobal(viewport.rect().topLeft()), viewport.size())
                assert visible.contains(focused.mapToGlobal(focused.rect().center()))
            parent = parent.parentWidget()
        QTest.keyClick(focused, Qt.Key.Key_Tab)
        app.processEvents()
    assert visited_duration
    assert app.focusWidget() is window.start
    window.close()


def test_keyboard_can_select_workflow_and_change_options(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.show()
    window.navigation.setFocus()
    QTest.keyClick(window.navigation, Qt.Key.Key_Down)
    app.processEvents()
    assert window.active_workflow == 1
    window.mode.setFocus()
    QTest.keyClick(window.mode, Qt.Key.Key_Down)
    assert window.mode.currentText() == "Latin"
    assert window.output_subfolder.isChecked()
    window.output_subfolder.setFocus()
    QTest.keyClick(window.output_subfolder, Qt.Key.Key_Space)
    assert not window.output_subfolder.isChecked()
    window.close()


@pytest.mark.parametrize("pattern", ["", "["])
def test_failed_preflight_preserves_previous_retryable_batch(tmp_path: Path, monkeypatch, pattern: str) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "bad.txt"
    source.write_bytes(b"\xff")
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.navigation.setCurrentRow(1)
    window.output_subfolder.setChecked(False)
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert window.retry_button.isEnabled()
    window.filter_kind.setCurrentIndex(2)
    window.pattern.setText(pattern)
    old_worker = window.worker
    window.start_job()
    assert window.worker is old_worker
    assert "Preflight failed" in window.details.toPlainText()
    assert window.batch_table.rowCount() == 1
    assert window.batch_table.item(0, 2).text() == "failed"
    assert window.retry_button.isEnabled()
    source.write_text("Salom", encoding="utf-8")
    window.retry_failed()
    assert window.worker.wait(5000)
    app.processEvents()
    assert (tmp_path / "bad CYR.txt").read_text(encoding="utf-8") == "Салом"
    window.close()


def test_project_context_is_restored_with_its_profiles(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    project = tmp_path / "project"
    project.mkdir()
    (project / "elevenlabs-toolkit.toml").write_text(
        "[profiles.custom.segmentation]\nmax_lines = 4\n", encoding="utf-8"
    )
    window = MainWindow(settings=settings)
    monkeypatch.setattr("elevenlabs_toolkit.gui.QFileDialog.getExistingDirectory", lambda *args: str(project))
    window.select_project()
    window.export_text.profile.setCurrentText("custom")
    window.save_preset("Custom project")
    window.close()
    restored = MainWindow(settings=settings)
    assert restored.project_directory == project
    assert restored.export_text.profile.currentText() == "custom"
    assert restored.export_text.settings(project).configuration()["segmentation"]["max_lines"] == 4
    restored.reset_preferences()
    restored.load_preset("Custom project")
    assert restored.project_directory == project
    assert restored.export_text.settings(project).configuration()["segmentation"]["max_lines"] == 4
    restored.close()
    app.processEvents()


def test_gui_remote_formats_match_cli_with_fake_provider(tmp_path: Path, monkeypatch) -> None:
    import base64

    from elevenlabs_toolkit import providers
    from elevenlabs_toolkit.cli import transcribe

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    app = QApplication.instance() or QApplication([])
    calls = []
    formats = ("pdf", "docx", "html", "segmented-json")

    class Provider:
        def __init__(self, *, env_file: Path) -> None:
            self.env_file = env_file

        def transcribe(self, path, options):
            calls.append(options.remote_formats)
            return {
                "text": "Salom",
                "words": [
                    {
                        "text": "Salom",
                        "type": "word",
                        "start": 0,
                        "end": 1,
                        "characters": [{"text": "S", "start": 0, "end": 0.2}],
                    }
                ],
                "additional_formats": [
                    {
                        "requested_format": "pdf",
                        "content": base64.b64encode(b"pdf content").decode(),
                        "is_base64_encoded": True,
                    },
                    {
                        "requested_format": "docx",
                        "content": base64.b64encode(b"docx content").decode(),
                        "is_base64_encoded": True,
                    },
                    {"requested_format": "html", "content": "<p>Salom</p>"},
                    {"requested_format": "segmented_json", "content": '{"segments": []}'},
                ],
            }

    monkeypatch.setattr(providers, "ElevenLabsProvider", Provider)
    monkeypatch.setattr(transcribe, "ElevenLabsProvider", Provider)
    source = tmp_path / "audio.wav"
    source.write_bytes(b"audio")
    cli_dir, gui_dir = tmp_path / "cli", tmp_path / "gui"
    args = ["transcribe", str(source), "-o", str(cli_dir)]
    for value in formats:
        args.extend(["--remote-format", value])
    assert main(args) == 0
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.project_directory = tmp_path
    window.output_directory.setText(str(gui_dir))
    for value in formats:
        window.transcription_options.remote_formats[value].setChecked(True)
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    expected_formats = ("pdf", "docx", "html", "segmented_json")
    assert calls == [expected_formats, expected_formats]
    for suffix in ("json", "pdf", "docx", "html", "segmented.json"):
        assert (gui_dir / f"audio.wav.transcript.{suffix}").read_bytes() == (cli_dir / f"audio.{suffix}").read_bytes()
    window.close()


def test_recent_input_and_output_folders_are_remembered(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    source = input_dir / "sample.txt"
    source.write_text("Salom", encoding="utf-8")
    window = MainWindow(settings=settings)
    monkeypatch.setattr("elevenlabs_toolkit.gui.QFileDialog.getOpenFileNames", lambda *args: ([str(source)], ""))
    monkeypatch.setattr("elevenlabs_toolkit.gui.QFileDialog.getExistingDirectory", lambda *args: str(output_dir))
    window.select_files()
    window.select_output_directory()
    window.close()
    restored = MainWindow(settings=settings)
    starts = []
    monkeypatch.setattr(
        "elevenlabs_toolkit.gui.QFileDialog.getOpenFileNames", lambda *args: (starts.append(args[2]) or [], "")
    )
    restored.select_files()
    assert starts == [str(input_dir)]
    restored.output_directory.clear()
    monkeypatch.setattr(
        "elevenlabs_toolkit.gui.QFileDialog.getExistingDirectory", lambda *args: starts.append(args[2]) or ""
    )
    restored.select_output_directory()
    assert starts[-1] == str(output_dir)
    restored.close()
    app.processEvents()


def test_show_effective_workflow_settings_identifies_overrides(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    (tmp_path / "elevenlabs-toolkit.toml").write_text('[text]\nscript = "cyrillic"\n', encoding="utf-8")
    window = MainWindow()
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(2)
    window.export_text.script.setCurrentText("Latin")
    window.export_text.numeric_fields["max_lines"][0].setText("3")
    window.show_workflow_configuration()
    report = json.loads(window.details.toPlainText())
    assert report["effective"]["text"]["script"] == "latin"
    assert report["effective"]["segmentation"]["max_lines"] == 3
    assert report["gui_overrides"]["text.script"] == "latin"
    assert report["gui_overrides"]["segmentation.max_lines"] == 3
    assert "segmentation.max_duration" not in report["gui_overrides"]
    assert "env_file" not in window.details.toPlainText()
    window.close()
    app.processEvents()


def test_effective_settings_identify_profile_and_override_sources(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    user_config = tmp_path / "user.toml"
    user_config.write_text(
        "[text]\nscript = 'latin'\n[profiles.custom.segmentation]\nmax_words = 5\n", encoding="utf-8"
    )
    monkeypatch.setattr("elevenlabs_toolkit.config.user_config_path", lambda: user_config)
    (tmp_path / "elevenlabs-toolkit.toml").write_text(
        "[segmentation]\nsrt_fps = 25\n[profiles.custom.segmentation]\nmax_lines = 4\nmax_duration = 4.0\n",
        encoding="utf-8",
    )
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.project_directory = tmp_path
    window.refresh_profiles()
    window.navigation.setCurrentRow(2)
    window.export_text.profile.setCurrentText("custom")
    window.export_text.numeric_fields["max_lines"][0].setText("3")
    window.show_workflow_configuration()
    report = json.loads(window.details.toPlainText())
    assert report["profile_settings"]["segmentation"]["max_lines"] == 4
    assert report["profile_settings"]["segmentation"]["max_words"] == 5
    assert report["effective"]["segmentation"]["max_lines"] == 3
    assert report["setting_sources"]["segmentation.max_lines"] == "workflow override"
    assert report["setting_sources"]["segmentation.max_duration"] == "profile: custom"
    assert report["setting_sources"]["segmentation.max_words"] == "profile: custom"
    assert report["setting_sources"]["segmentation.srt_fps"] == "project configuration"
    assert report["setting_sources"]["text.script"] == "user configuration"
    assert report["setting_sources"]["text.cleanup"] == "built-in default"
    window.close()
    app.processEvents()


def test_gui_script_selector_exports_cyrillic_text(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    source = transcript(tmp_path / "input.json")
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(2)
    window.output_subfolder.setChecked(False)
    window.export_text.script.setCurrentText("Cyrillic")
    window.export_formats[ArtifactFormat.SRT].setChecked(False)
    window.export_formats[ArtifactFormat.TXT].setChecked(True)
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert "Салом" in (tmp_path / "input.txt").read_text(encoding="utf-8")
    window.close()


def test_fail_fast_marks_unprocessed_batch_rows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    bad, good = tmp_path / "a.txt", tmp_path / "b.txt"
    bad.write_bytes(b"\xff")
    good.write_text("Salom", encoding="utf-8")
    window.navigation.setCurrentRow(1)
    window.fail_fast.setChecked(True)
    window.add_paths([bad, good])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert window.batch_table.item(0, 2).text() == "failed"
    assert "Not run" in window.batch_table.item(1, 2).text()
    assert not (tmp_path / "b CYR.txt").exists()
    window.close()


@pytest.mark.parametrize("invalid", ["", "../combined.txt", "CON.txt"])
def test_invalid_combined_filename_is_reported_before_start(tmp_path: Path, monkeypatch, invalid: str) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    source = transcript(tmp_path / "input.json")
    window.navigation.setCurrentRow(2)
    window.output_subfolder.setChecked(False)
    window.add_paths([source])
    window.export_formats[ArtifactFormat.COMBINED_TXT].setChecked(True)
    panel = window.export_text
    panel.combined_name.setText(invalid)
    window.start_job()
    app.processEvents()
    assert panel.combined_name_error.text()
    assert panel.advanced_section.toggle.isChecked()
    assert "combined" in window.details.toPlainText().lower()
    assert not (tmp_path / "input.srt").exists()
    panel.combined_name.setText("all.txt")
    assert not panel.combined_name_error.text()
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert (tmp_path / "all.txt").exists()
    window.close()


def test_unused_combined_filename_does_not_block_other_exports(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    source = transcript(tmp_path / "input.json")
    window.navigation.setCurrentRow(2)
    window.output_subfolder.setChecked(False)
    window.add_paths([source])
    combined = window.export_formats[ArtifactFormat.COMBINED_TXT]
    combined.setChecked(True)
    window.export_text.combined_name.setText("../invalid")
    window.start_job()
    assert window.export_text.combined_name_error.text()
    combined.setChecked(False)
    assert not window.export_text.combined_name_error.text()
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert (tmp_path / "input.srt").exists()
    window.close()


@pytest.mark.parametrize("case", ["timed_output", "profile_pause"])
def test_incompatible_transcription_timestamps_are_reported_inline(tmp_path: Path, monkeypatch, case: str) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    if case == "profile_pause":
        (tmp_path / "elevenlabs-toolkit.toml").write_text("[segmentation]\npause_detection = true\n", encoding="utf-8")
    source = tmp_path / "audio.flac"
    source.write_bytes(b"audio")
    window = MainWindow()
    window.project_directory = tmp_path
    panel = window.transcription_options
    panel.timestamps.setCurrentText("None" if case == "timed_output" else "Word")
    if case == "timed_output":
        panel.local_formats[ArtifactFormat.SRT].setChecked(True)
    window.add_paths([source])
    window.start_job()
    app.processEvents()
    assert panel.timestamps_error.text()
    assert "timestamp" in window.details.toPlainText().lower()
    assert not (tmp_path / "Output").exists()
    panel.timestamps.setCurrentText("Character")
    assert not panel.timestamps_error.text()
    window.close()


@pytest.mark.parametrize("workflow", [0, 2])
def test_empty_marker_colour_is_reported_inline_before_start(tmp_path: Path, monkeypatch, workflow: int) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.navigation.setCurrentRow(workflow)
    panel = window.transcription_options if workflow == 0 else window.export_text
    panel.marker_color.setText("   ")
    assert panel.marker_color_error.text()
    window.start_job()
    app.processEvents()
    assert "marker" in window.details.toPlainText().lower()
    assert panel.advanced_section.toggle.isChecked()
    panel.marker_color.setText("ResolveColorBlue")
    assert not panel.marker_color_error.text()
    window.close()


@pytest.mark.parametrize(
    ("field", "option", "value", "valid"),
    [
        ("max_chars_per_line", "--max-chars-per-line", "1", True),
        ("max_chars_per_line", "--max-chars-per-line", "0", False),
        ("max_lines", "--max-lines", "1", True),
        ("max_lines", "--max-lines", "0", False),
        ("max_duration", "--max-duration", "1", True),
        ("max_duration", "--max-duration", "0.5", False),
        ("srt_fps", "--srt-fps", "25", True),
        ("srt_fps", "--srt-fps", "0", False),
        ("srt_padding_frames", "--srt-padding-frames", "0", True),
        ("srt_padding_frames", "--srt-padding-frames", "-1", False),
        ("srt_gap_milliseconds", "--srt-gap-ms", "0", True),
        ("srt_gap_milliseconds", "--srt-gap-ms", "-1", False),
    ],
)
def test_subtitle_numeric_boundaries_match_cli(
    tmp_path: Path, monkeypatch, field: str, option: str, value: str, valid: bool
) -> None:
    from elevenlabs_toolkit.cli.export import _options
    from elevenlabs_toolkit.cli.main import build_parser

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("elevenlabs_toolkit.config.user_config_path", lambda: tmp_path / "missing-user.toml")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    panel = window.export_text
    panel.numeric_inputs[field].setText(value)
    args = build_parser().parse_args(["export", option, value])
    if valid:
        cli_options = _options(args)
        gui_config = panel.settings(tmp_path).configuration()
        assert gui_config["segmentation"][field] == getattr(cli_options.segmentation, field)
    else:
        with pytest.raises(ValueError):
            _options(args)
        with pytest.raises(ValueError):
            panel.settings(tmp_path)
    window.close()
    app.processEvents()


@pytest.mark.parametrize("name", ["max_lines", "max_duration", "srt_fps", "gap_seconds", "marker_fps"])
def test_subtitle_numeric_fields_reject_zero_inline(tmp_path: Path, monkeypatch, name: str) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    panel = window.export_text
    panel.numeric_inputs[name].setText("0")
    assert "greater than 0" in panel.numeric_errors[name].text()
    with pytest.raises(ValueError):
        panel.settings(tmp_path)
    assert panel.advanced_section is not None and panel.advanced_section.toggle.isChecked()
    panel.numeric_inputs[name].setText("1")
    assert not panel.numeric_errors[name].text()
    window.close()
    app.processEvents()


@pytest.mark.parametrize("name", ["min_duration", "srt_padding_frames", "srt_gap_milliseconds"])
def test_subtitle_numeric_fields_allow_zero(tmp_path: Path, monkeypatch, name: str) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    field = window.export_text.numeric_inputs[name]
    field.setText("0")
    assert not window.export_text.numeric_errors[name].text()
    window.close()
    app.processEvents()


@pytest.mark.parametrize(
    ("artifact_format", "suffix"),
    [
        (ArtifactFormat.SRT, "srt"),
        (ArtifactFormat.SRT_MINI, "mini.srt"),
        (ArtifactFormat.TXT, "txt"),
        (ArtifactFormat.RESOLVE_EDL, "resolve.edl"),
        (ArtifactFormat.CUE_INDEX_SRT, "cue-index.srt"),
    ],
)
def test_transcription_gui_matches_cli_provider_options_and_outputs(
    tmp_path: Path, monkeypatch, artifact_format: ArtifactFormat, suffix: str
) -> None:
    from elevenlabs_toolkit import providers
    from elevenlabs_toolkit.cli import transcribe

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    app = QApplication.instance() or QApplication([])
    calls = []

    class Provider:
        def __init__(self, *, env_file: Path) -> None:
            self.env_file = env_file

        def transcribe(self, path, options):
            calls.append((path, options, self.env_file))
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
    monkeypatch.setattr(transcribe, "ElevenLabsProvider", Provider)
    source = tmp_path / "input.wav"
    source.write_bytes(b"audio")
    cli_dir, gui_dir = tmp_path / "cli", tmp_path / "gui"
    assert (
        main(
            [
                "transcribe",
                str(source),
                "-o",
                str(cli_dir),
                "--format",
                artifact_format.value,
                "--language-code",
                "uz",
                "--num-speakers",
                "2",
                "--keyterm",
                "Salom",
                "--keyterm",
                "Toshkent",
                "--no-diarize",
                "--no-verbatim",
                "--temperature",
                "0.5",
                "--seed",
                "42",
                "--no-audio-events",
                "--env-file",
                str(tmp_path / ".env"),
            ]
        )
        == 0
    )
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.project_directory = tmp_path
    options = window.transcription_options
    options.language.setText("uz")
    options.num_speakers.setText("2")
    options.keyterms.setPlainText("Salom\nToshkent")
    options.diarize.setChecked(False)
    options.no_verbatim.setChecked(True)
    options.temperature.setText("0.5")
    options.seed.setText("42")
    options.audio_events.setChecked(False)
    options.local_formats[artifact_format].setChecked(True)
    window.output_directory.setText(str(gui_dir))
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert (gui_dir / "input.wav.transcript.json").read_bytes() == (cli_dir / "input.json").read_bytes()
    assert (gui_dir / f"input.wav.transcript.{suffix}").read_bytes() == (cli_dir / f"input.{suffix}").read_bytes()
    window.close()


def test_gui_and_cli_allow_untimed_plain_text_transcription(tmp_path: Path, monkeypatch, capsys) -> None:
    from elevenlabs_toolkit import providers
    from elevenlabs_toolkit.cli import transcribe

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    calls = []

    class Provider:
        def __init__(self, *, env_file: Path) -> None:
            pass

        def transcribe(self, path, options):
            calls.append(options.timestamps_granularity)
            return {"text": "Salom", "words": [{"text": "Salom", "start": None, "end": None, "type": "word"}]}

    monkeypatch.setattr(providers, "ElevenLabsProvider", Provider)
    monkeypatch.setattr(transcribe, "ElevenLabsProvider", Provider)
    source = tmp_path / "input.flac"
    source.write_bytes(b"audio")
    cli_dir, gui_dir = tmp_path / "cli", tmp_path / "gui"
    assert (
        main(
            [
                "transcribe",
                str(source),
                "--timestamps",
                "none",
                "--no-pause-detection",
                "--format",
                "txt",
                "-o",
                str(cli_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.project_directory = tmp_path
    panel = window.transcription_options
    panel.timestamps.setCurrentText("None")
    pause = panel.boolean_fields["pause_detection"]
    pause.setCurrentIndex(pause.findData(False))
    panel.local_formats[ArtifactFormat.TXT].setChecked(True)
    window.output_subfolder.setChecked(False)
    window.output_directory.setText(str(gui_dir))
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert calls == ["none", "none"]
    assert (gui_dir / "input.flac.transcript.txt").read_bytes() == (cli_dir / "input.txt").read_bytes()
    assert source.read_bytes() == b"audio"
    window.close()


def test_cli_and_gui_reject_timed_output_without_timestamps(tmp_path: Path, monkeypatch, capsys) -> None:
    from elevenlabs_toolkit import providers
    from elevenlabs_toolkit.cli import transcribe

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid settings must not construct a provider")

    monkeypatch.setattr(providers, "ElevenLabsProvider", forbidden)
    monkeypatch.setattr(transcribe, "ElevenLabsProvider", forbidden)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "input.wav"
    source.write_bytes(b"audio")
    assert main(["transcribe", str(source), "--timestamps", "none", "--format", "srt"]) != 0
    assert "timestamps" in capsys.readouterr().err
    window = MainWindow()
    window.project_directory = tmp_path
    window.transcription_options.timestamps.setCurrentText("None")
    window.transcription_options.local_formats[ArtifactFormat.SRT].setChecked(True)
    window.add_paths([source])
    window.start_job()
    app.processEvents()
    assert "timestamps" in window.details.toPlainText().lower()
    assert window.transcription_options.timestamps_error.text()
    assert list(tmp_path.iterdir()) == [source]
    window.close()


def test_transcription_skip_existing_matches_cli_without_provider(tmp_path: Path, monkeypatch) -> None:
    from elevenlabs_toolkit import providers
    from elevenlabs_toolkit.cli import transcribe

    def forbidden(*args, **kwargs):
        pytest.fail("Skipped transcription must not initialize a provider")

    monkeypatch.setattr(providers, "ElevenLabsProvider", forbidden)
    monkeypatch.setattr(transcribe, "ElevenLabsProvider", forbidden)
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "input.wav"
    source.write_bytes(b"audio")
    cli_dir, gui_dir = tmp_path / "cli", tmp_path / "gui"
    cli_dir.mkdir()
    gui_dir.mkdir()
    cli_target = cli_dir / "input.json"
    gui_target = gui_dir / "input.wav.transcript.json"
    cli_target.write_text("keep", encoding="utf-8")
    gui_target.write_text("keep", encoding="utf-8")
    assert main(["transcribe", str(source), "--on-conflict", "skip", "-o", str(cli_dir)]) == 0
    worker = Worker(
        "Transcribe",
        ScriptMode.SOURCE,
        [source],
        False,
        settings=DesktopSettings(project_directory=tmp_path, output_directory=gui_dir),
        policy=ConflictPolicy.SKIP,
    )
    reports = []
    worker.finished_result.connect(reports.append, Qt.ConnectionType.DirectConnection)
    worker.run()
    assert "failed" not in reports[0].lower()
    assert cli_target.read_text() == gui_target.read_text() == "keep"


@pytest.mark.parametrize("workflow", [0, 2])
@pytest.mark.parametrize(
    ("lower", "upper", "invalid_lower", "invalid_upper", "corrected_upper"),
    [("min_duration", "max_duration", "2", "1", "3"), ("gap_seconds", "hard_gap_seconds", "2", "1", "3")],
)
def test_inconsistent_subtitle_limits_have_inline_errors(
    tmp_path: Path,
    monkeypatch,
    workflow: int,
    lower: str,
    upper: str,
    invalid_lower: str,
    invalid_upper: str,
    corrected_upper: str,
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.navigation.setCurrentRow(workflow)
    panel = window.transcription_options if workflow == 0 else window.export_text
    panel.numeric_fields[lower][0].setText(invalid_lower)
    panel.numeric_fields[upper][0].setText(invalid_upper)
    assert panel.cross_errors[lower].text()
    window.start_job()
    assert "Preflight failed" in window.details.toPlainText()
    assert window.start.isEnabled()
    assert panel.advanced_section.toggle.isChecked()
    panel.numeric_fields[upper][0].setText(corrected_upper)
    assert not panel.cross_errors[lower].text()
    panel.numeric_fields[upper][0].setText("not a number")
    assert panel.numeric_errors[upper].text()
    assert not panel.cross_errors[lower].text()
    panel.numeric_fields[upper][0].clear()
    assert not panel.cross_errors[lower].text()
    window.close()
    app.processEvents()


@pytest.mark.parametrize("workflow", [0, 2])
@pytest.mark.parametrize(("field", "value", "fixed"), [("max_duration", "2", "4"), ("gap_seconds", "3", "1")])
def test_subtitle_override_is_checked_against_profile_values_before_start(
    tmp_path: Path, monkeypatch, workflow: int, field: str, value: str, fixed: str
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    (tmp_path / "elevenlabs-toolkit.toml").write_text(
        "[profiles.custom.segmentation]\nmin_duration = 3.0\nmax_duration = 5.0\n"
        "gap_seconds = 1.0\nhard_gap_seconds = 2.0\n",
        encoding="utf-8",
    )
    window = MainWindow()
    window.project_directory = tmp_path
    window.refresh_profiles()
    window.navigation.setCurrentRow(workflow)
    panel = window.transcription_options if workflow == 0 else window.export_text
    panel.profile.setCurrentText("custom")
    panel.numeric_fields[field][0].setText(value)
    window.start_job()
    assert panel.cross_errors[field].text()
    assert panel.advanced_section.toggle.isChecked()
    assert window.start.isEnabled()
    panel.numeric_fields[field][0].setText(fixed)
    assert not panel.cross_errors[field].text()
    panel.settings(tmp_path)
    window.close()
    app.processEvents()


def test_advanced_section_collapses_without_disabling_settings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.navigation.setCurrentRow(2)
    section = window.export_text.advanced_section
    section.toggle.setChecked(True)
    field = window.export_text.numeric_fields["max_lines"][0]
    field.setText("3")
    section.toggle.setChecked(False)
    assert field.isEnabled()
    assert ("max_lines", 3) in window.export_text.settings(tmp_path).segmentation
    window.close()
    app.processEvents()


@pytest.mark.parametrize("source", ["project", "selected", "environment"])
def test_transcription_credential_precedence_without_network(tmp_path: Path, monkeypatch, source: str) -> None:
    import sys
    from types import ModuleType, SimpleNamespace

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("ELEVENLABS_API_KEY=project-test-key\n", encoding="utf-8")
    selected = tmp_path / "selected.env"
    selected.write_text("ELEVENLABS_API_KEY=selected-test-key\n", encoding="utf-8")
    if source == "environment":
        monkeypatch.setenv("ELEVENLABS_API_KEY", "environment-test-key")
    received_keys = []

    class Client:
        def __init__(self, *, api_key, headers):
            assert headers == {"Accept-Encoding": "identity"}
            received_keys.append(api_key)
            self.speech_to_text = SimpleNamespace(convert=self.convert)

        def convert(self, **kwargs):
            return {
                "text": "Salom",
                "words": [
                    {
                        "text": "Salom",
                        "type": "word",
                        "start": 0,
                        "end": 1,
                        "characters": [{"text": "S", "start": 0, "end": 0.2}],
                    }
                ],
            }

    client_module = ModuleType("elevenlabs.client")
    client_module.ElevenLabs = Client
    monkeypatch.setitem(sys.modules, "elevenlabs.client", client_module)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings)
    window.output_subfolder.setChecked(False)
    window.project_directory = project
    if source != "project":
        monkeypatch.setattr(
            "elevenlabs_toolkit.gui_options.QFileDialog.getOpenFileName", lambda *args: (str(selected), "")
        )
        window.navigation.setCurrentRow(5)
        window.config_choose_credentials.click()
        assert window.config_credentials.text() == str(selected)
        window.navigation.setCurrentRow(0)
    window.add_paths([audio])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert received_keys == [f"{source}-test-key"]
    assert (tmp_path / "audio.wav.transcript.json").is_file()
    assert all(key not in window.details.toPlainText() for key in received_keys)
    window.close()
    assert all(key not in (tmp_path / "settings.ini").read_text() for key in received_keys)


@pytest.mark.parametrize("recovery", ["project_file", "selected_file"])
def test_missing_credentials_can_be_fixed_and_retried_without_network(
    tmp_path: Path, monkeypatch, recovery: str
) -> None:
    import sys
    from types import ModuleType, SimpleNamespace

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    received = []
    requested_models = []

    class Client:
        def __init__(self, *, api_key, headers):
            assert headers == {"Accept-Encoding": "identity"}
            received.append(api_key)
            self.speech_to_text = SimpleNamespace(convert=self.convert)

        def convert(self, **kwargs):
            requested_models.append(kwargs["model_id"])
            return {
                "text": "Salom",
                "words": [
                    {
                        "text": "Salom",
                        "type": "word",
                        "start": 0,
                        "end": 1,
                        "characters": [{"text": "S", "start": 0, "end": 0.2}],
                    }
                ],
            }

    client_module = ModuleType("elevenlabs.client")
    client_module.ElevenLabs = Client
    monkeypatch.setitem(sys.modules, "elevenlabs.client", client_module)
    reviews = []

    def approve(window, preview):
        reviews.append(preview)
        window.worker.answer_review(True)

    monkeypatch.setattr(MainWindow, "review_plan", approve)
    source = tmp_path / "audio.wav"
    source.write_bytes(b"audio")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.project_directory = tmp_path
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert window.retry_button.isEnabled()
    assert ".env" in window.batch_table.item(0, 3).text()
    assert not received
    credentials = tmp_path / (".env" if recovery == "project_file" else "selected.env")
    credentials.write_text("ELEVENLABS_API_KEY=test-key\n", encoding="utf-8")
    if recovery == "selected_file":
        window.transcription_options.env_file.setText(str(credentials))
        window.transcription_options.model.setText("different-model")
    window.retry_failed()
    for _ in range(100):
        app.processEvents()
        if window.worker.isFinished():
            break
        QTest.qWait(20)
    assert window.worker.wait(5000)
    app.processEvents()
    assert len(reviews) == 1
    assert received == ["test-key"]
    assert requested_models == ["scribe_v2"]
    assert (tmp_path / "audio.wav.transcript.json").is_file()
    assert not window.retry_button.isEnabled()
    window.close()


def test_credentials_file_can_be_selected_without_saving_it(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings)
    selected = tmp_path / ".env"
    selected.write_text("ELEVENLABS_API_KEY=secret-value\n", encoding="utf-8")
    monkeypatch.setattr("elevenlabs_toolkit.gui_options.QFileDialog.getOpenFileName", lambda *args: (str(selected), ""))
    window.transcription_options.choose_credentials.click()
    assert window.transcription_options.settings(tmp_path).env_file == selected
    window.close()
    saved = (tmp_path / "settings.ini").read_text(encoding="utf-8")
    assert "secret-value" not in saved
    assert str(selected) not in saved
    app.processEvents()


def test_credentials_help_is_available_from_help_menu(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    shown = []
    monkeypatch.setattr("elevenlabs_toolkit.gui.QMessageBox.information", lambda *args: shown.append(args))
    help_menu = next(action.menu() for action in window.menuBar().actions() if action.text() == "Help")
    action = next(action for action in help_menu.actions() if action.text() == "Credentials and API Costs")
    action.trigger()
    guidance = shown[0][2]
    for phrase in ("ELEVENLABS_API_KEY", "sk_", ".env", "Credentials file", "environment", "charges", "not saved"):
        assert phrase.lower() in guidance.lower()
    window.close()
    app.processEvents()


def test_transcription_options_explain_provider_settings(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    panel = window.transcription_options
    assert "charges" in panel.retries.toolTip().lower()
    assert "files" in panel.delay.toolTip().lower()
    assert "speaker" in panel.diarize.toolTip().lower()
    assert "timestamps" in panel.timestamps.toolTip().lower()
    assert "scribe_v2" in panel.no_verbatim.toolTip()
    window.close()
    app.processEvents()


@pytest.mark.parametrize(
    ("field", "option", "value", "valid"),
    [
        ("num_speakers", "--num-speakers", "1", True),
        ("num_speakers", "--num-speakers", "32", True),
        ("num_speakers", "--num-speakers", "0", False),
        ("num_speakers", "--num-speakers", "33", False),
        ("seed", "--seed", "0", True),
        ("seed", "--seed", "2147483647", True),
        ("seed", "--seed", "-1", False),
        ("seed", "--seed", "2147483648", False),
        ("temperature", "--temperature", "0", True),
        ("temperature", "--temperature", "2", True),
        ("temperature", "--temperature", "-0.1", False),
        ("temperature", "--temperature", "nan", False),
    ],
)
def test_recognition_numeric_boundaries_match_cli(
    tmp_path: Path, monkeypatch, field: str, option: str, value: str, valid: bool
) -> None:
    from elevenlabs_toolkit.cli.main import build_parser
    from elevenlabs_toolkit.cli.transcribe import _build_options

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    panel = window.transcription_options
    panel.numeric_inputs[field].setText(value)
    args = build_parser().parse_args(["transcribe", option, value])
    if valid:
        cli_options, _, _ = _build_options(args)
        gui_options = panel.settings(tmp_path).transcription
        assert getattr(gui_options, field) == getattr(cli_options, field)
        assert not panel.numeric_errors[field].text()
    else:
        with pytest.raises(ValueError):
            _build_options(args)
        with pytest.raises(ValueError):
            panel.settings(tmp_path)
        assert panel.numeric_errors[field].text()
    window.close()
    app.processEvents()


@pytest.mark.parametrize(
    ("field", "option", "setting", "value", "valid"),
    [
        ("retries", "--retries", "retries", "0", True),
        ("retries", "--retries", "retries", "2", True),
        ("retries", "--retries", "retries", "-1", False),
        ("backoff", "--retry-backoff", "retry_backoff", "0", True),
        ("backoff", "--retry-backoff", "retry_backoff", "1.5", True),
        ("backoff", "--retry-backoff", "retry_backoff", "-1", False),
        ("backoff", "--retry-backoff", "retry_backoff", "inf", False),
        ("delay", "--request-delay", "request_delay", "0", True),
        ("delay", "--request-delay", "request_delay", "2.5", True),
        ("delay", "--request-delay", "request_delay", "-1", False),
        ("delay", "--request-delay", "request_delay", "nan", False),
    ],
)
def test_request_pacing_validation_matches_cli(
    tmp_path: Path, monkeypatch, field: str, option: str, setting: str, value: str, valid: bool
) -> None:
    from elevenlabs_toolkit.cli.main import build_parser
    from elevenlabs_toolkit.cli.transcribe import _build_options

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    panel = window.transcription_options
    panel.numeric_inputs[field].setText(value)
    args = build_parser().parse_args(["transcribe", option, value])
    if valid:
        _build_options(args)
        assert getattr(panel.settings(tmp_path), setting) == getattr(args, setting)
        assert not panel.numeric_errors[field].text()
    else:
        with pytest.raises(ValueError):
            _build_options(args)
        with pytest.raises(ValueError):
            panel.settings(tmp_path)
        assert panel.numeric_errors[field].text()
    window.close()
    app.processEvents()


def test_invalid_speaker_count_has_inline_error_and_blocks_preflight(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.transcription_options.num_speakers.setText("33")
    app.processEvents()
    assert "between 1 and 32" in window.transcription_options.numeric_errors["num_speakers"].text()
    source = tmp_path / "input.wav"
    source.write_bytes(b"audio")
    window.add_paths([source])
    window.start_job()
    assert "Speaker count" in window.details.toPlainText()
    assert window.start.isEnabled()
    window.transcription_options.num_speakers.setText("2")
    assert not window.transcription_options.numeric_errors["num_speakers"].text()
    window.close()


@pytest.mark.parametrize("invalid", ["model", "keyterms", "no_verbatim"])
def test_invalid_transcription_text_options_have_inline_errors(monkeypatch, invalid: str) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    panel = window.transcription_options
    if invalid == "model":
        panel.model.clear()
        error = panel.model_error
    elif invalid == "keyterms":
        panel.keyterms.setPlainText("valid\ninvalid[term]")
        error = panel.keyterms_error
    else:
        panel.model.setText("scribe_v1")
        panel.no_verbatim.setChecked(True)
        error = panel.verbatim_error
    assert error.text()
    window.start_job()
    assert "preflight failed" in window.details.toPlainText().lower()
    assert window.start.isEnabled()
    panel.model.setText("scribe_v2")
    panel.keyterms.setPlainText("valid")
    assert not error.text()
    window.close()
    app.processEvents()


@pytest.mark.parametrize("workflow", [0, 2, 3])
def test_invalid_replacement_is_reported_inline_before_start(tmp_path: Path, monkeypatch, workflow: int) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.navigation.setCurrentRow(workflow)
    panel = (window.transcription_options, window.export_text, window.clean_text)[{0: 0, 2: 1, 3: 2}[workflow]]
    panel.override_replacements.setChecked(True)
    panel.replacements.setPlainText("valid=VALID\nnot-a-replacement")
    assert "line 2" in panel.replacement_error.text().lower()
    window.start_job()
    assert "replacement" in window.details.toPlainText().lower()
    assert window.start.isEnabled()
    panel.replacements.setPlainText("valid=VALID\nname=Name")
    assert panel.replacement_error.text() == ""
    panel.replacements.setPlainText("invalid")
    panel.override_replacements.setChecked(False)
    assert panel.replacement_error.text() == ""
    assert panel.settings(tmp_path).replacements is None
    window.close()
    app.processEvents()


def test_batch_table_distinguishes_individual_and_combined_export_results(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    good = transcript(tmp_path / "good.json")
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    window.navigation.setCurrentRow(2)
    window.export_formats[ArtifactFormat.SRT].setChecked(False)
    window.export_formats[ArtifactFormat.TXT].setChecked(True)
    window.export_formats[ArtifactFormat.COMBINED_TXT].setChecked(True)
    window.add_paths([good, bad])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    rows = [
        tuple(window.batch_table.item(i, col).text() for col in range(3)) for i in range(window.batch_table.rowCount())
    ]
    assert any(
        source == str(good) and output.endswith("good.txt") and status == "ok" for source, output, status in rows
    )
    assert any(
        source == str(good) and output.endswith("combined.txt") and status == "failed"
        for source, output, status in rows
    )
    window.close()


@pytest.mark.parametrize("preview_before_switch", [False, True])
def test_switching_workflows_preserves_failed_batch_for_retry(
    tmp_path: Path, monkeypatch, preview_before_switch: bool
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "bad.txt"
    source.write_bytes(b"\xff")
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.navigation.setCurrentRow(1)
    window.output_subfolder.setChecked(False)
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert window.retry_button.isEnabled()
    if preview_before_switch:
        window.show_preview()
        assert window.worker.wait(5000)
        app.processEvents()
    window.navigation.setCurrentRow(4)
    window.output_subfolder.setChecked(False)
    assert not window.retry_button.isEnabled()
    window.navigation.setCurrentRow(1)
    window.output_subfolder.setChecked(False)
    assert window.retry_button.isEnabled()
    assert window.batch_table.item(0, 2).text() == ("Pending" if preview_before_switch else "failed")
    source.write_text("Salom", encoding="utf-8")
    window.retry_failed()
    assert window.worker.wait(5000)
    app.processEvents()
    assert (tmp_path / "bad CYR.txt").is_file()
    window.close()


def test_switching_workflows_restores_its_own_batch_progress(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "input.txt"
    source.write_text("Salom", encoding="utf-8")
    window = MainWindow()
    window.navigation.setCurrentRow(1)
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert window.progress.value() == window.progress.maximum() == 1
    window.navigation.setCurrentRow(4)
    assert window.progress.value() == 0
    window.navigation.setCurrentRow(1)
    assert window.progress.value() == window.progress.maximum() == 1
    window.close()


def test_preview_does_not_discard_retryable_batch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    good, bad = tmp_path / "good.txt", tmp_path / "bad.txt"
    good.write_text("Salom", encoding="utf-8")
    bad.write_bytes(b"\xff")
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.navigation.setCurrentRow(1)
    window.output_subfolder.setChecked(False)
    window.add_paths([good, bad])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert window.retry_button.isEnabled()
    bad.write_text("Salom", encoding="utf-8")
    window.show_preview()
    assert window.worker.wait(5000)
    app.processEvents()
    assert window.retry_button.isEnabled()
    assert window.retry_button.text() == "Retry Previous Batch"
    window.retry_failed()
    assert window.worker.wait(5000)
    app.processEvents()
    assert (tmp_path / "bad CYR.txt").is_file()
    assert window.batch_table.rowCount() == 2
    assert all(window.batch_table.item(row, 2).text() == "written" for row in range(2))
    window.close()


def test_retry_failed_keeps_original_batch_destinations(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    good, bad = tmp_path / "one" / "good.txt", tmp_path / "two" / "bad.txt"
    good.parent.mkdir()
    bad.parent.mkdir()
    good.write_text("Salom", encoding="utf-8")
    bad.write_bytes(b"\xff")
    window.navigation.setCurrentRow(1)
    window.output_subfolder.setChecked(False)
    window.output_directory.setText(str(tmp_path / "out"))
    window.add_paths([good, bad])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert (tmp_path / "out" / "one" / "good CYR.txt").exists()
    bad.write_text("Salom", encoding="utf-8")
    window.retry_failed()
    assert window.worker.wait(5000)
    app.processEvents()
    assert (tmp_path / "out" / "two" / "bad CYR.txt").read_text(encoding="utf-8") == "Салом"
    assert not (tmp_path / "out" / "bad CYR.txt").exists()
    window.close()


@pytest.mark.parametrize("mode", ["Quiet", "Standard", "Verbose"])
def test_reporting_modes_keep_file_errors_visible(tmp_path: Path, monkeypatch, mode: str) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "bad.txt"
    source.write_bytes(b"\xff")
    window = MainWindow()
    window.navigation.setCurrentRow(1)
    window.report_mode.setCurrentText(mode)
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert window.batch_table.item(0, 2).text() == "failed"
    assert window.batch_table.item(0, 3).text()
    result = window.details.toPlainText()
    assert "failed" in result.lower()
    assert ("Traceback" in result) == (mode == "Verbose")
    assert source.read_bytes() == b"\xff"
    window.close()


@pytest.mark.parametrize("mode", ["Quiet", "Standard"])
def test_retry_respects_progress_reporting_mode(tmp_path: Path, monkeypatch, mode: str) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "input.txt"
    source.write_bytes(b"\xff")
    window = MainWindow()
    window.navigation.setCurrentRow(1)
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert window.retry_button.isEnabled()
    source.write_text("Salom", encoding="utf-8")
    window.report_mode.setCurrentText(mode)
    updates = []
    window.details.textChanged.connect(lambda: updates.append(window.details.toPlainText()))
    window.retry_failed()
    assert window.worker.wait(5000)
    app.processEvents()
    assert any("[1/1]" in update for update in updates) == (mode == "Standard")
    assert "written" in window.details.toPlainText().lower()
    window.close()


def test_retry_replaces_unchanged_existing_output_after_failed_replace(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "bad.txt"
    source.write_bytes(b"\xff")
    output = tmp_path / "bad CYR.txt"
    output.write_text("old output", encoding="utf-8")
    window = MainWindow()
    window.navigation.setCurrentRow(1)
    window.output_subfolder.setChecked(False)
    window.conflict.setCurrentIndex(1)
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert window.retry_button.isEnabled()
    assert output.read_text(encoding="utf-8") == "old output"
    source.write_text("Salom", encoding="utf-8")
    window.retry_failed()
    assert window.worker.wait(5000)
    app.processEvents()
    assert output.read_text(encoding="utf-8") == "Салом"
    window.close()


def test_retry_does_not_replace_output_created_after_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    source = tmp_path / "bad.txt"
    source.write_bytes(b"\xff")
    window.navigation.setCurrentRow(1)
    window.conflict.setCurrentIndex(1)
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    source.write_text("Salom", encoding="utf-8")
    output = tmp_path / "bad CYR.txt"
    output.write_text("keep this", encoding="utf-8")
    window.retry_failed()
    assert window.worker.wait(5000)
    app.processEvents()
    assert output.read_text(encoding="utf-8") == "keep this"
    window.close()


@pytest.mark.skipif(sys.platform == "win32", reason="Creating symlinks may require elevated Windows privileges")
def test_retry_does_not_replace_another_selected_input_via_symlink(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    first, second = tmp_path / "a.txt", tmp_path / "b.txt"
    first.write_bytes(b"\xff")
    second.write_text("Salom", encoding="utf-8")
    target = tmp_path / "a CYR.txt"
    target.write_text("old output", encoding="utf-8")
    window = MainWindow()
    window.navigation.setCurrentRow(1)
    window.output_subfolder.setChecked(False)
    window.conflict.setCurrentIndex(1)
    window.fail_fast.setChecked(True)
    window.add_paths([first, second])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    first.write_text("Salom", encoding="utf-8")
    second.unlink()
    second.symlink_to(target)
    window.retry_failed()
    assert window.worker.wait(5000)
    app.processEvents()
    assert target.read_text(encoding="utf-8") == "old output"
    assert second.read_text(encoding="utf-8") == "old output"
    window.close()


@pytest.mark.skipif(sys.platform == "win32", reason="Creating symlinks may require elevated Windows privileges")
def test_export_retry_does_not_replace_another_selected_input_via_symlink(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    first.write_text("broken", encoding="utf-8")
    transcript(second)
    target = tmp_path / "a.srt"
    target.write_text("old output", encoding="utf-8")
    window = MainWindow()
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(2)
    window.output_subfolder.setChecked(False)
    window.conflict.setCurrentIndex(1)
    window.fail_fast.setChecked(True)
    window.add_paths([first, second])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    transcript(first)
    second.unlink()
    second.symlink_to(target)
    window.retry_failed()
    assert window.worker.wait(5000)
    app.processEvents()
    assert target.read_text(encoding="utf-8") == "old output"
    assert second.read_text(encoding="utf-8") == "old output"
    window.close()


def test_retry_does_not_replace_output_changed_after_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "bad.txt"
    source.write_bytes(b"\xff")
    output = tmp_path / "bad CYR.txt"
    output.write_text("old output", encoding="utf-8")
    window = MainWindow()
    window.navigation.setCurrentRow(1)
    window.output_subfolder.setChecked(False)
    window.conflict.setCurrentIndex(1)
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert window.retry_button.isEnabled()
    source.write_text("Salom", encoding="utf-8")
    output.write_text("changed after failure", encoding="utf-8")
    window.retry_failed()
    assert window.worker.wait(5000)
    app.processEvents()
    assert output.read_text(encoding="utf-8") == "changed after failure"
    window.close()


@pytest.mark.parametrize("policy", ["skip", "replace"])
def test_content_preview_explains_existing_output_policy(tmp_path: Path, monkeypatch, policy: str) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "input.txt"
    source.write_text("Salom", encoding="utf-8")
    target = tmp_path / "input CYR.txt"
    target.write_text("old output", encoding="utf-8")
    window = MainWindow()
    window.navigation.setCurrentRow(1)
    window.output_subfolder.setChecked(False)
    window.conflict.setCurrentIndex(window.conflict.findData(policy))
    window.add_paths([source])
    window.preview_content_button.click()
    assert window.worker.wait(5000)
    app.processEvents()
    warning = window.details.toPlainText()
    assert ("skipped" if policy == "skip" else "replaced") in warning
    assert "if unchanged" not in warning
    target.write_text("changed after preview", encoding="utf-8")
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert target.read_text(encoding="utf-8") == ("changed after preview" if policy == "skip" else "Салом")
    assert source.read_text(encoding="utf-8") == "Salom"
    window.close()


@pytest.mark.parametrize("report_mode", ["Standard", "JSON"])
@pytest.mark.parametrize("clean_json", [False, True])
def test_combined_text_content_preview_matches_export_without_writes(
    tmp_path: Path, monkeypatch, report_mode: str, clean_json: bool
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    sources = [transcript(tmp_path / name) for name in ("lesson 10.json", "lesson 2.json")]
    originals = {source: source.read_bytes() for source in sources}
    output = tmp_path / "combined.txt"
    window = MainWindow()
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(2)
    window.output_subfolder.setChecked(False)
    window.export_formats[ArtifactFormat.SRT].setChecked(False)
    window.export_formats[ArtifactFormat.COMBINED_TXT].setChecked(True)
    window.export_formats[ArtifactFormat.CLEAN_JSON].setChecked(clean_json)
    window.report_mode.setCurrentText(report_mode)
    window.add_paths(sources)
    window.preview_content_button.click()
    assert window.worker.wait(5000)
    app.processEvents()
    preview = window.details.toPlainText()
    if report_mode == "JSON":
        data = json.loads(preview)
        assert set(data["sources"]) == {str(source) for source in sources}
        content = data["content"]
    else:
        assert all(str(source) in preview for source in sources)
        content = preview.split("\n\n", 1)[1]
    assert not output.exists()
    assert not list(tmp_path.glob("*.clean.json"))
    assert all(source.read_bytes() == originals[source] for source in sources)
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert output.read_text(encoding="utf-8") == content
    window.close()


def test_export_content_preview_displays_subtitles_without_writes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    source = transcript(tmp_path / "input.json")
    window.navigation.setCurrentRow(2)
    window.add_paths([source])
    window.preview_content_button.click()
    assert window.worker.wait(5000)
    app.processEvents()
    assert "Salom" in window.details.toPlainText()
    assert "-->" in window.details.toPlainText()
    assert not source.with_suffix(".srt").exists()
    window.close()


def test_nested_custom_output_warning_tracks_selection_and_settings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    folder = tmp_path / "inputs"
    folder.mkdir()
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.navigation.setCurrentRow(1)
    window.output_subfolder.setChecked(False)
    window.show()
    app.processEvents()
    window.add_paths([folder])
    window.output_directory.setText(str(folder / "results"))
    assert "scan" in window.output_warning.text().lower()
    assert window.output_warning.isVisible()
    window.recursive.setChecked(False)
    assert not window.output_warning.text()
    window.filter_kind.setCurrentIndex(1)
    window.pattern.setText("**/*.srt")
    assert window.output_warning.text()
    window.filter_kind.setCurrentIndex(0)
    assert not window.output_warning.text()
    window.recursive.setChecked(True)
    assert window.output_warning.text()
    window.output_subfolder.setChecked(True)
    assert not window.output_warning.text()
    window.output_subfolder.setChecked(False)
    window.output_directory.setText(str(tmp_path / "outside"))
    assert not window.output_warning.text()
    window.output_directory.setText(str(folder))
    assert window.output_warning.text()
    window.recursive.setChecked(False)
    assert window.output_warning.text()
    window.clear_files()
    assert not window.output_warning.text()
    window.close()
    app.processEvents()


def test_conversion_rejects_output_hardlinked_to_source(tmp_path: Path, monkeypatch) -> None:
    import os

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "sample LTN.srt"
    original = "1\n00:00:00,000 --> 00:00:01,000\nSalom\n"
    source.write_text(original, encoding="utf-8")
    output = tmp_path / "Output" / "sample CYR.srt"
    output.parent.mkdir()
    try:
        os.link(source, output)
    except OSError as exc:
        pytest.skip(f"Hard links are unavailable: {exc}")
    window = MainWindow()
    window.navigation.setCurrentRow(1)
    window.conflict.setCurrentIndex(window.conflict.findData("replace"))
    window.add_paths([source])
    window.show_preview()
    assert window.worker.wait(5000)
    app.processEvents()
    assert "conflict" in window.details.toPlainText().lower()
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert "overwrite" in window.details.toPlainText().lower()
    assert source.read_text(encoding="utf-8") == original
    assert output.read_text(encoding="utf-8") == original
    window.close()


def test_output_subfolder_is_default_for_new_workflows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    source = tmp_path / "sample.txt"
    source.write_text("Salom", encoding="utf-8")
    window.navigation.setCurrentRow(1)
    assert window.output_subfolder.isChecked()
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert (tmp_path / "Output" / "sample CYR.txt").read_text(encoding="utf-8") == "Салом"
    assert source.read_text(encoding="utf-8") == "Salom"
    opened = []
    monkeypatch.setattr("elevenlabs_toolkit.gui.QDesktopServices.openUrl", lambda url: opened.append(url) or True)
    window.batch_table.selectRow(0)
    window.open_output_folder()
    assert Path(opened[0].toLocalFile()) == tmp_path / "Output"
    window.close()


def test_repeated_folder_conversion_does_not_reprocess_default_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    nested = tmp_path / "nested"
    nested.mkdir()
    sources = [tmp_path / "first LTN.txt", nested / "second LTN.txt"]
    for source in sources:
        source.write_text("Salom", encoding="utf-8")
    window = MainWindow()
    window.navigation.setCurrentRow(1)
    window.add_paths([tmp_path])
    for _ in range(2):
        window.start_job()
        assert window.worker.wait(5000)
        app.processEvents()
        assert window.batch_table.rowCount() == 2
        assert {window.batch_table.item(row, 0).text() for row in range(2)} == {str(path) for path in sources}
    for source in sources:
        output = source.parent / "Output" / source.name.replace("LTN", "CYR")
        assert output.read_text(encoding="utf-8") == "Салом"
        assert source.read_text(encoding="utf-8") == "Salom"
        assert not (output.parent / "Output").exists()
    window.close()


@pytest.mark.parametrize(
    ("workflow", "output_name"), [(0, "audio.flac.transcript.json"), (2, "input.srt"), (3, "input.clean.json")]
)
def test_other_file_workflows_use_default_output_subfolder(
    tmp_path: Path, monkeypatch, workflow: int, output_name: str
) -> None:
    from elevenlabs_toolkit import providers

    class Provider:
        def __init__(self, *, env_file: Path) -> None:
            pass

        def transcribe(self, path, options):
            return {
                "text": "Salom",
                "words": [
                    {
                        "text": "Salom",
                        "type": "word",
                        "start": 0,
                        "end": 1,
                        "characters": [{"text": "S", "start": 0, "end": 0.2}],
                    }
                ],
            }

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(providers, "ElevenLabsProvider", Provider)
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "audio.flac" if workflow == 0 else transcript(tmp_path / "input.json")
    if workflow == 0:
        source.write_bytes(b"audio")
    original = source.read_bytes()
    window = MainWindow()
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(workflow)
    assert window.output_subfolder.isChecked()
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert (tmp_path / "Output" / output_name).is_file(), window.details.toPlainText()
    assert source.read_bytes() == original
    window.close()


def test_saved_source_adjacent_preference_is_preserved(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings)
    window.navigation.setCurrentRow(1)
    window.output_subfolder.setChecked(False)
    window.close()
    restored = MainWindow(settings=settings)
    restored.navigation.setCurrentRow(1)
    assert not restored.output_subfolder.isChecked()
    restored.close()
    app.processEvents()


def test_output_subfolder_option_is_used_for_conversion(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    source = tmp_path / "sample.txt"
    source.write_text("Salom", encoding="utf-8")
    window.navigation.setCurrentRow(1)
    custom = str(tmp_path / "custom")
    window.output_directory.setText(custom)
    window.output_subfolder.setChecked(True)
    assert not window.output_directory.isEnabled()
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert (tmp_path / "Output" / "sample CYR.txt").read_text(encoding="utf-8") == "Салом"
    window.output_subfolder.setChecked(False)
    assert window.output_directory.isEnabled()
    assert window.output_directory.text() == custom
    window.close()


@pytest.mark.parametrize("invalid", [{"recursive": "yes"}, {"filter_kind": 99}, {"output_directory": None}])
def test_damaged_preset_is_rejected_without_changing_current_settings(
    tmp_path: Path, monkeypatch, invalid: dict
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings)
    window.output_directory.setText(str(tmp_path / "current"))
    settings.setValue(
        "presets",
        json.dumps({"broken": {"options": {}, "workflows": {"0": invalid}, "project_directory": str(tmp_path)}}),
    )
    window.load_preset("broken")
    assert "could not load preset" in window.details.toPlainText().lower()
    assert window.output_directory.text() == str(tmp_path / "current")
    window.close()
    app.processEvents()


def test_damaged_saved_workflow_settings_do_not_prevent_launch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    settings.setValue("workflow_controls", json.dumps({"0": {"recursive": "yes"}}))
    window = MainWindow(settings=settings)
    assert window.recursive.isChecked()
    window.close()
    app.processEvents()


def test_invalid_saved_window_geometry_does_not_prevent_launch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    settings.setValue("geometry", "invalid geometry")
    window = MainWindow(settings=settings)
    assert window.size().width() > 0
    assert window.size().height() > 0
    window.close()
    app.processEvents()


def test_named_presets_and_reset_do_not_store_credentials(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings)
    window.transcription_options.model.setText("custom-model")
    window.transcription_options.env_file.setText("secret-file.env")
    window.output_subfolder.setChecked(False)
    window.output_directory.setText(str(tmp_path / "out"))
    window.save_preset("My preset")
    window.transcription_options.model.setText("changed")
    window.output_directory.clear()
    window.load_preset("My preset")
    assert window.transcription_options.model.text() == "custom-model"
    assert window.output_directory.text() == str(tmp_path / "out")
    assert not window.output_subfolder.isChecked()
    assert window.desktop_settings().output_directory == tmp_path / "out"
    window.reset_preferences()
    assert window.transcription_options.model.text() == "scribe_v2"
    assert window.transcription_options.env_file.text() == ""
    assert window.output_directory.text() == ""
    assert window.output_subfolder.isChecked()
    window.close()
    assert "secret-file.env" not in (tmp_path / "settings.ini").read_text()
    restored = MainWindow(settings=settings)
    restored.load_preset("My preset")
    assert restored.transcription_options.model.text() == "custom-model"
    assert not restored.output_subfolder.isChecked()
    assert restored.desktop_settings().output_directory == tmp_path / "out"
    restored.close()
    app.processEvents()


def test_delete_preset_requires_confirmation_and_preserves_other_settings(tmp_path: Path, monkeypatch) -> None:
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings)
    window.save_preset("Keep")
    window.save_preset("Remove")
    window.transcription_options.model.setText("current-model")
    monkeypatch.setattr("elevenlabs_toolkit.gui.QInputDialog.getItem", lambda *args: ("Remove", True))
    monkeypatch.setattr("elevenlabs_toolkit.gui.QMessageBox.question", lambda *args: QMessageBox.StandardButton.No)
    action = next(action for action in window.preferences_menu.actions() if action.text() == "Delete Preset…")
    action.trigger()
    assert set(window.presets()) == {"Keep", "Remove"}
    monkeypatch.setattr("elevenlabs_toolkit.gui.QMessageBox.question", lambda *args: QMessageBox.StandardButton.Yes)
    action.trigger()
    assert set(window.presets()) == {"Keep"}
    assert window.transcription_options.model.text() == "current-model"
    window.close()
    restored = MainWindow(settings=settings)
    assert set(restored.presets()) == {"Keep"}
    restored.close()
    app.processEvents()


def test_reset_restores_configuration_profile_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow(settings=QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    window.navigation.setCurrentRow(5)
    window.config_profile.setCurrentText("social")
    window.reset_preferences()
    assert window.config_profile.currentText() == "Project default"
    window.show_configuration()
    from elevenlabs_toolkit.config import effective_config

    assert json.loads(window.details.toPlainText())["effective"] == effective_config(cwd=window.project_directory)
    window.close()
    app.processEvents()


def test_advanced_workflow_options_persist_without_credentials(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings)
    window.transcription_options.model.setText("scribe_test")
    window.transcription_options.env_file.setText("supersecret.env")
    window.export_text.numeric_fields["max_lines"][0].setText("3")
    window.mode.setCurrentText("Latin")
    window.close()
    assert "supersecret.env" not in (tmp_path / "settings.ini").read_text()
    restored = MainWindow(settings=settings)
    assert restored.transcription_options.model.text() == "scribe_test"
    assert restored.transcription_options.env_file.text() == ""
    assert restored.export_text.numeric_fields["max_lines"][0].text() == "3"
    assert restored.mode.currentText() == "Latin"
    restored.close()
    app.processEvents()


def test_workflow_settings_are_restored_between_sessions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings)
    window.output_directory.setText(str(tmp_path / "audio"))
    window.navigation.setCurrentRow(1)
    window.filter_kind.setCurrentIndex(1)
    window.pattern.setText("*.srt")
    window.output_directory.setText(str(tmp_path / "output"))
    window.output_subfolder.setChecked(True)
    window.fail_fast.setChecked(True)
    window.close()
    restored = MainWindow(settings=settings)
    assert restored.output_directory.text() == str(tmp_path / "audio")
    restored.navigation.setCurrentRow(1)
    assert restored.filter_kind.currentIndex() == 1
    assert restored.pattern.text() == "*.srt"
    assert restored.output_directory.text() == str(tmp_path / "output")
    assert restored.output_subfolder.isChecked()
    assert not restored.output_directory.isEnabled()
    assert restored.fail_fast.isChecked()
    restored.close()
    app.processEvents()


@pytest.mark.parametrize("same_folder", [True, False])
def test_open_output_folder_without_selection_uses_unambiguous_batch_destination(
    tmp_path: Path, monkeypatch, same_folder: bool
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    sources = []
    for name in ("a", "b"):
        folder = tmp_path if same_folder else tmp_path / name
        folder.mkdir(exist_ok=True)
        source = folder / f"{name}.txt"
        source.write_text("Salom", encoding="utf-8")
        sources.append(source)
    window = MainWindow()
    window.navigation.setCurrentRow(1)
    window.add_paths(sources)
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert not window.batch_table.selectedItems()
    opened = []
    monkeypatch.setattr("elevenlabs_toolkit.gui.QDesktopServices.openUrl", lambda url: opened.append(url) or True)
    window.open_output_folder()
    if same_folder:
        assert [Path(url.toLocalFile()) for url in opened] == [tmp_path / "Output"]
    else:
        assert not opened
        assert "select" in window.details.toPlainText().lower()
    window.close()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows filenames cannot contain newlines")
def test_open_output_folder_with_newline_in_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    folder = tmp_path / "first\nsecond"
    folder.mkdir()
    source = folder / "sample.txt"
    source.write_text("Salom", encoding="utf-8")
    window = MainWindow()
    window.navigation.setCurrentRow(1)
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    opened = []
    monkeypatch.setattr("elevenlabs_toolkit.gui.QDesktopServices.openUrl", lambda url: opened.append(url) or True)
    window.open_output_folder()
    assert [Path(url.toLocalFile()) for url in opened] == [folder / "Output"]
    window.navigation.setCurrentRow(2)
    window.navigation.setCurrentRow(1)
    opened.clear()
    window.batch_table.selectRow(0)
    window.open_output_folder()
    assert [Path(url.toLocalFile()) for url in opened] == [folder / "Output"]
    window.close()


@pytest.mark.parametrize("folder_exists", [False, True])
def test_open_output_folder_explains_missing_folder_or_system_failure(
    tmp_path: Path, monkeypatch, folder_exists: bool
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "sample.txt"
    source.write_text("Salom", encoding="utf-8")
    window = MainWindow()
    window.navigation.setCurrentRow(1)
    window.add_paths([source])
    window.show_preview()
    assert window.worker.wait(5000)
    app.processEvents()
    folder = tmp_path / "Output"
    if folder_exists:
        folder.mkdir()
    calls = []
    monkeypatch.setattr("elevenlabs_toolkit.gui.QDesktopServices.openUrl", lambda url: calls.append(url) or False)
    window.open_output_folder()
    assert str(folder) in window.details.toPlainText()
    assert ("could not open" in window.details.toPlainText().lower()) is folder_exists
    assert ("does not exist" in window.details.toPlainText().lower()) is (not folder_exists)
    assert len(calls) == int(folder_exists)
    assert source.read_text(encoding="utf-8") == "Salom"
    window.close()


def test_open_selected_output_folder(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    source = tmp_path / "sample.txt"
    source.write_text("Salom", encoding="utf-8")
    window.navigation.setCurrentRow(1)
    window.output_subfolder.setChecked(False)
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    window.batch_table.selectRow(0)
    opened = []
    monkeypatch.setattr("elevenlabs_toolkit.gui.QDesktopServices.openUrl", lambda url: opened.append(url) or True)
    window.open_output_folder()
    assert Path(opened[0].toLocalFile()) == tmp_path
    window.close()


def test_save_batch_report_contains_results_and_preserves_input(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    source = tmp_path / "sample.txt"
    source.write_text("Salom", encoding="utf-8")
    window.navigation.setCurrentRow(1)
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    report = tmp_path / "report.json"
    monkeypatch.setattr("elevenlabs_toolkit.gui.QFileDialog.getSaveFileName", lambda *args: (str(report), ""))
    window.save_batch_report()
    rows = json.loads(report.read_text(encoding="utf-8"))["files"]
    assert rows[0]["input"] == str(source)
    assert rows[0]["status"] != "Pending"
    monkeypatch.setattr("elevenlabs_toolkit.gui.QFileDialog.getSaveFileName", lambda *args: (str(source), ""))
    window.save_batch_report()
    assert source.read_text(encoding="utf-8") == "Salom"
    window.close()


def test_batch_table_displays_plans_and_results(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    source = tmp_path / "sample.txt"
    source.write_text("Salom", encoding="utf-8")
    window.navigation.setCurrentRow(1)
    window.output_subfolder.setChecked(False)
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert window.batch_table.rowCount() == 1
    assert window.batch_table.item(0, 0).text() == str(source)
    assert window.batch_table.item(0, 1).text() == str(tmp_path / "sample CYR.txt")
    assert window.batch_table.item(0, 2).text() != "Pending"
    window.close()


@pytest.mark.parametrize(
    ("filename", "mode", "content", "output_name"),
    [
        ("04 T LTN.srt", "cyrillic", "Salom", "04 T CYR.srt"),
        ("04 T CYR.srt", "latin", "Салом", "04 T LTN.srt"),
        ("04 T.srt", "cyrillic", "Salom", "04 T CYR.srt"),
        ("04 T.srt", "latin", "Салом", "04 T LTN.srt"),
    ],
)
def test_gui_and_cli_transliteration_names_and_contents_match(
    tmp_path: Path, monkeypatch, capsys, filename: str, mode: str, content: str, output_name: str
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    source = tmp_path / filename
    source.write_text(f"1\n00:00:00,000 --> 00:00:01,000\n{content}\n", encoding="utf-8")
    original = source.read_bytes()
    cli_output, gui_output = tmp_path / "cli", tmp_path / "gui"
    assert main(["transliterate", str(source), "--to", mode, "-o", str(cli_output)]) == 0
    capsys.readouterr()
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.navigation.setCurrentRow(1)
    window.output_subfolder.setChecked(False)
    window.mode.setCurrentText(mode.title())
    window.output_directory.setText(str(gui_output))
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert (gui_output / output_name).read_bytes() == (cli_output / output_name).read_bytes()
    assert source.read_bytes() == original
    window.close()


def test_gui_and_cli_keep_distinct_outputs_for_same_named_inputs(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    sources = []
    originals = []
    for folder, content in (("first", "Salom"), ("second", "Dunyo")):
        source = tmp_path / "inputs" / folder / "lesson LTN.srt"
        source.parent.mkdir(parents=True)
        source.write_text(f"1\n00:00:00,000 --> 00:00:01,000\n{content}\n", encoding="utf-8")
        sources.append(source)
        originals.append(source.read_bytes())

    cli_output, gui_output = tmp_path / "cli", tmp_path / "gui"
    assert main(["transliterate", *(str(source) for source in sources), "--to", "cyrillic", "-o", str(cli_output)]) == 0
    capsys.readouterr()
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.navigation.setCurrentRow(1)
    window.output_subfolder.setChecked(False)
    window.mode.setCurrentText("Cyrillic")
    window.output_directory.setText(str(gui_output))
    window.add_paths(sources)
    window.show_preview()
    assert window.worker.wait(5000)
    app.processEvents()
    assert not gui_output.exists()
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    for folder in ("first", "second"):
        name = Path(folder) / "lesson CYR.srt"
        assert (gui_output / name).read_bytes() == (cli_output / name).read_bytes()
    assert [source.read_bytes() for source in sources] == originals
    window.close()


@pytest.mark.parametrize("policy", ["skip", "replace", "rename", "error"])
def test_transliteration_conflicts_match_cli(tmp_path: Path, monkeypatch, capsys, policy: str) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    source = tmp_path / "04 T LTN.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nSalom\n", encoding="utf-8")
    original = source.read_bytes()
    cli_dir, gui_dir = tmp_path / "cli", tmp_path / "gui"
    for folder in (cli_dir, gui_dir):
        folder.mkdir()
        (folder / "04 T CYR.srt").write_text("existing", encoding="utf-8")
    exit_code = main(["transliterate", str(source), "--to", "cyrillic", "--on-conflict", policy, "-o", str(cli_dir)])
    capsys.readouterr()
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.navigation.setCurrentRow(1)
    window.output_subfolder.setChecked(False)
    window.conflict.setCurrentIndex(window.conflict.findData(policy))
    window.output_directory.setText(str(gui_dir))
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert (exit_code != 0) == ("Preflight failed" in window.details.toPlainText())
    assert {path.name: path.read_bytes() for path in gui_dir.iterdir()} == {
        path.name: path.read_bytes() for path in cli_dir.iterdir()
    }
    assert source.read_bytes() == original
    window.close()


def test_transcription_conflict_preview_matches_cli_request_count(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    sources = [tmp_path / "a.wav", tmp_path / "b.wav"]
    for source in sources:
        source.write_bytes(b"audio")
    cli_dir, gui_dir = tmp_path / "cli", tmp_path / "gui"
    cli_dir.mkdir()
    gui_dir.mkdir()
    (cli_dir / "a.json").write_text("existing", encoding="utf-8")
    (gui_dir / "a.wav.transcript.json").write_text("existing", encoding="utf-8")
    assert (
        main(
            [
                "--json",
                "transcribe",
                *(str(source) for source in sources),
                "-o",
                str(cli_dir),
                "--on-conflict",
                "error",
                "--dry-run",
            ]
        )
        != 0
    )
    cli = json.loads(capsys.readouterr().out)
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.project_directory = tmp_path
    window.output_directory.setText(str(gui_dir))
    window.conflict.setCurrentIndex(window.conflict.findData("error"))
    window.report_mode.setCurrentText("JSON")
    window.add_paths(sources)
    window.show_preview()
    assert window.worker.wait(5000)
    app.processEvents()
    preview = json.loads(window.details.toPlainText())
    assert preview["status"] == cli["status"] == "conflict"
    assert preview["api_requests"] == cli["api_requests"] == 0
    assert preview["max_api_attempts"] == 0
    assert preview["conflicts"][0]["target"] == str(gui_dir / "a.wav.transcript.json")
    assert window.batch_table.item(0, 2).text() == "Conflict"
    assert window.batch_table.item(1, 2).text() == "Blocked"
    assert "another" in window.batch_table.item(1, 3).text().lower()
    assert not (gui_dir / "b.wav.transcript.json").exists()
    window.close()


def test_preview_reports_source_overwrite_conflict_without_writing(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "input CYR.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nSalom\n", encoding="utf-8")
    original = source.read_bytes()
    assert (
        main(
            [
                "--json",
                "transliterate",
                str(source),
                "--to",
                "cyrillic",
                "-o",
                str(tmp_path),
                "--on-conflict",
                "replace",
                "--dry-run",
            ]
        )
        != 0
    )
    cli = json.loads(capsys.readouterr().out)
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.navigation.setCurrentRow(1)
    window.output_subfolder.setChecked(False)
    window.conflict.setCurrentIndex(window.conflict.findData("replace"))
    window.report_mode.setCurrentText("JSON")
    window.add_paths([source])
    window.show_preview()
    assert window.worker.wait(5000)
    app.processEvents()
    preview = json.loads(window.details.toPlainText())
    assert preview["status"] == cli["status"] == "conflict"
    assert preview["conflicts"][0]["target"] == str(source)
    assert "overwrite" in window.batch_table.item(0, 3).text().lower()
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert source.read_bytes() == original
    window.close()


def test_preview_shows_conflicts_between_selected_inputs(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    sources = [tmp_path / "input LTN.srt", tmp_path / "input.latin.srt"]
    for source in sources:
        source.write_text("1\n00:00:00,000 --> 00:00:01,000\nSalom\n", encoding="utf-8")
    originals = [source.read_bytes() for source in sources]
    output = tmp_path / "out"
    assert (
        main(
            [
                "--json",
                "transliterate",
                *(str(source) for source in sources),
                "--to",
                "cyrillic",
                "-o",
                str(output),
                "--dry-run",
            ]
        )
        != 0
    )
    cli = json.loads(capsys.readouterr().out)
    window = MainWindow()
    window.navigation.setCurrentRow(1)
    window.output_directory.setText(str(output))
    window.report_mode.setCurrentText("JSON")
    window.add_paths(sources)
    window.show_preview()
    assert window.worker.wait(5000)
    app.processEvents()
    preview = json.loads(window.details.toPlainText())
    assert preview["status"] == cli["status"] == "conflict"
    assert len(preview["conflicts"]) == 1
    assert set(preview["conflicts"][0]["sources"]) == {str(source) for source in sources}
    assert all(window.batch_table.item(row, 2).text() == "Conflict" for row in range(2))
    assert not output.exists()
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert "failed" in window.details.toPlainText().lower()
    assert not output.exists()
    assert [source.read_bytes() for source in sources] == originals
    window.close()


def test_preview_shows_existing_output_conflict_without_writing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "input.txt"
    source.write_text("Salom", encoding="utf-8")
    target = tmp_path / "input CYR.txt"
    target.write_text("existing", encoding="utf-8")
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.navigation.setCurrentRow(1)
    window.output_subfolder.setChecked(False)
    window.conflict.setCurrentIndex(window.conflict.findData("error"))
    window.add_paths([source])
    window.show_preview()
    assert window.worker.wait(5000)
    app.processEvents()
    assert window.batch_table.item(0, 1).text() == str(target)
    assert "conflict" in window.batch_table.item(0, 2).text().lower()
    assert "already exists" in window.batch_table.item(0, 3).text().lower()
    assert "conflict" in window.details.toPlainText().lower()
    assert target.read_text(encoding="utf-8") == "existing"
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert "Preflight failed" in window.details.toPlainText()
    assert target.read_text(encoding="utf-8") == "existing"
    window.close()


def test_conversion_skip_existing_matches_cli_when_source_is_invalid(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "bad.srt"
    source.write_bytes(b"\xff")
    cli_dir = tmp_path / "cli"
    gui_dir = tmp_path / "gui"
    cli_dir.mkdir()
    gui_dir.mkdir()
    for folder in (cli_dir, gui_dir):
        (folder / "bad CYR.srt").write_text("existing", encoding="utf-8")
    code = main(["transliterate", str(source), "--to", "cyrillic", "--on-conflict", "skip", "-o", str(cli_dir)])
    capsys.readouterr()
    assert code != 0

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.navigation.setCurrentRow(1)
    window.output_directory.setText(str(gui_dir))
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert window.batch_table.item(0, 2).text() == "failed"
    assert (gui_dir / "bad CYR.srt").read_text(encoding="utf-8") == "existing"
    assert source.read_bytes() == b"\xff"
    window.close()


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


@pytest.mark.parametrize(
    ("format_name", "suffix"),
    [("resolve-edl", "resolve.edl"), ("cue-index-srt", "cue-index.srt"), ("clean-json", "clean.json")],
)
def test_advanced_export_formats_match_cli(tmp_path: Path, monkeypatch, format_name: str, suffix: str) -> None:
    monkeypatch.chdir(tmp_path)
    source = transcript(tmp_path / "input.json")
    cli_dir, gui_dir = tmp_path / "cli", tmp_path / "gui"
    args = ["export", str(source), "--format", format_name, "-o", str(cli_dir)]
    if format_name == "clean-json":
        args += ["--clean", "uzbek"]
    if format_name == "resolve-edl":
        args += ["--marker-fps", "30", "--marker-color", "ResolveColorRed", "--marker-prefix", "Cue"]
    assert main(args) == 0
    settings = DesktopSettings(
        project_directory=tmp_path,
        output_directory=gui_dir,
        cleanup="uzbek" if format_name == "clean-json" else None,
        marker_fps=30.0 if format_name == "resolve-edl" else 25.0,
        marker_color="ResolveColorRed" if format_name == "resolve-edl" else "ResolveColorBlue",
        marker_prefix="Cue" if format_name == "resolve-edl" else "Sentence",
    )
    worker = Worker("Export", ScriptMode.SOURCE, [source], False, (ArtifactFormat(format_name),), settings=settings)
    messages = []
    worker.finished_result.connect(messages.append, Qt.ConnectionType.DirectConnection)
    worker.run()
    assert "failed" not in messages[0].lower()
    assert (gui_dir / f"input.{suffix}").read_bytes() == (cli_dir / f"input.{suffix}").read_bytes()


def test_combined_text_export_matches_cli(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    sources = [transcript(tmp_path / "z.json"), transcript(tmp_path / "a.json")]
    cli_dir, gui_dir = tmp_path / "cli", tmp_path / "gui"
    assert (
        main(
            [
                "export",
                *(str(source) for source in sources),
                "--format",
                "combined-txt",
                "--combined-name",
                "all.txt",
                "-o",
                str(cli_dir),
            ]
        )
        == 0
    )
    settings = DesktopSettings(project_directory=tmp_path, output_directory=gui_dir, combined_name="all.txt")
    worker = Worker("Export", ScriptMode.SOURCE, sources, False, (ArtifactFormat.COMBINED_TXT,), settings=settings)
    worker.run()
    assert (gui_dir / "all.txt").read_bytes() == (cli_dir / "all.txt").read_bytes()


def test_gui_combined_export_keeps_cli_natural_input_order(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in ("lesson 10.json", "lesson 2.json"):
        transcript(inputs / name)
    cli_dir, gui_dir = tmp_path / "cli", tmp_path / "gui"
    assert main(["export", str(inputs), "--format", "combined-txt", "-o", str(cli_dir)]) == 0
    capsys.readouterr()
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(2)
    window.output_subfolder.setChecked(False)
    window.output_directory.setText(str(gui_dir))
    window.export_formats[ArtifactFormat.SRT].setChecked(False)
    window.export_formats[ArtifactFormat.COMBINED_TXT].setChecked(True)
    window.add_paths([inputs])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    gui_content = (gui_dir / "combined.txt").read_bytes()
    assert gui_content == (cli_dir / "combined.txt").read_bytes()
    assert gui_content.index(b"lesson 2") < gui_content.index(b"lesson 10")
    window.close()


@pytest.mark.parametrize("explicit_cleanup", [False, True])
def test_combined_text_and_clean_json_share_cli_cleanup(
    tmp_path: Path, monkeypatch, capsys, explicit_cleanup: bool
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("elevenlabs_toolkit.config.user_config_path", lambda: tmp_path / "missing-user.toml")
    source = tmp_path / "input.json"
    source.write_text(
        json.dumps(
            {
                "text": "man manga",
                "words": [{"text": "man", "start": 0, "end": 0.3}, {"text": "manga", "start": 0.4, "end": 0.8}],
            }
        ),
        encoding="utf-8",
    )
    original = source.read_bytes()
    cli_dir, gui_dir = tmp_path / "cli", tmp_path / "gui"
    args = ["export", str(source), "--format", "combined-txt", "--format", "clean-json", "-o", str(cli_dir)]
    if explicit_cleanup:
        args.extend(["--clean", "uzbek"])
    assert main(args) == 0
    capsys.readouterr()
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(2)
    window.output_subfolder.setChecked(False)
    window.output_directory.setText(str(gui_dir))
    window.export_formats[ArtifactFormat.SRT].setChecked(False)
    window.export_formats[ArtifactFormat.COMBINED_TXT].setChecked(True)
    window.export_formats[ArtifactFormat.CLEAN_JSON].setChecked(True)
    if explicit_cleanup:
        window.export_text.cleanup.setCurrentText("Uzbek editorial cleanup")
    window.show_workflow_configuration()
    report = json.loads(window.details.toPlainText())
    assert report["effective"]["text"]["cleanup"] == "uzbek"
    assert report["setting_sources"]["text.cleanup"] == (
        "workflow override" if explicit_cleanup else "automatic for cleaned JSON export"
    )
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert (gui_dir / "combined.txt").read_bytes() == (cli_dir / "combined.txt").read_bytes()
    assert (gui_dir / "input.clean.json").read_bytes() == (cli_dir / "input.clean.json").read_bytes()
    assert source.read_bytes() == original
    window.close()


def test_default_combined_exports_group_folders_and_match_cli_order(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "inputs"
    folders = [root / "one", root / "two"]
    originals = {}
    for folder in folders:
        folder.mkdir(parents=True)
        for name in ("lesson 10.json", "lesson 2.json"):
            source = transcript(folder / name)
            originals[source] = source.read_bytes()
        assert (
            main(["export", str(folder), "--format", "combined-txt", "-o", str(tmp_path / f"cli-{folder.name}")]) == 0
        )
        capsys.readouterr()
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(2)
    window.export_formats[ArtifactFormat.SRT].setChecked(False)
    window.export_formats[ArtifactFormat.COMBINED_TXT].setChecked(True)
    window.add_paths([root])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    for folder in folders:
        content = (folder / "Output" / "combined.txt").read_bytes()
        assert content == (tmp_path / f"cli-{folder.name}" / "combined.txt").read_bytes()
        assert content.index(b"lesson 2") < content.index(b"lesson 10")
    assert all(source.read_bytes() == content for source, content in originals.items())
    window.close()


def test_clean_json_with_cleanup_disabled_matches_cli_and_writes_nothing(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    source = transcript(tmp_path / "input.json")
    original = source.read_bytes()
    cli_dir, gui_dir = tmp_path / "cli", tmp_path / "gui"
    assert (
        main(
            ["export", str(source), "--format", "srt", "--format", "clean-json", "--clean", "none", "-o", str(cli_dir)]
        )
        != 0
    )
    assert "clean" in capsys.readouterr().err.lower()
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(2)
    window.output_subfolder.setChecked(False)
    window.output_directory.setText(str(gui_dir))
    window.export_formats[ArtifactFormat.CLEAN_JSON].setChecked(True)
    window.export_text.cleanup.setCurrentText("None")
    window.add_paths([source])
    window.start_job()
    app.processEvents()
    assert window.export_text.cleanup_error.text()
    assert window.start.isEnabled()
    assert "clean" in window.details.toPlainText().lower()
    assert not cli_dir.exists()
    assert not gui_dir.exists()
    assert source.read_bytes() == original
    window.export_formats[ArtifactFormat.CLEAN_JSON].setChecked(False)
    assert not window.export_text.cleanup_error.text()
    window.export_formats[ArtifactFormat.CLEAN_JSON].setChecked(True)
    window.export_text.cleanup.setCurrentText("Uzbek editorial cleanup")
    assert not window.export_text.cleanup_error.text()
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert (gui_dir / "input.clean.json").exists()
    window.close()


def test_gui_resolve_controls_match_cli_output(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    app = QApplication.instance() or QApplication([])
    source = transcript(tmp_path / "input.json")
    cli_dir = tmp_path / "cli"
    assert (
        main(
            [
                "export",
                str(source),
                "--format",
                "resolve-edl",
                "--marker-fps",
                "30",
                "--marker-color",
                "ResolveColorRed",
                "--marker-prefix",
                "Cue",
                "-o",
                str(cli_dir),
            ]
        )
        == 0
    )
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(2)
    window.output_subfolder.setChecked(False)
    window.export_formats[ArtifactFormat.SRT].setChecked(False)
    window.export_formats[ArtifactFormat.RESOLVE_EDL].setChecked(True)
    window.export_text.marker_fps.setText("30")
    window.export_text.marker_color.setText("ResolveColorRed")
    window.export_text.marker_prefix.setText("Cue")
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert (tmp_path / "input.resolve.edl").read_bytes() == (cli_dir / "input.resolve.edl").read_bytes()
    window.close()


def test_gui_advanced_subtitle_controls_match_cli(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    app = QApplication.instance() or QApplication([])
    source = transcript(tmp_path / "input.json")
    cli_dir = tmp_path / "cli"
    args = [
        "export",
        str(source),
        "-o",
        str(cli_dir),
        "--profile",
        "social",
        "--no-pause-detection",
        "--max-chars-per-line",
        "16",
        "--max-lines",
        "3",
        "--max-duration",
        "3.5",
        "--srt-fps",
        "60",
        "--srt-padding-frames",
        "1",
        "--srt-gap-ms",
        "20",
        "--srt-smart-line-breaks",
    ]
    assert main(args) == 0
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(2)
    window.output_subfolder.setChecked(False)
    panel = window.export_text
    panel.profile.setCurrentText("social")
    panel.boolean_fields["pause_detection"].setCurrentIndex(2)
    for name, value in {
        "max_chars_per_line": "16",
        "max_lines": "3",
        "max_duration": "3.5",
        "srt_fps": "60",
        "srt_padding_frames": "1",
        "srt_gap_milliseconds": "20",
    }.items():
        panel.numeric_fields[name][0].setText(value)
    panel.smart.setChecked(True)
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert (tmp_path / "input.srt").read_bytes() == (cli_dir / "input.srt").read_bytes()
    window.close()


@pytest.mark.parametrize("language", ["uz", "en", None])
def test_clean_language_validation_matches_cli(tmp_path: Path, monkeypatch, capsys, language: str | None) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("elevenlabs_toolkit.config.user_config_path", lambda: tmp_path / "missing-user.toml")
    source = tmp_path / "input.json"
    payload = {"text": "Salom dunyo.", "words": [{"text": "Salom", "start": 0, "end": 0.5}]}
    if language is not None:
        payload["language_code"] = language
    source.write_text(json.dumps(payload), encoding="utf-8")
    original = source.read_bytes()
    cli_dir, gui_dir = tmp_path / "cli", tmp_path / "gui"
    status = main(["clean", str(source), "-o", str(cli_dir)])
    capsys.readouterr()
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(3)
    window.output_subfolder.setChecked(False)
    window.output_directory.setText(str(gui_dir))
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    gui_output, cli_output = gui_dir / "input.clean.json", cli_dir / "input.clean.json"
    if language == "en":
        assert status != 0
        assert not gui_output.exists() and not cli_output.exists()
        assert window.batch_table.item(0, 2).text() == "failed"
        assert "Uzbek-only" in window.batch_table.item(0, 3).text()
    else:
        assert status == 0
        assert gui_output.read_bytes() == cli_output.read_bytes()
    assert source.read_bytes() == original
    window.close()


def test_gui_clean_script_and_replacements_match_cli(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    app = QApplication.instance() or QApplication([])
    source = transcript(tmp_path / "input.json")
    before = source.read_bytes()
    cli_dir = tmp_path / "cli"
    assert main(["clean", str(source), "--script", "cyrillic", "--replace", "Salom=Assalom", "-o", str(cli_dir)]) == 0
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(3)
    window.output_subfolder.setChecked(False)
    window.clean_text.script.setCurrentText("Cyrillic")
    window.clean_text.override_replacements.setChecked(True)
    window.clean_text.replacements.setPlainText("Salom=Assalom")
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert (tmp_path / "input.clean.json").read_bytes() == (cli_dir / "input.clean.json").read_bytes()
    assert source.read_bytes() == before
    window.close()


@pytest.mark.parametrize("policy", ["skip", "replace", "error", "rename"])
def test_combined_custom_name_conflicts_match_cli(tmp_path: Path, monkeypatch, capsys, policy: str) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    sources = [transcript(tmp_path / name) for name in ("a.json", "b.json")]
    originals = {source: source.read_bytes() for source in sources}
    cli_dir, gui_dir = tmp_path / "cli", tmp_path / "gui"
    for directory in (cli_dir, gui_dir):
        directory.mkdir()
        (directory / "all.txt").write_text("existing", encoding="utf-8")
    code = main(
        [
            "export",
            *map(str, sources),
            "--format",
            "combined-txt",
            "--combined-name",
            "all.txt",
            "--on-conflict",
            policy,
            "-o",
            str(cli_dir),
        ]
    )
    capsys.readouterr()
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(2)
    window.output_subfolder.setChecked(False)
    window.output_directory.setText(str(gui_dir))
    window.conflict.setCurrentIndex(["skip", "replace", "error", "rename"].index(policy))
    window.export_formats[ArtifactFormat.SRT].setChecked(False)
    window.export_formats[ArtifactFormat.COMBINED_TXT].setChecked(True)
    window.export_text.combined_name.setText("all.txt")
    window.add_paths(sources)
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert (code != 0) == (policy == "error")
    assert {p.name: p.read_bytes() for p in gui_dir.iterdir()} == {p.name: p.read_bytes() for p in cli_dir.iterdir()}
    assert all(source.read_bytes() == originals[source] for source in sources)
    window.close()


@pytest.mark.parametrize("policy", ["skip", "replace", "error", "rename"])
def test_clean_conflict_policy_matches_cli(tmp_path: Path, monkeypatch, capsys, policy: str) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    source = transcript(tmp_path / "input.json")
    original = source.read_bytes()
    cli_dir, gui_dir = tmp_path / "cli", tmp_path / "gui"
    for directory in (cli_dir, gui_dir):
        directory.mkdir()
        (directory / "input.clean.json").write_text("existing", encoding="utf-8")
    code = main(["clean", str(source), "--on-conflict", policy, "-o", str(cli_dir)])
    capsys.readouterr()
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(3)
    window.output_subfolder.setChecked(False)
    window.output_directory.setText(str(gui_dir))
    window.conflict.setCurrentIndex(["skip", "replace", "error", "rename"].index(policy))
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    assert (code != 0) == (policy == "error")
    assert {p.name: p.read_bytes() for p in gui_dir.iterdir()} == {p.name: p.read_bytes() for p in cli_dir.iterdir()}
    assert source.read_bytes() == original
    window.close()


@pytest.mark.parametrize("policy", ["skip", "replace", "error", "rename"])
def test_export_conflict_policy_matches_cli(tmp_path: Path, monkeypatch, policy: str) -> None:
    monkeypatch.chdir(tmp_path)
    source = transcript(tmp_path / "input.json")
    original = source.read_bytes()
    cli_dir, gui_dir = tmp_path / "cli", tmp_path / "gui"
    for directory in (cli_dir, gui_dir):
        directory.mkdir()
        (directory / "input.srt").write_text("existing", encoding="utf-8")
    exit_code = main(["export", str(source), "--format", "srt", "--on-conflict", policy, "-o", str(cli_dir)])
    worker = Worker(
        "Export",
        ScriptMode.SOURCE,
        [source],
        False,
        settings=DesktopSettings(project_directory=tmp_path, output_directory=gui_dir),
        policy=ConflictPolicy(policy),
    )
    reports = []
    worker.finished_result.connect(reports.append, Qt.ConnectionType.DirectConnection)
    worker.run()
    assert (exit_code != 0) == ("failed" in reports[0].lower())
    assert {p.name: p.read_bytes() for p in gui_dir.iterdir()} == {p.name: p.read_bytes() for p in cli_dir.iterdir()}
    assert source.read_bytes() == original


@pytest.mark.parametrize("suffix", ["srt", "txt", "mini.srt"])
def test_desktop_export_matches_cli_bytes(tmp_path: Path, monkeypatch, suffix: str) -> None:
    monkeypatch.chdir(tmp_path)
    source = transcript(tmp_path / "input.json")
    format_name = {"srt": "srt", "txt": "txt", "mini.srt": "srt-mini"}[suffix]
    cli_output = tmp_path / "cli"
    desktop_output = tmp_path / "desktop"
    result = main(["export", str(source), "--format", format_name, "-o", str(cli_output)])
    assert result == 0
    settings = DesktopSettings(project_directory=tmp_path, output_directory=desktop_output)
    worker = Worker(
        "Export",
        ScriptMode.SOURCE,
        [source],
        False,
        (ArtifactFormat(format_name),),
        settings=settings,
        policy=ConflictPolicy.ERROR,
    )
    messages = []
    worker.finished_result.connect(messages.append, Qt.ConnectionType.DirectConnection)
    worker.run()
    assert "failed" not in messages[0].lower()
    assert (desktop_output / f"input.{suffix}").read_bytes() == (cli_output / f"input.{suffix}").read_bytes()


def test_desktop_folder_filter_matches_cli_conversion(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    inputs = tmp_path / "input"
    inputs.mkdir()
    (inputs / "keep.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nSalom\n", encoding="utf-8")
    (inputs / "ignore.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nSalom\n", encoding="utf-8")
    cli_output, desktop_output = tmp_path / "cli", tmp_path / "desktop"
    assert main(["transliterate", str(inputs), "--glob", "keep*", "--to", "cyrillic", "-o", str(cli_output)]) == 0
    worker = Worker(
        "Convert Script",
        ScriptMode.CYRILLIC,
        [],
        False,
        settings=DesktopSettings(project_directory=tmp_path, output_directory=desktop_output),
        policy=ConflictPolicy.ERROR,
        selection=InputSpec((inputs,), glob="keep*", recursive=True),
    )
    worker.run()
    assert (desktop_output / "keep CYR.srt").read_bytes() == (cli_output / "keep CYR.srt").read_bytes()
    assert sorted(path.name for path in desktop_output.iterdir()) == ["keep CYR.srt"]


@pytest.mark.parametrize(("kind", "pattern"), [("glob", "*keep*.srt"), ("regex", r"keep.*\.srt$")])
@pytest.mark.parametrize("recursive", [False, True])
def test_gui_folder_filter_and_recursion_match_cli(
    tmp_path: Path, monkeypatch, kind: str, pattern: str, recursive: bool
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    app = QApplication.instance() or QApplication([])
    folder = tmp_path / "sources"
    nested = folder / "nested"
    nested.mkdir(parents=True)
    for path in (folder / "keep-a.srt", folder / "skip.srt", nested / "keep-b.srt"):
        path.write_text("1\n00:00:00,000 --> 00:00:01,000\nSalom\n", encoding="utf-8")
    cli_dir, gui_dir = tmp_path / "cli", tmp_path / "gui"
    args = ["transliterate", str(folder), "--to", "cyrillic", f"--{kind}", pattern, "-o", str(cli_dir)]
    args.append("--recursive" if recursive else "--no-recursive")
    assert main(args) == 0
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(1)
    window.output_subfolder.setChecked(False)
    window.output_directory.setText(str(gui_dir))
    window.filter_kind.setCurrentIndex(1 if kind == "glob" else 2)
    window.pattern.setText(pattern)
    window.recursive.setChecked(recursive)
    window.add_paths([folder])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    cli_outputs = {path.relative_to(cli_dir): path.read_bytes() for path in cli_dir.rglob("*.srt")}
    gui_outputs = {path.relative_to(gui_dir): path.read_bytes() for path in gui_dir.rglob("*.srt")}
    assert gui_outputs == cli_outputs
    assert len(gui_outputs) == (2 if recursive else 1)
    window.close()


@pytest.mark.parametrize(("kind", "pattern"), [("glob", "missing*.srt"), ("regex", "[")])
def test_gui_and_cli_report_invalid_or_empty_folder_filters(
    tmp_path: Path, monkeypatch, kind: str, pattern: str
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    app = QApplication.instance() or QApplication([])
    folder = tmp_path / "sources"
    folder.mkdir()
    (folder / "sample.srt").write_text("Salom", encoding="utf-8")
    cli_dir, gui_dir = tmp_path / "cli", tmp_path / "gui"
    assert main(["transliterate", str(folder), "--to", "cyrillic", f"--{kind}", pattern, "-o", str(cli_dir)]) != 0
    window = MainWindow()
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(1)
    window.output_directory.setText(str(gui_dir))
    window.filter_kind.setCurrentIndex(1 if kind == "glob" else 2)
    window.pattern.setText(pattern)
    window.add_paths([folder])
    window.start_job()
    if kind == "glob":
        assert window.worker.wait(5000)
        app.processEvents()
    assert "Preflight failed" in window.details.toPlainText()
    assert not cli_dir.exists()
    assert not gui_dir.exists()
    window.close()


def test_desktop_export_script_override_matches_cli_project_configuration(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "elevenlabs-toolkit.toml").write_text('[text]\nscript = "cyrillic"\n', encoding="utf-8")
    source = transcript(tmp_path / "input.json")
    cli_output, desktop_output = tmp_path / "cli", tmp_path / "desktop"
    assert main(["export", str(source), "--format", "txt", "--script", "latin", "-o", str(cli_output)]) == 0
    settings = DesktopSettings(
        project_directory=tmp_path, output_directory=desktop_output, output_script=ScriptMode.LATIN
    )
    worker = Worker(
        "Export",
        ScriptMode.SOURCE,
        [source],
        False,
        (ArtifactFormat.TXT,),
        settings=settings,
        policy=ConflictPolicy.ERROR,
    )
    worker.run()
    assert (desktop_output / "input.txt").read_bytes() == (cli_output / "input.txt").read_bytes()
    assert "Salom" in (desktop_output / "input.txt").read_text(encoding="utf-8")


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
    gui_clean = json.loads((tmp_path / "source.clean.json").read_text(encoding="utf-8"))
    assert gui_clean == json.loads((tmp_path / "cli/source.clean.json").read_text(encoding="utf-8"))
    assert "Салом" in gui_clean["text"]
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


@pytest.mark.parametrize(
    "payload",
    [
        {"segments": [{"text": "Salom", "start": 1, "end": 2}]},
        {"words": [{"text": "Salom", "start": None, "end": None}]},
        {"words": [{"text": "Salom", "start": 0, "end": 1}, {"type": "punctuation", "text": "!"}]},
    ],
)
def test_inspect_legacy_transcript_variants_match_cli_without_writes(
    tmp_path: Path, monkeypatch, capsys, payload
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    source = tmp_path / "legacy.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    original = source.read_bytes()
    assert main(["--json", "inspect", str(source)]) == 0
    cli = json.loads(capsys.readouterr().out)["transcripts"]
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.navigation.setCurrentRow(4)
    window.report_mode.setCurrentText("JSON")
    window.add_paths([source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    rows = json.loads(window.details.toPlainText())["results"]
    for row in rows:
        row.pop("sources")
    assert rows == cli
    assert rows[0]["valid"] is True
    assert source.read_bytes() == original
    assert list(tmp_path.iterdir()) == [source]
    window.close()


@pytest.mark.parametrize("include_generated", [False, True])
@pytest.mark.parametrize("recursive", [False, True])
def test_inspect_regex_recursion_and_generated_selection_match_cli(
    tmp_path: Path, monkeypatch, capsys, include_generated: bool, recursive: bool
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    root, nested = tmp_path / "inputs", tmp_path / "inputs" / "nested"
    nested.mkdir(parents=True)
    for name in (root / "lesson-a.json", root / "lesson.segmented.json", root / "other.json", nested / "lesson-b.json"):
        transcript(name)
    before = {path: path.read_bytes() for path in root.rglob("*.json")}
    args = [
        "--json",
        "inspect",
        str(root),
        "--regex",
        r"lesson.*\.json$",
        "--recursive" if recursive else "--no-recursive",
    ]
    if include_generated:
        args.append("--include-generated")
    assert main(args) == 0
    cli = json.loads(capsys.readouterr().out)["transcripts"]
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(4)
    window.filter_kind.setCurrentIndex(2)
    window.pattern.setText(r"lesson.*\.json$")
    window.recursive.setChecked(recursive)
    window.include_generated.setChecked(include_generated)
    window.report_mode.setCurrentText("JSON")
    window.add_paths([root])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    rows = json.loads(window.details.toPlainText())["results"]
    for row in rows:
        row.pop("sources")
    assert rows == cli
    assert len(rows) == 1 + int(recursive) + int(include_generated)
    assert {path: path.read_bytes() for path in root.rglob("*.json")} == before
    window.close()


@pytest.mark.parametrize("include_generated", [False, True])
@pytest.mark.parametrize("recursive", [False, True])
def test_clean_regex_recursion_and_generated_selection_match_cli(
    tmp_path: Path, monkeypatch, capsys, include_generated: bool, recursive: bool
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    root, nested = tmp_path / "inputs", tmp_path / "inputs" / "nested"
    nested.mkdir(parents=True)
    for name in (root / "lesson-a.json", root / "lesson.segmented.json", root / "other.json", nested / "lesson-b.json"):
        transcript(name)
    before = {path: path.read_bytes() for path in root.rglob("*.json")}
    cli_dir, gui_dir = tmp_path / "cli", tmp_path / "gui"
    args = [
        "clean",
        str(root),
        "--regex",
        r"lesson.*\.json$",
        "--recursive" if recursive else "--no-recursive",
        "-o",
        str(cli_dir),
    ]
    if include_generated:
        args.append("--include-generated")
    assert main(args) == 0
    capsys.readouterr()
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(3)
    window.output_subfolder.setChecked(False)
    window.output_directory.setText(str(gui_dir))
    window.filter_kind.setCurrentIndex(2)
    window.pattern.setText(r"lesson.*\.json$")
    window.recursive.setChecked(recursive)
    window.include_generated.setChecked(include_generated)
    window.add_paths([root])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    gui_outputs = {path.relative_to(gui_dir): path.read_bytes() for path in gui_dir.rglob("*.json")}
    cli_outputs = {path.relative_to(cli_dir): path.read_bytes() for path in cli_dir.rglob("*.json")}
    assert gui_outputs == cli_outputs
    assert len(gui_outputs) == 1 + int(recursive) + int(include_generated)
    assert {path: path.read_bytes() for path in root.rglob("*.json")} == before
    window.close()


@pytest.mark.parametrize("include_generated", [False, True])
def test_inspect_generated_and_malformed_files_matches_cli(
    tmp_path: Path, monkeypatch, capsys, include_generated: bool
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    app = QApplication.instance() or QApplication([])
    sources = tmp_path / "sources"
    sources.mkdir()
    original = transcript(sources / "valid.json")
    (sources / "bad.json").write_text("not json", encoding="utf-8")
    assert main(["clean", str(original), "-o", str(sources)]) == 0
    capsys.readouterr()
    before = {path: path.read_bytes() for path in sources.iterdir()}
    args = ["--json", "inspect", str(sources)]
    if include_generated:
        args.append("--include-generated")
    assert main(args) == 1
    cli = json.loads(capsys.readouterr().out)["transcripts"]
    window = MainWindow()
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(4)
    window.include_generated.setChecked(include_generated)
    window.report_mode.setCurrentText("JSON")
    window.add_paths([sources])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    rows = json.loads(window.details.toPlainText())["results"]
    for row in rows:
        row.pop("sources")
    assert rows == cli
    assert len(rows) == (3 if include_generated else 2)
    assert {path: path.read_bytes() for path in sources.iterdir()} == before
    window.close()


@pytest.mark.parametrize("include_generated", [False, True])
def test_export_generated_json_selection_matches_cli(
    tmp_path: Path, monkeypatch, capsys, include_generated: bool
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    app = QApplication.instance() or QApplication([])
    source = transcript(tmp_path / "valid.json")
    generated_dir = tmp_path / "generated"
    assert main(["clean", str(source), "-o", str(generated_dir)]) == 0
    capsys.readouterr()
    cli_dir, gui_dir = tmp_path / "cli", tmp_path / "gui"
    args = ["export", str(generated_dir), "--format", "txt", "-o", str(cli_dir)]
    if include_generated:
        args.append("--include-generated")
    code = main(args)
    capsys.readouterr()
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(2)
    window.output_subfolder.setChecked(False)
    window.output_directory.setText(str(gui_dir))
    window.export_formats[ArtifactFormat.SRT].setChecked(False)
    window.export_formats[ArtifactFormat.TXT].setChecked(True)
    window.include_generated.setChecked(include_generated)
    window.add_paths([generated_dir])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    if include_generated:
        assert code == 0
        assert (gui_dir / "valid.clean.txt").read_bytes() == (cli_dir / "valid.clean.txt").read_bytes()
    else:
        assert code != 0
        assert "Preflight failed" in window.details.toPlainText()
        assert not gui_dir.exists()
    window.close()


@pytest.mark.parametrize("workflow", [2, 3, 4])
@pytest.mark.parametrize("select_folder", [False, True])
def test_generated_input_error_names_gui_checkbox(
    tmp_path: Path, monkeypatch, workflow: int, select_folder: bool
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    folder = tmp_path / "inputs"
    folder.mkdir()
    source = transcript(folder / "sample.segmented.json")
    window = MainWindow()
    window.navigation.setCurrentRow(workflow)
    window.add_paths([folder if select_folder else source])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    error = window.details.toPlainText()
    assert "Include generated JSON" in error
    assert "exclude_generated=False" not in error
    assert source.is_file()
    window.close()


@pytest.mark.parametrize("include_generated", [False, True])
def test_clean_generated_input_selection_matches_cli(
    tmp_path: Path, monkeypatch, capsys, include_generated: bool
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    app = QApplication.instance() or QApplication([])
    source = transcript(tmp_path / "original.json")
    generated = tmp_path / "generated"
    assert main(["clean", str(source), "-o", str(generated)]) == 0
    capsys.readouterr()
    original_derivative = generated / "original.clean.json"
    before = original_derivative.read_bytes()
    cli_dir, gui_dir = tmp_path / "cli", tmp_path / "gui"
    args = ["clean", str(generated), "-o", str(cli_dir)]
    if include_generated:
        args.append("--include-generated")
    cli_result = main(args)
    capsys.readouterr()
    window = MainWindow()
    window.output_subfolder.setChecked(False)
    window.project_directory = tmp_path
    window.navigation.setCurrentRow(3)
    window.output_subfolder.setChecked(False)
    window.include_generated.setChecked(include_generated)
    window.output_directory.setText(str(gui_dir))
    window.add_paths([generated])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    if include_generated:
        assert cli_result == 0
        assert (gui_dir / "original.clean.clean.json").read_bytes() == (
            cli_dir / "original.clean.clean.json"
        ).read_bytes()
    else:
        assert cli_result != 0
        assert "Include generated JSON" in window.details.toPlainText()
        assert not gui_dir.exists()
    assert original_derivative.read_bytes() == before
    window.close()


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


@pytest.mark.parametrize("profile", [None, "custom"])
def test_configuration_view_matches_cli_with_user_project_and_profile_settings(
    tmp_path: Path, monkeypatch, capsys, profile: str | None
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "dummy-secret-not-for-display")
    user_config = tmp_path / "user.toml"
    user_config.write_text("[segmentation]\nmax_lines = 3\n", encoding="utf-8")
    monkeypatch.setattr("elevenlabs_toolkit.config.user_config_path", lambda: user_config)
    project = tmp_path / "project"
    project.mkdir()
    (project / "elevenlabs-toolkit.toml").write_text(
        "[segmentation]\nmax_lines = 4\n[profiles.custom.segmentation]\nmax_lines = 5\n", encoding="utf-8"
    )
    monkeypatch.chdir(project)
    args = ["--json", "config", "show"]
    if profile:
        args.extend(["--profile", profile])
    assert main(args) == 0
    cli = json.loads(capsys.readouterr().out)

    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings)
    window.project_directory = project
    window.refresh_profiles()
    window.navigation.setCurrentRow(5)
    if profile:
        window.config_profile.setCurrentText(profile)
    window.show_configuration()
    displayed = window.details.toPlainText()
    assert json.loads(displayed) == cli
    assert json.loads(displayed)["effective"]["segmentation"]["max_lines"] == (5 if profile else 4)
    assert "dummy-secret-not-for-display" not in displayed
    window.close()
    app.processEvents()


@pytest.mark.parametrize("case", ["invalid_project", "unknown_profile"])
def test_configuration_errors_match_cli_and_gui_can_recover(tmp_path: Path, monkeypatch, capsys, case: str) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr("elevenlabs_toolkit.config.user_config_path", lambda: tmp_path / "missing-user.toml")
    project = tmp_path / "project"
    project.mkdir()
    config_file = project / "elevenlabs-toolkit.toml"
    config_file.write_text(
        "[segmentation]\nunknown_option = 4\n" if case == "invalid_project" else "", encoding="utf-8"
    )
    monkeypatch.chdir(project)
    args = ["--json", "config", "show"]
    if case == "unknown_profile":
        args.extend(["--profile", "does-not-exist"])
    assert main(args) == 2
    cli_error = json.loads(capsys.readouterr().out)["message"]

    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings)
    window.project_directory = project
    window.refresh_profiles()
    window.navigation.setCurrentRow(5)
    if case == "unknown_profile":
        window.config_profile.setCurrentText("does-not-exist")
    window.show_configuration()
    assert window.details.toPlainText() == f"Configuration error: {cli_error}"

    config_file.write_text("[profiles.fixed.segmentation]\nmax_lines = 4\n", encoding="utf-8")
    window.config_profile.setCurrentText("Project default")
    window.refresh_profiles()
    window.config_profile.setCurrentText("fixed")
    window.show_configuration()
    assert json.loads(window.details.toPlainText())["effective"]["segmentation"]["max_lines"] == 4
    window.close()
    app.processEvents()


def test_configuration_display_updates_when_profile_or_project_changes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr("elevenlabs_toolkit.config.user_config_path", lambda: tmp_path / "missing-user.toml")
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "elevenlabs-toolkit.toml").write_text("[profiles.custom.segmentation]\nmax_lines = 4\n", encoding="utf-8")
    (second / "elevenlabs-toolkit.toml").write_text("[segmentation]\nmax_lines = 5\n", encoding="utf-8")
    app = QApplication.instance() or QApplication([])
    window = MainWindow(settings=QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    monkeypatch.setattr("elevenlabs_toolkit.gui.QFileDialog.getExistingDirectory", lambda *args: str(first))
    window.select_project()
    window.navigation.setCurrentRow(5)
    window.config_profile.setCurrentText("custom")
    assert json.loads(window.details.toPlainText())["effective"]["segmentation"]["max_lines"] == 4
    window.config_profile.setCurrentText("Project default")
    monkeypatch.setattr("elevenlabs_toolkit.gui.QFileDialog.getExistingDirectory", lambda *args: str(second))
    window.select_project()
    assert json.loads(window.details.toPlainText())["effective"]["segmentation"]["max_lines"] == 5
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
    window.output_subfolder.setChecked(False)
    window.navigation.setCurrentRow(2)
    window.output_subfolder.setChecked(False)
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


def test_json_export_summary_matches_cli_for_mixed_results(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "bad.json").write_text("invalid", encoding="utf-8")
    transcript(inputs / "good.json")
    cli_dir, gui_dir = tmp_path / "cli", tmp_path / "gui"
    assert main(["--json", "export", str(inputs), "-o", str(cli_dir)]) != 0
    cli = json.loads(capsys.readouterr().out)
    window = MainWindow()
    window.navigation.setCurrentRow(2)
    window.output_directory.setText(str(gui_dir))
    window.report_mode.setCurrentText("JSON")
    window.add_paths([inputs])
    window.start_job()
    assert window.worker.wait(5000)
    app.processEvents()
    report = json.loads(window.details.toPlainText())
    for key in ("status", "written", "skipped", "failed"):
        assert report[key] == cli[key]
    assert sorted(item["status"] for item in report["artifacts"]) == sorted(item["status"] for item in cli["artifacts"])
    assert len(report["results"]) == 2
    window.close()


def test_json_conversion_summary_includes_failed_and_skipped_outputs(tmp_path: Path) -> None:
    bad, good = tmp_path / "bad.txt", tmp_path / "good.txt"
    bad.write_bytes(b"\xff")
    good.write_text("Salom", encoding="utf-8")
    existing = tmp_path / "good CYR.txt"
    existing.write_text("existing", encoding="utf-8")
    worker = Worker("Convert Script", ScriptMode.CYRILLIC, [bad, good], False, report_mode="JSON")
    messages = []
    worker.finished_result.connect(messages.append, Qt.ConnectionType.DirectConnection)
    worker.run()
    report = json.loads(messages[0])
    assert (report["status"], report["written"], report["skipped"], report["failed"]) == ("failed", 0, 1, 1)
    assert {item["status"] for item in report["artifacts"]} == {"skipped", "failed"}
    assert existing.read_text(encoding="utf-8") == "existing"
    assert bad.read_bytes() == b"\xff"


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
@pytest.mark.parametrize("report_mode", ["Standard", "JSON"])
def test_review_uses_prepared_plan_and_requires_confirmation(tmp_path: Path, approved: bool, report_mode: str) -> None:
    source = transcript(tmp_path / "input.json")
    worker = Worker(
        "Export",
        ScriptMode.SOURCE,
        [source],
        False,
        settings=DesktopSettings(project_directory=tmp_path),
        review_before_running=True,
        report_mode=report_mode,
    )
    reviews = []
    retry_items = []
    messages = []
    worker.failed_item.connect(retry_items.append, Qt.ConnectionType.DirectConnection)
    worker.finished_result.connect(messages.append, Qt.ConnectionType.DirectConnection)
    target = source.with_suffix(".srt")

    def answer(preview: str) -> None:
        reviews.append((preview, target.exists()))
        worker.answer_review(approved)

    worker.review_requested.connect(answer, Qt.ConnectionType.DirectConnection)
    worker.start()
    assert worker.wait(5000)
    assert len(reviews) == 1
    if report_mode == "JSON":
        assert json.loads(reviews[0][0])["artifacts"][0]["target"] == str(target)
    else:
        assert str(target) in reviews[0][0]
    assert reviews[0][1] is False
    assert target.exists() is approved
    assert len(retry_items) == (0 if approved else 1)
    if report_mode == "JSON":
        report = json.loads(messages[0])
        assert report["status"] == ("ok" if approved else "cancelled")
        assert report["cancelled"] is (not approved)
        assert report["written"] == (1 if approved else 0)


def test_declined_transcription_review_never_constructs_provider(tmp_path: Path, monkeypatch) -> None:
    from PySide6.QtWidgets import QMessageBox

    from elevenlabs_toolkit import providers

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(providers, "ElevenLabsProvider", lambda **kwargs: pytest.fail("Unexpected provider request"))
    reviews = []

    def decline(dialog):
        reviews.append((dialog.standardButton(dialog.defaultButton()), dialog.detailedText()))
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "exec", decline)
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "audio.wav"
    source.write_bytes(b"audio")
    window = MainWindow()
    window.add_paths([source])
    window.review_before_running.setChecked(True)
    window.start_job()
    for _ in range(500):
        app.processEvents()
        if not window.worker.isRunning():
            break
        QTest.qWait(10)
    assert window.worker.wait(1000)
    app.processEvents()
    assert len(reviews) == 1
    assert reviews[0][0] == QMessageBox.StandardButton.Cancel
    assert "Planned API requests: 1" in reviews[0][1]
    assert not (tmp_path / "audio.wav.transcript.json").exists()
    assert window.retry_button.isEnabled()
    window.close()


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
    window.output_subfolder.setChecked(False)
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
    window.output_subfolder.setChecked(False)
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


def test_configuration_credentials_selection_is_shared_and_not_saved(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings)
    selected = tmp_path / ".env"
    monkeypatch.setattr("elevenlabs_toolkit.gui_options.QFileDialog.getOpenFileName", lambda *args: (str(selected), ""))
    window.navigation.setCurrentRow(5)
    window.config_choose_credentials.click()
    assert window.config_credentials.text() == str(selected)
    assert window.transcription_options.settings(tmp_path).env_file == selected
    window.config_credentials.setText("another.env")
    assert window.transcription_options.env_file.text() == "another.env"
    window.close()
    assert "another.env" not in (tmp_path / "settings.ini").read_text(encoding="utf-8")
    app.processEvents()


def test_transcription_options_adapt_to_available_width(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(1450, 900)
    window.show()
    app.processEvents()
    window.job_settings_button.setChecked(True)
    app.processEvents()
    panel = window.transcription_options
    assert panel.recognition_grid.getItemPosition(panel.recognition_grid.indexOf(panel.output_group))[1] == 1
    # Simulate platforms where the scrollable content remains wider than the viewport.
    panel.setMinimumWidth(1000)
    window.resize(800, 600)
    app.processEvents()
    assert panel.recognition_grid.getItemPosition(panel.recognition_grid.indexOf(panel.output_group))[1] == 0
    assert window.start.isVisible()
    window.close()


def test_transcription_settings_have_space_in_standard_window(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(1020, 800)
    window.show()
    app.processEvents()
    window.job_settings_button.setChecked(True)
    app.processEvents()
    assert window.options_scroll.height() >= 350
    assert window.start.isVisible()
    window.close()


@pytest.mark.parametrize(
    ("cli_args", "field"),
    [
        (["--model", ""], "model"),
        (["--keyterm", "invalid[term]"], "keyterms"),
        (["--model", "scribe_v1", "--no-verbatim"], "no_verbatim"),
    ],
)
def test_invalid_transcription_text_rejected_by_cli_and_gui_before_provider(
    tmp_path: Path, monkeypatch, capsys, cli_args: list[str], field: str
) -> None:
    from elevenlabs_toolkit import providers
    from elevenlabs_toolkit.cli import transcribe

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)

    def reject_provider(**kwargs):
        pytest.fail("Invalid options must not construct a provider")

    monkeypatch.setattr(providers, "ElevenLabsProvider", reject_provider)
    monkeypatch.setattr(transcribe, "ElevenLabsProvider", reject_provider)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    assert main(["transcribe", str(audio), "-o", str(tmp_path / "cli"), *cli_args]) != 0
    assert capsys.readouterr().err
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    panel = window.transcription_options
    if field == "model":
        panel.model.clear()
        error = panel.model_error
    elif field == "keyterms":
        panel.keyterms.setPlainText("invalid[term]")
        error = panel.keyterms_error
    else:
        panel.model.setText("scribe_v1")
        panel.no_verbatim.setChecked(True)
        error = panel.verbatim_error
    window.add_paths([audio])
    window.start_job()
    assert error.text()
    assert "preflight failed" in window.details.toPlainText().lower()
    assert not (tmp_path / "cli").exists()
    assert not (tmp_path / "Output").exists()
    assert audio.read_bytes() == b"audio"
    window.close()
    app.processEvents()


def test_batch_workspace_prioritizes_rows_until_settings_are_requested(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(1020, 800)
    window.show()
    app.processEvents()
    assert window.batch_table.height() >= 350
    assert not window.options_scroll.isVisible()
    assert not window.file_list.isVisible()
    assert window.start.isVisible()
    QTest.keyClick(window.job_settings_button, Qt.Key.Key_Space)
    assert window.options_scroll.isVisible()
    assert window.input_controls.isVisible()
    window.navigation.setCurrentRow(1)
    assert window.mode.isVisible()
    assert not window.input_controls.isVisible()
    window.close()


def test_main_workflow_menus_share_settings_and_actions(tmp_path: Path, monkeypatch) -> None:
    from PySide6.QtWidgets import QPushButton

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.project_directory = tmp_path
    window.show()
    app.processEvents()

    assert window.transcription_options.language.isVisible()
    assert not window.options_scroll.isVisible()
    button = window.quick_controls.currentWidget().findChild(QPushButton)
    local_formats = button.menu().actions()[0].menu()
    subtitle = next(action for action in local_formats.actions() if action.text() == "Subtitles (SRT)")
    subtitle.trigger()
    assert window.transcription_options.settings(tmp_path).additional_formats == (ArtifactFormat.SRT,)
    window.transcription_options.local_formats[ArtifactFormat.SRT].setChecked(False)
    assert not subtitle.isChecked()

    window.show_effective.trigger()
    assert window.details.isVisible()
    window.navigation.setCurrentRow(2)
    assert window.export_text.profile.isVisible()
    assert not window.options_scroll.isVisible()
    export_menu = window.quick_controls.currentWidget().findChild(QPushButton).menu()
    next(action for action in export_menu.actions() if action.text() == "Transcript text (TXT)").trigger()
    assert ArtifactFormat.TXT in window.selected_formats()

    window.navigation.setCurrentRow(5)
    assert window.preset_button.isVisible()
    assert window.preset_button.menu() is window.preferences_menu
    assert not window.more_actions.isVisible()
    window.close()


def test_batch_policies_live_in_job_settings_and_remain_visible_in_summary(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    assert window.output_subfolder.isVisible()
    assert not window.conflict.isVisible()
    assert not window.report_mode.isVisible()
    window.job_settings_button.click()
    assert window.batch_controls.isVisible()
    assert window.conflict.isVisible()
    window.conflict.setCurrentIndex(1)
    window.fail_fast.setChecked(True)
    window.job_settings_button.click()
    assert "Replace existing" in window.job_policy_summary.text()
    assert "Stop on error" in window.job_policy_summary.text()
    window.navigation.setCurrentRow(4)
    window.job_settings_button.click()
    assert window.report_mode.isVisible()
    assert not window.conflict.isVisible()
    window.navigation.setCurrentRow(5)
    assert not window.batch_controls.isVisible()
    window.close()


def test_batch_row_shows_short_paths_and_full_details(tmp_path: Path, monkeypatch) -> None:
    from PySide6.QtWidgets import QStyleOptionViewItem

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    source = str(tmp_path / "long input folder" / "clip.flac")
    outputs = [str(tmp_path / "long output folder" / name) for name in ("clip.json", "clip.srt")]
    window.show_batch_plan([{"source": source, "outputs": outputs, "operation": 0, "error": "Output conflict"}])
    for column, expected in ((0, "clip.flac"), (1, "clip.json (+1 more)")):
        option = QStyleOptionViewItem()
        delegate = window.batch_table.itemDelegateForColumn(column)
        delegate.initStyleOption(option, window.batch_table.model().index(0, column))
        assert option.text == expected
    assert window.batch_table.item(0, 0).text() == source
    assert not window.batch_row_details.isEnabled()
    window.batch_table.selectRow(0)
    assert window.batch_row_details.isEnabled()

    opened = []

    def inspect_dialog(dialog):
        opened.append(dialog.findChild(type(window.details)).toPlainText())

    monkeypatch.setattr("elevenlabs_toolkit.gui.QDialog.exec", inspect_dialog)
    window.batch_row_details.trigger()
    window.show()
    window.batch_table.setFocus()
    app.processEvents()
    QTest.keyClick(window.batch_table, Qt.Key.Key_Return)
    assert len(opened) == 2
    assert all(value in opened[0] for value in (source, *outputs, "Output conflict"))
    assert opened[0] == opened[1]
    window.close()
    app.processEvents()
