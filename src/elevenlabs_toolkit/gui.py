from __future__ import annotations

import re
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QRadioButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class Worker(QThread):
    status = Signal(str)
    finished_result = Signal(str)

    def __init__(self, action: str, mode: str, paths: list[Path], replace: bool) -> None:
        super().__init__()
        self.action, self.mode, self.paths, self.replace = action, mode, paths, replace

    def run(self) -> None:
        try:
            if self.action == "Convert Script":
                self._convert()
            else:
                self._transcribe()
        except Exception as exc:
            self.finished_result.emit(f"Failed: {exc}")

    def _convert(self) -> None:
        from .cli.transliterate import _convert_srt
        from .languages import get_language_processor

        outputs = []
        for source in self.paths:
            self.status.emit(f"Converting {source.name}")
            text = source.read_text(encoding="utf-8-sig")
            if source.suffix.casefold() == ".srt":
                converted = _convert_srt(text, self.mode)
            else:
                converted = get_language_processor("uzbek").transform_text(text, target=self.mode)
            target = source.with_name(f"{source.stem}.{self.mode}{source.suffix.lower()}")
            if target.exists() and not self.replace:
                outputs.append(f"Skipped existing {target.name}")
                continue
            target.write_text(converted, encoding="utf-8")
            outputs.append(f"Created {target}")
        self.finished_result.emit("\n".join(outputs) or "No files processed")

    def _transcribe(self) -> None:
        from .application import execute_transcription, plan_transcription
        from .config import effective_config, profile_options
        from .models import ArtifactFormat, ConflictPolicy, ExportOptions, TranscriptionOptions
        from .providers import ElevenLabsProvider

        config = effective_config(cwd=Path(__file__).resolve().parents[2])
        segmentation, text = profile_options(config["profile"], config)
        options = TranscriptionOptions()
        provider = ElevenLabsProvider(env_file=Path(__file__).resolve().parents[2] / ".env")
        completed = []
        for index, source in enumerate(self.paths, start=1):
            self.status.emit(f"[{index}/{len(self.paths)}] Preparing {source.name}")
            export = ExportOptions((ArtifactFormat.JSON,), source.parent, segmentation, text)
            plan = plan_transcription((source,), source.parent, (ArtifactFormat.JSON,),
                                      policy=ConflictPolicy.SKIP, transcription_options=options)
            if plan.api_requests:
                self.status.emit(f"Sending {source.name} to ElevenLabs. API charges may apply.")
            result = execute_transcription(
                plan, options, export, provider=provider,
                policy=ConflictPolicy.SKIP, progress=self.status.emit,
            )
            completed.append(f"{source.name}: {result.exit_code}")
        self.finished_result.emit("Completed\n" + "\n".join(completed))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ElevenLabs Toolkit")
        self.resize(960, 680)
        self.paths: list[Path] = []
        root = QWidget()
        outer = QHBoxLayout(root)
        self.navigation = QListWidget()
        self.navigation.addItems(["Transcribe", "Convert Script", "Export (coming later)",
                                  "Clean (coming later)", "Inspect (coming later)"])
        self.navigation.setFixedWidth(180)
        self.navigation.setCurrentRow(0)
        outer.addWidget(self.navigation)
        panel = QVBoxLayout()
        outer.addLayout(panel, 1)

        self.heading = QLabel("Transcribe audio or video to reusable JSON")
        panel.addWidget(self.heading)
        self.formats = QLabel("Output: JSON only · Recursive folder scanning")
        panel.addWidget(self.formats)

        buttons = QHBoxLayout()
        self.add_files = QPushButton("Add Files")
        self.add_folder = QPushButton("Add Folder")
        self.remove = QPushButton("Remove Selected")
        self.clear = QPushButton("Clear")
        for button in (self.add_files, self.add_folder, self.remove, self.clear):
            buttons.addWidget(button)
        panel.addLayout(buttons)

        self.file_list = QListWidget()
        self.file_list.setAcceptDrops(True)
        self.file_list.setDragDropMode(QListWidget.DragDropMode.DropOnly)
        self.file_list.setToolTip("Drop files or folders here")
        panel.addWidget(self.file_list, 1)
        self.file_list.dragEnterEvent = self.drag_enter
        self.file_list.dropEvent = self.drop

        self.options = QStackedWidget()
        convert = QWidget()
        convert_layout = QHBoxLayout(convert)
        self.mode = QComboBox()
        self.mode.addItems(["Cyrillic", "Latin"])
        convert_layout.addWidget(QLabel("Convert Uzbek script to"))
        convert_layout.addWidget(self.mode)
        convert_layout.addStretch(1)
        self.options.addWidget(QWidget())
        self.options.addWidget(convert)
        panel.addWidget(self.options)

        self.details = QLabel("Select files to see planned output paths.")
        panel.addWidget(self.details)
        bottom = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.replace = QRadioButton("Replace existing outputs")
        self.preview = QPushButton("Preview")
        self.stop = QPushButton("Stop After Current File")
        self.stop.setEnabled(False)
        self.start = QPushButton("Transcribe")
        self.start.setDefault(True)
        for widget in (self.progress, self.replace, self.preview, self.stop, self.start):
            bottom.addWidget(widget)
        panel.addLayout(bottom)

        self.setCentralWidget(root)
        self.add_files.clicked.connect(self.select_files)
        self.add_folder.clicked.connect(self.select_folder)
        self.remove.clicked.connect(self.remove_selected)
        self.clear.clicked.connect(self.clear_files)
        self.navigation.currentRowChanged.connect(self.change_workflow)
        self.preview.clicked.connect(self.show_preview)
        self.start.clicked.connect(self.start_job)

    def change_workflow(self, row: int) -> None:
        self.options.setCurrentIndex(1 if row == 1 else 0)
        self.heading.setText("Convert Uzbek SRT/TXT between Latin and Cyrillic" if row == 1
                             else "Transcribe audio or video to reusable JSON")
        self.start.setText("Convert" if row == 1 else "Transcribe")
        self.formats.setText("Output: suffixed files beside sources" if row == 1
                             else "Output: JSON only · Recursive folder scanning")

    def add_paths(self, paths: list[Path]) -> None:
        for path in paths:
            candidates = list(path.rglob("*") if path.is_dir() else [path])
            for candidate in candidates:
                if candidate.is_file() and candidate not in self.paths:
                    self.paths.append(candidate)
                    self.file_list.addItem(str(candidate))
        self.details.setText(f"{len(self.paths)} file(s) selected")

    def select_files(self) -> None:
        names, _ = QFileDialog.getOpenFileNames(self, "Select files")
        self.add_paths([Path(name) for name in names])

    def select_folder(self) -> None:
        name = QFileDialog.getExistingDirectory(self, "Select folder")
        if name:
            self.add_paths([Path(name)])

    def remove_selected(self) -> None:
        indices = sorted((self.file_list.row(item) for item in self.file_list.selectedItems()), reverse=True)
        for index in indices:
            self.file_list.takeItem(index)
            self.paths.pop(index)

    def clear_files(self) -> None:
        self.paths.clear()
        self.file_list.clear()

    def drag_enter(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def drop(self, event) -> None:
        self.add_paths([Path(url.toLocalFile()) for url in event.mimeData().urls()])
        event.acceptProposedAction()

    def show_preview(self) -> None:
        operation = "Convert to " + self.mode.currentText().lower() if self.navigation.currentRow() == 1 else "Transcribe to JSON"
        rows = [f"{operation}: {path}" for path in self.paths]
        QMessageBox.information(self, "Preview", "\n".join(rows) or "No files selected")

    def start_job(self) -> None:
        action = "Convert Script" if self.navigation.currentRow() == 1 else "Transcribe"
        supported = {".srt", ".txt"} if action == "Convert Script" else {
            ".aac", ".aiff", ".ogg", ".mp3", ".opus", ".wav", ".flac", ".m4a", ".webm",
            ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".mpeg", ".3gp"
        }
        selected = [path for path in self.paths if path.suffix.casefold() in supported]
        if not selected:
            QMessageBox.warning(self, "No supported files", "Add supported files for this workflow first.")
            return
        self.worker = Worker(action, self.mode.currentText().lower(), selected, self.replace.isChecked())
        self.worker.status.connect(self.details.setText)
        self.worker.finished_result.connect(self.job_finished)
        self.start.setEnabled(False)
        self.worker.start()

    def job_finished(self, message: str) -> None:
        self.details.setText(message)
        self.start.setEnabled(True)
        QMessageBox.information(self, "Finished", message)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
