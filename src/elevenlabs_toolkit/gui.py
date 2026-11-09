from __future__ import annotations

import json
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import replace as replace_settings
from pathlib import Path
from threading import Event

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDragMoveEvent, QDropEvent, QStandardItemModel
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .application.desktop import DesktopSettings, PreparedFile, prepare_files
from .files.discovery import AUDIO_VIDEO_SUFFIXES, DiscoveryCancelled, discover_inputs
from .gui_options import TextOptionsPanel, TranscriptionPanel
from .models import ArtifactFormat, ConflictPolicy, InputSpec, ScriptMode


class FileList(QListWidget):
    def __init__(self, add_paths: Callable[[list[Path]], None]) -> None:
        super().__init__()
        self.add_paths = add_paths
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.add_paths(paths)
            event.acceptProposedAction()


class Worker(QThread):
    status = Signal(str)
    finished_result = Signal(str)
    progress_changed = Signal(int, int)
    review_requested = Signal(str)

    def __init__(
        self,
        action: str,
        mode: ScriptMode,
        paths: list[Path],
        replace: bool,
        formats: tuple[ArtifactFormat, ...] = (ArtifactFormat.SRT,),
        settings: DesktopSettings | None = None,
        *,
        policy: ConflictPolicy | None = None,
        selection: InputSpec | None = None,
        include_generated: bool = False,
        preview_only: bool = False,
        report_mode: str = "Standard",
        review_before_running: bool = False,
    ) -> None:
        super().__init__()
        self.action = action
        self.mode = mode
        self.paths = tuple(paths)
        self.policy = policy or (ConflictPolicy.REPLACE if replace else ConflictPolicy.SKIP)
        self.formats = formats
        self.settings = settings or DesktopSettings()
        self.selection = selection
        self.include_generated = include_generated
        self.preview_only = preview_only
        self.report_mode = report_mode
        self.review_before_running = review_before_running
        self.review_answered = Event()
        self.review_approved = False

    def run(self) -> None:
        outcomes = []
        records: list[dict] = []
        cancelled = False
        try:
            paths = self.paths
            if self.selection is not None:
                self.status.emit("Scanning selected files and folders…")
                suffixes = {
                    "Transcribe": set(AUDIO_VIDEO_SUFFIXES),
                    "Convert Script": {".srt", ".txt"},
                    "Export": {".json"},
                    "Clean": {".json"},
                    "Inspect": {".json"},
                }
                paths = discover_inputs(
                    self.selection,
                    suffixes[self.action],
                    exclude_generated=self.action in {"Export", "Clean", "Inspect"} and not self.include_generated,
                    should_stop=self.isInterruptionRequested,
                    exclude_folder_suffixes=(
                        (".latin.srt", ".cyrillic.srt", ".latin.txt", ".cyrillic.txt")
                        if self.action == "Convert Script"
                        else ()
                    ),
                )
            self.status.emit("Planning outputs…")
            prepared = prepare_files(self.action, paths, self.mode, self.policy, self.formats, self.settings)
            if self.isInterruptionRequested():
                raise DiscoveryCancelled("Stopped before processing any files")
            if self.preview_only:
                self.finished_result.emit(self.preview_text(prepared))
                return
            if self.review_before_running:
                self.review_requested.emit(self.preview_text(prepared))
                while not self.review_answered.wait(0.1):
                    if self.isInterruptionRequested():
                        break
                if not self.review_approved or self.isInterruptionRequested():
                    raise DiscoveryCancelled("Cancelled before processing any files")
            previous_request = False
            for index, item in enumerate(prepared, start=1):
                if item.plan.api_requests and previous_request and self.settings.request_delay:
                    self.status.emit(f"Waiting {self.settings.request_delay:g} seconds before the next request")
                    deadline = time.monotonic() + self.settings.request_delay
                    while not self.isInterruptionRequested():
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        time.sleep(min(remaining, 0.1))
                if self.isInterruptionRequested():
                    cancelled = True
                    outcomes.append("Stopped before the next file.")
                    break
                previous_request |= bool(item.plan.api_requests)
                source = item.plan.sources[0]
                self.status.emit(f"[{index}/{len(prepared)}] {self.action}: {source.name}")
                try:
                    message, failed, data = self.execute_file(item)
                    outcomes.append(message)
                    records.append({"sources": [str(path) for path in item.plan.sources], **data})
                except Exception as exc:
                    message = traceback.format_exc() if self.report_mode == "Verbose" else str(exc)
                    outcomes.append(f"{source.name}: FAILED — {message}")
                    records.append({"source": str(source), "status": "failed", "error": str(exc)})
                    failed = True
                self.progress_changed.emit(index, len(prepared))
                if failed and self.settings.fail_fast:
                    outcomes.append("Stopped after the first failure.")
                    break
        except DiscoveryCancelled as exc:
            cancelled = True
            outcomes.append(str(exc))
        except Exception as exc:
            message = traceback.format_exc() if self.report_mode == "Verbose" else str(exc)
            outcomes.append(f"Preflight failed: {message}")
            records.append({"status": "failed", "error": str(exc)})
        if self.report_mode == "JSON":
            self.finished_result.emit(
                json.dumps(
                    {"cancelled": cancelled or self.isInterruptionRequested(), "results": records},
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            self.finished_result.emit("\n".join(outcomes))

    def answer_review(self, approved: bool) -> None:
        self.review_approved = approved
        self.review_answered.set()

    def preview_text(self, prepared: tuple[PreparedFile, ...]) -> str:
        lines = []
        artifacts = []
        requests = sum(item.plan.api_requests for item in prepared)
        attempts = requests * (1 + self.settings.retries)
        if self.action == "Transcribe":
            lines.append(f"Planned API requests: {requests}, up to {attempts} attempts. API charges may apply.")
        for item in prepared:
            if not item.plan.artifacts:
                lines.append(f"Inspect: {item.plan.sources[0]} (read-only)")
            for artifact in item.plan.artifacts:
                if artifact.format is ArtifactFormat.COMBINED_TXT:
                    lines.append("Combined text inputs: " + ", ".join(str(source) for source in item.plan.sources))
                status = "Create"
                if artifact.target.exists() or artifact.target.is_symlink():
                    status = "Replace" if self.policy is ConflictPolicy.REPLACE else "Skip existing"
                lines.append(f"{artifact.source} → {artifact.target} ({status})")
                artifacts.append(
                    {
                        "sources": [str(path) for path in item.plan.sources]
                        if artifact.format is ArtifactFormat.COMBINED_TXT
                        else [str(artifact.source)],
                        "target": str(artifact.target),
                        "format": artifact.format.value,
                        "action": status,
                    }
                )
        if self.report_mode == "JSON":
            return json.dumps(
                {
                    "sources": list(dict.fromkeys(str(path) for item in prepared for path in item.plan.sources)),
                    "api_requests": requests,
                    "max_api_attempts": attempts,
                    "artifacts": artifacts,
                },
                ensure_ascii=False,
                indent=2,
            )
        return "\n".join(lines)

    def execute_file(self, item: PreparedFile) -> tuple[str, bool, dict]:
        from .application import execute_export, execute_transcription
        from .application.conversion import convert_srt
        from .application.inspection import inspect_transcript
        from .files import atomic_write_text
        from .languages import get_language_processor
        from .providers import ElevenLabsProvider

        source = item.plan.sources[0]
        if self.action == "Inspect":
            summary = inspect_transcript(source)
            return json.dumps(summary, ensure_ascii=False, indent=2), not summary["valid"], dict(summary)
        if self.action == "Convert Script":
            artifact = item.plan.artifacts[0]
            if self.policy is ConflictPolicy.SKIP and (artifact.target.exists() or artifact.target.is_symlink()):
                return f"skipped: {artifact.target}", False, {"status": "skipped", "target": str(artifact.target)}
            text = source.read_text(encoding="utf-8-sig")
            if artifact.format is ArtifactFormat.SRT:
                converted = convert_srt(text, self.mode)
            else:
                converted = get_language_processor("uzbek").transform_text(text, target=self.mode)
            write_policy = ConflictPolicy.ERROR if self.policy is ConflictPolicy.RENAME else self.policy
            target, status = atomic_write_text(artifact.target, converted, write_policy)
            return f"{status.value}: {target}", False, {"status": status.value, "target": str(target)}
        if item.export is None:
            raise ValueError("Missing export configuration")
        if self.action == "Transcribe":
            if item.transcription is None:
                raise ValueError("Missing transcription configuration")
            provider = (
                ElevenLabsProvider(env_file=self.settings.env_file or self.settings.project_directory / ".env")
                if item.plan.api_requests
                else None
            )
            result = execute_transcription(
                item.plan,
                item.transcription,
                item.export,
                provider=provider,
                policy=self.policy,
                progress=self.status.emit,
                retries=self.settings.retries,
                backoff_seconds=self.settings.retry_backoff,
            )
        else:
            result = execute_export(
                item.plan,
                item.export,
                policy=self.policy,
                fail_fast=self.settings.fail_fast,
                should_stop=self.isInterruptionRequested,
                source_finished=lambda source: self.status.emit(f"Combined text: processed {source.name}"),
            )
        message = "\n".join(
            f"{artifact.status.value}: {artifact.artifact.target} {artifact.message}" for artifact in result.artifacts
        )
        return (
            message,
            bool(result.failed),
            {
                "status": "failed" if result.failed else "ok",
                "artifacts": [
                    {
                        "source": str(entry.artifact.source),
                        "target": str(entry.artifact.target),
                        "format": entry.artifact.format.value,
                        "status": entry.status.value,
                        "message": entry.message,
                    }
                    for entry in result.artifacts
                ],
            },
        )


@dataclass(frozen=True)
class WorkflowControls:
    recursive: bool = True
    include_generated: bool = False
    filter_kind: int = 0
    pattern: str = ""
    output_directory: str = ""
    conflict: int = 0
    fail_fast: bool = False
    review: bool = False


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ElevenLabs Toolkit")
        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction("Using the Toolkit", self.show_help)
        help_menu.addAction("About ElevenLabs Toolkit", self.show_about)
        self.resize(1020, 800)
        self.paths: list[Path] = []
        self.workflow_paths: dict[int, list[Path]] = {}
        self.workflow_controls: dict[int, WorkflowControls] = {}
        self.active_workflow = 0
        self.project_directory = DesktopSettings().project_directory
        root = QWidget()
        outer = QHBoxLayout(root)
        self.navigation = QListWidget()
        self.navigation.addItems(["Transcribe", "Convert Script", "Export", "Clean", "Inspect", "Configuration"])
        self.navigation.setFixedWidth(180)
        self.navigation.setCurrentRow(0)
        outer.addWidget(self.navigation)
        panel = QVBoxLayout()
        outer.addLayout(panel, 1)

        self.heading = QLabel("Transcribe audio or video to reusable JSON")
        panel.addWidget(self.heading)
        self.formats = QLabel("JSON is always saved · Recursive folder scanning")
        panel.addWidget(self.formats)

        buttons = QHBoxLayout()
        self.add_files = QPushButton("Add Files")
        self.add_folder = QPushButton("Add Folder")
        self.remove = QPushButton("Remove Selected")
        self.clear = QPushButton("Clear")
        for button in (self.add_files, self.add_folder, self.remove, self.clear):
            buttons.addWidget(button)
        panel.addLayout(buttons)

        self.input_controls = QWidget()
        filters = QHBoxLayout(self.input_controls)
        filters.setContentsMargins(0, 0, 0, 0)
        self.recursive = QCheckBox("Include subfolders")
        self.recursive.setChecked(True)
        self.include_generated = QCheckBox("Include generated JSON")
        self.filter_kind = QComboBox()
        self.filter_kind.addItems(["No folder filter", "Glob", "Regular expression"])
        self.pattern = QLineEdit()
        self.pattern.setPlaceholderText("Match paths inside folders")
        self.pattern.setEnabled(False)
        self.filter_kind.currentIndexChanged.connect(lambda index: self.pattern.setEnabled(index != 0))
        for control in (self.recursive, self.include_generated, self.filter_kind, self.pattern):
            filters.addWidget(control)
        self.include_generated.hide()
        panel.addWidget(self.input_controls)

        self.file_list = FileList(self.add_paths)
        self.file_list.setToolTip("Drop files or folders here")
        panel.addWidget(self.file_list, 1)

        self.options = QStackedWidget()
        convert = QWidget()
        convert_layout = QHBoxLayout(convert)
        self.mode = QComboBox()
        self.mode.addItems(["Cyrillic", "Latin"])
        convert_layout.addWidget(QLabel("Convert Uzbek script to"))
        convert_layout.addWidget(self.mode)
        convert_layout.addStretch(1)
        self.transcription_options = TranscriptionPanel()
        self.options.addWidget(self.transcription_options)
        self.options.addWidget(convert)
        export_panel = QWidget()
        export_layout = QVBoxLayout(export_panel)
        format_row = QHBoxLayout()
        export_layout.addLayout(format_row)
        self.export_formats = {}
        for artifact_format in (
            ArtifactFormat.SRT,
            ArtifactFormat.SRT_MINI,
            ArtifactFormat.TXT,
            ArtifactFormat.COMBINED_TXT,
            ArtifactFormat.RESOLVE_EDL,
            ArtifactFormat.CUE_INDEX_SRT,
            ArtifactFormat.CLEAN_JSON,
        ):
            checkbox = QCheckBox(artifact_format.value.upper())
            checkbox.setChecked(artifact_format is ArtifactFormat.SRT)
            self.export_formats[artifact_format] = checkbox
            format_row.addWidget(checkbox)
        self.export_text = TextOptionsPanel(advanced=True)
        export_layout.addWidget(self.export_text)
        self.options.addWidget(export_panel)
        clean_panel = QWidget()
        clean_layout = QVBoxLayout(clean_panel)
        clean_layout.addWidget(QLabel("Creates a cleaned JSON derivative using Uzbek editorial rules."))
        self.clean_text = TextOptionsPanel(clean=True)
        clean_layout.addWidget(self.clean_text)
        self.options.addWidget(clean_panel)
        self.options.addWidget(QLabel("Validate JSON and show language, timing, speakers, and transcript counts."))
        config_panel = QWidget()
        config_layout = QVBoxLayout(config_panel)
        self.project_label = QLabel(str(self.project_directory))
        self.project_label.setWordWrap(True)
        config_layout.addWidget(QLabel("Project configuration context"))
        config_layout.addWidget(self.project_label)
        self.choose_project = QPushButton("Choose Project Folder")
        self.choose_project.clicked.connect(self.select_project)
        config_layout.addWidget(self.choose_project)
        self.config_profile = QComboBox()
        self.config_profile.setEditable(True)
        self.config_profile.addItem("Project default")
        config_layout.addWidget(self.config_profile)
        config_layout.addWidget(QLabel("Shows the effective merged configuration and available profiles."))
        self.options.addWidget(config_panel)
        from PySide6.QtWidgets import QScrollArea

        options_scroll = QScrollArea()
        options_scroll.setWidgetResizable(True)
        options_scroll.setWidget(self.options)
        options_scroll.setMaximumHeight(300)
        panel.addWidget(options_scroll)

        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText("Preview output paths or inspect the results here.")
        self.details.setMinimumHeight(130)
        panel.addWidget(self.details)
        bottom = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.output_controls = QWidget()
        output_row = QHBoxLayout(self.output_controls)
        output_row.setContentsMargins(0, 0, 0, 0)
        self.output_directory = QLineEdit()
        self.output_directory.setPlaceholderText("Beside source files")
        choose_output = QPushButton("Browse…")
        choose_output.clicked.connect(self.select_output_directory)
        self.conflict = QComboBox()
        for label, policy in (
            ("Skip existing", ConflictPolicy.SKIP),
            ("Replace existing", ConflictPolicy.REPLACE),
            ("Report conflict", ConflictPolicy.ERROR),
            ("Choose a new name", ConflictPolicy.RENAME),
        ):
            self.conflict.addItem(label, policy.value)
        self.fail_fast = QCheckBox("Stop on error")
        for control_widget in (
            QLabel("Output folder"),
            self.output_directory,
            choose_output,
            self.conflict,
            self.fail_fast,
        ):
            output_row.addWidget(control_widget)
        panel.addWidget(self.output_controls)
        model = self.conflict.model()
        assert isinstance(model, QStandardItemModel)
        self.rename_option = model.item(3)
        self.rename_option.setEnabled(False)
        self.report_mode = QComboBox()
        self.report_mode.addItems(["Standard", "JSON", "Quiet", "Verbose"])
        self.report_mode.setToolTip("Result format and progress detail")
        self.review_before_running = QCheckBox("Review before running")
        self.preview = QPushButton("Preview")
        self.stop = QPushButton("Stop After Current File")
        self.stop.setEnabled(False)
        self.start = QPushButton("Transcribe")
        self.start.setDefault(True)
        for widget in (
            self.progress,
            self.report_mode,
            self.review_before_running,
            self.preview,
            self.stop,
            self.start,
        ):
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
        self.stop.clicked.connect(self.stop_job)
        self.refresh_profiles()

    def show_help(self) -> None:
        QMessageBox.information(
            self,
            "Using the Toolkit",
            "Choose a workflow, then add files or folders using the buttons or drag and drop. "
            "Use Preview to check destinations and planned API requests before running.\n\n"
            "Leave the output folder blank to save beside the sources. Transcribe saves JSON by default. "
            "Use Export to render saved JSON offline. Configuration displays the active project settings.",
        )

    def show_about(self) -> None:
        from .cli.main import _version

        QMessageBox.about(self, "About ElevenLabs Toolkit", f"ElevenLabs Toolkit {_version()}")

    def refresh_profiles(self) -> None:
        from .config import available_profiles

        try:
            self.transcription_options.refresh_profiles(self.project_directory)
            self.export_text.refresh_profiles(self.project_directory)
            self.clean_text.refresh_profiles(self.project_directory)
            selected = self.config_profile.currentText()
            self.config_profile.clear()
            self.config_profile.addItems(["Project default", *available_profiles(cwd=self.project_directory)])
            self.config_profile.setCurrentText(selected)
        except Exception as exc:
            self.details.setPlainText(f"Configuration error: {exc}")

    def select_project(self) -> None:
        name = QFileDialog.getExistingDirectory(self, "Project configuration folder", str(self.project_directory))
        if name:
            self.project_directory = Path(name)
            self.project_label.setText(name)
            self.refresh_profiles()

    def desktop_settings(self) -> DesktopSettings:
        settings = self.workflow_settings()
        folder = self.output_directory.text().strip()
        return replace_settings(
            settings,
            output_directory=self.project_directory / Path(folder).expanduser() if folder else None,
            fail_fast=self.fail_fast.isChecked() and self.active_workflow < 4,
        )

    def workflow_settings(self) -> DesktopSettings:
        if self.active_workflow == 0:
            return self.transcription_options.settings(self.project_directory)
        if self.active_workflow == 2:
            return self.export_text.settings(self.project_directory)
        if self.active_workflow == 3:
            return self.clean_text.settings(self.project_directory)
        return DesktopSettings(project_directory=self.project_directory)

    def show_configuration(self) -> None:
        from .config import available_profiles, effective_config

        try:
            profile = self.config_profile.currentText().strip()
            config = effective_config(None if profile == "Project default" else profile, cwd=self.project_directory)
            self.details.setPlainText(
                json.dumps(
                    {
                        "effective": config,
                        "available_profiles": available_profiles(cwd=self.project_directory),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
        except Exception as exc:
            self.details.setPlainText(f"Configuration error: {exc}")

    def change_workflow(self, row: int) -> None:
        self.workflow_paths[self.active_workflow] = list(self.paths)
        self.workflow_controls[self.active_workflow] = WorkflowControls(
            recursive=self.recursive.isChecked(),
            include_generated=self.include_generated.isChecked(),
            filter_kind=self.filter_kind.currentIndex(),
            pattern=self.pattern.text(),
            output_directory=self.output_directory.text(),
            conflict=self.conflict.currentIndex(),
            fail_fast=self.fail_fast.isChecked(),
            review=self.review_before_running.isChecked(),
        )
        self.active_workflow = row
        controls = self.workflow_controls.get(row, WorkflowControls())
        self.recursive.setChecked(controls.recursive)
        self.include_generated.setChecked(controls.include_generated)
        self.filter_kind.setCurrentIndex(controls.filter_kind)
        self.pattern.setText(controls.pattern)
        self.output_directory.setText(controls.output_directory)
        self.conflict.setCurrentIndex(controls.conflict)
        self.fail_fast.setChecked(controls.fail_fast)
        self.review_before_running.setChecked(controls.review)
        self.paths = list(self.workflow_paths.get(row, []))
        self.file_list.clear()
        self.file_list.addItems([str(path) for path in self.paths])
        self.options.setCurrentIndex(row)
        headings = (
            "Transcribe audio or video to reusable JSON",
            "Convert Uzbek SRT/TXT between Latin and Cyrillic",
            "Export saved transcript JSON to subtitles and text",
            "Clean Uzbek transcript JSON",
            "Inspect transcript JSON",
            "Configuration and profiles",
        )
        self.heading.setText(headings[row])
        self.start.setText(("Transcribe", "Convert", "Export", "Clean", "Inspect", "Show Configuration")[row])
        self.formats.setText(
            "JSON is always saved · API charges may apply"
            if row == 0
            else "Offline · Outputs beside source files"
            if row < 4
            else "Read-only"
        )
        for widget in (self.add_files, self.add_folder, self.remove, self.clear, self.file_list):
            widget.setVisible(row != 5)
        self.input_controls.setVisible(row != 5)
        self.include_generated.setVisible(row in (2, 3, 4))
        self.output_controls.setVisible(row < 4)
        self.rename_option.setEnabled(row != 0)
        if row == 0 and self.conflict_policy() is ConflictPolicy.RENAME:
            self.conflict.setCurrentIndex(0)
        self.preview.setVisible(row != 5)
        self.review_before_running.setVisible(row in (0, 2))
        self.details.clear()
        if row == 5:
            self.show_configuration()

    def add_paths(self, paths: list[Path]) -> None:
        for path in paths:
            path = path.resolve()
            if path not in self.paths:
                self.paths.append(path)
                self.file_list.addItem(str(path))
        self.details.setPlainText(f"{len(self.paths)} file(s) or folder(s) selected")

    def select_output_directory(self) -> None:
        name = QFileDialog.getExistingDirectory(self, "Output folder", self.output_directory.text())
        if name:
            self.output_directory.setText(name)

    def conflict_policy(self) -> ConflictPolicy:
        return ConflictPolicy(self.conflict.currentData())

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

    def selected_formats(self) -> tuple[ArtifactFormat, ...]:
        return tuple(value for value, checkbox in self.export_formats.items() if checkbox.isChecked())

    def show_preview(self) -> None:
        self.launch_worker(preview_only=True)

    def start_job(self) -> None:
        if self.active_workflow == 5:
            self.show_configuration()
        else:
            self.launch_worker(preview_only=False)

    def launch_worker(self, *, preview_only: bool) -> None:
        if not self.start.isEnabled() or self.active_workflow == 5:
            return
        try:
            settings = self.desktop_settings()
            kind = self.filter_kind.currentIndex()
            pattern = self.pattern.text()
            selection = InputSpec(
                tuple(self.paths),
                glob=pattern if kind == 1 else None,
                regex=pattern if kind == 2 else None,
                recursive=self.recursive.isChecked(),
            )
        except Exception as exc:
            self.details.setPlainText(f"Preflight failed: {exc}")
            return
        action = ("Transcribe", "Convert Script", "Export", "Clean", "Inspect")[self.active_workflow]
        mode = ScriptMode(self.mode.currentText().lower())
        self.worker = Worker(
            action,
            mode,
            [],
            False,
            self.selected_formats(),
            settings,
            policy=self.conflict_policy(),
            selection=selection,
            include_generated=self.include_generated.isChecked(),
            preview_only=preview_only,
            report_mode=self.report_mode.currentText(),
            review_before_running=not preview_only
            and self.active_workflow in (0, 2)
            and self.review_before_running.isChecked(),
        )
        if self.report_mode.currentText() != "Quiet":
            self.worker.status.connect(self.details.setPlainText)
        self.worker.review_requested.connect(self.review_plan)
        self.worker.finished_result.connect(self.job_finished)
        self.worker.finished.connect(self.worker_finished)
        self.worker.progress_changed.connect(self.update_progress)
        self.set_running(True)
        self.worker.start()

    def review_plan(self, preview: str) -> None:
        self.details.setPlainText(preview)
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Review output plan")
        dialog.setText("Run this plan? Review the destinations and any API requests before continuing.")
        dialog.setDetailedText(preview)
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        dialog.setDefaultButton(QMessageBox.StandardButton.Cancel)
        self.worker.answer_review(dialog.exec() == QMessageBox.StandardButton.Ok)

    def set_running(self, running: bool) -> None:
        for widget in (
            self.navigation,
            self.add_files,
            self.add_folder,
            self.remove,
            self.clear,
            self.file_list,
            self.options,
            self.input_controls,
            self.output_controls,
            self.report_mode,
            self.review_before_running,
            self.preview,
            self.start,
        ):
            widget.setEnabled(not running)
        self.stop.setEnabled(running)
        if running:
            self.progress.setRange(0, 0)

    def update_progress(self, completed: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(completed)

    def stop_job(self) -> None:
        self.worker.requestInterruption()
        self.stop.setEnabled(False)
        self.details.setPlainText("Stopping at the next safe point…")

    def job_finished(self, message: str) -> None:
        self.details.setPlainText(message)

    def worker_finished(self) -> None:
        self.set_running(False)
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
        self.worker.deleteLater()

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self.start.isEnabled():
            event.ignore()
            answer = QMessageBox.question(
                self,
                "Processing in progress",
                "Stop after the current file? You can close the window when it finishes.",
            )
            if answer == QMessageBox.StandardButton.Yes:
                self.stop_job()
            return
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
