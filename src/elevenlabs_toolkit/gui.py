from __future__ import annotations

import json
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from dataclasses import replace as replace_settings
from pathlib import Path
from threading import Event

from PySide6.QtCore import QSettings, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QDesktopServices, QDragEnterEvent, QDragMoveEvent, QDropEvent, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .application.desktop import DesktopSettings, PreparedFile, prepare_files
from .files.discovery import AUDIO_VIDEO_SUFFIXES, DiscoveryCancelled, discover_inputs
from .gui_options import FORMAT_DESCRIPTIONS, TextOptionsPanel, TranscriptionPanel
from .models import ArtifactFormat, ConflictPolicy, InputSpec, ScriptMode


class FileList(QListWidget):
    def __init__(self, add_paths: Callable[[list[Path]], None]) -> None:
        super().__init__()
        self.add_paths = add_paths
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

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


class BatchTable(QTableWidget):
    def __init__(self, add_paths: Callable[[list[Path]], None]) -> None:
        super().__init__(0, 4)
        self.add_paths = add_paths
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        if not self.isEnabled():
            event.ignore()
            return
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.add_paths(paths)
            event.acceptProposedAction()


class Worker(QThread):
    status = Signal(str)
    finished_result = Signal(str)
    progress_changed = Signal(int, int)
    review_requested = Signal(str)
    plan_ready = Signal(object)
    file_result = Signal(object)
    failed_item = Signal(object)
    batch_stopped = Signal(str)

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
        content_preview: bool = False,
        report_mode: str = "Standard",
        review_before_running: bool = False,
        retry_plan: tuple[PreparedFile, ...] | None = None,
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
        self.content_preview = content_preview
        self.report_mode = report_mode
        self.review_before_running = review_before_running
        self.retry_plan = retry_plan
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
                    exclude_directory_names=("output",) if self.settings.output_subfolder else (),
                )
            self.status.emit("Planning outputs…")
            prepared = (
                self.retry_plan
                if self.retry_plan is not None
                else prepare_files(self.action, paths, self.mode, self.policy, self.formats, self.settings)
            )
            self.plan_ready.emit(
                [
                    {
                        "source": str(source),
                        "outputs": [str(artifact.target) for artifact in item.plan.artifacts],
                        "operation": index,
                    }
                    for index, item in enumerate(prepared)
                    for source in item.plan.sources
                ]
            )
            if self.isInterruptionRequested():
                if not self.preview_only:
                    for retry_item in prepared:
                        self.failed_item.emit(retry_item)
                raise DiscoveryCancelled("Stopped before processing any files")
            if self.preview_only:
                self.finished_result.emit(
                    self.preview_content(prepared) if self.content_preview else self.preview_text(prepared)
                )
                return
            if self.review_before_running:
                self.review_requested.emit(self.preview_text(prepared))
                while not self.review_answered.wait(0.1):
                    if self.isInterruptionRequested():
                        break
                if not self.review_approved or self.isInterruptionRequested():
                    for retry_item in prepared:
                        self.failed_item.emit(retry_item)
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
                    for retry_item in prepared[index - 1 :]:
                        self.failed_item.emit(retry_item)
                    outcomes.append("Stopped before the next file.")
                    self.batch_stopped.emit("Stopped")
                    break
                previous_request |= bool(item.plan.api_requests)
                source = item.plan.sources[0]
                self.status.emit(f"[{index}/{len(prepared)}] {self.action}: {source.name}")
                data: dict = {}
                try:
                    message, failed, data = self.execute_file(item)
                    outcomes.append(message)
                    records.append({"sources": [str(path) for path in item.plan.sources], **data})
                except Exception as exc:
                    message = traceback.format_exc() if self.report_mode == "Verbose" else str(exc)
                    outcomes.append(f"{source.name}: FAILED — {message}")
                    records.append({"source": str(source), "status": "failed", "error": str(exc)})
                    failed = True
                if failed or data.get("status") == "cancelled":
                    self.failed_item.emit(item)
                for planned_source in item.plan.sources:
                    self.file_result.emit(
                        {
                            "source": str(planned_source),
                            "operation": index - 1,
                            "status": "failed" if failed else str(data.get("status", "completed")),
                            "error": message if failed else "",
                        }
                    )
                self.progress_changed.emit(index, len(prepared))
                if failed and self.settings.fail_fast:
                    for retry_item in prepared[index:]:
                        self.failed_item.emit(retry_item)
                    outcomes.append("Stopped after the first failure.")
                    self.batch_stopped.emit("Stopped after error")
                    break
        except DiscoveryCancelled as exc:
            cancelled = True
            outcomes.append(str(exc))
            self.batch_stopped.emit("Cancelled")
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

    def preview_content(self, prepared: tuple[PreparedFile, ...]) -> str:
        from .application.conversion import convert_srt
        from .application.exporter import _read_payload, render_artifact
        from .languages import get_language_processor
        from .models import Transcript

        for item in prepared:
            for artifact in item.plan.artifacts:
                if artifact.format not in (ArtifactFormat.SRT, ArtifactFormat.SRT_MINI, ArtifactFormat.TXT):
                    continue
                if self.isInterruptionRequested():
                    raise DiscoveryCancelled("Content preview cancelled")
                if self.action == "Convert Script":
                    text = artifact.source.read_text(encoding="utf-8-sig")
                    content = (
                        convert_srt(text, self.mode)
                        if artifact.format is ArtifactFormat.SRT
                        else get_language_processor("uzbek").transform_text(text, target=self.mode)
                    )
                elif self.action == "Export" and item.export is not None:
                    payload = _read_payload(artifact.source)
                    transcript = Transcript.from_payload(payload)
                    transcript = replace_settings(
                        transcript, metadata={**dict(transcript.metadata), "source_name": artifact.source.stem}
                    )
                    content = render_artifact(artifact.format, transcript, payload, item.export)
                else:
                    continue
                limit = 12000
                suffix = "\n\n[Preview truncated]" if len(content) > limit else ""
                action = ""
                if artifact.target.exists() or artifact.target.is_symlink():
                    if self.policy is ConflictPolicy.SKIP:
                        action = "\nThe existing output will be skipped if unchanged; the text below is a preview only."
                    elif self.policy is ConflictPolicy.REPLACE:
                        action = "\nThe existing output will be replaced if unchanged when the batch runs."
                return f"Content preview: {artifact.source} → {artifact.target}{action}\n\n{content[:limit]}{suffix}"
        return "No SRT, SRT-mini, or TXT content is available to preview for this selection."

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
            needs_request = self.policy is not ConflictPolicy.SKIP or any(
                not (artifact.target.exists() or artifact.target.is_symlink()) for artifact in item.plan.artifacts
            )
            if needs_request and not item.plan.api_requests:
                raise ValueError("Output files changed after planning; start a new batch and review the plan again.")
            provider = (
                ElevenLabsProvider(env_file=self.settings.env_file or self.settings.project_directory / ".env")
                if needs_request
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
                "status": (
                    "failed"
                    if result.failed
                    else "cancelled"
                    if any(entry.message == "cancelled before publication" for entry in result.artifacts)
                    else "skipped"
                    if result.artifacts and result.skipped == len(result.artifacts)
                    else "partial"
                    if result.skipped
                    else "ok"
                ),
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
    output_subfolder: bool = False
    conflict: int = 0
    fail_fast: bool = False
    review: bool = False


@dataclass
class BatchState:
    rows: list[list[QTableWidgetItem | None]]
    failed_items: list[PreparedFile]
    retry_context: tuple[str, ScriptMode, tuple[ArtifactFormat, ...], DesktopSettings] | None
    retry_table_snapshot: list[list[QTableWidgetItem | None]] | None
    preview_valid: bool
    details: str
    progress_maximum: int
    progress_value: int


class MainWindow(QMainWindow):
    def __init__(self, settings: QSettings | None = None) -> None:
        super().__init__()
        self.persistent_settings = settings
        self.setWindowTitle("ElevenLabs Toolkit")
        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction("Using the Toolkit", self.show_help)
        help_menu.addAction("Credentials and API Costs", self.show_credentials_help)
        help_menu.addAction("About ElevenLabs Toolkit", self.show_about)
        self.resize(1020, 800)
        self.paths: list[Path] = []
        self.workflow_paths: dict[int, list[Path]] = {}
        self.workflow_controls: dict[int, WorkflowControls] = {}
        self.batch_states: dict[int, BatchState] = {}
        self.failed_items: list[PreparedFile] = []
        self.retry_table_snapshot: list[list[QTableWidgetItem | None]] | None = None
        self.preview_valid = False
        self.retry_context: tuple[str, ScriptMode, tuple[ArtifactFormat, ...], DesktopSettings] | None = None
        if settings is not None:
            try:
                saved = json.loads(str(settings.value("workflow_controls", "{}")))
                self.workflow_controls = {int(key): WorkflowControls(**value) for key, value in saved.items()}
            except (ValueError, TypeError, AttributeError):
                self.workflow_controls = {}
            geometry = settings.value("geometry")
            if geometry is not None:
                self.restoreGeometry(geometry)
        self.active_workflow = 0
        self.project_directory = DesktopSettings().project_directory
        if settings is not None:
            saved_project = str(settings.value("project_directory", ""))
            if saved_project and Path(saved_project).is_dir():
                self.project_directory = Path(saved_project)
        self.recent_input_directory = str(settings.value("recent_input_directory", "")) if settings is not None else ""
        self.recent_output_directory = (
            str(settings.value("recent_output_directory", "")) if settings is not None else ""
        )
        root = QWidget()
        outer = QHBoxLayout(root)
        self.navigation = QListWidget()
        self.navigation.addItems(["Transcribe", "Convert Script", "Export", "Clean", "Inspect", "Configuration"])
        self.navigation.setFixedWidth(180)
        self.navigation.setCurrentRow(0)
        outer.addWidget(self.navigation)
        self.main_scroll = QScrollArea()
        self.main_scroll.setWidgetResizable(True)
        self.main_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        panel_widget = QWidget()
        panel = QVBoxLayout(panel_widget)
        panel.setContentsMargins(0, 0, 0, 0)
        self.main_scroll.setWidget(panel_widget)
        outer.addWidget(self.main_scroll, 1)

        self.heading = QLabel("Transcribe audio or video to reusable JSON")
        panel.addWidget(self.heading)
        self.formats = QLabel("JSON is always saved · API charges may apply")
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

        self.input_label = QLabel("Inputs — drop files or folders below")
        panel.addWidget(self.input_label)
        self.file_list = FileList(self.add_paths)
        self.file_list.setAccessibleName("Selected input files and folders")
        self.file_list.setToolTip("Drop files or folders here")
        panel.addWidget(self.file_list, 1)
        self.batch_label = QLabel("Batch plan and results")
        panel.addWidget(self.batch_label)
        self.batch_table = BatchTable(self.add_paths)
        self.batch_table.setAccessibleName("Batch plan and results")
        self.batch_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.batch_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.batch_table.setHorizontalHeaderLabels(["Input", "Output", "Status", "Error"])
        self.batch_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        header = self.batch_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.batch_table.setToolTip("Drop files or folders here; preview or run to see individual results")
        panel.addWidget(self.batch_table, 1)
        self.save_report = QPushButton("Save Batch Report…")
        self.save_report.clicked.connect(self.save_batch_report)
        self.open_output = QPushButton("Open Selected Output Folder")
        self.open_output.clicked.connect(self.open_output_folder)
        batch_actions = QHBoxLayout()
        batch_actions.addWidget(self.open_output)
        self.retry_button = QPushButton("Retry Unfinished")
        self.retry_button.setEnabled(False)
        self.retry_button.setToolTip(
            "Retry failed, cancelled or not-run items using their original settings and destinations; existing outputs are skipped"
        )
        self.retry_button.clicked.connect(self.retry_failed)
        batch_actions.addWidget(self.retry_button)
        batch_actions.addWidget(self.save_report)
        panel.addLayout(batch_actions)

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
        format_row = QGridLayout()
        export_layout.addLayout(format_row)
        self.export_formats = {}
        for index, artifact_format in enumerate(
            (
                ArtifactFormat.SRT,
                ArtifactFormat.SRT_MINI,
                ArtifactFormat.TXT,
                ArtifactFormat.COMBINED_TXT,
                ArtifactFormat.RESOLVE_EDL,
                ArtifactFormat.CUE_INDEX_SRT,
                ArtifactFormat.CLEAN_JSON,
            )
        ):
            label, description = FORMAT_DESCRIPTIONS[artifact_format]
            checkbox = QCheckBox(label)
            checkbox.setToolTip(description + " Rendered offline from transcript JSON.")
            checkbox.setChecked(artifact_format is ArtifactFormat.SRT)
            self.export_formats[artifact_format] = checkbox
            format_row.addWidget(checkbox, index // 3, index % 3)
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
        self.output_subfolder = QCheckBox("Use Output subfolder")
        self.output_subfolder.setToolTip(
            "Save into an Output folder inside each source folder instead of a custom folder."
        )
        choose_output = QPushButton("Browse…")
        choose_output.clicked.connect(self.select_output_directory)
        self.output_subfolder.toggled.connect(self.output_directory.setDisabled)
        self.output_subfolder.toggled.connect(choose_output.setDisabled)
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
            self.output_subfolder,
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
        self.show_effective = QPushButton("Show Effective Settings")
        self.show_effective.setToolTip("Show profile settings and values explicitly overridden in this workflow")
        batch_actions.addWidget(self.show_effective)
        self.preview_content_button = QPushButton("Preview Content")
        self.preview_content_button.setToolTip(
            "Preview the first supported subtitle or text output without writing files"
        )
        self.preview_content_button.hide()
        self.stop = QPushButton("Stop After Current File")
        self.stop.setEnabled(False)
        self.start = QPushButton("Transcribe")
        self.start.setDefault(True)
        status_row = QHBoxLayout()
        for status_widget in (self.progress, self.report_mode, self.review_before_running):
            status_row.addWidget(status_widget)
        panel.addLayout(status_row)
        for widget in (self.preview, self.preview_content_button, self.stop, self.start):
            bottom.addWidget(widget)
        panel.addLayout(bottom)

        self.setCentralWidget(root)
        self.add_files.clicked.connect(self.select_files)
        self.add_folder.clicked.connect(self.select_folder)
        self.remove.clicked.connect(self.remove_selected)
        self.clear.clicked.connect(self.clear_files)
        self.navigation.currentRowChanged.connect(self.change_workflow)
        self.preview.clicked.connect(self.show_preview)
        self.show_effective.clicked.connect(self.show_workflow_configuration)
        self.preview_content_button.clicked.connect(lambda: self.launch_worker(preview_only=True, content_preview=True))
        self.start.clicked.connect(self.start_job)
        self.stop.clicked.connect(self.stop_job)
        self.refresh_profiles()
        self.config_profile.currentTextChanged.connect(self.configuration_profile_changed)
        self.default_options = self.option_preferences()
        self.preferences_menu = self.menuBar().addMenu("Preferences")
        self.preferences_menu.addAction("Save Preset…", self.prompt_save_preset)
        self.preferences_menu.addAction("Load Preset…", self.prompt_load_preset)
        self.preferences_menu.addAction("Reset to Defaults", self.reset_preferences)
        controls = self.workflow_controls.get(0, WorkflowControls())
        self.recursive.setChecked(controls.recursive)
        self.include_generated.setChecked(controls.include_generated)
        self.filter_kind.setCurrentIndex(controls.filter_kind)
        self.pattern.setText(controls.pattern)
        self.output_directory.setText(controls.output_directory)
        self.output_subfolder.setChecked(controls.output_subfolder)
        self.conflict.setCurrentIndex(controls.conflict if controls.conflict != 3 else 0)
        self.fail_fast.setChecked(controls.fail_fast)
        self.review_before_running.setChecked(controls.review)
        self.restore_option_preferences()
        for line_field in self.findChildren(QLineEdit):
            line_field.textChanged.connect(self.invalidate_preview)
        for text_field in self.findChildren(QPlainTextEdit):
            if text_field is not self.details:
                text_field.textChanged.connect(self.invalidate_preview)
        for combo_field in self.findChildren(QComboBox):
            combo_field.currentIndexChanged.connect(self.invalidate_preview)
        for checkbox_field in self.findChildren(QCheckBox):
            checkbox_field.toggled.connect(self.invalidate_preview)

    def preference_controls(self) -> dict[str, QWidget]:
        controls: dict[str, QWidget] = {"mode": self.mode, "report_mode": self.report_mode}
        for prefix, panel in (
            ("transcription", self.transcription_options),
            ("export", self.export_text),
            ("clean", self.clean_text),
        ):
            controls.update({f"{prefix}.{key}": value for key, value in panel.preference_controls().items()})
        controls.update({f"export_format.{key.value}": value for key, value in self.export_formats.items()})
        return controls

    def restore_option_preferences(self) -> None:
        if self.persistent_settings is None:
            return
        try:
            values = json.loads(str(self.persistent_settings.value("option_preferences", "{}")))
            if not isinstance(values, dict):
                return
        except (ValueError, TypeError):
            return
        self.apply_option_preferences(values)

    def apply_option_preferences(self, values: dict) -> None:
        for key, widget in self.preference_controls().items():
            value = values.get(key)
            if isinstance(widget, QCheckBox) and isinstance(value, bool):
                widget.setChecked(value)
            elif isinstance(value, str):
                if isinstance(widget, QComboBox):
                    widget.setCurrentText(value)
                elif isinstance(widget, QLineEdit):
                    widget.setText(value)
                elif isinstance(widget, QPlainTextEdit):
                    widget.setPlainText(value)

    def option_preferences(self) -> dict[str, str | bool]:
        values: dict[str, str | bool] = {}
        for key, widget in self.preference_controls().items():
            if isinstance(widget, QCheckBox):
                values[key] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                values[key] = widget.currentText()
            elif isinstance(widget, QLineEdit):
                values[key] = widget.text()
            elif isinstance(widget, QPlainTextEdit):
                values[key] = widget.toPlainText()
        return values

    def apply_shared_controls(self, controls: WorkflowControls) -> None:
        self.recursive.setChecked(controls.recursive)
        self.include_generated.setChecked(controls.include_generated)
        self.filter_kind.setCurrentIndex(controls.filter_kind)
        self.pattern.setText(controls.pattern)
        self.output_directory.setText(controls.output_directory)
        self.output_subfolder.setChecked(controls.output_subfolder)
        self.conflict.setCurrentIndex(controls.conflict if self.active_workflow != 0 or controls.conflict != 3 else 0)
        self.fail_fast.setChecked(controls.fail_fast)
        self.review_before_running.setChecked(controls.review)

    def presets(self) -> dict:
        if self.persistent_settings is None:
            return {}
        try:
            value = json.loads(str(self.persistent_settings.value("presets", "{}")))
            return value if isinstance(value, dict) else {}
        except (ValueError, TypeError):
            return {}

    def save_preset(self, name: str) -> None:
        name = name.strip()
        if not name or self.persistent_settings is None:
            return
        self.workflow_controls[self.active_workflow] = self.current_controls()
        presets = self.presets()
        presets[name] = {
            "options": self.option_preferences(),
            "workflows": {str(key): asdict(value) for key, value in self.workflow_controls.items()},
            "project_directory": str(self.project_directory),
        }
        self.persistent_settings.setValue("presets", json.dumps(presets))
        self.persistent_settings.sync()
        self.details.setPlainText(f"Preset saved: {name}")

    def load_preset(self, name: str) -> None:
        preset = self.presets().get(name)
        try:
            if not isinstance(preset, dict) or not isinstance(preset.get("options"), dict):
                raise ValueError("Preset not found or invalid")
            workflows = {int(key): WorkflowControls(**value) for key, value in preset["workflows"].items()}
            project = Path(preset.get("project_directory", str(self.project_directory)))
            if not project.is_dir():
                raise ValueError(f"Project folder is unavailable: {project}")
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            self.details.setPlainText(f"Could not load preset: {exc}")
            return
        self.invalidate_all_previews()
        self.workflow_controls = workflows
        self.project_directory = project
        self.project_label.setText(str(project))
        self.refresh_profiles()
        self.apply_shared_controls(workflows.get(self.active_workflow, WorkflowControls()))
        self.apply_option_preferences(preset["options"])
        self.details.setPlainText(f"Preset loaded: {name}")

    def reset_preferences(self) -> None:
        self.invalidate_all_previews()
        self.workflow_controls.clear()
        self.project_directory = DesktopSettings().project_directory
        self.project_label.setText(str(self.project_directory))
        self.refresh_profiles()
        self.config_profile.setCurrentText("Project default")
        self.recent_input_directory = ""
        self.recent_output_directory = ""
        self.apply_shared_controls(WorkflowControls())
        self.apply_option_preferences(self.default_options)
        self.transcription_options.env_file.clear()
        self.details.setPlainText("Settings reset to defaults. Saved presets are unchanged.")

    def prompt_save_preset(self) -> None:
        name, accepted = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if accepted and name.strip():
            if (
                name.strip() in self.presets()
                and QMessageBox.question(self, "Replace Preset", f"Replace the preset '{name.strip()}'?")
                != QMessageBox.StandardButton.Yes
            ):
                return
            self.save_preset(name)

    def prompt_load_preset(self) -> None:
        names = sorted(self.presets())
        if not names:
            self.details.setPlainText("No saved presets.")
            return
        name, accepted = QInputDialog.getItem(self, "Load Preset", "Preset:", names, 0, False)
        if accepted:
            self.load_preset(name)

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

    def show_credentials_help(self) -> None:
        QMessageBox.information(
            self,
            "Credentials and API Costs",
            "Transcription sends audio to ElevenLabs and API charges may apply. "
            "Preview shows planned requests and the maximum number of attempts. "
            "Conversion, export, cleaning, and inspection work offline.\n\n"
            "Create a plain-text file named .env containing a line such as "
            "ELEVENLABS_API_KEY=your-key, using your own ElevenLabs API key. "
            "Place it in the project folder selected under Configuration, or use "
            "Transcribe → Credentials file → Browse to select an existing file. "
            "On macOS, Command-Shift-Period shows hidden files in the file picker.\n\n"
            "An ELEVENLABS_API_KEY in the process environment takes precedence over the file. "
            "The selected credentials-file path is not saved in settings or presets; "
            "select it again after restarting if it differs from the project default. "
            "Keep the key and file private.",
        )

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
            self.invalidate_all_previews()
            self.project_directory = Path(name)
            self.project_label.setText(name)
            self.refresh_profiles()
            self.invalidate_preview()
            if self.active_workflow == 5:
                self.show_configuration()

    def configuration_profile_changed(self, _profile: str) -> None:
        if self.active_workflow == 5:
            self.show_configuration()

    def desktop_settings(self) -> DesktopSettings:
        settings = self.workflow_settings()
        folder = self.output_directory.text().strip()
        return replace_settings(
            settings,
            output_directory=self.project_directory / Path(folder).expanduser()
            if folder and not self.output_subfolder.isChecked()
            else None,
            output_subfolder=self.output_subfolder.isChecked(),
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

    def show_workflow_configuration(self) -> None:
        from .config import configuration_sources

        try:
            settings = self.workflow_settings()
            clean = self.active_workflow == 3
            overrides = settings.configuration_overrides(clean=clean)
            profile_settings, sources = configuration_sources(
                settings.profile, overrides=overrides, cwd=settings.project_directory
            )
            self.details.setPlainText(
                json.dumps(
                    {
                        "effective": settings.configuration(clean=clean),
                        "profile_settings": profile_settings,
                        "gui_overrides": overrides,
                        "setting_sources": sources,
                        "note": "Profile settings are shown before workflow overrides; setting sources identify the winning value.",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        except Exception as exc:
            self.details.setPlainText(f"Configuration error: {exc}")

    def change_workflow(self, row: int) -> None:
        self.batch_states[self.active_workflow] = BatchState(
            self.copy_batch_table(),
            list(self.failed_items),
            self.retry_context,
            self.retry_table_snapshot,
            self.preview_valid,
            self.details.toPlainText(),
            self.progress.maximum(),
            self.progress.value(),
        )
        self.preview_valid = False
        self.batch_table.setRowCount(0)
        self.failed_items.clear()
        self.retry_table_snapshot = None
        self.retry_context = None
        self.retry_button.setText("Retry Unfinished")
        self.retry_button.setEnabled(False)
        self.workflow_paths[self.active_workflow] = list(self.paths)
        self.workflow_controls[self.active_workflow] = self.current_controls()
        self.active_workflow = row
        controls = self.workflow_controls.get(row, WorkflowControls())
        self.recursive.setChecked(controls.recursive)
        self.include_generated.setChecked(controls.include_generated)
        self.filter_kind.setCurrentIndex(controls.filter_kind)
        self.pattern.setText(controls.pattern)
        self.output_directory.setText(controls.output_directory)
        self.output_subfolder.setChecked(controls.output_subfolder)
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
        for widget in (
            self.add_files,
            self.add_folder,
            self.remove,
            self.clear,
            self.file_list,
            self.input_label,
            self.batch_label,
            self.batch_table,
            self.open_output,
            self.save_report,
            self.retry_button,
        ):
            widget.setVisible(row != 5)
        self.input_controls.setVisible(row != 5)
        self.include_generated.setVisible(row in (2, 3, 4))
        self.output_controls.setVisible(row < 4)
        self.rename_option.setEnabled(row != 0)
        if row == 0 and self.conflict_policy() is ConflictPolicy.RENAME:
            self.conflict.setCurrentIndex(0)
        self.preview.setVisible(row != 5)
        self.show_effective.setVisible(row in (0, 2, 3))
        self.preview_content_button.hide()
        self.review_before_running.hide()
        self.review_before_running.setVisible(row in (0, 2))
        self.preview_content_button.setVisible(row in (1, 2))
        state = self.batch_states.pop(row, None)
        if state is not None:
            self.restore_batch_table(state.rows)
            self.failed_items = state.failed_items
            self.retry_context = state.retry_context
            self.retry_table_snapshot = state.retry_table_snapshot
            self.preview_valid = state.preview_valid
            self.retry_button.setEnabled(bool(self.failed_items) and self.retry_context is not None)
            self.retry_button.setText(
                "Retry Previous Batch" if self.retry_table_snapshot is not None else "Retry Unfinished"
            )
            self.details.setPlainText(state.details)
            self.progress.setRange(0, state.progress_maximum)
            self.progress.setValue(state.progress_value)
        else:
            self.details.clear()
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
        if row == 5:
            self.show_configuration()

    def current_controls(self) -> WorkflowControls:
        return WorkflowControls(
            recursive=self.recursive.isChecked(),
            include_generated=self.include_generated.isChecked(),
            filter_kind=self.filter_kind.currentIndex(),
            pattern=self.pattern.text(),
            output_directory=self.output_directory.text(),
            output_subfolder=self.output_subfolder.isChecked(),
            conflict=self.conflict.currentIndex(),
            fail_fast=self.fail_fast.isChecked(),
            review=self.review_before_running.isChecked(),
        )

    def add_paths(self, paths: list[Path]) -> None:
        changed = False
        for path in paths:
            path = path.resolve()
            if path not in self.paths:
                self.paths.append(path)
                self.file_list.addItem(str(path))
                changed = True
        self.details.setPlainText(f"{len(self.paths)} file(s) or folder(s) selected")
        if changed:
            self.invalidate_preview()

    def select_output_directory(self) -> None:
        name = QFileDialog.getExistingDirectory(
            self, "Output folder", self.output_directory.text() or self.recent_output_directory
        )
        if name:
            self.output_directory.setText(name)
            self.recent_output_directory = name

    def conflict_policy(self) -> ConflictPolicy:
        return ConflictPolicy(self.conflict.currentData())

    def select_files(self) -> None:
        names, _ = QFileDialog.getOpenFileNames(self, "Select files", self.recent_input_directory)
        if names:
            self.recent_input_directory = str(Path(names[0]).parent)
        self.add_paths([Path(name) for name in names])

    def select_folder(self) -> None:
        name = QFileDialog.getExistingDirectory(self, "Select folder", self.recent_input_directory)
        if name:
            self.recent_input_directory = name
            self.add_paths([Path(name)])

    def remove_selected(self) -> None:
        indices = sorted((self.file_list.row(item) for item in self.file_list.selectedItems()), reverse=True)
        for index in indices:
            self.file_list.takeItem(index)
            self.paths.pop(index)
        if indices:
            self.invalidate_preview()

    def clear_files(self) -> None:
        had_paths = bool(self.paths)
        self.paths.clear()
        self.file_list.clear()
        if had_paths:
            self.invalidate_preview()

    def invalidate_preview(self, *_args: object) -> None:
        if not self.preview_valid:
            return
        self.preview_valid = False
        for row in range(self.batch_table.rowCount()):
            self.batch_table.setItem(row, 2, QTableWidgetItem("Preview out of date"))
        self.details.setPlainText("Inputs or settings changed. Preview again before relying on the displayed plan.")

    def invalidate_all_previews(self) -> None:
        self.invalidate_preview()
        for state in self.batch_states.values():
            if state.preview_valid:
                state.preview_valid = False
                for cells in state.rows:
                    cells[2] = QTableWidgetItem("Preview out of date")
                state.details = "Inputs or settings changed. Preview again before relying on the displayed plan."

    def selected_formats(self) -> tuple[ArtifactFormat, ...]:
        return tuple(value for value, checkbox in self.export_formats.items() if checkbox.isChecked())

    def show_preview(self) -> None:
        self.launch_worker(preview_only=True)

    def start_job(self) -> None:
        if self.active_workflow == 5:
            self.show_configuration()
        else:
            self.launch_worker(preview_only=False)

    def launch_worker(self, *, preview_only: bool, content_preview: bool = False) -> None:
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
        self.pending_retry_context = None if preview_only else (action, mode, self.selected_formats(), settings)
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
            content_preview=content_preview,
            report_mode=self.report_mode.currentText(),
            review_before_running=not preview_only
            and self.active_workflow in (0, 2)
            and self.review_before_running.isChecked(),
        )
        if self.report_mode.currentText() != "Quiet":
            self.worker.status.connect(self.details.setPlainText)
        self.worker.review_requested.connect(self.review_plan)
        self.worker.plan_ready.connect(self.activate_batch_plan)
        self.worker.file_result.connect(self.show_file_result)
        self.worker.batch_stopped.connect(self.mark_unprocessed)
        self.worker.failed_item.connect(self.record_failed_item)
        self.worker.finished_result.connect(self.job_finished)
        self.worker.finished.connect(self.worker_finished)
        self.worker.progress_changed.connect(self.update_progress)
        self.set_running(True)
        self.worker.start()

    def retry_failed(self) -> None:
        if not self.start.isEnabled() or not self.failed_items or self.retry_context is None:
            return
        action, mode, formats, settings = self.retry_context
        items = tuple(self.failed_items)
        if self.retry_table_snapshot is not None:
            self.restore_batch_table(self.retry_table_snapshot)
            self.retry_table_snapshot = None
        self.preview_valid = False
        self.failed_items.clear()
        self.retry_button.setEnabled(False)
        self.worker = Worker(
            action,
            mode,
            [],
            False,
            formats,
            settings,
            policy=ConflictPolicy.SKIP,
            retry_plan=items,
            report_mode=self.report_mode.currentText(),
            review_before_running=action == "Transcribe",
        )
        if self.report_mode.currentText() != "Quiet":
            self.worker.status.connect(self.details.setPlainText)
        self.worker.review_requested.connect(self.review_plan)
        self.worker.plan_ready.connect(self.show_retry_plan)
        self.worker.file_result.connect(self.show_file_result)
        self.worker.batch_stopped.connect(self.mark_unprocessed)
        self.worker.failed_item.connect(self.record_failed_item)
        self.worker.finished_result.connect(self.job_finished)
        self.worker.finished.connect(self.worker_finished)
        self.worker.progress_changed.connect(self.update_progress)
        self.set_running(True)
        self.worker.start()

    def record_failed_item(self, item: PreparedFile) -> None:
        self.failed_items.append(item)

    def activate_batch_plan(self, rows: list[dict]) -> None:
        if self.pending_retry_context is None:
            if self.failed_items and self.retry_table_snapshot is None:
                self.retry_table_snapshot = self.copy_batch_table()
        else:
            self.failed_items.clear()
            self.retry_table_snapshot = None
            self.retry_context = self.pending_retry_context
        self.preview_valid = self.pending_retry_context is None
        self.show_batch_plan(rows)

    def copy_batch_table(self) -> list[list[QTableWidgetItem | None]]:
        return [
            [
                item.clone() if (item := self.batch_table.item(row, column)) is not None else None
                for column in range(self.batch_table.columnCount())
            ]
            for row in range(self.batch_table.rowCount())
        ]

    def restore_batch_table(self, rows: list[list[QTableWidgetItem | None]]) -> None:
        self.batch_table.setRowCount(len(rows))
        for row, cells in enumerate(rows):
            for column, cell in enumerate(cells):
                if cell is not None:
                    self.batch_table.setItem(row, column, cell)

    def show_batch_plan(self, rows: list[dict]) -> None:
        self.batch_table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            for column, value in enumerate((row["source"], "\n".join(row["outputs"]), "Pending", "")):
                self.batch_table.setItem(index, column, QTableWidgetItem(value))
            source_item = self.batch_table.item(index, 0)
            if source_item is not None:
                source_item.setData(Qt.ItemDataRole.UserRole, row["operation"])

    def show_retry_plan(self, rows: list[dict]) -> None:
        available = set(range(self.batch_table.rowCount()))
        for index in available:
            source_item = self.batch_table.item(index, 0)
            if source_item is not None:
                source_item.setData(Qt.ItemDataRole.UserRole, None)
        for row in rows:
            outputs = "\n".join(row["outputs"])
            index = -1
            for candidate in sorted(available):
                source_item = self.batch_table.item(candidate, 0)
                output_item = self.batch_table.item(candidate, 1)
                if (
                    source_item
                    and output_item
                    and source_item.text() == row["source"]
                    and output_item.text() == outputs
                ):
                    index = candidate
                    break
            if index < 0:
                index = self.batch_table.rowCount()
                self.batch_table.insertRow(index)
                self.batch_table.setItem(index, 0, QTableWidgetItem(row["source"]))
                self.batch_table.setItem(index, 1, QTableWidgetItem(outputs))
            else:
                available.remove(index)
            source_item = self.batch_table.item(index, 0)
            if source_item is not None:
                source_item.setData(Qt.ItemDataRole.UserRole, row["operation"])
            self.batch_table.setItem(index, 2, QTableWidgetItem("Pending"))
            self.batch_table.setItem(index, 3, QTableWidgetItem(""))

    def show_file_result(self, result: dict) -> None:
        for row in range(self.batch_table.rowCount()):
            source_item = self.batch_table.item(row, 0)
            if (
                source_item is not None
                and source_item.text() == result["source"]
                and source_item.data(Qt.ItemDataRole.UserRole) == result["operation"]
            ):
                self.batch_table.setItem(row, 2, QTableWidgetItem(result["status"]))
                self.batch_table.setItem(row, 3, QTableWidgetItem(result["error"]))

    def mark_unprocessed(self, reason: str) -> None:
        for row in range(self.batch_table.rowCount()):
            status = self.batch_table.item(row, 2)
            if status is not None and status.text() == "Pending":
                self.batch_table.setItem(row, 2, QTableWidgetItem(f"Not run — {reason.lower()}"))

    def save_batch_report(self) -> None:
        from .files import atomic_write_text

        if not self.batch_table.rowCount():
            self.details.setPlainText("Preview or run a batch before saving a report.")
            return
        name, _ = QFileDialog.getSaveFileName(self, "Save Batch Report", "batch-report.json", "JSON files (*.json)")
        if not name:
            return
        rows = []
        for row in range(self.batch_table.rowCount()):
            values = []
            for column in range(4):
                item = self.batch_table.item(row, column)
                values.append(item.text() if item else "")
            rows.append(dict(zip(("input", "output", "status", "error"), values, strict=True)))
        try:
            atomic_write_text(
                Path(name), json.dumps({"files": rows}, ensure_ascii=False, indent=2), ConflictPolicy.ERROR
            )
            self.details.setPlainText(f"Batch report saved: {name}")
        except (OSError, ValueError) as exc:
            self.details.setPlainText(f"Could not save batch report: {exc}")

    def open_output_folder(self) -> None:
        selected = self.batch_table.selectedItems()
        if not selected:
            self.details.setPlainText("Select a batch row to open its output folder.")
            return
        item = self.batch_table.item(selected[0].row(), 1)
        targets = item.text().splitlines() if item else []
        if not targets:
            self.details.setPlainText("This input has no output folder.")
            return
        folder = Path(targets[0]).parent
        if not folder.is_dir():
            self.details.setPlainText(f"Output folder does not exist yet: {folder}")
        elif not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
            self.details.setPlainText(f"Could not open output folder: {folder}")

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
        self.preferences_menu.setEnabled(not running)
        self.retry_button.setText(
            "Retry Previous Batch" if self.retry_table_snapshot is not None else "Retry Unfinished"
        )
        self.retry_button.setEnabled(not running and bool(self.failed_items) and self.retry_context is not None)
        for widget in (
            self.navigation,
            self.add_files,
            self.add_folder,
            self.remove,
            self.clear,
            self.file_list,
            self.batch_table,
            self.save_report,
            self.open_output,
            self.options,
            self.input_controls,
            self.output_controls,
            self.report_mode,
            self.review_before_running,
            self.preview,
            self.show_effective,
            self.preview_content_button,
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
        if self.persistent_settings is not None:
            self.workflow_controls[self.active_workflow] = self.current_controls()
            self.persistent_settings.setValue(
                "workflow_controls", json.dumps({key: asdict(value) for key, value in self.workflow_controls.items()})
            )
            self.persistent_settings.setValue("geometry", self.saveGeometry())
            self.persistent_settings.setValue("project_directory", str(self.project_directory))
            self.persistent_settings.setValue("recent_input_directory", self.recent_input_directory)
            self.persistent_settings.setValue("recent_output_directory", self.recent_output_directory)
            self.persistent_settings.setValue("option_preferences", json.dumps(self.option_preferences()))
            self.persistent_settings.sync()
        event.accept()


def self_test() -> int:
    """Exercise a local conversion through the actual window and worker."""
    from tempfile import TemporaryDirectory

    import elevenlabs  # noqa: F401 - verify the optional SDK is included in the bundle

    app = QApplication.instance() or QApplication(["elevenlabs-toolkit-self-test"])
    with TemporaryDirectory(prefix="elevenlabs-desktop-test-") as temporary:
        source = Path(temporary) / "sample.srt"
        original = "1\n00:00:00,000 --> 00:00:01,000\nSalom\n"
        source.write_text(original, encoding="utf-8")
        window = MainWindow()
        window.show()
        app.processEvents()
        for row in range(6):
            window.navigation.setCurrentRow(row)
            app.processEvents()
        window.navigation.setCurrentRow(1)
        window.add_paths([source])
        window.start_job()
        if not window.worker.wait(10000):
            raise RuntimeError("Desktop self-test timed out during conversion")
        app.processEvents()
        output = source.parent / "sample CYR.srt"
        if source.read_text(encoding="utf-8") != original or "Салом" not in output.read_text(encoding="utf-8"):
            raise RuntimeError("Desktop self-test conversion produced an unexpected result")
        window.close()
        app.processEvents()
    print("Desktop self-test passed")
    return 0


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow(settings=QSettings("ElevenLabs Toolkit", "Desktop"))
    window.show()
    return app.exec()
