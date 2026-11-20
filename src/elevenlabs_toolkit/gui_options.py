from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .application.desktop import DesktopSettings
from .cli.transcribe import LOCAL_FORMATS, REMOTE_FORMATS
from .config import ConfigurationError, available_profiles
from .languages.replacements import replacement_parts
from .models import ArtifactFormat, ScriptMode, TranscriptionOptions

FORMAT_DESCRIPTIONS = {
    ArtifactFormat.SRT: ("Subtitles (SRT)", "Timed subtitles using the selected segmentation and timing settings."),
    ArtifactFormat.SRT_MINI: ("Short subtitles (SRT)", "Short subtitle cues for quick, compact display."),
    ArtifactFormat.TXT: ("Transcript text (TXT)", "Plain text organized into sentences."),
    ArtifactFormat.COMBINED_TXT: ("Combined text (TXT)", "Combine transcripts into one text file with source names."),
    ArtifactFormat.RESOLVE_EDL: (
        "Resolve markers (EDL)",
        "DaVinci Resolve marker list using the selected marker settings.",
    ),
    ArtifactFormat.CUE_INDEX_SRT: ("Cue index (SRT)", "Numbered subtitle cues for identifying corresponding segments."),
    ArtifactFormat.CLEAN_JSON: (
        "Cleaned transcript (JSON)",
        "An Uzbek-cleaned JSON derivative with timing information.",
    ),
}


class CollapsibleSection(QWidget):
    """Show or hide controls without changing whether their values apply."""

    def __init__(self, title: str) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.toggle = QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.content = QWidget()
        self.content.hide()
        self.toggle.toggled.connect(self.content.setVisible)
        self.toggle.toggled.connect(
            lambda expanded: self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        )
        layout.addWidget(self.toggle)
        layout.addWidget(self.content)


class TextOptionsPanel(QWidget):
    """Reusable profile and text controls for offline transcript workflows."""

    def __init__(self, *, clean: bool = False, advanced: bool = False) -> None:
        super().__init__()
        self.clean = clean
        self.numeric_errors: dict[str, QLabel] = {}
        self.cross_errors: dict[str, QLabel] = {}
        self.numeric_inputs: dict[str, QLineEdit] = {}
        self.numeric_labels: dict[str, str] = {}
        layout = QFormLayout(self)
        self.profile = QComboBox()
        self.profile.setEditable(True)
        self.profile.addItem("Project default", None)
        layout.addRow("Profile", self.profile)
        self.script = QComboBox()
        self.script.addItem("Profile default", None)
        for value in ScriptMode:
            self.script.addItem(value.value.title(), value)
        layout.addRow("Output script (Uzbek)", self.script)
        self.cleanup = QComboBox()
        self.cleanup.addItem("Profile default", None)
        self.cleanup.addItem("None", "none")
        self.cleanup.addItem("Uzbek editorial cleanup", "uzbek")
        self.cleanup_error = QLabel()
        self.cleanup_error.setStyleSheet("color: #b00020")
        self.cleanup_error.setWordWrap(True)
        self.cleanup.currentIndexChanged.connect(self.cleanup_error.clear)
        if not clean:
            cleanup_container = QWidget()
            cleanup_layout = QVBoxLayout(cleanup_container)
            cleanup_layout.setContentsMargins(0, 0, 0, 0)
            cleanup_layout.setSpacing(0)
            cleanup_layout.addWidget(self.cleanup)
            cleanup_layout.addWidget(self.cleanup_error)
            layout.addRow("Cleanup", cleanup_container)
        self.override_replacements = QCheckBox("Override profile replacements")
        layout.addRow(self.override_replacements)
        self.replacements = QPlainTextEdit()
        self.replacements.setPlaceholderText("One replacement per line, for example Acme=ACME")
        self.replacements.setMaximumHeight(65)
        self.replacements.setEnabled(False)
        self.replacements.setToolTip("Enter one non-empty TOKEN=TOKEN replacement per line, without spaces in tokens.")
        self.replacement_error = QLabel()
        self.replacement_error.setStyleSheet("color: #b00020")
        self.replacement_error.setWordWrap(True)
        replacement_container = QWidget()
        replacement_layout = QVBoxLayout(replacement_container)
        replacement_layout.setContentsMargins(0, 0, 0, 0)
        replacement_layout.setSpacing(0)
        replacement_layout.addWidget(self.replacements)
        replacement_layout.addWidget(self.replacement_error)
        self.override_replacements.toggled.connect(self.replacements.setEnabled)
        self.override_replacements.toggled.connect(self.validate_replacements)
        self.replacements.textChanged.connect(self.validate_replacements)
        layout.addRow("Token replacements", replacement_container)
        self.advanced = advanced
        self.numeric_fields = {}
        self.boolean_fields = {}
        self.smart = QCheckBox("Balance subtitle lines")
        self.marker_fps = QLineEdit("25")
        self.marker_color = QLineEdit("ResolveColorBlue")
        self.marker_color.setToolTip("DaVinci Resolve marker colour name, such as ResolveColorBlue.")
        self.marker_color_error = QLabel()
        self.marker_color_error.setStyleSheet("color: #b00020")
        self.marker_color_error.setWordWrap(True)
        self.marker_prefix = QLineEdit("Sentence")
        self.marker_prefix.setToolTip("Text placed before each Resolve marker number.")
        self.combined_name = QLineEdit("combined.txt")
        self.combined_name.setToolTip("Filename for combined text output; enter a name without a folder path.")
        self.combined_name_error = QLabel()
        self.combined_name_error.setStyleSheet("color: #b00020")
        self.combined_name_error.setWordWrap(True)
        self.advanced_section: CollapsibleSection | None = None
        if advanced:
            self.advanced_section = CollapsibleSection("Advanced subtitle and Resolve settings")
            advanced_layout = QFormLayout(self.advanced_section.content)
            fields = (
                ("max_chars_per_line", "Maximum characters per line", int),
                ("max_lines", "Maximum lines", int),
                ("max_duration", "Maximum duration (seconds)", float),
                ("min_duration", "Minimum duration (seconds)", float),
                ("gap_seconds", "Pause gap (seconds)", float),
                ("hard_gap_seconds", "Hard pause gap (seconds)", float),
                ("max_words", "Maximum words", int),
                ("srt_fps", "Subtitle FPS", float),
                ("srt_padding_frames", "Subtitle padding (frames)", int),
                ("srt_gap_milliseconds", "Subtitle minimum gap (ms)", int),
            )
            for name, label, converter in fields:
                field = QLineEdit()
                field.setPlaceholderText("Profile default")
                self.numeric_fields[name] = (field, converter)
                allow_zero = name in {"min_duration", "srt_padding_frames", "srt_gap_milliseconds"}
                self.add_numeric_field(
                    advanced_layout,
                    name,
                    label,
                    field,
                    converter,
                    optional=True,
                    minimum=0,
                    exclusive_minimum=not allow_zero,
                )
            for name in ("min_duration", "max_duration", "gap_seconds", "hard_gap_seconds"):
                error = QLabel()
                error.setStyleSheet("color: #b00020")
                error.setWordWrap(True)
                parent = self.numeric_errors[name].parentWidget()
                assert parent is not None
                field_layout = parent.layout()
                assert field_layout is not None
                field_layout.addWidget(error)
                self.cross_errors[name] = error
            for name in ("min_duration", "max_duration", "gap_seconds", "hard_gap_seconds"):
                self.numeric_fields[name][0].textChanged.connect(self.validate_numeric_relationships)
            self.profile.currentTextChanged.connect(self.validate_numeric_relationships)
            for name, label in (
                ("pause_detection", "Pause detection"),
                ("split_on_speaker_change", "Split on speaker change"),
            ):
                choice = QComboBox()
                choice.addItem("Profile default", None)
                choice.addItem("On", True)
                choice.addItem("Off", False)
                self.boolean_fields[name] = choice
                advanced_layout.addRow(label, choice)
            advanced_layout.addRow(self.smart)
            self.add_numeric_field(
                advanced_layout,
                "marker_fps",
                "Resolve marker FPS",
                self.marker_fps,
                float,
                minimum=0,
                exclusive_minimum=True,
            )
            marker_color_container = QWidget()
            marker_color_layout = QVBoxLayout(marker_color_container)
            marker_color_layout.setContentsMargins(0, 0, 0, 0)
            marker_color_layout.setSpacing(0)
            marker_color_layout.addWidget(self.marker_color)
            marker_color_layout.addWidget(self.marker_color_error)
            self.marker_color.textChanged.connect(self.validate_marker_color)
            advanced_layout.addRow("Resolve marker colour", marker_color_container)
            advanced_layout.addRow("Resolve marker prefix", self.marker_prefix)
            combined_container = QWidget()
            combined_layout = QVBoxLayout(combined_container)
            combined_layout.setContentsMargins(0, 0, 0, 0)
            combined_layout.setSpacing(0)
            combined_layout.addWidget(self.combined_name)
            combined_layout.addWidget(self.combined_name_error)
            self.combined_name.textChanged.connect(self.recheck_combined_name)
            advanced_layout.addRow("Combined text filename", combined_container)
            layout.addRow(self.advanced_section)

    def add_numeric_field(
        self,
        form: QFormLayout,
        name: str,
        label: str,
        field: QLineEdit,
        converter: type,
        *,
        optional: bool = False,
        minimum: float | None = None,
        maximum: float | None = None,
        exclusive_minimum: bool = False,
    ) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        error = QLabel()
        error.setStyleSheet("color: #b00020")
        error.setWordWrap(True)
        layout.addWidget(field)
        layout.addWidget(error)
        form.addRow(label, container)
        self.numeric_errors[name] = error
        self.numeric_inputs[name] = field
        self.numeric_labels[name] = label
        hint = "Enter a whole number" if converter is int else "Enter a finite number"
        if minimum is not None and maximum is not None:
            hint += f" between {minimum:g} and {maximum:g}"
        elif minimum is not None and exclusive_minimum:
            hint += f" greater than {minimum:g}"
        elif minimum is not None:
            hint += f" of at least {minimum:g}"
        if optional:
            hint += ", or leave blank"
        field.setToolTip(hint)

        def validate(text: str) -> None:
            if optional and not text.strip():
                error.clear()
                return
            try:
                number = converter(text)
                valid = (
                    math.isfinite(number)
                    and (minimum is None or (number > minimum if exclusive_minimum else number >= minimum))
                    and (maximum is None or number <= maximum)
                )
            except (ValueError, OverflowError):
                valid = False
            error.setText("" if valid else hint)

        field.textChanged.connect(validate)
        validate(field.text())

    def validate_replacements(self) -> None:
        self.replacement_error.clear()
        if not self.override_replacements.isChecked():
            return
        for line_number, line in enumerate(self.replacements.toPlainText().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                replacement_parts(line.strip())
            except ValueError as exc:
                self.replacement_error.setText(f"Line {line_number}: {exc}")
                return

    def validate_numeric_fields(self) -> None:
        for name, error in self.numeric_errors.items():
            if error.text():
                if self.advanced_section is not None and self.advanced_section.content.isAncestorOf(error):
                    self.advanced_section.toggle.setChecked(True)
                self.numeric_inputs[name].setFocus()
                raise ValueError(f"{self.numeric_labels[name]}: {error.text()}")
        self.validate_numeric_relationships()
        for name, error in self.cross_errors.items():
            if error.text():
                if self.advanced_section is not None:
                    self.advanced_section.toggle.setChecked(True)
                self.numeric_inputs[name].setFocus()
                raise ValueError(f"{self.numeric_labels[name]}: {error.text()}")

    def validate_numeric_relationships(self, *_args: object) -> None:
        pairs = (
            ("min_duration", "max_duration", "Minimum duration must not exceed maximum duration."),
            ("gap_seconds", "hard_gap_seconds", "Pause gap must not exceed hard pause gap."),
        )
        for lower, upper, message in pairs:
            if lower not in self.cross_errors:
                continue
            error = self.cross_errors[lower]
            error.clear()
            self.cross_errors[upper].clear()
            low_text = self.numeric_inputs[lower].text().strip()
            high_text = self.numeric_inputs[upper].text().strip()
            if not low_text or not high_text or self.numeric_errors[lower].text() or self.numeric_errors[upper].text():
                continue
            if float(low_text) > float(high_text):
                error.setText(message)

    def refresh_profiles(self, directory: Path) -> None:
        selected = self.profile.currentText()
        profiles = available_profiles(cwd=directory)
        self.profile.clear()
        self.profile.addItem("Project default", None)
        for profile in profiles:
            self.profile.addItem(profile, profile)
        if selected != "Project default":
            self.profile.setCurrentText(selected)

    def preference_controls(self) -> dict[str, QWidget]:
        controls: dict[str, QWidget] = {
            name: getattr(self, name)
            for name in (
                "profile",
                "script",
                "cleanup",
                "override_replacements",
                "replacements",
                "smart",
                "marker_fps",
                "marker_color",
                "marker_prefix",
                "combined_name",
            )
        }
        controls.update({f"number.{key}": field for key, (field, _) in self.numeric_fields.items()})
        controls.update({f"boolean.{key}": field for key, field in self.boolean_fields.items()})
        return controls

    def validate_marker_color(self) -> None:
        self.marker_color_error.setText("Enter a marker colour." if not self.marker_color.text().strip() else "")

    def recheck_combined_name(self) -> None:
        if self.combined_name_error.text():
            self.validate_combined_name()

    def validate_combined_name(self) -> bool:
        from .application.planner import PlanningError, validate_combined_name

        try:
            validate_combined_name(self.combined_name.text())
        except PlanningError:
            self.combined_name_error.setText("Enter a valid filename without a folder path.")
            return False
        self.combined_name_error.clear()
        return True

    def settings(self, directory: Path) -> DesktopSettings:
        self.validate_numeric_fields()
        if self.advanced:
            self.validate_marker_color()
            if self.marker_color_error.text():
                if self.advanced_section is not None:
                    self.advanced_section.toggle.setChecked(True)
                self.marker_color.setFocus()
                raise ValueError(f"Resolve marker colour: {self.marker_color_error.text()}")
        self.validate_replacements()
        if self.replacement_error.text():
            self.replacements.setFocus()
            raise ValueError(f"Token replacements: {self.replacement_error.text()}")
        profile = self.profile.currentText().strip()
        replacements = None
        if self.override_replacements.isChecked():
            replacements = tuple(line.strip() for line in self.replacements.toPlainText().splitlines() if line.strip())
        segmentation = []
        for name, (field, converter) in self.numeric_fields.items():
            value = field.text().strip()
            if value:
                try:
                    segmentation.append((name, converter(value)))
                except ValueError as exc:
                    raise ValueError(f"Invalid value for {name}: {value}") from exc
        for name, choice in self.boolean_fields.items():
            if choice.currentData() is not None:
                segmentation.append((name, choice.currentData()))
        settings = DesktopSettings(
            segmentation=tuple(segmentation),
            smart_line_breaks=self.smart.isChecked(),
            marker_fps=float(self.marker_fps.text()),
            marker_color=self.marker_color.text(),
            marker_prefix=self.marker_prefix.text(),
            combined_name=self.combined_name.text(),
            project_directory=directory,
            profile=None if profile == "Project default" else profile,
            output_script=ScriptMode(self.script.currentData()) if self.script.currentData() is not None else None,
            replacements=replacements,
            cleanup="uzbek" if self.clean else self.cleanup.currentData(),
        )
        relationship_fields = {"min_duration", "max_duration", "gap_seconds", "hard_gap_seconds"}
        if self.advanced and any(name in relationship_fields for name, _ in segmentation):
            try:
                settings.configuration(clean=self.clean)
            except ConfigurationError as exc:
                text = str(exc)
                relationships = (
                    (
                        "min_duration must be <= max_duration",
                        "min_duration",
                        "max_duration",
                        "Minimum duration must not exceed maximum duration.",
                    ),
                    (
                        "hard_gap_seconds must be >= gap_seconds",
                        "gap_seconds",
                        "hard_gap_seconds",
                        "Pause gap must not exceed hard pause gap.",
                    ),
                )
                for marker, lower, upper, message in relationships:
                    if marker in text:
                        field_name = lower if self.numeric_inputs[lower].text().strip() else upper
                        self.cross_errors[field_name].setText(message + " Check the selected profile's value.")
                        if self.advanced_section is not None:
                            self.advanced_section.toggle.setChecked(True)
                        self.numeric_inputs[field_name].setFocus()
                        raise ValueError(self.cross_errors[field_name].text()) from exc
                raise
        return settings


class TranscriptionPanel(TextOptionsPanel):
    def __init__(self) -> None:
        super().__init__(advanced=True)
        group = QWidget()
        self.recognition_grid = QGridLayout(group)
        self.recognition_group = QGroupBox("Recognition")
        form = QFormLayout(self.recognition_group)
        self.output_group = QGroupBox("Additional outputs")
        output_form = QFormLayout(self.output_group)
        self.advanced_group = QGroupBox("Advanced recognition and requests")
        advanced_form = QFormLayout(self.advanced_group)
        self.recognition_grid.addWidget(self.recognition_group, 0, 0)
        self.recognition_grid.addWidget(self.output_group, 1, 0)
        self.recognition_grid.addWidget(self.advanced_group, 2, 0)
        self.model = QLineEdit("scribe_v2")
        self.language = QLineEdit()
        self.language.setPlaceholderText("Auto detect")
        self.timestamps = QComboBox()
        self.timestamps.addItems(["Character", "Word", "None"])
        self.diarize = QCheckBox("Identify speakers")
        self.diarize.setChecked(True)
        self.audio_events = QCheckBox("Tag audio events")
        self.audio_events.setChecked(True)
        self.no_verbatim = QCheckBox("Remove fillers and false starts")
        self.num_speakers = QLineEdit()
        self.seed = QLineEdit()
        self.temperature = QLineEdit()
        self.keyterms = QPlainTextEdit()
        self.keyterms.setMaximumHeight(65)
        self.keyterms.setPlaceholderText("One keyterm per line")
        self.keyterms.setToolTip("One keyterm per line, fewer than 50 characters and at most 5 words each.")
        self.model_error = QLabel()
        self.keyterms_error = QLabel()
        self.verbatim_error = QLabel()
        self.timestamps_error = QLabel()
        for error in (self.model_error, self.keyterms_error, self.verbatim_error, self.timestamps_error):
            error.setStyleSheet("color: #b00020")
            error.setWordWrap(True)

        def with_error(control: QWidget, error: QLabel) -> QWidget:
            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            layout.addWidget(control)
            layout.addWidget(error)
            return container

        model_control = with_error(self.model, self.model_error)
        timestamps_control = with_error(self.timestamps, self.timestamps_error)
        keyterms_control = with_error(self.keyterms, self.keyterms_error)
        verbatim_control = with_error(self.no_verbatim, self.verbatim_error)
        self.model.textChanged.connect(self.validate_transcription_text)
        self.keyterms.textChanged.connect(self.validate_transcription_text)
        self.no_verbatim.toggled.connect(self.validate_transcription_text)
        self.timestamps.currentIndexChanged.connect(self.timestamps_error.clear)
        self.profile.currentTextChanged.connect(self.timestamps_error.clear)
        self.boolean_fields["pause_detection"].currentIndexChanged.connect(self.timestamps_error.clear)
        self.env_file = QLineEdit()
        self.env_file.setPlaceholderText("Project .env")
        self.env_file.setToolTip(
            "Select a .env file containing your secret API key (ELEVENLABS_API_KEY). "
            "The file path is not saved. "
            "See Help → Credentials and API Costs for setup."
        )
        self.choose_credentials = QPushButton("Browse…")
        self.choose_credentials.clicked.connect(self.select_credentials_file)
        credentials = QWidget()
        credentials_layout = QHBoxLayout(credentials)
        credentials_layout.setContentsMargins(0, 0, 0, 0)
        credentials_layout.addWidget(self.env_file)
        credentials_layout.addWidget(self.choose_credentials)
        self.retries = QLineEdit("0")
        self.backoff = QLineEdit("1")
        self.delay = QLineEdit("0")
        for label, control in (
            ("Model", model_control),
            ("Language code", self.language),
            ("Timestamps", timestamps_control),
            ("Speaker count", self.num_speakers),
            ("Seed", self.seed),
            ("Temperature", self.temperature),
            ("Keyterms", keyterms_control),
            ("Credentials file", credentials),
            ("Retries", self.retries),
            ("Retry backoff (seconds)", self.backoff),
            ("Request delay (seconds)", self.delay),
        ):
            numeric = {
                "Speaker count": ("num_speakers", int, True, 1, 32),
                "Seed": ("seed", int, True, 0, 2147483647),
                "Temperature": ("temperature", float, True, 0, 2),
                "Retries": ("retries", int, False, 0, None),
                "Retry backoff (seconds)": ("backoff", float, False, 0, None),
                "Request delay (seconds)": ("delay", float, False, 0, None),
            }
            target_form = (
                advanced_form
                if label
                in {
                    "Seed",
                    "Temperature",
                    "Credentials file",
                    "Retries",
                    "Retry backoff (seconds)",
                    "Request delay (seconds)",
                }
                else form
            )
            if label in numeric:
                name, converter, optional, minimum, maximum = numeric[label]
                assert isinstance(control, QLineEdit)
                self.add_numeric_field(
                    target_form, name, label, control, converter, optional=optional, minimum=minimum, maximum=maximum
                )
            else:
                target_form.addRow(label, control)
        self.model.setToolTip("ElevenLabs speech-to-text model ID. The default is scribe_v2.")
        self.language.setToolTip("Optional language code, such as uz. Leave blank for automatic detection.")
        self.timestamps.setToolTip(
            "Choose character, word, or no timestamps. Timed subtitles need timestamps; pause detection needs character timestamps."
        )
        self.diarize.setToolTip("Identify different speakers in the transcription.")
        self.audio_events.setToolTip("Include detected audio events, such as laughter, in the transcription.")
        self.no_verbatim.setToolTip("Ask the scribe_v2 model to remove fillers and false starts.")
        self.num_speakers.setToolTip(self.num_speakers.toolTip() + ". Used when speaker identification is enabled.")
        self.seed.setToolTip(self.seed.toolTip() + ". Optional provider seed for transcription.")
        self.temperature.setToolTip(self.temperature.toolTip() + ". Optional provider sampling setting.")
        self.retries.setToolTip(self.retries.toolTip() + ". Additional attempts may incur API charges.")
        self.backoff.setToolTip(self.backoff.toolTip() + ". Delay before retrying a transient provider error.")
        self.delay.setToolTip(self.delay.toolTip() + ". Minimum wait between requests for different files.")
        for control in (self.diarize, self.audio_events, verbatim_control):
            form.addRow(control)
        self.local_formats = {value: QCheckBox(FORMAT_DESCRIPTIONS[value][0]) for value in LOCAL_FORMATS}
        self.remote_formats = {value: QCheckBox(value.upper()) for value in REMOTE_FORMATS}
        for value, checkbox in self.local_formats.items():
            checkbox.setToolTip(FORMAT_DESCRIPTIONS[value][1] + " Rendered locally from the transcription response.")
            checkbox.toggled.connect(self.timestamps_error.clear)
            output_form.addRow("Local output", checkbox)
        for checkbox in self.remote_formats.values():
            checkbox.setToolTip("Requested from ElevenLabs during transcription; a new request may incur charges.")
            output_form.addRow("Provider output", checkbox)
        layout = self.layout()
        assert isinstance(layout, QFormLayout)
        layout.insertRow(0, group)

    def set_available_width(self, width: int) -> None:
        wide = width >= 850
        self.recognition_grid.addWidget(self.output_group, 0 if wide else 1, 1 if wide else 0)
        self.recognition_grid.addWidget(self.advanced_group, 1 if wide else 2, 0, 1, 2 if wide else 1)

    def validate_transcription_text(self) -> None:
        self.model_error.clear()
        self.keyterms_error.clear()
        self.verbatim_error.clear()
        try:
            TranscriptionOptions(model_id=self.model.text())
        except ValueError as exc:
            self.model_error.setText(str(exc))
        keyterms = tuple(line.strip() for line in self.keyterms.toPlainText().splitlines() if line.strip())
        try:
            TranscriptionOptions(keyterms=keyterms)
        except ValueError as exc:
            self.keyterms_error.setText(str(exc))
        if self.no_verbatim.isChecked() and self.model.text().strip() != "scribe_v2":
            self.verbatim_error.setText("Removing fillers is supported only by the scribe_v2 model.")

    def select_credentials_file(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Credentials File", self.env_file.text(), "All files (*)"
        )
        if filename:
            self.env_file.setText(filename)

    def preference_controls(self) -> dict[str, QWidget]:
        controls = super().preference_controls()
        controls.update(
            {
                name: getattr(self, name)
                for name in (
                    "model",
                    "language",
                    "timestamps",
                    "diarize",
                    "audio_events",
                    "no_verbatim",
                    "num_speakers",
                    "seed",
                    "temperature",
                    "keyterms",
                    "retries",
                    "backoff",
                    "delay",
                )
            }
        )
        controls.update({f"local.{key.value}": field for key, field in self.local_formats.items()})
        controls.update({f"remote.{key}": field for key, field in self.remote_formats.items()})
        return controls

    def settings(self, directory: Path) -> DesktopSettings:
        self.validate_transcription_text()
        for error, control in (
            (self.model_error, self.model),
            (self.keyterms_error, self.keyterms),
            (self.verbatim_error, self.no_verbatim),
        ):
            if error.text():
                control.setFocus()
                raise ValueError(error.text())
        env_file = self.env_file.text().strip()
        settings = replace(
            super().settings(directory),
            transcription=TranscriptionOptions(
                model_id=self.model.text(),
                language_code=self.language.text().strip() or None,
                timestamps_granularity=self.timestamps.currentText().lower(),
                diarize=self.diarize.isChecked(),
                tag_audio_events=self.audio_events.isChecked(),
                num_speakers=int(self.num_speakers.text()) if self.num_speakers.text().strip() else None,
                keyterms=tuple(line.strip() for line in self.keyterms.toPlainText().splitlines() if line.strip()),
                no_verbatim=self.no_verbatim.isChecked(),
                seed=int(self.seed.text()) if self.seed.text().strip() else None,
                temperature=float(self.temperature.text()) if self.temperature.text().strip() else None,
                remote_formats=tuple(value for value, control in self.remote_formats.items() if control.isChecked()),
            ),
            additional_formats=tuple(value for value, control in self.local_formats.items() if control.isChecked()),
            env_file=directory / Path(env_file).expanduser() if env_file else None,
            retries=int(self.retries.text()),
            retry_backoff=float(self.backoff.text()),
            request_delay=float(self.delay.text()),
        )
        timestamps = settings.transcription.timestamps_granularity
        timed_formats = {
            ArtifactFormat.SRT,
            ArtifactFormat.SRT_MINI,
            ArtifactFormat.RESOLVE_EDL,
            ArtifactFormat.CUE_INDEX_SRT,
        }
        message = ""
        if timestamps == "none" and any(value in timed_formats for value in settings.additional_formats):
            message = "Timed outputs require word or character timestamps."
        elif timestamps != "character" and settings.configuration()["segmentation"]["pause_detection"]:
            message = "Pause detection requires character timestamps."
        self.timestamps_error.setText(message)
        if message:
            self.timestamps.setFocus()
            raise ValueError(message)
        return settings
