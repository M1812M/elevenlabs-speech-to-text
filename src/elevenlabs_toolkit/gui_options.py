from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QGroupBox, QLineEdit, QPlainTextEdit, QWidget

from .application.desktop import DesktopSettings
from .cli.transcribe import LOCAL_FORMATS, REMOTE_FORMATS
from .config import available_profiles
from .models import ScriptMode, TranscriptionOptions


class TextOptionsPanel(QWidget):
    """Reusable profile and text controls for offline transcript workflows."""

    def __init__(self, *, clean: bool = False, advanced: bool = False) -> None:
        super().__init__()
        self.clean = clean
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
        if not clean:
            layout.addRow("Cleanup", self.cleanup)
        self.override_replacements = QCheckBox("Override profile replacements")
        layout.addRow(self.override_replacements)
        self.replacements = QPlainTextEdit()
        self.replacements.setPlaceholderText("One replacement per line, for example Acme=ACME")
        self.replacements.setMaximumHeight(65)
        self.replacements.setEnabled(False)
        self.override_replacements.toggled.connect(self.replacements.setEnabled)
        layout.addRow("Token replacements", self.replacements)
        self.advanced = advanced
        self.numeric_fields = {}
        self.boolean_fields = {}
        self.smart = QCheckBox("Balance subtitle lines")
        self.marker_fps = QLineEdit("25")
        self.marker_color = QLineEdit("ResolveColorBlue")
        self.marker_prefix = QLineEdit("Sentence")
        self.combined_name = QLineEdit("combined.txt")
        if advanced:
            group = QGroupBox("Advanced subtitle and Resolve settings")
            group.setCheckable(True)
            group.setChecked(False)
            advanced_layout = QFormLayout(group)
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
                advanced_layout.addRow(label, field)
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
            advanced_layout.addRow("Resolve marker FPS", self.marker_fps)
            advanced_layout.addRow("Resolve marker colour", self.marker_color)
            advanced_layout.addRow("Resolve marker prefix", self.marker_prefix)
            advanced_layout.addRow("Combined text filename", self.combined_name)
            layout.addRow(group)

    def refresh_profiles(self, directory: Path) -> None:
        selected = self.profile.currentText()
        profiles = available_profiles(cwd=directory)
        self.profile.clear()
        self.profile.addItem("Project default", None)
        for profile in profiles:
            self.profile.addItem(profile, profile)
        if selected != "Project default":
            self.profile.setCurrentText(selected)

    def settings(self, directory: Path) -> DesktopSettings:
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
        return DesktopSettings(
            segmentation=tuple(segmentation),
            smart_line_breaks=self.smart.isChecked(),
            marker_fps=float(self.marker_fps.text()),
            marker_color=self.marker_color.text(),
            marker_prefix=self.marker_prefix.text(),
            combined_name=self.combined_name.text(),
            project_directory=directory,
            profile=None if profile == "Project default" else profile,
            output_script=self.script.currentData(),
            replacements=replacements,
            cleanup="uzbek" if self.clean else self.cleanup.currentData(),
        )


class TranscriptionPanel(TextOptionsPanel):
    def __init__(self) -> None:
        super().__init__(advanced=True)
        group = QGroupBox("Recognition and additional outputs")
        form = QFormLayout(group)
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
        self.env_file = QLineEdit()
        self.env_file.setPlaceholderText("Project .env")
        self.retries = QLineEdit("0")
        self.backoff = QLineEdit("1")
        self.delay = QLineEdit("0")
        for label, control in (
            ("Model", self.model),
            ("Language code", self.language),
            ("Timestamps", self.timestamps),
            ("Speaker count", self.num_speakers),
            ("Seed", self.seed),
            ("Temperature", self.temperature),
            ("Keyterms", self.keyterms),
            ("Credentials file", self.env_file),
            ("Retries", self.retries),
            ("Retry backoff (seconds)", self.backoff),
            ("Request delay (seconds)", self.delay),
        ):
            form.addRow(label, control)
        for checkbox in (self.diarize, self.audio_events, self.no_verbatim):
            form.addRow(checkbox)
        self.local_formats = {value: QCheckBox(value.value.upper()) for value in LOCAL_FORMATS}
        self.remote_formats = {value: QCheckBox(value.upper()) for value in REMOTE_FORMATS}
        for checkbox in self.local_formats.values():
            form.addRow("Local output", checkbox)
        for checkbox in self.remote_formats.values():
            form.addRow("Provider output", checkbox)
        layout = self.layout()
        assert isinstance(layout, QFormLayout)
        layout.insertRow(0, group)

    def settings(self, directory: Path) -> DesktopSettings:
        env_file = self.env_file.text().strip()
        return replace(
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
