import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt

from elevenlabs_toolkit.application.desktop import DesktopSettings
from elevenlabs_toolkit.gui import Worker
from elevenlabs_toolkit.models import ArtifactFormat, ConflictPolicy, InputSpec, ScriptMode


def test_worker_reports_planned_outputs_and_per_file_failures(tmp_path: Path) -> None:
    good = tmp_path / "good.txt"
    bad = tmp_path / "bad.txt"
    good.write_text("Salom", encoding="utf-8")
    bad.write_bytes(b"\xff")
    worker = Worker("Convert Script", ScriptMode.CYRILLIC, [good, bad], False)
    plans = []
    results = []
    worker.plan_ready.connect(plans.append, Qt.ConnectionType.DirectConnection)
    worker.file_result.connect(results.append, Qt.ConnectionType.DirectConnection)
    worker.run()
    assert {row["source"] for row in plans[0]} == {str(good), str(bad)}
    assert next(row for row in plans[0] if row["source"] == str(good))["outputs"] == [str(tmp_path / "good CYR.txt")]
    assert next(row for row in results if row["source"] == str(good))["status"] != "failed"
    assert next(row for row in results if row["source"] == str(bad))["error"]


@pytest.mark.parametrize("existing", [False, True])
def test_json_content_preview_is_machine_readable_and_read_only(tmp_path: Path, existing: bool) -> None:
    source = tmp_path / "sample.srt"
    original = "1\n00:00:00,000 --> 00:00:01,000\nSalom\n"
    source.write_text(original, encoding="utf-8")
    target = tmp_path / "sample CYR.srt"
    if existing:
        target.write_text("existing", encoding="utf-8")
    worker = Worker(
        "Convert Script",
        ScriptMode.CYRILLIC,
        [source],
        False,
        policy=ConflictPolicy.SKIP,
        preview_only=True,
        content_preview=True,
        report_mode="JSON",
    )
    messages = []
    worker.finished_result.connect(messages.append, Qt.ConnectionType.DirectConnection)
    worker.run()
    preview = json.loads(messages[0])
    assert preview["status"] == "planned"
    assert preview["source"] == str(source)
    assert preview["target"] == str(target)
    assert "Салом" in preview["content"]
    assert preview["truncated"] is False
    assert ("skipped" in preview["warning"].lower()) is existing
    assert source.read_text(encoding="utf-8") == original
    assert target.exists() is existing
    if existing:
        assert target.read_text(encoding="utf-8") == "existing"


def test_conversion_content_preview_is_read_only(tmp_path: Path) -> None:
    source = tmp_path / "sample.srt"
    original = "1\n00:00:00,000 --> 00:00:01,000\nSalom\n"
    source.write_text(original, encoding="utf-8")
    worker = Worker("Convert Script", ScriptMode.CYRILLIC, [source], False, preview_only=True, content_preview=True)
    messages = []
    worker.finished_result.connect(messages.append, Qt.ConnectionType.DirectConnection)
    worker.run()
    assert "Салом" in messages[0]
    assert "00:00:00,000" in messages[0]
    assert source.read_text(encoding="utf-8") == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sample.srt"]


def test_json_content_preview_reports_truncation_and_conflict(tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("Salom " * 3000, encoding="utf-8")
    target = tmp_path / "sample CYR.txt"
    target.write_text("existing", encoding="utf-8")
    worker = Worker(
        "Convert Script",
        ScriptMode.CYRILLIC,
        [source],
        False,
        policy=ConflictPolicy.ERROR,
        preview_only=True,
        content_preview=True,
        report_mode="JSON",
    )
    messages = []
    worker.finished_result.connect(messages.append, Qt.ConnectionType.DirectConnection)
    worker.run()
    preview = json.loads(messages[0])
    assert preview["status"] == "conflict"
    assert preview["truncated"] is True
    assert len(preview["content"]) == 12000
    assert "conflict" in preview["warning"].lower()
    assert target.read_text(encoding="utf-8") == "existing"


@pytest.mark.parametrize("policy", [ConflictPolicy.SKIP, ConflictPolicy.REPLACE])
def test_content_preview_explains_existing_output_policy(tmp_path: Path, policy: ConflictPolicy) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("Salom", encoding="utf-8")
    output = tmp_path / "sample CYR.txt"
    output.write_text("existing output", encoding="utf-8")
    worker = Worker(
        "Convert Script", ScriptMode.CYRILLIC, [source], False, policy=policy, preview_only=True, content_preview=True
    )
    messages = []
    worker.finished_result.connect(messages.append, Qt.ConnectionType.DirectConnection)
    worker.run()
    assert "Салом" in messages[0]
    assert ("will be skipped" in messages[0].lower()) == (policy is ConflictPolicy.SKIP)
    assert ("will be replaced" in messages[0].lower()) == (policy is ConflictPolicy.REPLACE)
    assert output.read_text(encoding="utf-8") == "existing output"
    assert source.read_text(encoding="utf-8") == "Salom"


@pytest.mark.parametrize(
    ("artifact_format", "filename"),
    [(ArtifactFormat.SRT, "input.srt"), (ArtifactFormat.SRT_MINI, "input.mini.srt"), (ArtifactFormat.TXT, "input.txt")],
)
def test_export_content_preview_matches_written_output(
    tmp_path: Path, artifact_format: ArtifactFormat, filename: str
) -> None:
    source = tmp_path / "input.json"
    source.write_text(
        json.dumps({"language_code": "uz", "text": "Salom", "words": [{"text": "Salom", "start": 0, "end": 1}]}),
        encoding="utf-8",
    )
    original = source.read_bytes()
    settings = DesktopSettings(project_directory=tmp_path)
    preview = Worker(
        "Export",
        ScriptMode.SOURCE,
        [source],
        False,
        (artifact_format,),
        settings,
        preview_only=True,
        content_preview=True,
    )
    messages = []
    preview.finished_result.connect(messages.append, Qt.ConnectionType.DirectConnection)
    preview.run()
    assert not (tmp_path / filename).exists()
    execute = Worker("Export", ScriptMode.SOURCE, [source], False, (artifact_format,), settings)
    execute.run()
    assert messages[0].split("\n\n", 1)[1] == (tmp_path / filename).read_text(encoding="utf-8")
    assert source.read_bytes() == original


@pytest.mark.parametrize("artifact_format", [ArtifactFormat.SRT, ArtifactFormat.RESOLVE_EDL])
def test_json_export_content_preview_handles_supported_and_unavailable_formats(
    tmp_path: Path, artifact_format: ArtifactFormat
) -> None:
    source = tmp_path / "input.json"
    source.write_text(
        json.dumps({"language_code": "uz", "text": "Salom", "words": [{"text": "Salom", "start": 0, "end": 1}]}),
        encoding="utf-8",
    )
    original = source.read_bytes()
    worker = Worker(
        "Export",
        ScriptMode.SOURCE,
        [source],
        False,
        (artifact_format,),
        DesktopSettings(project_directory=tmp_path),
        preview_only=True,
        content_preview=True,
        report_mode="JSON",
    )
    messages = []
    worker.finished_result.connect(messages.append, Qt.ConnectionType.DirectConnection)
    worker.run()
    preview = json.loads(messages[0])
    if artifact_format is ArtifactFormat.SRT:
        assert preview["status"] == "planned"
        assert "Salom" in preview["content"]
    else:
        assert preview["status"] == "unavailable"
        assert "preview" in preview["message"].lower()
    assert source.read_bytes() == original
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("report_mode", ["Standard", "JSON"])
def test_unavailable_content_preview_still_reports_output_conflict(tmp_path: Path, report_mode: str) -> None:
    source = tmp_path / "input.json"
    source.write_text(json.dumps({"text": "Salom"}), encoding="utf-8")
    target = tmp_path / "input.resolve.edl"
    target.write_text("existing", encoding="utf-8")
    worker = Worker(
        "Export",
        ScriptMode.SOURCE,
        [source],
        False,
        (ArtifactFormat.RESOLVE_EDL,),
        DesktopSettings(project_directory=tmp_path),
        policy=ConflictPolicy.ERROR,
        preview_only=True,
        content_preview=True,
        report_mode=report_mode,
    )
    messages = []
    worker.finished_result.connect(messages.append, Qt.ConnectionType.DirectConnection)
    worker.run()
    assert "conflict" in messages[0].lower()
    assert "preview" in messages[0].lower()
    assert str(target) in messages[0]
    if report_mode == "JSON":
        preview = json.loads(messages[0])
        assert preview["status"] == "conflict"
        assert preview["conflicts"][0]["target"] == str(target)
    assert target.read_text(encoding="utf-8") == "existing"


def test_folder_conversion_includes_legacy_script_suffix(tmp_path: Path) -> None:
    source = tmp_path / "sample.latin.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nSalom\n", encoding="utf-8")
    worker = Worker("Convert Script", ScriptMode.CYRILLIC, [], False, selection=InputSpec((tmp_path,)))
    messages = []
    worker.finished_result.connect(messages.append, Qt.ConnectionType.DirectConnection)
    worker.run()
    assert (tmp_path / "sample CYR.srt").exists(), messages


def test_output_subfolder_is_not_rescanned_in_recursive_conversion(tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("Salom", encoding="utf-8")
    output = tmp_path / "Output"
    output.mkdir()
    (output / "sample CYR.txt").write_text("existing", encoding="utf-8")
    worker = Worker(
        "Convert Script",
        ScriptMode.LATIN,
        [],
        False,
        settings=DesktopSettings(output_subfolder=True),
        selection=InputSpec((tmp_path,), recursive=True),
        preview_only=True,
    )
    messages = []
    worker.finished_result.connect(messages.append, Qt.ConnectionType.DirectConnection)
    worker.run()
    assert str(output / "Output") not in messages[0]
    assert "sample LTN.txt" in messages[0]


def test_explicitly_selected_output_folder_and_file_are_not_excluded(tmp_path: Path) -> None:
    folder = tmp_path / "Output"
    folder.mkdir()
    source = folder / "input.txt"
    source.write_text("Salom", encoding="utf-8")
    for selected in (folder, source):
        worker = Worker(
            "Convert Script",
            ScriptMode.CYRILLIC,
            [],
            False,
            settings=DesktopSettings(output_subfolder=True),
            selection=InputSpec((selected,), recursive=True),
            preview_only=True,
        )
        messages = []
        worker.finished_result.connect(messages.append, Qt.ConnectionType.DirectConnection)
        worker.run()
        assert str(folder / "Output" / "input CYR.txt") in messages[0]
        assert not (folder / "Output").exists()


def test_conversion_continues_after_bad_input_and_preserves_source(tmp_path: Path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xff")
    good = tmp_path / "good.txt"
    good.write_text("Salom", encoding="utf-8")
    worker = Worker("Convert Script", ScriptMode.CYRILLIC, [bad, good], False)
    messages = []
    worker.finished_result.connect(messages.append, Qt.ConnectionType.DirectConnection)
    worker.run()
    assert good.read_text(encoding="utf-8") == "Salom"
    assert (tmp_path / "good CYR.txt").read_text(encoding="utf-8") == "Салом"
    assert "FAILED" in messages[0]


def test_skipped_transcription_needs_no_provider(tmp_path: Path, monkeypatch) -> None:
    import json

    import elevenlabs_toolkit.providers as providers

    def forbidden(*args, **kwargs):
        pytest.fail("Provider must not be constructed for a skipped job")

    monkeypatch.setattr(providers, "ElevenLabsProvider", forbidden)
    source = tmp_path / "audio.flac"
    source.write_bytes(b"placeholder")
    target = tmp_path / "audio.flac.transcript.json"
    target.write_text("{}")
    worker = Worker("Transcribe", ScriptMode.SOURCE, [source], False, report_mode="JSON")
    messages = []
    worker.finished_result.connect(messages.append, Qt.ConnectionType.DirectConnection)
    worker.run()
    assert "skipped" in messages[0]
    assert "FAILED" not in messages[0]
    assert json.loads(messages[0])["results"][0]["status"] == "skipped"


def test_stop_between_files(tmp_path: Path) -> None:
    sources = [tmp_path / f"{number}.txt" for number in range(3)]
    for source in sources:
        source.write_text("Salom")
    worker = Worker("Convert Script", ScriptMode.CYRILLIC, sources, False)
    stopped = []
    worker.batch_stopped.connect(stopped.append, Qt.ConnectionType.DirectConnection)
    worker.progress_changed.connect(
        lambda completed, total: worker.requestInterruption(),
        Qt.ConnectionType.DirectConnection,
    )
    worker.start()
    assert worker.wait(5000)
    assert (tmp_path / "0 CYR.txt").exists()
    assert not (tmp_path / "1 CYR.txt").exists()
    assert stopped == ["Stopped"]


def test_cancelled_combined_export_is_reported_and_can_be_retried(tmp_path: Path) -> None:
    import json

    sources = [tmp_path / f"{number}.json" for number in range(2)]
    for source in sources:
        source.write_text(
            json.dumps({"text": "Salom", "words": [{"type": "word", "text": "Salom", "start": 0, "end": 1}]}),
            encoding="utf-8",
        )
    worker = Worker("Export", ScriptMode.SOURCE, sources, False, (ArtifactFormat.COMBINED_TXT,))
    results, retry_items = [], []
    worker.file_result.connect(results.append, Qt.ConnectionType.DirectConnection)
    worker.failed_item.connect(retry_items.append, Qt.ConnectionType.DirectConnection)
    worker.status.connect(
        lambda text: worker.requestInterruption() if text.startswith("Combined text: processed") else None,
        Qt.ConnectionType.DirectConnection,
    )
    worker.start()
    assert worker.wait(5000)
    assert not (tmp_path / "combined.txt").exists()
    assert {item["status"] for item in results} == {"cancelled"}
    assert len(retry_items) == 1


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
