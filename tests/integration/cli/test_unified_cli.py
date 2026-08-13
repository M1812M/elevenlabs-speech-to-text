import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    return subprocess.run(
        [sys.executable, "-m", "elevenlabs_toolkit.cli.main", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "run_toolkit.py", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_direct_python_script_shows_help_without_an_installed_package() -> None:
    result = _run_script("--help")

    assert result.returncode == 0, result.stderr
    assert "Transcribe media and produce safe, reproducible post-production artifacts." in result.stdout


def _transcript(path: Path) -> Path:
    payload = {
        "text": "Салом дунё.",
        "words": [
            {"type": "word", "text": "Салом", "start": 0, "end": 0.3},
            {"type": "word", "text": "дунё.", "start": 0.4, "end": 0.8},
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_export_preserves_source_script_by_default(tmp_path: Path) -> None:
    source = _transcript(tmp_path / "sample.json")
    output = tmp_path / "out"

    result = _run("export", str(source), "-o", str(output), "--format", "srt")

    assert result.returncode == 0, result.stderr
    assert "Салом дунё." in (output / "sample.srt").read_text(encoding="utf-8")


def test_json_dry_run_is_one_machine_readable_document_and_writes_nothing(tmp_path: Path) -> None:
    source = _transcript(tmp_path / "sample.json")
    output = tmp_path / "out"

    result = _run("--json", "export", str(source), "-o", str(output), "--dry-run")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "planned"
    assert payload["dry_run"] is True
    assert not output.exists()


def test_nonexistent_path_is_not_treated_as_regex(tmp_path: Path) -> None:
    result = _run("export", str(tmp_path / ".*[.]json"), "--format", "srt")

    assert result.returncode == 2
    assert "does not exist" in result.stderr


def test_transcribe_dry_run_needs_no_sdk_or_api_key(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"not-real-audio")
    output = tmp_path / "build"

    result = _run("--json", "transcribe", str(source), "-o", str(output), "--dry-run")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["api_requests"] == 1
    assert payload["max_api_attempts"] == 1
    assert payload["provider"] == "elevenlabs"
    assert not output.exists()


def test_pause_detection_dependency_is_validated_before_api_work(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"not-real-audio")

    result = _run("transcribe", str(source), "--pause-detection", "--timestamps", "word", "--dry-run")

    assert result.returncode == 2
    assert "requires --timestamps character" in result.stderr


def test_inspect_reports_valid_transcript(tmp_path: Path) -> None:
    source = _transcript(tmp_path / "sample.json")

    result = _run("--json", "inspect", str(source))

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["transcripts"][0]["valid"] is True
    assert payload["transcripts"][0]["words"] == 2


def test_transliterate_preserves_srt_structure_and_html_tags(tmp_path: Path) -> None:
    source = tmp_path / "sample.latin.srt"
    source.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n<i>Salom</i> do'st\n",
        encoding="utf-8",
    )
    output = tmp_path / "out"

    result = _run("transliterate", str(source), "--to", "cyrillic", "-o", str(output))

    assert result.returncode == 0, result.stderr
    converted = (output / "sample.cyrillic.srt").read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:01,000" in converted
    assert "<i>Салом</i>" in converted
    assert "</i>" in converted


def test_transliterate_preserves_entities_and_timing_settings(tmp_path: Path) -> None:
    source = tmp_path / "sample.srt"
    timing = "00:00:00.000 --> 00:00:01.000 position:50% align:start"
    source.write_text(f"1\n{timing}\nSalom &amp; dunyo\n", encoding="utf-8")
    output = tmp_path / "out"

    result = _run("transliterate", str(source), "--to", "cyrillic", "-o", str(output))

    assert result.returncode == 0, result.stderr
    converted = (output / "sample.cyrillic.srt").read_text(encoding="utf-8")
    assert timing in converted
    assert "&amp;" in converted


def test_clean_command_writes_a_derivative_and_preserves_source(tmp_path: Path) -> None:
    source = tmp_path / "sample.json"
    original = {
        "text": "man manga boraman",
        "words": [
            {"type": "word", "text": "man", "start": 0.0, "end": 0.2},
            {"type": "word", "text": "manga", "start": 0.25, "end": 0.5},
        ],
    }
    source.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "out"

    result = _run("clean", str(source), "-o", str(output))

    assert result.returncode == 0, result.stderr
    cleaned = json.loads((output / "sample.clean.json").read_text(encoding="utf-8"))
    assert cleaned["text"] == "Men menga boraman"
    assert json.loads(source.read_text(encoding="utf-8")) == original


def test_pause_enabled_profile_automatically_requests_character_timestamps(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"not-real-audio")

    automatic = _run("transcribe", str(source), "--profile", "social", "--dry-run")
    incompatible = _run(
        "transcribe",
        str(source),
        "--profile",
        "social",
        "--timestamps",
        "word",
        "--dry-run",
    )

    assert automatic.returncode == 0, automatic.stderr
    assert incompatible.returncode == 2
    assert "requires --timestamps character" in incompatible.stderr


def test_dry_run_discloses_explicit_retry_attempt_ceiling(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"not-real-audio")

    result = _run("--json", "transcribe", str(source), "--retries", "2", "--dry-run")

    assert result.returncode == 0
    assert json.loads(result.stdout)["max_api_attempts"] == 3


def test_transcribe_rejects_non_finite_pacing_before_dry_run(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"not-real-audio")

    for option, value in (
        ("--retry-backoff", "inf"),
        ("--request-delay", "nan"),
        ("--lock-timeout", "inf"),
    ):
        result = _run("transcribe", str(source), option, value, "--dry-run")
        assert result.returncode == 2
        assert "finite number" in result.stderr


def test_transcribe_rejects_unsafe_source_stem_in_preflight(tmp_path: Path) -> None:
    source = tmp_path / "trailing..mp3"
    source.write_bytes(b"not-real-audio")
    output = tmp_path / "out"

    result = _run("transcribe", str(source), "-o", str(output), "--dry-run")

    assert result.returncode == 2
    assert "portable output name" in result.stderr
    assert not output.exists()


def test_transcription_cli_does_not_offer_unstable_cache_rename(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"not-real-audio")

    result = _run("transcribe", str(source), "--on-conflict", "rename", "--dry-run")

    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_clean_json_rejects_explicit_cleanup_opt_out(tmp_path: Path) -> None:
    source = _transcript(tmp_path / "sample.json")

    result = _run("export", str(source), "--format", "clean-json", "--clean", "none", "--dry-run")

    assert result.returncode == 2
    assert "cannot be combined" in result.stderr


def test_quiet_and_verbose_are_mutually_exclusive() -> None:
    result = _run("--quiet", "--verbose", "config", "show")

    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr
