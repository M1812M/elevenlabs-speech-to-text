import importlib.util
import sys
from pathlib import Path

import pytest


def test_transcription_uses_project_python_when_sdk_is_missing(tmp_path: Path, monkeypatch) -> None:
    script = Path(__file__).resolve().parents[2] / "run_toolkit.py"
    spec = importlib.util.spec_from_file_location("test_run_toolkit", script)
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    monkeypatch.setattr(launcher, "__file__", str(tmp_path / script.name))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    args = ["--json", "transcribe", "audio file.flac"]
    monkeypatch.setattr(sys, "argv", [str(script), *args])
    executable = tmp_path / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    executable.parent.mkdir(parents=True)
    executable.touch()

    def capture_exec(path, argv):
        assert path == str(executable)
        assert argv == [str(executable), str(tmp_path / script.name), *args]
        raise SystemExit(42)

    monkeypatch.setattr("os.execv", capture_exec)
    with pytest.raises(SystemExit, match="42"):
        launcher.main()


@pytest.mark.parametrize("location", ["checkout", "working directory", "explicit"])
def test_transcription_finds_credentials_from_another_directory(tmp_path: Path, monkeypatch, location: str) -> None:
    from elevenlabs_toolkit.cli import transcribe

    script = Path(__file__).resolve().parents[2] / "run_toolkit.py"
    spec = importlib.util.spec_from_file_location("test_run_toolkit", script)
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(launcher, "__file__", str(checkout / script.name))
    monkeypatch.chdir(tmp_path)
    (checkout / ".env").touch()
    source = tmp_path / "audio.flac"
    source.touch()
    args = ["transcribe", str(source)]
    expected = checkout / ".env"
    if location == "working directory":
        (tmp_path / ".env").touch()
        expected = tmp_path / ".env"
    elif location == "explicit":
        expected = tmp_path / "custom.env"
        args.extend(["--env-file", expected.name])

    paths = []

    def capture_provider(*, env_file):
        paths.append(env_file.resolve())
        raise SystemExit(42)

    monkeypatch.setattr(transcribe, "ElevenLabsProvider", capture_provider)
    with pytest.raises(SystemExit, match="42"):
        launcher.main(args)
    assert paths == [expected.resolve()]
