import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

LAUNCHER = Path(__file__).resolve().parents[2] / "run_gui.py"


def test_launcher_self_test_exercises_window_and_offline_conversion() -> None:
    pytest.importorskip("PySide6")
    environment = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    result = subprocess.run(
        [sys.executable, str(LAUNCHER), "--self-test"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Desktop self-test passed" in result.stdout


def test_launcher_uses_project_environment_when_pyside_is_missing(tmp_path: Path, monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location("test_run_gui", LAUNCHER)
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    monkeypatch.setattr(launcher, "__file__", str(tmp_path / "run_gui.py"))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setitem(sys.modules, "elevenlabs_toolkit.gui", SimpleNamespace(main=lambda: 0))
    venv_python = tmp_path / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    venv_python.parent.mkdir(parents=True)
    venv_python.touch()
    calls = []

    def capture_exec(path, args):
        calls.append((path, args))
        raise RuntimeError("exec called")

    monkeypatch.setattr("os.execv", capture_exec)
    with pytest.raises(RuntimeError, match="exec called"):
        launcher.main()
    assert calls[0][0] == str(venv_python)
    assert calls[0][1][1] == str(tmp_path / "run_gui.py")


def test_launcher_explains_missing_gui_dependencies(tmp_path: Path) -> None:
    launcher = tmp_path / "run_gui.py"
    launcher.write_text(LAUNCHER.read_text(encoding="utf-8"), encoding="utf-8")
    result = subprocess.run([sys.executable, "-S", str(launcher)], text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert "uv sync --locked --extra gui --extra stt" in result.stderr
    assert "Traceback" not in result.stderr
