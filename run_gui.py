"""Launch the optional desktop interface."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def main() -> int:
    project_directory = Path(__file__).resolve().parent
    if importlib.util.find_spec("PySide6") is None:
        executable = project_directory / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        if executable.is_file() and os.path.abspath(sys.executable) != str(executable):
            try:
                os.execv(str(executable), [str(executable), str(Path(__file__).resolve()), *sys.argv[1:]])
            except OSError as exc:
                print(f"Could not start the project Python environment: {exc}", file=sys.stderr)
        print(
            "PySide6 is missing. From the project folder, run:\n"
            "  uv sync --locked --extra gui --extra stt\n"
            "Then start the app again with: python run_gui.py",
            file=sys.stderr,
        )
        return 1
    source_directory = project_directory / "src"
    sys.path.insert(0, str(source_directory))
    from elevenlabs_toolkit.gui import main as gui_main

    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
