"""Build the desktop application with PyInstaller from an isolated environment."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    build = root / "build" / "desktop"
    build.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--windowed",
        "--name",
        "ElevenLabs Toolkit",
        "--paths",
        str(root / "src"),
        "--collect-all",
        "elevenlabs",
        "--copy-metadata",
        "elevenlabs-toolkit",
        "--distpath",
        str(root / "dist"),
        "--workpath",
        str(build / "work"),
        "--specpath",
        str(build),
        str(root / "run_gui.py"),
    ]
    return subprocess.call(command, cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
