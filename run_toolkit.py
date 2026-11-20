"""Run the ElevenLabs Toolkit directly from a checked-out project folder.

Use ``python .\\run_toolkit.py --help``. The launcher adds the local ``src``
directory to the import path, so it does not require an installed package or a
generated executable.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch to the local package without requiring installation."""
    script = Path(__file__).resolve()
    if argv is None:
        command = next((arg for arg in sys.argv[1:] if not arg.startswith("-")), None)
        if command in {"transcribe", "wizard"} and importlib.util.find_spec("elevenlabs") is None:
            executable = script.parent / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            if executable.is_file() and os.path.normcase(os.path.abspath(sys.executable)) != os.path.normcase(
                str(executable)
            ):
                try:
                    os.execv(str(executable), [str(executable), str(script), *sys.argv[1:]])
                except OSError as exc:
                    print(f"Could not start the project Python environment: {exc}", file=sys.stderr)
    source_directory = script.parent / "src"
    if str(source_directory) not in sys.path:
        sys.path.insert(0, str(source_directory))

    from elevenlabs_toolkit.cli.main import main as cli_main

    env_file = Path(".env")
    if not env_file.exists() and (script.parent / ".env").is_file():
        env_file = script.parent / ".env"
    return cli_main(argv, default_env_file=env_file)


if __name__ == "__main__":
    raise SystemExit(main())
