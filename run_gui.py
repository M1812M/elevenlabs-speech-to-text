"""Launch the optional desktop interface."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    source_directory = Path(__file__).resolve().parent / "src"
    sys.path.insert(0, str(source_directory))
    from elevenlabs_toolkit.gui import main as gui_main

    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
