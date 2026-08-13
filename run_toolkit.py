"""Run the ElevenLabs Toolkit directly from a checked-out project folder.

Use ``python .\\run_toolkit.py --help``. The launcher adds the local ``src``
directory to the import path, so it does not require an installed package or a
generated executable.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch to the local package without requiring installation."""
    source_directory = Path(__file__).resolve().parent / "src"
    if str(source_directory) not in sys.path:
        sys.path.insert(0, str(source_directory))

    from elevenlabs_toolkit.cli.main import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
