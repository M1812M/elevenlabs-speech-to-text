import argparse
import re
from pathlib import Path

from elevenlabs_toolkit.cli.main import build_parser


def test_gui_control_map_mentions_every_long_cli_option() -> None:
    document = (Path(__file__).resolve().parents[3] / "docs" / "cli-gui-controls.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"(?<![\w-])--[\w-]+", document))
    parsers = [build_parser()]
    options = set()
    while parsers:
        parser = parsers.pop()
        for action in parser._actions:
            options.update(option for option in action.option_strings if option.startswith("--"))
            if isinstance(action, argparse._SubParsersAction):
                parsers.extend(action.choices.values())
    assert options <= documented, f"CLI options missing from the GUI control map: {sorted(options - documented)}"
