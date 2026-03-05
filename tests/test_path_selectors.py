import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elevenlabs_toolkit.selectors import collect_json_sources


class PathSelectorTests(unittest.TestCase):
    def test_collect_json_sources_supports_regex_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "2025-06 Gulchihra 1 Shock.json").write_text("{}", encoding="utf-8")
            (tmp_path / "2025-06 Gulchihra 2 Surgery.json").write_text("{}", encoding="utf-8")
            (tmp_path / "2025-06 Gulchihra 3 Bible.json").write_text("{}", encoding="utf-8")

            pattern_path = tmp_path / r"^2025-06 Gulchihra [12] .*[.]json$"
            selected = collect_json_sources(pattern_path)
            names = [path.name for path in selected]

            self.assertEqual(
                names,
                [
                    "2025-06 Gulchihra 1 Shock.json",
                    "2025-06 Gulchihra 2 Surgery.json",
                ],
            )


if __name__ == "__main__":
    unittest.main()
