import tempfile
import unittest
from pathlib import Path

from elevenlabs_toolkit.selectors import collect_audio_files, collect_json_sources


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

    def test_collect_audio_files_supports_glob_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "a.flac").write_text("", encoding="utf-8")
            (tmp_path / "b.flac").write_text("", encoding="utf-8")
            (tmp_path / "c.mp3").write_text("", encoding="utf-8")

            selected = collect_audio_files(tmp_path / "*.flac")
            names = [path.name for path in selected]

            self.assertEqual(names, ["a.flac", "b.flac"])


if __name__ == "__main__":
    unittest.main()
