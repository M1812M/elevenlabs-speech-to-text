import json
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elevenlabs_toolkit.core.marker_builder import cues_to_marker_edl, seconds_to_edl_timecode


class MarkerBuilderTests(unittest.TestCase):
    def test_seconds_to_edl_timecode_uses_one_hour_start(self) -> None:
        self.assertEqual(seconds_to_edl_timecode(0.0, 25), "01:00:00:00")
        self.assertEqual(seconds_to_edl_timecode(1.0, 25), "01:00:01:00")

    def test_cues_to_marker_edl_creates_structured_resolve_markers(self) -> None:
        edl = cues_to_marker_edl(
            [(0.0, 0.8, "hello"), (2.0, 2.8, "world")],
            title="sample",
            fps=25,
        )

        self.assertIn("TITLE: sample", edl)
        self.assertIn("FCM: NON-DROP FRAME", edl)
        self.assertIn("001  001      V     C        01:00:00:00 01:00:00:01 01:00:00:00 01:00:00:01", edl)
        self.assertIn("|C:ResolveColorBlue |M:Sentence 1 |D:1", edl)
        self.assertIn("|C:ResolveColorBlue |M:Sentence 2 |D:1", edl)

    def test_cli_create_marker_writes_edl(self) -> None:
        tmp_root = Path("_tmp_marker_test")
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_root / f"case_{uuid.uuid4().hex}"
        tmp_path.mkdir()

        try:
            source_json = tmp_path / "sample.json"
            marker_out_dir = tmp_path / "markers"

            payload = {
                "text": "hello world again later",
                "words": [
                    {"type": "word", "text": "hello", "start": 0.0, "end": 0.2, "speaker": "speaker_0"},
                    {"type": "word", "text": "world", "start": 0.25, "end": 0.5, "speaker": "speaker_0"},
                    {"type": "word", "text": "again", "start": 1.8, "end": 2.0, "speaker": "speaker_0"},
                    {"type": "word", "text": "later", "start": 2.05, "end": 2.3, "speaker": "speaker_0"},
                ],
            }
            source_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            cmd = [
                sys.executable,
                "scripts/transform.py",
                "--path",
                str(source_json),
                "--create-marker",
                "--marker-out-dir",
                str(marker_out_dir),
                "--marker-fps",
                "25",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")

            edl_path = marker_out_dir / "sample.edl"
            self.assertTrue(edl_path.exists())
            edl_text = edl_path.read_text(encoding="utf-8")
            self.assertIn("|M:Sentence 1 |D:1", edl_text)
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
