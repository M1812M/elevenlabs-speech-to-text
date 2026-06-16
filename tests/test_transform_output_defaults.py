import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TransformOutputDefaultTests(unittest.TestCase):
    def test_cli_defaults_transform_outputs_to_source_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_json = tmp_path / "sample.json"
            payload = {
                "text": "hello world again",
                "words": [
                    {"type": "word", "text": "hello", "start": 0.0, "end": 0.2, "speaker": "speaker_0"},
                    {"type": "word", "text": "world", "start": 0.25, "end": 0.5, "speaker": "speaker_0"},
                    {"type": "word", "text": "again", "start": 1.8, "end": 2.0, "speaker": "speaker_0"},
                ],
            }
            source_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            cmd = [
                sys.executable,
                "scripts/transform.py",
                "--path",
                str(source_json),
                "--create-srt",
                "--create-marker",
                "--create-txt",
                "--create-txt-combined",
                "--create-social-srt-latin",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")

            self.assertTrue((tmp_path / "sample.srt").exists())
            self.assertTrue((tmp_path / "sample.edl").exists())
            self.assertTrue((tmp_path / "sample.txt").exists())
            self.assertTrue((tmp_path / "sample_comb.txt").exists())
            self.assertTrue((tmp_path / "sample_social_latin.srt").exists())


if __name__ == "__main__":
    unittest.main()
