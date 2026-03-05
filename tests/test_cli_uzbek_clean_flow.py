import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class UzbekCleanFlowTests(unittest.TestCase):
    def test_cli_creates_clean_json_and_readable_txt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_json = tmp_path / "sample.json"
            txt_out_dir = tmp_path / "txt"
            txt_out_dir.mkdir(parents=True, exist_ok=True)

            payload = {
                "text": "man manga boraman keyin iso masih haqida gapiraman",
                "words": [
                    {"type": "word", "text": "man", "start": 0.0, "end": 0.2, "speaker": "speaker_0"},
                    {"type": "word", "text": "manga", "start": 0.25, "end": 0.5, "speaker": "speaker_0"},
                    {"type": "word", "text": "boraman", "start": 0.55, "end": 0.9, "speaker": "speaker_0"},
                    {"type": "word", "text": "keyin", "start": 2.0, "end": 2.2, "speaker": "speaker_0"},
                    {"type": "word", "text": "iso", "start": 2.25, "end": 2.45, "speaker": "speaker_0"},
                    {"type": "word", "text": "masih", "start": 2.5, "end": 2.8, "speaker": "speaker_0"},
                    {"type": "word", "text": "haqida", "start": 2.85, "end": 3.05, "speaker": "speaker_0"},
                    {"type": "word", "text": "gapiraman", "start": 3.1, "end": 3.4, "speaker": "speaker_0"},
                ],
            }
            source_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            cmd = [
                sys.executable,
                "scripts/transform.py",
                "--path",
                str(source_json),
                "--create-clean-json",
                "--create-txt",
                "--txt-out-dir",
                str(txt_out_dir),
                "--uzbek-clean",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")

            clean_json = tmp_path / "sample_uz_clean.json"
            self.assertTrue(clean_json.exists())
            clean_payload = json.loads(clean_json.read_text(encoding="utf-8"))
            self.assertIn("Men", clean_payload.get("text", ""))
            self.assertIn("Iso Masih", clean_payload.get("text", ""))

            # Original source JSON stays untouched.
            original_payload = json.loads(source_json.read_text(encoding="utf-8"))
            self.assertEqual(original_payload["text"], payload["text"])

            txt_path = txt_out_dir / "sample.txt"
            self.assertTrue(txt_path.exists())
            txt_lines = [line for line in txt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertGreaterEqual(len(txt_lines), 2)
            self.assertTrue(any("Keyin" in line for line in txt_lines))


if __name__ == "__main__":
    unittest.main()

