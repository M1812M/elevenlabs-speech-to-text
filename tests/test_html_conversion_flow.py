import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class HtmlConversionFlowTests(unittest.TestCase):
    def test_cli_conversion_preserves_html_tags_and_timecodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "sample_latin.srt"
            source.write_text(
                "1\n"
                "00:00:00,000 --> 00:00:01,000\n"
                "<i>Salom</i> do'st\n\n"
                "2\n"
                "00:00:01,500 --> 00:00:02,200\n"
                "<b>Skoriy</b> kelyapti\n",
                encoding="utf-8",
            )

            cmd = [
                sys.executable,
                "scripts/transform.py",
                "--path",
                str(source),
                "--convert-latin-srt-to-cyrillic",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")

            target = tmp_path / "sample_cyrillic.srt"
            self.assertTrue(target.exists())
            converted = target.read_text(encoding="utf-8")

            self.assertIn("00:00:00,000 --> 00:00:01,000", converted)
            self.assertIn("00:00:01,500 --> 00:00:02,200", converted)
            self.assertIn("<i>", converted)
            self.assertIn("</i>", converted)
            self.assertIn("<b>", converted)
            self.assertIn("</b>", converted)
            self.assertIn("<i>Салом</i>", converted)
            self.assertIn("<b>Скорий</b>", converted)


if __name__ == "__main__":
    unittest.main()
