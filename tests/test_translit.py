import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elevenlabs_toolkit.translit import latin_srt_to_cyrillic_text, to_latin


class TransliterationTests(unittest.TestCase):
    def test_russian_letters_to_latin(self) -> None:
        source = "Скорый келяпти, виу-виу деб нимаси уж эшитилганда."
        expected = "Skoriy kelyapti, viu-viu deb nimasi uj eshitilganda."
        self.assertEqual(to_latin(source), expected)

    def test_latin_srt_to_cyrillic_preserves_html_tags(self) -> None:
        source = "1\n00:00:00,000 --> 00:00:01,000\n<i>Salom</i> do'st\n"
        converted = latin_srt_to_cyrillic_text(source)
        self.assertIn("<i>", converted)
        self.assertIn("</i>", converted)
        self.assertIn("Салом", converted)


if __name__ == "__main__":
    unittest.main()
