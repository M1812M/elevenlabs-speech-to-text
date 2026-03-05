import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elevenlabs_toolkit.uzbek_cleanup import clean_uzbek_payload, clean_uzbek_text


class UzbekCleanupTests(unittest.TestCase):
    def test_text_cleanup_normalizes_common_forms(self) -> None:
        source = "man manga misofir bo'lib keldim, iso masih va xudo o'zbekiston haqida gapirdi"
        cleaned = clean_uzbek_text(source)

        self.assertIn("Men", cleaned)
        self.assertIn("menga", cleaned)
        self.assertIn("musofir", cleaned)
        self.assertIn("Iso Masih", cleaned)
        self.assertIn("Xudo", cleaned)
        self.assertIn("O‘zbekiston", cleaned)

    def test_payload_cleanup_keeps_timing_intact(self) -> None:
        payload = {
            "text": "man manga boraman",
            "words": [
                {"type": "word", "text": "man", "start": 0.0, "end": 0.2},
                {"type": "word", "text": "manga", "start": 0.25, "end": 0.5},
            ],
        }
        cleaned = clean_uzbek_payload(payload)

        # Original remains untouched.
        self.assertEqual(payload["words"][0]["text"], "man")
        self.assertEqual(payload["words"][1]["text"], "manga")

        # Cleaned copy is normalized.
        self.assertEqual(cleaned["words"][0]["text"], "men")
        self.assertEqual(cleaned["words"][1]["text"], "menga")
        self.assertEqual(cleaned["words"][0]["start"], 0.0)
        self.assertEqual(cleaned["words"][1]["end"], 0.5)


if __name__ == "__main__":
    unittest.main()

