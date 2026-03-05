import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elevenlabs_toolkit.transcript_utils import payload_to_sentence_items


class TimingSentenceSplitTests(unittest.TestCase):
    def test_timing_split_breaks_on_pause_and_marker(self) -> None:
        payload = {
            "words": [
                {"type": "word", "text": "bugun", "start": 0.0, "end": 0.2, "speaker": "speaker_0"},
                {"type": "word", "text": "kelganman", "start": 0.25, "end": 0.5, "speaker": "speaker_0"},
                {"type": "word", "text": "gaplashamiz", "start": 0.55, "end": 0.8, "speaker": "speaker_0"},
                {"type": "word", "text": "keyin", "start": 2.0, "end": 2.2, "speaker": "speaker_0"},
                {"type": "word", "text": "yana", "start": 2.25, "end": 2.45, "speaker": "speaker_0"},
                {"type": "word", "text": "davom", "start": 2.5, "end": 2.7, "speaker": "speaker_0"},
            ]
        }

        no_timing = payload_to_sentence_items(payload)
        with_timing = payload_to_sentence_items(
            payload,
            use_timing_split=True,
            gap_split_seconds=0.9,
            hard_gap_split_seconds=1.8,
        )

        self.assertEqual(len(no_timing), 1)
        self.assertEqual(len(with_timing), 2)
        self.assertTrue((with_timing[1]["text"] or "").lower().startswith("keyin"))


if __name__ == "__main__":
    unittest.main()

