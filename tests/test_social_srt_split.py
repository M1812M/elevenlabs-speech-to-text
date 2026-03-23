import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elevenlabs_toolkit.core.srt_builder import tokens_to_social_cues


class SocialSrtSplitTests(unittest.TestCase):
    def test_keeps_json_word_timing_without_trimming(self) -> None:
        tokens = [
            {"text": "ibodat", "start": 410.88, "end": 411.38},
            {"text": "qiladigan,", "start": 411.42, "end": 413.00},
            {"text": "kalom", "start": 413.04, "end": 413.47},
            {"text": "o'qiydikan,", "start": 413.56, "end": 414.16},
        ]

        cues = tokens_to_social_cues(tokens)

        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0][2], "ibodat qiladigan,")
        self.assertEqual(cues[0][1], 413.00)
        self.assertEqual(cues[1][0], 413.04)

    def test_splits_on_json_boundaries_not_estimated_pauses(self) -> None:
        tokens = [
            {"text": "manga", "start": 459.20, "end": 459.48},
            {"text": "tegingan", "start": 459.58, "end": 460.18},
            {"text": "narsa", "start": 460.19, "end": 461.80},
            {"text": "uning", "start": 461.86, "end": 462.19},
            {"text": "ovozi", "start": 462.28, "end": 462.82},
            {"text": "na", "start": 462.86, "end": 463.12},
            {"text": "erkakni,", "start": 463.18, "end": 463.72},
        ]

        cues = tokens_to_social_cues(tokens)

        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0], (459.20, 461.80, "manga tegingan narsa"))
        self.assertEqual(cues[1], (461.86, 463.72, "uning ovozi na erkakni,"))


if __name__ == "__main__":
    unittest.main()
