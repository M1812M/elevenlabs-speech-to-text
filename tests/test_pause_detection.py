import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elevenlabs_toolkit.core.srt_builder import (
    build_social_word_tokens,
    build_standard_tokens,
    tokens_to_social_cues,
    tokens_to_standard_cues,
)
from elevenlabs_toolkit.pause_detection import detect_stretched_character_pause_end, effective_word_end
from elevenlabs_toolkit.transcript_utils import payload_to_sentence_items


class PauseDetectionTests(unittest.TestCase):
    def test_detects_stretched_final_character(self) -> None:
        word = {
            "type": "word",
            "text": "narsa",
            "start": 460.19,
            "end": 461.80,
            "characters": [
                {"text": "n", "start": 460.19, "end": 460.32},
                {"text": "a", "start": 460.32, "end": 460.34},
                {"text": "r", "start": 460.34, "end": 460.499},
                {"text": "s", "start": 460.50, "end": 460.54},
                {"text": "a", "start": 460.54, "end": 461.80},
            ],
        }

        detected = detect_stretched_character_pause_end(word)

        self.assertIsNotNone(detected)
        self.assertLess(detected or 0.0, 461.0)
        self.assertGreater(detected or 0.0, 460.65)

    def test_leaves_normal_word_end_unchanged(self) -> None:
        word = {
            "type": "word",
            "text": "ovozi",
            "start": 462.28,
            "end": 462.82,
            "characters": [
                {"text": "o", "start": 462.28, "end": 462.36},
                {"text": "v", "start": 462.36, "end": 462.38},
                {"text": "o", "start": 462.38, "end": 462.60},
                {"text": "z", "start": 462.60, "end": 462.61},
                {"text": "i", "start": 462.61, "end": 462.82},
            ],
        }

        self.assertIsNone(detect_stretched_character_pause_end(word))
        self.assertEqual(effective_word_end(word, pause_detection=True), 462.82)

    def test_standard_tokens_use_detected_pause_end(self) -> None:
        words = [
            {
                "type": "word",
                "text": "qiladigan,",
                "start": 411.42,
                "end": 413.0,
                "characters": [
                    {"text": "q", "start": 411.42, "end": 411.46},
                    {"text": "i", "start": 411.46, "end": 411.50},
                    {"text": "l", "start": 411.50, "end": 411.52},
                    {"text": "a", "start": 411.52, "end": 411.68},
                    {"text": "d", "start": 411.68, "end": 411.72},
                    {"text": "i", "start": 411.72, "end": 411.74},
                    {"text": "g", "start": 411.74, "end": 411.94},
                    {"text": "a", "start": 411.94, "end": 412.02},
                    {"text": "n", "start": 412.02, "end": 413.00},
                    {"text": ",", "start": 413.00, "end": 413.00},
                ],
            }
        ]

        plain = build_standard_tokens(words, pause_detection=False)
        detected = build_standard_tokens(words, pause_detection=True)

        self.assertEqual(plain[0]["end"], 413.0)
        self.assertLess(detected[0]["end"], 412.40)
        self.assertGreater(detected[0]["end"], 412.15)

    def test_social_cues_can_end_earlier_with_pause_detection(self) -> None:
        payload = {
            "words": [
                {
                    "type": "word",
                    "text": "ibodat",
                    "start": 410.88,
                    "end": 411.38,
                    "characters": [
                        {"text": "i", "start": 410.88, "end": 410.98},
                        {"text": "b", "start": 410.98, "end": 411.04},
                        {"text": "o", "start": 411.04, "end": 411.159},
                        {"text": "d", "start": 411.16, "end": 411.18},
                        {"text": "a", "start": 411.18, "end": 411.32},
                        {"text": "t", "start": 411.32, "end": 411.38},
                    ],
                },
                {
                    "type": "word",
                    "text": "qiladigan,",
                    "start": 411.42,
                    "end": 413.0,
                    "characters": [
                        {"text": "q", "start": 411.42, "end": 411.46},
                        {"text": "i", "start": 411.46, "end": 411.50},
                        {"text": "l", "start": 411.50, "end": 411.52},
                        {"text": "a", "start": 411.52, "end": 411.68},
                        {"text": "d", "start": 411.68, "end": 411.72},
                        {"text": "i", "start": 411.72, "end": 411.74},
                        {"text": "g", "start": 411.74, "end": 411.94},
                        {"text": "a", "start": 411.94, "end": 412.02},
                        {"text": "n", "start": 412.02, "end": 413.00},
                        {"text": ",", "start": 413.00, "end": 413.00},
                    ],
                },
                {
                    "type": "word",
                    "text": "kalom",
                    "start": 413.04,
                    "end": 413.47,
                    "characters": [
                        {"text": "k", "start": 413.04, "end": 413.18},
                        {"text": "a", "start": 413.18, "end": 413.24},
                        {"text": "l", "start": 413.24, "end": 413.32},
                        {"text": "o", "start": 413.32, "end": 413.46},
                        {"text": "m", "start": 413.46, "end": 413.47},
                    ],
                },
                {
                    "type": "word",
                    "text": "o'qiydikan,",
                    "start": 413.56,
                    "end": 414.16,
                    "characters": [
                        {"text": "o", "start": 413.56, "end": 413.70},
                        {"text": "'", "start": 413.70, "end": 413.70},
                        {"text": "q", "start": 413.70, "end": 413.78},
                        {"text": "i", "start": 413.78, "end": 413.82},
                        {"text": "y", "start": 413.82, "end": 413.92},
                        {"text": "d", "start": 413.92, "end": 414.00},
                        {"text": "i", "start": 414.00, "end": 414.04},
                        {"text": "k", "start": 414.04, "end": 414.10},
                        {"text": "a", "start": 414.10, "end": 414.14},
                        {"text": "n", "start": 414.14, "end": 414.16},
                        {"text": ",", "start": 414.16, "end": 414.16},
                    ],
                },
            ]
        }

        plain = tokens_to_social_cues(build_social_word_tokens(payload, pause_detection=False))
        detected = tokens_to_social_cues(build_social_word_tokens(payload, pause_detection=True))

        self.assertEqual(plain[0], (410.88, 413.0, "ibodat qiladigan,"))
        self.assertEqual(detected[0][2], "ibodat qiladigan,")
        self.assertLess(detected[0][1], plain[0][1])
        self.assertLess(detected[0][1], detected[1][0])

    def test_standard_cues_can_split_at_detected_pause(self) -> None:
        words = [
            {
                "type": "word",
                "text": "ibodat",
                "start": 410.88,
                "end": 411.38,
                "characters": [
                    {"text": "i", "start": 410.88, "end": 410.98},
                    {"text": "b", "start": 410.98, "end": 411.04},
                    {"text": "o", "start": 411.04, "end": 411.159},
                    {"text": "d", "start": 411.16, "end": 411.18},
                    {"text": "a", "start": 411.18, "end": 411.32},
                    {"text": "t", "start": 411.32, "end": 411.38},
                ],
            },
            {
                "type": "word",
                "text": "qiladigan,",
                "start": 411.42,
                "end": 413.0,
                "characters": [
                    {"text": "q", "start": 411.42, "end": 411.46},
                    {"text": "i", "start": 411.46, "end": 411.50},
                    {"text": "l", "start": 411.50, "end": 411.52},
                    {"text": "a", "start": 411.52, "end": 411.68},
                    {"text": "d", "start": 411.68, "end": 411.72},
                    {"text": "i", "start": 411.72, "end": 411.74},
                    {"text": "g", "start": 411.74, "end": 411.94},
                    {"text": "a", "start": 411.94, "end": 412.02},
                    {"text": "n", "start": 412.02, "end": 413.00},
                    {"text": ",", "start": 413.00, "end": 413.00},
                ],
            },
            {
                "type": "word",
                "text": "kalom",
                "start": 413.04,
                "end": 413.47,
                "characters": [
                    {"text": "k", "start": 413.04, "end": 413.18},
                    {"text": "a", "start": 413.18, "end": 413.24},
                    {"text": "l", "start": 413.24, "end": 413.32},
                    {"text": "o", "start": 413.32, "end": 413.46},
                    {"text": "m", "start": 413.46, "end": 413.47},
                ],
            },
            {
                "type": "word",
                "text": "o'qiydikan,",
                "start": 413.56,
                "end": 414.16,
                "characters": [
                    {"text": "o", "start": 413.56, "end": 413.70},
                    {"text": "'", "start": 413.70, "end": 413.70},
                    {"text": "q", "start": 413.70, "end": 413.78},
                    {"text": "i", "start": 413.78, "end": 413.82},
                    {"text": "y", "start": 413.82, "end": 413.92},
                    {"text": "d", "start": 413.92, "end": 414.00},
                    {"text": "i", "start": 414.00, "end": 414.04},
                    {"text": "k", "start": 414.04, "end": 414.10},
                    {"text": "a", "start": 414.10, "end": 414.14},
                    {"text": "n", "start": 414.14, "end": 414.16},
                    {"text": ",", "start": 414.16, "end": 414.16},
                ],
            },
        ]

        plain = tokens_to_standard_cues(build_standard_tokens(words, pause_detection=False))
        detected = tokens_to_standard_cues(build_standard_tokens(words, pause_detection=True))

        self.assertEqual(len(plain), 1)
        self.assertEqual(len(detected), 2)
        self.assertEqual(detected[0][2], "ibodat qiladigan,")
        self.assertEqual(detected[1][2], "kalom o'qiydikan,")

    def test_timing_sentence_split_can_use_detected_pause(self) -> None:
        payload = {
            "words": [
                {"type": "word", "text": "manga", "start": 459.20, "end": 459.48},
                {"type": "word", "text": "tegingan", "start": 459.58, "end": 460.18},
                {
                    "type": "word",
                    "text": "narsa",
                    "start": 460.19,
                    "end": 461.80,
                    "characters": [
                        {"text": "n", "start": 460.19, "end": 460.32},
                        {"text": "a", "start": 460.32, "end": 460.34},
                        {"text": "r", "start": 460.34, "end": 460.499},
                        {"text": "s", "start": 460.50, "end": 460.54},
                        {"text": "a", "start": 460.54, "end": 461.80},
                    ],
                },
                {"type": "word", "text": "uning", "start": 461.86, "end": 462.19},
                {"type": "word", "text": "ovozi", "start": 462.28, "end": 462.82},
            ]
        }

        plain = payload_to_sentence_items(payload, use_timing_split=True, pause_detection=False)
        detected = payload_to_sentence_items(payload, use_timing_split=True, pause_detection=True)

        self.assertEqual(len(plain), 1)
        self.assertEqual(len(detected), 2)
        self.assertEqual(detected[0]["text"], "manga tegingan narsa")
        self.assertEqual(detected[1]["text"], "uning ovozi")


if __name__ == "__main__":
    unittest.main()
