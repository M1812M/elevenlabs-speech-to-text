import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elevenlabs_toolkit.core.srt_builder import tokens_to_standard_cues


class SrtSplitTests(unittest.TestCase):
    def test_prevents_one_word_orphan_last_cue(self) -> None:
        words = [
            "shunda",
            "shu",
            "ovozni",
            "eshitganman,",
            "hushimdan",
            "ketganman-da,",
            "uje,",
            "uje",
            "skoriy",
            "kelyapti",
            "olgani.",
        ]
        tokens = []
        start = 26.0
        for idx, text in enumerate(words):
            st = start + idx * 0.55
            tokens.append({"text": text, "start": st, "end": st + 0.25, "speaker": "speaker_0"})

        cues = tokens_to_standard_cues(tokens)

        self.assertGreaterEqual(len(cues), 2)
        self.assertGreater(len(cues[-1][2].split()), 1)
        self.assertNotEqual(cues[-1][2].strip(), "olgani.")


if __name__ == "__main__":
    unittest.main()
