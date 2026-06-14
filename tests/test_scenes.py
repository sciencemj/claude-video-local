# tests/test_scenes.py
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import scenes  # noqa: E402


class SlideMathTests(unittest.TestCase):
    def test_cuts_to_slides_basic_spans(self):
        slides = scenes.cuts_to_slides([0.0, 30.0, 90.0], duration=120.0, min_slide_seconds=3.0)
        self.assertEqual([(s["start"], s["end"]) for s in slides],
                         [(0.0, 30.0), (30.0, 90.0), (90.0, 120.0)])
        self.assertEqual([s["index"] for s in slides], [1, 2, 3])

    def test_cuts_to_slides_prepends_zero(self):
        slides = scenes.cuts_to_slides([10.0, 20.0], duration=30.0)
        self.assertEqual(slides[0]["start"], 0.0)

    def test_cuts_to_slides_merges_short(self):
        # 0-1 (short) merges forward into the next span
        slides = scenes.cuts_to_slides([0.0, 1.0, 40.0], duration=80.0, min_slide_seconds=3.0)
        self.assertEqual([(s["start"], s["end"]) for s in slides], [(0.0, 40.0), (40.0, 80.0)])

    def test_merge_to_cap_reduces_count(self):
        slides = [{"index": i + 1, "start": float(i), "end": float(i) + 1.0} for i in range(5)]
        capped, merged = scenes.merge_to_cap(slides, max_slides=3)
        self.assertEqual(len(capped), 3)
        self.assertEqual(merged, 2)
        self.assertEqual(capped[-1]["end"], 5.0)  # coverage preserved end-to-end
        self.assertEqual([s["index"] for s in capped], [1, 2, 3])

    def test_representative_timestamp_offsets(self):
        self.assertEqual(scenes.representative_timestamp({"start": 10.0, "end": 40.0}), 10.5)

    def test_representative_timestamp_clamps_short_slide(self):
        t = scenes.representative_timestamp({"start": 10.0, "end": 10.2})
        self.assertTrue(10.0 <= t < 10.2)


if __name__ == "__main__":
    unittest.main()
