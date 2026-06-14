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


class SceneParseAndGroupTests(unittest.TestCase):
    def test_parse_scene_times(self):
        stderr = (
            "[Parsed_showinfo_1 @ 0x] n:0 pts:0 pts_time:0 ...\n"
            "[Parsed_showinfo_1 @ 0x] n:1 pts:720 pts_time:30.03 ...\n"
            "[Parsed_showinfo_1 @ 0x] n:2 pts:2160 pts_time:90.5 ...\n"
        )
        self.assertEqual(scenes.parse_scene_times(stderr), [0.0, 30.03, 90.5])

    def test_group_transcript_assigns_overlapping_segments(self):
        segments = [
            {"start": 0.0, "end": 10.0, "text": "intro"},
            {"start": 12.0, "end": 20.0, "text": "topic one"},
            {"start": 95.0, "end": 100.0, "text": "topic two"},
        ]
        slides = [{"index": 1, "start": 0.0, "end": 90.0},
                  {"index": 2, "start": 90.0, "end": 120.0}]
        grouped = scenes.group_transcript(slides, segments)
        self.assertEqual([s["text"] for s in grouped[0]["segments"]], ["intro", "topic one"])
        self.assertEqual([s["text"] for s in grouped[1]["segments"]], ["topic two"])
        self.assertIn("intro", grouped[0]["text"])


class AdaptiveThresholdTests(unittest.TestCase):
    def test_parse_scored(self):
        text = (
            "frame:0 pts:2091 pts_time:69.7\nlavfi.scene_score=0.050000\n"
            "frame:1 pts:7345 pts_time:244.83\nlavfi.scene_score=0.900000\n"
        )
        self.assertEqual(scenes.parse_scored(text), [(69.7, 0.05), (244.83, 0.9)])

    def test_cuts_from_scores_filters_and_prepends_zero(self):
        scored = [(10.0, 0.05), (20.0, 0.5), (30.0, 0.2)]
        self.assertEqual(scenes.cuts_from_scores(scored, 0.1), [0.0, 20.0, 30.0])
        self.assertEqual(scenes.cuts_from_scores(scored, 0.4), [0.0, 20.0])

    def test_adaptive_lowers_threshold_to_meet_target(self):
        # One strong cut (0.9) plus many weak (0.05) slide changes every 60s.
        scored = [(float(t), 0.05) for t in range(60, 1200, 60)] + [(30.0, 0.9)]
        cuts, used = scenes.adaptive_cuts(
            scored, duration=1300.0, start_threshold=0.3,
            min_slide_seconds=3.0, target_min=8, floor_threshold=0.03,
        )
        slides = scenes.cuts_to_slides(cuts, 1300.0, min_slide_seconds=3.0)
        self.assertGreaterEqual(len(slides), 8)
        self.assertLess(used, 0.3)

    def test_adaptive_keeps_start_threshold_when_target_already_met(self):
        scored = [(float(t), 0.5) for t in range(60, 900, 60)]  # 14 strong cuts
        cuts, used = scenes.adaptive_cuts(
            scored, duration=1000.0, start_threshold=0.3,
            min_slide_seconds=3.0, target_min=8, floor_threshold=0.03,
        )
        self.assertEqual(used, 0.3)


if __name__ == "__main__":
    unittest.main()
