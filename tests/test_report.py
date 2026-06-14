# tests/test_report.py
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import report  # noqa: E402


def uniform_ctx(**over):
    ctx = {
        "mode": "uniform", "source": "video.mp4", "title": "T", "uploader": None,
        "full_duration": 120.0, "focused": False,
        "effective_start": 0.0, "effective_end": 120.0, "effective_duration": 120.0,
        "width": 1280, "height": 720, "codec": "h264",
        "fps": 0.5, "target": 60, "max_frames": 100, "resolution": 512,
        "frames": [{"path": "/tmp/work/frames/frame_0001.jpg", "timestamp_seconds": 0.0}],
        "transcript_text": "[00:00] hi", "transcript_source": "captions",
        "transcript_segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
    }
    ctx.update(over)
    return ctx


def scene_ctx(**over):
    ctx = {
        "mode": "scenes", "source": "lecture.mp4", "title": "Lec", "uploader": None,
        "full_duration": 3600.0, "width": 1920, "height": 1080, "codec": "h264",
        "resolution": 768, "scene_threshold": 0.3, "merged_count": 0,
        "max_frames": 100, "transcript_source": "whisper (local: turbo)",
        "slides": [{"index": 1, "start": 0.0, "end": 90.0,
                    "path": "/tmp/work/frames/slide_0001.jpg",
                    "text": "[00:00] welcome", "segments": []}],
    }
    ctx.update(over)
    return ctx


class ReportTests(unittest.TestCase):
    def test_uniform_absolute_paths(self):
        out = report.render(uniform_ctx(), relative=False)
        self.assertIn("# watch: video report", out)
        self.assertIn("`/tmp/work/frames/frame_0001.jpg` (t=00:00)", out)

    def test_uniform_relative_paths(self):
        out = report.render(uniform_ctx(), relative=True)
        self.assertIn("`frames/frame_0001.jpg` (t=00:00)", out)
        self.assertNotIn("/tmp/work", out)

    def test_uniform_long_video_warning(self):
        out = report.render(uniform_ctx(full_duration=1800.0), relative=False)
        self.assertIn("Warning:", out)
        self.assertIn("--scenes", out)

    def test_scenes_structure_and_grouping(self):
        out = report.render(scene_ctx(), relative=False)
        self.assertIn("# watch: lecture report (scenes)", out)
        self.assertIn("### Slide 1 — 00:00 → 01:30", out)
        self.assertIn("`/tmp/work/frames/slide_0001.jpg`", out)
        self.assertIn("> [00:00] welcome", out)

    def test_scenes_relative_paths(self):
        out = report.render(scene_ctx(), relative=True)
        self.assertIn("`frames/slide_0001.jpg`", out)
        self.assertNotIn("/tmp/work", out)


if __name__ == "__main__":
    unittest.main()
