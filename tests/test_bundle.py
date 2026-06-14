# tests/test_bundle.py
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import bundle  # noqa: E402


def scene_ctx():
    return {
        "mode": "scenes", "source": "lecture.mp4", "title": "Lec", "uploader": None,
        "full_duration": 3600.0, "width": 1920, "height": 1080, "codec": "h264",
        "resolution": 768, "scene_threshold": 0.3, "merged_count": 0, "max_frames": 100,
        "transcript_source": "whisper (local: turbo)",
        "slides": [{"index": 1, "start": 0.0, "end": 90.0,
                    "path": "/abs/work/frames/slide_0001.jpg",
                    "text": "[00:00] welcome", "segments": []}],
    }


class BundleTests(unittest.TestCase):
    def test_build_meta(self):
        meta = bundle.build_meta(scene_ctx())
        self.assertEqual(meta["schema"], 1)
        self.assertEqual(meta["mode"], "scenes")
        self.assertEqual(meta["frame_count"], 1)
        self.assertEqual(meta["transcript_source"], "whisper (local: turbo)")

    def test_write_bundle_creates_files_with_relative_paths(self):
        work = Path(tempfile.mkdtemp())
        segs = [{"start": 0.0, "end": 1.0, "text": "welcome"}]
        bundle.write_bundle(work, scene_ctx(), segs)
        self.assertTrue((work / "report.md").exists())
        self.assertTrue((work / "transcript.json").exists())
        self.assertTrue((work / "transcript.txt").exists())
        self.assertTrue((work / "meta.json").exists())
        report_text = (work / "report.md").read_text()
        self.assertIn("`frames/slide_0001.jpg`", report_text)
        self.assertNotIn("/abs/work", report_text)
        self.assertEqual(json.loads((work / "transcript.json").read_text())["segments"], segs)


if __name__ == "__main__":
    unittest.main()
