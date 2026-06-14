# tests/test_load_bundle.py
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import load_bundle  # noqa: E402


class ResolveReportTests(unittest.TestCase):
    def test_relative_frame_links_become_absolute(self):
        md = "### Slide 1\n`frames/slide_0001.jpg`\n> [00:00] hi\n"
        out = load_bundle.resolve_report(md, "/abs/bundle")
        self.assertIn("`/abs/bundle/frames/slide_0001.jpg`", out)
        self.assertNotIn("`frames/slide_0001.jpg`", out)

    def test_non_frame_backticks_untouched(self):
        md = "use `--scenes` here and `frames/a.jpg`"
        out = load_bundle.resolve_report(md, "/abs/bundle")
        self.assertIn("`--scenes`", out)
        self.assertIn("`/abs/bundle/frames/a.jpg`", out)


if __name__ == "__main__":
    unittest.main()
