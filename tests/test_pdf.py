# tests/test_pdf.py
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import pdf  # noqa: E402


def _fake_jpeg(width: int, height: int) -> bytes:
    # SOI + SOF0 (dims) + EOI — enough for _jpeg_size and for embedding.
    sof = (b"\xff\xc0\x00\x11\x08"
           + bytes([height >> 8, height & 0xFF, width >> 8, width & 0xFF])
           + b"\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01")
    return b"\xff\xd8" + sof + b"\xff\xd9"


class JpegSizeTests(unittest.TestCase):
    def test_parses_dimensions(self):
        self.assertEqual(pdf._jpeg_size(_fake_jpeg(640, 480)), (640, 480))

    def test_rejects_non_jpeg(self):
        with self.assertRaises(ValueError):
            pdf._jpeg_size(b"not a jpeg at all")


class ImagesToPdfTests(unittest.TestCase):
    def test_writes_valid_pdf_structure(self):
        tmp = Path(tempfile.mkdtemp())
        imgs = []
        for i, (w, h) in enumerate([(640, 480), (800, 600)]):
            p = tmp / f"slide_{i:04d}.jpg"
            p.write_bytes(_fake_jpeg(w, h))
            imgs.append(p)
        out = tmp / "slides.pdf"
        n = pdf.images_to_pdf(imgs, out)
        self.assertEqual(n, 2)
        data = out.read_bytes()
        self.assertTrue(data.startswith(b"%PDF-"))
        self.assertTrue(data.rstrip().endswith(b"%%EOF"))
        self.assertIn(b"/Count 2", data)
        self.assertEqual(data.count(b"/Subtype /Image"), 2)
        self.assertIn(b"/MediaBox [0 0 640 480]", data)

    def test_raises_when_no_usable_images(self):
        tmp = Path(tempfile.mkdtemp())
        with self.assertRaises(SystemExit):
            pdf.images_to_pdf([tmp / "missing.jpg"], tmp / "out.pdf")


if __name__ == "__main__":
    unittest.main()
