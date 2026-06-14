# Slides → PDF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Combine a bundle's slide JPEGs into one slides-only PDF so it can be uploaded to Claude.ai as a single file.

**Architecture:** A pure-stdlib PDF writer embeds each JPEG directly via the `DCTDecode` filter (no re-encode, no dependency), one page per image sized to the image. A standalone `make_pdf.py` builds a PDF from any bundle; a `--pdf` flag on `watch.py` writes `slides.pdf` into the bundle on `--save`.

**Tech Stack:** Python 3 stdlib only; `unittest`.

---

Implements `docs/superpowers/specs/2026-06-14-slides-pdf-design.md`.

**Run tests:** `python3 -m unittest discover -s tests -v`

## Task 1: `pdf.py` — pure-stdlib JPEG→PDF

**Files:**
- Create: `tests/test_pdf.py`
- Create: `scripts/pdf.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m unittest tests.test_pdf -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pdf'`.

- [ ] **Step 3: Implement**

```python
# scripts/pdf.py
#!/usr/bin/env python3
"""Combine JPEG slides into a single PDF — pure stdlib.

JPEGs embed directly into PDF via the DCTDecode filter (no re-encode, no Pillow).
One page per image, sized to the image's pixel dimensions.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _jpeg_size(data: bytes) -> tuple[int, int]:
    """Return (width, height) parsed from a JPEG's SOF marker, without decoding."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG (no SOI marker)")
    i, n = 2, len(data)
    while i < n:
        if data[i] != 0xFF:
            i += 1
            continue
        while i < n and data[i] == 0xFF:  # skip fill bytes
            i += 1
        if i >= n:
            break
        marker = data[i]
        i += 1
        # Standalone markers (no length): TEM(01), RSTn(D0-D7), SOI(D8), EOI(D9).
        if marker == 0x01 or 0xD0 <= marker <= 0xD9:
            continue
        if i + 2 > n:
            break
        seg_len = (data[i] << 8) | data[i + 1]
        # SOFn = C0-CF except DHT(C4), JPG(C8), DAC(CC).
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if i + 7 > n:
                break
            height = (data[i + 3] << 8) | data[i + 4]
            width = (data[i + 5] << 8) | data[i + 6]
            return width, height
        i += seg_len
    raise ValueError("no SOF marker found")


def images_to_pdf(image_paths, out_path) -> int:
    """Write a slides-only PDF (one JPEG per page). Returns the page count."""
    images: list[tuple[bytes, int, int]] = []
    for p in image_paths:
        p = Path(p)
        try:
            data = p.read_bytes()
        except OSError as exc:
            print(f"[watch] pdf: skipping {p}: {exc}", file=sys.stderr)
            continue
        if not data:
            print(f"[watch] pdf: skipping empty {p}", file=sys.stderr)
            continue
        try:
            width, height = _jpeg_size(data)
        except ValueError as exc:
            print(f"[watch] pdf: skipping {p}: {exc}", file=sys.stderr)
            continue
        images.append((data, width, height))

    if not images:
        raise SystemExit("no usable JPEG images to write to PDF")

    buf = bytearray()
    offsets: dict[int, int] = {}

    def put(s: str) -> None:
        buf.extend(s.encode("latin-1"))

    def begin(n: int) -> None:
        offsets[n] = len(buf)
        put(f"{n} 0 obj\n")

    put("%PDF-1.7\n")
    buf.extend(b"%\xe2\xe3\xcf\xd3\n")  # binary marker

    n_images = len(images)
    page_nums = [3 + i * 3 for i in range(n_images)]

    begin(1)
    put("<< /Type /Catalog /Pages 2 0 R >>\n")
    put("endobj\n")

    begin(2)
    kids = " ".join(f"{pn} 0 R" for pn in page_nums)
    put(f"<< /Type /Pages /Kids [{kids}] /Count {n_images} >>\n")
    put("endobj\n")

    for i, (data, width, height) in enumerate(images):
        page_n = 3 + i * 3
        img_n = page_n + 1
        cont_n = page_n + 2

        begin(page_n)
        put(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            f"/Resources << /XObject << /Im0 {img_n} 0 R >> >> /Contents {cont_n} 0 R >>\n")
        put("endobj\n")

        begin(img_n)
        put(f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode "
            f"/Length {len(data)} >>\n")
        put("stream\n")
        buf.extend(data)
        put("\nendstream\n")
        put("endobj\n")

        content = f"q\n{width} 0 0 {height} 0 0 cm\n/Im0 Do\nQ\n".encode("latin-1")
        begin(cont_n)
        put(f"<< /Length {len(content)} >>\n")
        put("stream\n")
        buf.extend(content)
        put("endstream\n")
        put("endobj\n")

    total = 2 + n_images * 3
    xref_off = len(buf)
    put("xref\n")
    put(f"0 {total + 1}\n")
    put("0000000000 65535 f \n")
    for num in range(1, total + 1):
        put(f"{offsets[num]:010d} 00000 n \n")
    put("trailer\n")
    put(f"<< /Size {total + 1} /Root 1 0 R >>\n")
    put("startxref\n")
    put(f"{xref_off}\n")
    put("%%EOF\n")

    Path(out_path).write_bytes(bytes(buf))
    return n_images
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m unittest tests.test_pdf -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_pdf.py scripts/pdf.py
git commit -m "feat(pdf): pure-stdlib JPEG->PDF writer"
```

## Task 2: `make_pdf.py` — standalone bundle→PDF tool

**Files:**
- Create: `scripts/make_pdf.py`

- [ ] **Step 1: Implement**

```python
# scripts/make_pdf.py
#!/usr/bin/env python3
"""Combine a bundle's slide frames (or a folder of JPEGs) into one PDF."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from pdf import images_to_pdf  # noqa: E402


def collect_images(directory: Path) -> list[Path]:
    frames = directory / "frames"
    base = frames if frames.is_dir() else directory
    return sorted(base.glob("*.jpg"))


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: make_pdf.py <bundle-or-image-dir> [out.pdf]", file=sys.stderr)
        return 2
    directory = Path(sys.argv[1]).expanduser().resolve()
    if not directory.is_dir():
        raise SystemExit(f"not a directory: {directory}")
    images = collect_images(directory)
    if not images:
        raise SystemExit(f"no .jpg images found in {directory}")
    out = (Path(sys.argv[2]).expanduser().resolve()
           if len(sys.argv) > 2 else directory / "slides.pdf")
    pages = images_to_pdf(images, out)
    print(f"[watch] wrote {out} ({pages} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify against the existing real bundle**

Run: `python3 scripts/make_pdf.py /tmp/algo-bundle`
Expected: prints `wrote /tmp/algo-bundle/slides.pdf (53 pages)`; file exists and is a valid PDF.

- [ ] **Step 3: Commit**

```bash
git add scripts/make_pdf.py
git commit -m "feat(pdf): make_pdf.py standalone bundle->PDF tool"
```

## Task 3: `watch.py --pdf` flag

**Files:**
- Modify: `scripts/watch.py`

- [ ] **Step 1: Add the import**

Add to the import block near `import report as report_mod`:

```python
import pdf as pdf_mod  # noqa: E402
```

- [ ] **Step 2: Add the argument**

After the `--scenes` argument group (near `--save`), add:

```python
    ap.add_argument("--pdf", action="store_true", help="Also write slides.pdf (all frames combined) for easy upload")
```

- [ ] **Step 3: Generate the PDF after the report**

In `main()`, immediately after the `if args.save:` / `else:` block that prints the
work-dir line, add:

```python
    if args.pdf:
        frame_imgs = sorted((work / "frames").glob("*.jpg"))
        if frame_imgs:
            pages = pdf_mod.images_to_pdf(frame_imgs, work / "slides.pdf")
            print()
            print(f"_Combined {pages} frames into `{work / 'slides.pdf'}` — upload this one file to Claude.ai._")
        else:
            print("[watch] --pdf: no frames to combine", file=sys.stderr)
```

- [ ] **Step 4: Verify the flag works end-to-end**

Run: `python3 scripts/make_pdf.py /tmp/algo-bundle /tmp/smoke.pdf && python3 -c "print(open('/tmp/smoke.pdf','rb').read(8))"`
Expected: `b'%PDF-1.7'`.

(Full `--pdf` path is exercised by the manual smoke in Task 5.)

- [ ] **Step 5: Commit**

```bash
git add scripts/watch.py
git commit -m "feat(watch): --pdf flag writes combined slides.pdf"
```

## Task 4: Docs

**Files:**
- Modify: `commands/watch-save.md`, `SKILL.md`, `README.md`

- [ ] **Step 1: `commands/watch-save.md`** — in the Claude.ai conveyance bullet, add:

```markdown
   - **Easiest for Claude.ai:** if a `slides.pdf` is present in the bundle (pass `--pdf`, or run `scripts/make_pdf.py <bundle-dir>`), upload that single PDF instead of every image in `frames/`.
```

- [ ] **Step 2: `SKILL.md`** — add to the Step 2 flags list:

```markdown
- `--pdf` — also write `slides.pdf` (all slide frames combined into one PDF) into the bundle, so you can upload a single file to Claude.ai instead of many images. Or build one from an existing bundle later: `python3 "${CLAUDE_SKILL_DIR}/scripts/make_pdf.py" <bundle-dir>`.
```

- [ ] **Step 3: `README.md`** — in the "Convey a watched video to another session" section, add:

```markdown
To avoid attaching dozens of images on Claude.ai, combine the slides into one PDF:

```bash
/watch lecture.mp4 --scenes --save ./bundle --pdf   # writes ./bundle/slides.pdf
# or, from an existing bundle:
python3 scripts/make_pdf.py ./bundle                 # writes ./bundle/slides.pdf
```
```

- [ ] **Step 4: Commit**

```bash
git add commands/watch-save.md SKILL.md README.md
git commit -m "docs: document --pdf / make_pdf.py slide-PDF export"
```

## Task 5: Final verification

- [ ] **Step 1: Full suite**

Run: `python3 -m unittest discover -s tests -v`
Expected: all PASS (existing 36 + 4 new pdf = 40).

- [ ] **Step 2: Real PDF validity**

Run: `python3 scripts/make_pdf.py /tmp/algo-bundle` then open `/tmp/algo-bundle/slides.pdf`
with a PDF reader (or Read tool) and confirm it shows the slides (53 pages).

- [ ] **Step 3: `--pdf` integration**

Run: `python3 scripts/watch.py "Algorithm 6:11.mov" --scenes --whisper local --save /tmp/algo-bundle --pdf`
Expected: prints the `slides.pdf` line; transcript served from cache (fast).

## Self-review

- **Spec coverage:** slides-only PDF → Task 1; pure-stdlib DCTDecode → Task 1; standalone tool → Task 2; `--pdf` flag → Task 3; docs → Task 4; tests + e2e → Tasks 1, 5. ✓
- **Placeholders:** none; full code in every step. ✓
- **Type consistency:** `_jpeg_size` and `images_to_pdf(image_paths, out_path) -> int` used identically across pdf.py, make_pdf.py, watch.py, and tests. ✓
