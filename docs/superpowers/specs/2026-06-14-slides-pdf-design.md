# Design: combine slide frames into one PDF

- **Date:** 2026-06-14
- **Status:** Approved
- **Branch:** main

## Summary

When conveying a `/watch` bundle to Claude.ai / the web app, attaching dozens of
slide JPEGs is painful. Add the ability to combine a bundle's slide frames into a
single **slides-only PDF** (one slide image per page) so the user uploads one file.

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| PDF content | **Slides only** — one slide image per page, no transcript text. Transcript conveyed separately (`transcript.txt`). |
| Generation method | **Pure stdlib** — embed JPEGs directly via the PDF `DCTDecode` filter; no Pillow/img2pdf/ImageMagick dependency (matches the project's stdlib-only ethos). |
| Trigger | **Both** — a `--pdf` flag on `watch.py` (writes `slides.pdf` into the bundle on `--save`) **and** a standalone `make_pdf.py` that builds a PDF from any existing bundle. |

## Detailed design

### `scripts/pdf.py` (new, pure stdlib)

- `images_to_pdf(image_paths: list[Path], out_path) -> int` — writes a minimal,
  valid PDF with one page per JPEG; returns the page count.
  - `_jpeg_size(data: bytes) -> tuple[int, int]` — parse pixel dimensions from the
    JPEG `SOFn` marker (0xFFC0–0xFFCF, excluding DHT/DAC/DRI/SOS) without decoding.
  - Each image becomes an `XObject /Subtype /Image /Filter /DCTDecode` with the raw
    JPEG bytes as the stream (no re-encode), plus a page whose MediaBox equals the
    image's pixel size and whose content stream scales the image to fill the page.
  - Emits objects (catalog, pages, per-image page + xobject + content), a correct
    `xref` table, and `trailer`. JPEGs are embedded as `/ColorSpace /DeviceRGB`
    (the frames are RGB JPEGs from ffmpeg).
- Skips unreadable/zero-byte images with a stderr warning; raises `SystemExit` if
  no usable images remain.

### `scripts/make_pdf.py` (new, standalone CLI)

- `make_pdf.py <dir> [out.pdf]`:
  - If `<dir>/frames/` exists (a bundle), use `frames/*.jpg`; else use `<dir>/*.jpg`.
  - Sort by filename. Default output: `<dir>/slides.pdf`. Print the path + page count.

### `watch.py --pdf`

- New `--pdf` flag. After a `--save` run, call `pdf.images_to_pdf(<slide frame
  paths>, <save>/slides.pdf)`. Without `--save`, `--pdf` writes `slides.pdf` into
  the work dir and notes the path. Mentioned in the saved-bundle conveyance line.

### Docs

- `commands/watch-save.md`: note that `slides.pdf` (if generated) is the easiest
  thing to upload to Claude.ai.
- `SKILL.md` / `README.md`: document `--pdf` and `make_pdf.py`.

## Testing

- Unit (`tests/test_pdf.py`, stdlib `unittest`):
  - `_jpeg_size` on crafted SOF0 bytes returns the encoded dimensions.
  - `images_to_pdf` on small JPEG byte fixtures writes a file starting with `%PDF-`,
    ending with `%%EOF`, containing `/Count N` and N image XObjects.
- End-to-end (manual, run after implementing): build `slides.pdf` from the real
  `/tmp/algo-bundle` and open it with a PDF reader to confirm validity + that slides
  render.

## Out of scope (YAGNI)

Transcript text in the PDF; letter/A4 page fitting; bookmarks/outlines; non-JPEG
inputs (frames are always JPEG).
