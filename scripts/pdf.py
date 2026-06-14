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
