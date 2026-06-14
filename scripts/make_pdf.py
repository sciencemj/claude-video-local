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
