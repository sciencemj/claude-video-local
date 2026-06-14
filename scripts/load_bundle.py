#!/usr/bin/env python3
"""Re-ingest a /watch bundle into the current Claude Code session.

Prints the bundle's report with frame links resolved to absolute paths, then a
flat list of every frame path so Claude can Read them all.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def resolve_report(report_md: str, base_dir) -> str:
    base = str(Path(base_dir).resolve())
    return report_md.replace("`frames/", f"`{base}/frames/")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: load_bundle.py <bundle-dir>", file=sys.stderr)
        return 2
    d = Path(sys.argv[1]).expanduser().resolve()
    report_path = d / "report.md"
    meta_path = d / "meta.json"
    if not report_path.exists():
        raise SystemExit(f"Not a bundle: {report_path} is missing")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            print(f"[watch] loaded bundle: {meta.get('source')} "
                  f"({meta.get('mode')} mode, {meta.get('frame_count')} frames)", file=sys.stderr)
        except (OSError, json.JSONDecodeError):
            pass

    print(resolve_report(report_path.read_text(encoding="utf-8"), d))

    frames_dir = d / "frames"
    if frames_dir.exists():
        print("\n## Frame paths\n")
        print("**Read each path below with the Read tool.**\n")
        for p in sorted(frames_dir.glob("*.jpg")):
            print(f"- `{p}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
