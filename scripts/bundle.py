#!/usr/bin/env python3
"""Write a portable /watch bundle (report.md + frames/ + transcript + meta)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

import report  # noqa: E402
from transcribe import format_transcript  # noqa: E402


def build_meta(ctx: dict) -> dict:
    items = ctx.get("slides") if ctx.get("mode") == "scenes" else ctx.get("frames")
    return {
        "schema": 1,
        "tool": "watch",
        "source": ctx.get("source"),
        "title": ctx.get("title"),
        "uploader": ctx.get("uploader"),
        "duration_seconds": round(ctx.get("full_duration") or 0.0, 2),
        "mode": ctx.get("mode"),
        "resolution": ctx.get("resolution"),
        "frame_count": len(items or []),
        "transcript_source": ctx.get("transcript_source"),
        "scene_threshold": ctx.get("scene_threshold"),
    }


def write_bundle(work, ctx: dict, segments: list[dict]) -> None:
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    (work / "report.md").write_text(report.render(ctx, relative=True), encoding="utf-8")
    (work / "transcript.json").write_text(
        json.dumps({"source": ctx.get("transcript_source"), "segments": segments},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (work / "transcript.txt").write_text(format_transcript(segments) + "\n", encoding="utf-8")
    (work / "meta.json").write_text(
        json.dumps(build_meta(ctx), indent=2, ensure_ascii=False), encoding="utf-8"
    )
