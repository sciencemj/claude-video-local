#!/usr/bin/env python3
"""Detect slide changes in a lecture video and frame/group them per slide.

ffmpeg's content scene filter finds cuts; we turn cuts into slide spans, merge
build-animation flicker, cap to a frame budget, extract one representative frame
per slide, and group the transcript under each slide.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from transcribe import filter_range, format_transcript  # noqa: E402

PTS_RE = re.compile(r"pts_time:([0-9]+\.?[0-9]*)")


def _merge_short(spans: list[dict], min_seconds: float) -> list[dict]:
    out: list[dict] = []
    for s in spans:
        if out and (out[-1]["end"] - out[-1]["start"]) < min_seconds:
            out[-1]["end"] = s["end"]
        else:
            out.append(dict(s))
    while len(out) > 1 and (out[-1]["end"] - out[-1]["start"]) < min_seconds:
        out[-2]["end"] = out[-1]["end"]
        out.pop()
    return out


def cuts_to_slides(cuts: list[float], duration: float, min_slide_seconds: float = 3.0) -> list[dict]:
    cuts = sorted({round(c, 3) for c in cuts if c is not None and c >= 0})
    if not cuts or cuts[0] > 0:
        cuts = [0.0] + cuts
    spans: list[dict] = []
    for i, start in enumerate(cuts):
        end = cuts[i + 1] if i + 1 < len(cuts) else round(duration, 3)
        if end <= start:
            continue
        spans.append({"start": round(start, 3), "end": round(end, 3)})
    merged = _merge_short(spans, min_slide_seconds)
    for i, s in enumerate(merged):
        s["index"] = i + 1
    return merged


def merge_to_cap(slides: list[dict], max_slides: int) -> tuple[list[dict], int]:
    slides = [dict(s) for s in slides]
    merged = 0
    while len(slides) > max_slides:
        i = min(range(len(slides)), key=lambda j: slides[j]["end"] - slides[j]["start"])
        if i + 1 < len(slides):
            slides[i]["end"] = slides[i + 1]["end"]
            slides.pop(i + 1)
        else:
            slides[i - 1]["end"] = slides[i]["end"]
            slides.pop(i)
        merged += 1
    for k, s in enumerate(slides):
        s["index"] = k + 1
    return slides, merged


def representative_timestamp(slide: dict, offset: float = 0.5) -> float:
    start, end = slide["start"], slide["end"]
    t = start + offset
    if t >= end:
        t = max(start, (start + end) / 2.0)
    return round(t, 3)
