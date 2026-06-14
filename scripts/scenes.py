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


def parse_scene_times(stderr: str) -> list[float]:
    out: list[float] = []
    for m in PTS_RE.finditer(stderr or ""):
        try:
            out.append(round(float(m.group(1)), 3))
        except ValueError:
            pass
    return out


def group_transcript(slides: list[dict], segments: list[dict]) -> list[dict]:
    out: list[dict] = []
    for s in slides:
        segs = filter_range(segments, s["start"], s["end"])
        d = dict(s)
        d["segments"] = segs
        d["text"] = format_transcript(segs)
        out.append(d)
    return out


def detect_cuts(video_path: str, threshold: float = 0.3) -> list[float]:
    """Return slide-cut timestamps (always starting with 0.0) via ffmpeg scene filter."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")
    print(f"[watch] scanning for slide cuts (threshold {threshold}) — one decode pass…", file=sys.stderr)
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats",
        "-i", str(Path(video_path).resolve()),
        "-filter:v", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    times = [t for t in parse_scene_times(result.stderr) if t > 0.0]
    cuts = sorted({0.0, *(round(t, 3) for t in times)})
    print(f"[watch] detected {len(cuts)} slide boundaries", file=sys.stderr)
    return cuts


SCORE_RE = re.compile(r"scene_score=([0-9]+\.?[0-9]*)")


def parse_scored(text: str) -> list[tuple[float, float]]:
    """Parse ffmpeg metadata=print output into (timestamp, scene_score) pairs."""
    out: list[tuple[float, float]] = []
    current_t: float | None = None
    for line in (text or "").splitlines():
        ts = PTS_RE.search(line)
        if ts:
            try:
                current_t = float(ts.group(1))
            except ValueError:
                current_t = None
            continue
        sc = SCORE_RE.search(line)
        if sc and current_t is not None:
            try:
                out.append((round(current_t, 3), float(sc.group(1))))
            except ValueError:
                pass
            current_t = None
    return out


def scene_scores(video_path: str, floor: float = 0.02) -> list[tuple[float, float]]:
    """One decode pass: return (timestamp, scene_score) for every candidate transition.

    Captures scores once so the threshold can be tuned in software (no re-decode).
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")
    import os
    import tempfile

    fd, tmp = tempfile.mkstemp(suffix=".txt", prefix="watch-scenes-")
    os.close(fd)
    print("[watch] scanning for slide cuts (one decode pass)…", file=sys.stderr)
    try:
        cmd = [
            "ffmpeg", "-hide_banner", "-nostats",
            "-i", str(Path(video_path).resolve()),
            "-filter:v", f"select='gt(scene,{floor})',metadata=print:file={tmp}",
            "-an", "-f", "null", "-",
        ]
        subprocess.run(cmd, capture_output=True, text=True)
        text = Path(tmp).read_text(encoding="utf-8", errors="ignore")
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return parse_scored(text)


def cuts_from_scores(scored: list[tuple[float, float]], threshold: float) -> list[float]:
    """Cut timestamps (always including 0.0) for scores at or above `threshold`."""
    cuts = {0.0}
    cuts.update(round(t, 3) for t, s in scored if s >= threshold and t > 0.0)
    return sorted(cuts)


def adaptive_cuts(
    scored: list[tuple[float, float]],
    duration: float,
    start_threshold: float = 0.3,
    min_slide_seconds: float = 3.0,
    target_min: int = 8,
    floor_threshold: float = 0.025,
) -> tuple[list[float], float]:
    """Lower the threshold (in software) until at least `target_min` slides emerge.

    Returns (cuts, threshold_used). If the target is never reached, returns the
    threshold that produced the most slides.
    """
    ladder: list[float] = []
    th = start_threshold
    while th >= floor_threshold:
        ladder.append(round(th, 4))
        th /= 1.5
    if not ladder or ladder[-1] > floor_threshold:
        ladder.append(round(floor_threshold, 4))

    best: tuple[list[float], float, int] | None = None
    for th in ladder:
        cuts = cuts_from_scores(scored, th)
        count = len(cuts_to_slides(cuts, duration, min_slide_seconds))
        if best is None or count > best[2]:
            best = (cuts, th, count)
        if count >= target_min:
            return cuts, th
    assert best is not None
    return best[0], best[1]


def extract_slide_frames(video_path: str, slides: list[dict], out_dir, resolution: int = 768) -> list[dict]:
    """Extract one representative JPEG per slide. Returns slides with 'path' + 'timestamp_seconds'."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob("slide_*.jpg"):
        existing.unlink()

    out: list[dict] = []
    src = str(Path(video_path).resolve())
    for s in slides:
        ts = representative_timestamp(s)
        path = out_dir / f"slide_{s['index']:04d}.jpg"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{ts:.3f}", "-i", src,
            "-frames:v", "1", "-vf", f"scale={resolution}:-2", "-q:v", "4",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not path.exists():
            print(f"[watch] slide {s['index']} frame failed: {result.stderr.strip()}", file=sys.stderr)
            continue
        d = dict(s)
        d["path"] = str(path)
        d["timestamp_seconds"] = ts
        out.append(d)
    return out
