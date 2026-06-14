#!/usr/bin/env python3
"""Pure markdown rendering for /watch reports (uniform + scene modes).

One renderer drives both stdout (absolute frame paths) and the saved bundle
report.md (relative `frames/…` paths), so the two never diverge.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from frames import format_time  # noqa: E402


def _frame_ref(path: str, relative: bool) -> str:
    return f"frames/{Path(path).name}" if relative else path


def render(ctx: dict, relative: bool = False) -> str:
    if ctx.get("mode") == "scenes":
        return _render_scenes(ctx, relative)
    return _render_uniform(ctx, relative)


def _header(ctx: dict, lines: list[str]) -> None:
    lines.append(f"- **Source:** {ctx['source']}")
    if ctx.get("title"):
        lines.append(f"- **Title:** {ctx['title']}")
    if ctx.get("uploader"):
        lines.append(f"- **Uploader:** {ctx['uploader']}")
    full = ctx.get("full_duration") or 0.0
    lines.append(f"- **Duration:** {format_time(full)} ({full:.1f}s)")
    if ctx.get("width") and ctx.get("height"):
        lines.append(f"- **Resolution:** {ctx['width']}x{ctx['height']} ({ctx.get('codec') or 'unknown codec'})")


def _render_uniform(ctx: dict, relative: bool) -> str:
    L: list[str] = ["", "# watch: video report", ""]
    L.append(f"- **Source:** {ctx['source']}")
    if ctx.get("title"):
        L.append(f"- **Title:** {ctx['title']}")
    if ctx.get("uploader"):
        L.append(f"- **Uploader:** {ctx['uploader']}")
    full = ctx.get("full_duration") or 0.0
    L.append(f"- **Duration:** {format_time(full)} ({full:.1f}s)")
    if ctx.get("focused"):
        L.append(f"- **Focus range:** {format_time(ctx['effective_start'])} → "
                 f"{format_time(ctx['effective_end'])} ({ctx['effective_duration']:.1f}s)")
    if ctx.get("width") and ctx.get("height"):
        L.append(f"- **Resolution:** {ctx['width']}x{ctx['height']} ({ctx.get('codec') or 'unknown codec'})")
    mode = "focused" if ctx.get("focused") else "full"
    frames = ctx.get("frames") or []
    L.append(f"- **Frames:** {len(frames)} @ {ctx['fps']:.3f} fps, {mode} mode "
             f"(budget {ctx['target']}, max {ctx['max_frames']})")
    L.append(f"- **Frame size:** {ctx['resolution']}px wide")
    segs = ctx.get("transcript_segments") or []
    if segs:
        in_range = " in range" if ctx.get("focused") else ""
        L.append(f"- **Transcript:** {len(segs)} segments{in_range} "
                 f"(via {ctx.get('transcript_source') or 'captions'})")
    else:
        L.append("- **Transcript:** none available")

    if not ctx.get("focused") and full > 600:
        mins = int(full // 60)
        L += ["", (f"> **Warning:** This is a {mins}-minute video. Frame coverage is sparse at this "
                   "length — accuracy degrades on anything over 10 minutes. Re-run with "
                   "`--start HH:MM:SS --end HH:MM:SS` to zoom in, or `--scenes` for slide lectures.")]

    L += ["", "## Frames", ""]
    L.append("Frames live in: `frames/`" if relative else f"Frames live at: `{ctx.get('frames_dir', '')}`")
    L += ["", ("**Read each frame path below with the Read tool to view the image.** "
               "Frames are in chronological order; `t=MM:SS` is the absolute timestamp."), ""]
    for fr in frames:
        L.append(f"- `{_frame_ref(fr['path'], relative)}` (t={format_time(fr['timestamp_seconds'])})")

    L += ["", "## Transcript", ""]
    text = ctx.get("transcript_text")
    if text:
        label = ctx.get("transcript_source") or "captions"
        if ctx.get("focused"):
            L.append(f"_Source: {label}. Filtered to {format_time(ctx['effective_start'])} → "
                     f"{format_time(ctx['effective_end'])}:_")
        else:
            L.append(f"_Source: {label}._")
        L += ["", "```", text, "```"]
    else:
        L.append("_No transcript available — proceed with frames only._")
    return "\n".join(L)


def _render_scenes(ctx: dict, relative: bool) -> str:
    L: list[str] = ["", "# watch: lecture report (scenes)", ""]
    _header(ctx, L)
    slides = ctx.get("slides") or []
    auto = " auto" if ctx.get("auto_threshold") else ""
    L.append(f"- **Slides:** {len(slides)} detected "
             f"(threshold {ctx.get('scene_threshold')}{auto}, {ctx['resolution']}px)")
    if ctx.get("transcript_source"):
        L.append(f"- **Transcript:** via {ctx['transcript_source']}")
    else:
        L.append("- **Transcript:** none available")
    if ctx.get("merged_count"):
        L += ["", (f"> **Note:** {ctx['merged_count']} short slide(s) were merged to stay within "
                   f"the {ctx['max_frames']}-frame cap.")]

    L += ["", "## Slides", "",
          "**Read each slide image with the Read tool.** Each slide shows the frame plus the "
          "transcript spoken while it was on screen."]
    for s in slides:
        L += ["", f"### Slide {s['index']} — {format_time(s['start'])} → {format_time(s['end'])}",
              f"`{_frame_ref(s['path'], relative)}`"]
        if s.get("text"):
            L += [f"> {line}" for line in s["text"].splitlines()]
        else:
            L.append("> _(no speech in this span)_")
    return "\n".join(L)
