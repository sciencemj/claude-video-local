#!/usr/bin/env python3
"""/watch entry point: download video, extract frames (or slides), surface transcript.

Prints a markdown report to stdout listing frame paths + transcript. With
--scenes it detects slides and groups the transcript per slide. With --save it
writes a portable bundle. Claude then Reads each frame path to see the video.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

import bundle  # noqa: E402
import cache  # noqa: E402
import local_whisper  # noqa: E402
import report as report_mod  # noqa: E402
import scenes  # noqa: E402
from download import download, is_url  # noqa: E402
from frames import MAX_FPS, auto_fps, auto_fps_focus, extract, format_time, get_metadata, parse_time  # noqa: E402
from setup import local_available, venv_python  # noqa: E402
from transcribe import filter_range, format_transcript, parse_vtt  # noqa: E402
from whisper import extract_audio, load_api_key, transcribe_video  # noqa: E402


def choose_backend(requested: str | None) -> str | None:
    """Pick a Whisper backend: explicit choice, else auto (local → API)."""
    if requested in ("local", "groq", "openai"):
        return requested
    if local_available():
        return "local"
    backend, _ = load_api_key(None)
    return backend


def transcribe_local_cached(video_path: str, work: Path, model: str,
                            device: str | None, language: str | None) -> list[dict]:
    audio = extract_audio(video_path, work / "audio.mp3")
    key = cache.cache_key(audio, model, language)
    cached = cache.load(key)
    if cached is not None:
        print(f"[watch] using cached local transcript ({len(cached)} segments)", file=sys.stderr)
        return cached
    segs = local_whisper.launch_local(audio, model, device, language, venv_python())
    if not segs:
        raise SystemExit("local Whisper returned no segments")
    cache.save(key, segs, source=f"whisper (local: {model})")
    return segs


def run_whisper(video_path: str, work: Path, args) -> tuple[list[dict], str | None]:
    """Resolve precedence captions(→handled by caller) → local → API. Returns full segments."""
    model = args.whisper_model
    requested = args.whisper

    def _api(pref):
        backend, api_key = load_api_key(pref)
        if not (backend and api_key):
            return [], None
        try:
            segs, used = transcribe_video(video_path, work / "audio.mp3", backend=backend, api_key=api_key)
            return segs, f"whisper ({used})"
        except SystemExit as exc:
            print(f"[watch] whisper API failed: {exc}", file=sys.stderr)
            return [], None

    backend = choose_backend(requested)
    if backend == "local" and not local_available():
        print(f"[watch] --whisper local requested but the local venv is missing — run "
              f"`python3 {SCRIPT_DIR / 'setup.py'} --setup-local`", file=sys.stderr)
        backend = None  # fall through to API auto below

    if backend == "local":
        try:
            return transcribe_local_cached(video_path, work, model, args.whisper_device, args.whisper_language), \
                f"whisper (local: {model})"
        except SystemExit as exc:
            print(f"[watch] local whisper failed: {exc}", file=sys.stderr)
            return _api(None)
    if backend in ("groq", "openai"):
        return _api(requested if requested in ("groq", "openai") else None)

    print(f"[watch] no transcription configured — add an API key or run "
          f"`python3 {SCRIPT_DIR / 'setup.py'} --setup-local`", file=sys.stderr)
    return [], None


def _base_ctx(args, info, meta, full_duration, resolution) -> dict:
    return {
        "source": args.source,
        "title": info.get("title"),
        "uploader": info.get("uploader"),
        "full_duration": full_duration,
        "width": meta.get("width"),
        "height": meta.get("height"),
        "codec": meta.get("codec"),
        "resolution": resolution,
    }


def run_scenes(video_path, work, args, info, meta, full_duration, resolution, max_frames) -> tuple[dict, list[dict]]:
    cuts = scenes.detect_cuts(video_path, threshold=args.scene_threshold)
    slides = scenes.cuts_to_slides(cuts, full_duration, min_slide_seconds=args.min_slide_seconds)
    slides, merged = scenes.merge_to_cap(slides, max_slides=max_frames)
    if merged:
        print(f"[watch] merged {merged} short slide(s) to stay within {max_frames}-frame cap", file=sys.stderr)
    slides = scenes.extract_slide_frames(video_path, slides, work / "frames", resolution=resolution)

    segments, source = ([], None)
    if not args.no_whisper:
        segments, source = run_whisper(video_path, work, args)
    slides = scenes.group_transcript(slides, segments)

    ctx = _base_ctx(args, info, meta, full_duration, resolution)
    ctx.update({
        "mode": "scenes",
        "slides": slides,
        "scene_threshold": args.scene_threshold,
        "merged_count": merged,
        "max_frames": max_frames,
        "transcript_source": source,
    })
    return ctx, segments


def run_uniform(video_path, work, args, dl, info, meta, full_duration, resolution, max_frames) -> tuple[dict, list[dict]]:
    start_sec = parse_time(args.start)
    end_sec = parse_time(args.end)
    if start_sec is not None and start_sec < 0:
        raise SystemExit("--start must be non-negative")
    if end_sec is not None and start_sec is not None and end_sec <= start_sec:
        raise SystemExit("--end must be greater than --start")
    if full_duration > 0 and start_sec is not None and start_sec >= full_duration:
        raise SystemExit(f"--start {start_sec:.1f}s is past end of video ({full_duration:.1f}s)")

    effective_start = start_sec if start_sec is not None else 0.0
    effective_end = end_sec if end_sec is not None else full_duration
    effective_duration = max(0.0, effective_end - effective_start)
    focused = start_sec is not None or end_sec is not None

    if focused:
        fps, target = auto_fps_focus(effective_duration, max_frames=max_frames)
    else:
        fps, target = auto_fps(effective_duration, max_frames=max_frames)
    if args.fps is not None:
        fps = min(args.fps, MAX_FPS)
        target = max(1, int(round(fps * effective_duration)))

    scope = (f"{format_time(effective_start)}-{format_time(effective_end)} ({effective_duration:.1f}s)"
             if focused else f"full {effective_duration:.1f}s")
    print(f"[watch] extracting ~{target} frames at {fps:.3f} fps over {scope}…", file=sys.stderr)
    frames = extract(video_path, work / "frames", fps=fps, resolution=resolution,
                     max_frames=max_frames, start_seconds=start_sec, end_seconds=end_sec)

    segments: list[dict] = []
    source: str | None = None
    if dl.get("subtitle_path"):
        try:
            all_segments = parse_vtt(dl["subtitle_path"])
            segments = filter_range(all_segments, start_sec, end_sec) if focused else all_segments
            source = "captions"
        except Exception as exc:
            print(f"[watch] subtitle parse failed: {exc}", file=sys.stderr)
    if not segments and not args.no_whisper:
        all_segments, source = run_whisper(video_path, work, args)
        segments = filter_range(all_segments, start_sec, end_sec) if (focused and all_segments) else all_segments

    ctx = _base_ctx(args, info, meta, full_duration, resolution)
    ctx.update({
        "mode": "uniform",
        "focused": focused,
        "effective_start": effective_start,
        "effective_end": effective_end,
        "effective_duration": effective_duration,
        "fps": fps,
        "target": target,
        "max_frames": max_frames,
        "frames_dir": str(work / "frames"),
        "frames": frames,
        "transcript_segments": segments,
        "transcript_text": format_transcript(segments) if segments else None,
        "transcript_source": source,
    })
    return ctx, segments


def main() -> int:
    ap = argparse.ArgumentParser(prog="watch", description="Watch a video: frames/slides + transcript.")
    ap.add_argument("source", help="Video URL or local file path")
    ap.add_argument("--max-frames", type=int, default=100, help="Cap on frames/slides (hard max 100 in uniform)")
    ap.add_argument("--resolution", type=int, default=None, help="Frame width px (default 512; 768 with --scenes)")
    ap.add_argument("--fps", type=float, default=None, help="Override auto-fps (uniform mode)")
    ap.add_argument("--start", type=str, default=None, help="Range start (SS, MM:SS, HH:MM:SS) — uniform mode")
    ap.add_argument("--end", type=str, default=None, help="Range end — uniform mode")
    ap.add_argument("--out-dir", type=str, default=None, help="Working directory (default: tmp)")
    ap.add_argument("--save", type=str, default=None, help="Write a portable bundle to this dir")
    ap.add_argument("--scenes", action="store_true", help="Slide mode: one frame per detected slide + grouped transcript")
    ap.add_argument("--scene-threshold", type=float, default=0.3, help="Scene-cut sensitivity (default 0.3)")
    ap.add_argument("--min-slide-seconds", type=float, default=3.0, help="Merge slides shorter than this")
    ap.add_argument("--no-whisper", action="store_true", help="Disable Whisper fallback")
    ap.add_argument("--whisper", choices=["local", "groq", "openai"], default=None,
                    help="Force a backend. Default: captions → local → API.")
    ap.add_argument("--whisper-model", default="turbo", help="Local Whisper model (default turbo)")
    ap.add_argument("--whisper-device", choices=["cuda", "mps", "cpu"], default=None, help="Local Whisper device")
    ap.add_argument("--whisper-language", default=None, help="Local Whisper language code (e.g. en, ko)")
    args = ap.parse_args()

    max_frames = min(args.max_frames, 100)
    resolution = args.resolution if args.resolution is not None else (768 if args.scenes else 512)

    if args.save:
        work = Path(args.save).expanduser().resolve()
    elif args.out_dir:
        work = Path(args.out_dir).expanduser().resolve()
    else:
        work = Path(tempfile.mkdtemp(prefix="watch-"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"[watch] working dir: {work}", file=sys.stderr)

    print("[watch] downloading via yt-dlp…" if is_url(args.source) else "[watch] using local file…", file=sys.stderr)
    dl = download(args.source, work / "download")
    video_path = dl["video_path"]
    meta = get_metadata(video_path)
    full_duration = meta["duration_seconds"]
    info = dl.get("info") or {}

    if args.scenes:
        if args.start or args.end:
            print("[watch] note: --start/--end are ignored in --scenes mode (full video)", file=sys.stderr)
        ctx, segments = run_scenes(video_path, work, args, info, meta, full_duration, resolution, max_frames)
    else:
        ctx, segments = run_uniform(video_path, work, args, dl, info, meta, full_duration, resolution, max_frames)

    print(report_mod.render(ctx, relative=False))

    if args.save:
        bundle.write_bundle(work, ctx, segments)
        print()
        print(f"_Bundle saved to `{work}` — convey it with `/watch-load {work}` (Claude Code) "
              f"or upload `report.md` + `frames/` to Claude.ai._")
    else:
        print()
        print("---")
        print(f"_Work dir: `{work}` — delete when done._")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
