# Local Whisper + Scene Mode + Portable Bundles — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add local-Whisper transcription, a scene/slide mode for long lecture videos, and portable save/load bundles to the `/watch` skill.

**Architecture:** Three composable additions to the existing `download → frames → transcript → report` pipeline. (1) A vendored local-Whisper runner executed through a `uv`-managed Python-3.11 venv (keeps the heavy torch deps out of the stdlib core). (2) An ffmpeg scene-detection module that produces one frame per slide plus a per-slide transcript grouping. (3) A bundle writer/reader so a Claude Code run can be conveyed to Claude.ai or another session. Report rendering is extracted into a pure `report.py` so stdout and the saved `report.md` share one code path.

**Tech Stack:** Python 3 (stdlib only for the core; `unittest` for tests), `ffmpeg`/`ffprobe`, `yt-dlp`, `uv` (venv management), `openai-whisper` (inside the venv only).

---

## Spec

Implements `docs/superpowers/specs/2026-06-14-local-whisper-and-scene-mode-design.md`.

## File structure & interfaces

**New files**

| File | Responsibility | Key functions |
|---|---|---|
| `scripts/scenes.py` | Slide detection + per-slide framing/grouping | `parse_scene_times`, `detect_cuts`, `cuts_to_slides`, `merge_to_cap`, `representative_timestamp`, `group_transcript`, `extract_slide_frames` |
| `scripts/cache.py` | Local-transcript cache | `cache_dir`, `cache_key`, `load`, `save` |
| `scripts/local_whisper.py` | Local Whisper (venv-side script + system-side launcher) | `launch_local` (system), `_pick_device`/`_transcribe`/`main` (venv) |
| `scripts/report.py` | Pure markdown rendering for both modes | `render`, `_render_uniform`, `_render_scenes`, `_frame_ref` |
| `scripts/bundle.py` | Write a portable bundle | `build_meta`, `write_bundle` |
| `scripts/load_bundle.py` | Re-ingest a bundle | `resolve_report`, `main` |
| `commands/watch-save.md` | `/watch-save` slash command | — |
| `commands/watch-load.md` | `/watch-load` slash command | — |
| `tests/test_scenes.py` … | unittest suites | — |

**Modified files**

| File | Change |
|---|---|
| `scripts/setup.py` | venv helpers (`whisper_venv_dir`, `venv_python`, `local_available`), `--setup-local`, `_status`/`cmd_check`/`cmd_json` account for local. |
| `scripts/watch.py` | new flags, precedence (captions → local → API), scene branch, `--save`, render via `report.py`. Full replacement provided in Task 10. |
| `SKILL.md`, `README.md` | document the three features. |

**Cross-module contracts (use these signatures verbatim):**

```python
# scenes.py
parse_scene_times(stderr: str) -> list[float]
detect_cuts(video_path: str, threshold: float = 0.3) -> list[float]      # includes 0.0
cuts_to_slides(cuts: list[float], duration: float, min_slide_seconds: float = 3.0) -> list[dict]  # [{index,start,end}]
merge_to_cap(slides: list[dict], max_slides: int) -> tuple[list[dict], int]   # (slides, merged_count)
representative_timestamp(slide: dict, offset: float = 0.5) -> float
group_transcript(slides: list[dict], segments: list[dict]) -> list[dict]       # adds 'segments','text'
extract_slide_frames(video_path: str, slides: list[dict], out_dir, resolution: int = 768) -> list[dict]  # adds 'path','timestamp_seconds'

# cache.py
cache_dir(base=None) -> Path
cache_key(audio_path, model: str, language: str | None) -> str
load(key: str, base=None) -> list[dict] | None
save(key: str, segments: list[dict], source: str, base=None) -> None

# local_whisper.py
launch_local(audio_path, model: str, device: str | None, language: str | None, venv_python) -> list[dict]

# setup.py
whisper_venv_dir() -> Path
venv_python(venv=None) -> Path
local_available(venv=None) -> bool

# report.py
render(ctx: dict, relative: bool = False) -> str

# bundle.py
build_meta(ctx: dict) -> dict
write_bundle(work, ctx: dict, segments: list[dict]) -> None

# load_bundle.py
resolve_report(report_md: str, base_dir) -> str
```

The `ctx` dict passed to `report`/`bundle` has keys: `source, title, uploader, full_duration, focused, effective_start, effective_end, effective_duration, width, height, codec, mode ('uniform'|'scenes'), fps, target, max_frames, resolution, frames, slides, transcript_text, transcript_source, transcript_segments, scene_threshold, merged_count`.

**Running tests:** `python3 -m unittest discover -s tests -v` (from repo root).

---

## Task 1: scenes.py — slide math (pure)

**Files:**
- Create: `tests/test_scenes.py`
- Create: `scripts/scenes.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scenes.py
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import scenes  # noqa: E402


class SlideMathTests(unittest.TestCase):
    def test_cuts_to_slides_basic_spans(self):
        slides = scenes.cuts_to_slides([0.0, 30.0, 90.0], duration=120.0, min_slide_seconds=3.0)
        self.assertEqual([(s["start"], s["end"]) for s in slides],
                         [(0.0, 30.0), (30.0, 90.0), (90.0, 120.0)])
        self.assertEqual([s["index"] for s in slides], [1, 2, 3])

    def test_cuts_to_slides_prepends_zero(self):
        slides = scenes.cuts_to_slides([10.0, 20.0], duration=30.0)
        self.assertEqual(slides[0]["start"], 0.0)

    def test_cuts_to_slides_merges_short(self):
        # 0-1 (short) merges forward into the next span
        slides = scenes.cuts_to_slides([0.0, 1.0, 40.0], duration=80.0, min_slide_seconds=3.0)
        self.assertEqual([(s["start"], s["end"]) for s in slides], [(0.0, 40.0), (40.0, 80.0)])

    def test_merge_to_cap_reduces_count(self):
        slides = [{"index": i + 1, "start": float(i), "end": float(i) + 1.0} for i in range(5)]
        capped, merged = scenes.merge_to_cap(slides, max_slides=3)
        self.assertEqual(len(capped), 3)
        self.assertEqual(merged, 2)
        self.assertEqual(capped[-1]["end"], 5.0)  # coverage preserved end-to-end
        self.assertEqual([s["index"] for s in capped], [1, 2, 3])

    def test_representative_timestamp_offsets(self):
        self.assertEqual(scenes.representative_timestamp({"start": 10.0, "end": 40.0}), 10.5)

    def test_representative_timestamp_clamps_short_slide(self):
        t = scenes.representative_timestamp({"start": 10.0, "end": 10.2})
        self.assertTrue(10.0 <= t < 10.2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_scenes -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scenes'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/scenes.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_scenes -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_scenes.py scripts/scenes.py
git commit -m "feat(scenes): slide-span math (cuts→slides, cap-merge, rep timestamp)"
```

---

## Task 2: scenes.py — scene parsing + transcript grouping (pure)

**Files:**
- Modify: `tests/test_scenes.py`
- Modify: `scripts/scenes.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_scenes.py` (inside the file, new class):

```python
class SceneParseAndGroupTests(unittest.TestCase):
    def test_parse_scene_times(self):
        stderr = (
            "[Parsed_showinfo_1 @ 0x] n:0 pts:0 pts_time:0 ...\n"
            "[Parsed_showinfo_1 @ 0x] n:1 pts:720 pts_time:30.03 ...\n"
            "[Parsed_showinfo_1 @ 0x] n:2 pts:2160 pts_time:90.5 ...\n"
        )
        self.assertEqual(scenes.parse_scene_times(stderr), [0.0, 30.03, 90.5])

    def test_group_transcript_assigns_overlapping_segments(self):
        segments = [
            {"start": 0.0, "end": 10.0, "text": "intro"},
            {"start": 12.0, "end": 20.0, "text": "topic one"},
            {"start": 95.0, "end": 100.0, "text": "topic two"},
        ]
        slides = [{"index": 1, "start": 0.0, "end": 90.0},
                  {"index": 2, "start": 90.0, "end": 120.0}]
        grouped = scenes.group_transcript(slides, segments)
        self.assertEqual([s["text"] for s in grouped[0]["segments"]], ["intro", "topic one"])
        self.assertEqual([s["text"] for s in grouped[1]["segments"]], ["topic two"])
        self.assertIn("intro", grouped[0]["text"])
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m unittest tests.test_scenes -v`
Expected: FAIL — `AttributeError: module 'scenes' has no attribute 'parse_scene_times'`.

- [ ] **Step 3: Implement**

Append to `scripts/scenes.py`:

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m unittest tests.test_scenes -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_scenes.py scripts/scenes.py
git commit -m "feat(scenes): parse scene timestamps + group transcript per slide"
```

---

## Task 3: scenes.py — ffmpeg integration (detect_cuts, extract_slide_frames)

These shell out to ffmpeg, so they are verified manually, not in unit tests.

**Files:**
- Modify: `scripts/scenes.py`

- [ ] **Step 1: Implement the ffmpeg functions**

Append to `scripts/scenes.py`:

```python
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
```

- [ ] **Step 2: Manual verification (skip if no sample video handy)**

If you have any short video at `/tmp/sample.mp4`:

```bash
python3 - <<'PY'
import sys; sys.path.insert(0, "scripts")
import scenes
cuts = scenes.detect_cuts("/tmp/sample.mp4", threshold=0.3)
slides = scenes.cuts_to_slides(cuts, duration=9999.0)
slides, merged = scenes.merge_to_cap(slides, max_slides=50)
slides = scenes.extract_slide_frames("/tmp/sample.mp4", slides, "/tmp/slides_out", resolution=768)
print("slides:", len(slides), "merged:", merged, "first frame:", slides[0]["path"] if slides else None)
PY
ls -la /tmp/slides_out
```

Expected: prints a slide count and writes `slide_0001.jpg …` into `/tmp/slides_out`.

- [ ] **Step 3: Run the existing unit tests to confirm no regression**

Run: `python3 -m unittest tests.test_scenes -v`
Expected: PASS (still 9 tests; ffmpeg fns untested by design).

- [ ] **Step 4: Commit**

```bash
git add scripts/scenes.py
git commit -m "feat(scenes): ffmpeg cut detection + per-slide frame extraction"
```

---

## Task 4: cache.py — local transcript cache (pure)

**Files:**
- Create: `tests/test_cache.py`
- Create: `scripts/cache.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_cache.py
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import cache  # noqa: E402


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.audio = self.tmp / "a.mp3"
        self.audio.write_bytes(b"fake audio bytes")

    def test_key_is_deterministic_and_param_sensitive(self):
        k1 = cache.cache_key(self.audio, "turbo", "en")
        k2 = cache.cache_key(self.audio, "turbo", "en")
        k3 = cache.cache_key(self.audio, "small", "en")
        k4 = cache.cache_key(self.audio, "turbo", None)
        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, k3)
        self.assertNotEqual(k1, k4)

    def test_load_miss_returns_none(self):
        self.assertIsNone(cache.load("nope", base=self.tmp))

    def test_save_then_load_roundtrips(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "hi"}]
        key = cache.cache_key(self.audio, "turbo", "en")
        cache.save(key, segs, source="whisper (local: turbo)", base=self.tmp)
        self.assertEqual(cache.load(key, base=self.tmp), segs)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m unittest tests.test_cache -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cache'`.

- [ ] **Step 3: Implement**

```python
# scripts/cache.py
#!/usr/bin/env python3
"""Content-addressed cache for local-Whisper transcripts.

Keyed on the extracted audio bytes + model + language, so re-running `--scenes`
at a different threshold does not re-transcribe a 2-hour video.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

DEFAULT_BASE = Path.home() / ".config" / "watch"


def cache_dir(base: Path | None = None) -> Path:
    return (Path(base) if base else DEFAULT_BASE) / "cache"


def cache_key(audio_path, model: str, language: str | None) -> str:
    h = hashlib.sha256()
    with open(audio_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    h.update(f"|model={model}|lang={language or ''}".encode())
    return h.hexdigest()


def load(key: str, base: Path | None = None) -> list[dict] | None:
    path = cache_dir(base) / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("segments")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def save(key: str, segments: list[dict], source: str, base: Path | None = None) -> None:
    directory = cache_dir(base)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{key}.json").write_text(
        json.dumps({"source": source, "segments": segments}, ensure_ascii=False),
        encoding="utf-8",
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m unittest tests.test_cache -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_cache.py scripts/cache.py
git commit -m "feat(cache): content-addressed local transcript cache"
```

---

## Task 5: setup.py — venv helpers, local-aware status, --setup-local

**Files:**
- Create: `tests/test_setup_local.py`
- Modify: `scripts/setup.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_setup_local.py
import os
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import setup  # noqa: E402


class VenvHelperTests(unittest.TestCase):
    def test_venv_python_path_shape(self):
        venv = Path("/tmp/some-venv")
        py = setup.venv_python(venv)
        if os.name == "nt":
            self.assertEqual(py, venv / "Scripts" / "python.exe")
        else:
            self.assertEqual(py, venv / "bin" / "python")

    def test_local_available_false_when_missing(self):
        self.assertFalse(setup.local_available(Path(tempfile.mkdtemp()) / "novenv"))

    def test_local_available_true_when_python_present(self):
        venv = Path(tempfile.mkdtemp()) / "venv"
        py = setup.venv_python(venv)
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_text("#!/bin/sh\n")
        self.assertTrue(setup.local_available(venv))

    def test_status_reports_local_available_key(self):
        self.assertIn("local_available", setup._status())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m unittest tests.test_setup_local -v`
Expected: FAIL — `AttributeError: module 'setup' has no attribute 'venv_python'`.

- [ ] **Step 3: Implement — add venv helpers**

In `scripts/setup.py`, after the `CONFIG_FILE = ...` line (near line 32), add:

```python
WHISPER_VENV = CONFIG_DIR / "whisper-venv"
LOCAL_WHISPER_PYTHON = "3.11"  # openai-whisper / torch wheels are reliable here
```

After `def _which(...)` (near line 53), add:

```python
def whisper_venv_dir() -> Path:
    return WHISPER_VENV


def venv_python(venv: Path | None = None) -> Path:
    venv = Path(venv) if venv else WHISPER_VENV
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def local_available(venv: Path | None = None) -> bool:
    return venv_python(venv).exists()
```

- [ ] **Step 4: Implement — make status local-aware**

Replace the body of `_status()` (lines 199-221) with:

```python
def _status() -> dict:
    """Structured preflight snapshot."""
    missing = _check_binaries()
    has_key, backend = _have_api_key()
    local = local_available()
    transcription_ok = has_key or local

    if not missing and transcription_ok:
        status = "ready"
    elif missing and not transcription_ok:
        status = "needs_install_and_key"
    elif missing:
        status = "needs_install"
    else:
        status = "needs_key"

    return {
        "status": status,
        "first_run": is_first_run(),
        "missing_binaries": missing,
        "whisper_backend": backend,
        "has_api_key": has_key,
        "local_available": local,
        "config_file": str(CONFIG_FILE),
        "platform": platform.system(),
    }
```

Then in `cmd_check()` (lines 224-253), replace the `if not s["has_api_key"]:` block so the message mentions local. Change:

```python
    if not s["has_api_key"]:
        parts.append("no Whisper API key (GROQ_API_KEY or OPENAI_API_KEY)")
```

to:

```python
    if not s["has_api_key"] and not s["local_available"]:
        parts.append("no transcription configured (API key, or local venv via --setup-local)")
```

and change the trailing exit-code logic:

```python
    if s["missing_binaries"] and not s["has_api_key"]:
        return 4
    if s["missing_binaries"]:
        return 2
    return 3
```

to:

```python
    transcription_ok = s["has_api_key"] or s["local_available"]
    if s["missing_binaries"] and not transcription_ok:
        return 4
    if s["missing_binaries"]:
        return 2
    return 3
```

- [ ] **Step 5: Implement — `--setup-local` builder**

Add this function before `def main()` (near line 315):

```python
def cmd_setup_local() -> int:
    """Build/repair the local-Whisper venv (uv-preferred, Python 3.11, openai-whisper)."""
    venv = whisper_venv_dir()
    py = venv_python(venv)
    if py.exists():
        print(f"[setup] local Whisper venv already present: {venv}")
        return 0

    venv.parent.mkdir(parents=True, exist_ok=True)
    if _which("uv"):
        print(f"[setup] creating venv with uv (Python {LOCAL_WHISPER_PYTHON})…", file=sys.stderr)
        if subprocess.run(["uv", "venv", "--python", LOCAL_WHISPER_PYTHON, str(venv)]).returncode != 0:
            print("[setup] `uv venv` failed.", file=sys.stderr)
            return 2
        print("[setup] installing openai-whisper (this downloads ~GB)…", file=sys.stderr)
        if subprocess.run(["uv", "pip", "install", "--python", str(py), "openai-whisper"]).returncode != 0:
            print("[setup] `uv pip install openai-whisper` failed.", file=sys.stderr)
            return 2
    else:
        py311 = _which(f"python{LOCAL_WHISPER_PYTHON}") or _which("python3.11")
        if not py311:
            print(
                f"[setup] need Python {LOCAL_WHISPER_PYTHON} or `uv`. Install uv "
                "(https://docs.astral.sh/uv/) or python3.11, then re-run --setup-local.",
                file=sys.stderr,
            )
            return 2
        print(f"[setup] creating venv with {py311}…", file=sys.stderr)
        if subprocess.run([py311, "-m", "venv", str(venv)]).returncode != 0:
            return 2
        subprocess.run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
        print("[setup] installing openai-whisper (this downloads ~GB)…", file=sys.stderr)
        if subprocess.run([str(py), "-m", "pip", "install", "openai-whisper"]).returncode != 0:
            return 2

    check = subprocess.run([str(py), "-c", "import whisper"], capture_output=True, text=True)
    if check.returncode != 0:
        print(f"[setup] venv built but `import whisper` failed: {check.stderr.strip()}", file=sys.stderr)
        return 2
    print(f"[setup] local Whisper ready: {venv}")
    print("[setup] model weights (turbo ≈ 1.5 GB) download on first transcription.")
    return 0
```

Wire it into `main()` — change:

```python
        if arg == "--json":
            return cmd_json()
```

to:

```python
        if arg == "--json":
            return cmd_json()
        if arg == "--setup-local":
            return cmd_setup_local()
```

Also extend the interactive installer's no-key message in `cmd_install()` (after line 311, before `return 3`) — add a line offering local:

```python
    print("")
    print("  Or run local Whisper (no key, no length limit; downloads ~GB):")
    print(f"    python3 {Path(__file__).resolve()} --setup-local")
```

- [ ] **Step 6: Run tests + smoke the help**

Run: `python3 -m unittest tests.test_setup_local -v`
Expected: PASS (4 tests).

Run: `python3 scripts/setup.py --json`
Expected: JSON now includes `"local_available": false` (assuming venv not built).

- [ ] **Step 7: Commit**

```bash
git add tests/test_setup_local.py scripts/setup.py
git commit -m "feat(setup): local Whisper venv helpers, --setup-local, local-aware status"
```

---

## Task 6: local_whisper.py — venv-side transcribe + system-side launcher

The venv-side path needs torch/whisper and is verified manually. The system-side
`launch_local` is stdlib (subprocess + JSON) and gets a light test with a fake venv python.

**Files:**
- Create: `tests/test_local_whisper.py`
- Create: `scripts/local_whisper.py`

- [ ] **Step 1: Write failing test (system-side launcher with a fake interpreter)**

```python
# tests/test_local_whisper.py
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import local_whisper  # noqa: E402


class LaunchLocalTests(unittest.TestCase):
    def _fake_python(self, body: str) -> Path:
        d = Path(tempfile.mkdtemp())
        fake = d / "python"
        fake.write_text("#!/bin/sh\n" + body + "\n")
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        return fake

    @unittest.skipIf(os.name == "nt", "shell-script fake interpreter is POSIX-only")
    def test_launch_local_parses_json_segments(self):
        payload = json.dumps({"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]})
        fake = self._fake_python(f"cat <<'EOF'\n{payload}\nEOF")
        segs = local_whisper.launch_local(Path("/tmp/a.mp3"), "turbo", None, None, fake)
        self.assertEqual(segs, [{"start": 0.0, "end": 1.0, "text": "hi"}])

    @unittest.skipIf(os.name == "nt", "shell-script fake interpreter is POSIX-only")
    def test_launch_local_raises_on_nonzero_exit(self):
        fake = self._fake_python("echo boom 1>&2\nexit 3")
        with self.assertRaises(SystemExit):
            local_whisper.launch_local(Path("/tmp/a.mp3"), "turbo", None, None, fake)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m unittest tests.test_local_whisper -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'local_whisper'`.

- [ ] **Step 3: Implement**

```python
# scripts/local_whisper.py
#!/usr/bin/env python3
"""Local OpenAI-Whisper transcription.

Two roles in one file:
  - SYSTEM side (`launch_local`): stdlib only. Spawns the venv's Python to run
    this same file's `main()` and parses the JSON segments it prints.
  - VENV side (`main`): runs *inside* ~/.config/watch/whisper-venv where torch
    and openai-whisper are installed. Lazily imports them, transcribes, and
    prints {"segments": [...], "device": "..."} as JSON to stdout.

Top-level imports stay stdlib so the system Python can import this module without
torch installed. Heavy imports live inside the venv-side functions.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


# ----- system side (stdlib only) -----

def launch_local(audio_path, model: str, device: str | None, language: str | None, venv_python) -> list[dict]:
    """Run local Whisper through the venv interpreter; return {start,end,text} segments."""
    cmd = [str(venv_python), str(Path(__file__).resolve()), str(audio_path), "--model", model]
    if device:
        cmd += ["--device", device]
    if language:
        cmd += ["--language", language]
    # stderr streams live (progress); stdout is captured for the JSON payload.
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"local Whisper failed (exit {proc.returncode}) — see log above")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"local Whisper returned non-JSON: {exc}: {proc.stdout[:200]}")
    return data.get("segments") or []


# ----- venv side (torch/whisper, imported lazily) -----

def _pick_device(requested: str | None) -> str:
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _transcribe(audio: str, model_name: str, device: str, language: str | None) -> list[dict]:
    import os
    import whisper

    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    print(f"[watch] loading local Whisper '{model_name}' on {device}…", file=sys.stderr)
    model = whisper.load_model(model_name, device=device)
    print(f"[watch] transcribing {Path(audio).name}…", file=sys.stderr)
    result = model.transcribe(str(audio), language=language, verbose=False)

    segs: list[dict] = []
    for s in result.get("segments") or []:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        segs.append({"start": round(float(s.get("start") or 0.0), 2),
                     "end": round(float(s.get("end") or 0.0), 2),
                     "text": text})
    if not segs:
        full = (result.get("text") or "").strip()
        if full:
            segs = [{"start": 0.0, "end": 0.0, "text": full}]
    return segs


def main() -> int:
    ap = argparse.ArgumentParser(prog="local_whisper")
    ap.add_argument("audio")
    ap.add_argument("--model", default="turbo")
    ap.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    ap.add_argument("--language", default=None)
    args = ap.parse_args()

    device = _pick_device(args.device)
    segments = _transcribe(args.audio, args.model, device, args.language)
    json.dump({"segments": segments, "device": device}, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m unittest tests.test_local_whisper -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Manual smoke (only if you've run `--setup-local`)**

```bash
python3 scripts/setup.py --setup-local        # one-time, downloads ~GB
ffmpeg -y -i /tmp/sample.mp4 -vn -ar 16000 -ac 1 -b:a 64k /tmp/a.mp3
~/.config/watch/whisper-venv/bin/python scripts/local_whisper.py /tmp/a.mp3 --model tiny
```

Expected: a JSON object with a `segments` array on stdout.

- [ ] **Step 6: Commit**

```bash
git add tests/test_local_whisper.py scripts/local_whisper.py
git commit -m "feat(local-whisper): vendored local transcription via venv subprocess"
```

---

## Task 7: report.py — pure markdown rendering (uniform + scenes)

**Files:**
- Create: `tests/test_report.py`
- Create: `scripts/report.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_report.py
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import report  # noqa: E402


def uniform_ctx(**over):
    ctx = {
        "mode": "uniform", "source": "video.mp4", "title": "T", "uploader": None,
        "full_duration": 120.0, "focused": False,
        "effective_start": 0.0, "effective_end": 120.0, "effective_duration": 120.0,
        "width": 1280, "height": 720, "codec": "h264",
        "fps": 0.5, "target": 60, "max_frames": 100, "resolution": 512,
        "frames": [{"path": "/tmp/work/frames/frame_0001.jpg", "timestamp_seconds": 0.0}],
        "transcript_text": "[00:00] hi", "transcript_source": "captions",
        "transcript_segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
    }
    ctx.update(over)
    return ctx


def scene_ctx(**over):
    ctx = {
        "mode": "scenes", "source": "lecture.mp4", "title": "Lec", "uploader": None,
        "full_duration": 3600.0, "width": 1920, "height": 1080, "codec": "h264",
        "resolution": 768, "scene_threshold": 0.3, "merged_count": 0,
        "max_frames": 100, "transcript_source": "whisper (local: turbo)",
        "slides": [{"index": 1, "start": 0.0, "end": 90.0,
                    "path": "/tmp/work/frames/slide_0001.jpg",
                    "text": "[00:00] welcome", "segments": []}],
    }
    ctx.update(over)
    return ctx


class ReportTests(unittest.TestCase):
    def test_uniform_absolute_paths(self):
        out = report.render(uniform_ctx(), relative=False)
        self.assertIn("# watch: video report", out)
        self.assertIn("`/tmp/work/frames/frame_0001.jpg` (t=00:00)", out)

    def test_uniform_relative_paths(self):
        out = report.render(uniform_ctx(), relative=True)
        self.assertIn("`frames/frame_0001.jpg` (t=00:00)", out)
        self.assertNotIn("/tmp/work", out)

    def test_uniform_long_video_warning(self):
        out = report.render(uniform_ctx(full_duration=1800.0), relative=False)
        self.assertIn("Warning:", out)
        self.assertIn("--scenes", out)

    def test_scenes_structure_and_grouping(self):
        out = report.render(scene_ctx(), relative=False)
        self.assertIn("# watch: lecture report (scenes)", out)
        self.assertIn("### Slide 1 — 00:00 → 01:30", out)
        self.assertIn("`/tmp/work/frames/slide_0001.jpg`", out)
        self.assertIn("> [00:00] welcome", out)

    def test_scenes_relative_paths(self):
        out = report.render(scene_ctx(), relative=True)
        self.assertIn("`frames/slide_0001.jpg`", out)
        self.assertNotIn("/tmp/work", out)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m unittest tests.test_report -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'report'`.

- [ ] **Step 3: Implement**

```python
# scripts/report.py
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
    L.append(f"- **Slides:** {len(slides)} detected "
             f"(threshold {ctx.get('scene_threshold')}, {ctx['resolution']}px)")
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
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m unittest tests.test_report -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_report.py scripts/report.py
git commit -m "feat(report): pure renderer for uniform + scene reports"
```

---

## Task 8: bundle.py — write a portable bundle

**Files:**
- Create: `tests/test_bundle.py`
- Create: `scripts/bundle.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_bundle.py
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import bundle  # noqa: E402


def scene_ctx():
    return {
        "mode": "scenes", "source": "lecture.mp4", "title": "Lec", "uploader": None,
        "full_duration": 3600.0, "width": 1920, "height": 1080, "codec": "h264",
        "resolution": 768, "scene_threshold": 0.3, "merged_count": 0, "max_frames": 100,
        "transcript_source": "whisper (local: turbo)",
        "slides": [{"index": 1, "start": 0.0, "end": 90.0,
                    "path": "/abs/work/frames/slide_0001.jpg",
                    "text": "[00:00] welcome", "segments": []}],
    }


class BundleTests(unittest.TestCase):
    def test_build_meta(self):
        meta = bundle.build_meta(scene_ctx())
        self.assertEqual(meta["schema"], 1)
        self.assertEqual(meta["mode"], "scenes")
        self.assertEqual(meta["frame_count"], 1)
        self.assertEqual(meta["transcript_source"], "whisper (local: turbo)")

    def test_write_bundle_creates_files_with_relative_paths(self):
        work = Path(tempfile.mkdtemp())
        segs = [{"start": 0.0, "end": 1.0, "text": "welcome"}]
        bundle.write_bundle(work, scene_ctx(), segs)
        self.assertTrue((work / "report.md").exists())
        self.assertTrue((work / "transcript.json").exists())
        self.assertTrue((work / "transcript.txt").exists())
        self.assertTrue((work / "meta.json").exists())
        report_text = (work / "report.md").read_text()
        self.assertIn("`frames/slide_0001.jpg`", report_text)
        self.assertNotIn("/abs/work", report_text)
        self.assertEqual(json.loads((work / "transcript.json").read_text())["segments"], segs)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m unittest tests.test_bundle -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bundle'`.

- [ ] **Step 3: Implement**

```python
# scripts/bundle.py
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
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m unittest tests.test_bundle -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_bundle.py scripts/bundle.py
git commit -m "feat(bundle): write portable report.md + transcript + meta"
```

---

## Task 9: load_bundle.py — re-ingest a bundle

**Files:**
- Create: `tests/test_load_bundle.py`
- Create: `scripts/load_bundle.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_load_bundle.py
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import load_bundle  # noqa: E402


class ResolveReportTests(unittest.TestCase):
    def test_relative_frame_links_become_absolute(self):
        md = "### Slide 1\n`frames/slide_0001.jpg`\n> [00:00] hi\n"
        out = load_bundle.resolve_report(md, "/abs/bundle")
        self.assertIn("`/abs/bundle/frames/slide_0001.jpg`", out)
        self.assertNotIn("`frames/slide_0001.jpg`", out)

    def test_non_frame_backticks_untouched(self):
        md = "use `--scenes` here and `frames/a.jpg`"
        out = load_bundle.resolve_report(md, "/abs/bundle")
        self.assertIn("`--scenes`", out)
        self.assertIn("`/abs/bundle/frames/a.jpg`", out)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m unittest tests.test_load_bundle -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'load_bundle'`.

- [ ] **Step 3: Implement**

```python
# scripts/load_bundle.py
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
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m unittest tests.test_load_bundle -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_load_bundle.py scripts/load_bundle.py
git commit -m "feat(load-bundle): resolve + re-emit a saved bundle"
```

---

## Task 10: watch.py — wire flags, precedence, scene mode, --save

Full replacement of `scripts/watch.py`. Includes a `choose_backend` helper that
gets a focused unit test.

**Files:**
- Create: `tests/test_watch_backend.py`
- Modify (replace): `scripts/watch.py`

- [ ] **Step 1: Write failing test for backend precedence**

```python
# tests/test_watch_backend.py
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import watch  # noqa: E402


class ChooseBackendTests(unittest.TestCase):
    def test_explicit_choice_wins(self):
        self.assertEqual(watch.choose_backend("groq"), "groq")
        self.assertEqual(watch.choose_backend("local"), "local")
        self.assertEqual(watch.choose_backend("openai"), "openai")

    def test_auto_prefers_local_when_available(self):
        with mock.patch.object(watch, "local_available", return_value=True):
            self.assertEqual(watch.choose_backend(None), "local")

    def test_auto_falls_back_to_api_key(self):
        with mock.patch.object(watch, "local_available", return_value=False), \
             mock.patch.object(watch, "load_api_key", return_value=("groq", "k")):
            self.assertEqual(watch.choose_backend(None), "groq")

    def test_auto_returns_none_when_nothing(self):
        with mock.patch.object(watch, "local_available", return_value=False), \
             mock.patch.object(watch, "load_api_key", return_value=(None, None)):
            self.assertIsNone(watch.choose_backend(None))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m unittest tests.test_watch_backend -v`
Expected: FAIL — `AttributeError: module 'watch' has no attribute 'choose_backend'`.

- [ ] **Step 3: Replace `scripts/watch.py` with:**

```python
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
```

- [ ] **Step 4: Run the backend test to verify pass**

Run: `python3 -m unittest tests.test_watch_backend -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Smoke the CLI surface (no network/model needed)**

Run: `python3 scripts/watch.py --help`
Expected: help text shows `--scenes`, `--save`, `--whisper {local,groq,openai}`, `--whisper-model`, etc.

- [ ] **Step 6: Run the full suite**

Run: `python3 -m unittest discover -s tests -v`
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/test_watch_backend.py scripts/watch.py
git commit -m "feat(watch): local/scene/save flags, captions→local→API precedence"
```

---

## Task 11: Slash commands — /watch-save and /watch-load

**Files:**
- Create: `commands/watch-save.md`
- Create: `commands/watch-load.md`

- [ ] **Step 1: Create `commands/watch-save.md`**

```markdown
---
description: Watch a video and save a portable bundle (slides + transcript) to convey to another session (Claude.ai or a fresh Claude Code session).
argument-hint: <video-url-or-path> <bundle-dir> [flags]
allowed-tools: [Bash, Read, AskUserQuestion]
---

Run the `watch` skill (SKILL.md) on the user's video, writing a portable bundle.

Arguments: $ARGUMENTS — parse a video source, a destination bundle directory, and any flags.

Steps:
1. Run the preflight check (`scripts/setup.py --check`) as in SKILL.md.
2. Run: `python3 "${CLAUDE_PLUGIN_ROOT}/scripts/watch.py" "<source>" --save "<bundle-dir>" [flags]`.
   For lecture videos, pass `--scenes`; local Whisper is used automatically once the venv exists.
3. Read the frames listed in the printed report and answer any question, as usual.
4. Tell the user how to convey the bundle:
   - **To another Claude Code session:** `/watch-load <bundle-dir>`.
   - **To Claude.ai:** upload `<bundle-dir>/report.md` and the images in `<bundle-dir>/frames/`.
     For a lighter transfer, upload `report.md` / `transcript.txt` alone (transcript-only).

Pass `--save` from the start when sharing is the goal — re-running later re-decodes
the video (the transcript itself is cached).
```

- [ ] **Step 2: Create `commands/watch-load.md`**

```markdown
---
description: Load a previously saved /watch bundle into this session (frames + grouped transcript) without reprocessing.
argument-hint: <bundle-dir>
allowed-tools: [Bash, Read]
---

Re-ingest a saved `/watch` bundle so you can answer questions about the video
without re-downloading or re-transcribing.

Argument: $ARGUMENTS — a bundle directory created by `/watch-save` or `watch.py --save`.

Steps:
1. Run: `python3 "${CLAUDE_PLUGIN_ROOT}/scripts/load_bundle.py" "<bundle-dir>"`.
   It prints the report (frame links resolved to absolute paths) and a flat list
   of every frame path.
2. Read each frame path with the Read tool (parallel reads) to see the slides.
3. Combine frames + transcript to answer the user's question.
```

- [ ] **Step 3: Verify load command path works against a real bundle**

If you produced a bundle in Task 10 (e.g. via a smoke run with `--save /tmp/bundle`):

Run: `python3 scripts/load_bundle.py /tmp/bundle`
Expected: prints the report with absolute frame paths + a `## Frame paths` list. (Skip if no bundle yet.)

- [ ] **Step 4: Commit**

```bash
git add commands/watch-save.md commands/watch-load.md
git commit -m "feat(commands): /watch-save and /watch-load subskills"
```

---

## Task 12: Documentation — SKILL.md and README.md

**Files:**
- Modify: `SKILL.md`
- Modify: `README.md`

- [ ] **Step 1: Update `SKILL.md`**

In the Step 0 exit-code table, change the row for exit `3`:

```
| `3` | No transcription configured | Offer: API key, OR `python3 "${CLAUDE_SKILL_DIR}/scripts/setup.py" --setup-local` (local Whisper, no key, no length limit) |
```

In the "Optional flags" list under Step 2, add:

```
- `--scenes` — lecture/slide mode: detect slides and emit one frame per slide with the transcript grouped per slide. Best for 1–2 hour talks over slide decks. Defaults to 768px frames; tune with `--scene-threshold` (default 0.3) and `--min-slide-seconds` (default 3).
- `--whisper local` — transcribe locally (no API key, no 25 MB / 50 min cap). Requires a one-time `setup.py --setup-local`. Default precedence is captions → local → API.
- `--whisper-model` / `--whisper-device` / `--whisper-language` — local Whisper knobs (default model `turbo`, auto device, auto language).
- `--save DIR` — write a portable bundle (report.md + frames/ + transcript.json/.txt + meta.json) to convey to another session.
```

Add a new section after "Recommended limits":

```markdown
## Long lectures (1–2 hours of slides)

For recorded talks over slide decks, use `--scenes` with local Whisper:

```bash
python3 "${CLAUDE_SKILL_DIR}/scripts/watch.py" lecture.mp4 --scenes
```

This detects each slide (one frame per slide instead of a sparse uniform scan),
transcribes the whole video locally (no length cap), and groups the transcript
under each slide. First run needs `setup.py --setup-local` (builds a Python-3.11
venv with openai-whisper; ~GB). The transcript is cached, so re-running with a
different `--scene-threshold` does not re-transcribe.

## Conveying a result to another session

Local Whisper only runs in Claude Code. To use a result in Claude.ai or a fresh
session, save a bundle and convey it:

```bash
python3 "${CLAUDE_SKILL_DIR}/scripts/watch.py" lecture.mp4 --scenes --save ./lecture-bundle
```

- Another Claude Code session: `python3 "${CLAUDE_SKILL_DIR}/scripts/load_bundle.py" ./lecture-bundle`.
- Claude.ai: upload `report.md` and the images in `frames/` (or `report.md`/`transcript.txt` alone for transcript-only).
```

- [ ] **Step 2: Update `README.md`**

Add to the "Other knobs" list (near line 146):

```markdown
- `--scenes` — lecture/slide mode: one frame per detected slide + transcript grouped per slide. For 1–2 hour talks over slides.
- `--whisper local` — local Whisper (no API key, no length cap); one-time setup via `scripts/setup.py --setup-local`. Precedence: captions → local → API.
- `--save DIR` — write a portable bundle to convey a run to Claude.ai or another session (`/watch-load DIR` to re-ingest).
```

Add a new section after "Bring your own keys":

```markdown
## Local Whisper & long lectures

For long videos (lectures, talks) and offline/no-key transcription, run Whisper
locally:

```bash
python3 scripts/setup.py --setup-local      # one-time: Python-3.11 venv + openai-whisper (~GB)
/watch lecture.mp4 --scenes                  # slide detection + local transcript, grouped per slide
```

`--scenes` extracts one frame per slide (not a sparse uniform scan) and aligns
each slide with what was said while it was on screen. Local Whisper has no 25 MB /
50 min API cap, so full 2-hour videos transcribe end to end. The transcript is
cached for cheap re-runs.

## Convey a watched video to another session

Local Whisper is Claude-Code-only. Save a portable bundle and hand it off:

```bash
/watch-save lecture.mp4 ./lecture-bundle --scenes   # writes report.md + frames/ + transcript
/watch-load ./lecture-bundle                        # re-ingest in another Claude Code session
```

For Claude.ai, upload `report.md` and the frame images from the bundle.
```

- [ ] **Step 3: Commit**

```bash
git add SKILL.md README.md
git commit -m "docs: document local Whisper, scene mode, and bundle save/load"
```

---

## Task 13: Final verification

- [ ] **Step 1: Full test suite**

Run: `python3 -m unittest discover -s tests -v`
Expected: every test PASSES (scenes 9, cache 3, setup 4, local_whisper 2, report 5, bundle 2, load_bundle 2, watch_backend 4).

- [ ] **Step 2: CLI smoke (uniform, no model needed) on any short clip**

```bash
python3 scripts/watch.py /tmp/sample.mp4 --no-whisper --save /tmp/uni-bundle
python3 scripts/load_bundle.py /tmp/uni-bundle | head -40
```
Expected: report renders; bundle has `report.md`, `frames/`, `transcript.txt` (empty transcript ok), `meta.json`; load resolves absolute frame paths.

- [ ] **Step 3: End-to-end lecture smoke (only if `--setup-local` done)**

```bash
python3 scripts/watch.py /tmp/lecture.mp4 --scenes --whisper-model tiny --save /tmp/lec-bundle
```
Expected: slides detected, per-slide transcript grouped, bundle written. Confirm `meta.json` `mode` is `scenes`.

- [ ] **Step 4: Confirm clean tree + final state**

Run: `git status` and `git log --oneline -15`
Expected: clean working tree; commits for tasks 1–12 present on `feat/local-whisper-and-scenes`.

---

## Self-review

**Spec coverage:**
- Local Whisper backend → Tasks 5 (venv), 6 (runner), 10 (precedence). ✓
- captions → local → API precedence → Task 10 `choose_backend`/`run_whisper`. ✓
- Dedicated uv-managed Py3.11 venv → Task 5 `cmd_setup_local`. ✓
- Scene/slide detection, merge, cap, rep-frame, per-slide grouping → Tasks 1–3, 10. ✓
- 768px scene default → Task 10 (`resolution` sentinel). ✓
- Transcript cache → Tasks 4, 10. ✓
- Scene-mode report format → Task 7 `_render_scenes`. ✓
- `--check`/`--json` local-aware → Task 5. ✓
- Bundle save (`--save`) + `/watch-save` + `/watch-load` → Tasks 8, 9, 10, 11. ✓
- Docs → Task 12. ✓
- `--multi-lang` intentionally deferred (spec non-goal). ✓

**Type consistency:** `choose_backend`, `run_whisper`, `transcribe_local_cached`, `launch_local`, `local_available`, `venv_python`, `cache_key/load/save`, `cuts_to_slides`/`merge_to_cap`/`group_transcript`/`extract_slide_frames`, `report.render`, `bundle.write_bundle`/`build_meta`, `load_bundle.resolve_report` are used with the same signatures across tasks and the interfaces table. ✓

**Placeholder scan:** No TBD/TODO; every code step contains complete code. ✓

**Note for executor:** `run_scenes` always transcribes via `run_whisper` (captions are not pulled in scene mode — slide lectures are local files without captions, and local Whisper has no length cap). This matches the spec's lecture use case.
