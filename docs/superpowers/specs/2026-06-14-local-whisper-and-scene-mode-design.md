# Design: Local Whisper transcription + scene/slide mode

- **Date:** 2026-06-14
- **Status:** Approved (pending spec review)
- **Branch:** `feat/local-whisper-and-scenes`

## Summary

Add two orthogonal, composable capabilities to the `/watch` skill so it can handle
**long (1–2 hour) lecture-style videos with distinct slides** (e.g. recorded talks
over PPT decks):

1. **`--whisper local`** — transcribe with a locally-run OpenAI Whisper instead of
   the Groq/OpenAI API. Removes the API's 25 MB (~50 min) upload cap, so full 2-hour
   videos transcribe end to end. Also joins the default transcription precedence
   chain: **captions → local → API**.
2. **`--scenes`** — replace uniform frame sampling with **slide detection**: one
   representative frame per distinct slide, with the transcript grouped under each
   slide so a lecture becomes a slide-by-slide digest.

The two flags are independent and can be used separately or together.

## Goals

- Transcribe arbitrarily long local videos with no API key and no length cap.
- Turn a 1–2 hour slide lecture into a compact, complete set of slide frames
  (one per slide) instead of a useless 100-frame sparse uniform scan.
- Align each slide with the words spoken while it was on screen.
- Keep the existing API + captions paths and the pure-stdlib core untouched for
  users who don't opt into local Whisper.

## Non-goals (YAGNI)

- Auto-detecting whether a video "is a lecture" — `--scenes` is explicit.
- OCR of on-slide text — frames + transcript already cover the content.
- `--multi-lang` per-segment language detection (deferred; see Open decisions).
- GPU/batch perf tuning beyond device auto-pick.

## Background: current pipeline

`watch.py` orchestrates `download (yt-dlp) → frames (ffmpeg auto-fps) →
transcript → markdown report`. Transcript precedence today is **captions
(yt-dlp VTT) → Whisper API (Groq preferred, OpenAI fallback)**. Relevant limits:

- Frames hard-capped at **100 / 2 fps**; >10 min prints a "sparse scan" warning.
- API audio upload capped at **25 MB ≈ 50 min** of mono 16 kHz mp3.
- Scripts are **pure stdlib** (no pip installs) on the captions/API path.

`../Whisper` (`whisper-transcribe`) is a local OpenAI-Whisper CLI: auto device
pick (cuda > mps > cpu, `PYTORCH_ENABLE_MPS_FALLBACK=1`), model sizes (default
`turbo`), and a `--multi-lang` VAD mode. Its `model.transcribe()` returns
`{start, end, text}` segments — the **same shape** the watch pipeline already
consumes, so integration is a drop-in at the segment boundary.

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Local Whisper integration | **Vendor** a minimal module into the plugin (don't depend on the separate `../Whisper` install). |
| Feature packaging | **Independent flags** (`--scenes`, `--whisper local`), composable — not one combined mode. |
| Transcript precedence | **captions → local → API**. |
| `--multi-lang` | **Defer** (extra `silero-vad` dep). |
| Slide frame resolution | **768px** default in scene mode (balanced legibility vs tokens). |

## Detailed design

### A. Local Whisper backend

**New file: `scripts/local_whisper.py`** — adapted from `../Whisper`, runs as a
standalone script under the managed venv's Python:

- CLI: `local_whisper.py <audio-path> [--model M] [--device D] [--language L]`.
- Lazily imports `torch` and `whisper` (only this script needs them).
- Device auto-pick: `cuda > mps > cpu`; sets `PYTORCH_ENABLE_MPS_FALLBACK=1` on mps.
- `model.transcribe()` → emit `{"segments": [{start, end, text}, ...]}` as JSON to
  stdout. Progress/log lines go to stderr.
- Default model `turbo` (matches `../Whisper`).

**Why a subprocess, not an import:** the heavy deps (`torch`, `openai-whisper`,
~GB) must not pollute the user's system Python, and the system Python here is
3.14 where those wheels are unreliable. So:

**Dependency isolation — managed venv:**
- `setup.py` builds a dedicated venv at `~/.config/watch/whisper-venv`, pinned to
  **Python 3.11** (matching `../Whisper`'s `>=3.11,<3.12`), with `openai-whisper`
  installed. Uses **`uv venv --python 3.11` + `uv pip install`** when `uv` is on
  PATH (auto-downloads 3.11 if absent); falls back to `python3.11 -m venv` + `pip`
  otherwise.
- `watch.py` (system Python, stdlib) extracts audio once via the existing
  `whisper.py:extract_audio`, then invokes `local_whisper.py` through the venv's
  Python as a subprocess and parses the JSON segments.
- Model **weights** (turbo ≈ 1.5 GB) download to `~/.cache/whisper/` on first
  transcription (Whisper's own cache) — not pre-pulled by setup.

**`watch.py` precedence (replaces current captions→API):**
1. Native captions if present (unchanged).
2. Else, unless `--no-whisper`, pick a Whisper backend:
   - `--whisper local` → local. If the venv is missing, print a hint to run
     `setup.py --setup-local` and fall through to API if a key exists.
   - `--whisper groq|openai` → that API (unchanged).
   - **auto (no `--whisper`)** → **prefer local if the venv exists**, else API
     (Groq > OpenAI). This realizes captions → local → API.

**New flags on `watch.py`:** `--whisper local` (added to existing choices);
`--whisper-model` (default `turbo`), `--whisper-device`, `--whisper-language`
(passed through to `local_whisper.py`; ignored for API backends).

**`scripts/whisper.py`:** unchanged transcription logic; only the backend
*selection* in `watch.py` grows a `local` branch. `extract_audio` is reused as-is.

### B. Scene / slide mode (`--scenes`)

**New file: `scripts/scenes.py`.**

- **Detection:** ffmpeg content scene filter in a single decode pass:
  `-vf select='gt(scene,T)',showinfo` (or `metadata=print`) to capture cut
  timestamps; `T` from `--scene-threshold` (default ~0.3). Chosen over
  sampled perceptual-hashing because slide changes are large, clean cuts the
  scene filter detects reliably with no extra dependency.
- **Cut list → slides:** slide *i* spans `[cut_i, cut_{i+1})` (slide 0 starts at
  0; last slide runs to end). Merge any slide shorter than `--min-slide-seconds`
  (default 3 s) into its neighbor so PPT build-animations don't each become a
  slide.
- **Representative frame:** extract one frame at `cut_i + 0.5 s` (clamped within
  the slide span) to avoid transition blur, at `--resolution` (default **768** in
  scene mode), `-q:v 4` JPEG — reusing the ffmpeg invocation style from
  `frames.py:extract`.
- **Overflow cap:** if detected slides exceed `--max-frames` (default 100), merge
  the shortest-duration slides until under the cap and print a warning naming how
  many were merged (no silent truncation).
- **Per-slide transcript grouping:** for each slide, select transcript segments
  overlapping its span (reusing `transcribe.py:filter_range` semantics) and render
  them beneath the slide.

**Scene-mode report format** (replaces the flat Frames + Transcript sections when
`--scenes` is set):

```
# watch: lecture report
- Source / Title / Duration / Resolution …
- Slides: 42 detected (threshold 0.30, 768px); transcript via whisper (local: turbo)

## Slides
### Slide 1 — 00:00 → 02:14
`/tmp/watch-xxxx/frames/slide_0001.jpg`
> [00:00] Welcome everyone, today we'll cover…
> [01:30] The first topic is…

### Slide 2 — 02:14 → 05:48
`/tmp/watch-xxxx/frames/slide_0002.jpg`
> [02:14] Moving on to the architecture…
```

`SKILL.md` Step 3 stays the same — Claude `Read`s each `slide_*.jpg` path; the
grouping just gives it slide↔speech alignment for free.

### C. Transcript cache

Local transcription of a 2-hour video takes many minutes, and re-running
`--scenes` with a different threshold should not re-transcribe.

- Cache file: `~/.config/watch/cache/<key>.json`, `key = sha256(audio bytes) +
  model + language`.
- On cache hit, skip transcription and load segments. Applies to the local
  backend only (API is fast/keyed differently and length-limited).
- Cache is keyed on the **extracted audio** (~57 MB for 2 h), which is small and
  already produced for transcription.

### D. `setup.py` changes

- **New:** `setup.py --setup-local` — builds/repairs the `~/.config/watch/whisper-venv`
  (uv-preferred, Python 3.11, `openai-whisper`). Idempotent. Prints progress; this
  is the only heavy (~GB) action and is never run implicitly.
- **`--check` status:** transcription is now "configured" if **either** an API key
  exists **or** the local venv is present. So `needs_key` is returned only when
  there is no API key *and* no local venv. New `--json` field: `local_available`
  (bool) alongside the existing `whisper_backend` / `has_api_key`.
- **Interactive installer:** when no transcription path is configured, offer three
  choices instead of two: (a) Groq key, (b) OpenAI key, (c) **build local Whisper
  venv** (warn ~GB download), or proceed `--no-whisper`.

### E. Docs

- `SKILL.md`: document `--whisper local` (+ model/device/language), the new
  precedence, `--scenes` and its flags, scene-mode report shape, and the
  `--setup-local` remediation path. Update the "Recommended limits" section to note
  that local Whisper + `--scenes` is the recommended path for >10 min lectures.
- `README.md`: add a "Local Whisper" + "Lecture / slide mode" section and a usage
  example: `/watch lecture.mp4 --scenes` (auto-uses local Whisper once the venv
  exists).

## Error handling

- `--whisper local` but venv missing → hint to run `setup.py --setup-local`; fall
  through to API if a key exists, else frames-only with a clear message.
- Local transcription subprocess fails (model download, OOM, etc.) → stderr from
  the venv subprocess is surfaced; pipeline falls through to API if available,
  else frames-only (mirrors current Whisper-API failure handling).
- `--scenes` on a video with no detectable cuts (e.g. a talking head) → fall back
  to a single "slide" spanning the whole video plus a note suggesting plain mode.
- ffmpeg scene pass over a 2-hour file is one decode pass (minutes); a progress
  note is printed to stderr.

## Testing

Unit tests (`tests/`) for pure logic with synthetic data — no model download or
real video needed:

- cut-list → slide-span construction (boundaries, last-slide-to-end).
- short-slide merging (`min-slide-seconds`) and overflow merge-to-cap.
- per-slide transcript grouping vs `filter_range`.
- cache key determinism.

A real-video smoke test (`--scenes --whisper local` on a short slide clip) is
documented in the spec/README but kept out of CI (needs ffmpeg + model weights).

## Affected files

| File | Change |
|---|---|
| `scripts/local_whisper.py` | **new** — vendored local Whisper runner (venv subprocess). |
| `scripts/scenes.py` | **new** — scene detection, slide spans, per-slide frames + transcript grouping. |
| `scripts/watch.py` | `--whisper local`, model/device/language flags, `--scenes` + scene flags, precedence chain, scene-mode report, transcript cache wiring. |
| `scripts/whisper.py` | reuse `extract_audio`; (selection moves to `watch.py`). |
| `scripts/setup.py` | `--setup-local` venv builder; `--check`/`--json` account for local; installer offers local option. |
| `SKILL.md`, `README.md` | document local Whisper + scene mode. |
| `tests/` | **new** — unit tests for slide/grouping/cache logic. |

## Open decisions

None outstanding — all resolved in brainstorming. `--multi-lang` intentionally
deferred and can be added later as a flag on `local_whisper.py` without changing
the architecture.
