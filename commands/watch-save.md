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
4. **Ask the user how they'll use the bundle** with `AskUserQuestion` (header "Convey to"),
   then act on the answer — do not just print options:
   - **Claude.ai / web or app** → build the single PDF (best for upload):
     `python3 "${CLAUDE_PLUGIN_ROOT}/scripts/make_pdf.py" "<bundle-dir>"` (instant — it
     just wraps the frames already on disk; no re-run, no `--pdf` needed up front). Then
     tell them to upload `<bundle-dir>/slides.pdf`, optionally with `transcript.txt` for the words.
   - **Another Claude Code session** → tell them to run `/watch-load <bundle-dir>` there
     (no PDF needed — it reads the frames directly).
   - **Transcript only** → point them to `<bundle-dir>/transcript.txt` (no images to upload).

   If the user already stated where it's going (e.g. "for Claude.ai"), skip the question and
   act directly.

Note: `make_pdf.py` works on any existing bundle, so the PDF is generated on demand only
when the user wants it. Pass `--save` from the start when sharing is the goal — re-running
later re-decodes the video (the transcript itself is cached).
