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
   - **Easiest for Claude.ai:** pass `--pdf` (or run `scripts/make_pdf.py <bundle-dir>`) to get a single `slides.pdf`, and upload that one file instead of every image in `frames/`.

Pass `--save` from the start when sharing is the goal — re-running later re-decodes
the video (the transcript itself is cached).
