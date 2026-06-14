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
