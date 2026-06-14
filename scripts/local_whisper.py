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
import contextlib
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
    # Whisper may print a "Detected language: …" line to stdout before our JSON;
    # take the payload from the first '{' so leading chatter doesn't break parsing.
    out = proc.stdout or ""
    brace = out.find("{")
    try:
        data = json.loads(out[brace:]) if brace != -1 else json.loads(out)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"local Whisper returned non-JSON: {exc}: {out[:200]}")
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
    # Running under the venv interpreter: Python puts this script's directory
    # (scripts/) on sys.path[0], and scripts/whisper.py would shadow the pip
    # `openai-whisper` package. Strip our own dir so `import whisper` (inside
    # _transcribe) resolves to the installed package, not the API wrapper.
    here = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != here]

    ap = argparse.ArgumentParser(prog="local_whisper")
    ap.add_argument("audio")
    ap.add_argument("--model", default="turbo")
    ap.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    ap.add_argument("--language", default=None)
    args = ap.parse_args()

    device = _pick_device(args.device)
    # Whisper prints "Detected language: …" to stdout; route all transcription
    # chatter to stderr so our stdout stays a clean JSON payload.
    with contextlib.redirect_stdout(sys.stderr):
        segments = _transcribe(args.audio, args.model, device, args.language)
    json.dump({"segments": segments, "device": device}, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
