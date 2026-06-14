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
