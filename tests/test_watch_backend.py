# tests/test_watch_backend.py
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import argparse  # noqa: E402

import watch  # noqa: E402


class RunWhisperFallbackTests(unittest.TestCase):
    def test_local_missing_falls_back_to_api_when_key_present(self):
        args = argparse.Namespace(
            whisper="local", whisper_model="turbo", whisper_device=None, whisper_language=None
        )
        with mock.patch.object(watch, "local_available", return_value=False), \
             mock.patch.object(watch, "load_api_key", return_value=("groq", "KEY")), \
             mock.patch.object(watch, "transcribe_video",
                               return_value=([{"start": 0.0, "end": 1.0, "text": "hi"}], "groq")) as tv:
            segs, source = watch.run_whisper("v.mp4", Path("/tmp"), args)
        self.assertTrue(segs)
        self.assertEqual(source, "whisper (groq)")
        tv.assert_called_once()


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
