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
