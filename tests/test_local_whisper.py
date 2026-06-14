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
    def test_launch_local_tolerates_leading_stdout_noise(self):
        # Whisper prints "Detected language: English" to stdout before our JSON.
        payload = json.dumps({"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]})
        fake = self._fake_python(f"echo 'Detected language: English'\ncat <<'EOF'\n{payload}\nEOF")
        segs = local_whisper.launch_local(Path("/tmp/a.mp3"), "turbo", None, None, fake)
        self.assertEqual(segs, [{"start": 0.0, "end": 1.0, "text": "hi"}])

    @unittest.skipIf(os.name == "nt", "shell-script fake interpreter is POSIX-only")
    def test_launch_local_raises_on_nonzero_exit(self):
        fake = self._fake_python("echo boom 1>&2\nexit 3")
        with self.assertRaises(SystemExit):
            local_whisper.launch_local(Path("/tmp/a.mp3"), "turbo", None, None, fake)


if __name__ == "__main__":
    unittest.main()
