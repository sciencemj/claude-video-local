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

    def _tmpdir(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)

    def test_local_available_false_when_missing(self):
        self.assertFalse(setup.local_available(self._tmpdir() / "novenv"))

    def test_local_available_true_when_python_present(self):
        venv = self._tmpdir() / "venv"
        py = setup.venv_python(venv)
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_text("#!/bin/sh\n")
        self.assertTrue(setup.local_available(venv))

    def test_status_reports_local_available_key(self):
        self.assertIn("local_available", setup._status())


if __name__ == "__main__":
    unittest.main()
