#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v7" / "scripts" / "check_disk_headroom_v7.py"


class CheckDiskHeadroomV7Tests(unittest.TestCase):
    def test_check_passes_with_tiny_threshold(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_disk_ok_") as tmp:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--path",
                    tmp,
                    "--min-free-gb",
                    "0.001",
                    "--label",
                    "unit_ok",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["label"], "unit_ok")

    def test_check_fails_with_unreachable_threshold(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_disk_fail_") as tmp:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--path",
                    tmp,
                    "--min-free-gb",
                    "1000000",
                    "--label",
                    "unit_fail",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(proc.returncode, 0)
            payload = json.loads(proc.stdout)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["label"], "unit_fail")


if __name__ == "__main__":
    unittest.main()
