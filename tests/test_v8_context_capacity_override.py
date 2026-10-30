"""A build request must not silently shrink its declared context capacity."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "version" / "v8" / "scripts" / "build_ir_v8.py"


class ContextCapacityOverrideTests(unittest.TestCase):
    def test_context_outside_manifest_capacity_fails_before_lowering(self) -> None:
        for requested in (0, -1, 8):
            with self.subTest(requested=requested), tempfile.TemporaryDirectory() as root:
                tmp_path = Path(root)
                manifest = tmp_path / "weights_manifest.json"
                manifest.write_text(json.dumps({
                    "version": 5,
                    "config": {"context_length": 4, "max_seq_len": 4},
                    "template": {},
                    "entries": [],
                }))
                output = tmp_path / "ir.json"
                result = subprocess.run(
                    [sys.executable, str(BUILDER), "--manifest", str(manifest),
                     "--mode", "decode", "--context-len", str(requested),
                     "--output", str(output)],
                    cwd=ROOT, text=True, capture_output=True, timeout=30,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(output.exists())
                diagnostic = json.loads((tmp_path / "build_diagnostic.json").read_text())
                self.assertEqual(diagnostic["failure"]["code"], "CKE-V8-CONTEXT-CAPACITY")
                self.assertEqual(diagnostic["failure"]["observed"], {
                    "requested_context_len": requested,
                    "declared_context_length": 4,
                })


if __name__ == "__main__":
    unittest.main()
