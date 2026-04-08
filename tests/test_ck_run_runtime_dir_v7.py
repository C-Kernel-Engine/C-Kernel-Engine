#!/usr/bin/env python3
import os
import sys
import tempfile
import time
import unittest
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "version" / "v7" / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "version" / "v7" / "scripts"))

import ck_run_v7  # type: ignore


class TestCKRunRuntimeDir(unittest.TestCase):
    def test_prefers_freshest_valid_runtime_dir(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_run_runtime_dir_") as td:
            run_dir = Path(td)
            root_lib = run_dir / "libmodel.so"
            root_weights = run_dir / "weights.bump"
            root_lib.write_bytes(b"root")
            root_weights.write_bytes(b"weights")

            stale_dir = run_dir / ".ck_build"
            stale_dir.mkdir()
            stale_lib = stale_dir / "libmodel.so"
            stale_weights = stale_dir / "weights.bump"
            stale_lib.write_bytes(b"stale")
            stale_weights.write_bytes(b"weights")

            # Make the root runtime clearly newer than the stale subdir.
            now = time.time()
            stale_time = now - 100.0
            fresh_time = now
            for path in (stale_lib, stale_weights):
                path.touch()
                path.chmod(0o644)
                os.utime(path, (stale_time, stale_time))
            for path in (root_lib, root_weights):
                path.touch()
                path.chmod(0o644)
                os.utime(path, (fresh_time, fresh_time))

            resolved = ck_run_v7._resolve_chat_runtime_dir(run_dir)
            self.assertEqual(resolved, run_dir)

    def test_detects_hybrid_recurrent_train_template_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_run_train_manifest_") as td:
            run_dir = Path(td)
            (run_dir / "weights_manifest.json").write_text(
                json.dumps(
                    {
                        "template": {
                            "contract": {
                                "attention_contract": {
                                    "attn_variant": "hybrid_recurrent_attention"
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(ck_run_v7._train_run_uses_hybrid_recurrent_attention(run_dir))


if __name__ == "__main__":
    unittest.main()
