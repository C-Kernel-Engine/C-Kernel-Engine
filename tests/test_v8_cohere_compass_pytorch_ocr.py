#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v8" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "certify_cohere_compass_pytorch_ocr_v8",
    SCRIPTS / "certify_cohere_compass_pytorch_ocr_v8.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CohereCompassPytorchOcrTests(unittest.TestCase):
    def test_model_identity_covers_config_and_every_safetensors_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cke-cohere-oracle-") as temporary:
            checkpoint = Path(temporary)
            (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
            (checkpoint / "model-2.safetensors").write_bytes(b"second")
            (checkpoint / "model-1.safetensors").write_bytes(b"first")
            identity = MODULE._model_identity(checkpoint)

        self.assertEqual(
            [entry["name"] for entry in identity["weight_files"]],
            ["model-1.safetensors", "model-2.safetensors"],
        )
        self.assertTrue(identity["config_sha256"])
        self.assertTrue(all(entry["sha256"] for entry in identity["weight_files"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
