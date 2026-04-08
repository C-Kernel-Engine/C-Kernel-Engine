#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v7" / "kernel_maps" / "validate_kernel_maps.py"
KERNEL_MAPS = ROOT / "version" / "v7" / "kernel_maps"


def _load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


validator_mod = _load_module("validate_kernel_maps_v7_test", SCRIPT)


class TestV7KernelMapValidator(unittest.TestCase):
    def test_tokenizer_map_uses_signature_schema_without_errors(self) -> None:
        errors, warnings = validator_mod.validate_kernel_map(
            KERNEL_MAPS / "tokenizer_bpe_trie.json",
            check_paths=False,
        )
        self.assertEqual(errors, [])
        self.assertIsInstance(warnings, list)

    def test_mixed_quant_map_no_longer_crashes_on_quant_metadata(self) -> None:
        errors, warnings = validator_mod.validate_kernel_map(
            KERNEL_MAPS / "mega_fused_attention_decode_q5_0.json",
            check_paths=False,
        )
        self.assertEqual(errors, [])
        self.assertIsInstance(warnings, list)

    def test_memcpy_style_kernel_is_accepted(self) -> None:
        errors, warnings = validator_mod.validate_kernel_map(
            KERNEL_MAPS / "memcpy.json",
            check_paths=False,
        )
        self.assertEqual(errors, [])
        self.assertIsInstance(warnings, list)


if __name__ == "__main__":
    unittest.main()
