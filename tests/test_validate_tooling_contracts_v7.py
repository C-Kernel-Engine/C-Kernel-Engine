#!/usr/bin/env python3
"""Regression checks for v7 static tooling/path contracts."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "version" / "v7" / "scripts" / "validate_tooling_contracts.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_tooling_contracts_v7", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestValidateToolingContractsV7(unittest.TestCase):
    def test_path_contract_layers_present_and_passing(self) -> None:
        mod = _load_module()
        rows = mod.run_checks()
        by_layer = {row.layer: row for row in rows}

        self.assertIn("L7", by_layer)
        self.assertIn("L8", by_layer)
        self.assertEqual(by_layer["L7"].status, "PASS", by_layer["L7"].detail)
        self.assertEqual(by_layer["L8"].status, "PASS", by_layer["L8"].detail)


if __name__ == "__main__":
    unittest.main()
