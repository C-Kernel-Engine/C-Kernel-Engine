#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v7" / "scripts" / "init_tiny_train_model_v7.py"


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


init_tiny = _load_module("init_tiny_train_model_v7_test", SCRIPT)


class InitTinyTrainModelV7Tests(unittest.TestCase):
    def test_template_aliases_cover_family_names(self) -> None:
        self.assertEqual(init_tiny._resolve_template_name("gemma"), "gemma3")
        self.assertEqual(init_tiny._resolve_template_name("nanbeige"), "nanbeige")
        self.assertEqual(init_tiny._resolve_template_name("qwen35"), "qwen35")
        self.assertEqual(init_tiny._resolve_template_name(""), "qwen3")


if __name__ == "__main__":
    unittest.main()
