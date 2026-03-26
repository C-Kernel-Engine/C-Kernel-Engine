#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V7_BUILD_PATH = ROOT / "version" / "v7" / "scripts" / "build_ir_v7.py"
V8_BUILD_PATH = ROOT / "version" / "v8" / "scripts" / "build_ir_v8.py"


def _load_module(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build_ir_v7 = _load_module("build_ir_v7_for_v8_scaffold_tests", V7_BUILD_PATH)
build_ir_v8 = _load_module("build_ir_v8_for_tests", V8_BUILD_PATH)


class BuildIrV8ScaffoldTests(unittest.TestCase):
    def test_v8_template_root_is_isolated(self) -> None:
        self.assertEqual(build_ir_v8.V8_ROOT.name, "v8")
        self.assertTrue((build_ir_v8.V8_ROOT / "templates" / "qwen3.json").exists())

    def test_v8_templates_match_current_v7_seed(self) -> None:
        for name in ("gemma3", "llama", "qwen2", "qwen3", "qwen35"):
            with self.subTest(template=name):
                v7_doc = json.loads((ROOT / "version" / "v7" / "templates" / f"{name}.json").read_text(encoding="utf-8"))
                v8_doc = json.loads((ROOT / "version" / "v8" / "templates" / f"{name}.json").read_text(encoding="utf-8"))
                self.assertEqual(v8_doc, v7_doc)

    def test_v8_uses_same_rope_resolution_as_v7(self) -> None:
        cases = [
            ({}, {"rope_qk": "rope_forward_qk_pairwise"}),
            ({"rope_layout": "split"}, {"rope_qk": "rope_forward_qk_pairwise"}),
            ({"rope_layout": "interleaved"}, {}),
            ({"rope_layout": "split"}, {"rope_qk": "rope_forward_qk_custom"}),
        ]
        for config, kernels in cases:
            with self.subTest(config=config, kernels=kernels):
                self.assertEqual(
                    build_ir_v8._resolve_rope_qk_kernel(config, kernels),
                    build_ir_v7._resolve_rope_qk_kernel(config, kernels),
                )

    def test_v8_uses_v7_kernel_registry_until_runtime_diverges(self) -> None:
        registry = build_ir_v8.load_kernel_registry()
        self.assertIsInstance(registry, dict)
        self.assertTrue(registry)
        self.assertEqual(build_ir_v8.V7_ROOT.name, "v7")
        self.assertTrue((build_ir_v8.V7_ROOT / "kernel_maps" / "KERNEL_REGISTRY.json").exists())


if __name__ == "__main__":
    unittest.main()
