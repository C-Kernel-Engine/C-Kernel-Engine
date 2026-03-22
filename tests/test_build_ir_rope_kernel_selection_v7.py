#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version" / "v7" / "scripts"))

import build_ir_v7  # type: ignore


class RopeKernelSelectionTests(unittest.TestCase):
    def test_unspecified_rope_layout_preserves_pairwise_template_default(self) -> None:
        kernel = build_ir_v7._resolve_rope_qk_kernel(
            {},
            {"rope_qk": "rope_forward_qk_pairwise"},
        )
        self.assertEqual(kernel, "rope_forward_qk_pairwise")

    def test_split_rope_layout_ignores_pairwise_template_override(self) -> None:
        kernel = build_ir_v7._resolve_rope_qk_kernel(
            {"rope_layout": "split"},
            {"rope_qk": "rope_forward_qk_pairwise"},
        )
        self.assertEqual(kernel, "rope_forward_qk")

    def test_pairwise_rope_layout_selects_pairwise_kernel(self) -> None:
        kernel = build_ir_v7._resolve_rope_qk_kernel(
            {"rope_layout": "interleaved"},
            {},
        )
        self.assertEqual(kernel, "rope_forward_qk_pairwise")

    def test_non_pairwise_override_is_preserved_for_split_layout(self) -> None:
        kernel = build_ir_v7._resolve_rope_qk_kernel(
            {"rope_layout": "split"},
            {"rope_qk": "rope_forward_qk_custom"},
        )
        self.assertEqual(kernel, "rope_forward_qk_custom")

    def test_templates_declare_explicit_rope_layout_defaults(self) -> None:
        cases = {
            "version/v7/templates/llama.json": "pairwise",
            "version/v7/templates/gemma3.json": "split",
            "version/v7/templates/qwen2.json": "split",
            "version/v7/templates/qwen3.json": "split",
            "version/v7/templates/qwen35.json": "split",
        }
        for rel_path, expected in cases.items():
            doc = json.loads((ROOT / rel_path).read_text(encoding="utf-8"))
            actual = (
                ((doc.get("contract") or {}).get("attention_contract") or {}).get("rope_layout")
            )
            self.assertEqual(actual, expected, rel_path)


if __name__ == "__main__":
    unittest.main()
