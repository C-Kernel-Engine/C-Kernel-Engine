#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version" / "v7" / "scripts"))

import build_ir_train_ropefix_v7  # type: ignore
import build_ir_train_v7  # type: ignore
import codegen_train_runtime_v7  # type: ignore
import lower_ir2_backward_v7  # type: ignore


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _entry(name: str) -> dict:
    return {
        "name": name,
        "dtype": "fp32",
        "shape": [1, 1],
        "offset": 0,
        "file_offset": 0,
        "size": 4,
    }


def _test_manifest(template_name: str) -> dict:
    template = _load_json(ROOT / "version" / "v7" / "templates" / f"{template_name}.json")
    return {
        "config": {
            "model": template_name,
            "num_layers": 1,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_heads": 8,
            "num_kv_heads": 4,
            "head_dim": 16,
            "vocab_size": 512,
            "train_tokens": 2,
        },
        "template": template,
        "entries": [
            _entry("token_emb"),
            _entry("output.weight"),
            _entry("norm.weight"),
            _entry("layer.0.ln1_gamma"),
            _entry("layer.0.ln2_gamma"),
            _entry("layer.0.wq"),
            _entry("layer.0.wk"),
            _entry("layer.0.wv"),
            _entry("layer.0.q_norm"),
            _entry("layer.0.k_norm"),
            _entry("layer.0.wo"),
            _entry("layer.0.w1"),
            _entry("layer.0.w2"),
            _entry("layer.0.w3"),
        ],
    }


def _first_rope_kernel(ir1: dict) -> str | None:
    for op in ir1.get("ops", []):
        if str(op.get("op")) == "rope_qk":
            kernel_id = op.get("kernel_id")
            return str(kernel_id) if kernel_id is not None else None
    return None


class TrainRopeFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = _load_json(ROOT / "version" / "v7" / "kernel_maps" / "KERNEL_REGISTRY.json")
        cls.bindings = _load_json(ROOT / "version" / "v7" / "kernel_maps" / "kernel_bindings.json")
        cls.grad_rules = _load_json(ROOT / "version" / "v7" / "scripts" / "grad_rules_v7.json")

    def test_ropefix_builder_selects_pairwise_rope_for_llama(self) -> None:
        ir1 = build_ir_train_ropefix_v7.build_ir1_train_ropefix(
            manifest=_test_manifest("llama"),
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            max_layers=1,
            strict=False,
        )
        self.assertEqual(_first_rope_kernel(ir1), "rope_forward_qk_pairwise")
        self.assertEqual((ir1.get("ropefix") or {}).get("rope_qk_kernel"), "rope_forward_qk_pairwise")

    def test_stable_builder_selects_pairwise_rope_for_llama(self) -> None:
        ir1 = build_ir_train_v7.build_ir1_train(
            manifest=_test_manifest("llama"),
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            max_layers=1,
            strict=False,
        )
        self.assertEqual(_first_rope_kernel(ir1), "rope_forward_qk_pairwise")
        self.assertEqual((ir1.get("config") or {}).get("rope_layout"), "pairwise")

    def test_ropefix_builder_preserves_split_rope_for_qwen3(self) -> None:
        ir1 = build_ir_train_ropefix_v7.build_ir1_train_ropefix(
            manifest=_test_manifest("qwen3"),
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            max_layers=1,
            strict=False,
        )
        self.assertEqual(_first_rope_kernel(ir1), "rope_forward_qk")

    def test_stable_builder_preserves_split_rope_for_qwen3(self) -> None:
        ir1 = build_ir_train_v7.build_ir1_train(
            manifest=_test_manifest("qwen3"),
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            max_layers=1,
            strict=False,
        )
        self.assertEqual(_first_rope_kernel(ir1), "rope_forward_qk")
        self.assertEqual((ir1.get("config") or {}).get("rope_layout"), "split")

    def test_ropefix_builder_restores_stable_kernel_map_after_use(self) -> None:
        self.assertEqual(build_ir_train_v7.FORWARD_KERNEL_BY_OP["rope_qk"], "rope_forward_qk")
        _ = build_ir_train_ropefix_v7.build_ir1_train_ropefix(
            manifest=_test_manifest("llama"),
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            max_layers=1,
            strict=False,
        )
        self.assertEqual(build_ir_train_v7.FORWARD_KERNEL_BY_OP["rope_qk"], "rope_forward_qk")

    def test_codegen_train_runtime_recognizes_pairwise_rope_forward_kernel(self) -> None:
        self.assertTrue(codegen_train_runtime_v7._is_rope_forward_qk_kernel("rope_forward_qk"))
        self.assertTrue(codegen_train_runtime_v7._is_rope_forward_qk_kernel("rope_forward_qk_pairwise"))
        self.assertEqual(
            codegen_train_runtime_v7._rope_forward_qk_bridge_function("rope_forward_qk_pairwise"),
            "rope_forward_qk_pairwise_with_rotary_dim",
        )

    def test_lower_ir2_uses_pairwise_rope_backward_for_llama_ropefix_ir1(self) -> None:
        ir1 = build_ir_train_ropefix_v7.build_ir1_train_ropefix(
            manifest=_test_manifest("llama"),
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            max_layers=1,
            strict=False,
        )
        ir2 = lower_ir2_backward_v7.synthesize_ir2_backward(
            ir1=ir1,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            strict=False,
            allow_partial=True,
            checkpoint_policy="none",
        )
        rope_backward_kernels = [
            str(op.get("kernel_id"))
            for op in ir2.get("backward", [])
            if str(op.get("op", "")).startswith("rope_qk_backward_")
        ]
        self.assertIn("rope_backward_qk_pairwise_f32", rope_backward_kernels)

    def test_lower_ir2_keeps_split_rope_backward_for_qwen3(self) -> None:
        ir1 = build_ir_train_ropefix_v7.build_ir1_train_ropefix(
            manifest=_test_manifest("qwen3"),
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            max_layers=1,
            strict=False,
        )
        ir2 = lower_ir2_backward_v7.synthesize_ir2_backward(
            ir1=ir1,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            strict=False,
            allow_partial=True,
            checkpoint_policy="none",
        )
        rope_backward_kernels = [
            str(op.get("kernel_id"))
            for op in ir2.get("backward", [])
            if str(op.get("op", "")).startswith("rope_qk_backward_")
        ]
        self.assertIn("rope_backward_qk_f32", rope_backward_kernels)
        self.assertNotIn("rope_backward_qk_pairwise_f32", rope_backward_kernels)

    def test_lower_ir2_uses_config_rope_layout_not_forward_kernel_name(self) -> None:
        ir1 = build_ir_train_ropefix_v7.build_ir1_train_ropefix(
            manifest=_test_manifest("llama"),
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            max_layers=1,
            strict=False,
        )
        tampered = deepcopy(ir1)
        tampered["config"]["rope_layout"] = "split"
        for op in tampered.get("ops", []):
            if str(op.get("op")) == "rope_qk":
                op["kernel_id"] = "rope_forward_qk_pairwise"
                break
        ir2 = lower_ir2_backward_v7.synthesize_ir2_backward(
            ir1=tampered,
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            strict=False,
            allow_partial=True,
            checkpoint_policy="none",
        )
        rope_backward_kernels = [
            str(op.get("kernel_id"))
            for op in ir2.get("backward", [])
            if str(op.get("op", "")).startswith("rope_qk_backward_")
        ]
        self.assertIn("rope_backward_qk_f32", rope_backward_kernels)
        self.assertNotIn("rope_backward_qk_pairwise_f32", rope_backward_kernels)

    def test_codegen_train_runtime_recognizes_pairwise_rope_backward_kernel(self) -> None:
        self.assertTrue(codegen_train_runtime_v7._is_rope_backward_qk_kernel("rope_backward_qk_f32"))
        self.assertTrue(codegen_train_runtime_v7._is_rope_backward_qk_kernel("rope_backward_qk_pairwise_f32"))
        self.assertTrue(codegen_train_runtime_v7._is_pairwise_rope_backward_qk_kernel("rope_backward_qk_pairwise_f32"))


if __name__ == "__main__":
    unittest.main()
