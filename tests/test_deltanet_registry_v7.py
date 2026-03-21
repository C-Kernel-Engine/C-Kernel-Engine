#!/usr/bin/env python3
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v7" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_ir_v7 as build_ir


class TestDeltaNetRegistryV7(unittest.TestCase):
    def test_registry_contains_deltanet_kernel(self) -> None:
        registry = build_ir.load_kernel_registry()
        kernel = next(
            (k for k in registry.get("kernels", []) if k.get("id") == "gated_deltanet_autoregressive_forward"),
            None,
        )
        self.assertIsNotNone(kernel)
        self.assertEqual(kernel.get("op"), "gated_deltanet")

    def test_registry_contains_ssm_conv_kernels(self) -> None:
        registry = build_ir.load_kernel_registry()
        ids = {k.get("id") for k in registry.get("kernels", [])}
        self.assertIn("ssm_conv1d_forward", ids)
        self.assertIn("ssm_conv1d_backward", ids)
        self.assertIn("split_q_gate_forward", ids)
        self.assertIn("split_q_gate_backward_f32", ids)
        self.assertIn("attn_gate_sigmoid_mul_forward", ids)
        self.assertIn("attn_gate_sigmoid_mul_backward_f32", ids)
        self.assertIn("recurrent_split_qkv_forward", ids)
        self.assertIn("recurrent_split_qkv_backward_f32", ids)
        self.assertIn("recurrent_dt_gate_forward", ids)
        self.assertIn("recurrent_dt_gate_backward_f32", ids)
        self.assertIn("recurrent_conv_state_update_forward", ids)
        self.assertIn("recurrent_conv_state_update_backward_f32", ids)
        self.assertIn("recurrent_silu_forward", ids)
        self.assertIn("recurrent_silu_backward_f32", ids)
        self.assertIn("recurrent_split_conv_qkv_forward", ids)
        self.assertIn("recurrent_split_conv_qkv_backward_f32", ids)
        self.assertIn("recurrent_qk_l2_norm_forward", ids)
        self.assertIn("recurrent_qk_l2_norm_backward_f32", ids)
        self.assertIn("recurrent_norm_gate_forward", ids)
        self.assertIn("recurrent_norm_gate_backward_f32", ids)

    def test_bindings_contain_deltanet_signature(self) -> None:
        bindings = build_ir.load_kernel_bindings()
        binding = bindings.get("gated_deltanet_autoregressive_forward")
        self.assertIsNotNone(binding)
        params = [p.get("name") for p in binding.get("params", [])]
        self.assertEqual(
            params,
            [
                "q",
                "k",
                "v",
                "g",
                "beta",
                "state_in",
                "state_out",
                "out",
                "num_heads",
                "state_dim",
                "norm_eps",
            ],
        )

    def test_bindings_contain_ssm_conv_signature(self) -> None:
        bindings = build_ir.load_kernel_bindings()
        binding = bindings.get("ssm_conv1d_forward")
        self.assertIsNotNone(binding)
        params = [p.get("name") for p in binding.get("params", [])]
        self.assertEqual(
            params,
            [
                "conv_x",
                "kernel",
                "out",
                "kernel_size",
                "num_channels",
                "num_tokens",
                "num_seqs",
            ],
        )

    def test_bindings_contain_recurrent_split_signature(self) -> None:
        bindings = build_ir.load_kernel_bindings()
        binding = bindings.get("recurrent_split_qkv_forward")
        self.assertIsNotNone(binding)
        params = [p.get("name") for p in binding.get("params", [])]
        self.assertEqual(
            params,
            [
                "packed_qkv",
                "q",
                "k",
                "v",
                "rows",
                "q_dim",
                "k_dim",
                "v_dim",
            ],
        )

    def test_bindings_contain_split_q_gate_signature(self) -> None:
        bindings = build_ir.load_kernel_bindings()
        binding = bindings.get("split_q_gate_forward")
        self.assertIsNotNone(binding)
        params = [p.get("name") for p in binding.get("params", [])]
        self.assertEqual(
            params,
            [
                "packed_qg",
                "q",
                "gate",
                "rows",
                "q_dim",
                "gate_dim",
            ],
        )

    def test_bindings_contain_recurrent_dt_gate_signature(self) -> None:
        bindings = build_ir.load_kernel_bindings()
        binding = bindings.get("recurrent_dt_gate_forward")
        self.assertIsNotNone(binding)
        params = [p.get("name") for p in binding.get("params", [])]
        self.assertEqual(
            params,
            [
                "alpha",
                "dt_bias",
                "a",
                "gate",
                "rows",
                "dim",
            ],
        )

    def test_bindings_contain_recurrent_conv_state_signature(self) -> None:
        bindings = build_ir.load_kernel_bindings()
        binding = bindings.get("recurrent_conv_state_update_forward")
        self.assertIsNotNone(binding)
        params = [p.get("name") for p in binding.get("params", [])]
        self.assertEqual(
            params,
            [
                "state_in",
                "q",
                "k",
                "v",
                "conv_x",
                "state_out",
                "history_len",
                "num_seqs",
                "num_tokens",
                "q_dim",
                "k_dim",
                "v_dim",
            ],
        )

    def test_bindings_contain_recurrent_norm_gate_signature(self) -> None:
        bindings = build_ir.load_kernel_bindings()
        binding = bindings.get("recurrent_norm_gate_forward")
        self.assertIsNotNone(binding)
        params = [p.get("name") for p in binding.get("params", [])]
        self.assertEqual(
            params,
            [
                "x",
                "gate",
                "weight",
                "out",
                "rows",
                "num_heads",
                "head_dim",
                "eps",
            ],
        )

    def test_bindings_contain_recurrent_qk_l2_norm_signature(self) -> None:
        bindings = build_ir.load_kernel_bindings()
        binding = bindings.get("recurrent_qk_l2_norm_forward")
        self.assertIsNotNone(binding)
        params = [p.get("name") for p in binding.get("params", [])]
        self.assertEqual(
            params,
            [
                "q",
                "k",
                "rows",
                "q_dim",
                "k_dim",
                "head_dim",
                "eps",
            ],
        )

    def test_bindings_contain_attn_gate_sigmoid_mul_signature(self) -> None:
        bindings = build_ir.load_kernel_bindings()
        binding = bindings.get("attn_gate_sigmoid_mul_forward")
        self.assertIsNotNone(binding)
        params = [p.get("name") for p in binding.get("params", [])]
        self.assertEqual(
            params,
            [
                "x",
                "gate",
                "out",
                "rows",
                "dim",
            ],
        )

    def test_inference_prefers_forward_recurrent_kernels(self) -> None:
        registry = build_ir.load_kernel_registry()
        expected = {
            "recurrent_conv_state_update": ("recurrent_conv_state_update_forward", {"weight": "none"}),
            "recurrent_silu": ("recurrent_silu_forward", {"weight": "none"}),
            "recurrent_split_conv_qkv": ("recurrent_split_conv_qkv_forward", {"weight": "none"}),
            "recurrent_qk_l2_norm": ("recurrent_qk_l2_norm_forward", {"weight": "none"}),
            "recurrent_norm_gate": ("recurrent_norm_gate_forward", {"weight": "fp32"}),
        }
        for op, (kernel_id, quant) in expected.items():
            with self.subTest(op=op):
                self.assertEqual(
                    build_ir.find_kernel(
                        registry,
                        op=op,
                        quant=quant,
                        mode="decode",
                    ),
                    kernel_id,
                )


if __name__ == "__main__":
    unittest.main()
