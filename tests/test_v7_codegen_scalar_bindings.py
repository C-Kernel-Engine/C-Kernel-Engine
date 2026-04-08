#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v7" / "scripts" / "codegen_train_runtime_v7.py"


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


codegen = _load_module("codegen_train_runtime_v7_scalar_bindings", SCRIPT)


class V7CodegenScalarBindingTests(unittest.TestCase):
    def test_runtime_seq_len_uses_active_tokens(self) -> None:
        expr = codegen._arg_scalar_expr("int", "rows", {}, param_source="runtime:seq_len")
        self.assertEqual(expr, "g_active_tokens")

    def test_recurrent_dims_derive_from_binding_source(self) -> None:
        cfg = {
            "q_dim": 64,
            "k_dim": 64,
            "v_dim": 64,
            "gate_dim": 4,
            "ssm_conv_kernel": 4,
            "ssm_conv_history": 3,
            "ssm_conv_channels": 192,
        }
        self.assertEqual(
            codegen._arg_scalar_expr("int", "history_len", cfg, param_source="dim:ssm_conv_history"),
            "3",
        )
        self.assertEqual(
            codegen._arg_scalar_expr("int", "num_channels", cfg, param_source="dim:ssm_conv_channels"),
            "192",
        )
        self.assertEqual(
            codegen._arg_scalar_expr("int", "state_dim", cfg, param_source="dim:recurrent_head_dim"),
            "16",
        )

    def test_generic_name_fallback_still_works(self) -> None:
        cfg = {"num_heads": 8, "num_kv_heads": 4, "head_dim": 16}
        self.assertEqual(codegen._arg_scalar_expr("int", "num_heads", cfg), "8")
        self.assertEqual(codegen._arg_scalar_expr("int", "num_kv_heads", cfg), "4")
        self.assertEqual(codegen._arg_scalar_expr("int", "head_dim", cfg), "16")

    def test_pointer_binding_source_prefers_weight_contract(self) -> None:
        tensors = {
            "act.alpha": {"dtype": "fp32"},
            "weight.layer.0.ssm_a": {"dtype": "fp32"},
        }
        expr = codegen._choose_tensor_for_ptr(
            "const float*",
            "a",
            io_inputs={"alpha": "act.alpha"},
            io_outputs={"gate": "act.gate"},
            io_weights={"ssm_a": "weight.layer.0.ssm_a"},
            tensors=tensors,
            tvars_f32={"act.alpha": "act_alpha", "weight.layer.0.ssm_a": "weight_ssm_a"},
            tvars_i32={},
            param_source="weight:ssm_a",
        )
        self.assertEqual(expr, "weight_ssm_a")

    def test_pointer_binding_source_resolves_weight_suffix_when_ir_key_differs(self) -> None:
        tensors = {
            "act.alpha": {"dtype": "fp32"},
            "weight.layer.0.ssm_a": {"dtype": "fp32"},
        }
        expr = codegen._choose_tensor_for_ptr(
            "const float*",
            "a",
            io_inputs={"alpha": "act.alpha"},
            io_outputs={"gate": "act.gate"},
            io_weights={"A": "weight.layer.0.ssm_a"},
            tensors=tensors,
            tvars_f32={"act.alpha": "act_alpha", "weight.layer.0.ssm_a": "weight_ssm_a"},
            tvars_i32={},
            param_source="weight:ssm_a",
        )
        self.assertEqual(expr, "weight_ssm_a")

    def test_pointer_binding_source_resolves_output_tensor(self) -> None:
        tensors = {
            "act.rstd": {"dtype": "fp32"},
        }
        expr = codegen._choose_tensor_for_ptr(
            "float*",
            "rstd_cache",
            io_inputs={"input": "act.input"},
            io_outputs={"rstd_cache": "act.rstd"},
            io_weights={},
            tensors=tensors,
            tvars_f32={"act.rstd": "act_rstd"},
            tvars_i32={},
            param_source="output:rstd_cache",
        )
        self.assertEqual(expr, "act_rstd")


if __name__ == "__main__":
    unittest.main()
