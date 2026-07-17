from __future__ import annotations

import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v8" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = _load("codegen_core_v8_capability_tests", SCRIPTS / "codegen_core_v8.py")
prefill = _load("codegen_prefill_v8_capability_tests", SCRIPTS / "codegen_prefill_v8.py")
resolver = _load(
    "resolve_attention_contracts_v8_capability_tests",
    SCRIPTS / "resolve_attention_contracts_v8.py",
)

GOVERNED_SYMBOLS = {
    "gemv_q4_k_q8_k",
    "gemv_q6_k_q8_k",
    "gemm_nt_q4_k_q8_k",
    "gemm_nt_q6_k_q8_k",
    "quantize_row_q8_0",
    "quantize_row_q8_k",
}


def _execution(weight: str, *, prefill_mode: bool = False) -> dict:
    is_q4 = weight == "q4_k"
    prefix = "gemm_nt" if prefill_mode else "gemv"
    return {
        "numerical_contract": f"{weight}_x_q8_k_fp32_block_order",
        "implementation": {
            "weight_storage": {
                "format": weight,
                "block_elements": 256,
                "block_bytes": 144 if is_q4 else 210,
            },
            "activation_storage": {"format": "q8_k", "block_elements": 256},
            "diagnostic_providers": {
                "fp32_activation": f"{prefix}_{weight}",
                **(
                    {"row_quantized": f"renamed_{weight}_row_provider"}
                    if prefill_mode
                    else {}
                ),
            },
        },
    }


def _decode_op(op_name: str, weight: str) -> dict:
    return {
        "idx": 1,
        "op": op_name,
        "function": f"renamed_{weight}_production",
        "layer": 0,
        "section": "body",
        "resolved_execution": _execution(weight),
        "args": [
            {"name": "y", "expr": "Y"},
            {"name": "W", "expr": "W"},
            {"name": "x_q8", "expr": "XQ"},
            {"name": "M", "expr": "M"},
            {"name": "K", "expr": "K"},
        ],
    }


def _prefill_op(op_name: str, weight: str) -> dict:
    return {
        "idx": 1,
        "op": op_name,
        "function": f"renamed_{weight}_production",
        "layer": 0,
        "section": "body",
        "resolved_execution": _execution(weight, prefill_mode=True),
        "args": [
            {"name": "A", "expr": "A"},
            {"name": "B", "expr": "B"},
            {"name": "bias", "expr": "BIAS"},
            {"name": "C", "expr": "C"},
            {"name": "M", "expr": "M", "source": "dim:_m"},
            {"name": "N", "expr": "N"},
            {"name": "K", "expr": "K"},
        ],
    }


def _quantize_op(
    op_name: str,
    storage: str,
    *,
    rows: str = "ROWS",
    prefill_batch: bool = False,
) -> dict:
    is_q8_k = storage == "q8_k"
    function = f"renamed_{storage}_quantizer"
    capability = {
        "schema_version": 1,
        "kernel_id": f"fake_{storage}_quantizer",
        "operator_family": "activation_quantization",
        "function": function,
        "output_storage": {
            "format": storage,
            "block_elements": 256 if is_q8_k else 32,
            "block_elements_symbol": "QK_K" if is_q8_k else "QK8_0",
            "c_block_type": "block_q8_K" if is_q8_k else "block_q8_0",
        },
    }
    if prefill_batch:
        capability["prefill_batch"] = {
            "function": "grouped_quantizer",
            "row_group": 4,
            "tail_function": function,
            "rounding_contract": "test_grouped_rounding",
        }
    return {
        "idx": 2,
        "op": op_name,
        "function": function,
        "layer": 0,
        "section": "body",
        "resolved_codegen_capability": capability,
        "args": [
            {"name": "x", "expr": "X"},
            {"name": "y", "expr": "Y"},
            {"name": "k", "expr": "K"},
            {"name": "rows", "expr": rows},
        ],
    }


class V8CodegenCapabilityTests(unittest.TestCase):
    def test_kernel_maps_own_q4_q6_storage_and_diagnostic_providers(self) -> None:
        kernels = resolver.load_kernel_execution_capabilities()["kernels"]
        expected = {
            "gemv_q4_k_q8_k": ("q4_k", 144, "gemv_q4_k", None),
            "gemv_q6_k_q8_k": ("q6_k", 210, "gemv_q6_k", None),
            "gemm_nt_q4_k_q8_k": ("q4_k", 144, "gemm_nt_q4_k", "gemv_q4_k_q8_k"),
            "gemm_nt_q6_k_q8_k": ("q6_k", 210, "gemm_nt_q6_k", "gemv_q6_k_q8_k"),
        }
        for kernel_id, (weight, block_bytes, fp32, row) in expected.items():
            with self.subTest(kernel=kernel_id):
                implementation = kernels[kernel_id]["implementation"]
                self.assertEqual(implementation["weight_storage"]["format"], weight)
                self.assertEqual(implementation["weight_storage"]["block_bytes"], block_bytes)
                self.assertEqual(implementation["activation_storage"]["format"], "q8_k")
                self.assertEqual(implementation["diagnostic_providers"]["fp32_activation"], fp32)
                self.assertEqual(implementation["diagnostic_providers"].get("row_quantized"), row)

    def test_missing_storage_capability_is_a_hard_failure(self) -> None:
        source = ROOT / "version" / "v8" / "kernel_maps" / "gemv_q4_k_q8_k.json"
        document = json.loads(source.read_text(encoding="utf-8"))
        del document["implementation"]["weight_storage"]
        with tempfile.TemporaryDirectory(prefix="cke_missing_linear_storage_") as td:
            path = Path(td) / source.name
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(resolver.ContractError, "no explicit storage capability"):
                resolver.load_kernel_execution_capabilities(Path(td))

    def test_storage_capability_must_match_numerical_contract(self) -> None:
        source = ROOT / "version" / "v8" / "kernel_maps" / "gemv_q4_k_q8_k.json"
        document = json.loads(source.read_text(encoding="utf-8"))
        document["implementation"]["weight_storage"]["format"] = "q6_k"
        with tempfile.TemporaryDirectory(prefix="cke_wrong_linear_storage_") as td:
            path = Path(td) / source.name
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(resolver.ContractError, "disagrees with its contract"):
                resolver.load_kernel_execution_capabilities(Path(td))

    def test_missing_diagnostic_provider_is_a_hard_failure(self) -> None:
        source = ROOT / "version" / "v8" / "kernel_maps" / "gemm_nt_q4_k_q8_k.json"
        document = json.loads(source.read_text(encoding="utf-8"))
        del document["implementation"]["diagnostic_providers"]["row_quantized"]
        with tempfile.TemporaryDirectory(prefix="cke_missing_linear_diagnostic_") as td:
            path = Path(td) / source.name
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(resolver.ContractError, "no row-quantized provider"):
                resolver.load_kernel_execution_capabilities(Path(td))

    def test_decode_projection_categories_ignore_production_symbol_spelling(self) -> None:
        for weight in ("q4_k", "q6_k"):
            for op_name in ("q_proj", "k_proj", "v_proj", "logits", "out_proj", "mlp_down"):
                with self.subTest(weight=weight, op=op_name):
                    code = core.emit_op(_decode_op(op_name, weight))
                    self.assertIn(f"renamed_{weight}_production(", code)
                    self.assertIn(f"gemv_{weight}(", code)

    def test_decode_gate_up_uses_map_owned_row_layout(self) -> None:
        for weight, block_bytes in (("q4_k", 144), ("q6_k", 210)):
            with self.subTest(weight=weight):
                code = core.emit_op(_decode_op("mlp_gate_up", weight))
                self.assertIn(f"/ 256u) * {block_bytes}u", code)
                self.assertIn(f"gemv_{weight}(", code)
                self.assertIn(f"renamed_{weight}_production(", code)

    def test_prefill_projection_categories_ignore_production_symbol_spelling(self) -> None:
        for weight in ("q4_k", "q6_k"):
            with self.subTest(weight=weight, op="out_proj"):
                code = prefill.emit_prefill_op(_prefill_op("out_proj", weight), 1, {})
                self.assertIn(f"renamed_{weight}_production(", code)
                self.assertIn(f"gemm_nt_{weight}(", code)
            with self.subTest(weight=weight, op="mlp_gate_up"):
                code = prefill.emit_prefill_op(_prefill_op("mlp_gate_up", weight), 1, {})
                self.assertIn(f"renamed_{weight}_row_provider(", code)
                self.assertIn(f"gemm_nt_{weight}(", code)

    def test_prefill_override_requires_resolved_capability(self) -> None:
        op = _prefill_op("mlp_down", "q4_k")
        del op["resolved_execution"]
        with self.assertRaisesRegex(RuntimeError, "requires resolved Q4/Q6"):
            prefill._emit_prefill_gemm_fp32_override(
                op,
                1,
                debug_flag_name="debug",
                debug_input_name="input",
            )

    def test_quantization_maps_own_exact_output_storage_abi(self) -> None:
        expected = {
            "quantize_row_q8_0.json": ("q8_0", 32, "QK8_0", "block_q8_0"),
            "quantize_row_q8_k.json": ("q8_k", 256, "QK_K", "block_q8_K"),
            "quantize_row_q8_k_llama_repack.json": (
                "q8_k",
                256,
                "QK_K",
                "block_q8_K",
            ),
        }
        for filename, storage in expected.items():
            with self.subTest(map=filename):
                document = json.loads(
                    (ROOT / "version" / "v8" / "kernel_maps" / filename).read_text(
                        encoding="utf-8"
                    )
                )
                capability = document["codegen_capability"]
                self.assertEqual(capability["operator_family"], "activation_quantization")
                self.assertEqual(
                    (
                        capability["output_storage"]["format"],
                        capability["output_storage"]["block_elements"],
                        capability["output_storage"]["block_elements_symbol"],
                        capability["output_storage"]["c_block_type"],
                    ),
                    storage,
                )

    def test_quantization_codegen_capability_rejects_function_drift(self) -> None:
        path = ROOT / "version" / "v8" / "kernel_maps" / "quantize_row_q8_k.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["codegen_capability"]["function"] = "wrong_quantizer"
        build_ir = _load("build_ir_v8_quant_drift_tests", SCRIPTS / "build_ir_v8.py")
        with self.assertRaisesRegex(RuntimeError, "advertises function"):
            build_ir._validated_kernel_codegen_capability(document["id"], document)

    def test_decode_quantization_uses_storage_capability_not_symbol(self) -> None:
        for storage, symbol, block_type in (
            ("q8_0", "QK8_0", "block_q8_0"),
            ("q8_k", "QK_K", "block_q8_K"),
        ):
            with self.subTest(storage=storage):
                op = _quantize_op("quantize_input_1", storage)
                code = core.emit_op(op)
                self.assertIn(f"renamed_{storage}_quantizer(", code)
                self.assertIn(f"/ {symbol}) * sizeof({block_type})", code)

    def test_prefill_quantization_uses_storage_capability_not_symbol(self) -> None:
        for storage, symbol, block_type in (
            ("q8_0", "QK8_0", "block_q8_0"),
            ("q8_k", "QK_K", "block_q8_K"),
        ):
            with self.subTest(storage=storage):
                op = _quantize_op("quantize_input_0", storage)
                code = prefill.emit_prefill_op(op, 2, {})
                self.assertIn(f"renamed_{storage}_quantizer(", code)
                self.assertIn(f"/ {symbol}) * sizeof({block_type})", code)

    def test_prefill_quantization_uses_declared_grouped_provider(self) -> None:
        op = _quantize_op("quantize_input_0", "q8_k", prefill_batch=True)
        code = prefill.emit_prefill_op(op, 2, {})
        self.assertIn("grouped_quantizer(_x_base, (void*)_y_base, num_tokens, _k);", code)
        self.assertNotIn("for (int _t = 0; _t < num_tokens; ++_t)", code)
        decode_code = core.emit_op(op)
        self.assertIn("renamed_q8_k_quantizer(", decode_code)
        self.assertNotIn("grouped_quantizer(", decode_code)

    def test_quantization_without_resolved_capability_hard_fails(self) -> None:
        op = _quantize_op("quantize_input_0", "q8_k")
        del op["resolved_codegen_capability"]
        with self.assertRaisesRegex(RuntimeError, "requires resolved map-owned"):
            core.emit_op(op)
        with self.assertRaisesRegex(RuntimeError, "requires resolved map-owned"):
            prefill.emit_prefill_op(op, 2, {})

    def test_quantize_provider_resolution_is_unique_and_map_driven(self) -> None:
        registry = {
            "kernels": [
                {"id": "renamed_q8_0", "op": "quantize", "quant": {"output": "q8_0"}},
                {
                    "id": "renamed_q8_k",
                    "op": "quantize",
                    "quant": {"output": "q8_k"},
                    "codegen_capability": {
                        "rounding_contract": "canonical_q8_k_row_nearest"
                    },
                },
                {
                    "id": "renamed_q8_k_grouped",
                    "op": "quantize",
                    "quant": {"output": "q8_k"},
                    "codegen_capability": {
                        "rounding_contract": "llama_repack_q8_k_4row_nearest_even"
                    },
                },
            ]
        }
        build_ir = _load("build_ir_v8_quant_capability_tests", SCRIPTS / "build_ir_v8.py")
        self.assertEqual(
            build_ir.get_quantize_kernel_for_activation(
                registry, "q8_k", "canonical_q8_k_row_nearest"
            ),
            "renamed_q8_k",
        )
        self.assertEqual(
            build_ir.get_quantize_kernel_for_activation(
                registry,
                "q8_k",
                "llama_repack_q8_k_4row_nearest_even",
            ),
            "renamed_q8_k_grouped",
        )
        registry["kernels"].append(
            {
                "id": "ambiguous_q8_k",
                "op": "quantize",
                "quant": {"output": "q8_k"},
                "codegen_capability": {
                    "rounding_contract": "canonical_q8_k_row_nearest"
                },
            }
        )
        with self.assertRaisesRegex(RuntimeError, "resolved 2 quantization providers"):
            build_ir.get_quantize_kernel_for_activation(
                registry, "q8_k", "canonical_q8_k_row_nearest"
            )

    def test_q4_q6_consumers_select_phase_specific_quantization_contracts(self) -> None:
        build_ir = _load("build_ir_v8_quant_consumer_contract_tests", SCRIPTS / "build_ir_v8.py")
        registry = build_ir.load_kernel_registry()
        self.assertEqual(
            build_ir.get_kernel_activation_quantization_contract(
                registry, "gemm_nt_q4_k_q8_k", "prefill"
            ),
            ("q8_k", "llama_repack_q8_k_4row_nearest_even"),
        )
        self.assertEqual(
            build_ir.get_kernel_activation_quantization_contract(
                registry, "gemm_nt_q6_k_q8_k", "prefill"
            ),
            ("q8_k", "canonical_q8_k_row_nearest"),
        )
        self.assertEqual(
            build_ir.get_kernel_activation_quantization_contract(
                registry, "gemv_q4_k_q8_k", "decode"
            ),
            ("q8_k", "canonical_q8_k_row_nearest"),
        )

    def test_codegen_conditions_do_not_name_governed_q4_q6_providers(self) -> None:
        for filename in ("codegen_core_v8.py", "codegen_prefill_v8.py"):
            tree = ast.parse((SCRIPTS / filename).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.If, ast.IfExp, ast.While)):
                    continue
                literals = {
                    child.value
                    for child in ast.walk(node.test)
                    if isinstance(child, ast.Constant) and isinstance(child.value, str)
                }
                with self.subTest(file=filename, line=node.lineno):
                    self.assertFalse(literals & GOVERNED_SYMBOLS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
