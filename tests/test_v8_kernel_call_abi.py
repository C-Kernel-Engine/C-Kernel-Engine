from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
MAPS = ROOT / "version" / "v8" / "kernel_maps"
SCHEMA = ROOT / "version" / "v8" / "schemas" / "kernel_call_abi.schema.json"
REGISTRY = MAPS / "KERNEL_REGISTRY.json"
EXCLUDED = {"KERNEL_REGISTRY.json", "kernel_bindings.json", "kernel_bindings.overlay.json"}
BUILD_IR = ROOT / "version" / "v8" / "scripts" / "build_ir_v8.py"
EXPECTED_GOVERNED_MAP_COUNT = 42
QWEN3VL_PARITY_PROVIDERS = {
    "attention_forward_decode_head_major_gqa_flash_f16cache_contract",
    "attention_forward_causal_head_major_gqa_prefill_append_f16cache_flash_auto_qtile64",
    "attention_forward_causal_head_major_gqa_prefill_append_f16cache_single_range",
    "qk_norm_forward_fp64_sum",
    "rmsnorm_forward_fp64_sum",
    "swiglu_forward_ggml",
}


if str(BUILD_IR.parent) not in sys.path:
    sys.path.insert(0, str(BUILD_IR.parent))
SPEC = importlib.util.spec_from_file_location("build_ir_v8_call_abi_tests", BUILD_IR)
assert SPEC is not None and SPEC.loader is not None
build_ir_v8 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = build_ir_v8
SPEC.loader.exec_module(build_ir_v8)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class V8KernelCallABITests(unittest.TestCase):
    def test_all_contract_governed_maps_own_valid_call_abi(self) -> None:
        validator = Draft202012Validator(load_json(SCHEMA))
        governed = 0
        for path in sorted(MAPS.glob("*.json")):
            if path.name in EXCLUDED:
                continue
            doc = load_json(path)
            is_governed = bool(
                doc.get("numerical_capabilities")
                or doc.get("supported_reductions")
                or ("numerical_contract" in doc and "production" in doc)
            )
            if not is_governed:
                continue
            governed += 1
            with self.subTest(kernel=doc.get("id")):
                errors = sorted(
                    validator.iter_errors(doc.get("call_abi")),
                    key=lambda error: tuple(str(part) for part in error.absolute_path),
                )
                self.assertEqual(errors, [])
        self.assertEqual(governed, EXPECTED_GOVERNED_MAP_COUNT)

    def test_map_owned_abis_do_not_exist_in_legacy_registries(self) -> None:
        call_abis = build_ir_v8.load_kernel_call_abis()
        legacy = build_ir_v8.load_kernel_bindings()
        self.assertEqual(len(call_abis), EXPECTED_GOVERNED_MAP_COUNT)
        self.assertTrue(QWEN3VL_PARITY_PROVIDERS.issubset(call_abis))
        for kernel_id, entry in call_abis.items():
            with self.subTest(kernel=kernel_id):
                self.assertNotIn(kernel_id, legacy)
                self.assertNotIn(entry["function"], legacy)

    def test_generated_registry_retains_exact_map_owned_abis(self) -> None:
        registry = {
            kernel["id"]: kernel
            for kernel in load_json(REGISTRY)["kernels"]
        }
        for kernel_id, entry in build_ir_v8.load_kernel_call_abis().items():
            with self.subTest(kernel=kernel_id):
                self.assertIn(kernel_id, registry)
                self.assertEqual(registry[kernel_id].get("call_abi"), entry["call_abi"])

    def test_duplicate_map_and_legacy_ownership_is_a_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cke_call_abi_duplicate_") as td:
            root = Path(td)
            (root / "synthetic.json").write_text(
                json.dumps({
                    "id": "synthetic",
                    "impl": {"function": "synthetic_fn"},
                    "call_abi": {"version": 1, "params": []},
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "legacy bindings still define"):
                build_ir_v8.load_kernel_call_abis(
                    root,
                    legacy_bindings={"synthetic_fn": {"params": []}},
                )

    def test_unknown_call_source_is_a_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cke_call_abi_source_") as td:
            root = Path(td)
            (root / "synthetic.json").write_text(
                json.dumps({
                    "id": "synthetic",
                    "impl": {"function": "synthetic_fn"},
                    "call_abi": {
                        "version": 1,
                        "params": [{"name": "x", "source": "guessed:model_default"}],
                    },
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "unsupported source"):
                build_ir_v8.load_kernel_call_abis(root, legacy_bindings={})

    def test_malformed_optional_call_metadata_is_a_hard_failure(self) -> None:
        bad_params = [
            {"name": "x", "source": "null:guessed"},
            {"name": "x", "source": "null", "cast": ""},
            {"name": "x", "source": "null", "alt": ["x", "x"]},
        ]
        for index, param in enumerate(bad_params):
            with self.subTest(param=param), tempfile.TemporaryDirectory(
                prefix=f"cke_call_abi_metadata_{index}_"
            ) as td:
                root = Path(td)
                (root / "synthetic.json").write_text(
                    json.dumps({
                        "id": "synthetic",
                        "impl": {"function": "synthetic_fn"},
                        "call_abi": {"version": 1, "params": [param]},
                    }),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(RuntimeError, "HARD CALL ABI FAULT"):
                    build_ir_v8.load_kernel_call_abis(root, legacy_bindings={})

    def test_resolved_operation_cannot_fall_back_to_legacy_binding(self) -> None:
        lowered = {
            "config": {},
            "operations": [{
                "idx": 0,
                "kernel": "im2patch",
                "function": "im2patch",
                "op": "patchify",
                "layer": -1,
                "section": "header",
                "resolved_contract": {"function": "im2patch", "kernel_id": "im2patch"},
            }],
        }
        call_ir = build_ir_v8.generate_ir_lower_3(lowered, "prefill")
        self.assertIn("missing map-owned call_abi", call_ir["operations"][0]["errors"][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
