import importlib.util
import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "build_ir_v8.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("build_ir_v8_provider_selection", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
build_ir = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_ir)


def provider(
    kernel_id,
    *,
    activation="fp32",
    status=None,
    priority=0,
    group="linear.fp32.v1",
):
    value = {
        "id": kernel_id,
        "op": "gemm",
        "variant": "forward",
        "quant": {"weight": "fp32", "activation": activation},
        "modes": {"inference": True, "backward": False},
    }
    if status is not None:
        value["selection"] = {
            "status": status,
            "priority": priority,
            "equivalence_group": group,
            "phases": ["prefill", "decode"],
        }
    return value


class ProviderSelectionTests(unittest.TestCase):
    def resolve(self, providers, *, prefer_q8=False):
        return build_ir.find_kernel(
            {"kernels": providers},
            op="gemm",
            quant={"weight": "fp32"},
            mode="prefill",
            prefer_q8_activation=prefer_q8,
        )

    def test_higher_priority_production_provider_wins(self):
        self.assertEqual(
            self.resolve(
                [
                    provider("baseline", status="production", priority=100),
                    provider("packed", status="production", priority=200),
                ]
            ),
            "packed",
        )

    def test_candidate_does_not_override_production_by_priority(self):
        self.assertEqual(
            self.resolve(
                [
                    provider("baseline", status="production", priority=10),
                    provider("experiment", status="candidate", priority=1000),
                ]
            ),
            "baseline",
        )

    def test_dtype_policy_precedes_provider_priority(self):
        self.assertEqual(
            self.resolve(
                [
                    provider("fp32", status="production", priority=1000),
                    provider(
                        "q8",
                        activation="q8_0",
                        status="production",
                        priority=1,
                    ),
                ],
                prefer_q8=True,
            ),
            "q8",
        )

    def test_equal_priority_equivalent_providers_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "equal-priority production providers"):
            self.resolve(
                [
                    provider("one", status="production", priority=100),
                    provider("two", status="production", priority=100),
                ]
            )

    def test_priority_cannot_compare_different_equivalence_groups(self):
        with self.assertRaisesRegex(RuntimeError, "different equivalence groups"):
            self.resolve(
                [
                    provider("one", status="production", priority=100, group="linear.a.v1"),
                    provider("two", status="production", priority=100, group="linear.b.v1"),
                ]
            )

    def test_candidate_only_provider_is_not_selected(self):
        self.assertIsNone(
            self.resolve([provider("experiment", status="candidate", priority=1000)])
        )

    def test_provider_phase_is_required(self):
        decode_only = provider("decode_only", status="production", priority=100)
        decode_only["selection"]["phases"] = ["decode"]
        self.assertIsNone(self.resolve([decode_only]))

    def test_malformed_selection_metadata_fails_closed(self):
        malformed = provider("bad")
        malformed["selection"] = {
            "status": "production",
            "priority": "high",
            "equivalence_group": "linear.fp32.v1",
            "phases": ["prefill"],
        }
        with self.assertRaisesRegex(RuntimeError, "priority must be an integer"):
            self.resolve([malformed])

    def test_checked_in_selection_metadata_matches_schema(self):
        schema = json.loads(
            (ROOT / "version" / "v8" / "schemas" / "kernel_provider_selection.schema.json").read_text()
        )
        validator = Draft202012Validator(schema)
        maps = ROOT / "version" / "v8" / "kernel_maps"
        selected = []
        for path in maps.glob("*.json"):
            document = json.loads(path.read_text())
            if "selection" not in document:
                continue
            selected.append(document["id"])
            errors = sorted(validator.iter_errors(document["selection"]), key=str)
            self.assertEqual(errors, [], path.name)
        self.assertEqual(len(selected), 52)

    def test_gemma_q5_prefill_providers_are_production_selected(self):
        maps = ROOT / "version" / "v8" / "kernel_maps"
        expected = {
            "gemm_nt_q5_1": (
                "q5_1_weight_q8_1_internal_fp32_output",
                "gemm_nt_q5_1_q8_1_parallel_dispatch",
            ),
            "gemm_nt_q5_k": (
                "q5_k_weight_q8_k_input_avx2_fma_fp32_output",
                "gemm_nt_q5_k_parallel_dispatch_with_scratch",
            ),
        }
        registry = {
            kernel["id"]: kernel
            for kernel in json.loads(
                (maps / "KERNEL_REGISTRY.json").read_text()
            )["kernels"]
        }
        for kernel_id, (group, function) in expected.items():
            with self.subTest(kernel_id=kernel_id):
                document = json.loads((maps / f"{kernel_id}.json").read_text())
                self.assertEqual(document["selection"]["status"], "production")
                self.assertEqual(document["selection"]["equivalence_group"], group)
                self.assertIn("prefill", document["selection"]["phases"])
                self.assertIn("call_abi", document)
                self.assertEqual(document["impl"]["function"], function)
                self.assertEqual(document["production"]["function"], function)
                self.assertEqual(registry[kernel_id]["impl"]["function"], function)
                self.assertEqual(registry[kernel_id]["production"]["function"], function)
    def test_embedding_production_priorities_preserve_dtype_dispatch(self):
        registry = json.loads(
            (ROOT / "version" / "v8" / "kernel_maps" / "KERNEL_REGISTRY.json").read_text()
        )
        expected = {
            "fp32": "embedding_forward_fp32",
            "bf16": "embedding_forward_bf16_fp32",
            "q4_k": "embedding_forward_q4_k",
            "q5_0": "embedding_forward_q5_0",
            "q6_k": "embedding_forward_q6_k",
            "q8_0": "embedding_forward_q8_0",
        }
        for weight_dtype, kernel_id in expected.items():
            with self.subTest(weight_dtype=weight_dtype):
                self.assertEqual(
                    build_ir.find_kernel(
                        registry,
                        op="embedding",
                        quant={"weight": weight_dtype},
                        mode="prefill",
                        prefer_q8_activation=False,
                    ),
                    kernel_id,
                )


if __name__ == "__main__":
    unittest.main()
