from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_nightly_runner():
    path = ROOT / "scripts" / "nightly_runner.py"
    spec = importlib.util.spec_from_file_location("nightly_runner_v8_dsl_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


nightly = _load_nightly_runner()


class V8DSLNightlyRegistrationTests(unittest.TestCase):
    EXPECTED_TARGETS = {
        "test-v8-dsl-policy": "v8 DSL Zero-Hardcoding Policy",
        "test-numerical-contracts": "v8 Numerical Kernel Contracts",
        "test-v8-template-circuit-audit": "v8 Template Circuit/Dataflow Audit",
    }

    def test_instella_circuit_codegen_is_a_visible_nightly_row(self) -> None:
        suite = nightly.TEST_SUITES.get("v8_instella_moe_circuit_contracts")
        self.assertIsNotNone(suite)
        self.assertEqual(suite.name, "Instella-MoE Circuit/Codegen Contracts")
        self.assertEqual(suite.category, "inference")
        self.assertEqual(
            suite.test_file,
            ROOT / "tests" / "test_v8_instella_moe_bringup.py",
        )

        yarn_fp32 = nightly.TEST_SUITES.get("yarn_rope_explicit_positions")
        yarn_bf16 = nightly.TEST_SUITES.get("yarn_rope_explicit_positions_bf16")
        self.assertEqual(yarn_fp32.category, "kernels")
        self.assertEqual(yarn_bf16.category, "bf16")

    def test_dense_qwen_metadata_has_a_separate_nightly_target(self) -> None:
        entry = nightly.MAKE_TARGETS["v8_qwen38_dense_contracts"]
        self.assertEqual(entry["category"], "inference")
        self.assertEqual(entry["target"], "test-v8-qwen38-dense-contracts")
        self.assertIn("v8_qwen38_dense_contracts", nightly.NIGHTLY_PROFILES["demo-readiness"])

    def test_flash_numerical_providers_have_a_nightly_target(self) -> None:
        entry = nightly.MAKE_TARGETS["v8_qwen38_flash_contracts"]
        self.assertEqual(entry["category"], "parity")
        self.assertEqual(entry["target"], "test-v8-qwen38-flash-contracts")

    def test_cohere_and_laguna_contracts_are_a_visible_nightly_row(self) -> None:
        entry = nightly.MAKE_TARGETS.get("v8_cohere_laguna_contracts")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["name"], "Cohere/Laguna Compiler/Circuit Contracts")
        self.assertEqual(entry["category"], "inference")
        self.assertEqual(entry["target"], "test-v8-cohere-laguna-contracts")
        self.assertIn("v8_cohere_laguna_contracts", nightly.NIGHTLY_PROFILES["demo-readiness"])

    def test_cohere_and_laguna_numerical_providers_are_in_the_nightly_gate(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        for test_file in (
            "unittest/test_moe_swiglu_q4k_mixed_parallel.py",
            "unittest/test_attn_gate_softplus_mul.py",
            "tests/test_v8_rope_split_direct_parallel.py",
            "unittest/bf16/test_vision_position_fp32_interp_bf16.py",
        ):
            with self.subTest(test_file=test_file):
                self.assertIn(f"@$(PYTHON) {test_file}", makefile)

    def test_full_dsl_gate_dependencies_are_explicit_nightly_rows(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        match = re.search(r"^test-v8-dsl:\s+([^\n]+)$", makefile, re.MULTILINE)
        self.assertIsNotNone(match, "Makefile must define the aggregate test-v8-dsl gate")
        dependencies = set(match.group(1).split())
        self.assertEqual(dependencies, set(self.EXPECTED_TARGETS))

        registered = {
            entry["target"]: entry
            for entry in nightly.MAKE_TARGETS.values()
            if entry.get("target") in self.EXPECTED_TARGETS
        }
        self.assertEqual(set(registered), set(self.EXPECTED_TARGETS))
        for target, expected_name in self.EXPECTED_TARGETS.items():
            with self.subTest(target=target):
                self.assertEqual(registered[target]["name"], expected_name)
                self.assertIn(registered[target]["category"], {"inference", "parity"})

    def test_report_documents_the_three_visible_rows(self) -> None:
        source = (ROOT / "docs" / "site" / "_pages" / "test-report.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("v8 DSL and Codegen Contracts", source)
        self.assertIn('id="v8-dsl-contract-dashboard"', source)
        self.assertIn("function renderV8DSLContracts(results)", source)
        self.assertIn("renderV8DSLContracts(data.results || [])", source)
        for target in self.EXPECTED_TARGETS:
            with self.subTest(target=target):
                self.assertIn(f"<code>{target}</code>", source)

    def test_report_publishes_kernel_allocation_debt(self) -> None:
        source = (ROOT / "docs" / "site" / "_pages" / "test-report.html").read_text(encoding="utf-8")
        workflow = (ROOT / ".github" / "workflows" / "nightly.yml").read_text(encoding="utf-8")
        self.assertIn("v8 Kernel Allocation Debt", source)
        self.assertIn('id="kernel-allocation-tbody"', source)
        self.assertIn("function renderKernelAllocation(data)", source)
        self.assertIn("renderKernelAllocation(data.kernel_allocation || null)", source)
        self.assertIn('data["kernel_allocation"] = kernel_allocation_payload', workflow)
        self.assertIn('"allocation_sites": [', workflow)
        self.assertIn("row.path || '-')}:${Number(row.line || 0)", source)
        self.assertIn("mapped_allocating_without_scratch_contract", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
