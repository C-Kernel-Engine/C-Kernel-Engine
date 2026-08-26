from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_nightly_runner():
    path = ROOT / "scripts" / "nightly_runner.py"
    spec = importlib.util.spec_from_file_location("nightly_runner_q6k_perf_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Qwen36Q6KM4PerformanceGateTests(unittest.TestCase):
    def test_nightly_registers_qwen36_shape_gate(self) -> None:
        nightly = _load_nightly_runner()
        entry = nightly.MAKE_TARGETS["qwen36_q6k_m4_performance"]
        self.assertEqual(entry["category"], "bench")
        self.assertEqual(entry["target"], "test-qwen36-q6k-m4-performance")
        self.assertGreaterEqual(entry["timeout_sec"], 120)

    def test_make_gate_checks_exactness_and_speed_at_real_shape(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        match = re.search(
            r"^test-qwen36-q6k-m4-performance:.*?(?=^\.PHONY:|\Z)",
            makefile,
            re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(match)
        recipe = match.group(0)
        self.assertIn("--mode compare-m4", recipe)
        self.assertIn("CK_QWEN36_Q6_M:-23", recipe)
        self.assertIn("CK_QWEN36_Q6_N:-5120", recipe)
        self.assertIn("CK_QWEN36_Q6_K:-17408", recipe)
        self.assertIn("CK_QWEN36_Q6_MIN_SPEEDUP:-1.05", recipe)
        self.assertIn("CK_QWEN36_Q6_RECURRENT_M:-23", recipe)
        self.assertIn("CK_QWEN36_Q6_RECURRENT_N:-10240", recipe)
        self.assertIn("CK_QWEN36_Q6_RECURRENT_K:-5120", recipe)
        self.assertIn("CK_QWEN36_Q6_RECURRENT_MIN_SPEEDUP:-1.10", recipe)

    def test_performance_gate_executes_dispatch_boundaries(self) -> None:
        benchmark = (
            ROOT / "benchmarks" / "bench_q6k_prefill_tile.py"
        ).read_text(encoding="utf-8")
        self.assertIn("for boundary_m in (63, 64):", benchmark)
        self.assertIn("'--mode', 'default', '--verify-row-exact'", benchmark)
        self.assertIn("boundary_exact=", benchmark)

    def test_nightly_excludes_candidate_prepared_q6_gate(self) -> None:
        nightly = _load_nightly_runner()
        self.assertNotIn("q6k_prepared_performance", nightly.MAKE_TARGETS)

        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        match = re.search(
            r"^test-q6k-prepared-performance:.*?(?=^\.PHONY:|\Z)",
            makefile,
            re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(match)
        recipe = match.group(0)
        self.assertIn("--mode compare-prepared", recipe)
        self.assertIn("--m 128 --n 2560 --k 10496", recipe)
        self.assertIn("CK_Q6K_PREPARED_MIN_SPEEDUP:-1.15", recipe)


if __name__ == "__main__":
    unittest.main()
