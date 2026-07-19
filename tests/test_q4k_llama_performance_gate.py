from __future__ import annotations

import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_nightly_runner():
    path = ROOT / "scripts" / "nightly_runner.py"
    spec = importlib.util.spec_from_file_location("nightly_runner_q4k_perf_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Q4KLlamaPerformanceGateTests(unittest.TestCase):
    def test_nightly_registers_comparative_benchmark(self) -> None:
        nightly = _load_nightly_runner()
        entry = nightly.MAKE_TARGETS["q4k_q8k_llama_performance"]
        self.assertEqual(entry["category"], "bench")
        self.assertEqual(entry["target"], "test-q4k-q8k-llama-performance")
        self.assertGreaterEqual(entry["timeout_sec"], 120)

    def test_make_gate_uses_production_shape_and_hard_ratio(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        match = re.search(
            r"^test-q4k-q8k-llama-performance:.*?(?=^\.PHONY:|\Z)",
            makefile,
            re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(match)
        recipe = match.group(0)
        self.assertIn("CK_Q4K_PERF_M:-1028", recipe)
        self.assertIn("CK_Q4K_PERF_N:-4096", recipe)
        self.assertIn("CK_Q4K_PERF_K:-4096", recipe)
        self.assertIn("CK_Q4K_LLAMA_MAX_RATIO:-2.5", recipe)
        self.assertIn("CK_Q4K_PERF_DEFAULT_THREADS", recipe)
        self.assertIn("CK_Q4K_PERF_CAPACITY_THREADS", recipe)
        self.assertIn("CK_Q4K_LLAMA_THREADS", recipe)
        self.assertIn("--perf", recipe)

    def test_vnni_provider_is_an_explicit_kernel_map_capability(self) -> None:
        path = ROOT / "version" / "v8" / "kernel_maps" / "gemm_nt_q4_k_q8_k.json"
        kernel = json.loads(path.read_text(encoding="utf-8"))
        candidates = {
            row["name"]: row
            for row in kernel["phase_selection"]["prefill"]["candidates"]
        }
        provider = candidates["packed_vnni_x8_split_min_4m8n"]
        self.assertEqual(
            provider["function"],
            "gemm_nt_q4_k_packed_vnni_x8_q8_k_split_min_threaded_4m",
        )
        self.assertEqual(provider["layout"], "q4_k_packed_vnni_x8")
        self.assertEqual(provider["activation_layout"], "canonical_q8_k")
        threading = provider["threading"]
        self.assertEqual(threading["default_concurrency"], "physical_cores")
        self.assertEqual(
            threading["capacity"], "physical_cores_plus_half_available_smt"
        )
        self.assertEqual(threading["smt_extension"], "half_available_siblings")
        self.assertEqual(
            threading["shape_minimum"], {"M": 512, "N": 4096, "K": 4096}
        )
        self.assertEqual(threading["reduction_order_effect"], "none")
        self.assertIn("avx_vnni", provider["requires"])
        self.assertIn("pairwise_split_min_reduction", provider["requires"])

        source = (
            ROOT / "version" / "v8" / "src" / "ck_parallel_prefill_v8.c"
        ).read_text(encoding="utf-8")
        selector = re.search(
            r"static int ck_select_q4k_vnni_active_threads\(.*?^}",
            source,
            re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(selector)
        limits = re.search(
            r"M < (\d+) \|\| N < (\d+) \|\| K < (\d+)",
            selector.group(0),
        )
        self.assertIsNotNone(limits)
        self.assertEqual(
            tuple(map(int, limits.groups())),
            (
                threading["shape_minimum"]["M"],
                threading["shape_minimum"]["N"],
                threading["shape_minimum"]["K"],
            ),
        )

    def test_forced_avx2_compile_row_does_not_enable_vnni(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        match = re.search(
            r"^test-q4k-q8k-isa-compile:.*?(?=^endif$)",
            makefile,
            re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(match)
        recipe = match.group(0)
        avx2_match = re.search(
            r"\$\(CC\)(.*?)Q4K_Q8K_ISA_AVX2_OBJ",
            recipe,
            re.DOTALL,
        )
        self.assertIsNotNone(avx2_match)
        self.assertNotIn("mavxvnni", avx2_match.group(1))

    def test_production_uses_persistent_weight_pack_not_q8_repack(self) -> None:
        source = (ROOT / "version" / "v8" / "src" / "ck_parallel_prefill_v8.c").read_text(
            encoding="utf-8"
        )
        self.assertIn("ck_get_q4k_packed_vnni_x8_cached", source)
        self.assertIn("gemm_nt_q4_k_packed_vnni_x8_q8_k_split_min_threaded_4m", source)
        self.assertIn("ck_select_q4k_vnni_active_threads", source)
        self.assertIn("ck_threadpool_capacity(pool)", source)
        self.assertNotIn("pack_q8_k_rows_x4", source)

    def test_exact_vnni_provider_honors_explicit_pool_capacity(self) -> None:
        source = (
            ROOT / "src" / "kernels" / "gemm_kernels_q4k_q8k_vnni.c"
        ).read_text(encoding="utf-8")
        function = re.search(
            r"void gemm_nt_q4_k_packed_vnni_x8_q8_k_split_min_threaded_4m\("
            r".*?^}",
            source,
            re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(function)
        body = function.group(0)
        self.assertIn("ck_threadpool_capacity(pool)", body)
        self.assertNotIn("ck_threadpool_n_threads(pool)", body)

    def test_openmp_isolation_does_not_disable_ck_provider_capacity(self) -> None:
        source = (ROOT / "src" / "ck_threadpool.c").read_text(encoding="utf-8")
        init = re.search(
            r"static void global_pool_init\(void\).*?^}",
            source,
            re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(init)
        body = init.group(0)
        self.assertIn('!getenv("CK_NUM_THREADS")', body)
        self.assertIn("default_threads == physical_threads", body)
        self.assertNotIn('!getenv("OMP_NUM_THREADS")', body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
