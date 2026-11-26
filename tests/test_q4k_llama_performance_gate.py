from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _q4_dispatch_policies():
    path = ROOT / "version/v8/kernel_maps/gemm_nt_q4_k_q8_k.json"
    return json.loads(path.read_text(encoding="utf-8"))["implementation"]["runtime_dispatch"]["policies"]


def _load_nightly_runner():
    path = ROOT / "scripts" / "nightly_runner.py"
    spec = importlib.util.spec_from_file_location("nightly_runner_q4k_perf_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Q4KLlamaPerformanceGateTests(unittest.TestCase):
    def test_packed_q4_tile_schedules_cover_each_output_exactly_once(self) -> None:
        compiler = shutil.which("cc")
        taskset = shutil.which("taskset")
        if compiler is None or taskset is None or sys.platform != "linux":
            self.skipTest("Linux C toolchain and taskset required for the native pool probe")
        fixture = ROOT / "tests/fixtures/q4k_tile_schedule/probe.c"
        cpus = sorted(os.sched_getaffinity(0))
        isa_flags = []
        cpu_flags = set(Path("/proc/cpuinfo").read_text().split())
        if {"avx2", "fma", "f16c"} <= cpu_flags:
            isa_flags = ["-mavx2", "-mfma", "-mf16c"]
        if {"avx512f", "avx512bw", "avx512vl", "avx512_vnni"} <= cpu_flags:
            isa_flags += ["-mavx512f", "-mavx512bw", "-mavx512vl", "-mavx512vnni"]
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "probe"
            subprocess.run([
                compiler, "-std=gnu11", "-O2", *isa_flags,
                "-ffunction-sections", "-fdata-sections",
                "-I", str(ROOT / "include"), "-I", str(ROOT / "src/kernels"),
                str(fixture), str(ROOT / "src/ck_threadpool.c"), str(ROOT / "src/ckernel_strict.c"),
                "-Wl,--gc-sections", "-pthread", "-lm", "-o", str(binary),
            ], check=True, capture_output=True, text=True)
            for count in sorted({1, min(4, len(cpus))}):
                with self.subTest(workers=count):
                    env = {**os.environ, "CK_NUM_THREADS": str(count), "OMP_NUM_THREADS": "1"}
                    env.pop("CK_THREADPOOL_CAPACITY", None)
                    subprocess.run(
                        [taskset, "-c", ",".join(map(str, cpus[:count])), str(binary)],
                        env=env, check=True, capture_output=True, text=True,
                    )

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

    def test_avx512_x16_provider_is_qwen36_scoped_capability(self) -> None:
        path = ROOT / "version" / "v8" / "kernel_maps" / "gemm_nt_q4_k_q8_k.json"
        kernel = json.loads(path.read_text(encoding="utf-8"))
        candidates = {
            row["name"]: row
            for row in kernel["phase_selection"]["prefill"]["candidates"]
        }
        provider = candidates["packed_vnni_x16_split_min_16m16n"]
        self.assertEqual(
            provider["function"],
            "gemm_nt_q4_k_packed_vnni_x16_q8_k_split_min_threaded_16m",
        )
        self.assertEqual(provider["layout"], "q4_k_packed_vnni_x16")
        self.assertEqual(provider["threading"]["tile"], {"M": 16, "N": 16})
        self.assertEqual(provider["threading"]["internal_row_register_group"], 8)
        self.assertEqual(provider["promotion"], "experimental_sweep_only")
        self.assertIn("avx512_vnni", provider["requires"])

        source = (
            ROOT / "src" / "kernels" / "gemm_kernels_q4k_q8k_vnni.c"
        ).read_text(encoding="utf-8")
        self.assertIn("CK_HAS_AVX512_VNNI_512", source)
        self.assertIn("pack_q4_k_to_packed_vnni_x16", source)
        self.assertIn(
            "gemm_nt_q4_k_packed_vnni_x16_q8_k_split_min_threaded_16m",
            source,
        )

        dispatcher = (
            ROOT / "version" / "v8" / "src" / "ck_parallel_prefill_v8.c"
        ).read_text(encoding="utf-8")
        self.assertIn("ck_get_q4k_packed_vnni_x16_cached", dispatcher)
        self.assertIn("ck_q4k_avx512_x16_prefill_route", dispatcher)
        self.assertIn("ck_policy_gemm_nt_q4_k_q8_k_avx512_vnni_x16_prefill", dispatcher)
        prefill = _q4_dispatch_policies()["avx512_vnni_x16_prefill"]
        self.assertEqual(prefill["status"], "candidate")
        self.assertEqual({(r["min_m"], r["n"], r["k"]) for r in prefill["routes"]}, {
            (16, 1024, 5120), (16, 6144, 5120), (16, 12288, 5120),
            (16, 34816, 5120), (16, 5120, 6144), (16, 5120, 17408),
        })
        self.assertIn("CK_ENABLE_Q4K_AVX512_X16_EXPERIMENTAL", dispatcher)
        self.assertIn("CK_V8_FORCE_BATCHED_PREFILL", dispatcher)
        self.assertIn("CK_DISABLE_Q4K_AVX512_X16_PREFILL", dispatcher)

    def test_qwen36_x16_recurrent_tail_batches_only_exact_gemv_rows(self) -> None:
        dispatcher = (
            ROOT / "version" / "v8" / "src" / "ck_parallel_prefill_v8.c"
        ).read_text(encoding="utf-8")
        worker = re.search(
            r"static void work_gemv_q4_k_q8_k_repacked_x16\(.*?^}",
            dispatcher,
            re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(worker)
        self.assertIn("for (int row = 0; row < a->M; ++row)", worker.group(0))
        self.assertIn("row * a->A_row_bytes", worker.group(0))
        self.assertIn("row * (size_t)a->N", worker.group(0))
        self.assertIn(
            "gemm_nt_q4_k_packed_vnni_x16_q8_k_gemv_order",
            worker.group(0),
        )

        routes = _q4_dispatch_policies()["avx512_vnni_x16_prefill"]["routes"]
        self.assertEqual([(r["n"], r["k"]) for r in routes if "batched_tail" in r.get("flags", [])],
                         [(6144, 5120)])
        self.assertIn("packed_vnni_x16 && x16_route", dispatcher)
        self.assertIn("x16_route->flags & CK_GEMM_ROUTE_BATCHED_TAIL", dispatcher)
        self.assertIn("CK_DISABLE_Q4K_X16_BATCHED_TAIL", dispatcher)

    def test_qwen36_x16_decode_reuses_only_prepared_weight_views(self) -> None:
        source = (ROOT / "version/v8/src/ck_parallel_prefill_v8.c").read_text(
            encoding="utf-8"
        )
        dispatcher = re.search(
            r"void gemv_q4_k_q8_k_repacked_parallel_dispatch\(.*?^}",
            source,
            flags=re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(dispatcher)
        body = dispatcher.group(0)
        self.assertNotIn("CK_ENABLE_QWEN36_Q4K_X16_DECODE", source)
        self.assertIn("ck_policy_gemm_nt_q4_k_q8_k_avx512_vnni_x16_decode_prepared", body)
        routes = _q4_dispatch_policies()["avx512_vnni_x16_decode_prepared"]["routes"]
        self.assertEqual({(r["m"], r["n"], r["k"]) for r in routes}, {
            (1, 6144, 5120), (1, 12288, 5120), (1, 17408, 5120),
            (1, 5120, 6144), (1, 5120, 17408),
        })
        self.assertIn("run_gemv_q4_k_q8_k_repacked_x16_parallel", body)
        self.assertIn("ck_find_prepared_q4k_packed_vnni_x16", body)
        self.assertLess(
            body.index("ck_find_prepared_q4k_packed_vnni_x16"),
            body.index("ck_get_q4k_packed_meta_x8_cached"),
        )

        lookup = re.search(
            r"static void \*ck_find_prepared_q4k_packed_vnni_x16\(.*?^}",
            source,
            flags=re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(lookup)
        lookup_body = lookup.group(0)
        self.assertIn("query_begin < entry_begin", lookup_body)
        self.assertIn("query_end > entry_end", lookup_body)
        self.assertIn("row_offset % 16u", lookup_body)
        self.assertIn("row_offset / 16u", lookup_body)
        self.assertNotIn("ck_get_q4k_packed_vnni_x16_cached", lookup_body)

    def test_production_uses_persistent_weight_pack_not_q8_repack(self) -> None:
        source = (ROOT / "version" / "v8" / "src" / "ck_parallel_prefill_v8.c").read_text(
            encoding="utf-8"
        )
        dispatcher = re.search(
            r"void gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch\("
            r".*?^}",
            source,
            re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(dispatcher)
        body = dispatcher.group(0)
        self.assertIn("ck_get_q4k_packed_vnni_x8_cached", body)
        self.assertIn("ck_q4k_packed_vnni_x8_available()", body)
        self.assertIn(
            "gemm_nt_q4_k_packed_vnni_x8_q8_k_split_min_threaded_4m",
            body,
        )
        self.assertNotIn("#if defined(__AVXVNNI__)", body)
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
