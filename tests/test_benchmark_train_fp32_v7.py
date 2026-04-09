#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v7" / "scripts" / "benchmark_train_fp32_v7.py"


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


bench = _load_module("benchmark_train_fp32_v7_test", SCRIPT)
cpu_policy = _load_module("cpu_policy_v7_test", SCRIPT.parent / "cpu_policy_v7.py")


class BenchmarkTrainFp32V7Tests(unittest.TestCase):
    def test_cpu_policy_auto_prefers_fast_physical_cores_on_hybrid_sysfs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cpu_base = Path(td)
            spec = {
                0: {"siblings": "0-1", "core_id": "0", "core_type": "0", "max_freq": "4700000"},
                1: {"siblings": "0-1", "core_id": "0", "core_type": "0", "max_freq": "4700000"},
                2: {"siblings": "2-3", "core_id": "1", "core_type": "0", "max_freq": "4700000"},
                3: {"siblings": "2-3", "core_id": "1", "core_type": "0", "max_freq": "4700000"},
                4: {"siblings": "4", "core_id": "2", "core_type": "1", "max_freq": "3400000"},
                5: {"siblings": "5", "core_id": "3", "core_type": "1", "max_freq": "3400000"},
            }
            for cpu, meta in spec.items():
                cpu_dir = cpu_base / f"cpu{cpu}"
                (cpu_dir / "topology").mkdir(parents=True)
                (cpu_dir / "cpufreq").mkdir(parents=True)
                (cpu_dir / "topology" / "thread_siblings_list").write_text(meta["siblings"], encoding="utf-8")
                (cpu_dir / "topology" / "core_id").write_text(meta["core_id"], encoding="utf-8")
                (cpu_dir / "topology" / "core_type").write_text(meta["core_type"], encoding="utf-8")
                (cpu_dir / "cpufreq" / "cpuinfo_max_freq").write_text(meta["max_freq"], encoding="utf-8")

            resolved = cpu_policy.resolve_dense_cpu_policy(
                8,
                None,
                cpu_policy="auto",
                cpu_base=cpu_base,
            )
            self.assertEqual(resolved["resolved_threads"], 2)
            self.assertEqual(resolved["resolved_affinity_cpulist"], "0,2")
            self.assertEqual(resolved["policy_source"], "auto-fast-cores")

    def test_cpu_policy_respects_explicit_affinity(self) -> None:
        resolved = cpu_policy.resolve_dense_cpu_policy(
            8,
            "0-7",
            cpu_policy="auto",
            cpu_base=Path("/definitely/missing"),
        )
        self.assertEqual(resolved["resolved_threads"], 8)
        self.assertEqual(resolved["resolved_affinity_cpulist"], "0-7")
        self.assertEqual(resolved["policy_source"], "explicit")

    def test_extract_step_profile_accepts_train_report_shape(self) -> None:
        report = {
            "step_profile": {
                "processed_tokens": 2048,
                "ck_total_ms": 100.0,
                "torch_total_ms": 125.0,
                "train_tok_s": 20480.0,
            }
        }
        step = bench._extract_step_profile(report)
        self.assertEqual(step["processed_tokens"], 2048)
        self.assertEqual(step["ck_total_ms"], 100.0)

    def test_extract_step_profile_accepts_flat_profile_shape(self) -> None:
        report = {
            "processed_tokens": 1024,
            "ck_total_ms": 50.0,
            "train_tok_s": 20480.0,
        }
        step = bench._extract_step_profile(report)
        self.assertEqual(step["processed_tokens"], 1024)
        self.assertEqual(step["ck_total_ms"], 50.0)

    def test_render_markdown_contains_speedup_and_parity(self) -> None:
        summary = {
            "generated_at": "2026-04-08T00:00:00Z",
            "runtime": {"backend": "both", "template": "qwen3"},
            "workload": {
                "layers": 2,
                "d_model": 256,
                "hidden": 1024,
                "vocab_size": 1024,
                "seq_len": 8,
                "total_tokens": 2048,
                "grad_accum": 8,
                "epochs": 1,
            },
            "env_policy": {
                "threads": 8,
                "affinity_cpulist": "",
                "CK_CACHE_DIR": "/tmp/cache",
                "CK_NUM_THREADS": "8",
                "OMP_NUM_THREADS": "8",
                "MKL_NUM_THREADS": "8",
                "OPENBLAS_NUM_THREADS": "8",
                "NUMEXPR_NUM_THREADS": "8",
            },
            "performance": {
                "init_wall_s": 0.5,
                "train_wall_s": 2.0,
                "processed_tokens": 2048,
                "ck_total_ms": 100.0,
                "torch_total_ms": 125.0,
                "ck_avg_step_ms": 10.0,
                "torch_avg_step_ms": 12.5,
                "ck_train_tok_s": 20480.0,
                "torch_train_tok_s": 16384.0,
                "ck_speedup_vs_torch": 1.25,
            },
            "parity": {
                "pass_parity": True,
                "max_loss_abs_diff": 1.2e-5,
                "final_param_max_abs_diff": 0.0,
            },
            "artifacts": {
                "run_dir": "/tmp/run",
                "train_report_json": "/tmp/report.json",
                "init_log": "/tmp/init.log",
                "train_log": "/tmp/train.log",
            },
        }
        md = bench._render_markdown(summary)
        self.assertIn("ck_speedup_vs_torch", md)
        self.assertIn("pass_parity", md)
        self.assertIn("/tmp/report.json", md)


if __name__ == "__main__":
    unittest.main()
