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


class BenchmarkTrainFp32V7Tests(unittest.TestCase):
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
