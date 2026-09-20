#!/usr/bin/env python3
"""Regression tests for generated Cohere performance evidence parsing."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "version/v8/scripts/run_cohere_generated_long_audio_v8.py"
SPEC = importlib.util.spec_from_file_location("cohere_generated_long", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GeneratedAudioProfileTest(unittest.TestCase):
    def test_parses_complete_shape_inventory(self) -> None:
        report = MODULE.parse_performance_profile(
            "audio_runtime_profile requested_threads=8 actual_threads=8\n"
            "gemm_profile window=0 M=9 N=5120 K=1280 active_threads=8 "
            "mode=parallel calls=96 elapsed_ns=1234\n"
            "gemm_profile_summary window=0 shapes=1 overflow_calls=0\n"
            "threadpool_profile window=0 dispatches=96 total_ns=1200 "
            "main_work_ns=800 completion_wait_ns=400\n",
            1,
        )
        self.assertEqual(report["actual_threads"], 8)
        self.assertEqual(report["total_profiled_calls"], 96)
        self.assertEqual(report["gemm_shapes"][0]["N"], 5120)

    def test_rejects_incomplete_window_inventory(self) -> None:
        with self.assertRaisesRegex(ValueError, "inventory is incomplete"):
            MODULE.parse_performance_profile(
                "audio_runtime_profile requested_threads=auto actual_threads=16\n"
                "gemm_profile_summary window=0 shapes=0 overflow_calls=0\n"
                "threadpool_profile window=0 dispatches=0 total_ns=0 "
                "main_work_ns=0 completion_wait_ns=0\n",
                2,
            )

    def test_rejects_mismatched_shape_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape count differs"):
            MODULE.parse_performance_profile(
                "audio_runtime_profile requested_threads=4 actual_threads=4\n"
                "gemm_profile_summary window=0 shapes=1 overflow_calls=0\n"
                "threadpool_profile window=0 dispatches=0 total_ns=0 "
                "main_work_ns=0 completion_wait_ns=0\n",
                1,
            )

    def test_rejects_profile_overflow(self) -> None:
        with self.assertRaisesRegex(ValueError, "overflowed"):
            MODULE.parse_performance_profile(
                "audio_runtime_profile requested_threads=4 actual_threads=4\n"
                "gemm_profile_summary window=0 shapes=0 overflow_calls=1\n"
                "threadpool_profile window=0 dispatches=1 total_ns=1 "
                "main_work_ns=1 completion_wait_ns=0\n",
                1,
            )

    def test_rejects_duplicate_shape_identity(self) -> None:
        row = (
            "gemm_profile window=0 M=9 N=5120 K=1280 active_threads=8 "
            "mode=parallel calls=1 elapsed_ns=10\n"
        )
        with self.assertRaisesRegex(ValueError, "duplicate shape"):
            MODULE.parse_performance_profile(
                "audio_runtime_profile requested_threads=8 actual_threads=8\n"
                + row + row
                + "gemm_profile_summary window=0 shapes=2 overflow_calls=0\n"
                + "threadpool_profile window=0 dispatches=2 total_ns=20 "
                "main_work_ns=10 completion_wait_ns=10\n",
                1,
            )


if __name__ == "__main__":
    unittest.main()
