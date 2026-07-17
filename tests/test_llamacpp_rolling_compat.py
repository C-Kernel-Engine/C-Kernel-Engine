from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_llamacpp_rolling_compat.py"
SPEC = importlib.util.spec_from_file_location("rolling_compat", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RollingCompatReportTests(unittest.TestCase):
    def test_parse_quick_log(self) -> None:
        content = """
PERFORMANCE SUMMARY
  Average CK/llama.cpp speedup: 0.90x
  Average CK GFLOPS:            7.26
  Average llama.cpp GFLOPS:     9.77
PARITY SMOKETEST SUMMARY
  Passed:  6
  Failed:  0
  Skipped: 1
"""
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp) / "quick.log"
            path.write_text(content, encoding="utf-8")
            summary = MODULE.parse_quick_log(path)
        self.assertEqual(summary["smoketest"], {"passed": 6, "failed": 0, "skipped": 1})
        self.assertEqual(
            summary["performance_summaries"],
            [{"ck_over_llama_speedup": 0.9, "ck_gflops": 7.26, "llama_gflops": 9.77}],
        )

    def test_optional_patch_drift_does_not_fail_runtime_compatibility(self) -> None:
        phases = {
            "patch_compatibility": {"status": "warn", "blocking": False},
            "ck_build": {"status": "pass"},
            "quick_parity": {"status": "pass"},
        }
        self.assertEqual(MODULE.compatibility_status(phases), "pass")

    def test_build_or_parity_failure_remains_blocking(self) -> None:
        for phase_name in ("ck_build", "quick_parity"):
            with self.subTest(phase=phase_name):
                phases = {
                    "patch_compatibility": {"status": "pass", "blocking": False},
                    "ck_build": {"status": "pass"},
                    "quick_parity": {"status": "pass"},
                }
                phases[phase_name]["status"] = "fail"
                self.assertEqual(MODULE.compatibility_status(phases), "fail")


if __name__ == "__main__":
    unittest.main()
