from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_nightly_results.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("verify_nightly_results_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NightlyVerdictTests(unittest.TestCase):
    def _write(
        self,
        root: Path,
        results: list[dict],
        summary: dict | None = None,
        regression_fast: dict | None = None,
    ) -> Path:
        counts = {
            "total": len(results),
            "passed": sum(row["status"] == "pass" for row in results),
            "failed": sum(row["status"] == "fail" for row in results),
            "skipped": sum(row["status"] == "skip" for row in results),
            "timeout": sum(row["status"] == "timeout" for row in results),
        }
        path = root / "latest.json"
        payload = {
            "timestamp": "2026-09-06T12:00:00+00:00",
            "summary": summary or counts,
            "results": results,
        }
        if regression_fast is not None:
            payload["regression_fast"] = regression_fast
        path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        return path

    def test_clean_report_passes(self) -> None:
        verifier = _load_verifier()
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), [{"name": "contracts", "status": "pass"}])
            self.assertEqual(verifier.verify_report(path), [])

    def test_collects_failures_timeouts_and_failed_subtests(self) -> None:
        verifier = _load_verifier()
        rows = [
            {"name": "lowering", "status": "fail"},
            {"name": "audio", "status": "timeout"},
            {
                "name": "provider matrix",
                "status": "pass",
                "sub_tests": [{"name": "nvfp4", "status": "fail"}],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            errors = verifier.verify_report(self._write(Path(tmp), rows))
        self.assertTrue(any("lowering: fail" in error for error in errors))
        self.assertTrue(any("audio: timeout" in error for error in errors))
        self.assertTrue(any("failed subtest nvfp4" in error for error in errors))

    def test_rejects_stale_and_inconsistent_report(self) -> None:
        verifier = _load_verifier()
        rows = [{"name": "contracts", "status": "pass"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), rows, {"total": 99})
            errors = verifier.verify_report(path, not_before_epoch=2_000_000_000)
        self.assertTrue(any("predates" in error for error in errors))
        self.assertTrue(any("summary total mismatch" in error for error in errors))

    def test_required_fast_regression_needs_current_summary_evidence(self) -> None:
        verifier = _load_verifier()
        rows = [{"name": "contracts", "status": "pass"}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = verifier.verify_report(
                self._write(root, rows), fast_regression_required=True
            )
            failed = verifier.verify_report(
                self._write(
                    root,
                    rows,
                    regression_fast={
                        "status": "fail",
                        "summary_path": "",
                        "family_rows": [],
                    },
                ),
                fast_regression_required=True,
            )
            passed = verifier.verify_report(
                self._write(
                    root,
                    rows,
                    regression_fast={
                        "status": "pass",
                        "summary_path": "build/regression-reports/run/summary.json",
                        "family_rows": [{"family_id": "qwen38", "status": "PASS"}],
                    },
                ),
                fast_regression_required=True,
            )
        self.assertTrue(any("payload is missing" in error for error in missing))
        self.assertTrue(any("payload status: fail" in error for error in failed))
        self.assertTrue(any("summary path is missing" in error for error in failed))
        self.assertTrue(any("no family rows" in error for error in failed))
        self.assertEqual(passed, [])

    def test_workflow_collects_evidence_before_required_verdict(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "nightly.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("name: Required nightly verdict", workflow)
        nightly_start = workflow.index("- name: Run Nightly Test Suite")
        upload_start = workflow.index("- name: Upload test results")
        verdict_start = workflow.index("- name: Enforce nightly required verdict")
        nightly_block = workflow[nightly_start:workflow.index("- name: Attach Nightly Summaries")]
        self.assertIn("continue-on-error: true", nightly_block)
        self.assertIn("set -o pipefail", nightly_block)
        self.assertNotIn("--no-fail", nightly_block)
        attach_block = workflow[
            workflow.index("- name: Attach Nightly Summaries"):upload_start
        ]
        self.assertIn("elif requested:", attach_block)
        self.assertLess(nightly_start, upload_start)
        self.assertLess(upload_start, verdict_start)


if __name__ == "__main__":
    unittest.main()
