from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version/v8/scripts/run_training_composition_v8.py"


class TrainingCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("training_composition_v8_test", SCRIPT)
        assert spec is not None and spec.loader is not None
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def test_profiles_exercise_dense_and_unequal_head_gqa(self) -> None:
        profiles = self.runner.PROFILES
        self.assertEqual(profiles["2l_dense"]["heads"], profiles["2l_dense"]["kv_heads"])
        self.assertGreater(profiles["2l_gqa_tail"]["heads"], profiles["2l_gqa_tail"]["kv_heads"])
        self.assertEqual({profile["layers"] for profile in profiles.values()}, {2})

    def test_missing_update_comparisons_and_manifest_cannot_pass(self) -> None:
        report = {"configuration": {"layers": 2}, "checks": {
            "pytorch_trajectory": {"passed": True, "trajectory": [
                {"step": index, "passed": True} for index in range(1, 11)]},
        }}
        experiment = SimpleNamespace(manifest_path=Path("/nonexistent/cke-composition-manifest.json"))
        errors = self.runner.validate_case("2l_dense", experiment, report)
        self.assertIn("trajectory.update_1.forward_logits", errors)
        self.assertIn("trajectory.update_1.gradients", errors)
        self.assertIn("manifest.missing", errors)

    def test_case_launch_failure_still_writes_aggregate_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            report_path = Path(temp) / "aggregate.json"
            argv = [str(SCRIPT), "--run-root", temp, "--json-out", str(report_path),
                    "--case", "2l_dense", "--case", "2l_gqa_tail"]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                self.runner, "build_experiment", side_effect=OSError("injected launch failure")
            ):
                self.assertEqual(self.runner.main(), 1)
            aggregate = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(aggregate["status"], "FAIL")
            self.assertEqual(len(aggregate["cases"]), 2)
            self.assertTrue(all("injected launch failure" in row["errors"][0] for row in aggregate["cases"]))


if __name__ == "__main__":
    unittest.main()
