#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version" / "v7" / "scripts"))

import training_policy_v7 as policy  # type: ignore


FAMILIES = ("memory_map", "timeline", "system_diagram")
HIDDEN_SPLITS = ("hidden_train", "hidden_test")


def _probe_row(*, family: str, split: str, exact: bool, renderable: bool) -> dict[str, object]:
    return {
        "prompt": f"[task:svg] [layout:{family}] [OUT]",
        "split": split,
        "exact_match": exact,
        "renderable": renderable,
    }


class TrainingPolicyV7Tests(unittest.TestCase):
    def test_pilot_gate_blocks_family_regression_even_when_target_family_improves(self) -> None:
        baseline_doc = {
            "results": [
                _probe_row(family="memory_map", split="train", exact=True, renderable=True),
                _probe_row(family="timeline", split="train", exact=True, renderable=True),
                _probe_row(family="system_diagram", split="train", exact=False, renderable=True),
                _probe_row(family="memory_map", split="hidden_train", exact=True, renderable=True),
                _probe_row(family="timeline", split="hidden_test", exact=True, renderable=True),
                _probe_row(family="system_diagram", split="hidden_test", exact=False, renderable=True),
            ]
        }
        current_doc = {
            "results": [
                _probe_row(family="memory_map", split="train", exact=True, renderable=True),
                _probe_row(family="timeline", split="train", exact=False, renderable=True),
                _probe_row(family="system_diagram", split="train", exact=True, renderable=True),
                _probe_row(family="memory_map", split="hidden_train", exact=True, renderable=True),
                _probe_row(family="timeline", split="hidden_test", exact=True, renderable=True),
                _probe_row(family="system_diagram", split="hidden_test", exact=True, renderable=True),
            ]
        }

        baseline_metrics = policy.probe_metrics_from_doc(
            baseline_doc,
            families=FAMILIES,
            hidden_splits=HIDDEN_SPLITS,
        )
        current_metrics = policy.probe_metrics_from_doc(
            current_doc,
            families=FAMILIES,
            hidden_splits=HIDDEN_SPLITS,
        )
        payload = policy.build_pilot_gate_payload(
            spec="specx",
            baseline_probe=Path("/tmp/baseline.json"),
            current_probe=Path("/tmp/current.json"),
            baseline_metrics=baseline_metrics,
            current_metrics=current_metrics,
            families=FAMILIES,
            hidden_splits=HIDDEN_SPLITS,
            improve_families=("system_diagram",),
        )

        self.assertFalse(payload["clears_gate"])
        self.assertTrue(payload["checks"]["system_diagram_improved"])
        self.assertFalse(payload["checks"]["family_non_regression"])
        self.assertIn("regressed at least one family", " ".join(payload["reasons"]))

    def test_training_decision_blocks_after_strong_winner_and_failed_descendants(self) -> None:
        frozen_doc = {
            "results": [
                _probe_row(family="memory_map", split="train", exact=True, renderable=True),
                _probe_row(family="timeline", split="train", exact=True, renderable=True),
                _probe_row(family="system_diagram", split="train", exact=True, renderable=True),
                _probe_row(family="memory_map", split="hidden_train", exact=True, renderable=True),
                _probe_row(family="timeline", split="hidden_test", exact=True, renderable=True),
                _probe_row(family="system_diagram", split="hidden_test", exact=True, renderable=True),
            ]
        }
        frozen_metrics = policy.probe_metrics_from_doc(
            frozen_doc,
            families=FAMILIES,
            hidden_splits=HIDDEN_SPLITS,
        )
        descendants = [
            {
                "run_name": "r10",
                "overall_exact": 0.72,
                "renderable": 0.84,
                "beats_frozen": False,
                "pilot_gate_clears": False,
            },
            {
                "run_name": "r11",
                "overall_exact": 0.81,
                "renderable": 0.95,
                "beats_frozen": False,
                "pilot_gate_clears": False,
            },
        ]

        payload = policy.build_training_decision_payload(
            spec="specx",
            frozen_run=Path("/tmp/specx_r9"),
            frozen_metrics=frozen_metrics,
            descendants=descendants,
            hidden_splits=HIDDEN_SPLITS,
        )

        self.assertFalse(payload["training_allowed"])
        self.assertEqual(payload["default_action"], "decode_repair")
        self.assertTrue(payload["block_raw_repair_rungs"])
        self.assertIn("frozen raw baseline", " ".join(payload["reasons"]))
        self.assertIn("latest pilot failed", " ".join(payload["reasons"]))


if __name__ == "__main__":
    unittest.main()
