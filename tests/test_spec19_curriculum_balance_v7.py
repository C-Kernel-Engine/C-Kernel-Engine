#!/usr/bin/env python3
"""Tests for spec19 cumulative curriculum balancing helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "version" / "v7" / "scripts" / "dataset"
sys.path.insert(0, str(SCRIPT_DIR))

import spec19_curriculum_balance_v7 as balance  # type: ignore


class Spec19CurriculumBalanceV7Tests(unittest.TestCase):
    def test_balance_caps_dominant_corrective_surface(self) -> None:
        base_surface_counts = {
            "clean_stop_anchor": 60,
            "routebook_paraphrase": 30,
            "routebook_direct_hint": 30,
            "style_topology_bridge": 30,
        }
        proposed_meta = (
            [{"prompt_surface": "clean_stop_anchor"} for _ in range(45)]
            + [{"prompt_surface": "routebook_paraphrase"} for _ in range(9)]
            + [{"prompt_surface": "routebook_direct_hint"} for _ in range(9)]
            + [{"prompt_surface": "style_topology_bridge"} for _ in range(9)]
        )
        proposed_rows = [f"row_{idx}" for idx in range(len(proposed_meta))]

        kept_rows, kept_meta, report = balance.balance_delta_rows(
            stage="pretrain",
            base_train_rows=300,
            base_surface_counts=base_surface_counts,
            proposed_rows=proposed_rows,
            proposed_meta=proposed_meta,
            max_surface_growth_fraction=0.2,
            dominant_to_soft_ratio_max=1.5,
        )

        self.assertEqual(len(kept_rows), len(kept_meta))
        self.assertEqual(report["kept_surface_counts"]["routebook_paraphrase"], 6)
        self.assertEqual(report["kept_surface_counts"]["routebook_direct_hint"], 6)
        self.assertEqual(report["kept_surface_counts"]["style_topology_bridge"], 6)
        self.assertEqual(report["kept_surface_counts"]["clean_stop_anchor"], 12)
        self.assertFalse(report["dominant_ratio_applied"])

    def test_balance_preserves_existing_order(self) -> None:
        base_surface_counts = {"clean_stop_anchor": 20, "style_topology_bridge": 20}
        proposed_meta = [
            {"prompt_surface": "clean_stop_anchor", "id": "a1"},
            {"prompt_surface": "style_topology_bridge", "id": "b1"},
            {"prompt_surface": "clean_stop_anchor", "id": "a2"},
            {"prompt_surface": "style_topology_bridge", "id": "b2"},
        ]
        proposed_rows = ["a1", "b1", "a2", "b2"]

        kept_rows, kept_meta, _ = balance.balance_delta_rows(
            stage="pretrain",
            base_train_rows=100,
            base_surface_counts=base_surface_counts,
            proposed_rows=proposed_rows,
            proposed_meta=proposed_meta,
            max_surface_growth_fraction=0.05,
            dominant_to_soft_ratio_max=1.0,
        )

        self.assertEqual(kept_rows, ["a1", "b1"])
        self.assertEqual([item["id"] for item in kept_meta], ["a1", "b1"])


if __name__ == "__main__":
    unittest.main()
