#!/usr/bin/env python3
"""Tests for asset-to-DSL coverage audit helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "version" / "v7" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import build_asset_dsl_coverage_audit_v7 as audit  # type: ignore


class AssetDslCoverageAuditTest(unittest.TestCase):
    def test_bucket_assignment_is_directionally_useful(self) -> None:
        self.assertEqual(audit._bucket_for_asset("ir-v66-evolution-timeline"), "timeline")
        self.assertEqual(audit._bucket_for_asset("memory-layout-map"), "memory-map-or-training")
        self.assertEqual(audit._bucket_for_asset("pipeline-overview"), "system-diagram-or-flow")
        self.assertEqual(audit._bucket_for_asset("performance-balance"), "board-chart-or-comparison")

    def test_summary_counts_covered_vs_missing(self) -> None:
        asset_paths = [Path("/tmp/a.svg"), Path("/tmp/b.svg"), Path("/tmp/c.svg")]
        asset_map = {"a.svg": ["gen_a.py"], "c.svg": ["gen_c.py"]}
        summary = audit._summarize(asset_paths, asset_map)
        self.assertEqual(summary["asset_count"], 3)
        self.assertEqual(summary["covered_count"], 2)
        self.assertEqual(summary["missing_count"], 1)
        self.assertEqual(summary["covered_assets"], ["a.svg", "c.svg"])
        self.assertEqual(summary["missing_assets"], ["b.svg"])


if __name__ == "__main__":
    unittest.main()
