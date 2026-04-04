#!/usr/bin/env python3
"""Tests for the unified spec19 curriculum materializer helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "version" / "v7" / "scripts" / "dataset"
sys.path.insert(0, str(SCRIPT_DIR))

import materialize_spec19_unified_curriculum_v7 as unified_materializer  # type: ignore


class Spec19UnifiedCurriculumMaterializerTest(unittest.TestCase):
    def test_build_unified_manifest_sets_lineage(self) -> None:
        route_manifest = {
            "line": "spec19_route_recovery_replay",
            "source_runs": ["/tmp/r2", "/tmp/r3d"],
            "stages": {"pretrain": {"added_rows": 53}, "midtrain": {"added_rows": 65}},
            "eval_collision_filter": {"removed_total": 0},
        }
        manifest = unified_materializer._build_unified_manifest(
            workspace=Path("/tmp/spec19"),
            prefix="spec19_scene_bundle",
            route_recovery_manifest=route_manifest,
        )
        self.assertEqual(manifest["line"], unified_materializer.LINE_NAME)
        self.assertEqual(manifest["derived_from_line"], "spec19_route_recovery_replay")
        self.assertEqual(manifest["curriculum_mode"], "fresh_retrain_from_frozen_base_seed")
        self.assertEqual(manifest["stages"]["pretrain"]["added_rows"], 53)


if __name__ == "__main__":
    unittest.main()
