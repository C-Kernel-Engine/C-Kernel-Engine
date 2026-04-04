#!/usr/bin/env python3
"""Tests for spec19 balanced coverage materializer helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "version" / "v7" / "scripts" / "dataset"
sys.path.insert(0, str(SCRIPT_DIR))

import materialize_spec19_balanced_coverage_replay_v7 as balanced_materializer  # type: ignore


class Spec19BalancedCoverageMaterializerTest(unittest.TestCase):
    def test_humanize(self) -> None:
        self.assertEqual(balanced_materializer._humanize("ordered_steps"), "ordered steps")


if __name__ == "__main__":
    unittest.main()
