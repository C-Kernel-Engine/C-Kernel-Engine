#!/usr/bin/env python3
"""Tests for spec19 cumulative neighbor augmentation helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "version" / "v7" / "scripts" / "dataset"
sys.path.insert(0, str(SCRIPT_DIR))

import materialize_spec19_cumulative_neighbor_replay_v7 as neighbor_materializer  # type: ignore


class Spec19CumulativeNeighborMaterializerTest(unittest.TestCase):
    def test_humanize(self) -> None:
        self.assertEqual(neighbor_materializer._humanize("ordered_steps"), "ordered steps")

    def test_dedupe_preserve(self) -> None:
        rows = ["a", "b", "a", "", "c"]
        self.assertEqual(neighbor_materializer._dedupe_preserve(rows), ["a", "b", "c"])

    def test_extract_prompt(self) -> None:
        row = "[task:svg] [goal:show_flow] [OUT] [bundle] [family:system_diagram] [/bundle]"
        self.assertEqual(neighbor_materializer._extract_prompt(row), "[task:svg] [goal:show_flow] [OUT]")


if __name__ == "__main__":
    unittest.main()
