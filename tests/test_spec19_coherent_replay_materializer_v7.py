#!/usr/bin/env python3
"""Unit tests for the spec19 coherent replay materializer helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "version" / "v7" / "scripts" / "dataset"
sys.path.insert(0, str(SCRIPT_DIR))

import materialize_spec19_coherent_replay_union_v7 as coherent_materializer  # type: ignore


class Spec19CoherentReplayMaterializerHelpersTest(unittest.TestCase):
    def test_dedupe_preserve_keeps_first_order(self) -> None:
        rows = ["a", "b", "a", "", "c", "b", "d"]
        self.assertEqual(coherent_materializer._dedupe_preserve(rows), ["a", "b", "c", "d"])

    def test_extract_prompt_stops_before_bundle(self) -> None:
        row = "[task:svg] [goal:show_flow] [OUT] [bundle] [family:system_diagram] [/bundle]"
        self.assertEqual(coherent_materializer._extract_prompt(row), "[task:svg] [goal:show_flow] [OUT]")

    def test_filter_prompts_removes_collisions(self) -> None:
        prompts = ["p1", "p2", "p1", "p3"]
        kept, removed = coherent_materializer._filter_prompts(prompts, {"p2"})
        self.assertEqual(kept, ["p1", "p3"])
        self.assertEqual(removed, ["p2"])


if __name__ == "__main__":
    unittest.main()
