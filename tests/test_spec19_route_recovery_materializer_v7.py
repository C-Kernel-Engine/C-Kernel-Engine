#!/usr/bin/env python3
"""Tests for spec19 route recovery materializer helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "version" / "v7" / "scripts" / "dataset"
sys.path.insert(0, str(SCRIPT_DIR))

import materialize_spec19_route_recovery_replay_v7 as recovery_materializer  # type: ignore


class Spec19RouteRecoveryMaterializerTest(unittest.TestCase):
    def test_humanize(self) -> None:
        self.assertEqual(recovery_materializer._humanize("process_route"), "process route")


if __name__ == "__main__":
    unittest.main()
