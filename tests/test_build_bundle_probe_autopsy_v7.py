#!/usr/bin/env python3
"""Unit tests for bundle probe autopsy classification."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "version" / "v7" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import build_bundle_probe_autopsy_v7 as autopsy  # type: ignore


class BuildBundleProbeAutopsyV7Test(unittest.TestCase):
    def _row(self, *, expected: str, parsed: str, renderable: bool = True, exact: bool = False) -> dict[str, object]:
        return {
            "expected_output": expected,
            "parsed_output": parsed,
            "renderable": renderable,
            "exact_match": exact,
        }

    def test_classify_family_mismatch(self) -> None:
        row = self._row(
            expected="[bundle] [family:timeline] [form:stage_sequence] [/bundle]",
            parsed="[bundle] [family:system_diagram] [form:build_path] [/bundle]",
        )
        self.assertEqual(autopsy.classify_case(row)["primary_failure"], "family")

    def test_classify_form_mismatch(self) -> None:
        row = self._row(
            expected="[bundle] [family:timeline] [form:stage_sequence] [/bundle]",
            parsed="[bundle] [family:timeline] [form:milestone_chain] [/bundle]",
        )
        self.assertEqual(autopsy.classify_case(row)["primary_failure"], "form")

    def test_classify_style_mismatch(self) -> None:
        row = self._row(
            expected="[bundle] [family:timeline] [form:stage_sequence] [theme:paper_editorial] [/bundle]",
            parsed="[bundle] [family:timeline] [form:stage_sequence] [theme:infra_dark] [/bundle]",
        )
        self.assertEqual(autopsy.classify_case(row)["primary_failure"], "style")

    def test_classify_syntax_when_non_renderable(self) -> None:
        row = self._row(
            expected="[bundle] [family:timeline] [form:stage_sequence] [/bundle]",
            parsed="[bundle] [family:timeline] [OUT] [bundle]",
            renderable=False,
        )
        self.assertEqual(autopsy.classify_case(row)["primary_failure"], "syntax")


if __name__ == "__main__":
    unittest.main()
