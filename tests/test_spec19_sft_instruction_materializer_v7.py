#!/usr/bin/env python3
"""Tests for spec19 SFT instruction materializer helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "version" / "v7" / "scripts" / "dataset"
sys.path.insert(0, str(SCRIPT_DIR))

import materialize_spec19_sft_instruction_v7 as sft_materializer  # type: ignore


class Spec19SftInstructionMaterializerTest(unittest.TestCase):
    def test_parse_surface_multiplier_accepts_surface_count_pairs(self) -> None:
        parsed = sft_materializer._parse_surface_multiplier(
            ["clean_stop_anchor=3", "explicit_bundle_anchor=2"]
        )
        self.assertEqual(parsed["clean_stop_anchor"], 3)
        self.assertEqual(parsed["explicit_bundle_anchor"], 2)

    def test_instruction_variants_keep_original_and_add_instructional_forms(self) -> None:
        prompt = "choose one shared bundle for topic memory layout goal compare regions audience infra ops [OUT]"
        variants = sft_materializer._instruction_variants(prompt, prompt_surface="routebook_direct")
        self.assertGreaterEqual(len(variants), 3)
        self.assertEqual(variants[0], prompt)
        self.assertTrue(any("Plan exactly one compiler-facing shared visual bundle" in row for row in variants))

    def test_build_sft_rows_can_preserve_surface_weights(self) -> None:
        class _Base:
            @staticmethod
            def _row_from_catalog(prompt: str, output_tokens: str) -> str:
                return f"{prompt} {output_tokens}"

        rows, _, counts = sft_materializer._build_sft_rows(
            [
                {
                    "prompt": "prompt a",
                    "output_tokens": "[bundle] a [/bundle]",
                    "prompt_surface": "clean_stop_anchor",
                }
            ],
            base=_Base(),
            max_variants=1,
            surface_multipliers={"clean_stop_anchor": 3},
            preserve_surface_weights=True,
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(counts["clean_stop_anchor"], 3)

    def test_build_sft_manifest_tracks_variant_policy(self) -> None:
        manifest = sft_materializer._build_sft_manifest(
            workspace=Path("/tmp/spec19"),
            prefix="spec19_scene_bundle",
            unified_manifest={"line": "spec19_unified_curriculum", "source_runs": ["/tmp/r2", "/tmp/r3d"]},
            train_rows=100,
            dev_rows=20,
            test_rows=20,
            source_surface_counts={"routebook_direct": 12},
            instruction_surface_counts={"routebook_direct": 36},
            train_variants=3,
            eval_variants=2,
            line_name="spec19_sft_instruction_b",
            format_version="ck.spec19_sft_instruction_b.v1",
            train_surface_multipliers={"clean_stop_anchor": 3},
        )
        self.assertEqual(manifest["derived_from_line"], "spec19_unified_curriculum")
        self.assertEqual(manifest["instruction_variant_policy"]["train_variants_per_prompt"], 3)
        self.assertEqual(manifest["instruction_variant_policy"]["train_surface_multipliers"]["clean_stop_anchor"], 3)
        self.assertEqual(manifest["stages"]["sft"]["train_rows"], 100)


if __name__ == "__main__":
    unittest.main()
