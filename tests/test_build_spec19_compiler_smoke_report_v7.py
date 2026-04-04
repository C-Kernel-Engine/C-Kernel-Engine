#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version" / "v7" / "scripts"))

import build_spec19_compiler_smoke_report_v7 as smoke_report  # type: ignore
import generate_svg_structured_spec19_v7 as spec19_generator  # type: ignore
from render_svg_structured_scene_spec16_v7 import render_structured_scene_spec16_svg  # type: ignore
from spec16_scene_bundle_canonicalizer_v7 import serialize_scene_bundle  # type: ignore


class Spec19CompilerSmokeReportTests(unittest.TestCase):
    def test_build_report_compiles_representative_rows(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spec19_smoke_") as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            catalog_dir = run_dir / "dataset" / "manifests" / "generated" / "structured_atoms"
            catalog_dir.mkdir(parents=True, exist_ok=True)
            out_dir = run_dir / "smoke"

            profile_cases = spec19_generator._build_profile_cases()
            lookup = spec19_generator._profile_lookup(profile_cases)
            by_family = {}
            for case in profile_cases:
                by_family.setdefault(case.bundle.family, case)

            rows: list[dict[str, object]] = []
            preferred_surfaces = {"explicit_bundle_anchor", "routebook_direct", "style_topology_bridge"}
            for case in by_family.values():
                output_tokens = serialize_scene_bundle(case.bundle)
                svg_xml = render_structured_scene_spec16_svg(output_tokens, content=case.content_json)
                seen_surface: set[str] = set()
                for prompt, prompt_surface, split, training_prompt in spec19_generator._prompt_rows(case, lookup):
                    if split != "train" or not training_prompt or prompt_surface not in preferred_surfaces or prompt_surface in seen_surface:
                        continue
                    seen_surface.add(prompt_surface)
                    rows.append(
                        {
                            "prompt": prompt,
                            "output_tokens": output_tokens,
                            "content_json": dict(case.content_json),
                            "svg_xml": svg_xml,
                            "split": "train",
                            "layout": case.bundle.family,
                            "family": case.bundle.family,
                            "case_id": case.case_id,
                            "profile_id": case.profile_id,
                            "prompt_surface": prompt_surface,
                        }
                    )

            catalog_path = catalog_dir / "spec19_scene_bundle_render_catalog.json"
            catalog_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

            report = smoke_report.build_report(run_dir, "spec19_scene_bundle", out_dir, max_per_family=3)

            self.assertTrue(report["pass"])
            self.assertEqual(report["count"], 9)
            self.assertEqual(report["compiled_count"], 9)
            self.assertEqual(report["svg_exact_count"], 9)
            for case in report["cases"]:
                self.assertTrue(Path(case["bundle_path"]).exists())
                self.assertTrue(Path(case["scene_dsl_path"]).exists())
                self.assertTrue(Path(case["svg_path"]).exists())


if __name__ == "__main__":
    unittest.main()
