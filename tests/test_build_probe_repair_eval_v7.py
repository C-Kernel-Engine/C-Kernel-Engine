#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "version" / "v7" / "scripts" / "build_probe_repair_eval_v7.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class ProbeRepairEvalTests(unittest.TestCase):
    def test_replay_scores_raw_and_repaired_paths_separately(self) -> None:
        mod = _load_module(MODULE_PATH, "build_probe_repair_eval_v7")
        with tempfile.TemporaryDirectory(prefix="ck_probe_repair_eval_") as tmp:
            report_path = Path(tmp) / "probe_report.json"
            report_doc = {
                "run_name": "spec16_test_run",
                "output_adapter": "text_renderer",
                "output_adapter_config": {
                    "name": "text_renderer",
                    "stop_markers": ["[/bundle]"],
                    "renderer": "structured_svg_scene_spec16.v1",
                    "preview_mime": "image/svg+xml",
                    "repairer": "spec16_scene_bundle.v1",
                },
                "results": [
                    {
                        "id": "fixable_01",
                        "split": "test",
                        "split_label": "test",
                        "label": "Fixable System",
                        "prompt": "[task:svg] [layout:system_diagram] [form:build_path] [theme:signal_glow] [tone:blue] [density:balanced] [background:mesh] [stages:4] [links:4] [terminal:1] [footer:1] [OUT]",
                        "expected_output": "[bundle] [family:system_diagram] [form:build_path] [theme:signal_glow] [tone:blue] [density:balanced] [background:mesh] [stages:4] [links:4] [terminal:1] [footer:1] [/bundle]",
                        "raw_output": "[bundle] [footer:1] [/bundle]",
                        "response_text": "[bundle] [footer:1] [/bundle]",
                    },
                    {
                        "id": "wrong_form_01",
                        "split": "test",
                        "split_label": "test",
                        "label": "Wrong Timeline Form",
                        "prompt": "[task:svg] [layout:timeline] [form:stage_sequence] [theme:infra_dark] [tone:amber] [density:compact] [background:none] [stages:3] [arrows:2] [footer:0] [OUT]",
                        "expected_output": "[bundle] [family:timeline] [form:stage_sequence] [theme:infra_dark] [tone:amber] [density:compact] [background:none] [stages:3] [arrows:2] [footer:0] [/bundle]",
                        "raw_output": "[bundle] [family:timeline] [form:arena_sections] [theme:infra_dark] [tone:amber] [density:compact] [background:none] [segments:6] [brackets:0] [cards:3] [/bundle]",
                        "response_text": "[bundle] [family:timeline] [form:arena_sections] [theme:infra_dark] [tone:amber] [density:compact] [background:none] [segments:6] [brackets:0] [cards:3] [/bundle]",
                    },
                ],
            }
            report_path.write_text(json.dumps(report_doc, indent=2), encoding="utf-8")

            payload = mod.build_repair_eval(report_path)

            self.assertAlmostEqual(payload["raw"]["totals"]["exact_rate"], 0.0)
            self.assertAlmostEqual(payload["repaired"]["totals"]["exact_rate"], 0.5)
            self.assertEqual(payload["summary"]["repair_applied_count"], 1)
            self.assertEqual(payload["summary"]["exact_improved_count"], 1)
            self.assertEqual(payload["summary"]["exact_regressed_count"], 0)
            self.assertEqual(payload["summary"]["exact_improved_ids"], ["fixable_01"])


if __name__ == "__main__":
    unittest.main()
