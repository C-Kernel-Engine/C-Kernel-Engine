from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HUB_SCRIPT = ROOT / "version/v8/tools/open_ir_hub_v8.py"


def _load_hub():
    spec = importlib.util.spec_from_file_location("v8_ir_hub_likwid_test", HUB_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {HUB_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class V8IrHubLikwidTests(unittest.TestCase):
    def test_hub_indexes_likwid_status_metrics_and_raw_links(self) -> None:
        hub = _load_hub()
        with tempfile.TemporaryDirectory() as tmp:
            models_root = Path(tmp)
            run_dir = models_root / "demo"
            build_dir = run_dir / ".ck_build_v8"
            raw_dir = build_dir / "likwid"
            raw_dir.mkdir(parents=True)
            (run_dir / "ir_report.html").write_text("<html></html>")
            raw_csv = raw_dir / "mem.csv"
            raw_csv.write_text("Metric,Memory bandwidth [MBytes/s],12345\n")
            (build_dir / "likwid_summary.json").write_text(
                json.dumps(
                    {
                        "schema": "cke.profile.likwid.v1",
                        "status": "pass",
                        "selected_groups": ["MEM"],
                        "cpu_ids": [4, 8],
                        "normalized": {
                            "MEM": {"memory_bandwidth_mbytes_per_second": 12345.0}
                        },
                        "artifacts": [{"kind": "MEM csv", "path": str(raw_csv)}],
                    }
                )
            )

            payload = hub.build_index(models_root)
            rendered = hub.render_html(payload)

        self.assertEqual(payload["summary"]["runs_with_likwid"], 1)
        self.assertEqual(len(payload["runs"]), 1)
        summary = payload["runs"][0]["likwid_summary"]
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["selected_groups"], ["MEM"])
        self.assertEqual(summary["cpu_ids"], [4, 8])
        self.assertEqual(summary["metrics"][0]["value"], 12345.0)
        self.assertTrue(summary["summary_uri"].startswith("file://"))
        self.assertTrue(summary["artifacts"][0]["uri"].startswith("file://"))
        deep = next(
            section
            for section in payload["runs"][0]["artifact_sections"]
            if section["key"] == "deep_profiling"
        )
        self.assertIn({"label": "likwid", "ready": True}, deep["items"])
        self.assertIn("renderLikwidSummary", rendered)
        self.assertIn("LIKWID Profiles", rendered)

    def test_rendered_hub_javascript_is_valid(self) -> None:
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is not installed")
        hub = _load_hub()
        html = hub.render_html({"schema": "ck.ir.hub.v2", "runs": [], "summary": {}})
        script = html.rsplit("<script>", 1)[1].split("</script>", 1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            script_path = Path(tmp) / "hub.js"
            script_path.write_text(script)
            completed = subprocess.run(
                [node, "--check", str(script_path)], text=True, capture_output=True
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_hub_surfaces_skip_reason_without_metrics(self) -> None:
        hub = _load_hub()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "likwid_summary.json").write_text(
                json.dumps(
                    {
                        "schema": "cke.profile.likwid.v1",
                        "status": "skip",
                        "reason": "counter access denied",
                    }
                )
            )
            summary = hub._extract_likwid_summary(run_dir)

        self.assertEqual(summary["status"], "skip")
        self.assertEqual(summary["reason"], "counter access denied")
        self.assertEqual(summary["metrics"], [])


if __name__ == "__main__":
    unittest.main()
