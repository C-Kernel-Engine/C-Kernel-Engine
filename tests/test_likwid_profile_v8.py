from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version/v8/scripts/likwid_profile_v8.py"
VISUALIZER = ROOT / "version/v8/tools/open_ir_visualizer_v8.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("likwid_profile_v8_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LikwidProfileV8Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_module()

    def test_available_groups_are_detected_and_preferred_dynamically(self) -> None:
        available = self.module.parse_available_groups(
            """
            Group name      Description
            FLOPS_DP        Double precision floating point
            MEM             Main memory bandwidth
            CLOCK           Clock frequency
            """
        )
        self.assertEqual([row["name"] for row in available], ["FLOPS_DP", "MEM", "CLOCK"])
        self.assertEqual(
            self.module.choose_groups(available, "auto", 2),
            ["MEM", "CLOCK"],
        )
        self.assertEqual(
            self.module.choose_groups(available, "CACHE,MEM,NOPE", 4),
            ["MEM"],
        )

    def test_csv_metrics_are_preserved_and_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mem.csv"
            path.write_text(
                "\n".join(
                    [
                        "TABLE,Region CKE,Core 0,Core 1",
                        "Metric,Runtime (RDTSC) [s],1.0,1.2",
                        "Metric,Memory bandwidth [MBytes/s],24000,26000",
                        "Metric,CPI,0.80,1.00",
                        "Event,UNC_M_CAS_COUNT_RD,100,120",
                    ]
                )
            )
            metrics, normalized = self.module.parse_likwid_csv(path)
        self.assertEqual(len(metrics), 4)
        self.assertAlmostEqual(normalized["runtime_seconds"], 1.1)
        self.assertAlmostEqual(normalized["memory_bandwidth_mbytes_per_second"], 25000)
        self.assertAlmostEqual(normalized["cpi"], 0.9)

    def test_missing_likwid_is_a_recorded_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with mock.patch.object(self.module.shutil, "which", return_value=None):
                rc = self.module.main(
                    ["--output-dir", str(output), "--", "/bin/true"]
                )
            summary = json.loads((output / "likwid_summary.json").read_text())
        self.assertEqual(rc, 0)
        self.assertEqual(summary["status"], "skip")
        self.assertIn("not installed", summary["reason"])
        self.assertEqual(summary["schema"], "cke.profile.likwid.v1")

    def test_capture_runs_only_available_groups_with_cpu_pinning(self) -> None:
        calls: list[list[str]] = []

        def fake_run(command, **_kwargs):
            calls.append(command)
            if "-a" in command:
                return self.module.subprocess.CompletedProcess(
                    command, 0, "MEM Main memory bandwidth\nCLOCK CPU clock\n", ""
                )
            if "-v" in command:
                return self.module.subprocess.CompletedProcess(
                    command, 0, "likwid-perfctr 5.3\n", ""
                )
            if "-i" in command:
                return self.module.subprocess.CompletedProcess(
                    command, 0, "CPU type: test\n", ""
                )
            csv_path = Path(command[command.index("-o") + 1])
            csv_path.write_text("Metric,Memory bandwidth [MBytes/s],12345\n")
            return self.module.subprocess.CompletedProcess(
                command, 0, "workload output\n", ""
            )

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with mock.patch.object(
                self.module.shutil, "which", return_value="/usr/bin/likwid-perfctr"
            ), mock.patch.object(
                self.module.subprocess, "run", side_effect=fake_run
            ):
                rc = self.module.main(
                    [
                        "--output-dir",
                        str(output),
                        "--groups",
                        "auto",
                        "--max-groups",
                        "1",
                        "--cpus",
                        "4,8",
                        "--",
                        "/bin/true",
                    ]
                )
            summary = json.loads((output / "likwid_summary.json").read_text())
        self.assertEqual(rc, 0)
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["selected_groups"], ["MEM"])
        self.assertEqual(summary["cpu_ids"], [4, 8])
        wrapped = next(command for command in calls if "-g" in command)
        self.assertEqual(wrapped[wrapped.index("-C") + 1], "4,8")
        self.assertEqual(
            summary["normalized"]["MEM"]["memory_bandwidth_mbytes_per_second"],
            12345,
        )

    def test_auto_cpu_selection_stays_inside_allowed_affinity(self) -> None:
        with mock.patch.object(
            self.module.os, "sched_getaffinity", return_value={4, 8, 12}
        ):
            self.assertEqual(self.module.default_cpu_ids(2), [4, 8])

    def test_exported_plot_is_preserved_as_an_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "memory.svg"
            source.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>')
            artifact_dir = root / "run" / "likwid"
            artifact_dir.mkdir(parents=True)
            summary = {"artifacts": []}
            self.module.register_plot_artifacts([source], artifact_dir, summary)
        artifact = summary["artifacts"][0]
        self.assertEqual(artifact["kind"], "plot")
        self.assertEqual(artifact["media_type"], "image/svg+xml")
        self.assertTrue(artifact["path"].endswith("plot_memory.svg"))

    def test_visualizer_loads_summary_and_resolves_raw_artifacts(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "likwid_visualizer_v8_test", VISUALIZER
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {VISUALIZER}")
        visualizer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(visualizer)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            raw = run_dir / "likwid" / "mem.csv"
            raw.parent.mkdir()
            raw.write_text("Metric,CPI,1.0\n")
            plot = run_dir / "likwid" / "memory.svg"
            plot.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>')
            (run_dir / "likwid_summary.json").write_text(
                json.dumps(
                    {
                        "schema": "cke.profile.likwid.v1",
                        "status": "pass",
                        "artifacts": [
                            {"kind": "MEM csv", "path": str(raw)},
                            {"kind": "plot", "path": str(plot)},
                        ],
                        "runs": [{"group": "MEM", "csv_path": str(raw)}],
                    }
                )
            )
            data = visualizer.load_model_data(
                run_dir, run_dir=run_dir, strict_run_artifacts=True
            )
        summary = data["files"]["likwid_summary"]
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["artifacts"][0]["resolved_path"], str(raw))
        self.assertTrue(
            summary["artifacts"][1]["image_data_uri"].startswith(
                "data:image/svg+xml;base64,"
            )
        )
        self.assertEqual(summary["runs"][0]["csv_path_resolved"], str(raw))

    def test_visualizer_discovers_v8_build_directory(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "likwid_visualizer_v8_build_test", VISUALIZER
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {VISUALIZER}")
        visualizer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(visualizer)
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            build_dir = model_dir / ".ck_build_v8"
            build_dir.mkdir()
            (build_dir / "likwid_summary.json").write_text(
                json.dumps({"schema": "cke.profile.likwid.v1", "status": "pass"})
            )
            resolved_build, _ = visualizer.resolve_model_target(str(model_dir))
            data = visualizer.load_model_data(resolved_build)
        self.assertEqual(resolved_build, build_dir)
        self.assertEqual(data["files"]["likwid_summary"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
