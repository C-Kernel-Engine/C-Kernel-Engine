from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "report_model_novelty_v8.py"


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = _load("report_model_novelty_v8", SCRIPT)

# Cohere2 Command R bring-up commit (exists in this repository).
COHERE2_COMMIT = "f22de7b17"


class V8ModelNoveltyGitRangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        subprocess.check_output(
            ["git", "cat-file", "-e", f"{COHERE2_COMMIT}^{{commit}}"], cwd=ROOT
        )
        cls.report = report.build_git_range_report(
            f"{COHERE2_COMMIT}^", COHERE2_COMMIT, ROOT
        )

    def test_advisory_schema(self) -> None:
        self.assertEqual(self.report["schema"], "cke.v8_model_novelty")
        self.assertTrue(self.report["advisory"])
        self.assertEqual(self.report["mode"], "git_range")

    def test_bucket_classification(self) -> None:
        buckets = self.report["buckets"]
        expected_counts = {
            "circuit": 1,
            "model_map": 1,
            "kernel_map": 2,
            "kernel_c_source": 2,
            "core_compiler": 2,
            "converters": 1,
            "tests_evidence": 5,
            "docs": 0,
            "other": 1,
        }
        for bucket, count in expected_counts.items():
            self.assertEqual(
                buckets[bucket]["files"],
                count,
                f"bucket {bucket}: {buckets[bucket]['paths']}",
            )
        self.assertEqual(self.report["totals"]["files"], 15)
        self.assertEqual(
            sum(buckets[b]["files"] for b in report.BUCKET_ORDER),
            self.report["totals"]["files"],
        )

    def test_core_compiler_attribution(self) -> None:
        core = self.report["core_compiler"]
        self.assertEqual(core["target_trend"], "zero")
        self.assertEqual(
            sorted(core["paths"]),
            [
                "version/v8/scripts/build_ir_v8.py",
                "version/v8/scripts/codegen_prefill_v8.py",
            ],
        )
        self.assertEqual(core["files"], 2)
        self.assertEqual(core["added"], 73)
        self.assertEqual(core["deleted"], 9)

    def test_bucket_paths(self) -> None:
        buckets = self.report["buckets"]
        self.assertEqual(buckets["circuit"]["paths"], ["version/v8/circuits/cohere2.json"])
        self.assertEqual(
            buckets["model_map"]["paths"], ["version/v8/model_maps/gguf_ck_map.json"]
        )
        self.assertIn(
            "version/v8/kernel_maps/final_logit_scale_f32.json",
            buckets["kernel_map"]["paths"],
        )
        self.assertIn("src/kernels/logit_kernels.c", buckets["kernel_c_source"]["paths"])
        self.assertIn(
            "version/v8/scripts/convert_gguf_to_bump_v8.py",
            buckets["converters"]["paths"],
        )
        self.assertIn("tests/test_v8_cohere2_contract.py", buckets["tests_evidence"]["paths"])
        self.assertIn(
            "version/v8/contracts/numerical_execution.json",
            buckets["tests_evidence"]["paths"],
        )

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_out = pathlib.Path(tmp) / "report.json"
            proc = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--base",
                    f"{COHERE2_COMMIT}^",
                    "--head",
                    COHERE2_COMMIT,
                    "--json-out",
                    str(json_out),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("ADVISORY", proc.stdout)
            self.assertIn("Core-compiler surface", proc.stdout)
            payload = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(payload["core_compiler"]["files"], 2)


class V8ModelNoveltyCircuitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = report.build_circuit_report("kimi_vl")

    def test_advisory_schema(self) -> None:
        self.assertEqual(self.report["schema"], "cke.v8_model_novelty")
        self.assertEqual(self.report["schema_version"], 1)
        self.assertTrue(self.report["advisory"])
        self.assertEqual(self.report["mode"], "circuit")
        self.assertEqual(self.report["circuit"], "kimi_vl")

    def test_operations_schema(self) -> None:
        ops = self.report["operations"]
        for key in (
            "used",
            "total",
            "shared",
            "shared_count",
            "unique_to_circuit",
            "unique_count",
            "circuits_compared",
        ):
            self.assertIn(key, ops)
        self.assertEqual(ops["total"], len(ops["used"]))
        self.assertEqual(ops["shared_count"] + ops["unique_count"], ops["total"])
        self.assertGreater(ops["circuits_compared"], 0)
        self.assertIn("mla_attention", ops["used"])

    def test_providers_schema(self) -> None:
        providers = self.report["providers"]
        self.assertGreater(providers["total"], 0)
        counts = providers["status_counts"]
        self.assertEqual(
            sum(counts.values()),
            providers["total"],
            f"status counts must cover every bound provider: {counts}",
        )
        self.assertEqual(counts["missing_from_registry"], 0)
        for detail in providers["detail"]:
            self.assertIn(detail["status"], counts)

    def test_no_fabricated_metadata(self) -> None:
        # Kernel maps do not track structured boundaries or added-by-PR
        # provenance yet; the report must surface explicit nulls, not numbers.
        self.assertIsNone(self.report["numerical_boundaries"])
        self.assertIn("not tracked", self.report["numerical_boundaries_note"])
        self.assertIsNone(self.report["provider_provenance"])
        self.assertIn("not track", self.report["provider_provenance_note"])

    def test_unknown_circuit_errors(self) -> None:
        with self.assertRaises(SystemExit):
            report.build_circuit_report("no_such_circuit_v8")


if __name__ == "__main__":
    unittest.main()
