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

    def test_flat_circuit_is_not_composite(self) -> None:
        self.assertFalse(self.report["composite"])
        self.assertIsNone(self.report["composition"])

    def test_unknown_circuit_errors(self) -> None:
        with self.assertRaises(SystemExit):
            report.build_circuit_report("no_such_circuit_v8")


class V8ModelNoveltyCompositeCircuitTests(unittest.TestCase):
    """Mode B must resolve components + stitch for composite circuits."""

    @staticmethod
    def _circuit_json(name: str) -> dict:
        return json.loads(
            (ROOT / "version" / "v8" / "circuits" / f"{name}.json").read_text(
                encoding="utf-8"
            )
        )

    def test_cohere_compass_reports_nonzero_with_attribution(self) -> None:
        result = report.build_circuit_report("cohere_compass")
        self.assertTrue(result["composite"])
        composition = result["composition"]
        self.assertEqual(composition["component_count"], 2)
        self.assertEqual(composition["stitch_edge_count"], 1)
        self.assertGreater(result["operations"]["total"], 0)
        self.assertGreater(result["providers"]["total"], 0)

        components = {c["name"]: c for c in composition["components"]}
        self.assertEqual(set(components), {"vision_encoder", "decoder"})
        self.assertEqual(components["vision_encoder"]["circuit"], "cohere_compass_vision")
        self.assertEqual(components["decoder"]["circuit"], "cohere_compass_text")
        self.assertGreater(components["vision_encoder"]["operation_count"], 0)
        self.assertGreater(components["decoder"]["operation_count"], 0)

        # Per-component ops come from the real component circuits: the
        # decoder inherits cohere2's block ops via extends, the vision
        # encoder inherits qwen3_vl_vision's.
        cohere2_ops = report._extract_circuit_ops(self._circuit_json("cohere2"))
        vision_ops = report._extract_circuit_ops(self._circuit_json("qwen3_vl_vision"))
        self.assertTrue(cohere2_ops)
        self.assertTrue(cohere2_ops <= set(components["decoder"]["operations"]))
        self.assertTrue(vision_ops <= set(components["vision_encoder"]["operations"]))

        # Providers keep their component attribution and respect extends
        # overrides: cohere_compass_text overrides rope_qk from cohere2.
        bindings = result["providers"]["bindings"]
        cohere2_kernels = self._circuit_json("cohere2")["kernels"]
        for binding, provider_id in cohere2_kernels.items():
            if binding in ("rope_qk", "rope_qk_decode"):
                continue  # overridden by cohere_compass_text
            self.assertEqual(bindings[f"decoder.{binding}"], provider_id)
        self.assertEqual(bindings["decoder.rope_qk"], "mrope_qk_text_imrope")
        vision_kernels = self._circuit_json("qwen3_vl_vision")["kernels"]
        for binding, provider_id in vision_kernels.items():
            if binding == "position_embeddings":
                continue  # overridden by cohere_compass_vision
            self.assertEqual(bindings[f"vision_encoder.{binding}"], provider_id)
        self.assertEqual(
            bindings["vision_encoder.position_embeddings"],
            "position_embeddings_add_tiled_2d_align_corners_fp32_interp_bf16",
        )

        # Stitch edge ops and providers are aggregated with attribution.
        stitch = composition["stitch"]
        self.assertEqual(len(stitch), 1)
        edge = stitch[0]
        self.assertEqual(edge["id"], "vision_embeddings_to_decoder_prefix")
        self.assertEqual(edge["op"], "multimodal_prefix_stitch")
        self.assertEqual(
            edge["providers"],
            {
                "prefix_insert": "multimodal_prefix_insert_f32",
                "position_builder": "multimodal_mrope_positions_2d",
                "position_transform": "mrope_qk_imrope_positions",
            },
        )
        ops_used = set(result["operations"]["used"])
        self.assertIn("multimodal_prefix_stitch", ops_used)
        self.assertIn("multimodal_prefix_insert", ops_used)
        self.assertIn("multimodal_position_builder", ops_used)
        for binding, provider_id in edge["providers"].items():
            self.assertEqual(
                bindings[f"stitch.vision_embeddings_to_decoder_prefix.{binding}"],
                provider_id,
            )

        # Every aggregated provider still resolves in the registry.
        self.assertEqual(
            result["providers"]["status_counts"]["missing_from_registry"], 0
        )
        self.assertEqual(
            sum(result["providers"]["status_counts"].values()),
            result["providers"]["total"],
        )

    def test_qwen36vl_reports_nonzero_with_attribution(self) -> None:
        result = report.build_circuit_report("qwen36vl")
        self.assertTrue(result["composite"])
        composition = result["composition"]
        self.assertGreater(result["operations"]["total"], 0)
        self.assertGreater(result["providers"]["total"], 0)

        components = {c["name"]: c for c in composition["components"]}
        self.assertEqual(components["vision_encoder"]["circuit"], "qwen3_vl_vision")
        self.assertEqual(components["decoder"]["circuit"], "qwen35")

        qwen35_ops = report._extract_circuit_ops(self._circuit_json("qwen35"))
        self.assertTrue(qwen35_ops <= set(components["decoder"]["operations"]))
        qwen35_kernels = self._circuit_json("qwen35")["kernels"]
        bindings = result["providers"]["bindings"]
        for binding, provider_id in qwen35_kernels.items():
            self.assertEqual(bindings[f"decoder.{binding}"], provider_id)

        edge = composition["stitch"][0]
        self.assertEqual(edge["op"], "multimodal_prefix_stitch")
        self.assertIn("multimodal_prefix_stitch", result["operations"]["used"])
        self.assertEqual(
            result["providers"]["status_counts"]["missing_from_registry"], 0
        )

    def test_composite_cli_markdown(self) -> None:
        proc = subprocess.run(
            ["python3", str(SCRIPT), "--circuit", "cohere_compass"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Composite circuit: yes (2 components, 1 stitch edge)", proc.stdout)
        self.assertIn("multimodal_prefix_stitch", proc.stdout)
        self.assertIn("`decoder` | `cohere_compass_text`", proc.stdout)


class V8ModelNoveltyCompositionFaultTests(unittest.TestCase):
    """Unresolvable compositions must fail loudly, never report zeros."""

    def _run_with_circuits(self, fixtures: dict[str, dict], circuit: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            for name, doc in fixtures.items():
                (tmp_path / f"{name}.json").write_text(
                    json.dumps(doc), encoding="utf-8"
                )
            original = report.CIRCUITS_DIR
            report.CIRCUITS_DIR = tmp_path
            try:
                report.build_circuit_report(circuit)
            finally:
                report.CIRCUITS_DIR = original

    def test_extends_cycle_fails_loudly(self) -> None:
        fixtures = {
            "cycle_a": {"version": 1, "name": "cycle_a", "extends": "cycle_b"},
            "cycle_b": {"version": 1, "name": "cycle_b", "extends": "cycle_a"},
        }
        with self.assertRaises(SystemExit) as ctx:
            self._run_with_circuits(fixtures, "cycle_a")
        self.assertIn("cyclic circuit reference", str(ctx.exception))

    def test_component_cycle_fails_loudly(self) -> None:
        def composite(name: str, other: str) -> dict:
            return {
                "version": 1,
                "name": name,
                "sequence": ["part"],
                "components": {
                    "part": {
                        "runtime_role": "decoder",
                        "circuit": other,
                        "block": "decoder",
                    }
                },
            }

        fixtures = {
            "cycle_c": composite("cycle_c", "cycle_d"),
            "cycle_d": composite("cycle_d", "cycle_c"),
        }
        with self.assertRaises(SystemExit) as ctx:
            self._run_with_circuits(fixtures, "cycle_c")
        self.assertIn("cyclic circuit reference", str(ctx.exception))

    def test_missing_component_circuit_fails_loudly(self) -> None:
        fixtures = {
            "broken": {
                "version": 1,
                "name": "broken",
                "sequence": ["part"],
                "components": {
                    "part": {
                        "runtime_role": "encoder",
                        "circuit": "no_such_component_circuit",
                        "block": "vision_encoder",
                    }
                },
            }
        }
        with self.assertRaises(SystemExit) as ctx:
            self._run_with_circuits(fixtures, "broken")
        self.assertIn("unsupported composition", str(ctx.exception))

    def test_missing_component_block_fails_loudly(self) -> None:
        fixtures = {
            "leaf": {
                "version": 1,
                "name": "leaf",
                "block_types": {"decoder": {"header": ["layernorm"]}},
            },
            "broken": {
                "version": 1,
                "name": "broken",
                "sequence": ["part"],
                "components": {
                    "part": {
                        "runtime_role": "decoder",
                        "circuit": "leaf",
                        "block": "no_such_block",
                    }
                },
            },
        }
        with self.assertRaises(SystemExit) as ctx:
            self._run_with_circuits(fixtures, "broken")
        self.assertIn("unsupported composition", str(ctx.exception))
        self.assertIn("missing block", str(ctx.exception))

    def test_malformed_stitch_edge_fails_loudly(self) -> None:
        fixtures = {
            "broken": {
                "version": 1,
                "name": "broken",
                "block_types": {"decoder": {"header": ["layernorm"]}},
                "stitch": [{"id": "edge_without_op"}],
            }
        }
        with self.assertRaises(SystemExit) as ctx:
            self._run_with_circuits(fixtures, "broken")
        self.assertIn("unsupported composition", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
