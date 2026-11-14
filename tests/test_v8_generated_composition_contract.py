"""Structural composition is mandatory, but never a substitute for execution."""

from copy import deepcopy
import unittest
import subprocess
import sys
import tempfile
from pathlib import Path
from version.v8.scripts.generated_composition_contract_v8 import (
    audit_generated_composition,
    component_only_status,
    require_generated_e2e,
)
from tests.test_v8_selective_training import fixture


class CompositionGateTests(unittest.TestCase):
    def test_real_internal_edge_is_structural_only(self):
        ir = fixture()
        # Give operations unique identities without changing their semantics.
        for n, op in enumerate(ir["ops"]):
            op["op"] = f"node{n}"
        edges = [dict(tensor="act.bridge", producer="node2", consumer="node3")]
        result = audit_generated_composition(
            ir, edges, "act.score", {"skipped_ops": []}
        )
        self.assertEqual(result["structural"], "PASS")
        self.assertEqual(result["execution"], "NOT_TESTED")
        with self.assertRaises(ValueError):
            require_generated_e2e(dict(generated_composition=result))
        for defect in ("host", "shape", "order", "dead", "skip", "missing"):
            graph, boundaries, summary = (
                deepcopy(ir),
                deepcopy(edges),
                {"skipped_ops": []},
            )
            if defect == "host":
                graph["ops"].pop(2)
                graph["tensors"]["act.bridge"]["kind"] = "input"
            elif defect == "shape":
                graph["ops"][3]["dataflow"]["inputs"]["input"]["shape"] = [1, 16]
            elif defect == "order":
                graph["ops"][2:4] = reversed(graph["ops"][2:4])
            elif defect == "dead":
                graph["ops"][3]["dataflow"]["inputs"]["input"] = deepcopy(
                    graph["ops"][1]["dataflow"]["inputs"]["input"]
                )
            elif defect == "skip":
                summary["skipped_ops"] = ["bridge"]
            else:
                boundaries = []
            with self.subTest(defect=defect):
                self.assertEqual(
                    audit_generated_composition(
                        graph, boundaries, "act.score", summary
                    )["structural"],
                    "FAIL",
                )

    def test_dead_declared_consumer_cannot_borrow_live_sibling_path(self):
        ir = fixture()
        for n, op in enumerate(ir["ops"]):
            op["op"] = f"node{n}"
        dead = deepcopy(ir["ops"][3])
        dead["op"] = "dead_consumer"
        for ref in dead["dataflow"]["outputs"].values():
            old = ref["tensor"]
            ref["tensor"] = "unused." + old
            ir["tensors"][ref["tensor"]] = deepcopy(ir["tensors"][old])
        ir["ops"].append(dead)
        for consumer, verdict in (("node3", "PASS"), ("dead_consumer", "FAIL")):
            result = audit_generated_composition(
                ir,
                [dict(tensor="act.bridge", producer="node2", consumer=consumer)],
                "act.score",
                {"skipped_ops": []},
            )
            self.assertEqual(result["structural"], verdict)
        self.assertIn(
            "declared boundary consumer is disconnected from final output",
            result["errors"],
        )

    def test_incomplete_boundary_metadata_fails_even_when_refs_agree(self):
        ir = fixture()
        for n, op in enumerate(ir["ops"]):
            op["op"] = f"node{n}"
        edges = [dict(tensor="act.bridge", producer="node2", consumer="node3")]
        for key in ("shape", "dtype", "numel"):
            graph = deepcopy(ir)
            graph["tensors"]["act.bridge"].pop(key)
            for op in graph["ops"]:
                for field in ("inputs", "outputs"):
                    for ref in op["dataflow"].get(field, {}).values():
                        if ref.get("tensor") == "act.bridge":
                            ref.pop(key, None)
            with self.subTest(key=key):
                self.assertEqual(
                    audit_generated_composition(
                        graph, edges, "act.score", {"skipped_ops": []}
                    )["structural"],
                    "FAIL",
                )
        for boundary in (None, {}, dict(tensor="act.bridge", producer="node2")):
            self.assertEqual(
                audit_generated_composition(
                    ir, [boundary], "act.score", {"skipped_ops": []}
                )["structural"],
                "FAIL",
            )

    def test_real_commands_reject_e2e_before_assets_or_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            cases = {
                "run_native_clef": (
                    "release",
                    "lock",
                    "bundle",
                    "bundle-descriptor",
                    "request",
                    "lock-sha256",
                    "bundle-sha256",
                ),
                "train_native_clef": (
                    "release",
                    "native-run",
                    "labels",
                    "report-sha256",
                ),
            }
            for module, names in cases.items():
                command = [
                    sys.executable,
                    "-m",
                    "version.v8.decision." + module,
                    "--require-generated-e2e",
                    "--output",
                    str(output),
                ]
                for name in names:
                    command += ["--" + name, "intentionally-unavailable"]
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=30
                )
                with self.subTest(module=module):
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(
                        "generated E2E lacks verified composition", result.stderr
                    )
                    self.assertFalse(output.exists())

    def test_component_pass_never_implies_e2e(self):
        for mode in ("inference", "training"):
            report = dict(
                execution="PASS", generated_composition=component_only_status(mode)
            )
            with self.assertRaises(ValueError):
                require_generated_e2e(report)
        for report in ({}, dict(execution="PASS"), dict(generated_composition={})):
            with self.assertRaises(ValueError):
                require_generated_e2e(report)

    def test_individual_evidence_cannot_be_omitted(self):
        good = dict(
            mode="training",
            structural="PASS",
            execution="PASS",
            numerical_parity="PASS",
            export_reload="PASS",
            gradients="PASS",
            optimizer="PASS",
            frozen_parameters="PASS",
            host_component_handoff=False,
        )
        # Only tests report validation; this fabricated record is NOT model evidence.
        ir = fixture()
        for n, op in enumerate(ir["ops"]):
            op["op"] = f"node{n}"
        audit_args = dict(
            ir=ir,
            boundaries=[dict(tensor="act.bridge", producer="node2", consumer="node3")],
            output="act.score",
            codegen_report={"skipped_ops": []},
        )
        with self.assertRaises(ValueError):
            require_generated_e2e(dict(generated_composition=good))
        for key in good:
            candidate = dict(good)
            candidate.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                require_generated_e2e(
                    dict(generated_composition=candidate), **audit_args
                )
