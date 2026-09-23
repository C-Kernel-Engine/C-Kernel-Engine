from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V7_PACKAGE_ROOT = ROOT / "version" / "v7"
if str(V7_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(V7_PACKAGE_ROOT))

import ckernel_engine as cke  # noqa: E402


def _model(*, layers: int = 4, vocab: int = 384):
    return cke.models.qwen3_tiny(
        vocab=vocab, dim=32, layers=layers, hidden=64, heads=4, kv_heads=2,
        context_len=32, dtype="float32", name="v8_python_authoring_test",
    )


class V8PythonAuthoringTests(unittest.TestCase):
    def test_v8_reuses_existing_nn_surface_and_emits_generated_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            experiment = cke.v8.compile(
                _model(), run_name="authoring-test", run_dir=Path(td),
                dataset=cke.v8.DatasetConfig(max_train_tokens=320, max_validation_tokens=64),
            )
            command = experiment.command()
            self.assertIsInstance(experiment.model, cke.nn.Module)
            self.assertTrue(command[1].endswith("version/v8/scripts/run_training_workflow_v8.py"))
            self.assertEqual(command[command.index("--layers") + 1], "4")
            self.assertEqual(command[command.index("--num-kv-heads") + 1], "2")
            self.assertNotIn("torch", " ".join(command).lower())

    def test_preflight_derives_providers_and_tests_from_kernel_maps(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            experiment = cke.v8.compile(_model(), run_name="preflight-test", run_dir=Path(td))
            report = experiment.preflight()
            self.assertTrue(report["passed"])
            self.assertEqual(report["missing"], [])
            by_label = {row["label"]: row for row in report["capabilities"]}
            attention = by_label["causal GQA gradient"]["candidates"]
            self.assertTrue(any(row["provider"] == "attention_backward_causal_head_major_gqa" for row in attention))
            self.assertTrue(all(row["tests"] for row in attention))
            self.assertTrue(all("tensor_contract" in row for row in attention))
            self.assertTrue(all("saved_for_backward" in row for row in attention))
            self.assertTrue(all("implementation" in row for row in attention))
            persisted = json.loads(experiment.preflight_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["experiment_sha256"], report["experiment_sha256"])

    def test_experiment_document_records_python_boundary_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            experiment = cke.v8.compile(_model(), run_name="document-test", run_dir=Path(td))
            document = experiment.experiment_document()
            self.assertEqual(document["execution_backend"], "generated_c")
            self.assertEqual(document["model"]["graph"]["schema"], "ck.python_authoring.graph.v1")
            self.assertEqual(document["training"]["optimizer"], "generated_c_adamw")
            self.assertEqual(
                document["deployment"]["inference_runtime"], "independently_generated_v8_c"
            )
            self.assertFalse(document["deployment"]["python_required_by_generated_runtime"])
            self.assertEqual(document["deployment"]["standalone_native_executable"], "NOT_CERTIFIED")

    def test_run_preflights_then_reads_generated_workflow_report(self) -> None:
        calls: list[tuple[list[str], Path]] = []

        def fake_runner(command, cwd) -> None:
            rendered = [str(part) for part in command]
            calls.append((rendered, Path(cwd)))
            report = Path(rendered[rendered.index("--json-out") + 1])
            report.write_text(json.dumps({"schema": "cke.v8.training_workflow.v1", "status": "PASS", "passed": True}))

        with tempfile.TemporaryDirectory() as td:
            experiment = cke.v8.compile(
                _model(), run_name="run-test", run_dir=Path(td), command_runner=fake_runner,
            )
            report = experiment.run()
            self.assertTrue(report["passed"])
            self.assertEqual(len(calls), 1)
            self.assertTrue(experiment.preflight_path.is_file())
            self.assertTrue(experiment.experiment_path.is_file())

    def test_failed_generated_workflow_names_its_report(self) -> None:
        def failed_runner(command, _cwd) -> None:
            rendered = [str(part) for part in command]
            report = Path(rendered[rendered.index("--json-out") + 1])
            report.write_text(json.dumps({"status": "FAIL", "passed": False, "failures": ["injected"]}))
            raise subprocess.CalledProcessError(7, rendered)

        with tempfile.TemporaryDirectory() as td:
            experiment = cke.v8.compile(
                _model(), run_name="failed-run", run_dir=Path(td), command_runner=failed_runner,
            )
            with self.assertRaisesRegex(RuntimeError, r"report=.*training_workflow.json.*injected"):
                experiment.run()

    def test_unsupported_depth_and_module_fail_before_execution(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 4, 6, or 10"):
            cke.v8.compile(_model(layers=2), run_name="bad-depth")

        model = _model()
        model.extra = cke.nn.Linear(32, 32)
        with self.assertRaisesRegex(ValueError, "currently expects"):
            cke.v8.compile(model, run_name="bad-module")

    def test_rwkv_is_not_exposed_by_the_supported_authoring_surface(self) -> None:
        self.assertFalse(hasattr(cke.nn, "RWKV"))
        self.assertFalse(hasattr(cke.models, "rwkv"))

    def test_notebook_and_ci_companion_use_the_same_v8_adapter(self) -> None:
        notebook_path = ROOT / "version" / "v8" / "notebooks" / "01_generated_training_quickstart.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )
        self.assertIn("cke.v8.compile", source)
        self.assertIn("experiment.preflight()", source)
        self.assertIn("experiment.run()", source)
        self.assertIn("checkpoint/resume", source)
        self.assertIn("inference export", source)

        example = (ROOT / "version" / "v8" / "examples" / "python_authoring_tiny_lm_v8.py").read_text()
        self.assertIn("cke.v8.compile", example)
        self.assertIn("experiment.preflight()", example)
        makefile = (ROOT / "Makefile").read_text()
        self.assertIn("v8-training-python-authoring-smoke:", makefile)
        self.assertIn("version/v8/examples/python_authoring_tiny_lm_v8.py", makefile)

        runbook = (ROOT / "docs" / "site" / "_pages" / "v8-runbook.html").read_text()
        self.assertIn("id=\"python-training-authoring\"", runbook)
        self.assertIn("v8-training-python-authoring-smoke", runbook)
        self.assertIn("version/v8/notebooks/01_generated_training_quickstart.ipynb", runbook)
        self.assertIn("RWKV is outside this workstream", runbook)


if __name__ == "__main__":
    unittest.main()
