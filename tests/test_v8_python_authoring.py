from __future__ import annotations

import json
import hashlib
import io
import math
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
V7_PACKAGE_ROOT = ROOT / "version" / "v7"
if str(V7_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(V7_PACKAGE_ROOT))

import ckernel_engine as cke  # noqa: E402


def _model(*, layers: int = 4, vocab: int = 384):
    return cke.models.qwen3_tiny(
        vocab=vocab, dim=32, layers=layers, hidden=64, heads=4, kv_heads=2,
        context_len=32, init="normal_0p02", dtype="float32", name="v8_python_authoring_test",
    )


def _arg(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _passing_report(command: list[str]) -> dict:
    configuration = {
        "architecture": "qwen3_style_dense_reduced", "dtype": "fp32",
        "layers": int(_arg(command, "--layers")), "d_model": int(_arg(command, "--d-model")),
        "hidden": int(_arg(command, "--hidden")), "heads": int(_arg(command, "--num-heads")),
        "kv_heads": int(_arg(command, "--num-kv-heads")),
        "vocab_size": int(_arg(command, "--vocab-size")), "tokenizer": _arg(command, "--tokenizer"),
        "seq_len": int(_arg(command, "--seq-len")), "epochs": int(_arg(command, "--epochs")),
        "grad_accum": int(_arg(command, "--grad-accum")), "optimizer": "generated_c_adamw",
        "lr": float(_arg(command, "--lr")), "beta1": float(_arg(command, "--beta1")),
        "beta2": float(_arg(command, "--beta2")), "eps": float(_arg(command, "--eps")),
        "weight_decay": float(_arg(command, "--weight-decay")),
    }
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    configuration["training_config_sha256"] = hashlib.sha256(encoded).hexdigest()
    corpus_path = Path(_arg(command, "--corpus"))
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True,
    ).stdout.strip()
    case_id = _arg(command, "--matrix-case-id")
    return {
        "schema": "cke.v8.training_workflow.v1", "status": "PASS", "passed": True,
        "matrix_identity": {
            "run_id": _arg(command, "--matrix-run-id"), "case_id": case_id, "profile": case_id,
            "run_dir": _arg(command, "--run-dir"), "report": _arg(command, "--json-out"),
        },
        "execution": {"git_commit": git_commit}, "configuration": configuration,
        "corpus": {"spec_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest()},
    }


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
            self.assertFalse(report["passed"])
            self.assertTrue(report["candidate_inventory_complete"])
            self.assertTrue(report["can_launch_generated_workflow"])
            self.assertEqual(report["resolved_plan_status"], "PENDING_GENERATED_WORKFLOW")
            self.assertEqual(report["missing"], [])
            by_label = {row["label"]: row for row in report["capabilities"]}
            attention = by_label["causal GQA gradient"]["candidates"]
            self.assertTrue(any(row["provider"] == "attention_backward_causal_head_major_gqa" for row in attention))
            self.assertTrue(all(row["registered_tests"] for row in attention))
            self.assertTrue(all(row["provider_selection_status"] == "CANDIDATE_NOT_RESOLVED" for row in attention))
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
            report.write_text(json.dumps(_passing_report(rendered)))

        with tempfile.TemporaryDirectory() as td:
            experiment = cke.v8.compile(
                _model(), run_name="run-test", run_dir=Path(td), command_runner=fake_runner,
            )
            report = experiment.run()
            self.assertTrue(report["passed"])
            self.assertEqual(len(calls), 1)
            self.assertTrue(experiment.preflight_path.is_file())
            self.assertTrue(experiment.experiment_path.is_file())
            preflight = json.loads(experiment.preflight_path.read_text())
            self.assertEqual(preflight["status"], "RESOLVED_EXECUTION_PASS")
            self.assertEqual(preflight["executed_numerical_evidence"], "PASS")

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

    def test_unsupported_authored_semantics_fail_closed(self) -> None:
        variants = {}
        variants["rope_theta"] = cke.models.qwen3_tiny(
            vocab=384, dim=32, layers=4, hidden=64, heads=4, kv_heads=2,
            context_len=32, rope_theta=10_000.0, init="normal_0p02", dtype="float32",
        )
        variants["initialization"] = cke.models.qwen3_tiny(
            vocab=384, dim=32, layers=4, hidden=64, heads=4, kv_heads=2,
            context_len=32, init="xavier_uniform", dtype="float32",
        )
        for field in ("activation", "norm_epsilon", "bias", "changed_child"):
            variants[field] = _model()
        for block in variants["activation"].children()[1:-2]:
            block.activation = "geglu"
        variants["norm_epsilon"].children()[-2].eps = 1e-5
        variants["bias"].children()[1].bias = True
        variants["changed_child"].children()[1].attention = cke.nn.Linear(32, 32, bias=False)
        for field, model in variants.items():
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "cannot preserve authored semantics"):
                    cke.v8.compile(model, run_name=f"unsupported-{field}")

    def test_stale_pass_and_noop_runner_cannot_satisfy_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            stale = run_dir / "training_workflow.json"
            stale.write_text(json.dumps({"schema": "cke.v8.training_workflow.v1", "status": "PASS", "passed": True}))
            experiment = cke.v8.compile(
                _model(), run_name="stale-pass", run_dir=run_dir,
                command_runner=lambda _command, _cwd: None,
            )
            with self.assertRaisesRegex(RuntimeError, "did not publish"):
                experiment.run()
            self.assertFalse(stale.exists())

    def test_historical_inspection_is_read_only_and_identity_bound(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            experiment = cke.v8.compile(_model(), run_name="historical", run_dir=Path(td))
            preflight = experiment.preflight()
            experiment_sha = preflight["experiment_sha256"]
            command = experiment.command(invocation_id="historical-run", experiment_sha256=experiment_sha)
            report = _passing_report(command)
            experiment.report_path.write_text(json.dumps(report) + "\n")
            identity = {
                "run_id": report["matrix_identity"]["run_id"],
                "case_id": report["matrix_identity"]["case_id"],
                "repository_commit": report["execution"]["git_commit"],
                "training_config_sha256": report["configuration"]["training_config_sha256"],
                "corpus_spec_sha256": report["corpus"]["spec_sha256"],
                "train_token_ids_sha256": "tokens-1",
                "python_experiment_sha256": hashlib.sha256(experiment.experiment_path.read_bytes()).hexdigest(),
            }
            report["corpus"]["splits"] = {"train": {"token_ids_sha256": "tokens-1"}}
            experiment.report_path.write_text(json.dumps(report) + "\n")
            manifest = {
                "schema": "cke.v8.training_experiment_manifest.v1", "identity": identity,
                "verdict": {"status": "PASS", "passed": True},
                "artifacts": [{
                    "role": "authored_experiment", "path": experiment.experiment_path.name,
                    "required": True,
                    "sha256": hashlib.sha256(experiment.experiment_path.read_bytes()).hexdigest(),
                }],
            }
            experiment.manifest_path.write_text(json.dumps(manifest) + "\n")
            before = {path: path.read_bytes() for path in (
                experiment.experiment_path, experiment.preflight_path,
                experiment.report_path, experiment.manifest_path,
            )}
            inspection = experiment.inspect_existing()
            self.assertEqual(inspection["status"], "MATCHED")
            self.assertTrue(inspection["read_only"])
            self.assertEqual(before, {path: path.read_bytes() for path in before})

            report["matrix_identity"]["run_id"] = "different-run"
            experiment.report_path.write_text(json.dumps(report) + "\n")
            mismatch = experiment.inspect_existing()
            self.assertEqual(mismatch["status"], "MISMATCH")
            self.assertIn("identity.run_id", mismatch["failures"])

    def test_malformed_stale_and_mismatched_reports_fail_closed(self) -> None:
        def runner(mode: str):
            def write_report(command, _cwd) -> None:
                rendered = [str(part) for part in command]
                report_path = Path(_arg(rendered, "--json-out"))
                if mode == "malformed":
                    report_path.write_text("{bad")
                    return
                report = _passing_report(rendered)
                if mode == "mismatched":
                    report["matrix_identity"]["run_id"] = "old-invocation"
                report_path.write_text(json.dumps(report))
                if mode == "stale":
                    os.utime(report_path, ns=(1, 1))
            return write_report

        for mode, message in (
            ("malformed", "malformed report"),
            ("stale", "report_stale_mtime"),
            ("mismatched", "matrix_identity.run_id"),
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td:
                experiment = cke.v8.compile(
                    _model(), run_name=f"bad-report-{mode}", run_dir=Path(td),
                    command_runner=runner(mode),
                )
                with self.assertRaisesRegex(RuntimeError, message):
                    experiment.run()

    def test_training_configuration_rejects_nonfinite_values(self) -> None:
        for kwargs in (
            {"learning_rate": math.nan}, {"learning_rate": math.inf},
            {"gradient_tolerance": math.nan}, {"gradient_tolerance": math.inf},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "finite"):
                    cke.v8.TrainingConfig(**kwargs)

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
        self.assertIn("Training loss by epoch", source)
        self.assertIn("CKE versus PyTorch discrepancy", source)
        self.assertIn("Open the interactive training IR visualizer", source)
        self.assertIn("tokenizer_roundtrip.json", source)
        self.assertIn("fresh identity-bound execution", source)
        self.assertIn("experiment.inspect_existing()", source)
        self.assertIn("READ-ONLY historical inspection", source)
        self.assertIn("experiment.manifest_path", source)

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

    def test_notebook_executes_through_preflight_without_training(self) -> None:
        notebook_path = (
            ROOT / "version" / "v8" / "notebooks" / "01_generated_training_quickstart.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertTrue(all(cell.get("execution_count") is None for cell in code_cells))
        self.assertTrue(all(not cell.get("outputs") for cell in code_cells))
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(
            os.environ,
            {
                "CKE_NOTEBOOK_RUN_DIR": td,
                "CKE_NOTEBOOK_EXECUTE": "0",
                "CKE_NOTEBOOK_LOAD_EXISTING": "0",
                "CKE_NOTEBOOK_EMBED_IR": "0",
            },
        ), redirect_stdout(io.StringIO()):
            namespace = {"__name__": "__cke_notebook_smoke__"}
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell.get("source", []))
                exec(compile(source, f"{notebook_path}#cell-{index}", "exec"), namespace)
            run_dir = Path(td)
            self.assertTrue((run_dir / "python_training_experiment.json").is_file())
            preflight = json.loads((run_dir / "training_capability_preflight.json").read_text())
            self.assertEqual(preflight["status"], "CANDIDATE_INVENTORY_COMPLETE")
            self.assertTrue(preflight["can_launch_generated_workflow"])
            self.assertFalse((run_dir / "training_workflow.json").exists())


if __name__ == "__main__":
    unittest.main()
