from __future__ import annotations

import copy
import json
import hashlib
import importlib.util
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
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
V7_PACKAGE_ROOT = ROOT / "version" / "v7"
if str(V7_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(V7_PACKAGE_ROOT))

import ckernel_engine as cke  # noqa: E402


def _load_workflow_module():
    path = ROOT / "version" / "v8" / "scripts" / "run_training_workflow_v8.py"
    spec = importlib.util.spec_from_file_location("v8_training_workflow_authoring_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model(*, layers: int = 4, vocab: int = 384):
    return cke.models.qwen3_tiny(
        vocab=vocab, dim=32, layers=layers, hidden=64, heads=4, kv_heads=2,
        context_len=32, init="normal_0p02", dtype="float32", name="v8_python_authoring_test",
    )


def _arg(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _passing_report(command: list[str]) -> dict:
    semantic = json.loads(Path(_arg(command, "--semantic-model")).read_text())
    model = semantic["model_contract"]
    configuration = {
        "architecture": "qwen3_style_dense_reduced", "dtype": "fp32",
        "layers": model["layers"], "d_model": model["d_model"],
        "hidden": model["hidden"], "heads": model["heads"],
        "kv_heads": model["kv_heads"], "rope_theta": model["rope_theta"],
        "vocab_size": int(_arg(command, "--vocab-size")), "tokenizer": _arg(command, "--tokenizer"),
        "seq_len": model["seq_len"], "epochs": int(_arg(command, "--epochs")),
        "grad_accum": int(_arg(command, "--grad-accum")), "optimizer": "generated_c_adamw",
        "lr": float(_arg(command, "--lr")), "beta1": float(_arg(command, "--beta1")),
        "beta2": float(_arg(command, "--beta2")), "eps": float(_arg(command, "--eps")),
        "weight_decay": float(_arg(command, "--weight-decay")),
        "semantic_model_sha256": _arg(command, "--semantic-model-sha256"),
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
            self.assertNotIn("--layers", command)
            semantic = json.loads(Path(_arg(command, "--semantic-model")).read_text()) if Path(_arg(command, "--semantic-model")).exists() else experiment.semantic_model_document()
            self.assertEqual(semantic["model_contract"]["layers"], 4)
            self.assertEqual(semantic["model_contract"]["kv_heads"], 2)
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

    def test_five_layer_graph_lowers_authored_semantics_and_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            experiment = cke.v8.compile(
                _model(layers=5), run_name="five-layer-semantic", run_dir=Path(td),
            )
            preflight = experiment.preflight()
            semantic = json.loads(experiment.semantic_model_path.read_text())
            self.assertEqual(semantic["model_contract"]["layers"], 5)
            self.assertEqual(semantic["model_contract"]["rope_theta"], 1_000_000.0)
            self.assertEqual(len(semantic["operation_trace"]["layers"]), 5)
            self.assertEqual(len(semantic["parameter_contract"]), 53)
            self.assertEqual(preflight["semantic_model_sha256"], hashlib.sha256(
                experiment.semantic_model_path.read_bytes()
            ).hexdigest())
            command = experiment.command()
            self.assertNotIn("--layers", command)
            self.assertEqual(_arg(command, "--semantic-model"), str(experiment.semantic_model_path))

    def test_allowed_rope_change_changes_lowering_and_ordered_authored_identities_are_preserved(self) -> None:
        base = _model(layers=5)
        changed = cke.models.qwen3_tiny(
            vocab=384, dim=32, layers=5, hidden=64, heads=4, kv_heads=2,
            context_len=32, rope_theta=10_000.0, init="normal_0p02", dtype="float32",
            name="v8_python_authoring_test",
        )
        base_doc = cke.v8.compile(base, run_name="base-rope").semantic_model_document()
        changed_doc = cke.v8.compile(changed, run_name="changed-rope").semantic_model_document()
        self.assertNotEqual(base_doc["model_contract"]["rope_theta"], changed_doc["model_contract"]["rope_theta"])
        self.assertNotEqual(
            base_doc["operation_trace"]["layers"][0]["rope_theta"],
            changed_doc["operation_trace"]["layers"][0]["rope_theta"],
        )
        self.assertNotEqual(
            base_doc["template"]["python_semantic_lowering"]["layers"][0]["rope_theta"],
            changed_doc["template"]["python_semantic_lowering"]["layers"][0]["rope_theta"],
        )
        self.assertNotEqual(
            hashlib.sha256(json.dumps(base_doc, sort_keys=True).encode()).hexdigest(),
            hashlib.sha256(json.dumps(changed_doc, sort_keys=True).encode()).hexdigest(),
        )

        first, second = base._modules["1"], base._modules["2"]
        first._name = "distinguishable_alpha"
        second._name = "distinguishable_beta"
        ordered = cke.v8.compile(base, run_name="ordered-blocks").semantic_model_document()
        base._modules["1"], base._modules["2"] = second, first
        reordered = cke.v8.compile(base, run_name="reordered-blocks").semantic_model_document()
        self.assertEqual(
            [row["block_label"] for row in ordered["operation_trace"]["layers"][:2]],
            ["distinguishable_alpha", "distinguishable_beta"],
        )
        self.assertEqual(
            [row["block_label"] for row in reordered["operation_trace"]["layers"][:2]],
            ["distinguishable_beta", "distinguishable_alpha"],
        )

    def test_semantic_trace_rejects_wrong_ownership_missing_phases_and_bad_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            experiment = cke.v8.compile(_model(layers=5), run_name="trace-negative", run_dir=run_dir)
            experiment.preflight()
            workflow = _load_workflow_module()
            args = SimpleNamespace(
                semantic_model=experiment.semantic_model_path,
                semantic_model_sha256=_arg(experiment.command(), "--semantic-model-sha256"),
                run_dir=run_dir, vocab_size=384,
            )
            semantic, template_path = workflow._apply_semantic_model(args)
            subprocess.run([
                sys.executable, str(ROOT / "version/v7/scripts/ck_run_v7.py"), "init",
                "--run", str(run_dir), "--allow-non-cache-run-dir", "--train-seed", "42",
                "--init", "normal_0p02", "--layers", "5", "--vocab-size", "384",
                "--embed-dim", "32", "--hidden-dim", "64", "--num-heads", "4",
                "--num-kv-heads", "2", "--context-len", "32", "--rope-theta", "1000000",
                "--template", "qwen3", "--template-file", str(template_path),
                "--omit-linear-biases", "--generate-ir", "--train-bridge-lowering", "explicit",
            ], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            assert semantic is not None
            evidence = workflow._validate_semantic_operation_trace(run_dir, semantic)
            self.assertEqual(evidence["lowered_kernel_operations"], 302)
            self.assertNotIn("executed_kernel_operations", evidence)

            ir_path = run_dir / "ir2_train_backward.json"
            original = json.loads(ir_path.read_text())

            wrong_owner = copy.deepcopy(original)
            q_proj = next(row for row in wrong_owner["forward"] if row.get("op") == "q_proj")
            q_proj["authored_semantic_id"] = semantic["operation_trace"]["layers"][0]["feed_forward"]
            ir_path.write_text(json.dumps(wrong_owner))
            with self.assertRaisesRegex(RuntimeError, "ownership"):
                workflow._validate_semantic_operation_trace(run_dir, semantic)

            for phase in ("forward", "backward"):
                missing_phase = copy.deepcopy(original)
                missing_phase[phase] = []
                ir_path.write_text(json.dumps(missing_phase))
                with self.subTest(phase=phase), self.assertRaisesRegex(RuntimeError, f"no {phase}"):
                    workflow._validate_semantic_operation_trace(run_dir, semantic)

            missing_operation = copy.deepcopy(original)
            embedding_id = semantic["operation_trace"]["embedding"]
            missing_operation["forward"] = [
                row for row in missing_operation["forward"]
                if row.get("authored_semantic_id") != embedding_id
            ]
            ir_path.write_text(json.dumps(missing_operation))
            with self.assertRaisesRegex(RuntimeError, "missing_operations"):
                workflow._validate_semantic_operation_trace(run_dir, semantic)

            bad_lineage = copy.deepcopy(original)
            bad_lineage["backward"][0]["authored_semantic_id"] = embedding_id
            ir_path.write_text(json.dumps(bad_lineage))
            with self.assertRaisesRegex(RuntimeError, "lineage"):
                workflow._validate_semantic_operation_trace(run_dir, semantic)

            wrong_forward_operation = copy.deepcopy(original)
            q_backward = next(
                row for row in wrong_forward_operation["backward"]
                if row.get("op") == "q_proj_backward_core"
            )
            k_forward = next(
                row for row in wrong_forward_operation["forward"]
                if row.get("op") == "k_proj" and row.get("layer") == q_backward.get("layer")
            )
            q_backward["forward_ref"] = k_forward["op_id"]
            ir_path.write_text(json.dumps(wrong_forward_operation))
            with self.assertRaisesRegex(RuntimeError, "operation_mismatch"):
                workflow._validate_semantic_operation_trace(run_dir, semantic)
            ir_path.write_text(json.dumps(original))

    def test_semantic_parameter_contract_rejects_wrong_valid_owner(self) -> None:
        experiment = cke.v8.compile(_model(layers=5), run_name="parameter-owner")
        semantic = experiment.semantic_model_document()
        expected = [
            {"name": row["runtime_parameter"], "shape": row["shape"]}
            for row in semantic["parameter_contract"]
        ]
        bad = copy.deepcopy(semantic)
        q_weight = next(row for row in bad["parameter_contract"] if row["runtime_parameter"] == "layer.0.wq")
        q_weight["authored_semantic_id"] = bad["operation_trace"]["layers"][0]["feed_forward"]
        workflow = _load_workflow_module()
        with self.assertRaisesRegex(RuntimeError, "parameter ownership mismatch"):
            workflow._validate_semantic_parameter_contract(bad, expected)

    def test_manifest_source_ignores_stale_semantic_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            stale_semantic = run_dir / "python_training_semantic_model.json"
            stale_template = run_dir / "python_training_lowered_template.json"
            stale_semantic.write_text('{"stale": true}\n')
            stale_template.write_text('{"stale": true}\n')
            workflow = _load_workflow_module()
            authored, source, artifacts = workflow._resolve_authored_manifest_source(
                run_dir=run_dir, identity={}, semantic_model_path=None,
                semantic_model_sha256=None, semantic_template_path=None,
            )
            self.assertEqual(authored, run_dir / "template_train.json")
            self.assertEqual(source, "workflow_configuration")
            self.assertEqual(artifacts, [])
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                workflow._resolve_authored_manifest_source(
                    run_dir=run_dir, identity={}, semantic_model_path=stale_semantic,
                    semantic_model_sha256="0" * 64, semantic_template_path=stale_template,
                )

    def test_allowed_rope_change_reaches_generated_ir_and_independent_torch_math(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            model = cke.models.qwen3_tiny(
                vocab=384, dim=32, layers=5, hidden=64, heads=4, kv_heads=2,
                context_len=32, rope_theta=10_000.0, init="normal_0p02", dtype="float32",
            )
            experiment = cke.v8.compile(model, run_name="rope-ir", run_dir=run_dir)
            experiment.preflight()
            workflow = _load_workflow_module()
            args = SimpleNamespace(
                semantic_model=experiment.semantic_model_path,
                semantic_model_sha256=_arg(experiment.command(), "--semantic-model-sha256"),
                run_dir=run_dir, vocab_size=384,
            )
            _document, template_path = workflow._apply_semantic_model(args)
            subprocess.run([
                sys.executable, str(ROOT / "version/v7/scripts/ck_run_v7.py"), "init",
                "--run", str(run_dir), "--allow-non-cache-run-dir", "--train-seed", "42",
                "--init", "normal_0p02", "--layers", "5", "--vocab-size", "384",
                "--embed-dim", "32", "--hidden-dim", "64", "--num-heads", "4",
                "--num-kv-heads", "2", "--context-len", "32", "--rope-theta", "10000",
                "--template", "qwen3", "--template-file", str(template_path),
                "--omit-linear-biases", "--generate-ir", "--train-bridge-lowering", "explicit",
            ], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            ir1 = json.loads((run_dir / "ir1_train_forward.json").read_text())
            rope_ops = [row for row in ir1["ops"] if row.get("op") == "rope_qk"]
            self.assertEqual(
                [row["authored_semantic_properties"]["rope_theta"] for row in rope_ops],
                [10_000.0] * 5,
            )

            oracle_path = ROOT / "version/v7/scripts/oracle_snapshot_torch_v7.py"
            spec = importlib.util.spec_from_file_location("v8_rope_oracle_test", oracle_path)
            assert spec is not None and spec.loader is not None
            oracle = importlib.util.module_from_spec(spec); spec.loader.exec_module(oracle)
            q = torch.arange(32, dtype=torch.float32).reshape(1, 1, 4, 8)
            k = q.flip(-1)
            q_10k, _ = oracle._apply_rope(q, k, 10_000.0, "split")
            q_1m, _ = oracle._apply_rope(q, k, 1_000_000.0, "split")
            self.assertGreater(float(torch.max(torch.abs(q_10k - q_1m))), 0.0)

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
        with self.assertRaisesRegex(ValueError, "exactly 2, 4, 5, 6, or 10"):
            cke.v8.compile(_model(layers=3), run_name="bad-depth")
        self.assertEqual(cke.v8.compile(_model(layers=2), run_name="two-layer").contract["layers"], 2)

        model = _model()
        model.extra = cke.nn.Linear(32, 32)
        with self.assertRaisesRegex(ValueError, "currently expects"):
            cke.v8.compile(model, run_name="bad-module")

    def test_unsupported_authored_semantics_fail_closed(self) -> None:
        variants = {}
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

    def test_semantic_model_identity_mismatch_fails_before_lowering(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            experiment = cke.v8.compile(_model(layers=5), run_name="semantic-tamper", run_dir=Path(td))
            experiment.preflight()
            workflow = _load_workflow_module()
            args = SimpleNamespace(
                semantic_model=experiment.semantic_model_path,
                semantic_model_sha256="0" * 64,
                run_dir=Path(td), vocab_size=384,
            )
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                workflow._apply_semantic_model(args)
            self.assertFalse((Path(td) / "python_training_lowered_template.json").exists())

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

    def test_svg_notebook_preflights_the_frozen_fixture_without_training(self) -> None:
        notebook_path = ROOT / "version/v8/notebooks/02_frozen_svg_training_fixture.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertTrue(all(cell.get("execution_count") is None and not cell.get("outputs") for cell in code_cells))
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(
            os.environ,
            {
                "CKE_NOTEBOOK_RUN_DIR": td,
                "CKE_NOTEBOOK_EXECUTE": "0",
                "CKE_NOTEBOOK_LOAD_EXISTING": "0",
                "CKE_NOTEBOOK_EMBED_IR": "0",
            },
        ), redirect_stdout(io.StringIO()):
            namespace = {"__name__": "__cke_svg_notebook_smoke__"}
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell.get("source", [])), f"{notebook_path}#cell-{index}", "exec"), namespace)
            run_dir = Path(td)
            preflight = json.loads((run_dir / "training_capability_preflight.json").read_text())
            authored = json.loads((run_dir / "python_training_experiment.json").read_text())
            self.assertEqual(preflight["status"], "CANDIDATE_INVENTORY_COMPLETE")
            self.assertTrue(preflight["can_launch_generated_workflow"])
            self.assertIn("svg_single_document_v1.json", json.dumps(authored))
            self.assertFalse((run_dir / "training_workflow.json").exists())


if __name__ == "__main__":
    unittest.main()
