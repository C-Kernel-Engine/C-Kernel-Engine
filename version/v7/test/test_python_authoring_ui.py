#!/usr/bin/env python3
"""Focused tests for the experimental v7 Python authoring layer."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ckernel_engine.v7 import (  # noqa: E402
    DataSource,
    MaterializeOptions,
    TemplateSpec,
    TinyModelSpec,
    TrainConfig,
    TrainingProject,
    notebook_artifact_dashboard_html,
)


class _FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], Path]] = []

    def __call__(self, command, cwd) -> None:
        rendered = [str(part) for part in command]
        self.calls.append((rendered, Path(cwd)))

        for flag in ("--output", "--index-out"):
            if flag in rendered:
                output_path = Path(rendered[rendered.index(flag) + 1])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text("{}", encoding="utf-8")

        if any(part.endswith("prepare_run_viewer.py") for part in rendered):
            run_dir = Path(rendered[2])
            for artifact_name in ("dataset_viewer.html", "embeddings.json", "attention.json"):
                artifact_path = run_dir / artifact_name
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                artifact_path.write_text("{}", encoding="utf-8")


class PythonAuthoringLayerTest(unittest.TestCase):
    def test_materialize_writes_template_and_init_command(self) -> None:
        fake_runner = _FakeRunner()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            template = TemplateSpec.from_document(
                {
                    "name": "qwen3",
                    "sequence": ["header", "transformer", "footer"],
                    "block_types": {
                        "transformer": {
                            "body": ["qkv_proj", "qk_norm", "attention", "mlp_gate_up", "mlp_down"],
                        }
                    },
                    "contract": {
                        "attention_contract": {
                            "train_runtime_contract": {
                                "save_attn_weights": True,
                                "use_flash_when_safe": False,
                                "rope_layout": "qwen"
                            }
                        }
                    },
                }
            )
            project = TrainingProject(
                run_name="py-ui-test",
                run_dir=run_dir,
                model=TinyModelSpec(),
                template=template,
                command_runner=fake_runner,
            )

            result = project.materialize(MaterializeOptions(generate_ir=True, generate_runtime=True, strict=True))

            self.assertEqual(result.action, "materialize")
            self.assertEqual(len(fake_runner.calls), 1)
            command, cwd = fake_runner.calls[0]
            self.assertEqual(cwd, REPO_ROOT)
            self.assertIn("init", command)
            self.assertIn("--generate-ir", command)
            self.assertIn("--generate-runtime", command)
            self.assertIn("--template-file", command)
            self.assertIn("--train-seed", command)
            self.assertNotIn("--seed", command)

            template_path = run_dir / "template_python_ui.json"
            self.assertTrue(template_path.exists())
            payload = json.loads(template_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["name"], "qwen3")

            plan_path = run_dir / "python_authoring_plan.json"
            self.assertTrue(plan_path.exists())
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(plan["schema"], "ck.python_authoring.v1")
            self.assertEqual(plan["history"][0]["action"], "materialize")

    def test_train_command_uses_prompt_and_report_path(self) -> None:
        fake_runner = _FakeRunner()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            # Simulate init output so train does not need to auto-materialize.
            (run_dir / "weights_manifest.json").write_text("{}", encoding="utf-8")

            project = TrainingProject(
                run_name="py-ui-train",
                run_dir=run_dir,
                command_runner=fake_runner,
            )
            result = project.train(
                DataSource.inline_text("hello kernel engine"),
                TrainConfig(
                    backend="ck",
                    strict=True,
                    epochs=1,
                    seq_len=8,
                    total_tokens=64,
                    grad_accum=2,
                    memory_check=False,
                ),
                auto_materialize=False,
            )

            self.assertEqual(result.action, "train")
            self.assertEqual(len(fake_runner.calls), 1)
            command, _cwd = fake_runner.calls[0]
            self.assertIn("train", command)
            self.assertIn("--prompt", command)
            self.assertIn("hello kernel engine", command)
            self.assertIn("--train-json-out", command)
            self.assertIn("--no-train-memory-check", command)
            report_idx = command.index("--train-json-out") + 1
            self.assertEqual(command[report_idx], str(run_dir / "train_e2e_latest.json"))

            plan = json.loads((run_dir / "python_authoring_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["history"][0]["action"], "train")
            self.assertEqual(plan["history"][0]["payload"]["data"]["kind"], "inline_text")

    def test_prepare_viewers_runs_visualizer_dataset_and_hub_commands(self) -> None:
        fake_runner = _FakeRunner()
        with tempfile.TemporaryDirectory() as tmp:
            models_root = Path(tmp) / "models"
            run_dir = models_root / "train" / "py-ui-visualizers"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "weights_manifest.json").write_text("{}", encoding="utf-8")

            project = TrainingProject(
                run_name="py-ui-visualizers",
                run_dir=run_dir,
                command_runner=fake_runner,
            )

            viewer_artifacts = project.prepare_viewers(force=True)

            self.assertEqual(len(fake_runner.calls), 3)

            generate_command, generate_cwd = fake_runner.calls[0]
            self.assertEqual(generate_cwd, REPO_ROOT)
            self.assertTrue(any(part.endswith("open_ir_visualizer.py") for part in generate_command))
            self.assertIn("--generate", generate_command)
            self.assertIn("--html-only", generate_command)
            self.assertIn("--strict-run-artifacts", generate_command)

            dataset_command, _dataset_cwd = fake_runner.calls[1]
            self.assertTrue(any(part.endswith("prepare_run_viewer.py") for part in dataset_command))
            self.assertIn("--force", dataset_command)

            hub_command, _hub_cwd = fake_runner.calls[2]
            self.assertTrue(any(part.endswith("open_ir_hub.py") for part in hub_command))
            self.assertIn("--models-root", hub_command)
            self.assertIn(str(models_root), hub_command)

            self.assertEqual(viewer_artifacts.run_dir, run_dir)
            self.assertEqual(viewer_artifacts.models_root, models_root)
            self.assertEqual(viewer_artifacts.ir_report, run_dir / "ir_report.html")
            self.assertEqual(viewer_artifacts.dataset_viewer, run_dir / "dataset_viewer.html")
            self.assertEqual(viewer_artifacts.embeddings, run_dir / "embeddings.json")
            self.assertEqual(viewer_artifacts.attention, run_dir / "attention.json")
            self.assertEqual(viewer_artifacts.ir_hub, models_root / "ir_hub.html")
            self.assertEqual(viewer_artifacts.hub_index, models_root / "runs_hub_index.json")

            plan = json.loads((run_dir / "python_authoring_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [entry["action"] for entry in plan["history"]],
                ["generate_ir_report", "prepare_run_viewer", "refresh_ir_hub"],
            )
            self.assertEqual(plan["artifacts"]["ir_report"], str(run_dir / "ir_report.html"))
            self.assertEqual(plan["artifacts"]["dataset_viewer"], str(run_dir / "dataset_viewer.html"))
            self.assertEqual(plan["artifacts"]["ir_hub"], str(models_root / "ir_hub.html"))

    def test_notebook_artifact_dashboard_html_surfaces_optional_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_root = Path(tmp) / "models"
            run_dir = models_root / "train" / "py-ui-dashboard"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "python_authoring_plan.json").write_text("{}", encoding="utf-8")
            (run_dir / "weights_manifest.json").write_text("{}", encoding="utf-8")
            (run_dir / "train_e2e_latest.json").write_text("{}", encoding="utf-8")
            (run_dir / "ir_report.html").write_text("<html></html>", encoding="utf-8")
            (models_root / "ir_hub.html").write_text("<html></html>", encoding="utf-8")

            project = TrainingProject(
                run_name="py-ui-dashboard",
                run_dir=run_dir,
            )

            html = notebook_artifact_dashboard_html(run_dir)
            self.assertIn("v7 Run Artifact Dashboard", html)
            self.assertIn("Open IR Visualizer", html)
            self.assertIn("Open IR Hub", html)
            self.assertIn("Open Dataset Viewer unavailable", html)
            self.assertIn((run_dir / "python_authoring_plan.json").resolve().as_uri(), html)
            self.assertIn((models_root / "ir_hub.html").resolve().as_uri(), html)
            self.assertIn("Requires dataset manifests or a staged dataset workspace.", html)
            self.assertIn("Requires tokenizer.json plus a probe path.", html)

            method_html = project.notebook_artifact_dashboard_html(title="Notebook Surface")
            self.assertIn("Notebook Surface", method_html)
            self.assertIn((run_dir / "ir_report.html").resolve().as_uri(), method_html)


if __name__ == "__main__":
    unittest.main()
