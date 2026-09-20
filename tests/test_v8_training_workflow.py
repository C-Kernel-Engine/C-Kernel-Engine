from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "run_training_workflow_v8.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_training_workflow_v8_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class V8TrainingWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = _load()

    def test_corpus_sources_are_pinned_and_split_before_tokenization(self) -> None:
        spec = json.loads(self.workflow.CORPUS_SPEC.read_text(encoding="utf-8"))
        self.assertNotEqual(spec["train"]["path"], spec["validation"]["path"])
        for split in ("train", "validation"):
            source = ROOT / spec[split]["path"]
            self.assertEqual(self.workflow._git_blob(source), spec[split]["git_blob"])
            self.assertEqual(len(spec[split]["source_revision"]), 40)
            self.assertEqual(len(spec[split]["source_blob"]), 40)
            self.assertEqual(source.stat().st_size, spec[split]["token_count"])
        self.assertEqual(spec["tokenizer"]["vocab_size"], 256)

    def test_final_short_batch_preserves_exact_token_presentations(self) -> None:
        batches = self.workflow._batches(np.arange(10, dtype=np.int32), seq_len=4, epochs=3)
        self.assertEqual(sum(row[2] for row in batches), 30)
        self.assertEqual([row[2] for row in batches[:3]], [4, 4, 2])
        self.assertTrue(all(row[0].shape == (4,) and row[1].shape == (4,) for row in batches))

    def test_v8_export_adds_loader_header_and_rebases_file_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); raw = root / "raw"; out = root / "out"; raw.mkdir()
            (raw / "weights.bump").write_bytes(b"abcdefgh")
            (raw / "weights_manifest.json").write_text(json.dumps({"entries": [{"name": "w", "offset": 4}]}) + "\n")
            config = root / "config.json"; config.write_text("{}\n")
            self.workflow._export_v8_bundle(raw, config, out)
            self.assertEqual((out / "weights.bump").read_bytes()[:8], b"BUMPWGT4")
            manifest = json.loads((out / "weights_manifest.json").read_text())
            self.assertEqual(manifest["entries"][0]["offset"], 132)
            self.assertEqual((out / "weights.bump").read_bytes()[128:], b"abcdefgh")

    def test_workflow_does_not_delegate_training_to_v7_cli(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn('"train-e2e"', source)
        self.assertNotIn('"train", "--run"', source)
        self.assertIn('"v7_training_cli_invoked": False', source)

    def test_generated_runtime_persists_token_weighted_accumulation_count(self) -> None:
        source = (ROOT / "version" / "v7" / "scripts" / "codegen_train_runtime_v7.py").read_text(encoding="utf-8")
        self.assertIn("g_accum_tokens", source)
        self.assertIn("ck_train_get_accum_tokens", source)
        self.assertIn("total_tokens = g_accum_tokens", source)
        self.assertNotIn("int accum_denom = (g_accum_step > 0)", source)

    def test_checkpoint_identity_binds_runtime_dataset_config_and_cursor(self) -> None:
        config = {"seq_len": 32, "epochs": 10, "grad_accum": 8, "lr": 3e-4}
        doc = {
            "schema": "cke.v8.training_checkpoint.v1", "complete": True,
            "runtime_contract_sha256": "a" * 64,
            "training_config_sha256": self.workflow._json_sha256(config),
            "training_config": config, "dataset_sha256": "b" * 64,
            "next_microstep": 11, "total_microsteps": 100,
            "optimizer_step": 1, "accum_counter": 3, "accum_tokens": 96,
            "files": {name: {"file": name, "numel": 1, "sha256": "c" * 64}
                      for name in ("weight", "optimizer_state", "accum")},
        }
        doc["checkpoint_identity_sha256"] = self.workflow._checkpoint_identity(doc)
        expected = {"runtime_contract_sha256": "a" * 64,
                    "training_config_sha256": self.workflow._json_sha256(config),
                    "dataset_sha256": "b" * 64, "total_microsteps": 100, "grad_accum": 8}
        self.workflow._validate_checkpoint_document(doc, expected)
        for field in ("runtime_contract_sha256", "dataset_sha256"):
            altered = dict(expected); altered[field] = "0" * 64
            with self.assertRaises(RuntimeError):
                self.workflow._validate_checkpoint_document(doc, altered)
        altered_doc = copy.deepcopy(doc); altered_doc["next_microstep"] += 1
        with self.assertRaises(RuntimeError):
            self.workflow._validate_checkpoint_document(altered_doc, expected)

    def test_workflow_emits_visualizer_trajectory_and_scoped_performance_evidence(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("open_ir_visualizer_v8.py", source)
        self.assertIn("training_pipeline_latest.json", source)
        self.assertIn("final_partial_flush_wrong_lr", source)
        self.assertIn("generated_c_tokens_per_second", source)
        self.assertIn("generation_tokens_match", source)
        self.assertNotIn('"start_microstep": int(', source)

    def test_v8_runbook_exposes_bounded_training_workflow(self) -> None:
        required = (
            'id="certified-fp32-training"',
            "make v8-training-certify-fp32",
            "make v8-training-workflow-fp32",
            "training_workflow_latest.json",
            "checkpoints/latest.json",
            "ir_report.html",
            "backprop_stitch_runtime_latest.json",
            "open_ir_visualizer_v8.py",
            "v7 training IR builder",
            "LoRA/QLoRA",
        )
        for path in (ROOT / "docs/site/_pages/v8-runbook.html", ROOT / "docs/site/v8-runbook.html"):
            runbook = path.read_text(encoding="utf-8")
            for marker in required:
                with self.subTest(path=path.name, marker=marker):
                    self.assertIn(marker, runbook)
            self.assertNotIn("Training workflows remain in <code>v7</code>.", runbook)
            self.assertNotIn("v7 still owns the promoted training/backprop visualizer lane", runbook)


if __name__ == "__main__":
    unittest.main()
