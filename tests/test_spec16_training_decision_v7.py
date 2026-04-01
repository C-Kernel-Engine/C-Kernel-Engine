#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECISION_MODULE_PATH = ROOT / "version" / "v7" / "scripts" / "spec16_training_decision_v7.py"
TRAIN_PIPELINE_MODULE_PATH = ROOT / "version" / "v7" / "scripts" / "train_data_pipeline_v7.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _probe_row(*, family: str, split: str, exact: bool, renderable: bool) -> dict[str, object]:
    prompt = f"[task:svg] [layout:{family}] [OUT]"
    return {
        "prompt": prompt,
        "split": split,
        "exact_match": exact,
        "renderable": renderable,
    }


class Spec16TrainingDecisionTests(unittest.TestCase):
    def test_strong_frozen_baseline_blocks_more_raw_repair_training(self) -> None:
        mod = _load_module(DECISION_MODULE_PATH, "spec16_training_decision_v7")
        frozen_doc = {
            "results": [
                _probe_row(family="memory_map", split="train", exact=True, renderable=True),
                _probe_row(family="timeline", split="train", exact=True, renderable=True),
                _probe_row(family="system_diagram", split="train", exact=True, renderable=True),
                _probe_row(family="memory_map", split="hidden_train", exact=True, renderable=True),
                _probe_row(family="timeline", split="hidden_train", exact=True, renderable=True),
                _probe_row(family="system_diagram", split="hidden_test", exact=True, renderable=True),
            ]
        }
        frozen_metrics = mod._probe_metrics_from_doc(frozen_doc)
        descendants = [
            {"run_name": "r10", "overall_exact": 0.70, "renderable": 0.81, "beats_frozen": False, "pilot_gate_clears": False},
            {"run_name": "r11", "overall_exact": 0.81, "renderable": 0.95, "beats_frozen": False, "pilot_gate_clears": False},
        ]

        payload = mod._build_decision_payload(
            frozen_run=Path("/tmp/spec16_r9"),
            frozen_metrics=frozen_metrics,
            descendants=descendants,
        )

        self.assertFalse(payload["training_allowed"])
        self.assertEqual(payload["default_action"], "decode_repair")
        self.assertTrue(payload["block_raw_repair_rungs"])
        self.assertEqual(payload["suggested_next_training_branch"], "capacity_branch")
        self.assertIn("clean capacity branch", " ".join(payload["training_reenable_conditions"]).lower())
        self.assertTrue(any("warning-language rows" in item for item in payload["banned_training_patterns"]))

    def test_training_plan_builder_preserves_operator_policy_keys(self) -> None:
        mod = _load_module(TRAIN_PIPELINE_MODULE_PATH, "train_data_pipeline_v7")
        with tempfile.TemporaryDirectory(prefix="ck_plan_preserve_") as tmp:
            run_dir = Path(tmp)
            training_pipeline = {
                "active_stage": "pretrain",
                "stage_sequence": {
                    "entries": [
                        {"stage": "pretrain", "seq": 1},
                        {"stage": "midtrain", "seq": 2},
                    ]
                },
                "tokenizer_lineage": {
                    "type": "ascii_bpe",
                    "vocab_size": 150,
                    "tokenizer_path": "/tmp/tokenizer.json",
                    "tokenizer_sha256": "abc",
                    "reused_run_tokenizer": True,
                    "tokenizer_corpora": [],
                },
            }
            existing_plan = {
                "schema": "ck.training_plan.v1",
                "created_at": "2026-03-31T00:00:00+00:00",
                "run_policy": {"mode": "pilot"},
                "token_budget": {"selected_pretrain_total_tokens": 123},
                "training_decision": {"default_action": "decode_repair"},
                "stage_order": ["pretrain", "midtrain"],
                "stages": [],
            }

            plan = mod._build_or_update_training_plan_payload(
                run_dir=run_dir,
                training_pipeline=training_pipeline,
                existing_plan=existing_plan,
            )

            self.assertEqual(plan["run_policy"], {"mode": "pilot"})
            self.assertEqual(plan["token_budget"], {"selected_pretrain_total_tokens": 123})
            self.assertEqual(plan["training_decision"], {"default_action": "decode_repair"})


if __name__ == "__main__":
    unittest.main()
