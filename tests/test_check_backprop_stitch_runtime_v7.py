#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v7" / "scripts" / "check_backprop_stitch_runtime_v7.py"


def _load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stitch = _load_module("check_backprop_stitch_runtime_v7_test", SCRIPT)


class CheckBackpropStitchRuntimeTests(unittest.TestCase):
    def test_smoke_mode_does_not_require_raw_pass_parity_when_first_checked_step_is_clean(self) -> None:
        report = {
            "pass_parity": False,
            "train_dims": {
                "requested": {"vocab": 256, "d_model": 64, "hidden": 128, "num_layers": 1},
                "effective": {"vocab": 128, "d_model": 32, "hidden": 64, "num_layers": 2},
                "mismatches": {
                    "vocab": {"requested": 256, "effective": 128},
                    "d_model": {"requested": 64, "effective": 32},
                    "hidden": {"requested": 128, "effective": 64},
                    "num_layers": {"requested": 1, "effective": 2},
                },
                "source": "run_manifest",
            },
            "oracle": {"check_dump_files": ["a.bin"]},
            "parity_steps": [
                {
                    "checked": True,
                    "step": 1,
                    "oracle_error": None,
                    "first_bad_tensor": None,
                    "first_bad_op": None,
                    "first_bad_diff": None,
                    "slots_compared": 36,
                    "slots_matched": 36,
                    "loss_diff": 0.0,
                    "logits_max_abs_diff": 1e-6,
                },
                {
                    "checked": True,
                    "step": 2,
                    "oracle_error": None,
                    "first_bad_tensor": None,
                    "first_bad_op": None,
                    "first_bad_diff": None,
                    "slots_compared": 36,
                    "slots_matched": 36,
                    "loss_diff": 0.54,
                    "logits_max_abs_diff": 1e-6,
                },
            ],
        }
        out = stitch._evaluate(
            report,
            manifest_dims={"vocab": 128, "d_model": 32, "hidden": 64, "num_layers": 2},
            expect_mismatch=True,
            max_first_loss_diff=1e-5,
            max_first_logits_diff=2e-4,
            require_check_dumps=True,
            require_all_checked_clean=False,
        )
        self.assertTrue(out["passed"])
        self.assertFalse(out["checks"]["pass_parity"])
        self.assertTrue(out["checks"]["pass_parity_gate_applied"]["passed"])
        self.assertFalse(out["checks"]["pass_parity_gate_applied"]["required"])

    def test_strict_mode_still_requires_raw_pass_parity(self) -> None:
        report = {
            "pass_parity": False,
            "train_dims": {
                "requested": {"vocab": 256, "d_model": 64, "hidden": 128, "num_layers": 1},
                "effective": {"vocab": 128, "d_model": 32, "hidden": 64, "num_layers": 2},
                "mismatches": {"vocab": {"requested": 256, "effective": 128}},
                "source": "run_manifest",
            },
            "oracle": {"check_dump_files": ["a.bin"]},
            "parity_steps": [
                {
                    "checked": True,
                    "step": 1,
                    "oracle_error": None,
                    "first_bad_tensor": None,
                    "first_bad_op": None,
                    "first_bad_diff": None,
                    "slots_compared": 36,
                    "slots_matched": 36,
                    "loss_diff": 0.0,
                    "logits_max_abs_diff": 1e-6,
                }
            ],
        }
        out = stitch._evaluate(
            report,
            manifest_dims={"vocab": 128, "d_model": 32, "hidden": 64, "num_layers": 2},
            expect_mismatch=True,
            max_first_loss_diff=1e-5,
            max_first_logits_diff=2e-4,
            require_check_dumps=True,
            require_all_checked_clean=True,
        )
        self.assertFalse(out["passed"])
        self.assertFalse(out["checks"]["pass_parity_gate_applied"]["passed"])
        self.assertTrue(out["checks"]["pass_parity_gate_applied"]["required"])

    def test_hybrid_recurrent_single_token_mode_allows_loss_only_step_when_replay_is_clean(self) -> None:
        report = {
            "pass_parity": True,
            "train_dims": {
                "requested": {"vocab": 256, "d_model": 64, "hidden": 128, "num_layers": 1},
                "effective": {"vocab": 128, "d_model": 32, "hidden": 64, "num_layers": 2},
                "mismatches": {"vocab": {"requested": 256, "effective": 128}},
                "source": "run_manifest",
            },
            "oracle": {"check_dump_files": ["a.bin"]},
            "parity_steps": [
                {
                    "checked": True,
                    "step": 1,
                    "oracle_error": None,
                    "first_bad_tensor": None,
                    "first_bad_op": None,
                    "first_bad_diff": None,
                    "slots_compared": 0,
                    "slots_matched": 0,
                    "loss_diff": 0.43,
                    "logits_max_abs_diff": None,
                    "replay_ok": True,
                    "replay_weight_max_abs_diff": 0.0,
                    "replay_weight_threshold": 3e-5,
                    "replay_optimizer_state_max_abs_diff": 0.0,
                    "replay_optimizer_state_threshold": 3e-5,
                    "replay_accum_snapshot_max_abs_diff": 0.0,
                    "replay_accum_snapshot_threshold": 3e-5,
                }
            ],
        }
        out = stitch._evaluate(
            report,
            manifest_dims={"vocab": 128, "d_model": 32, "hidden": 64, "num_layers": 2},
            expect_mismatch=True,
            max_first_loss_diff=1e-5,
            max_first_logits_diff=2e-4,
            require_check_dumps=True,
            require_all_checked_clean=False,
            allow_loss_only_relaxation=True,
        )
        self.assertTrue(out["passed"])
        self.assertTrue(out["checks"]["first_checked_parity_step"]["loss_only_relaxation_applied"])


if __name__ == "__main__":
    unittest.main()
