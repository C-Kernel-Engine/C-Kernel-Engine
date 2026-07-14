#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v8" / "scripts"
PROFILE = ROOT / "version" / "v8" / "parity_profiles" / "qwen3vl_llamacpp_q8_v1.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


import sys
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

frontend = load_module("xray_vision_parity_v8_test", SCRIPTS / "xray_vision_parity_v8.py")
llama = load_module("xray_qwen3vl_llamacpp_v8_test", SCRIPTS / "xray_qwen3vl_llamacpp_v8.py")
xray = load_module("xray_numerical_parity_v8_test_interface", SCRIPTS / "xray_numerical_parity_v8.py")


class XRayVisionInterfaceTests(unittest.TestCase):
    def test_frontend_dispatches_llamacpp_without_owning_backend_arguments(self) -> None:
        with mock.patch.object(frontend.llamacpp_adapter, "main", return_value=7) as adapter:
            result = frontend.dispatch(["--backend", "llamacpp", "--gguf", "model.gguf"])
        self.assertEqual(result, 7)
        adapter.assert_called_once_with(["--gguf", "model.gguf"])

    def test_frontend_dispatches_pytorch_without_owning_backend_arguments(self) -> None:
        with mock.patch.object(frontend.pytorch_adapter, "main", return_value=0) as adapter:
            result = frontend.dispatch(["--backend", "pytorch", "--checkpoint", "model"])
        self.assertEqual(result, 0)
        adapter.assert_called_once_with(["--checkpoint", "model"])

    def test_llama_profile_is_schema_valid(self) -> None:
        profile = xray.load_json(PROFILE)
        xray.validate(profile, xray.PROFILE_SCHEMA, "llama profile")
        self.assertEqual(profile["backend"], "llamacpp")

    def test_legacy_results_are_reordered_by_semantic_circuit_position(self) -> None:
        profile = xray.load_json(PROFILE)
        report = {
            "results": [
                {"layer": 0, "op": "Kcur_rope", "status": "FAIL", "max_abs_diff": 0.2},
                {"layer": 0, "op": "q_proj", "status": "FAIL", "max_abs_diff": 0.1},
                {"layer": 0, "op": "ln1", "status": "PASS", "max_abs_diff": 0.0},
                {"layer": -1, "op": "patch_bias", "status": "PASS", "max_abs_diff": 0.0},
                {"layer": -1, "op": "inp_pos_emb", "status": "PASS", "max_abs_diff": 0.0},
            ]
        }
        result = llama.normalize_capture_report(report, profile, layer=0)
        self.assertEqual(result["last_passing_checkpoint"], "vision.layer.0.norm1.output")
        self.assertEqual(result["first_divergence"]["checkpoint_id"], "vision.layer.0.q.pre_rope")
        self.assertEqual(
            result["first_divergence"]["classification"],
            "KERNEL_IMPLEMENTATION_DIVERGENCE",
        )

    def test_nonzero_passing_checkpoint_prevents_false_downstream_blame(self) -> None:
        profile = xray.load_json(PROFILE)
        report = {
            "results": [
                {"layer": -1, "op": "patch_bias", "status": "PASS", "max_abs_diff": 0.0},
                {"layer": -1, "op": "inp_pos_emb", "status": "PASS", "max_abs_diff": 0.0},
                {"layer": 0, "op": "ln1", "status": "PASS", "max_abs_diff": 0.0},
                {"layer": 0, "op": "q_proj", "status": "PASS", "max_abs_diff": 0.0},
                {"layer": 0, "op": "k_proj", "status": "PASS", "max_abs_diff": 0.0},
                {"layer": 0, "op": "v_proj", "status": "PASS", "max_abs_diff": 0.0},
                {"layer": 0, "op": "Qcur_rope", "status": "PASS", "max_abs_diff": 0.0},
                {"layer": 0, "op": "Kcur_rope", "status": "PASS", "max_abs_diff": 0.0},
                {"layer": 0, "op": "kqv_out", "status": "PASS", "max_abs_diff": 1e-6},
                {"layer": 0, "op": "attn_output", "status": "FAIL", "max_abs_diff": 5e-4},
            ]
        }
        result = llama.normalize_capture_report(report, profile, layer=0, execution_mode="production")
        self.assertEqual(result["execution_mode"], "production")
        self.assertEqual(
            result["first_non_exact_checkpoint"]["checkpoint_id"],
            "vision.layer.0.attention.output",
        )
        self.assertEqual(
            result["first_divergence"]["classification"],
            "DOWNSTREAM_OR_PROPAGATED_DIVERGENCE",
        )
        self.assertEqual(result["first_divergence"]["fix_owner"], "exact_input_control")

    def test_capture_mode_controls_strict_parity_flag(self) -> None:
        profile = xray.load_json(PROFILE)
        base = {
            "gguf": Path("model.gguf"),
            "output_dir": Path("out"),
            "threads": 1,
            "ck_threads": 20,
            "layer": 0,
            "image": None,
            "image_mode": "gradient",
            "image_min_tokens": None,
            "image_max_tokens": 1024,
        }
        strict = llama._capture_args(
            argparse.Namespace(**base, execution_mode="strict"), profile, Path("report.json")
        )
        production = llama._capture_args(
            argparse.Namespace(**base, execution_mode="production"), profile, Path("report.json")
        )
        self.assertIn("--strict-parity", strict)
        self.assertNotIn("--strict-parity", production)
        self.assertEqual(strict[strict.index("--llama-flash-attn") + 1], "disabled")
        self.assertEqual(production[production.index("--llama-flash-attn") + 1], "enabled")
        self.assertEqual(strict[strict.index("--ck-dump-layer") + 1], "0")
        self.assertEqual(strict[strict.index("--threads") + 1], "1")
        self.assertEqual(strict[strict.index("--ck-threads") + 1], "20")

    def test_exact_llama_oracle_rejects_multiple_threads(self) -> None:
        args = argparse.Namespace(threads=20, allow_nondeterministic_oracle=False)
        with self.assertRaisesRegex(RuntimeError, "requires --threads 1"):
            llama._validate_oracle_execution(args)

    def test_nondeterministic_llama_oracle_requires_explicit_opt_in(self) -> None:
        args = argparse.Namespace(threads=20, allow_nondeterministic_oracle=True)
        result = llama._validate_oracle_execution(args)
        self.assertFalse(result["deterministic"])
        self.assertTrue(result["nondeterministic_opt_in"])

    def test_ck_capture_is_bounded_to_global_and_requested_layer(self) -> None:
        observed: dict[str, str | None] = {}

        def capture_environment(**_kwargs) -> None:
            observed["layer_filter"] = os.environ.get("CK_PARITY_LAYER_FILTER")

        with tempfile.TemporaryDirectory(prefix="cke_xray_layer_filter_") as td:
            with mock.patch.object(
                llama.capture_adapter.npv8,
                "_run_generated_encoder",
                side_effect=capture_environment,
            ):
                llama.capture_adapter._run_generated_encoder_with_dump(
                    model_so=Path("model.so"),
                    weights_bump=Path("weights.bump"),
                    manifest_map=Path("weights.map"),
                    layout_path=Path("layout.json"),
                    planar_image=[],
                    dump_dir=Path(td),
                    strict_parity=False,
                    strict_dump_layer=None,
                    dump_layer=26,
                    ck_stop_op=None,
                    dump_names="ln1,layer_out",
                )

        self.assertEqual(observed["layer_filter"], "-1,26")
        self.assertNotIn("CK_PARITY_LAYER_FILTER", os.environ)

    def test_capture_adapter_is_documented_as_internal_to_xray(self) -> None:
        source = (SCRIPTS / "activation_parity_qwen3vl_mmproj_v8.py").read_text(encoding="utf-8")
        self.assertIn("must invoke ``xray_vision_parity_v8.py --backend", source)
        self.assertIn("Do not add model-family branches here", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
