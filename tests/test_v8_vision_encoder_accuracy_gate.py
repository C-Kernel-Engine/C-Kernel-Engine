#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import argparse
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version/v8/scripts/vision_encoder_accuracy_gate_v8.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("vision_encoder_accuracy_gate_v8", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VisionEncoderAccuracyGateTests(unittest.TestCase):
    def test_relative_rmse_is_scale_aware(self) -> None:
        import numpy as np
        spec = importlib.util.spec_from_file_location("compare_qwen3vl_bf16", ROOT / "version/v8/scripts/compare_qwen3vl_bf16_vision_hidden_v8.py")
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metrics = module._metrics(np.array([1000.0, -1000.0], dtype=np.float32), np.array([1010.0, -1010.0], dtype=np.float32))
        self.assertAlmostEqual(metrics["rmse"], 10.0, places=5)
        self.assertAlmostEqual(metrics["relative_rmse"], 0.01, places=5)

    def test_ck_checkpoint_capture_uses_an_isolated_worker(self) -> None:
        import numpy as np
        spec = importlib.util.spec_from_file_location(
            "compare_qwen3vl_bf16_isolated",
            ROOT / "version/v8/scripts/compare_qwen3vl_bf16_vision_hidden_v8.py",
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(prefix="vision_ck_capture_") as tmp:
            output = Path(tmp) / "selector.f32"
            args = argparse.Namespace(
                checkpoint=Path("checkpoint"), runtime_dir=Path("runtime"),
                weights_bump=Path("weights.bump"), image=Path("image.ppm"),
                out_dir=Path(tmp), threads=20, attn_implementation="sdpa",
                architecture="qwen3vl", model_so_name="libqwen3vl_bf16_encoder_v8.so",
                ck_import_layer_input=None, ck_import_layer=None,
                ck_import_checkpoint="layer_input",
            )

            def fake_run(command, **_kwargs):
                self.assertIn("--ck-worker-selector", command)
                self.assertEqual(command[command.index("--ck-worker-selector") + 1], "ln1@0")
                self.assertEqual(command[command.index("--ck-worker-elements") + 1], "2")
                np.array([1.0, 2.0], dtype=np.float32).tofile(
                    command[command.index("--ck-worker-output") + 1]
                )
                return subprocess.CompletedProcess(command, 0)

            with mock.patch.object(module.subprocess, "run", side_effect=fake_run):
                captured = module._run_ck_selector_isolated(
                    args, "ln1@0", output, expected_elements=2
                )
            np.testing.assert_array_equal(captured, np.array([1.0, 2.0], dtype=np.float32))

    def test_matrix_checkpoint_extent_comes_from_call_abi(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "compare_qwen3vl_bf16_extent",
            ROOT / "version/v8/scripts/compare_qwen3vl_bf16_vision_hidden_v8.py",
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        operation = {
            "args": [
                {"name": "M", "expr": "3657"},
                {"name": "N", "expr": "2048"},
                {"name": "K", "expr": "4608"},
            ]
        }
        self.assertEqual(module._operation_output_elements(operation), 3657 * 2048)
        self.assertIsNone(module._operation_output_elements({"args": [{"name": "M", "expr": "rows"}]}))

    def test_elementwise_checkpoint_extent_comes_from_call_abi(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "compare_qwen3vl_bf16_element_extent",
            ROOT / "version/v8/scripts/compare_qwen3vl_bf16_vision_hidden_v8.py",
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        operation = {
            "args": [
                {"name": "data", "source": "output:out", "buffer_ref": "shared"},
                {"name": "n", "source": "dim:gelu_elems", "expr": "62958912"},
            ]
        }
        self.assertEqual(module._operation_output_elements(operation), 62958912)

    def test_attention_capture_keeps_projection_consumption_order(self) -> None:
        script = (ROOT / "version/v8/scripts/compare_qwen3vl_bf16_vision_hidden_v8.py").read_text()
        declaration = next(
            line for line in script.splitlines() if line.strip().startswith("head_major_names =")
        )
        self.assertNotIn("attn_out_head_major", declaration)

    def test_position_capture_preserves_model_storage_boundary(self) -> None:
        script = (ROOT / "version/v8/scripts/compare_qwen3vl_bf16_vision_hidden_v8.py").read_text()
        self.assertIn("pos_embeds.to(patch_sum.dtype)", script)
        self.assertNotIn("patch_sum + pos_embeds).detach()", script)

    def test_ck_checkpoint_trims_shared_buffer_only_when_call_abi_proves_extent(self) -> None:
        import numpy as np
        spec = importlib.util.spec_from_file_location(
            "compare_qwen3vl_bf16_shared_buffer",
            ROOT / "version/v8/scripts/compare_qwen3vl_bf16_vision_hidden_v8.py",
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(prefix="vision_shared_buffer_") as tmp:
            runtime_dir = Path(tmp)
            (runtime_dir / "config.json").write_text(json.dumps({
                "image_height": 2,
                "image_width": 2,
            }))
            (runtime_dir / "call.json").write_text(json.dumps({
                "operations": [{
                    "idx": 7,
                    "args": [
                        {"name": "C", "source": "output:c", "buffer_ref": "shared"},
                        {"name": "M", "expr": "2"},
                        {"name": "N", "expr": "2"},
                    ],
                    "semantic_checkpoints": [{
                        "tensor": "vision_projector_out",
                        "layer": -1,
                    }],
                }],
            }))
            args = argparse.Namespace(
                runtime_dir=runtime_dir,
                checkpoint=runtime_dir,
                image=runtime_dir / "image.ppm",
                architecture="cohere_compass",
                weights_bump=runtime_dir / "weights.bump",
                model_so_name="model.so",
                ck_import_layer_input=None,
            )
            numeric = mock.Mock()
            numeric._run_generated_encoder.return_value = np.arange(8, dtype=np.float32)
            with mock.patch.object(module, "_load_processor_planar", return_value=[0.0] * 12):
                captured = module._run_ck_selector(
                    args,
                    "vision_projector_out",
                    numeric,
                    expected_elements=4,
                )
            np.testing.assert_array_equal(captured, np.arange(4, dtype=np.float32))
            self.assertNotIn("CK_STOP_OP", __import__("os").environ)

    def test_torch_checkpoint_capture_releases_model_before_ck_workers(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "compare_qwen3vl_bf16_torch_isolated",
            ROOT / "version/v8/scripts/compare_qwen3vl_bf16_vision_hidden_v8.py",
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(prefix="vision_torch_capture_") as tmp:
            out_dir = Path(tmp)
            args = argparse.Namespace(
                checkpoint=Path("checkpoint"), runtime_dir=Path("runtime"),
                weights_bump=Path("weights.bump"), image=Path("image.ppm"),
                out_dir=out_dir, threads=20, attn_implementation="sdpa",
                torch_prefix=None, architecture="qwen3vl",
                model_so_name="libqwen3vl_bf16_encoder_v8.so",
            )

            def fake_run(command, **_kwargs):
                self.assertIn("--skip-ck", command)
                (out_dir / "report.json").write_text(json.dumps({
                    "torch": {"tensors": {"ln1@0": {"path": "ln1.f32", "shape": [1, 2]}}}
                }))
                return subprocess.CompletedProcess(command, 0)

            with mock.patch.object(module.subprocess, "run", side_effect=fake_run):
                captured = module._run_torch_captures_isolated(args, ["ln1@0"])
            self.assertIn("ln1@0", captured["tensors"])

    def test_large_public_fixture_is_deterministic_ppm(self) -> None:
        gate = _load_gate()
        with tempfile.TemporaryDirectory(prefix="vision_encoder_fixture_") as tmp:
            first = Path(tmp) / "first.ppm"
            second = Path(tmp) / "second.ppm"
            gate._write_large_form(first)
            gate._write_large_form(second)
            data = first.read_bytes()
            self.assertTrue(data.startswith(b"P6\n1152 896\n255\n"))
            self.assertEqual(data, second.read_bytes())
            self.assertEqual(len(data), len(b"P6\n1152 896\n255\n") + 1152 * 896 * 3)

    def test_missing_artifact_is_skip_or_required_failure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vision_encoder_gate_") as tmp:
            out = Path(tmp) / "skip"
            base = [
                sys.executable,
                str(SCRIPT),
                "--mode",
                "q4",
                "--allow-non-avx512",
                "--output-dir",
                str(out),
            ]
            skipped = subprocess.run(base, cwd=ROOT, check=False, capture_output=True, text=True)
            self.assertEqual(skipped.returncode, 0, skipped.stdout + skipped.stderr)
            self.assertEqual(json.loads((out / "summary.json").read_text())["status"], "skip")

            required_out = Path(tmp) / "required"
            required = subprocess.run(
                [*base[:-1], str(required_out), "--require-artifacts"],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(required.returncode, 0)
            report = json.loads((required_out / "summary.json").read_text())
            self.assertEqual(report["status"], "fail")
            self.assertTrue(report["failures"])


if __name__ == "__main__":
    unittest.main()
