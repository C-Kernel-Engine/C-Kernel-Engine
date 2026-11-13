"""Biased projection and generated-head contracts, not mixed trajectory certification."""

import unittest
from tests import test_v8_clef_full_training as full


class BiasedProjectionKernel(unittest.TestCase):
    def test_bias_gradients_and_rejection(self):
        import ctypes as C, tempfile, subprocess
        from pathlib import Path

        try:
            import numpy as np
            import torch
        except ImportError as exc:
            self.skipTest("independent PyTorch unavailable: " + str(exc))
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            so = Path(tmp) / "bias.so"
            subprocess.run(
                [
                    "cc",
                    "-O3",
                    "-shared",
                    "-fPIC",
                    "-I" + str(root / "include"),
                    str(root / "src/kernels/projection_bf16_train.c"),
                    "-lm",
                    "-o",
                    str(so),
                ],
                check=True,
            )
            lib = C.CDLL(str(so))
            for phase in ("forward", "backward"):
                fn = getattr(lib, "ck_projection_bf16_train_bias_" + phase)
                fn.argtypes = [
                    t for _ in range(7) for t in (C.c_void_p, C.c_size_t)
                ] + [C.c_size_t] * 3
                fn.restype = C.c_int

            def run(phase, arrays, override=None):
                args = []
                for a in arrays:
                    args.extend([a.ctypes.data, a.size])
                args.extend([3, 5, 7])
                for i, v in (override or {}).items():
                    args[i] = v
                return getattr(lib, "ck_projection_bf16_train_bias_" + phase)(*args)

            rng = np.random.default_rng(851)
            x = rng.normal(0, 0.2, (3, 5)).astype("f")
            w = rng.normal(0, 0.2, (7, 5)).astype("f")
            b = rng.normal(0, 0.2, 7).astype("f")
            y = np.full(24, 71.0, "f")
            sx = np.zeros(15, "uint16")
            sw = np.zeros(35, "uint16")
            scratch = np.zeros(21, "f")
            arrays = [x, w, b, y, sx, sw, scratch]
            for override in ({4: None}, {5: 6}, {4: y.ctypes.data + 4}, {13: 20}):
                self.assertLess(run("forward", arrays, override), 0)
                np.testing.assert_array_equal(y, 71.0)
                np.testing.assert_array_equal(sx, 0)
                np.testing.assert_array_equal(sw, 0)
            original = b.copy()
            b[0] = np.nan
            self.assertLess(run("forward", arrays), 0)
            np.testing.assert_array_equal(y, 71.0)
            b[:] = original
            original_x, original_w = x.copy(), w.copy()
            x[:] = 1.0
            w[:] = 1e38
            b[:] = 3e38
            self.assertLess(run("forward", arrays), 0)
            np.testing.assert_array_equal(y, 71.0)
            np.testing.assert_array_equal(sx, 0)
            np.testing.assert_array_equal(sw, 0)
            x[:], w[:], b[:] = original_x, original_w, original
            self.assertEqual(run("forward", arrays), 0)
            xx = torch.tensor(x).bfloat16().float().requires_grad_()
            ww = torch.tensor(w).bfloat16().float().requires_grad_()
            bb = torch.tensor(b, requires_grad=True)
            expected = xx @ ww.T + bb
            np.testing.assert_allclose(
                y[:21].reshape(3, 7), expected.detach().numpy(), atol=2e-7, rtol=2e-6
            )
            dy = rng.normal(0, 0.3, (3, 7)).astype("f")
            expected.backward(torch.tensor(dy))
            dx = np.full(18, 71.0, "f")
            dw = np.full(38, 71.0, "f")
            db = np.full(10, 71.0, "f")
            stage = np.zeros(57, "f")
            back = [dy, sx, sw, dx, dw, db, stage]
            for override in ({10: None}, {11: 6}, {10: dw.ctypes.data + 4}, {13: 56}):
                self.assertLess(run("backward", back, override), 0)
                for a in (dx, dw, db):
                    np.testing.assert_array_equal(a, 71.0)
            original_dy = dy.copy()
            dy[:] = 3e38
            self.assertLess(run("backward", back), 0)
            for a in (dx, dw, db):
                np.testing.assert_array_equal(a, 71.0)
            dy[:] = original_dy
            self.assertEqual(run("backward", back), 0)
            for actual, oracle in (
                (dx[:15].reshape(3, 5), xx.grad),
                (dw[:35].reshape(7, 5), ww.grad),
                (db[:7], bb.grad),
            ):
                np.testing.assert_allclose(actual, oracle.numpy(), atol=2e-7, rtol=2e-6)
            for a, n in ((y, 21), (dx, 15), (dw, 35), (db, 7)):
                np.testing.assert_array_equal(a[n:], 71.0)


class MixedHeadContracts(unittest.TestCase):
    def test_complete_topology_generates_all_parameter_gradients(self):
        ir, registry = full.FullHeadTrainingContracts.graph(self, mixed=True)
        source, report = full.FullHeadTrainingContracts.emit(self, ir, registry)
        self.assertEqual(report["skipped_ops"], [])
        self.assertEqual(len(report["parameter_gradient_order"]), 162)
        self.assertIn("ck_projection_bf16_train_bias_forward(", source)
        self.assertIn("ck_projection_bf16_train_bias_backward(", source)
        self.assertIn("ck_projection_bf16_train_backward(", source)
        self.assertNotIn("linear_rows_checked_f32(", source)
        for name, tensor in ir["tensors"].items():
            if name.startswith(("weight.", "grad.weight.")):
                self.assertEqual(tensor["dtype"], "fp32", name)

    def test_oracle_failure_and_missing_assets_are_not_pass(self):
        import json, tempfile, sys
        from pathlib import Path
        from unittest.mock import patch
        from tests import run_clef_training_oracle as runner

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            for status, code in (("FAIL", 1), ("NOT_TESTED", 2)):
                with self.subTest(status=status):
                    result = {"numerical_parity": status}
                    kwargs = (
                        {"side_effect": unittest.SkipTest("missing reference")}
                        if status == "NOT_TESTED"
                        else {"return_value": result}
                    )
                    with patch.object(
                        sys, "argv", ["oracle", "--mixed", "--report", str(path)]
                    ), patch.object(runner.FullHeadTraining, "certify", **kwargs):
                        self.assertEqual(runner.main(), code)
                    self.assertEqual(
                        json.loads(path.read_text())["numerical_parity"], status
                    )


if __name__ == "__main__":
    unittest.main()
