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

            # Branched inputs have accumulated gradients, not the selected VJP.
            # No accumulated dX is provided: diagnosis must replay into fresh buffers.
            from tests.clef_mixed_oracle import diagnose

            xx = np.array([[0.25, 0.5]], dtype="f4")
            ww = np.array([[0.5, 0.25]], dtype="f4")
            reference_x = xx + 0.25
            op = dict(
                op="branch",
                op_id=1,
                kernel_id="projection_bf16_train_forward",
                weights={"weight": {"name": "w"}},
                dataflow={
                    "inputs": {"input": {"tensor": "act.x"}},
                    "outputs": {"output": {"tensor": "act.y"}},
                },
            )
            captures = {
                "w": [dict(input=reference_x, weight=ww, output=reference_x @ ww.T)]
            }
            layout = {
                "regions": [
                    dict(name="grads", bytes=0),
                    dict(name="grad_activations", offset=0),
                ],
                "tensors": [dict(id="grad.act.act.y", offset=0, numel=1, shape=[1, 1])],
            }
            result, packet = diagnose(
                {"ops": [op]},
                captures,
                {"act.x": xx, "act.y": xx @ ww.T},
                {"w": ww},
                {"w": xx},
                layout,
                np.ones(1, dtype="f4"),
                lib,
            )
            self.assertEqual(result[0]["same_input_d_input_error"], 0.0)
            self.assertIn("isolated provider", result[0]["d_input_evidence"])
            np.testing.assert_array_equal(packet["d_input"], ww)


class MixedHeadContracts(unittest.TestCase):
    def test_memory_exception_requires_mixed_experiment(self):
        import sys
        from unittest.mock import patch
        from tests import run_clef_training_oracle as runner

        with patch.object(
            sys,
            "argv",
            ["oracle", "--fp32-memory-projection", "--report", "unused.json"],
        ), patch.object(runner.FullHeadTraining, "certify") as execute:
            with self.assertRaises(SystemExit) as error:
                runner.main()
            self.assertEqual(error.exception.code, 2)
            execute.assert_not_called()

    def test_fp32_memory_exception_changes_only_declared_projection(self):
        ir, registry = full.FullHeadTrainingContracts.graph(
            self, mixed=True, fp32_memory_projection=True
        )
        baseline, _ = full.FullHeadTrainingContracts.graph(self, mixed=True)
        original = {op["op"]: op for op in baseline["forward"]}
        memory = next(op for op in ir["forward"] if op["op"] == "memory_projection")
        self.assertEqual(memory["kernel_id"], "linear_rows_checked_no_bias_f32")
        for op in ir["forward"]:
            if op["op"] != "memory_projection":
                self.assertEqual(op["kernel_id"], original[op["op"]]["kernel_id"])
        self.assertNotIn(
            "aux.projection_%s.saved_input" % memory["op_id"], ir["tensors"]
        )
        _, report = full.FullHeadTrainingContracts.emit(self, ir, registry)
        self.assertEqual(report["skipped_ops"], [])
        self.assertEqual(len(report["parameter_gradient_order"]), 162)
        from tests.clef_mixed_oracle import projection_policy

        with self.assertRaisesRegex(ValueError, "one declared FP32 memory projection"):
            projection_policy({"ops": []}, fp32_memory_projection=True)

    def test_oracle_preserves_fp32_exception_forward_and_backward(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest("independent PyTorch unavailable: " + str(exc))
        from tests.clef_mixed_oracle import install

        head = torch.nn.Module()
        head.memory_projection = torch.nn.Linear(2, 2, bias=False)
        head.other = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            for layer in (head.memory_projection, head.other):
                layer.weight.copy_(
                    torch.tensor([[0.123456, 0.654321], [-0.321234, 0.234567]])
                )
        captures = install(
            self, torch, head, fp32_weights=("memory_projection.weight",)
        )
        x = torch.tensor([[0.345678, -0.765432]], requires_grad=True)
        y = head.memory_projection(x)
        expected = x @ head.memory_projection.weight.T
        torch.testing.assert_close(y, expected, rtol=0, atol=0)
        y.sum().backward()
        torch.testing.assert_close(
            x.grad, head.memory_projection.weight.sum(0)[None], rtol=0, atol=0
        )
        torch.testing.assert_close(
            head.memory_projection.weight.grad, x.detach().expand(2, 2), rtol=0, atol=0
        )
        mixed = head.other(x)
        reference = x.bfloat16().float() @ head.other.weight.bfloat16().float().T
        torch.testing.assert_close(mixed, reference, rtol=0, atol=0)
        self.assertNotIn("memory_projection.weight", captures)
        self.assertIn("other.weight", captures)
        self.assertFalse(torch.equal(y, mixed))

    def test_same_state_restores_packed_qkv_scalar_and_moments(self):
        try:
            import numpy as np
            import torch
        except ImportError as exc:
            self.skipTest("independent PyTorch unavailable: " + str(exc))
        from tests.clef_same_state_oracle import synchronize

        head = torch.nn.Module()
        head.attn = torch.nn.MultiheadAttention(2, 1)
        head.scale = torch.nn.Parameter(torch.tensor(0.0))
        weights = {}
        expected = {}
        for i, (name, p) in enumerate(head.named_parameters()):
            value = np.arange(p.numel(), dtype="f4").reshape(tuple(p.shape)) + i * 10
            expected[name] = value
            if name.endswith(("in_proj_weight", "in_proj_bias")):
                suffix = "in_proj_weight" if name.endswith("weight") else "in_proj_bias"
                leaf = "weight" if suffix.endswith("weight") else "bias"
                for j, axis in enumerate(("q", "k", "v")):
                    weights[name[: -len(suffix)] + axis + "_proj." + leaf] = value[
                        j * 2 : (j + 1) * 2
                    ].copy()
            else:
                weights[name] = value.reshape(1) if value.ndim == 0 else value.copy()
        order = sorted(weights)
        moments = np.concatenate(
            [weights[n].ravel() + offset for offset in (100.0, 200.0) for n in order]
        )
        original = moments.copy()
        opt = torch.optim.AdamW(head.parameters())
        report = synchronize(torch, head, opt, weights, moments, order, 7)
        self.assertEqual(report["optimizer_step"], 7)
        for name, p in head.named_parameters():
            np.testing.assert_array_equal(p.detach().numpy(), expected[name])
            for key, offset in (("exp_avg", 100.0), ("exp_avg_sq", 200.0)):
                np.testing.assert_array_equal(
                    opt.state[p][key].numpy(), expected[name] + offset
                )
            self.assertEqual(opt.state[p]["step"].item(), 7)
        np.testing.assert_array_equal(moments, original)
        with self.assertRaises(ValueError):
            synchronize(torch, head, opt, weights, moments[:-1], order, 7)

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
