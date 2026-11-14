"""Typed FP32 inputs are request data, never parameter snapshots."""

import ctypes as C
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
from tests import test_v8_selective_training as backend

ROOT = Path(__file__).resolve().parents[1]


def graph():
    ir = backend.fixture()
    ir["ops"].pop(0)
    ir["tensors"].pop("weight.features")
    meta = ir["tensors"].pop("act.features")
    meta.update(kind="input", requires_grad=False, persistent=False)
    ir["tensors"]["input.features"] = meta

    def replace(value):
        if isinstance(value, dict):
            if value.get("tensor") == "act.features":
                value.update(meta, tensor="input.features")
            else:
                for item in value.values():
                    replace(item)
        elif isinstance(value, list):
            for item in value:
                replace(item)

    replace(ir["ops"])
    ir["config"].update(
        train_batch_policy="fixed", training_float_inputs=["input.features"]
    )
    return ir


class FloatInputContractTests(unittest.TestCase):
    def test_declarations_reject_weights_duplicates_and_unbound_inputs(self):
        for defect in (
            "weight",
            "duplicate",
            "missing",
            "trainable",
            "persistent",
            "dynamic",
        ):
            ir = graph()
            if defect == "weight":
                ir["config"]["training_float_inputs"] = ["weight.up"]
            if defect == "duplicate":
                ir["config"]["training_float_inputs"] *= 2
            if defect == "missing":
                ir["config"]["training_float_inputs"] = []
            if defect == "trainable":
                ir["tensors"]["input.features"]["requires_grad"] = True
            if defect == "persistent":
                ir["tensors"]["input.features"]["persistent"] = True
            if defect == "dynamic":
                ir["config"].pop("train_batch_policy")
            with self.subTest(defect=defect), self.assertRaises(
                (ValueError, RuntimeError)
            ):
                lowered = backend.lower(ir)
                layout = backend.planner.build_layout(
                    lowered, {"config": lowered["config"]}, 64, strict=True
                )
                backend.codegen.generate_c(
                    lowered,
                    backend.REGISTRY,
                    manifest={"config": lowered["config"]},
                    layout=layout,
                )

    def test_native_rejection_reuse_and_independent_gradient(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest("independent PyTorch unavailable: " + str(exc))
        engine = ROOT / "build/libckernel_engine.so"
        if not engine.exists():
            self.skipTest("native engine unavailable")
        import sys

        sys.path.insert(0, str(ROOT / "version/v8/scripts"))
        import run_training_workflow_v8 as wf

        ir = backend.lower(graph())
        layout = backend.planner.build_layout(
            ir, {"config": ir["config"]}, 64, strict=True
        )
        source, summary = backend.codegen.generate_c(
            ir, backend.REGISTRY, manifest={"config": ir["config"]}, layout=layout
        )
        self.assertNotIn("features", summary["init_weight_order"])
        self.assertEqual(summary["float_input_count"], 16)
        rng = np.random.default_rng(731)
        values = {
            n: rng.normal(0, 0.4, (4, 4)).astype("f4")
            for n in summary["init_weight_order"]
        }
        a = rng.normal(0, 0.4, (4, 4)).astype("f4")
        b = rng.normal(0, 0.7, (4, 4)).astype("f4")
        packed = np.concatenate(
            [values[n].ravel() for n in summary["init_weight_order"]]
        )
        sizes = np.array(summary["init_weight_numel"], dtype="i4")
        ids = np.arange(4, dtype="i4")
        fp = C.POINTER(C.c_float)
        ip = C.POINTER(C.c_int32)
        ptr = lambda x: x.ctypes.data_as(fp)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            # Test-only address probes exercise arena overlap without adding a production export.
            (folder / "train.c").write_text(
                source
                + "\nfloat *test_arena(void) { return g_memory; }\nfloat *test_features(void) { return input_features; }\n"
            )
            subprocess.run(
                [
                    "cc",
                    "-O2",
                    "-shared",
                    "-fPIC",
                    "-DCK_NUM_TOKENS=4",
                    "-I" + str(ROOT / "include"),
                    str(folder / "train.c"),
                    str(engine),
                    "-Wl,-rpath," + str(engine.parent),
                    "-lm",
                    "-o",
                    str(folder / "train.so"),
                ],
                check=True,
            )
            lib = C.CDLL(str(folder / "train.so"))
            wf._configure(lib)
            lib.ck_train_init.argtypes = [fp, ip, C.c_int]
            lib.ck_train_set_float_inputs_ex.argtypes = [C.c_void_p, C.c_size_t]
            lib.ck_train_set_batch_ex.argtypes = [ip, ip, C.c_int]
            lib.ck_train_eval_step.argtypes = [fp]
            lib.ck_train_step_ex.argtypes = [ip, ip, C.c_int, fp, C.c_float]
            lib.test_arena.restype = C.c_void_p
            lib.test_features.restype = C.c_void_p

            def state():
                return (
                    wf._weight_export(lib).tobytes(),
                    wf._array_export(lib, "optimizer_state").tobytes(),
                    lib.ck_train_get_opt_step(),
                )

            def supply(x):
                self.assertEqual(
                    lib.ck_train_set_float_inputs_ex(x.ctypes.data, x.size), x.size
                )

            def evaluate(x):
                supply(x)
                self.assertGreater(
                    lib.ck_train_set_batch_ex(
                        ids.ctypes.data_as(ip), ids.ctypes.data_as(ip), 4
                    ),
                    0,
                )
                loss = C.c_float(-91)
                self.assertGreater(lib.ck_train_eval_step(C.byref(loss)), 0)
                return loss.value

            try:
                self.assertEqual(
                    lib.ck_train_init(
                        ptr(packed), sizes.ctypes.data_as(ip), len(sizes)
                    ),
                    len(sizes),
                )
                lib.ck_set_strict_parity(1)
                loss = C.c_float(-91)
                self.assertLess(lib.ck_train_eval_step(C.byref(loss)), 0)
                self.assertEqual(loss.value, -91)
                initial = state()
                first = evaluate(a)
                other = evaluate(b)
                again = evaluate(a)
                self.assertEqual(first, again)
                self.assertNotEqual(first, other)
                self.assertEqual(initial, state())
                bad = a.copy()
                bad.flat[-1] = np.nan
                old_features = C.string_at(lib.test_features(), a.nbytes)
                misaligned = C.create_string_buffer(a.nbytes + 4)
                cases = [
                    (None, 16),
                    (a.ctypes.data, 15),
                    (a.ctypes.data, 17),
                    (bad.ctypes.data, 16),
                    (C.addressof(misaligned) + 1, 16),
                    (C.c_void_p(-4).value, 16),
                    (lib.test_arena(), 16),
                    (lib.test_features() + 4, 16),
                ]
                for address, count in cases:
                    with self.subTest(address=address, count=count):
                        self.assertLess(
                            lib.ck_train_set_float_inputs_ex(address, count), 0
                        )
                        self.assertEqual(state(), initial)
                        self.assertEqual(
                            C.string_at(lib.test_features(), a.nbytes), old_features
                        )
                        loss = C.c_float(-91)
                        self.assertLess(lib.ck_train_eval_step(C.byref(loss)), 0)
                        self.assertEqual(loss.value, -91)
                        self.assertEqual(evaluate(a), first)
                supply(a)
                self.assertLess(lib.ck_train_backward_step(), 0)
                self.assertEqual(state(), initial)
                params = {
                    n: torch.tensor(v, requires_grad=n != "bridge")
                    for n, v in values.items()
                }
                opt = torch.optim.AdamW(
                    [p for p in params.values() if p.requires_grad],
                    lr=0.03,
                    weight_decay=0.01,
                    foreach=False,
                )
                for _ in range(4):
                    opt.zero_grad()
                    z = (
                        torch.from_numpy(a)
                        @ params["up"].T
                        @ params["bridge"].T
                        @ params["score"].T
                    )
                    expected = torch.nn.functional.cross_entropy(z, torch.arange(4))
                    expected.backward()
                    supply(a)
                    loss = C.c_float()
                    self.assertGreater(
                        lib.ck_train_step_ex(
                            ids.ctypes.data_as(ip),
                            ids.ctypes.data_as(ip),
                            4,
                            C.byref(loss),
                            0.03,
                        ),
                        0,
                    )
                    self.assertAlmostEqual(loss.value, expected.item(), delta=3e-5)
                    grads = np.concatenate(
                        [
                            params[n].grad.numpy().ravel()
                            for n in summary["parameter_gradient_order"]
                        ]
                    )
                    np.testing.assert_allclose(
                        wf._array_export(lib, "parameter_gradient"),
                        grads,
                        atol=3e-6,
                        rtol=3e-4,
                    )
                    opt.step()
                    expected_weights = np.concatenate(
                        [
                            params[n].detach().numpy().ravel()
                            for n in summary["init_weight_order"]
                        ]
                    )
                    np.testing.assert_allclose(
                        wf._weight_export(lib), expected_weights, atol=3e-5, rtol=3e-4
                    )
                lib.ck_train_free()
                self.assertEqual(
                    lib.ck_train_init(
                        ptr(packed), sizes.ctypes.data_as(ip), len(sizes)
                    ),
                    len(sizes),
                )
                loss = C.c_float(-91)
                self.assertLess(lib.ck_train_eval_step(C.byref(loss)), 0)
                self.assertEqual(evaluate(a), first)
            finally:
                lib.ck_train_free()
