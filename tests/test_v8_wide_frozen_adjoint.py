"""A wide frozen projection uses the declared FP64 input-only VJP."""
import ctypes
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

from tests.test_v8_decoder_lora_training import build, load


ROOT = Path(__file__).resolve().parents[1]


def native_tail_probe(threads):
    import numpy as np

    f32 = ctypes.POINTER(ctypes.c_float)
    lib = ctypes.CDLL(str(ROOT / "build/libckernel_engine.so"))
    lib.ck_threadpool_global.restype = ctypes.c_void_p
    lib.ck_threadpool_n_threads.argtypes = [ctypes.c_void_p]
    lib.ck_threadpool_n_threads.restype = ctypes.c_int
    assert lib.ck_threadpool_n_threads(lib.ck_threadpool_global()) == threads
    fn = lib.ck_gemm_backward_input_accum64_f32
    fn.argtypes = [f32] * 3 + [ctypes.c_int] * 3
    fn.restype = ctypes.c_int
    rng = np.random.default_rng(321)
    tokens, width, reduction = 7, 135, 16389
    weight = rng.normal(0, 0.1, (reduction, width)).astype("f4")
    weight[0] = 1e7
    weight[-1] = -1e7
    grad = rng.uniform(.9, 1.1, (tokens, reduction)).astype("f4")
    grad[:, (0, -1)] = 1
    result = np.full(tokens * width + 1, 4482, dtype="f4")
    pointers = [array.ctypes.data_as(f32) for array in (grad, weight, result)]
    assert fn(*pointers, 0, width, reduction) != 0
    np.testing.assert_array_equal(result, 4482)
    assert fn(*pointers, tokens, width, reduction) == 0
    reference = grad.astype("f8") @ weight.astype("f8")
    np.testing.assert_allclose(result[:-1].reshape(tokens, width), reference, rtol=1e-6, atol=1e-6)
    assert result[-1] == 4482
    sys.stdout.buffer.write(result.tobytes())


class WideFrozenAdjointTests(unittest.TestCase):
    def test_native_reduction_handles_token_and_feature_tails(self):
        if not (ROOT / "build/libckernel_engine.so").is_file():
            self.skipTest("native engine unavailable")
        probe = (
            "from tests.test_v8_wide_frozen_adjoint import native_tail_probe; "
            "import sys; native_tail_probe(int(sys.argv[1]))"
        )
        args = [sys.executable, "-c", probe]
        workers = min(len(os.sched_getaffinity(0)), 4)
        outputs = [
            subprocess.check_output([*args, str(threads)], cwd=ROOT,
                                    env={**os.environ, "CK_NUM_THREADS": str(threads)})
            for threads in (1, workers)
        ]
        self.assertEqual(*outputs)

    def test_declared_selection_removes_only_frozen_weight_destination(self):
        from tests.test_v8_selective_training import codegen, planner
        import lower_ir2_backward_v7 as backward

        rules = load("version/v8/scripts/grad_rules_v8.json")
        bindings = load("version/v7/kernel_maps/kernel_bindings.json")
        with tempfile.TemporaryDirectory() as tmp:
            ir, default, registry, _, _, _, _ = build(Path(tmp), 3, vocab=16389)
        provider = "gemm_backward_input_accum64_f32"
        weight = "weight.output.weight"
        ir["config"].update(train_batch_policy="fixed", frozen_linear_dx="fp64_wide_v1")
        def lower():
            return backward.synthesize_ir2_backward(
                ir, registry, bindings, rules, strict=True, allow_partial=False, checkpoint_policy="none")
        ir["config"]["frozen_linear_dx"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unsupported frozen linear"):
            lower()
        ir["config"]["frozen_linear_dx"] = "fp64_wide_v1"
        del ir["config"]["train_batch_policy"]
        with self.assertRaisesRegex(ValueError, "fixed training batch"):
            lower()
        ir["config"]["train_batch_policy"] = "fixed"
        result = lower()
        selected = [op for op in result["backward"] if op["kernel_id"] == provider]
        self.assertEqual(len(selected), 1)
        op = selected[0]
        self.assertEqual(op["dataflow"]["inputs"]["weight"]["tensor"], weight)
        self.assertEqual(set(op["dataflow"]["outputs"]), {"d_input"})
        parent = op["forward_ref"]
        removed = [t for t in default["tensors"] if t.startswith(f"tmp.grad.weight.op{parent}.")]
        self.assertTrue(removed)
        self.assertTrue(all(t not in result["tensors"] for t in removed))
        layout = planner.build_layout(result, {"config": result["config"]}, 64, strict=True)
        old_layout = planner.build_layout(default, {"config": default["config"]}, 64, strict=True)
        self.assertGreaterEqual(old_layout["total_bytes"] - layout["total_bytes"], 16389 * 32 * 4)
        source, summary = codegen.generate_c(result, registry, layout=layout)
        self.assertFalse(summary["skipped_ops"])
        self.assertIn("ck_gemm_backward_input_accum64_f32(", source)
        ir["tensors"][weight]["requires_grad"] = True
        for forward in ir["ops"]:
            for ref in forward.get("weights", {}).values():
                if ref["tensor"] == weight:
                    ref["requires_grad"] = True
        trainable = lower()
        self.assertFalse(any(op["kernel_id"] == provider for op in trainable["backward"]))
        self.assertIn("grad.weight.output.weight", trainable["tensors"])
