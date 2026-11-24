import ctypes
from pathlib import Path
import platform
import subprocess
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from version.v8.scripts.build_inkling2_cross_pass_v8 import CONTROL_PORTS, control_weight_shapes, cross_pass_controls
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph


class InklingCrossPassTests(unittest.TestCase):
    @unittest.skipUnless(platform.system() == "Linux", "optional CKE pool uses ELF linkage")
    def test_depth_rows_keep_exact_reduction_order_in_pool(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as tmp:
            library = Path(tmp) / "depth-pool.so"
            subprocess.run([
                "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                "-I", str(root / "include"), str(root / "src/kernels/depth_residual_mix_checked.c"),
                str(root / "src/ck_threadpool.c"), str(root / "src/ckernel_strict.c"),
                "-lm", "-lpthread", "-ldl", "-o", str(library),
            ], check=True)
            lib = ctypes.CDLL(str(library))
            fn = lib.depth_residual_mix_checked_f32
            fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t] * 4 + [ctypes.c_size_t] * 3 + [ctypes.c_float]
            fn.restype = ctypes.c_int
            lib.ck_threadpool_global.restype = ctypes.c_void_p
            lib.ck_threadpool_profile_reset.argtypes = [ctypes.c_void_p]
            lib.ck_threadpool_profile_snapshot.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
            lib.ck_set_num_threads(2)
            pool = lib.ck_threadpool_global()
            try:
                rng = np.random.default_rng(6202)
                rows, states, width = 19, 3, 133
                bank = rng.standard_normal((rows, states, width), dtype=np.float32)
                current = rng.standard_normal((rows, width), dtype=np.float32)
                query = rng.standard_normal(width, dtype=np.float32)
                output = np.empty_like(current)

                def run():
                    tensors = [value for array in (bank, current, query, output)
                               for value in (array.ctypes.data, array.size)]
                    return fn(*tensors, rows, states, width, 1e-6)

                lib.ck_set_num_threads(1)
                self.assertEqual(run(), 0)
                expected = output.view("u4").copy()
                lib.ck_set_num_threads(2)
                counts = (ctypes.c_uint64 * 4)()
                lib.ck_threadpool_profile_reset(pool)
                self.assertEqual(run(), 0)
                lib.ck_threadpool_profile_snapshot(pool, counts)
                self.assertGreater(counts[0], 0)
                np.testing.assert_array_equal(output.view("u4"), expected)
                bank[-1, -1, -1] = np.nan
                output.fill(-5)
                self.assertEqual(run(), -3)
                self.assertTrue(np.all(output == -5))
            finally:
                lib.ck_threadpool_global_destroy()

    def test_prev_pass_messages_run_all_four_controllers(self):
        rows, states, width, rank = 3, 2, 7, 4
        rng = np.random.default_rng(437)
        graph = cross_pass_controls(rows, width, rank, states)
        decl = {name: (shape, "F32") for name, shape in control_weight_shapes(graph).items()}
        weights = {name: rng.normal(0, .2, shape).astype(np.float32) for name, (shape, _) in decl.items()}
        with TemporaryDirectory() as tmp:
            fn, arena, view, _ = compile_checked_weight_graph(Path(tmp), graph, weights, decl)
            initial = {}
            for name in ("attention_input", "ffn_input", "prior_bank", "prior_output"):
                initial[name] = rng.standard_normal(view(name).shape).astype(np.float32)
                view(name)[:] = initial[name]
            self.assertEqual(fn(arena, len(arena)), 0)

            def normalized(x):
                return x / np.sqrt(1e-6 + np.mean(x * x, axis=-1, keepdims=True))

            bank = normalized(np.concatenate(
                (initial["prior_bank"], initial["prior_output"][:, None]), axis=1))
            outputs = {}
            for port in CONTROL_PORTS:
                query = weights[f"blk.6.loop_msg_query.{port}"]
                scores = bank @ query / np.sqrt(width)
                probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                message = (bank * probabilities[..., None]).sum(axis=1)
                source = initial["ffn_input" if port == "ffn" else "attention_input"]
                joined = np.concatenate((normalized(source), normalized(message)), axis=1)
                stem = f"blk.6.loop_{port}_ctrl_"
                down = joined @ weights[stem + "down.weight"].T
                gain = np.tanh(down @ weights[stem + "gain.weight"].T)
                forget = np.tanh(down @ weights[stem + "forget.weight"].T)
                expected = (1 - forget) * source + gain * message
                np.testing.assert_allclose(view(f"{port}_message"), message, atol=1e-6)
                np.testing.assert_allclose(view(f"{port}_controlled"), expected, atol=2e-6)
                self.assertGreater(float(abs(expected - source).max()), 0.01)
                outputs[port] = view(f"{port}_controlled").copy()

            view("prior_bank")[:] = initial["prior_bank"] * -1
            view("prior_output")[:] = initial["prior_output"] * -1
            self.assertEqual(fn(arena, len(arena)), 0)
            self.assertGreater(float(abs(outputs["q"] - view("q_controlled")).max()), .01)
            view("prior_bank")[:] = initial["prior_bank"]
            view("prior_output")[:] = initial["prior_output"]
            self.assertEqual(fn(arena, len(arena)), 0)
            for port in CONTROL_PORTS:
                np.testing.assert_array_equal(view(f"{port}_controlled"), outputs[port])


if __name__ == "__main__":
    unittest.main()
