"""Independent PyTorch fixture and checked bounds for reusable FP32 linear rows."""
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/tts/kokoro_projection_pinned.npz"


class LinearRowsOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        lib = Path(cls.temp.name) / "linear.so"
        subprocess.run([
            "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
            "-shared", "-fPIC", str(ROOT / "src/kernels/linear_rows_checked.c"),
            "-I", str(ROOT / "include"), "-lm", "-o", str(lib),
        ], check=True)
        cls.fn = ctypes.CDLL(str(lib)).linear_rows_checked_f32
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
                           ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
        cls.fn.restype = ctypes.c_int
        cls.meta = json.loads(FIXTURE.with_suffix(".json").read_text())
        if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != cls.meta["fixture_sha256"]:
            raise RuntimeError("projection fixture hash mismatch")
        cls.fixture = dict(np.load(FIXTURE))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def call(self, x, w, b, out, m, k, n, xs=None, ws=None, ys=None,
             x_capacity=None, w_capacity=None, b_capacity=None, y_capacity=None):
        return self.fn(x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       x.size if x_capacity is None else x_capacity, k if xs is None else xs,
                       w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       w.size if w_capacity is None else w_capacity, k if ws is None else ws,
                       b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       b.size if b_capacity is None else b_capacity,
                       out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       out.size if y_capacity is None else y_capacity,
                       n if ys is None else ys, m, k, n)

    def test_channel_vector_keeps_ordered_fp64_cancellation(self):
        for rows in (1, 2, 3, 4, 5, 7):
            x = np.tile(np.array([2.0**60, 1.0, -2.0**60, 0.0], np.float32), (rows, 1))
            x[:, 3] = np.arange(rows, dtype=np.float32) / 8
            w = np.ones((5, 4), np.float32)
            bias = np.zeros(5, np.float32)
            output = np.full((rows, 5), -777, np.float32)
            expected = np.repeat(x[:, 3:4], 5, axis=1)
            self.assertEqual(self.call(x, w, bias, output, rows, 4, 5), 0)
            np.testing.assert_array_equal(output.view(np.uint32), expected.view(np.uint32))
            # Horizontal reassociation gives 1, not the contracted first-row 0.
            reassociated = ((float(x[0, 0]) + float(x[0, 2])) +
                            (float(x[0, 1]) + float(x[0, 3])))
            self.assertEqual(reassociated, 1.0)
            self.assertNotEqual(float(output[0, 0]), reassociated)

    def test_simd_validation_preserves_large_cancellation_and_atomic_overflow(self):
        x = np.full((1, 4), np.float32(1e30), dtype=np.float32)
        w = np.tile(np.array([1e10, -1e10, 1e10, -1e10], dtype=np.float32), (5, 1))
        bias = np.arange(5, dtype=np.float32)
        out = np.full((1, 5), -777., dtype=np.float32)
        # The conservative L1 bound requires the two-pass path, while ordered
        # cancellation still produces representable results.
        expected = bias.astype(np.float64)[None, :].copy()
        for c in range(4):
            expected += x[:, c].astype(np.float64)[:, None] * w[:, c].astype(np.float64)[None, :]
        self.assertEqual(self.call(x, w, bias, out, 1, 4, 5), 0)
        np.testing.assert_array_equal(out.view(np.uint32), expected.astype(np.float32).view(np.uint32))
        out.fill(-777.)
        w[-1, :] = np.float32(1e10)
        self.assertEqual(self.call(x, w, bias, out, 1, 4, 5), -1)
        self.assertTrue(np.all(out == -777.))

    def test_pinned_kokoro_projection_against_pytorch(self):
        x, w, b, expected = (self.fixture[key] for key in ("input", "weight", "bias", "expected"))
        out = np.full(expected.shape, -777., dtype=np.float32)
        self.assertEqual(self.call(x, w, b, out, 36, 128, 768), 0)
        self.assertTrue(np.isfinite(out).all())
        error = np.abs(out - expected)
        worst = np.unravel_index(int(np.argmax(error)), error.shape)
        self.assertLessEqual(float(error[worst]), 3e-5)
        print("TTS_LINEAR_PROJECTION_EVIDENCE " + json.dumps({
            "status": "PASS", "oracle": self.meta["oracle"],
            "shape": [36, 768], "max_abs_error": float(error[worst]),
            "worst_row_channel": list(map(int, worst)),
            "reproduce": "python3 -m unittest tests.test_v8_linear_rows_oracle",
            "complete_encoder": "NOT_TESTED", "generated_waveform": "NOT_TESTED",
        }, sort_keys=True))

    def test_strided_rows_and_padding(self):
        x = np.array([[1., 2., 3., 99.], [-2., 1., 4., 99.]], dtype=np.float32)
        w = np.array([[1., 0., 2., 99.], [0., -1., 3., 99.]], dtype=np.float32)
        b = np.array([1., -2.], dtype=np.float32)
        out = np.full((2, 4), -777., dtype=np.float32)
        self.assertEqual(self.call(x, w, b, out, 2, 3, 2, 4, 4, 4), 0)
        np.testing.assert_array_equal(out[:, :2], np.array([[8., 5.], [7., 9.]], np.float32))
        self.assertTrue(np.all(out[:, 2:] == -777.))
        self.assertEqual(self.call(x, w, b, out, 2, 3, 2, 4, 4, 4), 0)

    def test_fp64_order_on_ordinary_nonzero_rows(self):
        rng = np.random.default_rng(37)
        x = rng.standard_normal((3, 17)).astype(np.float32)
        w = rng.standard_normal((7, 17)).astype(np.float32)
        b = rng.standard_normal(7).astype(np.float32)
        out = np.full((3, 9), -777., dtype=np.float32)
        self.assertEqual(self.call(x, w, b, out, 3, 17, 7, ys=9), 0)
        expected = np.empty((3, 7), dtype=np.float32)
        for row in range(3):
            for channel in range(7):
                acc = float(b[channel])
                for column in range(17):
                    acc += float(x[row, column]) * float(w[channel, column])
                expected[row, channel] = np.float32(acc)
        np.testing.assert_array_equal(out[:, :7], expected)
        self.assertTrue(np.all(out[:, 7:] == -777.))

    def test_extreme_cancellation_and_late_overflow_staging(self):
        x = np.array([[1e38, 1e38], [1., 1.]], dtype=np.float32)
        w = np.array([[1e38, -1e38]], dtype=np.float32)
        b = np.array([0.], dtype=np.float32)
        out = np.array([[-777.], [-777.]], dtype=np.float32)
        # The absolute bound is inconclusive, but the exact result is finite.
        self.assertEqual(self.call(x, w, b, out, 2, 2, 1), 0)
        np.testing.assert_array_equal(out, np.zeros((2, 1), dtype=np.float32))

        x = np.array([[1., 1.], [1e38, 1e38]], dtype=np.float32)
        w = np.array([[1., 1.], [1e38, 1e38]], dtype=np.float32)
        b = np.array([0., 0.], dtype=np.float32)
        out = np.full((2, 2), -777., dtype=np.float32)
        # A later result exceeds FP32 after earlier results would have passed.
        self.assertNotEqual(self.call(x, w, b, out, 2, 2, 2), 0)
        self.assertTrue(np.all(out == -777.))
        w[1] = [1., 1.]
        self.assertEqual(self.call(x, w, b, out, 2, 2, 2), 0)
        self.assertTrue(np.isfinite(out).all())

    def test_rejections_leave_output_unchanged(self):
        x = np.array([1., 2., 3., 4.], dtype=np.float32)
        w = np.array([1., 2., 3., 4.], dtype=np.float32)
        b = np.array([1.], dtype=np.float32)
        out = np.array([-777.], dtype=np.float32)
        cases = [
            dict(m=1, k=4, n=1, x_capacity=3),
            dict(m=1, k=4, n=1, w_capacity=3),
            dict(m=1, k=4, n=1, b_capacity=0),
            dict(m=1, k=4, n=1, y_capacity=0),
            dict(m=0, k=4, n=1),
            dict(m=1, k=0, n=1),
            dict(m=1, k=4, n=0),
            dict(m=1, k=4, n=1, xs=3),
            dict(m=1, k=4, n=1, ws=3),
            dict(m=1, k=4, n=1, ys=0),
            dict(m=2**63, k=4, n=1, xs=2**63),
        ]
        for kw in cases:
            with self.subTest(kw=kw):
                self.assertNotEqual(self.call(x, w, b, out, **kw), 0)
                self.assertEqual(float(out[0]), -777.)
        for target in (x, w, b):
            original = float(target[0])
            for invalid in (np.nan, np.inf, -np.inf):
                target[0] = invalid
                self.assertNotEqual(self.call(x, w, b, out, 1, 4, 1), 0)
                self.assertEqual(float(out[0]), -777.)
            target[0] = original
        x[0], w[0] = 1e38, 1e38
        self.assertNotEqual(self.call(x, w, b, out, 1, 4, 1), 0)
        self.assertEqual(float(out[0]), -777.)

    def test_small_pytorch_live_when_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable; live oracle NOT_TESTED")
        x = np.arange(15, dtype=np.float32).reshape(3, 5) / 7
        w = np.arange(35, dtype=np.float32).reshape(7, 5) / 11
        b = np.arange(7, dtype=np.float32) / 13
        expected = torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(w),
                                              torch.from_numpy(b)).numpy()
        out = np.empty((3, 7), dtype=np.float32)
        self.assertEqual(self.call(x, w, b, out, 3, 5, 7), 0)
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
