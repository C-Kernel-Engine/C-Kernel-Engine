"""Checked right-cropped dense transposed convolution against an independent oracle."""

import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POINTER = ctypes.POINTER(ctypes.c_float)
SIZE = ctypes.c_size_t


class AudioDenseDeconvCropRightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.functions = []
        for flag in ("-O0", "-O3"):
            library = Path(cls.temp.name) / f"crop_{flag}.so"
            subprocess.run([
                "cc", "-std=c11", flag, "-Wall", "-Wextra", "-Werror",
                "-pedantic", "-shared", "-fPIC", "-I", str(ROOT / "include"),
                str(ROOT / "src/kernels/audio_conv_transpose_dense_checked.c"),
                "-lm", "-o", str(library),
            ], check=True, capture_output=True)
            fn = ctypes.CDLL(str(library)).audio_conv_transpose1d_dense_crop_right_channel_major_f32_checked
            fn.argtypes = [POINTER, SIZE, SIZE, POINTER, SIZE, POINTER, SIZE,
                           POINTER, SIZE, SIZE, POINTER, SIZE] + [SIZE] * 9
            fn.restype = ctypes.c_int
            cls.functions.append(fn)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def pointer(value):
        return value.ctypes.data_as(POINTER)

    def test_oracle_padding_tail_rejection_and_recovery(self):
        rng = np.random.default_rng(20261024)
        for frames, stride in ((1, 2), (3, 2), (5, 3), (1, 4), (3, 6), (5, 10)):
            with self.subTest(frames=frames, stride=stride):
                kernel, crop = 2 * stride, stride
                valid = frames * stride
                source = np.full((2, frames + 3), np.nan, np.float32)
                source[:, :frames] = rng.normal(size=(2, frames)).astype(np.float32)
                weight = rng.normal(size=(2, 3, kernel)).astype(np.float32) / 5
                bias = rng.normal(size=3).astype(np.float32) / 5
                reference = np.tile(bias[:, None].astype(np.float64), (1, valid))
                for ic in range(2):
                    for frame in range(frames):
                        for tap in range(kernel):
                            position = frame * stride + tap
                            if position < valid:
                                reference[:, position] += (float(source[ic, frame]) *
                                                         weight[ic, :, tap])
                results = []
                for fn in self.functions:
                    output = np.full((3, valid + 4), -91, np.float32)
                    scratch = np.full(3 * valid, -37, np.float32)
                    args = [self.pointer(source), source.size, source.shape[1],
                            self.pointer(weight), weight.size,
                            self.pointer(bias), bias.size,
                            self.pointer(output), output.size, output.shape[1],
                            self.pointer(scratch), scratch.size,
                            2, 3, frames, kernel, stride, 0, 0, crop, valid]
                    self.assertEqual(fn(*args), 0)
                    np.testing.assert_allclose(output[:, :valid], reference,
                                               atol=3e-6, rtol=3e-6)
                    np.testing.assert_array_equal(output[:, valid:], -91)
                    results.append(output.copy())
                    for slot, replacement in ((1, 0), (4, weight.size - 1),
                                              (8, 0), (11, 0), (19, crop - 1),
                                              (19, crop + 1), (20, valid + 1)):
                        bad = args.copy()
                        bad[slot] = replacement
                        output[:] = -91
                        self.assertNotEqual(fn(*bad), 0, (slot, replacement))
                        np.testing.assert_array_equal(output, -91)
                    saved_weight = weight[0, 0, 0].copy()
                    weight[0, 0, 0] = np.nan
                    output[:] = -91
                    self.assertNotEqual(fn(*args), 0)
                    np.testing.assert_array_equal(output, -91)
                    weight[0, 0, 0] = saved_weight
                    self.assertEqual(fn(*args), 0)
                    np.testing.assert_allclose(output[:, :valid], reference,
                                               atol=3e-6, rtol=3e-6)
                # Independent calls must retain their own padded output rows.
                np.testing.assert_array_equal(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
