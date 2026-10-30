"""Independent NumPy controls for the checked native attention aligner."""

from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/kernels/alignment_attention_dtw_f32.c"


def reference(attention: np.ndarray, width: int) -> np.ndarray:
    # Independent array implementation of the pinned reference's order:
    # normalize over text tokens, reflect-median over frames, average heads,
    # then cost-minimizing DTW with its exact tie priority.
    heads, tokens, frames = attention.shape
    mean = attention.mean(axis=1, keepdims=True)
    std = attention.std(axis=1, keepdims=True)
    norm = (attention - mean) / std
    if frames > width // 2:
        padded = np.pad(norm, ((0, 0), (0, 0), (width // 2, width // 2)),
                        mode="reflect")
        norm = np.median(np.stack([padded[..., i:i + frames]
                                   for i in range(width)], axis=-1), axis=-1)
    scores = norm.mean(axis=0)
    cost = np.full((tokens + 1, frames + 1), np.inf, dtype=np.float32)
    trace = np.full((tokens + 1, frames + 1), -1, dtype=np.int8)
    cost[0, 0] = 0
    for frame in range(1, frames + 1):
        for token in range(1, tokens + 1):
            diagonal = cost[token - 1, frame - 1]
            vertical = cost[token - 1, frame]
            horizontal = cost[token, frame - 1]
            if diagonal < vertical and diagonal < horizontal:
                prior, direction = diagonal, 0
            elif vertical < diagonal and vertical < horizontal:
                prior, direction = vertical, 1
            else:
                prior, direction = horizontal, 2
            cost[token, frame] = -scores[token - 1, frame - 1] + prior
            trace[token, frame] = direction
    starts = np.full(tokens, -1, dtype=np.int32)
    token, frame = tokens, frames
    while token or frame:
        if token:
            starts[token - 1] = max(0, frame - 1)
        direction = trace[token, frame] if token and frame else (1 if token else 2)
        if direction in (0, 1):
            token -= 1
        if direction in (0, 2):
            frame -= 1
    return starts


class AlignmentAttentionDtwTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        library = Path(cls.temp.name) / "align.so"
        subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-fPIC", "-shared", str(SOURCE), "-lm", "-o", str(library)],
                       check=True)
        cls.fn = ctypes.CDLL(str(library)).alignment_attention_dtw_f32
        cls.fn.argtypes = ([ctypes.c_void_p, ctypes.c_size_t] +
                           [ctypes.c_size_t] * 6 +
                           [ctypes.c_void_p, ctypes.c_size_t] * 4 +
                           [ctypes.c_void_p, ctypes.c_size_t])
        cls.fn.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def call_native(self, attention, width=3, *, output_capacity=None):
        values = np.ascontiguousarray(attention, dtype=np.float32)
        heads, tokens, frames = values.shape
        matrix = tokens * frames
        cells = (tokens + 1) * (frames + 1)
        score = np.zeros(matrix, np.float32)
        mean = np.zeros(matrix, np.float32)
        cost = np.zeros(cells, np.float32)
        trace = np.zeros(cells, np.uint8)
        capacity = tokens if output_capacity is None else output_capacity
        result = np.full(max(capacity, 1), -777, np.int32)
        status = self.fn(values.ctypes.data, values.size, heads, tokens, frames,
                         frames, tokens * frames, width,
                         score.ctypes.data, score.size,
                         mean.ctypes.data, mean.size,
                         cost.ctypes.data, cost.size,
                         trace.ctypes.data, trace.size,
                         result.ctypes.data, capacity)
        return status, result

    def test_multiple_heads_reflect_median_and_dtw(self):
        rng = np.random.default_rng(441)
        for heads, tokens, frames, width in ((2, 5, 9, 3), (3, 7, 13, 7),
                                             (1, 3, 4, 3), (2, 4, 2, 7)):
            with self.subTest(shape=(heads, tokens, frames), width=width):
                attention = rng.uniform(0.01, 1.0, (heads, tokens, frames)).astype(np.float32)
                status, actual = self.call_native(attention, width)
                self.assertEqual(status, 0)
                np.testing.assert_array_equal(actual, reference(attention, width))
                self.assertTrue(np.all(actual[:-1] <= actual[1:]))

    def test_rejection_preserves_output_and_recovery(self):
        rng = np.random.default_rng(8)
        attention = rng.uniform(0.01, 1.0, (2, 4, 8)).astype(np.float32)
        bad = attention.copy()
        bad[0, 0, 1] = np.nan
        for values, width, capacity in ((bad, 3, None), (attention, 4, None),
                                        (attention, 3, 3),
                                        (np.ones_like(attention), 3, None)):
            with self.subTest(width=width, capacity=capacity):
                status, actual = self.call_native(values, width, output_capacity=capacity)
                self.assertNotEqual(status, 0)
                np.testing.assert_array_equal(actual, -777)
        status, actual = self.call_native(attention)
        self.assertEqual(status, 0)
        np.testing.assert_array_equal(actual, reference(attention, 3))

    def test_physical_strides_padding_and_exact_capacity(self):
        heads, tokens, frames = 2, 4, 6
        token_stride, head_stride = 9, 42
        source = np.random.default_rng(42).uniform(.02, 1.,
                                                   (heads, tokens, frames)).astype(np.float32)
        padded = np.full(heads * head_stride, -909., np.float32)
        padding_mask = np.ones(padded.size, dtype=bool)
        for h in range(heads):
            for t in range(tokens):
                row = h * head_stride + t * token_stride
                padded[row:row + frames] = source[h, t]
                padding_mask[row:row + frames] = False
        matrix = tokens * frames
        cells = (tokens + 1) * (frames + 1)
        score = np.zeros(matrix, np.float32)
        mean = np.zeros(matrix, np.float32)
        cost = np.zeros(cells, np.float32)
        trace = np.zeros(cells, np.uint8)
        output = np.full(tokens + 1, -501, np.int32)

        def call(capacity):
            return self.fn(padded.ctypes.data, capacity, heads, tokens, frames,
                           token_stride, head_stride, 3,
                           score.ctypes.data, score.size,
                           mean.ctypes.data, mean.size,
                           cost.ctypes.data, cost.size,
                           trace.ctypes.data, trace.size,
                           output.ctypes.data, output.size)

        required = (heads - 1) * head_stride + (tokens - 1) * token_stride + frames
        self.assertNotEqual(call(required - 1), 0)
        np.testing.assert_array_equal(output, -501)
        self.assertEqual(call(required), 0)
        np.testing.assert_array_equal(output[:tokens], reference(source, 3))
        self.assertEqual(int(output[tokens]), -501)
        self.assertTrue(np.all(padded[padding_mask] == -909.))


if __name__ == "__main__":
    unittest.main()
