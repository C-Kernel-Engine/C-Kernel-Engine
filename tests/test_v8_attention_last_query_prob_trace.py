"""Native selected-layer attention observation and rejection controls."""

import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/kernels/attention_last_query_prob_trace_f32.c"


class AttentionLastQueryProbTraceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        library = Path(cls.temp.name) / "trace.so"
        subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-fPIC", "-shared", str(SOURCE), "-lm", "-o", str(library)],
                       check=True)
        cls.fn = ctypes.CDLL(str(library)).attention_last_query_prob_trace_f32
        cls.fn.argtypes = ([ctypes.c_void_p, ctypes.c_size_t] * 4 +
                           [ctypes.c_size_t] * 7 + [ctypes.c_float])
        cls.fn.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def invoke(self, q, k, trace, scratch, *, layer=1, valid_frames=5,
               query_elements=None, key_elements=None, trace_elements=None,
               score_elements=None, scale=0.5):
        layers, heads, frame_capacity = trace.shape
        return self.fn(q.ctypes.data, q.size if query_elements is None else query_elements,
                       k.ctypes.data, k.size if key_elements is None else key_elements,
                       trace.ctypes.data, trace.size if trace_elements is None else trace_elements,
                       scratch.ctypes.data, scratch.size if score_elements is None else score_elements,
                       layers, layer, heads, q.shape[1], valid_frames,
                       frame_capacity, q.shape[2], scale)

    @staticmethod
    def sample():
        rng = np.random.default_rng(818)
        q = rng.normal(size=(3, 2, 4)).astype(np.float32)
        k = rng.normal(size=(3, 7, 4)).astype(np.float32)
        trace = np.full((4, 3, 7), -77, np.float32)
        scratch = np.zeros((3, 7), np.float32)
        return q, k, trace, scratch

    def test_last_query_and_layer_slice_match_independent_softmax(self):
        q, k, trace, scratch = self.sample()
        self.assertEqual(self.invoke(q, k, trace, scratch), 0)
        scores = np.einsum("hd,hfd->hf", q[:, -1].astype(np.float64),
                           k[:, :5].astype(np.float64)) * 0.5
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(trace[1, :, :5], weights, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(trace[0], -77)
        np.testing.assert_array_equal(trace[2:], -77)
        np.testing.assert_array_equal(trace[1, :, 5:], -77)
        np.testing.assert_allclose(trace[1, :, :5].sum(axis=1), 1, atol=2e-6)

    def test_rejects_bad_capacity_and_nonfinite_without_publishing(self):
        q, k, trace, scratch = self.sample()
        for kwargs in ({"layer": 4}, {"valid_frames": 8},
                       {"query_elements": q.size - 1},
                       {"key_elements": k.size - 1},
                       {"trace_elements": trace.size - 1},
                       {"score_elements": 3 * 5 - 1}):
            with self.subTest(kwargs=kwargs):
                trace.fill(-77)
                self.assertNotEqual(self.invoke(q, k, trace, scratch, **kwargs), 0)
                np.testing.assert_array_equal(trace, -77)
        k[0, 0, 0] = np.nan
        self.assertNotEqual(self.invoke(q, k, trace, scratch), 0)
        np.testing.assert_array_equal(trace, -77)
        k[0, 0, 0] = 0
        self.assertEqual(self.invoke(q, k, trace, scratch), 0)


if __name__ == "__main__":
    unittest.main()
