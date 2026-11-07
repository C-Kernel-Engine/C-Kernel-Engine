"""Independent arithmetic and rejection tests for unmasked cross attention."""

import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def reference(query, key, value, heads):
    q_tokens, width = query.shape
    memory_tokens = key.shape[0]
    head_dim = width // heads
    q = query.reshape(q_tokens, heads, head_dim).astype("f8")
    k = key.reshape(memory_tokens, heads, head_dim).astype("f8")
    v = value.reshape(memory_tokens, heads, head_dim).astype("f8")
    result = np.empty((q_tokens, heads, head_dim), dtype="f4")
    for t in range(q_tokens):
        for h in range(heads):
            scores = (k[:, h] @ q[t, h] / np.sqrt(head_dim)).astype("f4")
            weights = np.exp(scores.astype("f8") - float(scores.max())).astype("f4")
            result[t, h] = ((weights.astype("f8") @ v[:, h]) /
                            weights.astype("f8").sum()).astype("f4")
    return result.reshape(q_tokens, width)


class CrossAttentionCheckedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        path = Path(cls.temp.name) / "cross.so"
        subprocess.run([
            "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
            "-shared", "-fPIC", "-ffp-contract=off",
            "-I" + str(ROOT / "include"),
            str(ROOT / "src/kernels/attention_cross_token_major_checked.c"),
            "-lm", "-o", str(path)], check=True, capture_output=True)
        cls.library = ctypes.CDLL(str(path))
        cls.fn = cls.library.ck_attention_cross_token_major_checked_f32
        fp = ctypes.POINTER(ctypes.c_float)
        size = ctypes.c_size_t
        cls.fn.argtypes = [fp, size, fp, size, fp, size, fp, size,
                           fp, size, size, size, size, size]
        cls.fn.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_case(self, query, key, value, heads, *, output=None,
                 scratch=None, memory_tokens=None):
        q_tokens, width = query.shape
        mem = key.shape[0] if memory_tokens is None else memory_tokens
        if output is None:
            output = np.full_like(query, -91)
        if scratch is None:
            scratch = np.zeros(key.shape[0] + query.size, dtype="f4")
        ptr = ctypes.POINTER(ctypes.c_float)
        status = self.fn(
            query.ctypes.data_as(ptr), query.size,
            key.ctypes.data_as(ptr), key.size,
            value.ctypes.data_as(ptr), value.size,
            output.ctypes.data_as(ptr), output.size,
            scratch.ctypes.data_as(ptr), scratch.size,
            q_tokens, mem, heads, width // heads)
        return status, output

    def test_unequal_lengths_and_head_tails(self):
        rng = np.random.default_rng(801)
        for q_tokens, memory_tokens, heads, head_dim in (
                (1, 3, 1, 5), (3, 7, 2, 3), (5, 2, 4, 7)):
            with self.subTest(shape=(q_tokens, memory_tokens, heads, head_dim)):
                shape = (q_tokens, heads * head_dim)
                mem_shape = (memory_tokens, heads * head_dim)
                q = rng.normal(0, 0.2, shape).astype("f4")
                k = rng.normal(0, 0.2, mem_shape).astype("f4")
                v = rng.normal(0, 0.2, mem_shape).astype("f4")
                status, actual = self.run_case(q, k, v, heads)
                self.assertEqual(status, 0)
                np.testing.assert_allclose(actual, reference(q, k, v, heads),
                                           atol=1e-7, rtol=0)

    def test_rejection_and_recovery(self):
        q = np.ones((2, 4), dtype="f4")
        k = np.ones((3, 4), dtype="f4")
        v = np.arange(12, dtype="f4").reshape(3, 4)
        output = np.full_like(q, -91)
        for kwargs in (
                {"scratch": np.zeros(3 + q.size - 1, dtype="f4")},
                {"memory_tokens": 4},
                {"output": q},
                {"scratch": v.reshape(-1)},
        ):
            with self.subTest(kwargs=tuple(kwargs)):
                selected = {"output": output, **kwargs}
                status, _ = self.run_case(q, k, v, 2, **selected)
                self.assertNotEqual(status, 0)
                np.testing.assert_array_equal(output, -91)
        k[0, 0] = np.nan
        status, _ = self.run_case(q, k, v, 2, output=output)
        self.assertNotEqual(status, 0)
        np.testing.assert_array_equal(output, -91)
        k[0, 0] = 1
        overlap_base = np.ones(q.size + 1, dtype="f4")
        overlap_query = overlap_base[:q.size].reshape(q.shape)
        overlap_output = overlap_base[1:].reshape(q.shape)
        status, _ = self.run_case(overlap_query, k, v, 2,
                                  output=overlap_output)
        self.assertNotEqual(status, 0)
        np.testing.assert_array_equal(overlap_base, 1)
        high = np.full_like(q, 1e30)
        status, _ = self.run_case(high, np.full_like(k, 1e30), v, 2,
                                  output=output)
        self.assertNotEqual(status, 0)
        np.testing.assert_array_equal(output, -91)
        status, _ = self.run_case(q, k, v, 2, output=output)
        self.assertEqual(status, 0)
        np.testing.assert_allclose(output, reference(q, k, v, 2), atol=1e-7, rtol=0)

    def test_pytorch_when_available(self):
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError:
            self.skipTest("PyTorch comparison unavailable")
        rng = np.random.default_rng(91)
        q = rng.normal(0, 0.2, (3, 8)).astype("f4")
        k = rng.normal(0, 0.2, (6, 8)).astype("f4")
        v = rng.normal(0, 0.2, (6, 8)).astype("f4")
        status, actual = self.run_case(q, k, v, 2)
        self.assertEqual(status, 0)
        def shaped(x):
            return torch.from_numpy(x).reshape(1, -1, 2, 4).transpose(1, 2)
        expected = functional.scaled_dot_product_attention(
            shaped(q), shaped(k), shaped(v), dropout_p=0.0,
            is_causal=False).transpose(1, 2).reshape(3, 8)
        np.testing.assert_allclose(actual, expected.numpy(), atol=2e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
