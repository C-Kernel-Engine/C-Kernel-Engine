"""Native regular FP16 decode scheduling and failure publication."""
import ctypes as C
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class PoolProfile(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in (
        "dispatch_count", "dispatch_total_ns", "main_work_ns", "completion_wait_ns",
    )]


class RegularFP16DecodeHeadsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="cke-regular-f16-heads-")
        cls.addClassCleanup(cls.temp.cleanup)
        binary = Path(cls.temp.name) / "attention.so"
        sources = [
            "kernels/attention_kernels.c", "kernels/attention_kernels_sliding.c",
            "kernels/attention_flash_true.c", "kernels/softmax_kernels.c",
            "kernels/gemm_kernels_bf16.c", "ckernel_strict.c", "ck_threadpool.c",
        ]
        subprocess.run([
            "gcc", "-O2", "-std=gnu11", "-shared", "-fPIC",
            "-I", str(ROOT / "include"), *(str(ROOT / "src" / path) for path in sources),
            "-lm", "-lpthread", "-o", str(binary),
        ], check=True)
        cls.lib = C.CDLL(str(binary))
        cls.lib.ck_set_num_threads.argtypes = [C.c_int]
        cls.lib.ck_set_num_threads.restype = None
        cls.lib.ck_threadpool_global.restype = C.c_void_p
        cls.lib.ck_threadpool_profile_reset.argtypes = [C.c_void_p]
        cls.lib.ck_threadpool_profile_snapshot.argtypes = [C.c_void_p, C.POINTER(PoolProfile)]
        cls.lib.ck_threadpool_n_threads.argtypes = [C.c_void_p]
        cls.lib.ck_threadpool_n_threads.restype = C.c_int
        with patch.dict(os.environ, {"CK_NUM_THREADS": "8"}):
            cls.lib.ck_set_num_threads(8)
            cls.pool = cls.lib.ck_threadpool_global()
        cls.workers = cls.lib.ck_threadpool_n_threads(cls.pool)
        cls.fn = cls.lib.attention_forward_decode_head_major_gqa_f16cache_regular_noscale
        cls.fn.argtypes = [C.c_void_p] * 4 + [C.c_int] * 7
        cls.fn.restype = C.c_int

    @staticmethod
    def inputs(heads=6, groups=2, dim=257, stride=272, capacity=263, seed=1211):
        rng = np.random.default_rng(seed)
        return (
            rng.normal(0, .1, (heads, stride)).astype("f4"),
            rng.normal(0, .1, (groups, capacity, stride)).astype("f2"),
            rng.normal(0, .1, (groups, capacity, stride)).astype("f2"),
            dim,
        )

    def invoke(self, inputs, tokens, window, output=None):
        q, k, v, dim = inputs
        guarded = np.full(q.size + 2, 99, dtype="f4")
        dst = guarded[1:-1] if output is None else output
        status = self.fn(q.ctypes.data, k.ctypes.data, v.ctypes.data, dst.ctypes.data,
                         q.shape[0], k.shape[0], tokens, k.shape[1], dim, q.shape[1], window)
        self.assertEqual((guarded[0], guarded[-1]), (99, 99))
        return status, dst.tobytes()

    def compare(self, inputs, tokens, window, expected_dispatches):
        with patch.dict(os.environ, {"CK_DISABLE_ATTENTION_THREADPOOL": "1"}):
            expected = self.invoke(inputs, tokens, window)
        with patch.dict(os.environ, {"CK_DISABLE_ATTENTION_THREADPOOL": "0", "CK_ATTENTION_THREAD_CAP": "8"}):
            self.lib.ck_threadpool_profile_reset(self.pool)
            actual = self.invoke(inputs, tokens, window)
            profile = PoolProfile()
            self.lib.ck_threadpool_profile_snapshot(self.pool, C.byref(profile))
        self.assertEqual(actual, expected)
        self.assertEqual(profile.dispatch_count, expected_dispatches if self.workers > 1 else 0)
        return actual

    def test_effective_window_and_worker_count(self):
        for heads, groups in ((3, 1), (6, 2), (10, 2), (1, 1)):
            for tokens, window in ((63, 0), (256, 32), (64, 0), (256, 256)):
                with self.subTest(heads=heads, tokens=tokens, window=window):
                    live = min(tokens, window) if window else tokens
                    expected = int(heads > 1 and live >= 64 and heads * live * 257 >= 131072)
                    self.assertEqual(self.compare(self.inputs(heads, groups), tokens, window, expected)[0], 0)
        with patch.dict(os.environ, {"CK_DISABLE_ATTENTION_THREADPOOL": "0", "CK_ATTENTION_THREAD_CAP": "1"}):
            self.lib.ck_threadpool_profile_reset(self.pool)
            self.assertEqual(self.invoke(self.inputs(), 256, 0)[0], 0)
            counters = PoolProfile()
            self.lib.ck_threadpool_profile_snapshot(self.pool, C.byref(counters))
        self.assertEqual(counters.dispatch_count, 0)

    def test_failure_prefix_and_recovery(self):
        for index in (0, 3, 5):
            inputs = self.inputs()
            q = inputs[0]
            saved = q[index].copy()
            q[index, :] = np.nan
            status, data = self.compare(inputs, 256, 0, 1)
            self.assertNotEqual(status, 0)
            rows = np.frombuffer(data, dtype="f4").reshape(q.shape)
            self.assertTrue(np.all(rows[index:] == 99))
            self.assertTrue(np.all(rows[:index, :inputs[-1]] != 99))
            q[index] = saved
            self.assertEqual(self.compare(inputs, 256, 0, 1)[0], 0)

    def test_score_reuse_matches_recomputation(self):
        flag = "CK_EXPERIMENT_REGULAR_FP16_CACHE_SCORES"
        inputs = self.inputs()
        for window in (0, 129):
            for bad_head in (None, 0, 3, 5):
                with self.subTest(window=window, bad_head=bad_head):
                    q = inputs[0]
                    saved = q.copy()
                    if bad_head is not None:
                        q[bad_head] = np.nan
                    with patch.dict(os.environ, {flag: "0"}):
                        expected = self.compare(inputs, 256, window, 1)
                    with patch.dict(os.environ, {flag: "1"}):
                        self.assertEqual(self.compare(inputs, 256, window, 1), expected)
                    q[:] = saved

        long_cache = self.inputs(1, 1, 17, 32, 16385)
        for tokens in (16384, 16385):
            with self.subTest(tokens=tokens):
                with patch.dict(os.environ, {flag: "0"}):
                    expected = self.invoke(long_cache, tokens, 0)
                with patch.dict(os.environ, {flag: "1"}):
                    self.assertEqual(self.invoke(long_cache, tokens, 0), expected)

    def test_alias_staging_limit_and_concurrent_callers(self):
        inputs = self.inputs()
        q = inputs[0]
        saved = q.copy()
        with patch.dict(os.environ, {"CK_DISABLE_ATTENTION_THREADPOOL": "1"}):
            expected = self.invoke(inputs, 256, 0, q)
        q[:] = saved
        with patch.dict(os.environ, {"CK_DISABLE_ATTENTION_THREADPOOL": "0"}):
            self.lib.ck_threadpool_profile_reset(self.pool)
            self.assertEqual(self.invoke(inputs, 256, 0, q), expected)
            profile = PoolProfile()
            self.lib.ck_threadpool_profile_snapshot(self.pool, C.byref(profile))
            self.assertEqual(profile.dispatch_count, 0)
        self.assertEqual(self.compare(self.inputs(8, 2, 1025, 1040), 256, 0, 0)[0], 0)
        cases = [self.inputs(seed=i) for i in range(12)]
        with patch.dict(os.environ, {"CK_DISABLE_ATTENTION_THREADPOOL": "1"}):
            reference = [self.invoke(case, 256, 128) for case in cases]
        with patch.dict(os.environ, {"CK_DISABLE_ATTENTION_THREADPOOL": "0", "CK_ATTENTION_THREAD_CAP": "8"}):
            with ThreadPoolExecutor(max_workers=4) as workers:
                actual = list(workers.map(lambda case: self.invoke(case, 256, 128), cases))
        self.assertEqual(actual, reference)


if __name__ == "__main__":
    unittest.main()
