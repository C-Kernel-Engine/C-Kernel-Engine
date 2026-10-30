"""Captured publisher BF16 MLA arithmetic at Instella decoder layer 11."""

import ctypes
import copy
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/instella_mla11"
FLOAT_PTR = ctypes.POINTER(ctypes.c_float)


class InstellaMlaBf16PhasesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(
            ["make", "--no-print-directory", "build/libckernel_engine.so"],
            cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
        )
        cls.engine = ctypes.CDLL(str(ROOT / "build/libckernel_engine.so"))
        cls.prefill_args = [FLOAT_PTR] * 4 + [ctypes.c_int] * 5 + [ctypes.c_float, FLOAT_PTR, ctypes.c_size_t]
        cls.decode_args = [FLOAT_PTR] * 4 + [ctypes.c_int] * 7 + [ctypes.c_float, FLOAT_PTR, ctypes.c_size_t]
        cls.prefill = cls.engine.deepseek_mla_attention_bf16_phases_workspace
        cls.prefill.argtypes = cls.prefill_args
        cls.prefill_parallel = cls.engine.deepseek_mla_attention_bf16_phases_parallel_dispatch
        cls.prefill_parallel.argtypes = cls.prefill_args
        cls.old_prefill = cls.engine.deepseek_mla_attention_f32_workspace
        cls.old_prefill.argtypes = cls.prefill_args
        cls.old_prefill_parallel = cls.engine.deepseek_mla_attention_f32_parallel_dispatch
        cls.old_prefill_parallel.argtypes = cls.prefill_args
        cls.decode = cls.engine.deepseek_mla_attention_decode_bf16_phases_workspace
        cls.decode.argtypes = cls.decode_args
        cls.old_decode = cls.engine.deepseek_mla_attention_decode_f32_workspace
        cls.old_decode.argtypes = cls.decode_args

    def setUp(self):
        metadata = json.loads((FIXTURE / "provenance.json").read_text())
        fixture_bytes = (FIXTURE / "captured_head8.npz").read_bytes()
        self.assertEqual(hashlib.sha256(fixture_bytes).hexdigest(), metadata["fixture_sha256"])
        self.scale = metadata["scale"]
        with np.load(FIXTURE / "captured_head8.npz") as fixture:
            self.q = np.ascontiguousarray(fixture["q"][:, None, :])
            self.k = np.ascontiguousarray(fixture["k"][:, None, :])
            self.v = np.ascontiguousarray(fixture["v"][:, None, :])
            self.expected = np.ascontiguousarray(fixture["publisher_out"][:, None, :])

    @staticmethod
    def ptr(array):
        return array.ctypes.data_as(FLOAT_PTR)

    def run_prefill(self, function, q=None, k=None, v=None):
        q = self.q if q is None else q
        k = self.k if k is None else k
        v = self.v if v is None else v
        output = np.full_like(self.expected, np.nan)
        scores = np.empty(len(q), dtype=np.float32)
        function(self.ptr(q), self.ptr(k), self.ptr(v), self.ptr(output),
                 1, 1, len(q), q.shape[2], v.shape[2], self.scale,
                 self.ptr(scores), scores.nbytes)
        return output

    def test_captured_publisher_prefill_and_parallel_are_exact(self):
        np.testing.assert_array_equal(self.run_prefill(self.prefill), self.expected)
        np.testing.assert_array_equal(self.run_prefill(self.prefill_parallel), self.expected)
        old = self.run_prefill(self.old_prefill)
        self.assertGreater(float(np.max(np.abs(old[20] - self.expected[20]))), 0.01)

    def test_decode_matches_each_causal_prefill_prefix(self):
        length, _, dim = self.k.shape
        k_cache = np.ascontiguousarray(self.k.transpose(1, 0, 2))
        v_cache = np.ascontiguousarray(self.v.transpose(1, 0, 2))
        for pos in range(length):
            with self.subTest(pos=pos):
                output = np.full((1, dim), np.nan, dtype=np.float32)
                scores = np.empty(pos + 1, dtype=np.float32)
                self.decode(self.ptr(self.q[pos]), self.ptr(k_cache),
                            self.ptr(v_cache), self.ptr(output),
                            1, 1, pos + 1, dim, dim, length, dim,
                            self.scale, self.ptr(scores), scores.nbytes)
                np.testing.assert_array_equal(output, self.expected[pos])

    def test_cache_position_and_value_perturbations_are_detected(self):
        prefix = self.run_prefill(self.prefill)
        shifted = np.ascontiguousarray(np.roll(self.k, 1, axis=0))
        self.assertFalse(np.array_equal(self.run_prefill(self.prefill, k=shifted)[20], prefix[20]))
        perturbed = self.v.copy()
        perturbed[20, 0, 51] += 16.0
        self.assertFalse(np.array_equal(self.run_prefill(self.prefill, v=perturbed)[20], prefix[20]))

    def test_float_buffers_obey_bf16_operand_storage_boundary(self):
        q = self.q.copy()
        k = self.k.copy()
        v = self.v.copy()
        q[20, 0, 0] = np.nextafter(q[20, 0, 0], np.float32(np.inf))
        k[20, 0, 0] = np.nextafter(k[20, 0, 0], np.float32(np.inf))
        v[20, 0, 0] = np.nextafter(v[20, 0, 0], np.float32(np.inf))
        np.testing.assert_array_equal(self.run_prefill(self.prefill, q, k, v), self.expected)

    def test_instella_selects_declared_phase_providers(self):
        circuit = json.loads((ROOT / "version/v8/circuits/instella_moe.json").read_text())
        self.assertEqual(circuit["kernels"]["mla_attention_prefill"],
                         "deepseek_mla_attention_bf16_phases")
        self.assertEqual(circuit["kernels"]["mla_attention_decode"],
                         "deepseek_mla_attention_decode_bf16_phases")

    def test_numerical_selector_rejects_fp32_arithmetic(self):
        scripts = ROOT / "version/v8/scripts"
        sys.path.insert(0, str(scripts))
        import resolve_numerical_execution_contracts_v8 as resolver
        v8 = ROOT / "version/v8"
        circuit = json.loads((v8 / "circuits/instella_moe.json").read_text())
        contracts = json.loads((v8 / "contracts/numerical_execution.json").read_text())
        kernels = resolver.load_kernel_capabilities(v8 / "kernel_maps", contracts)
        for phase, expected in (("prefill", "deepseek_mla_attention_bf16_phases"),
                                ("decode", "deepseek_mla_attention_decode_bf16_phases")):
            with self.subTest(phase=phase):
                selected = resolver.resolve_contract(
                    circuit, contracts, kernels, "decoder.mla_attention_bf16_phases",
                    phase, mode="production")
                self.assertEqual(selected["kernel"]["id"], expected)
                broken = copy.deepcopy(kernels)
                broken["kernels"][expected]["capabilities"] = []
                with self.assertRaises(resolver.ContractError):
                    resolver.resolve_contract(
                        circuit, contracts, broken,
                        "decoder.mla_attention_bf16_phases", phase,
                        mode="production")

    def test_parallel_gqa_unequal_widths_and_invalid_inputs(self):
        get_threads = self.engine.ck_get_num_threads
        get_threads.restype = ctypes.c_int
        set_threads = self.engine.ck_set_num_threads
        set_threads.argtypes = [ctypes.c_int]
        destroy = self.engine.ck_threadpool_global_destroy
        pool = self.engine.ck_threadpool_global
        pool.restype = ctypes.c_void_p
        count = self.engine.ck_threadpool_n_threads
        count.argtypes = [ctypes.c_void_p]
        count.restype = ctypes.c_int
        original = get_threads()
        try:
            destroy()
            set_threads(4)
            self.assertGreaterEqual(count(pool()), 2)
            t, h, kv, qk, width = 96, 8, 2, 64, 32
            rng = np.random.default_rng(812)
            q = rng.normal(size=(t, h, qk)).astype(np.float32)
            k = rng.normal(size=(t, kv, qk)).astype(np.float32)
            v = rng.normal(size=(t, kv, width)).astype(np.float32)
            serial = np.full((t, h, width), -91, dtype=np.float32)
            parallel = serial.copy()
            scores = np.full(4 * t + 1, -77, dtype=np.float32)
            args = (self.ptr(q), self.ptr(k), self.ptr(v), self.ptr(serial),
                    h, kv, t, qk, width, self.scale, self.ptr(scores),
                    scores.nbytes)
            self.prefill(*args)
            scores.fill(-77)
            args = args[:3] + (self.ptr(parallel),) + args[4:]
            self.prefill_parallel(*args)
            np.testing.assert_array_equal(parallel, serial)
            for worker in range(4):
                self.assertFalse(np.all(scores[worker*t:(worker+1)*t] == -77),
                                 f"worker {worker}'s private score row must be used")
            self.assertEqual(scores[-1], -77)

            for function in (self.prefill, self.prefill_parallel,
                             self.old_prefill, self.old_prefill_parallel):
                for bad in ({"scale": -1.0}, {"scale": float("nan")},
                            {"kv": 0}, {"kv": 3}, {"qk": 0},
                            {"scores_bytes": 0}, {"q": None}):
                    with self.subTest(function=function.__name__, bad=bad):
                        output = np.full_like(serial, -91)
                        values = [self.ptr(q), self.ptr(k), self.ptr(v),
                                  self.ptr(output), h, kv, t, qk, width,
                                  self.scale, self.ptr(scores), scores.nbytes]
                        for key, index in (("scale", 9), ("kv", 5), ("qk", 7),
                                           ("scores_bytes", 11), ("q", 0)):
                            if key in bad:
                                values[index] = bad[key]
                        function(*values)
                        self.assertTrue(np.all(output == -91))
        finally:
            destroy()
            set_threads(original)

    def test_decode_rejects_over_capacity_without_read_or_write(self):
        q = np.ones((2, 3), dtype=np.float32)
        # One cache element per head beyond this allocation would hit the canary.
        cache = np.full((2, 2, 4), 0.5, dtype=np.float32)
        for function in (self.decode, self.old_decode):
            for cache_len in (3, 2):
                with self.subTest(function=function.__name__, cache_len=cache_len):
                    out = np.full((2, 2), -91, dtype=np.float32)
                    scores = np.full(4, -77, dtype=np.float32)
                    function(self.ptr(q), self.ptr(cache), self.ptr(cache),
                             self.ptr(out), 2, 2, cache_len, 3, 2,
                             2, 4, self.scale, self.ptr(scores), scores.nbytes)
                    if cache_len > 2:
                        self.assertTrue(np.all(out == -91))
                        self.assertTrue(np.all(scores == -77))
                    else:
                        self.assertTrue(np.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
