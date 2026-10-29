#!/usr/bin/env python3
"""Q4_K routed expert packing reuse versus the production row provider."""

from __future__ import annotations

import ctypes as C
from concurrent.futures import ThreadPoolExecutor
import os
import struct
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get("CKE_MOE_REUSE_BASE_LIBRARY")
BASE_LIB = C.CDLL(BASE, mode=C.RTLD_GLOBAL) if BASE else None
LIB = C.CDLL(os.environ.get("CKE_MOE_REUSE_TEST_LIBRARY", str(ROOT / "build/libckernel_engine.so")),
             mode=C.RTLD_GLOBAL | getattr(os, "RTLD_DEEPBIND", 0))
F32 = C.POINTER(C.c_float)
I32 = C.POINTER(C.c_int)
ARGS = [F32, I32, F32, C.c_void_p, C.c_void_p, C.c_void_p,
        F32, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
        C.c_void_p, C.c_size_t]


def q4_weights(blocks: int, seed: int) -> C.Array:
    rng = np.random.default_rng(seed)
    raw = bytearray(blocks * 144)
    for block in range(blocks):
        pos = block * 144
        struct.pack_into("<H", raw, pos, 0x211F + block % 7)
        struct.pack_into("<H", raw, pos + 2, 0x1800 + block % 5)
        raw[pos + 4:pos + 144] = rng.integers(0, 256, 140, dtype=np.uint8).tobytes()
    return C.create_string_buffer(bytes(raw), len(raw))


def down_weights(blocks: int, block_bytes: int, seed: int) -> C.Array:
    rng = np.random.default_rng(seed)
    raw = bytearray(blocks * block_bytes)
    for block in range(blocks):
        pos = block * block_bytes
        struct.pack_into("<H", raw, pos, 0x211F + block % 7)
        raw[pos + 2:pos + block_bytes] = rng.integers(
            0, 256, block_bytes - 2, dtype=np.uint8).tobytes()
    return C.create_string_buffer(bytes(raw), len(raw))


def stats() -> tuple[int, ...]:
    values = [C.c_uint64() for _ in range(6)]
    LIB.ck_moe_q4k_reuse_stats(*(C.byref(value) for value in values))
    return tuple(value.value for value in values)


class ExpertReuseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        threading_lib = BASE_LIB or LIB
        threading_lib.ck_set_num_threads.argtypes = [C.c_int]
        threading_lib.ck_set_num_threads(4)
        LIB.moe_swiglu_expert_q4k_q8_0_reuse_workspace_bytes.argtypes = [C.c_int] * 5
        LIB.moe_swiglu_expert_q4k_q8_0_reuse_workspace_bytes.restype = C.c_size_t
        cls.reuse_supported = bool(
            LIB.moe_swiglu_expert_q4k_q8_0_reuse_workspace_bytes(517, 256, 256, 4, 3)
        )
        for kind in ("q5_0", "q8_0"):
            for suffix in ("parallel", "reuse"):
                fn = getattr(LIB, f"moe_swiglu_expert_forward_q4k_{kind}_{suffix}_workspace")
                fn.argtypes = ARGS
                fn.restype = C.c_int
        LIB.moe_swiglu_expert_q4k_q8_0_workspace_bytes.argtypes = [C.c_int, C.c_int]
        LIB.moe_swiglu_expert_q4k_q8_0_workspace_bytes.restype = C.c_size_t
        LIB.ck_moe_q4k_reuse_stats.argtypes = [C.POINTER(C.c_uint64)] * 6

    def run_case(self, kind: str, rows: int, experts: int, top_k: int,
                 distinct_routes: bool = False, baseline_workspace: bool = False,
                 intermediate: int = 256) -> tuple[np.ndarray, tuple[int, ...]]:
        hidden = 256
        rng = np.random.default_rng(0xC0DE + rows + experts)
        x = np.ascontiguousarray(rng.normal(0, 0.2, (rows, hidden)), dtype=np.float32)
        indices = np.empty((rows, top_k), dtype=np.int32)
        if distinct_routes:
            indices[:, 0] = np.arange(rows, dtype=np.int32)
        else:
            indices[:] = rng.integers(0, experts, (rows, top_k), dtype=np.int32)
        routing = np.ascontiguousarray(rng.uniform(0.1, 0.9, (rows, top_k)), dtype=np.float32)
        gate = q4_weights(experts * intermediate * (hidden // 256), 1)
        up = q4_weights(experts * intermediate * (hidden // 256), 2)
        block_bytes = 22 if kind == "q5_0" else 34
        down = down_weights(experts * hidden * (intermediate // 32), block_bytes, 3)
        expected = np.empty((rows, hidden), dtype=np.float32)
        actual = np.empty_like(expected)
        stride = LIB.moe_swiglu_expert_q4k_q8_0_workspace_bytes(hidden, intermediate)
        original_space = C.create_string_buffer(stride * 64)
        reuse_bytes = LIB.moe_swiglu_expert_q4k_q8_0_reuse_workspace_bytes(
            rows, hidden, intermediate, experts, top_k)
        if not baseline_workspace and not reuse_bytes:
            self.skipTest("AVX-512 reuse provider unavailable on this runner")
        available = stride * 64 if baseline_workspace else reuse_bytes
        self.assertGreater(available, 0)
        reuse_space = C.create_string_buffer(available)
        common = (x.ctypes.data_as(F32), indices.ctypes.data_as(I32),
                  routing.ctypes.data_as(F32), gate, up, down)
        baseline = getattr(LIB, f"moe_swiglu_expert_forward_q4k_{kind}_parallel_workspace")
        reuse = getattr(LIB, f"moe_swiglu_expert_forward_q4k_{kind}_reuse_workspace")
        self.assertEqual(baseline(*common, expected.ctypes.data_as(F32), rows,
                                  hidden, intermediate, experts, top_k,
                                  original_space, len(original_space)), 0)
        LIB.ck_moe_q4k_reuse_stats_reset()
        self.assertEqual(reuse(*common, actual.ctypes.data_as(F32), rows,
                               hidden, intermediate, experts, top_k,
                               reuse_space, len(reuse_space)), 0)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        return actual, stats()

    def test_repeated_routes_pack_once_per_expert_and_slot(self) -> None:
        for kind in ("q5_0", "q8_0"):
            for intermediate in (256, 640):
                with self.subTest(kind=kind, intermediate=intermediate):
                    _, measured = self.run_case(kind, rows=517, experts=4,
                                                top_k=3, intermediate=intermediate)
                    assignments, distinct, packs, packed_bytes, reuse_calls, fallbacks = measured
                    self.assertEqual(assignments, 517 * 3)
                    self.assertEqual(distinct, 4 * 3)
                    self.assertEqual(packs, 2 * distinct)
                    self.assertGreater(packed_bytes, 0)
                    self.assertEqual((reuse_calls, fallbacks), (1, 0))

    def test_sparse_routes_keep_baseline(self) -> None:
        _, measured = self.run_case("q8_0", rows=512, experts=512,
                                    top_k=1, distinct_routes=True)
        self.assertEqual(measured[2], 0)
        self.assertEqual(measured[4:], (0, 1))

    def test_insufficient_reuse_workspace_falls_back(self) -> None:
        _, measured = self.run_case("q8_0", rows=517, experts=4,
                                    top_k=3, baseline_workspace=True)
        self.assertEqual(measured[2], 0)
        self.assertEqual(measured[4:], (0, 1))

    def test_independent_workspaces_can_execute_concurrently(self) -> None:
        if not self.reuse_supported:
            self.skipTest("AVX-512 reuse provider unavailable on this runner")
        threading_lib = BASE_LIB or LIB
        threading_lib.ck_set_num_threads(1)
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(
                    lambda kind: self.run_case(kind, rows=517, experts=4,
                                               top_k=3)[0],
                    ("q5_0", "q8_0"),
                ))
            self.assertEqual(len(results), 2)
        finally:
            threading_lib.ck_set_num_threads(4)


if __name__ == "__main__":
    unittest.main()
