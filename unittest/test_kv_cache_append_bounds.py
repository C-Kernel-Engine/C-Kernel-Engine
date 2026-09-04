"""KV-cache append/read bounds tests at valid rows 256 and 257, capacity 1034.

Hardened scheduling-extent rule: scheduling may use a rounded/physical extent
(here capacity S_max = 1034 rows), but cache providers must never read or write
beyond the valid KV rows declared by the append index. These tests append at
the boundary rows 256 and 257 of a 1034-row physical cache and verify:

- the appended rows round-trip exactly (fp32, fp16 and bf16 storage contracts),
- rows outside the valid append set keep their sentinel bytes (no out-of-bounds
  or strided writes into unused capacity or neighboring heads).

Run: python unittest/test_kv_cache_append_bounds.py
"""

import ctypes
import unittest

import numpy as np

from lib_loader import load_lib


lib = load_lib("libckernel_engine.so")

CAPACITY = 1034
VALID_ROWS = (256, 257)
NUM_KV_HEADS = 4
HEAD_DIM = 128


def _make_src(values_offset=0.0):
    """Exactly fp16/bf16-representable source values (small integers / .5 steps)."""
    base = (np.arange(NUM_KV_HEADS * HEAD_DIM, dtype=np.float32) % 64) * 0.5 - 16.0
    return (base + values_offset).astype(np.float32)


def _fp16_roundtrip(x):
    return x.astype(np.float16).astype(np.float32)


def _bf16_roundtrip(x):
    # bf16 keeps the top 16 bits of fp32; values chosen are exact either way.
    u = x.view(np.uint32)
    return (u & np.uint32(0xFFFF0000)).view(np.float32)


def _readback_fp32(block):
    return block.astype(np.float32)


def _readback_f16(block):
    return block.view(np.float16).astype(np.float32)


def _readback_bf16(block):
    return (block.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


class TestKvCacheAppendBounds(unittest.TestCase):
    def _fresh_cache(self, storage):
        if storage == "fp32":
            return np.full((NUM_KV_HEADS, CAPACITY, HEAD_DIM), -7777.0, dtype=np.float32)
        # uint16 storage: fill with a sentinel bit pattern.
        return np.full((NUM_KV_HEADS, CAPACITY, HEAD_DIM), 0xBEEF, dtype=np.uint16)

    def _check_store(self, fn_name, storage, roundtrip, readback):
        fn = getattr(lib, fn_name)
        fn.restype = None
        cache_k = self._fresh_cache(storage)
        cache_v = self._fresh_cache(storage)
        sentinel = cache_k[0, 0, 0]
        for pos in VALID_ROWS:
            # Small offsets keep every value exactly representable in fp16/bf16.
            src_k = _make_src(float(pos - VALID_ROWS[0]) * 0.25)
            src_v = _make_src(float(pos - VALID_ROWS[0]) * 0.25 + 0.125)
            fn(
                cache_k.ctypes.data_as(ctypes.c_void_p),
                cache_v.ctypes.data_as(ctypes.c_void_p),
                src_k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                src_v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                0,  # layer
                pos,
                NUM_KV_HEADS,
                HEAD_DIM,
                CAPACITY,
            )
            np.testing.assert_array_equal(
                readback(cache_k[:, pos, :]), roundtrip(src_k.reshape(NUM_KV_HEADS, HEAD_DIM))
            )
            np.testing.assert_array_equal(
                readback(cache_v[:, pos, :]), roundtrip(src_v.reshape(NUM_KV_HEADS, HEAD_DIM))
            )
        # No writes outside the valid rows: all other capacity keeps the sentinel.
        mask = np.ones(CAPACITY, dtype=bool)
        mask[list(VALID_ROWS)] = False
        self.assertTrue(np.all(cache_k[:, mask, :] == sentinel))
        self.assertTrue(np.all(cache_v[:, mask, :] == sentinel))

    def _check_batch(self, fn_name, storage, roundtrip, readback):
        fn = getattr(lib, fn_name)
        fn.restype = None
        num_tokens = 2  # rows 256 and 257
        cache_k = self._fresh_cache(storage)
        cache_v = self._fresh_cache(storage)
        sentinel = cache_k[0, 0, 0]
        # Compact head-major source: [head, token, dim]
        src_k = np.stack([_make_src(0.25 * t).reshape(NUM_KV_HEADS, HEAD_DIM) for t in range(num_tokens)], axis=1)
        src_v = np.stack([_make_src(0.25 * t + 0.125).reshape(NUM_KV_HEADS, HEAD_DIM) for t in range(num_tokens)], axis=1)
        fn(
            cache_k.ctypes.data_as(ctypes.c_void_p),
            cache_v.ctypes.data_as(ctypes.c_void_p),
            np.ascontiguousarray(src_k).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            np.ascontiguousarray(src_v).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            VALID_ROWS[0],  # start_pos
            num_tokens,
            NUM_KV_HEADS,
            HEAD_DIM,
            CAPACITY,
        )
        for t, pos in enumerate(VALID_ROWS):
            np.testing.assert_array_equal(readback(cache_k[:, pos, :]), roundtrip(src_k[:, t, :]))
            np.testing.assert_array_equal(readback(cache_v[:, pos, :]), roundtrip(src_v[:, t, :]))
        mask = np.ones(CAPACITY, dtype=bool)
        mask[list(VALID_ROWS)] = False
        self.assertTrue(np.all(cache_k[:, mask, :] == sentinel))
        self.assertTrue(np.all(cache_v[:, mask, :] == sentinel))

    def test_store_fp32(self):
        self._check_store("kv_cache_store", "fp32", lambda x: x, _readback_fp32)

    def test_store_f16(self):
        self._check_store("kv_cache_store_f16", "f16", _fp16_roundtrip, _readback_f16)

    def test_store_bf16(self):
        self._check_store("kv_cache_store_bf16", "bf16", _bf16_roundtrip, _readback_bf16)

    def test_store_batch_fp32(self):
        self._check_batch("kv_cache_store_batch_f32", "fp32", lambda x: x, _readback_fp32)

    def test_store_batch_f16(self):
        self._check_batch("kv_cache_store_batch_f16", "f16", _fp16_roundtrip, _readback_f16)

    def test_store_batch_bf16(self):
        self._check_batch("kv_cache_store_batch_bf16", "bf16", _bf16_roundtrip, _readback_bf16)


if __name__ == "__main__":
    unittest.main()
