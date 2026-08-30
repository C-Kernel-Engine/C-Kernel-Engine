#!/usr/bin/env python3
"""Bit-exact contract for persistent-pool BF16 GEMV row partitioning."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LIB = ctypes.CDLL(
    os.environ.get("CK_ENGINE_SO", str(ROOT / "build" / "libckernel_engine.so"))
)
FLOAT_P = ctypes.POINTER(ctypes.c_float)
U16_P = ctypes.POINTER(ctypes.c_uint16)
SERIAL = LIB.gemv_bf16
PARALLEL = LIB.gemv_bf16_parallel_dispatch
for kernel in (SERIAL, PARALLEL):
    kernel.argtypes = [FLOAT_P, U16_P, FLOAT_P, ctypes.c_int, ctypes.c_int]
LIB.ck_set_num_threads.argtypes = [ctypes.c_int]
LIB.ck_threadpool_init.argtypes = []
LIB.ck_threadpool_shutdown.argtypes = []


def bf16_bits(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((source >> 16) & np.uint32(1))
    return ((source + rounding) >> 16).astype(np.uint16)


def main() -> int:
    rng = np.random.default_rng(20260829)
    rows, cols = 257, 513
    weights = bf16_bits(rng.standard_normal((rows, cols), dtype=np.float32))
    activation = rng.standard_normal(cols, dtype=np.float32)
    serial = np.empty(rows, dtype=np.float32)
    parallel = np.empty(rows, dtype=np.float32)

    SERIAL(
        serial.ctypes.data_as(FLOAT_P),
        weights.ctypes.data_as(U16_P),
        activation.ctypes.data_as(FLOAT_P),
        rows,
        cols,
    )
    LIB.ck_set_num_threads(4)
    LIB.ck_threadpool_init()
    try:
        PARALLEL(
            parallel.ctypes.data_as(FLOAT_P),
            weights.ctypes.data_as(U16_P),
            activation.ctypes.data_as(FLOAT_P),
            rows,
            cols,
        )
    finally:
        LIB.ck_threadpool_shutdown()

    mismatch = np.flatnonzero(serial.view(np.uint32) != parallel.view(np.uint32))
    if mismatch.size:
        index = int(mismatch[0])
        raise AssertionError(
            f"parallel BF16 GEMV differs at {mismatch.size}/{rows}; first={index} "
            f"serial={serial[index]!r} parallel={parallel[index]!r}"
        )
    print(f"persistent-pool BF16 GEMV: {rows}/{rows} exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
