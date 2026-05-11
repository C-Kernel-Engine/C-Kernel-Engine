"""Q8_K activation quantizer dispatch consistency.

The Q4_K/Q5_K/Q6_K decode paths all consume Q8_K activations. Keep the
architecture-dispatched quantizers byte-identical to each other so long-context
parity debugging is not polluted by different local SIMD choices.
"""

import ctypes
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UNITS = ROOT / "unittest"
for path in (ROOT, UNITS):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from lib_loader import load_lib

QK_K = 256
BLOCK_Q8_K_SIZE = 4 + 256 + 32


def _bind(fn):
    fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int]
    fn.restype = None
    return fn


def _quantize(fn, row: np.ndarray) -> bytes:
    assert row.dtype == np.float32
    out = ctypes.create_string_buffer((row.size // QK_K) * BLOCK_Q8_K_SIZE)
    fn(row.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
       ctypes.cast(out, ctypes.c_void_p),
       ctypes.c_int(row.size))
    return out.raw


def _edge_row() -> np.ndarray:
    vals = np.zeros(QK_K, dtype=np.float32)
    vals[0] = 1.0
    vals[1] = -1.0
    # Values landing close to integer and half-integer quantized bins.
    for i in range(2, QK_K):
        bin_val = ((i % 255) - 127) + (0.5 if i % 2 else -0.5)
        vals[i] = np.float32(-bin_val / 127.0)
    return vals


def test_q8_k_quantizer_variants_match_dispatch() -> None:
    lib = load_lib("libckernel_engine.so")
    variants = {
        "sse": _bind(lib.quantize_row_q8_k_sse),
        "avx": _bind(lib.quantize_row_q8_k_avx),
        "avx2": _bind(lib.quantize_row_q8_k_avx2),
    }
    dispatch = _bind(lib.quantize_row_q8_k)

    rng = np.random.default_rng(1234)
    rows = [
        np.zeros(QK_K, dtype=np.float32),
        _edge_row(),
        (rng.standard_normal(QK_K).astype(np.float32) * np.float32(0.2)),
        (rng.standard_normal(QK_K * 4).astype(np.float32) * np.float32(1.7)),
    ]

    for row in rows:
        expected = _quantize(dispatch, row)
        for name, fn in variants.items():
            got = _quantize(fn, row)
            assert got == expected, f"{name} Q8_K quantizer differs from dispatch for k={row.size}"


if __name__ == "__main__":
    test_q8_k_quantizer_variants_match_dispatch()
