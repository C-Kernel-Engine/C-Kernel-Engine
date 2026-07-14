#!/usr/bin/env python3
"""Production F16 GEMM parity against llama.cpp's actual mul_mat graph."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / os.environ.get("CK_BUILD_DIR", "build")
LLAMA_ROOT = Path(os.environ.get("CK_LLAMA_CPP_ROOT", ROOT / "llama.cpp"))
LLAMA_BIN = LLAMA_ROOT / "build" / "bin"
M, N, K = 17, 288, 4304


def _load() -> ctypes.CDLL:
    for name in ("libggml-base.so", "libggml.so", "libggml-cpu.so"):
        path = LLAMA_BIN / name
        if not path.exists():
            raise RuntimeError(f"missing llama.cpp library: {path}")
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    ck_path = BUILD / "libckernel_engine.so"
    if not ck_path.exists():
        raise RuntimeError(f"missing CK library: {ck_path}")
    ck = ctypes.CDLL(str(ck_path), mode=ctypes.RTLD_GLOBAL)
    ck.ck_set_strict_parity.argtypes = [ctypes.c_int]
    ck.gemm_nt_f16.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    ck.ck_gemm_nt_f16_ggml_oracle.argtypes = ck.gemm_nt_f16.argtypes
    ck.ck_gemm_nt_f16_ggml_oracle.restype = ctypes.c_int
    return ck


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row = np.arange(M, dtype=np.float32)[:, None]
    col = np.arange(K, dtype=np.float32)[None, :]
    activation = (
        np.sin(col * np.float32(0.013) + row * np.float32(0.17)) * np.float32(0.37)
        + np.cos(col * np.float32(0.0031) - row * np.float32(0.11)) * np.float32(0.09)
    ).astype(np.float32)
    out = np.arange(N, dtype=np.float32)[:, None]
    weights = (
        np.sin(col * np.float32(0.007) + out * np.float32(0.019)) * np.float32(0.12)
        + np.cos(col * np.float32(0.0023) - out * np.float32(0.023)) * np.float32(0.04)
    ).astype(np.float16)
    bias = (np.sin(np.arange(N, dtype=np.float32) * np.float32(0.021)) * np.float32(0.03)).astype(np.float32)
    return np.ascontiguousarray(activation), np.ascontiguousarray(weights), bias


def _run_production(ck: ctypes.CDLL, activation: np.ndarray, weights: np.ndarray,
                    bias: np.ndarray | None) -> np.ndarray:
    output = np.empty((M, N), dtype=np.float32)
    ck.ck_set_strict_parity(0)
    bias_ptr = (
        bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        if bias is not None else None
    )
    ck.gemm_nt_f16(
        activation.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_void_p(weights.ctypes.data),
        bias_ptr,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        M, N, K,
    )
    return output


def _run_oracle(ck: ctypes.CDLL, activation: np.ndarray, weights: np.ndarray,
                bias: np.ndarray | None) -> np.ndarray:
    output = np.empty((M, N), dtype=np.float32)
    bias_ptr = (
        bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        if bias is not None else None
    )
    ok = ck.ck_gemm_nt_f16_ggml_oracle(
        activation.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_void_p(weights.ctypes.data),
        bias_ptr,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        M, N, K,
    )
    if not ok:
        raise RuntimeError("ggml F16 mul_mat oracle is unavailable; refusing CK fallback")
    return output


def _metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {
        "bit_exact": bool(np.array_equal(actual, expected)),
        "different_values": int(np.count_nonzero(actual != expected)),
        "compared_values": int(actual.size),
        "max_abs": float(np.max(delta, initial=0.0)),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
    }


def main() -> int:
    ck = _load()
    activation, weights, bias = _fixture()
    oracle_no_bias = _run_oracle(ck, activation, weights, None)
    production_no_bias = _run_production(ck, activation, weights, None)
    oracle_bias = _run_oracle(ck, activation, weights, bias)
    production_bias = _run_production(ck, activation, weights, bias)
    repeat_bias = _run_production(ck, activation, weights, bias)

    no_bias = _metrics(production_no_bias, oracle_no_bias)
    with_bias = _metrics(production_bias, oracle_bias)
    repeat = _metrics(repeat_bias, production_bias)
    passed = bool(no_bias["bit_exact"] and with_bias["bit_exact"] and repeat["bit_exact"])
    report = {
        "schema": "cke.f16_gemm_llama_contract",
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "shape": {"tokens": M, "outputs": N, "width": K},
        "contract": {
            "weight_storage": "fp16",
            "activation_rounding": "fp16_rne_before_dot",
            "accumulator": "fp32_simd_lanes",
            "horizontal_reduction": "llamafile_avx_movehl_movehdup",
            "output_storage": "fp32",
        },
        "before_bias": no_bias,
        "after_bias": with_bias,
        "thread_determinism": repeat,
    }
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
