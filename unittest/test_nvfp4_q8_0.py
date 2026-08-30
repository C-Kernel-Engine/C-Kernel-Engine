#!/usr/bin/env python3
"""NVFP4 packed storage and Q8_0 CPU projection contracts."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / os.environ.get("CK_BUILD_DIR", "build")
QK_NVFP4 = 64
NVFP4_BLOCK_BYTES = 36
QK8_0 = 32
Q8_0_BLOCK_BYTES = 34
E2M1_X2 = np.array(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
    dtype=np.int8,
)


def ue4m3_to_f32(value: int) -> float:
    value &= 0x7F
    if value in (0, 0x7F):
        return 0.0
    exponent = (value >> 3) & 0xF
    mantissa = value & 0x7
    if exponent == 0:
        return float(mantissa) * 2.0**-10
    return (1.0 + float(mantissa) / 8.0) * 2.0 ** (exponent - 8)


def packed_fixture(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0xC0A4)
    blocks = rows * cols // QK_NVFP4
    raw = np.empty((blocks, NVFP4_BLOCK_BYTES), dtype=np.uint8)
    scale_codes = np.array([0x01, 0x08, 0x17, 0x38, 0x47, 0x6E], dtype=np.uint8)
    raw[:, :4] = rng.choice(scale_codes, size=(blocks, 4))
    raw[:, 4:] = rng.integers(0, 256, size=(blocks, 32), dtype=np.uint8)
    global_scales = np.resize(
        np.array([0.25, 1.0, 1.75, 0.03125], dtype=np.float32), rows
    )
    return np.ascontiguousarray(raw.reshape(rows, -1)), global_scales


def dequant_python(row: np.ndarray, cols: int, global_scale: float) -> np.ndarray:
    output = np.empty(cols, dtype=np.float32)
    blocks = row.reshape(-1, NVFP4_BLOCK_BYTES)
    for block_index, block in enumerate(blocks):
        for sub in range(4):
            scale = np.float32(ue4m3_to_f32(int(block[sub])) * global_scale)
            packed = block[4 + sub * 8 : 4 + (sub + 1) * 8]
            base = block_index * QK_NVFP4 + sub * 16
            output[base : base + 8] = E2M1_X2[packed & 0x0F].astype(np.float32) * scale
            output[base + 8 : base + 16] = E2M1_X2[packed >> 4].astype(np.float32) * scale
    return output


def load_ck() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(BUILD / "libckernel_engine.so"))
    f32p = ctypes.POINTER(ctypes.c_float)
    lib.ck_ue4m3_to_fp32.argtypes = [ctypes.c_uint8]
    lib.ck_ue4m3_to_fp32.restype = ctypes.c_float
    lib.dequantize_row_nvfp4.argtypes = [
        ctypes.c_void_p, f32p, ctypes.c_int, ctypes.c_float,
    ]
    lib.vec_dot_nvfp4_q8_0_ref.argtypes = [
        ctypes.c_int, f32p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float,
    ]
    lib.vec_dot_nvfp4_q8_0.argtypes = list(lib.vec_dot_nvfp4_q8_0_ref.argtypes)
    lib.gemv_nvfp4_q8_0.argtypes = [
        f32p, ctypes.c_void_p, f32p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int,
    ]
    lib.quantize_row_q8_0.argtypes = [f32p, ctypes.c_void_p, ctypes.c_int]
    lib.moe_swiglu_nvfp4_workspace_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.moe_swiglu_nvfp4_workspace_bytes.restype = ctypes.c_size_t
    lib.moe_swiglu_expert_forward_nvfp4_workspace.argtypes = [
        f32p, ctypes.POINTER(ctypes.c_int), f32p,
        ctypes.c_void_p, f32p, ctypes.c_void_p, f32p, ctypes.c_void_p, f32p,
        f32p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
    ]
    lib.moe_swiglu_expert_forward_nvfp4_workspace.restype = ctypes.c_int
    lib.moe_swiglu_shared_forward_nvfp4_workspace.argtypes = [
        f32p, f32p,
        ctypes.c_void_p, f32p, ctypes.c_void_p, f32p, ctypes.c_void_p, f32p,
        f32p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_void_p, ctypes.c_size_t,
    ]
    lib.moe_swiglu_shared_forward_nvfp4_workspace.restype = ctypes.c_int
    return lib


def load_llama() -> ctypes.CDLL | None:
    root_value = os.environ.get("CK_LLAMA_CPP_ROOT")
    if not root_value:
        default = Path.home() / ".cache/ck-engine/llamacpp-rolling-full/llama.cpp"
        if not default.exists():
            return None
        root = default
    else:
        root = Path(root_value)
    bin_dir = root / "build/bin"
    try:
        for name in ("libggml-base.so", "libggml.so"):
            ctypes.CDLL(str(bin_dir / name), mode=ctypes.RTLD_GLOBAL)
        cpu = ctypes.CDLL(str(bin_dir / "libggml-cpu.so"))
    except OSError:
        return None
    if not hasattr(cpu, "ggml_vec_dot_nvfp4_q8_0"):
        return None
    cpu.ggml_vec_dot_nvfp4_q8_0.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_int,
    ]
    cpu.ggml_cpu_init.argtypes = []
    cpu.ggml_cpu_init()
    return cpu


def main() -> int:
    ck = load_ck()
    llama = load_llama()

    for code in range(0x80):
        got = float(ck.ck_ue4m3_to_fp32(code))
        expected = ue4m3_to_f32(code)
        if got != expected:
            raise AssertionError(f"UE4M3 0x{code:02x}: {got} != {expected}")

    rows, cols = 4, 256
    packed, global_scales = packed_fixture(rows, cols)
    dequantized = np.empty((rows, cols), dtype=np.float32)
    for row in range(rows):
        ck.dequantize_row_nvfp4(
            ctypes.c_void_p(packed[row].ctypes.data),
            dequantized[row].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            cols, ctypes.c_float(global_scales[row]),
        )
        expected = dequant_python(packed[row], cols, float(global_scales[row]))
        np.testing.assert_array_equal(dequantized[row], expected)

    phase = np.arange(cols, dtype=np.float32)
    activation = np.ascontiguousarray(
        np.sin(phase * np.float32(0.031)) * np.float32(1.7)
        + np.cos(phase * np.float32(0.007)) * np.float32(0.2),
        dtype=np.float32,
    )
    q8 = np.empty((cols // QK8_0) * Q8_0_BLOCK_BYTES, dtype=np.uint8)
    ck.quantize_row_q8_0(
        activation.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_void_p(q8.ctypes.data), cols,
    )

    selected = np.empty(rows, dtype=np.float32)
    reference = np.empty(rows, dtype=np.float32)
    for row in range(rows):
        ck.vec_dot_nvfp4_q8_0_ref(
            cols, reference[row : row + 1].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_void_p(packed[row].ctypes.data), ctypes.c_void_p(q8.ctypes.data),
            ctypes.c_float(global_scales[row]),
        )
        ck.vec_dot_nvfp4_q8_0(
            cols, selected[row : row + 1].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_void_p(packed[row].ctypes.data), ctypes.c_void_p(q8.ctypes.data),
            ctypes.c_float(global_scales[row]),
        )
    np.testing.assert_allclose(selected, reference, rtol=2e-6, atol=2e-6)

    gemv = np.empty(rows, dtype=np.float32)
    ck.gemv_nvfp4_q8_0(
        gemv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_void_p(packed.ctypes.data),
        global_scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_void_p(q8.ctypes.data), rows, cols,
    )
    np.testing.assert_array_equal(gemv, selected)

    # Exercise routed and shared expert composition with expert-major storage.
    hidden_dim = intermediate_dim = 64
    n_experts = 2
    expert_rows = n_experts * intermediate_dim
    gate, gate_scales = packed_fixture(expert_rows, hidden_dim)
    up, up_scales = packed_fixture(expert_rows, hidden_dim)
    down, down_scales = packed_fixture(n_experts * hidden_dim, intermediate_dim)
    hidden = np.ascontiguousarray(
        np.sin(np.arange(hidden_dim, dtype=np.float32) * np.float32(0.13))
    )
    indices = np.array([1], dtype=np.int32)
    routes = np.array([0.75], dtype=np.float32)
    workspace_bytes = int(
        ck.moe_swiglu_nvfp4_workspace_bytes(hidden_dim, intermediate_dim)
    )
    workspace = np.empty(workspace_bytes, dtype=np.uint8)
    routed = np.empty(hidden_dim, dtype=np.float32)
    rc = ck.moe_swiglu_expert_forward_nvfp4_workspace(
        hidden.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        routes.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_void_p(gate.ctypes.data),
        gate_scales.reshape(n_experts, intermediate_dim)[:, 0].ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        ),
        ctypes.c_void_p(up.ctypes.data),
        up_scales.reshape(n_experts, intermediate_dim)[:, 0].ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        ),
        ctypes.c_void_p(down.ctypes.data),
        down_scales.reshape(n_experts, hidden_dim)[:, 0].ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        ),
        routed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        1, hidden_dim, intermediate_dim, n_experts, 1,
        ctypes.c_void_p(workspace.ctypes.data), workspace_bytes,
    )
    assert rc == 0
    assert np.isfinite(routed).all()
    assert np.linalg.norm(routed) > 0.0

    shared = np.empty(hidden_dim, dtype=np.float32)
    rc = ck.moe_swiglu_shared_forward_nvfp4_workspace(
        hidden.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        routed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_void_p(gate.ctypes.data),
        gate_scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_void_p(up.ctypes.data),
        up_scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_void_p(down.ctypes.data),
        down_scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        shared.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        1, hidden_dim, intermediate_dim,
        ctypes.c_float(0.5),
        ctypes.c_void_p(workspace.ctypes.data), workspace_bytes,
    )
    assert rc == 0
    assert np.isfinite(shared).all()
    assert not np.array_equal(shared, routed)

    llama_status = "skip"
    if llama is not None:
        llama_outputs = np.empty(rows, dtype=np.float32)
        for row in range(rows):
            value = ctypes.c_float()
            llama.ggml_vec_dot_nvfp4_q8_0(
                cols, ctypes.byref(value), 0,
                ctypes.c_void_p(packed[row].ctypes.data), 0,
                ctypes.c_void_p(q8.ctypes.data), 0, 1,
            )
            llama_outputs[row] = value.value * global_scales[row]
        np.testing.assert_allclose(selected, llama_outputs, rtol=2e-6, atol=2e-6)
        llama_status = "pass"

    print(f"NVFP4 storage/dequant/dot/gemv: PASS; llama.cpp parity: {llama_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
