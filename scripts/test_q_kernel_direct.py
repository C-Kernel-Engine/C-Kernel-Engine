#!/usr/bin/env python3
"""
Direct test of gemm_nt_q4_k kernel for Q projection

The weights are identical, so the bug must be in the kernel.
This tests the kernel directly with the same input/weights as llama.cpp.
"""

import ctypes
import numpy as np
from pathlib import Path
import json

BASE_DIR = Path("/home/antshiv/Workspace/C-Kernel-Engine")
MODEL_DIR = Path("/home/antshiv/.cache/ck-engine-v5/models/qwen2.5-3b-instruct-q4_k_m")
LLAMA_DUMP = BASE_DIR / "llama_dump"

EMBED_DIM = 2048
NUM_HEADS = 16
HEAD_DIM = 128
QK_K = 256
BLOCK_Q4_K_SIZE = 144

def load_llama_tensor(name: str) -> np.ndarray:
    path = LLAMA_DUMP / f"{name}.bin"
    if path.exists():
        return np.fromfile(str(path), dtype=np.float32)
    return None

def main():
    print("=" * 70)
    print("DIRECT Q KERNEL TEST: gemm_nt_q4_k")
    print("=" * 70)

    # Load libraries
    libck = ctypes.CDLL(str(MODEL_DIR / "libmodel.so"))
    libggml = ctypes.CDLL(str(BASE_DIR / "llama.cpp/libggml_kernel_test.so"))

    # Setup ggml dequant function
    libggml.test_dequant_q4_k.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]

    # Setup CK gemm function
    # void gemm_nt_q4_k(const float *A, const void *B, const float *bias, float *C, int M, int N, int K)
    libck.gemm_nt_q4_k.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # A (input)
        ctypes.c_void_p,                  # B (weight Q4_K)
        ctypes.c_void_p,                  # bias
        ctypes.POINTER(ctypes.c_float),  # C (output)
        ctypes.c_int,                     # M (tokens)
        ctypes.c_int,                     # N (output dim)
        ctypes.c_int,                     # K (input dim)
    ]

    # Load llama attn_norm (same as CK ln1_out)
    llama_ln1 = load_llama_tensor("attn_norm-0")
    if llama_ln1 is None:
        print("ERROR: llama attn_norm-0 not found!")
        return 1

    print(f"\n[1] Input (attn_norm-0):")
    print(f"    shape: {llama_ln1.shape}")
    print(f"    range: ({llama_ln1.min():.4f}, {llama_ln1.max():.4f})")

    # Load WQ from bump
    manifest_path = MODEL_DIR / "weights_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    wq_entry = None
    bq_entry = None
    for e in manifest.get("entries", []):
        if e["name"] == "layer.0.wq":
            wq_entry = e
        if e["name"] == "layer.0.bq":
            bq_entry = e

    bump_path = MODEL_DIR / "weights.bump"
    with open(bump_path, "rb") as f:
        f.seek(wq_entry["file_offset"])
        wq_data = f.read(wq_entry["size"])

        # Load bias if present
        bq_data = None
        if bq_entry:
            f.seek(bq_entry["file_offset"])
            bq_data = np.frombuffer(f.read(bq_entry["size"]), dtype=np.float32).copy()

    n_out = NUM_HEADS * HEAD_DIM  # 2048
    n_in = EMBED_DIM  # 2048

    print(f"\n[2] WQ weight:")
    print(f"    size: {len(wq_data)} bytes")
    if bq_data is not None:
        print(f"    BQ bias: size={len(bq_data)}, range=({bq_data.min():.4f}, {bq_data.max():.4f})")
    else:
        print(f"    BQ bias: None")

    # Run CK gemm_nt_q4_k
    print(f"\n[3] Running CK gemm_nt_q4_k...")

    input_arr = llama_ln1.astype(np.float32)
    output_arr = np.zeros(n_out, dtype=np.float32)

    bias_ptr = bq_data.ctypes.data_as(ctypes.c_void_p) if bq_data is not None else None
    libck.gemm_nt_q4_k(
        input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        wq_data,
        bias_ptr,
        output_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(1),       # M = 1 token
        ctypes.c_int(n_out),   # N = output dim
        ctypes.c_int(n_in),    # K = input dim
    )

    print(f"    CK Q output: range=({output_arr.min():.4f}, {output_arr.max():.4f})")

    # Compare with llama Qcur
    llama_qcur = load_llama_tensor("Qcur-0")
    if llama_qcur is not None:
        print(f"    Llama Qcur:  range=({llama_qcur.min():.4f}, {llama_qcur.max():.4f})")

        # RMS comparison
        ck_rms = np.sqrt(np.mean(output_arr**2))
        llama_rms = np.sqrt(np.mean(llama_qcur**2))
        print(f"\n    CK Q RMS:    {ck_rms:.4f}")
        print(f"    Llama Qcur RMS: {llama_rms:.4f}")
        print(f"    Ratio (llama/CK): {llama_rms/ck_rms:.4f}")

        # Element-wise comparison
        diff = np.abs(output_arr - llama_qcur)
        max_diff = diff.max()
        rel_err = max_diff / (np.abs(llama_qcur).max() + 1e-9)
        print(f"\n    Max absolute diff: {max_diff:.4f}")
        print(f"    Max relative error: {100*rel_err:.2f}%")

        # First few elements
        print(f"\n    First 8 elements:")
        print(f"      CK:    {output_arr[:8]}")
        print(f"      Llama: {llama_qcur[:8]}")

    # Also compute reference via dequant + FP32 matmul
    print(f"\n[4] Reference via FP32 matmul...")
    n_elements = n_out * n_in
    wq_dequant = np.zeros(n_elements, dtype=np.float32)
    libggml.test_dequant_q4_k(
        wq_data,
        wq_dequant.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n_elements
    )

    wq_matrix = wq_dequant.reshape(n_out, n_in)
    ref_q = wq_matrix @ input_arr
    if bq_data is not None:
        ref_q = ref_q + bq_data

    print(f"    FP32 ref Q (with bias): range=({ref_q.min():.4f}, {ref_q.max():.4f})")
    print(f"    FP32 ref Q RMS: {np.sqrt(np.mean(ref_q**2)):.4f}")

    # Compare CK with FP32 reference
    ref_diff = np.abs(output_arr - ref_q)
    print(f"\n    CK vs FP32 ref:")
    print(f"      Max diff: {ref_diff.max():.6f}")
    print(f"      Mean diff: {ref_diff.mean():.6f}")

    return 0

if __name__ == "__main__":
    exit(main())
