#!/usr/bin/env python3
"""
Test Q projection specifically

Embedding and RMSNorm pass, but Q projection fails. Let's isolate the issue.

The Q projection is: Q = gemm_nt_q4_k(ln1_out, WQ)

This tests:
1. Is the WQ weight loaded correctly?
2. Is the gemm_nt_q4_k kernel producing correct output?
"""

import ctypes
import numpy as np
from pathlib import Path
import json

BASE_DIR = Path("/home/antshiv/Workspace/C-Kernel-Engine")
MODEL_DIR = Path("/home/antshiv/.cache/ck-engine-v5/models/qwen2.5-3b-instruct-q4_k_m")
LLAMA_DUMP = BASE_DIR / "llama_dump"

# Constants
EMBED_DIM = 2048
NUM_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 128
QK_K = 256
BLOCK_Q4_K_SIZE = 144

# Offsets from header
HEADER_EMBEDDED_INPUT = 0x0A6EC0C0
LAYER0_LN1_OUT = 0x0A6F01C0
LAYER0_WQ = 0x0A6F2200

def load_llama_tensor(name: str) -> np.ndarray:
    path = LLAMA_DUMP / f"{name}.bin"
    if path.exists():
        return np.fromfile(str(path), dtype=np.float32)
    return None

def main():
    print("=" * 70)
    print("Q PROJECTION PARITY TEST")
    print("=" * 70)

    # Load libraries
    print("\nLoading libraries...")
    libggml = ctypes.CDLL(str(BASE_DIR / "llama.cpp/libggml_kernel_test.so"))
    lib = ctypes.CDLL(str(MODEL_DIR / "libmodel.so"))

    # Setup ggml functions
    libggml.test_dequant_q4_k.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]

    # Setup CK functions
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_get_base_ptr.argtypes = []
    lib.ck_model_get_base_ptr.restype = ctypes.c_void_p
    lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.ck_model_embed_tokens.restype = ctypes.c_int
    lib.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_forward.restype = ctypes.c_int

    # Initialize model
    weights_path = str(MODEL_DIR / "weights.bump")
    ret = lib.ck_model_init(weights_path.encode())
    if ret != 0:
        print(f"Failed to init: {ret}")
        return 1

    # Run forward for token 9707 to populate buffers
    token_arr = (ctypes.c_int32 * 1)(9707)
    lib.ck_model_embed_tokens(token_arr, 1)
    lib.ck_model_forward(None)

    base_ptr = lib.ck_model_get_base_ptr()

    # Read CK ln1_out (input to Q projection)
    ln1_out_ptr = ctypes.cast(base_ptr + LAYER0_LN1_OUT, ctypes.POINTER(ctypes.c_float))
    ck_ln1_out = np.ctypeslib.as_array(ln1_out_ptr, shape=(EMBED_DIM,)).copy()

    print(f"\n[1] CK ln1_out (input to Q projection):")
    print(f"    shape={ck_ln1_out.shape}, range=({ck_ln1_out.min():.4f}, {ck_ln1_out.max():.4f})")

    # Load llama attn_norm (should match ln1_out)
    llama_ln1 = load_llama_tensor("attn_norm-0")
    if llama_ln1 is not None:
        print(f"    llama attn_norm: range=({llama_ln1.min():.4f}, {llama_ln1.max():.4f})")
        diff = np.abs(ck_ln1_out - llama_ln1)
        print(f"    max diff: {diff.max():.6f} ({100*diff.max()/np.abs(llama_ln1).max():.2f}%)")

    # Read WQ weight (Q4_K quantized)
    print(f"\n[2] WQ weight analysis:")
    wq_size = NUM_HEADS * HEAD_DIM * EMBED_DIM  # Total elements
    wq_blocks = wq_size // QK_K
    wq_bytes = wq_blocks * BLOCK_Q4_K_SIZE

    print(f"    Shape: [{NUM_HEADS*HEAD_DIM}, {EMBED_DIM}] = {wq_size} elements")
    print(f"    Blocks: {wq_blocks}, Bytes: {wq_bytes}")

    wq_ptr = ctypes.cast(base_ptr + LAYER0_WQ, ctypes.c_void_p)

    # Dequantize first block of WQ to check
    dequant_out = np.zeros(QK_K, dtype=np.float32)
    libggml.test_dequant_q4_k(
        wq_ptr,
        dequant_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        QK_K
    )

    print(f"    First block dequant: range=({dequant_out.min():.4f}, {dequant_out.max():.4f})")
    print(f"    Mean: {dequant_out.mean():.4f}, Std: {dequant_out.std():.4f}")

    # Dequantize entire WQ
    print(f"\n[3] Full WQ dequantization:")
    wq_dequant = np.zeros(wq_size, dtype=np.float32)
    libggml.test_dequant_q4_k(
        wq_ptr,
        wq_dequant.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        wq_size
    )
    print(f"    Full WQ dequant: range=({wq_dequant.min():.4f}, {wq_dequant.max():.4f})")
    print(f"    Mean: {wq_dequant.mean():.6f}, Std: {wq_dequant.std():.4f}")

    # Manual Q projection: Q = ln1_out @ WQ.T
    print(f"\n[4] Manual Q projection:")
    wq_matrix = wq_dequant.reshape(NUM_HEADS * HEAD_DIM, EMBED_DIM)
    q_manual = wq_matrix @ ck_ln1_out  # Shape: [NUM_HEADS * HEAD_DIM]

    print(f"    Manual Q: range=({q_manual.min():.4f}, {q_manual.max():.4f})")

    # Compare with llama Qcur
    llama_qcur = load_llama_tensor("Qcur-0")
    if llama_qcur is not None:
        print(f"    Llama Qcur: range=({llama_qcur.min():.4f}, {llama_qcur.max():.4f})")
        print(f"    Llama Qcur shape: {llama_qcur.shape}")

        if len(llama_qcur) == len(q_manual):
            diff = np.abs(q_manual - llama_qcur)
            print(f"    Max diff: {diff.max():.4f} ({100*diff.max()/np.abs(llama_qcur).max():.2f}%)")
        else:
            print(f"    Shape mismatch! Manual: {q_manual.shape}, Llama: {llama_qcur.shape}")

    # Check if Qcur in llama includes RoPE
    print(f"\n[5] Analysis:")
    if llama_qcur is not None:
        # llama Qcur might be after RoPE, which would explain large values
        llama_mag = np.sqrt(np.mean(llama_qcur**2))
        manual_mag = np.sqrt(np.mean(q_manual**2))
        print(f"    Llama Qcur RMS magnitude: {llama_mag:.4f}")
        print(f"    Manual Q RMS magnitude: {manual_mag:.4f}")
        print(f"    Ratio (llama/manual): {llama_mag/manual_mag:.4f}")

    return 0

if __name__ == "__main__":
    exit(main())
