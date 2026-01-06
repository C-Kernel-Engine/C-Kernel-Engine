#!/usr/bin/env python3
"""
Layer 0 Parity Test: Compare CK vs llama.cpp intermediate tensors

This script:
1. Loads llama.cpp tensor dumps
2. Loads CK model and runs forward pass
3. Reads CK intermediate tensors from memory layout
4. Compares each tensor to find the divergence point

Usage:
    python scripts/test_layer0_parity.py --model-dir /path/to/ck/model --token 9707
"""

import ctypes
import numpy as np
import argparse
import json
from pathlib import Path

# Model config from generated header
EMBED_DIM = 2048
NUM_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 128
INTERMEDIATE = 11008
VOCAB_SIZE = 151936
MAX_SEQ_LEN = 32768

# Header offsets (from generated_qwen2_decode.h)
HEADER_TOKEN_EMB = 0x00000080
HEADER_EMBEDDED_INPUT = 0x0A6EC0C0

# Layer 0 offsets
LAYER0_OFFSETS = {
    "ln1_gamma": 0x0A6EE180,
    "ln1_out": 0x0A6F01C0,
    "wq": 0x0A6F2200,
    "wk": 0x0A934280,
    "wv": 0x0A97C700,
    "q": 0x0A9E5B80,
    "k": 0x0A9E7BC0,
    "v": 0x0C9E7C00,
    "attn_out": 0x0E9E7C40,
    "wo": 0x0E9E9C80,
    "proj_tmp": 0x0EC29CC0,
    "residual1": 0x0EC2DD40,
    "ln2_gamma": 0x0EC2FD80,
    "ln2_out": 0x0EC31DC0,
    "w1": 0x0EC33E00,
    "fc1_out": 0x10463E40,
    "swiglu_out": 0x10479680,
    "w2": 0x104842C0,
    "mlp_out": 0x11627300,
    "output": 0x11629340,
}

# Footer offsets (approximate - need to calculate from layout)
FOOTER_FINAL_LN_WEIGHT = None  # Will be calculated
FOOTER_FINAL_OUTPUT = None
FOOTER_LM_HEAD = None
FOOTER_LOGITS = None

def load_llama_tensor(name: str, dump_dir: str = "llama_dump") -> np.ndarray:
    """Load a tensor from llama.cpp dump."""
    path = Path(dump_dir) / f"{name}.bin"
    if not path.exists():
        return None
    data = np.fromfile(str(path), dtype=np.float32)
    return data

def read_tensor_from_model(lib, offset: int, num_elements: int) -> np.ndarray:
    """Read a tensor from the CK model's memory buffer."""
    # Get base pointer via a helper function
    if not hasattr(lib, 'ck_model_get_base_ptr'):
        return None

    base_ptr = lib.ck_model_get_base_ptr()
    if not base_ptr:
        return None

    # Cast to correct type and read
    ptr = ctypes.cast(base_ptr + offset, ctypes.POINTER(ctypes.c_float))
    return np.ctypeslib.as_array(ptr, shape=(num_elements,)).copy()

def compare_tensors(name: str, llama: np.ndarray, ck: np.ndarray, tol: float = 0.05) -> dict:
    """Compare two tensors."""
    if llama is None:
        return {"name": name, "status": "llama_missing"}
    if ck is None:
        return {"name": name, "status": "ck_missing"}

    # Handle different shapes
    llama_flat = llama.flatten()
    ck_flat = ck.flatten()

    min_len = min(len(llama_flat), len(ck_flat))
    llama_flat = llama_flat[:min_len]
    ck_flat = ck_flat[:min_len]

    diff = np.abs(llama_flat - ck_flat)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Relative error
    max_abs = max(np.max(np.abs(llama_flat)), 1e-9)
    rel_err = max_diff / max_abs

    passed = rel_err < tol

    return {
        "name": name,
        "status": "pass" if passed else "FAIL",
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
        "rel_err": float(rel_err),
        "llama_range": (float(np.min(llama_flat)), float(np.max(llama_flat))),
        "ck_range": (float(np.min(ck_flat)), float(np.max(ck_flat))),
        "size": min_len,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Path to CK model directory")
    parser.add_argument("--token", type=int, default=9707, help="Token ID")
    parser.add_argument("--dump-dir", default="llama_dump", help="llama.cpp dump directory")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    print("=" * 70)
    print("LAYER 0 PARITY TEST: CK Engine vs llama.cpp")
    print("=" * 70)
    print(f"\nToken ID: {args.token}")
    print(f"Model dir: {model_dir}")
    print(f"Dump dir: {args.dump_dir}")

    # Load CK library
    lib_path = model_dir / "libmodel.so"
    print(f"\nLoading CK library: {lib_path}")
    lib = ctypes.CDLL(str(lib_path))

    # Check if we have the parity helper function
    try:
        lib.ck_model_get_base_ptr.argtypes = []
        lib.ck_model_get_base_ptr.restype = ctypes.c_void_p
        has_base_ptr = True
    except:
        has_base_ptr = False
        print("WARNING: ck_model_get_base_ptr not available")

    # Setup standard functions
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_get_vocab_size.argtypes = []
    lib.ck_model_get_vocab_size.restype = ctypes.c_int
    lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.ck_model_embed_tokens.restype = ctypes.c_int
    lib.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_forward.restype = ctypes.c_int
    lib.ck_model_get_logits.argtypes = []
    lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)

    # Initialize model
    weights_path = str(model_dir / "weights.bump")
    ret = lib.ck_model_init(weights_path.encode())
    if ret != 0:
        print(f"Failed to initialize CK model: {ret}")
        return 1

    vocab_size = lib.ck_model_get_vocab_size()
    print(f"CK vocab size: {vocab_size}")

    # Run forward pass
    print(f"\nRunning CK forward pass for token {args.token}...")
    token_arr = (ctypes.c_int32 * 1)(args.token)
    lib.ck_model_embed_tokens(token_arr, 1)
    lib.ck_model_forward(None)

    # Get CK logits
    logits_ptr = lib.ck_model_get_logits()
    ck_logits = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,)).copy()

    # Compare with llama.cpp
    print("\n" + "=" * 70)
    print("COMPARISON: Embedding")
    print("=" * 70)

    llama_embd = load_llama_tensor("inp_embd", args.dump_dir)
    if llama_embd is not None:
        print(f"  llama inp_embd: shape={llama_embd.shape}, range=({llama_embd.min():.6f}, {llama_embd.max():.6f})")

        # CK embedding is at HEADER_EMBEDDED_INPUT
        if has_base_ptr:
            ck_embd = read_tensor_from_model(lib, HEADER_EMBEDDED_INPUT, EMBED_DIM)
            if ck_embd is not None:
                print(f"  CK embedded:    shape={ck_embd.shape}, range=({ck_embd.min():.6f}, {ck_embd.max():.6f})")
                result = compare_tensors("embedding", llama_embd, ck_embd)
                print(f"  Status: {result['status']}, rel_err={result.get('rel_err', 0):.2%}")
            else:
                print("  CK embedded: could not read")
        else:
            print("  CK embedded: base_ptr not available")

    # Compare layer 0 tensors
    print("\n" + "=" * 70)
    print("COMPARISON: Layer 0 Intermediate Tensors")
    print("=" * 70)

    # Mapping: llama.cpp tensor name -> (CK offset name, size)
    tensor_mapping = [
        ("attn_norm-0", "ln1_out", EMBED_DIM),
        ("Qcur-0", "q", NUM_HEADS * HEAD_DIM),  # Note: q is after projection
        ("Kcur-0", "k", NUM_KV_HEADS * HEAD_DIM),  # k is per-position
        ("Vcur-0", "v", NUM_KV_HEADS * HEAD_DIM),
        ("__fattn__-0", "attn_out", NUM_HEADS * HEAD_DIM),
        ("ffn_inp-0", "residual1", EMBED_DIM),
        ("ffn_norm-0", "ln2_out", EMBED_DIM),
        ("ffn_gate-0", "fc1_out", INTERMEDIATE * 2),  # gate+up combined?
        ("ffn_up-0", None, INTERMEDIATE),  # May be combined with gate
        ("ffn_swiglu-0", "swiglu_out", INTERMEDIATE),
        ("ffn_out-0", "mlp_out", EMBED_DIM),
        ("l_out-0", "output", EMBED_DIM),
    ]

    first_fail = None
    results = []

    for llama_name, ck_name, size in tensor_mapping:
        llama_tensor = load_llama_tensor(llama_name, args.dump_dir)

        if ck_name and has_base_ptr and ck_name in LAYER0_OFFSETS:
            ck_tensor = read_tensor_from_model(lib, LAYER0_OFFSETS[ck_name], size)
        else:
            ck_tensor = None

        result = compare_tensors(llama_name, llama_tensor, ck_tensor)
        results.append(result)

        status = result["status"]
        if status == "pass":
            status_str = "\033[92mPASS\033[0m"
        elif status == "FAIL":
            status_str = "\033[91mFAIL\033[0m"
            if first_fail is None:
                first_fail = llama_name
        else:
            status_str = f"\033[93m{status}\033[0m"

        llama_range = result.get("llama_range", ("?", "?"))
        ck_range = result.get("ck_range", ("?", "?"))

        print(f"  {llama_name:<15} {status_str:>12}  rel_err={result.get('rel_err', 0):.2%}")
        print(f"    llama: [{llama_range[0]:.4f}, {llama_range[1]:.4f}]")
        if ck_tensor is not None:
            print(f"    CK:    [{ck_range[0]:.4f}, {ck_range[1]:.4f}]")

    # Final logits comparison
    print("\n" + "=" * 70)
    print("COMPARISON: Final Output")
    print("=" * 70)

    llama_logits = load_llama_tensor("result_output", args.dump_dir)
    if llama_logits is not None:
        result = compare_tensors("logits", llama_logits, ck_logits)
        print(f"  Logits: {result['status']}, rel_err={result.get('rel_err', 0):.2%}")
        print(f"    llama top-1: token {np.argmax(llama_logits)} (logit={np.max(llama_logits):.4f})")
        print(f"    CK top-1:    token {np.argmax(ck_logits)} (logit={np.max(ck_logits):.4f})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if first_fail:
        print(f"\n  First divergence detected at: {first_fail}")
        print(f"\n  This means the bug is likely in or before the {first_fail} computation.")
    elif not has_base_ptr:
        print("\n  Cannot read CK intermediate tensors without ck_model_get_base_ptr().")
        print("  Need to add this function to the generated model.c")
    else:
        print("\n  All layer 0 tensors match! Bug may be in later layers.")

    return 0

if __name__ == "__main__":
    exit(main())
