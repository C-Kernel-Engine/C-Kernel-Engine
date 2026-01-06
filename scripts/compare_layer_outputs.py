#!/usr/bin/env python3
"""
Compare llama.cpp tensor dumps with CK engine outputs

This script:
1. Reads llama.cpp tensor dumps from llama_dump/
2. Runs CK engine and extracts intermediate tensors
3. Compares them layer by layer

Usage:
    python scripts/compare_layer_outputs.py --model-dir /path/to/ck/model --token 9707
"""

import ctypes
import numpy as np
import argparse
from pathlib import Path

def load_llama_tensor(name: str, dump_dir: str = "llama_dump") -> np.ndarray:
    """Load a tensor from llama.cpp dump."""
    path = Path(dump_dir) / f"{name}.bin"
    if not path.exists():
        return None
    data = np.fromfile(str(path), dtype=np.float32)
    return data

def compare_tensors(name: str, llama: np.ndarray, ck: np.ndarray) -> dict:
    """Compare two tensors."""
    if llama is None or ck is None:
        return {"status": "missing", "name": name}

    if llama.shape != ck.shape:
        return {
            "status": "shape_mismatch",
            "name": name,
            "llama_shape": llama.shape,
            "ck_shape": ck.shape
        }

    diff = np.abs(llama - ck)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    rel_err = max_diff / (np.max(np.abs(llama)) + 1e-9)

    passed = rel_err < 0.05  # 5% tolerance

    return {
        "status": "pass" if passed else "fail",
        "name": name,
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
        "rel_err": float(rel_err),
        "llama_range": (float(np.min(llama)), float(np.max(llama))),
        "ck_range": (float(np.min(ck)), float(np.max(ck)))
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Path to CK model directory")
    parser.add_argument("--token", type=int, default=9707, help="Token ID")
    parser.add_argument("--dump-dir", default="llama_dump", help="llama.cpp dump directory")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    print("=" * 70)
    print("LAYER-BY-LAYER PARITY TEST: CK Engine vs llama.cpp")
    print("=" * 70)

    # First, compare final logits
    print("\n[1] Comparing final logits (result_output)...")

    llama_logits = load_llama_tensor("result_output", args.dump_dir)
    if llama_logits is not None:
        print(f"    llama logits: shape={llama_logits.shape}, range=({llama_logits.min():.4f}, {llama_logits.max():.4f})")

    # Load CK and run forward
    lib_path = model_dir / "libmodel.so"
    print(f"\n    Loading CK model from {lib_path}...")
    lib = ctypes.CDLL(str(lib_path))

    # Setup functions
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

    # Initialize
    weights_path = str(model_dir / "weights.bump")
    ret = lib.ck_model_init(weights_path.encode())
    if ret != 0:
        print(f"    Failed to initialize CK model: {ret}")
        return 1

    vocab_size = lib.ck_model_get_vocab_size()
    print(f"    CK vocab size: {vocab_size}")

    # Run forward
    token_arr = (ctypes.c_int32 * 1)(args.token)
    lib.ck_model_embed_tokens(token_arr, 1)
    lib.ck_model_forward(None)

    # Get CK logits
    logits_ptr = lib.ck_model_get_logits()
    ck_logits = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,)).copy()

    print(f"    CK logits: shape={ck_logits.shape}, range=({ck_logits.min():.4f}, {ck_logits.max():.4f})")

    # Compare logits
    if llama_logits is not None:
        result = compare_tensors("logits", llama_logits, ck_logits)
        status_color = "\033[92m" if result["status"] == "pass" else "\033[91m"
        print(f"\n    Logits comparison: {status_color}{result['status'].upper()}\033[0m")
        print(f"      max_diff={result.get('max_diff', 'N/A'):.6f}")
        print(f"      rel_err={result.get('rel_err', 'N/A'):.2%}")

        # Top token comparison
        llama_top1 = np.argmax(llama_logits)
        ck_top1 = np.argmax(ck_logits)
        print(f"\n    llama top-1: token {llama_top1} (logit={llama_logits[llama_top1]:.4f})")
        print(f"    CK top-1:    token {ck_top1} (logit={ck_logits[ck_top1]:.4f})")
        print(f"    Top-1 match: {llama_top1 == ck_top1}")

    # Now compare layer 0 tensors
    print("\n" + "=" * 70)
    print("[2] Comparing Layer 0 intermediate tensors...")
    print("=" * 70)

    # Tensor name mapping: llama.cpp name -> size
    layer0_tensors = [
        ("inp_embd", 2048, "Embedding output"),
        ("attn_norm-0", 2048, "Attention RMSNorm output"),
        ("Qcur-0", 2048, "Q projection (reshaped to n_head*head_dim)"),
        ("Kcur-0", 256, "K projection (n_kv_heads*head_dim)"),
        ("Vcur-0", 256, "V projection (n_kv_heads*head_dim)"),
        ("__fattn__-0", 2048, "Attention output (before O projection)"),
        ("ffn_inp-0", 2048, "FFN input (after attention residual)"),
        ("ffn_norm-0", 2048, "FFN RMSNorm output"),
        ("ffn_gate-0", 11008, "FFN gate projection"),
        ("ffn_up-0", 11008, "FFN up projection"),
        ("ffn_swiglu-0", 11008, "FFN SwiGLU output"),
        ("ffn_out-0", 2048, "FFN down projection output"),
        ("l_out-0", 2048, "Layer output (after MLP residual)"),
        ("result_norm", 2048, "Final RMSNorm output"),
    ]

    print(f"\n    {'Tensor':<20} {'Shape':<12} {'Status':<8} {'Max Diff':<12} {'Rel Err':<10}")
    print("    " + "-" * 70)

    for name, expected_size, desc in layer0_tensors:
        llama_tensor = load_llama_tensor(name, args.dump_dir)
        if llama_tensor is None:
            print(f"    {name:<20} {'N/A':<12} {'SKIP':<8} (not dumped)")
            continue

        # For now, we don't have CK intermediates - just show llama values
        actual_size = llama_tensor.shape[0]
        status = "\033[93mPENDING\033[0m"
        print(f"    {name:<20} {actual_size:<12} {status:<8} llama: [{llama_tensor.min():.3f}, {llama_tensor.max():.3f}]")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if llama_logits is not None:
        if np.argmax(llama_logits) == np.argmax(ck_logits):
            print("\n  Top-1 token MATCHES between llama.cpp and CK!")
            print("  (This is good - basic inference is working)")
        else:
            print("\n  Top-1 token MISMATCH between llama.cpp and CK!")
            print("  Need to investigate layer-by-layer where divergence starts.")
            print("\n  Next steps:")
            print("    1. Add tensor dump hooks to CK engine")
            print("    2. Compare intermediate tensors at each layer")
            print("    3. Find first divergence point")

    return 0

if __name__ == "__main__":
    exit(main())
