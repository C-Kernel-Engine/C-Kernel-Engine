#!/usr/bin/env python3
"""
Layer Sweep Parity Test
=======================

Systematically compare CK engine vs llama.cpp outputs layer by layer.

Approach:
1. Run llama.cpp with tensor dump hack (dumps to llama_dump/)
2. Run CK engine and read intermediate buffers
3. Compare each tensor: embedding, ln1, Q, K, V, attn, FFN, output per layer
"""

import ctypes
import numpy as np
from pathlib import Path
import json
import subprocess
import sys

BASE_DIR = Path("/home/antshiv/Workspace/C-Kernel-Engine")
MODEL_DIR = Path("/home/antshiv/.cache/ck-engine-v5/models/qwen2.5-3b-instruct-q4_k_m")
LLAMA_DUMP = BASE_DIR / "llama_dump"
GGUF_PATH = BASE_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf"

# Model constants
EMBED_DIM = 2048
NUM_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 128
NUM_LAYERS = 36

def load_llama_tensor(name: str) -> np.ndarray:
    """Load tensor from llama.cpp dump."""
    path = LLAMA_DUMP / f"{name}.bin"
    if path.exists():
        return np.fromfile(str(path), dtype=np.float32)
    return None

def run_llama_dump(token: int = 9707):
    """Run llama.cpp with tensor dump."""
    print(f"[1] Running llama.cpp with tensor dump (token={token})...")

    # Clean old dumps
    if LLAMA_DUMP.exists():
        for f in LLAMA_DUMP.glob("*.bin"):
            f.unlink()
    LLAMA_DUMP.mkdir(exist_ok=True)

    # Run llama-cli with our hacked version
    llama_cli = BASE_DIR / "llama.cpp/build/bin/llama-cli"
    cmd = [
        str(llama_cli),
        "-m", str(GGUF_PATH),
        "-p", "Hello",  # Single token prompt
        "-n", "1",
        "--temp", "0",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    # List dumped tensors
    dumps = list(LLAMA_DUMP.glob("*.bin"))
    print(f"    Dumped {len(dumps)} tensors")
    return len(dumps) > 0

def load_ck_model():
    """Load CK model and return library handle."""
    lib = ctypes.CDLL(str(MODEL_DIR / "libmodel.so"))

    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.ck_model_embed_tokens.restype = ctypes.c_int
    lib.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_forward.restype = ctypes.c_int
    lib.ck_model_get_vocab_size.argtypes = []
    lib.ck_model_get_vocab_size.restype = ctypes.c_int

    # Check if parity helpers exist
    try:
        lib.ck_model_get_base_ptr.argtypes = []
        lib.ck_model_get_base_ptr.restype = ctypes.c_void_p
        has_parity = True
    except AttributeError:
        has_parity = False

    weights_path = str(MODEL_DIR / "weights.bump")
    ret = lib.ck_model_init(weights_path.encode())
    if ret != 0:
        print(f"    ERROR: Failed to init CK model: {ret}")
        return None, None, False

    return lib, weights_path, has_parity

def compare_tensor(name: str, ck_tensor: np.ndarray, llama_tensor: np.ndarray,
                   tolerance: float = 0.01) -> dict:
    """Compare two tensors and return metrics."""
    if llama_tensor is None:
        return {"status": "SKIP", "reason": "llama tensor not found"}
    if ck_tensor is None:
        return {"status": "SKIP", "reason": "ck tensor not found"}

    if ck_tensor.shape != llama_tensor.shape:
        return {
            "status": "FAIL",
            "reason": f"shape mismatch: CK={ck_tensor.shape} vs llama={llama_tensor.shape}"
        }

    diff = np.abs(ck_tensor - llama_tensor)
    max_diff = diff.max()
    mean_diff = diff.mean()

    ck_rms = np.sqrt(np.mean(ck_tensor**2))
    llama_rms = np.sqrt(np.mean(llama_tensor**2))
    ratio = llama_rms / (ck_rms + 1e-9)

    rel_err = max_diff / (np.abs(llama_tensor).max() + 1e-9)

    status = "PASS" if rel_err < tolerance else "FAIL"

    return {
        "status": status,
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
        "rel_err": float(rel_err),
        "ck_rms": float(ck_rms),
        "llama_rms": float(llama_rms),
        "ratio": float(ratio),
    }

def print_result(name: str, result: dict):
    """Print comparison result."""
    status = result["status"]
    if status == "SKIP":
        print(f"  {name}: SKIP ({result['reason']})")
    elif status == "PASS":
        print(f"  {name}: \033[92mPASS\033[0m (rel_err={100*result['rel_err']:.2f}%, ratio={result['ratio']:.4f})")
    else:
        print(f"  {name}: \033[91mFAIL\033[0m (rel_err={100*result['rel_err']:.2f}%, ratio={result['ratio']:.4f})")
        print(f"      CK  RMS={result['ck_rms']:.4f}, range unknown")
        print(f"      Llama RMS={result['llama_rms']:.4f}")

def main():
    print("=" * 70)
    print("LAYER SWEEP PARITY TEST: CK Engine vs llama.cpp")
    print("=" * 70)

    # Step 1: Run llama.cpp dump
    if not run_llama_dump():
        print("ERROR: llama.cpp dump failed")
        return 1

    # Step 2: Load CK model
    print(f"\n[2] Loading CK engine...")
    lib, weights_path, has_parity = load_ck_model()
    if lib is None:
        return 1

    print(f"    Parity helpers: {'YES' if has_parity else 'NO'}")

    # Step 3: Run CK forward
    print(f"\n[3] Running CK forward pass...")
    token_arr = (ctypes.c_int32 * 1)(9707)  # Same token as llama
    lib.ck_model_embed_tokens(token_arr, 1)

    vocab_size = lib.ck_model_get_vocab_size()
    logits_out = (ctypes.c_float * vocab_size)()
    lib.ck_model_forward(logits_out)
    ck_logits = np.ctypeslib.as_array(logits_out, shape=(vocab_size,)).copy()

    print(f"    CK logits: range=({ck_logits.min():.4f}, {ck_logits.max():.4f})")
    print(f"    CK top-5 tokens: {np.argsort(ck_logits)[-5:][::-1]}")

    # Step 4: Compare logits
    print(f"\n[4] Comparing final logits...")
    llama_logits = load_llama_tensor("result_output-0")
    if llama_logits is not None:
        result = compare_tensor("logits", ck_logits, llama_logits)
        print_result("logits", result)
        print(f"    Llama top-5 tokens: {np.argsort(llama_logits)[-5:][::-1]}")
    else:
        print("    Llama logits not found in dump")

    # Step 5: Layer-by-layer comparison (if parity helpers available)
    print(f"\n[5] Layer-by-layer comparison...")

    # Map llama tensor names to what we expect
    layer_tensors = [
        ("attn_norm", "RMSNorm attn input"),
        ("Qcur", "Q projection"),
        ("Kcur", "K projection"),
        ("Vcur", "V projection"),
        ("attn_out", "Attention output"),
        ("ffn_inp", "FFN input (residual1)"),
        ("ffn_norm", "RMSNorm FFN input"),
        ("ffn_out", "FFN output"),
        ("l_out", "Layer output"),
    ]

    # Check which layers have dumps
    layers_found = set()
    for f in LLAMA_DUMP.glob("*.bin"):
        name = f.stem
        if "-" in name:
            layer = name.split("-")[-1]
            if layer.isdigit():
                layers_found.add(int(layer))

    print(f"    Found dumps for layers: {sorted(layers_found)}")

    # Compare each layer's tensors
    for layer in sorted(layers_found)[:5]:  # First 5 layers
        print(f"\n  === Layer {layer} ===")
        for tensor_name, desc in layer_tensors:
            llama_name = f"{tensor_name}-{layer}"
            llama_data = load_llama_tensor(llama_name)
            if llama_data is not None:
                print(f"    {desc} ({tensor_name}): shape={llama_data.shape}, "
                      f"range=({llama_data.min():.4f}, {llama_data.max():.4f}), "
                      f"rms={np.sqrt(np.mean(llama_data**2)):.4f}")

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print("To enable full layer comparison, rebuild libmodel.so with parity helpers:")
    print("  Add ck_model_get_base_ptr() and ck_model_get_tensor_ptr() to model.c")

    return 0

if __name__ == "__main__":
    sys.exit(main())
