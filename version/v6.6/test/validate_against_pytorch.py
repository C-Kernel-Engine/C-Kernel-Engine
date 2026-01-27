#!/usr/bin/env python3
"""
Validate v6.5 and v6.6 against PyTorch/HuggingFace reference.

Determines which implementation is closer to the ground truth.
"""

import ctypes
import numpy as np
import os
import sys
from pathlib import Path

V66_DIR = Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF"
V65_DIR = Path.home() / ".cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF"

VOCAB_SIZE = 151936


def run_ck_model(model_dir, token):
    """Run CK model and return logits."""
    engine_path = model_dir / "libckernel_engine.so"
    if engine_path.exists():
        ctypes.CDLL(str(engine_path), mode=ctypes.RTLD_GLOBAL)

    lib_path = model_dir / "ck-kernel-inference.so"
    if not lib_path.exists():
        lib_path = model_dir / "libmodel.so"

    lib = ctypes.CDLL(str(lib_path))

    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_free.argtypes = []
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    lib.ck_model_get_vocab_size.argtypes = []
    lib.ck_model_get_vocab_size.restype = ctypes.c_int

    weights_path = model_dir / "weights.bump"
    ret = lib.ck_model_init(str(weights_path).encode())
    if ret != 0:
        return None

    vocab_size = lib.ck_model_get_vocab_size()
    output = (ctypes.c_float * vocab_size)()
    lib.ck_model_decode(token, output)

    logits = np.array(output[:], dtype=np.float32)
    lib.ck_model_free()
    return logits


def run_pytorch(token):
    """Run PyTorch/HuggingFace model and return logits."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("PyTorch/transformers not available")
        return None

    model_name = "Qwen/Qwen2-0.5B-Instruct"
    print(f"Loading {model_name}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use FP32 for fair comparison
        trust_remote_code=True
    )
    model.eval()

    input_ids = torch.tensor([[token]], dtype=torch.long)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, 0].cpu().numpy()

    return logits


def compare_logits(name1, logits1, name2, logits2):
    """Compare two logit arrays."""
    if logits1 is None or logits2 is None:
        return None

    diff = np.abs(logits1 - logits2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Correlation
    corr = np.corrcoef(logits1, logits2)[0, 1]

    # Top-5 agreement
    top5_1 = set(np.argsort(logits1)[-5:])
    top5_2 = set(np.argsort(logits2)[-5:])
    top5_overlap = len(top5_1 & top5_2)

    # Argmax match
    argmax_match = np.argmax(logits1) == np.argmax(logits2)

    print(f"\n{name1} vs {name2}:")
    print(f"  Max diff: {max_diff:.4f}")
    print(f"  Mean diff: {mean_diff:.4f}")
    print(f"  Correlation: {corr:.6f}")
    print(f"  Top-5 overlap: {top5_overlap}/5")
    print(f"  Argmax match: {argmax_match}")
    print(f"  {name1} argmax: {np.argmax(logits1)}")
    print(f"  {name2} argmax: {np.argmax(logits2)}")

    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'correlation': corr,
        'top5_overlap': top5_overlap,
        'argmax_match': argmax_match
    }


def main():
    token = 100  # Test token

    print("=" * 70)
    print("VALIDATION AGAINST PYTORCH REFERENCE")
    print("=" * 70)
    print(f"Test token: {token}")

    # Run v6.5
    print("\nRunning v6.5...")
    logits_65 = run_ck_model(V65_DIR, token)
    if logits_65 is not None:
        print(f"  v6.5 logits: min={logits_65.min():.4f}, max={logits_65.max():.4f}, argmax={np.argmax(logits_65)}")

    # Run v6.6
    print("\nRunning v6.6...")
    logits_66 = run_ck_model(V66_DIR, token)
    if logits_66 is not None:
        print(f"  v6.6 logits: min={logits_66.min():.4f}, max={logits_66.max():.4f}, argmax={np.argmax(logits_66)}")

    # Run PyTorch
    print("\nRunning PyTorch reference...")
    logits_pt = run_pytorch(token)
    if logits_pt is not None:
        print(f"  PyTorch logits: min={logits_pt.min():.4f}, max={logits_pt.max():.4f}, argmax={np.argmax(logits_pt)}")

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISONS")
    print("=" * 70)

    if logits_65 is not None and logits_66 is not None:
        compare_logits("v6.5", logits_65, "v6.6", logits_66)

    if logits_pt is not None:
        if logits_65 is not None:
            result_65 = compare_logits("v6.5", logits_65, "PyTorch", logits_pt)

        if logits_66 is not None:
            result_66 = compare_logits("v6.6", logits_66, "PyTorch", logits_pt)

        # Determine winner
        if logits_65 is not None and logits_66 is not None:
            print("\n" + "=" * 70)
            print("VERDICT")
            print("=" * 70)

            diff_65 = np.mean(np.abs(logits_65 - logits_pt))
            diff_66 = np.mean(np.abs(logits_66 - logits_pt))

            if diff_65 < diff_66:
                print(f"v6.5 is CLOSER to PyTorch (mean diff: {diff_65:.4f} vs {diff_66:.4f})")
            elif diff_66 < diff_65:
                print(f"v6.6 is CLOSER to PyTorch (mean diff: {diff_66:.4f} vs {diff_65:.4f})")
            else:
                print("Both are equally close to PyTorch")

            # Check argmax
            pt_argmax = np.argmax(logits_pt)
            if np.argmax(logits_65) == pt_argmax and np.argmax(logits_66) != pt_argmax:
                print("v6.5 matches PyTorch argmax, v6.6 does not!")
            elif np.argmax(logits_66) == pt_argmax and np.argmax(logits_65) != pt_argmax:
                print("v6.6 matches PyTorch argmax, v6.5 does not!")
            elif np.argmax(logits_65) == pt_argmax and np.argmax(logits_66) == pt_argmax:
                print("Both match PyTorch argmax")
            else:
                print("Neither matches PyTorch argmax!")


if __name__ == "__main__":
    main()
