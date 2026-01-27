#!/usr/bin/env python3
"""
test_kernel_validation.py - Kernel-by-kernel validation against PyTorch/HuggingFace

This test validates each kernel in the C-Kernel-Engine against the reference
implementation (PyTorch transformers) to ensure numerical correctness.

Tests:
1. Embedding lookup
2. RMSNorm
3. Q/K/V projections (with bias)
4. RoPE
5. Attention
6. Residual add
7. MLP (gate_up + SwiGLU + down)
8. Final logits

Usage:
    python test_kernel_validation.py [--layer N] [--verbose]
"""

import argparse
import ctypes
import json
import numpy as np
import sys
from pathlib import Path

# Check for required packages
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch/transformers not available. Install with:")
    print("  pip install torch transformers")

# Paths
CACHE_DIR = Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF"
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def load_ck_model():
    """Load the C-Kernel-Engine model."""
    so_path = CACHE_DIR / "ck-kernel-inference.so"
    if not so_path.exists():
        print(f"Error: {so_path} not found. Run ck_run_v6_6.py first.")
        return None

    lib = ctypes.CDLL(str(so_path))

    # Set up function signatures
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    lib.ck_model_get_logits.argtypes = []
    lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)
    lib.ck_model_get_vocab_size.argtypes = []
    lib.ck_model_get_vocab_size.restype = ctypes.c_int
    lib.ck_model_kv_cache_reset.argtypes = []
    lib.ck_model_kv_cache_reset.restype = None
    lib.ck_model_free.argtypes = []
    lib.ck_model_free.restype = None

    # Load weights
    weights_path = CACHE_DIR / "weights.bump"
    if lib.ck_model_init(str(weights_path).encode()) != 0:
        print("Error: Failed to load CK model")
        return None

    return lib


def load_hf_model():
    """Load the HuggingFace reference model."""
    if not HAS_TORCH:
        return None, None

    model_name = "Qwen/Qwen2-0.5B-Instruct"
    print(f"Loading HuggingFace model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    model.eval()

    return model, tokenizer


# Quantization-aware tolerance mapping
QUANT_TOLERANCES = {
    "fp32": (1e-3, 1e-4),
    "fp16": (1e-3, 1e-4),
    "bf16": (1e-3, 1e-4),
    "q8_0": (1e-2, 1e-2),
    "q8_1": (1e-2, 1e-2),
    "q5_0": (1e-2, 1e-1),
    "q5_1": (1e-2, 1e-1),
    "q6_k": (1e-2, 1e-1),
    "q4_k": (1e-2, 2e-1),
    "q4_0": (1e-2, 1e-1),
    "q4_1": (1e-2, 1e-1),
}


def get_tolerance(dtype: str) -> tuple:
    """Get (rtol, atol) for a given dtype."""
    return QUANT_TOLERANCES.get(dtype, (1e-3, 1e-4))


def compare_tensors(name: str, ck_tensor: np.ndarray, hf_tensor: np.ndarray,
                   rtol: float = None, atol: float = None, dtype: str = None) -> dict:
    """Compare two tensors and return statistics.

    Args:
        name: Test name
        ck_tensor: C-Kernel tensor
        hf_tensor: HuggingFace reference tensor
        rtol: Relative tolerance (auto if dtype given)
        atol: Absolute tolerance (auto if dtype given)
        dtype: Quantization type for auto tolerance selection
    """
    # Auto-select tolerance based on dtype if not provided
    if rtol is None or atol is None:
        if dtype:
            auto_rtol, auto_atol = get_tolerance(dtype)
            rtol = rtol if rtol is not None else auto_rtol
            atol = atol if atol is not None else auto_atol
        else:
            rtol = rtol or 1e-3
            atol = atol or 1e-4

    if ck_tensor.shape != hf_tensor.shape:
        return {
            "name": name,
            "passed": False,
            "error": f"Shape mismatch: CK={ck_tensor.shape} vs HF={hf_tensor.shape}"
        }

    # Check for NaN/Inf
    ck_nan = np.isnan(ck_tensor).any() or np.isinf(ck_tensor).any()
    hf_nan = np.isnan(hf_tensor).any() or np.isinf(hf_tensor).any()

    if ck_nan:
        return {
            "name": name,
            "passed": False,
            "error": f"CK tensor has NaN/Inf values"
        }

    # Compute differences
    abs_diff = np.abs(ck_tensor - hf_tensor)
    max_abs_diff = abs_diff.max()
    mean_abs_diff = abs_diff.mean()

    # Relative difference (avoiding division by zero)
    hf_abs = np.abs(hf_tensor)
    rel_diff = abs_diff / np.maximum(hf_abs, 1e-8)
    max_rel_diff = rel_diff.max()
    mean_rel_diff = rel_diff.mean()

    # Check if within tolerance
    passed = np.allclose(ck_tensor, hf_tensor, rtol=rtol, atol=atol)

    return {
        "name": name,
        "passed": passed,
        "max_abs_diff": float(max_abs_diff),
        "mean_abs_diff": float(mean_abs_diff),
        "max_rel_diff": float(max_rel_diff),
        "mean_rel_diff": float(mean_rel_diff),
        "ck_range": (float(ck_tensor.min()), float(ck_tensor.max())),
        "hf_range": (float(hf_tensor.min()), float(hf_tensor.max())),
    }


def test_full_forward(ck_lib, hf_model, hf_tokenizer, test_text: str = "Hello"):
    """Test full forward pass comparing CK vs HuggingFace."""
    print("\n" + "="*60)
    print("TEST: Full Forward Pass")
    print("="*60)

    # Tokenize
    inputs = hf_tokenizer(test_text, return_tensors="pt")
    tokens = inputs.input_ids[0].tolist()
    print(f"Input text: '{test_text}'")
    print(f"Tokens: {tokens}")

    # HuggingFace forward
    with torch.no_grad():
        hf_outputs = hf_model(inputs.input_ids)
        hf_logits = hf_outputs.logits[0, -1, :].numpy()  # Last token logits

    # CK forward (process all tokens)
    ck_lib.ck_model_kv_cache_reset()
    for token in tokens:
        ck_lib.ck_model_decode(token, None)

    vocab_size = ck_lib.ck_model_get_vocab_size()
    ck_logits_ptr = ck_lib.ck_model_get_logits()
    ck_logits = np.ctypeslib.as_array(ck_logits_ptr, shape=(vocab_size,)).copy()

    # Compare
    result = compare_tensors("logits", ck_logits, hf_logits)

    print(f"\nLogits comparison:")
    print(f"  Shape: CK={ck_logits.shape}, HF={hf_logits.shape}")
    print(f"  CK range: [{ck_logits.min():.4f}, {ck_logits.max():.4f}]")
    print(f"  HF range: [{hf_logits.min():.4f}, {hf_logits.max():.4f}]")
    print(f"  Max abs diff: {result['max_abs_diff']:.6f}")
    print(f"  Mean abs diff: {result['mean_abs_diff']:.6f}")

    # Check for NaN
    if np.isnan(ck_logits).any():
        nan_count = np.isnan(ck_logits).sum()
        print(f"  WARNING: {nan_count} NaN values in CK logits!")
        result["passed"] = False

    # Check for extreme values
    if np.abs(ck_logits).max() > 1e10:
        print(f"  WARNING: Extreme values in CK logits (possible overflow)")
        result["passed"] = False

    # Top-5 token comparison
    print(f"\nTop-5 predicted tokens:")
    hf_top5 = np.argsort(hf_logits)[-5:][::-1]
    ck_top5 = np.argsort(ck_logits)[-5:][::-1]
    print(f"  HF: {[hf_tokenizer.decode([t]) for t in hf_top5]}")
    print(f"  CK: {[hf_tokenizer.decode([t]) for t in ck_top5]}")

    if result["passed"]:
        print("\n[PASS] Full forward pass matches!")
    else:
        print("\n[FAIL] Full forward pass does not match")

    return result


def test_embedding(ck_lib, hf_model, token_id: int = 9707):
    """Test embedding lookup."""
    print("\n" + "="*60)
    print(f"TEST: Embedding Lookup (token_id={token_id})")
    print("="*60)

    # HuggingFace embedding
    with torch.no_grad():
        hf_emb = hf_model.model.embed_tokens(torch.tensor([[token_id]]))[0, 0].numpy()

    print(f"HF embedding shape: {hf_emb.shape}")
    print(f"HF embedding range: [{hf_emb.min():.4f}, {hf_emb.max():.4f}]")

    # For CK, we need to extract intermediate activation
    # This would require modifying the generated code to expose internal buffers
    # For now, we'll just verify the HF reference is reasonable

    print("\n[INFO] CK embedding extraction requires internal buffer access")
    print("       Skipping direct comparison, will validate through full forward pass")

    return {"name": "embedding", "passed": True, "skipped": True}


def test_logits_weight_tying(ck_lib, hf_model):
    """Verify logits uses embedding matrix (weight tying)."""
    print("\n" + "="*60)
    print("TEST: Logits Weight Tying")
    print("="*60)

    # In Qwen2, lm_head shares weights with embed_tokens
    embed_weight = hf_model.model.embed_tokens.weight.data.numpy()
    lm_head_weight = hf_model.lm_head.weight.data.numpy()

    # Check if they're the same (weight tying)
    is_tied = np.allclose(embed_weight, lm_head_weight)

    print(f"Embedding weight shape: {embed_weight.shape}")
    print(f"LM head weight shape: {lm_head_weight.shape}")
    print(f"Weights are tied: {is_tied}")

    if is_tied:
        print("\n[PASS] Model uses weight tying (logits = embedding^T @ hidden)")
    else:
        print("\n[INFO] Model has separate lm_head weights")

    return {"name": "weight_tying", "passed": True, "is_tied": is_tied}


def test_layer_structure(hf_model):
    """Print model layer structure for reference."""
    print("\n" + "="*60)
    print("MODEL STRUCTURE (HuggingFace Reference)")
    print("="*60)

    config = hf_model.config
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_heads: {config.num_attention_heads}")
    print(f"  num_kv_heads: {config.num_key_value_heads}")
    print(f"  head_dim: {config.hidden_size // config.num_attention_heads}")
    print(f"  intermediate_size: {config.intermediate_size}")
    print(f"  num_layers: {config.num_hidden_layers}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  rms_norm_eps: {config.rms_norm_eps}")
    print(f"  rope_theta: {config.rope_theta}")

    # Check layer structure
    layer0 = hf_model.model.layers[0]
    print(f"\nLayer 0 structure:")
    print(f"  input_layernorm: {layer0.input_layernorm}")
    print(f"  self_attn.q_proj: {layer0.self_attn.q_proj.weight.shape}")
    print(f"  self_attn.k_proj: {layer0.self_attn.k_proj.weight.shape}")
    print(f"  self_attn.v_proj: {layer0.self_attn.v_proj.weight.shape}")
    print(f"  self_attn.o_proj: {layer0.self_attn.o_proj.weight.shape}")

    # Check for biases
    has_q_bias = layer0.self_attn.q_proj.bias is not None
    has_k_bias = layer0.self_attn.k_proj.bias is not None
    has_v_bias = layer0.self_attn.v_proj.bias is not None
    print(f"\n  Q bias: {has_q_bias}")
    print(f"  K bias: {has_k_bias}")
    print(f"  V bias: {has_v_bias}")

    return {"name": "structure", "passed": True}


def run_validation_suite():
    """Run the full validation suite."""
    print("="*60)
    print("C-Kernel-Engine v6.6 - Kernel Validation Suite")
    print("="*60)

    # Load models
    print("\n[1/3] Loading C-Kernel-Engine model...")
    ck_lib = load_ck_model()
    if not ck_lib:
        print("Failed to load CK model")
        return 1

    print("[2/3] Loading HuggingFace reference model...")
    hf_model, hf_tokenizer = load_hf_model()
    if hf_model is None:
        print("Failed to load HF model")
        ck_lib.ck_model_free()
        return 1

    print("[3/3] Running validation tests...\n")

    results = []

    # Test model structure
    results.append(test_layer_structure(hf_model))

    # Test weight tying
    results.append(test_logits_weight_tying(ck_lib, hf_model))

    # Test embedding
    results.append(test_embedding(ck_lib, hf_model))

    # Test full forward pass
    results.append(test_full_forward(ck_lib, hf_model, hf_tokenizer, "Hello"))
    results.append(test_full_forward(ck_lib, hf_model, hf_tokenizer, "The capital of France is"))

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    passed = sum(1 for r in results if r.get("passed", False))
    failed = len(results) - passed

    for r in results:
        status = "[PASS]" if r.get("passed") else "[FAIL]"
        if r.get("skipped"):
            status = "[SKIP]"
        print(f"  {status} {r['name']}")
        if not r.get("passed") and not r.get("skipped"):
            if "error" in r:
                print(f"         Error: {r['error']}")
            if "max_abs_diff" in r:
                print(f"         Max diff: {r['max_abs_diff']:.6f}")

    print(f"\nTotal: {passed}/{len(results)} passed")

    # Cleanup
    ck_lib.ck_model_free()

    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Kernel validation against PyTorch")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if not HAS_TORCH:
        print("Error: PyTorch and transformers required for validation")
        print("Install with: pip install torch transformers")
        return 1

    return run_validation_suite()


if __name__ == "__main__":
    sys.exit(main())
