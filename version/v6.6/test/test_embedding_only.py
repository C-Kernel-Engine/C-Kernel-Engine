#!/usr/bin/env python3
"""
test_embedding_only.py - Test JUST the embedding layer

This is the first checkpoint test. If embedding fails, nothing else matters.

USAGE:
    cd ~/.cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF
    LD_LIBRARY_PATH=. python /path/to/test_embedding_only.py --token 25
"""

import argparse
import ctypes
import json
import numpy as np
import os
import struct
from pathlib import Path


def load_v66_model(model_path: Path):
    """Load v6.6 model and return library handle."""
    # Load kernel engine first
    engine_path = model_path / "libckernel_engine.so"
    if engine_path.exists():
        ctypes.CDLL(str(engine_path), mode=ctypes.RTLD_GLOBAL)

    lib_path = model_path / "libmodel.so"
    lib = ctypes.CDLL(str(lib_path))

    # Setup function signatures
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    lib.ck_model_get_logits.argtypes = []
    lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)
    lib.ck_model_free.argtypes = []
    lib.ck_model_free.restype = None

    return lib


def get_pytorch_embedding(token: int, model_name: str = "Qwen/Qwen2-0.5B-Instruct") -> np.ndarray:
    """Get embedding from PyTorch/HuggingFace."""
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("ERROR: PyTorch/transformers not installed")
        print("  pip install torch transformers")
        return None

    print(f"Loading PyTorch model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    model.eval()

    # Get embedding weight and look up token
    embed_weight = model.model.embed_tokens.weight.detach().numpy()
    print(f"  Embedding shape: {embed_weight.shape}")

    embedding = embed_weight[token].copy()
    print(f"  Token {token} embedding: [{embedding[:5]}...]")
    print(f"  Range: [{embedding.min():.6f}, {embedding.max():.6f}]")

    return embedding


def get_v66_embedding_from_weights(model_path: Path, token: int, config: dict) -> np.ndarray:
    """
    Directly read embedding from weights file.
    This bypasses the inference code to test just the data.
    """
    weights_path = model_path / "weights.bump"
    manifest_path = model_path / "weights_manifest.json"

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Find token_emb in manifest (v6.6 uses "entries" not "weights")
    token_emb_info = None
    for entry in manifest.get("entries", manifest.get("weights", [])):
        if entry.get("name") == "token_emb":
            token_emb_info = entry
            break

    if not token_emb_info:
        print("ERROR: token_emb not found in manifest")
        return None

    # v6.6 uses "file_offset", older versions use "offset"
    offset = token_emb_info.get("file_offset", token_emb_info.get("offset", 0))
    dtype = token_emb_info.get("dtype", "q8_0")
    embed_dim = config.get("embed_dim", 896)
    vocab_size = config.get("vocab_size", 151936)

    print(f"  token_emb: offset={offset}, dtype={dtype}, vocab={vocab_size}, embed={embed_dim}")

    # Read the embedding for this token
    with open(weights_path, "rb") as f:
        if dtype == "q8_0":
            # Q8_0: 32 elements per block, 34 bytes per block (32 int8 + 1 fp16 scale)
            block_size = 32
            bytes_per_block = 34
            blocks_per_row = embed_dim // block_size

            # Calculate byte offset for this token's row
            row_bytes = blocks_per_row * bytes_per_block
            token_offset = offset + token * row_bytes

            f.seek(token_offset)
            row_data = f.read(row_bytes)

            # Dequantize Q8_0
            embedding = np.zeros(embed_dim, dtype=np.float32)
            for b in range(blocks_per_row):
                block_start = b * bytes_per_block
                # Scale is fp16 at start of block
                scale_bytes = row_data[block_start:block_start+2]
                scale = struct.unpack('<e', scale_bytes)[0]  # fp16

                # Quantized values are int8
                quants = np.frombuffer(
                    row_data[block_start+2:block_start+34],
                    dtype=np.int8
                )

                # Dequantize: val = quant * scale
                start_idx = b * block_size
                embedding[start_idx:start_idx+block_size] = quants.astype(np.float32) * scale

            return embedding

        elif dtype == "fp32":
            token_offset = offset + token * embed_dim * 4
            f.seek(token_offset)
            return np.frombuffer(f.read(embed_dim * 4), dtype=np.float32)

        else:
            print(f"ERROR: Unsupported dtype {dtype}")
            return None


def run_v66_and_get_embedding_output(model_path: Path, token: int, config: dict) -> np.ndarray:
    """
    Run v6.6 inference and read the embedding output from activations.
    """
    lib = load_v66_model(model_path)

    weights_path = model_path / "weights.bump"
    result = lib.ck_model_init(str(weights_path).encode())
    if result != 0:
        print(f"ERROR: init failed with {result}")
        return None

    # Run one decode step
    vocab_size = config.get("vocab_size", 151936)
    output = (ctypes.c_float * vocab_size)()

    print(f"  Running decode with token {token}...")
    result = lib.ck_model_decode(token, output)
    print(f"  Decode result: {result}")

    # Try to read embedding output from activations
    # The embedding output goes to layer_output at offset 813751828
    # But we need to access model->activations which we can't directly...

    # For now, just check logits
    logits_ptr = lib.ck_model_get_logits()
    logits = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,)).copy()

    lib.ck_model_free()

    return logits


def main():
    parser = argparse.ArgumentParser(description="Test embedding layer only")
    parser.add_argument("--model", type=Path,
                        default=Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF")
    parser.add_argument("--token", type=int, default=25)
    parser.add_argument("--skip-pytorch", action="store_true", help="Skip PyTorch comparison")

    args = parser.parse_args()

    # Load config
    config_path = args.model / "lowered_decode.json"
    with open(config_path) as f:
        data = json.load(f)
        config = data.get("config", {})

    print(f"\n{'='*70}")
    print(f"EMBEDDING TEST: token={args.token}")
    print(f"{'='*70}")
    print(f"  embed_dim: {config.get('embed_dim')}")
    print(f"  vocab_size: {config.get('vocab_size')}")

    # Test 1: Read embedding directly from weights
    print(f"\n{'='*70}")
    print("TEST 1: Direct weight read (bypass inference)")
    print(f"{'='*70}")

    v66_emb_direct = get_v66_embedding_from_weights(args.model, args.token, config)
    if v66_emb_direct is not None:
        print(f"  v6.6 embedding: [{v66_emb_direct[:5]}...]")
        print(f"  Range: [{v66_emb_direct.min():.6f}, {v66_emb_direct.max():.6f}]")
        print(f"  All zeros: {np.all(v66_emb_direct == 0)}")

    # Test 2: Compare with PyTorch
    if not args.skip_pytorch:
        print(f"\n{'='*70}")
        print("TEST 2: PyTorch comparison")
        print(f"{'='*70}")

        pytorch_emb = get_pytorch_embedding(args.token)
        if pytorch_emb is not None and v66_emb_direct is not None:
            # Compare
            diff = np.abs(pytorch_emb - v66_emb_direct)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            print(f"\n  Comparison:")
            print(f"    Max diff: {max_diff:.6f}")
            print(f"    Mean diff: {mean_diff:.6f}")

            if max_diff < 0.01:
                print(f"    [PASS] Embeddings match within tolerance")
            else:
                print(f"    [FAIL] Embeddings differ significantly")
                # Find where they differ
                diff_idx = np.argmax(diff)
                print(f"    First big diff at index {diff_idx}:")
                print(f"      PyTorch: {pytorch_emb[diff_idx]:.6f}")
                print(f"      v6.6:    {v66_emb_direct[diff_idx]:.6f}")

    # Test 3: Run v6.6 inference and check output
    print(f"\n{'='*70}")
    print("TEST 3: v6.6 inference output")
    print(f"{'='*70}")

    os.environ["LD_LIBRARY_PATH"] = str(args.model) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

    logits = run_v66_and_get_embedding_output(args.model, args.token, config)
    if logits is not None:
        print(f"  Logits range: [{logits.min():.6f}, {logits.max():.6f}]")
        print(f"  Argmax: {np.argmax(logits)}")
        print(f"  All zeros: {np.all(logits == 0)}")

        if np.all(logits == 0):
            print("\n  [FAIL] Logits are all zeros!")
            print("  This means something in the inference pipeline is broken.")
            print("  The embedding data itself looks OK, so the bug is in:")
            print("    1. How the token is stored/read")
            print("    2. How embedding output is passed to next layer")
            print("    3. Some layer zeroing out the values")
        else:
            print("\n  [PASS] Logits are non-zero!")


if __name__ == "__main__":
    main()
