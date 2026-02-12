#!/usr/bin/env python3
"""
Layer-by-layer validation for v7 artifacts only.

Examples:
  python test_layer_by_layer.py --model ~/.cache/ck-engine-v7/models/gemma-3-270m-it-Q5_K_M
  python test_layer_by_layer.py --model ~/.cache/ck-engine-v7/models/Qwen--Qwen2-0.5B-Instruct-GGUF --token 25
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

DEFAULT_MODEL = Path.home() / ".cache/ck-engine-v7/models/Qwen--Qwen2-0.5B-Instruct-GGUF"

MODEL_DIR: Path = DEFAULT_MODEL
TEST_TOKEN: int = 100


def load_q8_0_embedding(bump_path: Path, offset: int, token_id: int, embed_dim: int) -> np.ndarray:
    """Load and dequantize one token row from a Q8_0 embedding matrix."""
    block_size = 32
    num_blocks = (embed_dim + (block_size - 1)) // block_size
    bytes_per_block = 2 + 32  # fp16 scale + 32 int8 quants
    row_start = offset + (token_id * num_blocks * bytes_per_block)

    values = []
    with open(bump_path, "rb") as f:
        for b in range(num_blocks):
            f.seek(row_start + b * bytes_per_block)
            block_data = f.read(bytes_per_block)
            if len(block_data) != bytes_per_block:
                raise ValueError(f"Short read in {bump_path} for token={token_id}, block={b}")
            scale = np.frombuffer(block_data[:2], dtype=np.float16)[0]
            quants = np.frombuffer(block_data[2:], dtype=np.int8)
            values.extend((quants.astype(np.float32) * float(scale)).tolist())

    return np.array(values[:embed_dim], dtype=np.float32)


def load_fp32_vector(bump_path: Path, offset: int, count: int) -> np.ndarray:
    """Load fp32 vector from bump file."""
    with open(bump_path, "rb") as f:
        f.seek(offset)
        data = f.read(count * 4)
    if len(data) != count * 4:
        raise ValueError(f"Short read in {bump_path} offset={offset} count={count}")
    return np.frombuffer(data, dtype=np.float32)


def rmsnorm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute RMSNorm."""
    x = np.asarray(x, dtype=np.float32)
    gamma = np.asarray(gamma, dtype=np.float32)
    variance = np.mean(x ** 2)
    x_norm = x / np.sqrt(variance + eps)
    return x_norm * gamma


def load_manifest_entries(model_dir: Path) -> Dict[str, dict]:
    manifest_path = model_dir / "weights_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"weights_manifest.json not found: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    return {e["name"]: e for e in manifest.get("entries", [])}


def infer_embed_dim(entries: Dict[str, dict]) -> int:
    tok = entries.get("token_emb")
    if tok and isinstance(tok, dict):
        shape = tok.get("shape") or tok.get("dims")
        if isinstance(shape, list) and len(shape) >= 2 and isinstance(shape[-1], int):
            return int(shape[-1])
    for key in ("layer.0.ln1_gamma", "layer.0.attention_norm", "final_norm"):
        entry = entries.get(key)
        if entry and entry.get("size"):
            return int(entry["size"] // 4)
    return 896


def test_v7_embedding() -> None:
    print("\n" + "=" * 60)
    print("TEST 1: Token Embedding Lookup (v7)")
    print("=" * 60)

    entries = load_manifest_entries(MODEL_DIR)
    if "token_emb" not in entries:
        print("FAIL: token_emb missing in manifest")
        return

    tok_entry = entries["token_emb"]
    embed_dim = infer_embed_dim(entries)
    print(f"Token embedding: offset={tok_entry['file_offset']}, dtype={tok_entry.get('dtype', '?')}, dim={embed_dim}")

    test_tokens = sorted({0, 1, TEST_TOKEN, 1000})
    v7_bump = MODEL_DIR / "weights.bump"
    if not v7_bump.exists():
        raise FileNotFoundError(f"weights.bump not found: {v7_bump}")

    for token_id in test_tokens:
        emb = load_q8_0_embedding(v7_bump, int(tok_entry["file_offset"]), token_id, embed_dim)
        has_nan = bool(np.any(np.isnan(emb)))
        print(f"\nToken {token_id}:")
        print(f"  first 5: {emb[:5]}")
        print(f"  stats: min={emb.min():.6f}, max={emb.max():.6f}, mean={emb.mean():.6f}, nan={has_nan}")


def test_v7_rmsnorm() -> None:
    print("\n" + "=" * 60)
    print("TEST 2: Layer 0 RMSNorm (v7)")
    print("=" * 60)

    entries = load_manifest_entries(MODEL_DIR)
    ln_keys = ["layer.0.ln1_gamma", "layer.0.attention_norm"]
    ln_key = next((k for k in ln_keys if k in entries), None)
    if not ln_key:
        print("FAIL: no layer0 norm gamma found (expected one of layer.0.ln1_gamma or layer.0.attention_norm)")
        return

    ln_entry = entries[ln_key]
    embed_dim = int(ln_entry["size"] // 4)
    print(f"{ln_key}: offset={ln_entry['file_offset']} dim={embed_dim}")

    gamma = load_fp32_vector(MODEL_DIR / "weights.bump", int(ln_entry["file_offset"]), embed_dim)

    tok_entry = entries.get("token_emb")
    if not tok_entry:
        print("FAIL: token_emb missing for RMSNorm input probe")
        return

    input_emb = load_q8_0_embedding(
        MODEL_DIR / "weights.bump",
        int(tok_entry["file_offset"]),
        TEST_TOKEN,
        infer_embed_dim(entries),
    )
    output = rmsnorm(input_emb[:embed_dim], gamma)
    print(f"\nRMSNorm probe (token {TEST_TOKEN} embedding -> layer 0 input):")
    print(f"  Input first 5: {input_emb[:5]}")
    print(f"  Output first 5: {output[:5]}")
    print(f"  Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")


def test_v7_layer0_weights() -> None:
    print("\n" + "=" * 60)
    print("TEST 3: Layer 0 Weight Integrity (v7)")
    print("=" * 60)

    entries = load_manifest_entries(MODEL_DIR)
    v7_bump = MODEL_DIR / "weights.bump"
    layer0_names = sorted(n for n in entries if n.startswith("layer.0."))

    if not layer0_names:
        print("FAIL: no layer.0.* entries found in manifest")
        return

    all_readable = True
    with open(v7_bump, "rb") as f:
        for name in layer0_names:
            e = entries[name]
            offset = int(e.get("file_offset", -1))
            size = int(e.get("size", 0))
            dtype = e.get("dtype", "?")
            sample_n = min(32, max(size, 0))

            if offset < 0 or size <= 0:
                all_readable = False
                print(f"✗ {name}: invalid metadata offset={offset} size={size} dtype={dtype}")
                continue

            f.seek(offset)
            sample = f.read(sample_n)
            if len(sample) != sample_n:
                all_readable = False
                print(f"✗ {name}: short read offset={offset} size={size} dtype={dtype}")
                continue

            checksum = sum(sample) % 256
            print(
                f"✓ {name}: offset={offset}, size={size}, dtype={dtype}, "
                f"sample_crc8={checksum:02x}, first8={sample[:8].hex()}"
            )

    if all_readable:
        print("\n✓ All layer.0 weights are readable with valid metadata")
    else:
        print("\n✗ Some layer.0 weights have invalid metadata or unreadable payload")


def map_define_to_manifest_name(define_name: str) -> Optional[str]:
    if define_name == "W_TOKEN_EMB":
        return "token_emb"
    if define_name == "W_LM_HEAD":
        return "lm_head"
    if define_name == "W_FINAL_NORM":
        return "final_norm"
    if define_name.startswith("W_LAYER_"):
        rest = define_name[len("W_LAYER_"):]
        parts = rest.split("_")
        if len(parts) >= 2 and parts[0].isdigit():
            layer = parts[0]
            tail = "_".join(parts[1:]).lower()
            return f"layer.{layer}.{tail}"
    return None


def check_c_code_offsets() -> None:
    print("\n" + "=" * 60)
    print("TEST 4: C Code Offset Verification (v7)")
    print("=" * 60)

    entries = load_manifest_entries(MODEL_DIR)
    candidates = [
        MODEL_DIR / "model_v7.c",
        MODEL_DIR / "generated_model.c",
        MODEL_DIR / "model.c",
    ]
    model_c = next((p for p in candidates if p.exists()), None)
    if model_c is None:
        print("SKIP: no generated C file found (checked model_v7.c, generated_model.c, model.c)")
        return

    c_defines: Dict[str, int] = {}
    with open(model_c) as f:
        for line in f:
            m = re.match(r"#define\s+(W_\w+)\s+(\d+)", line)
            if m:
                c_defines[m.group(1)] = int(m.group(2))

    print(f"Found {len(c_defines)} W_* defines in {model_c.name}")
    if not c_defines:
        print("SKIP: no W_* offset defines found")
        return

    compared = 0
    all_match = True
    for c_name, c_offset in sorted(c_defines.items()):
        manifest_name = map_define_to_manifest_name(c_name)
        if not manifest_name:
            continue
        m_entry = entries.get(manifest_name)
        if not m_entry:
            continue
        compared += 1
        m_offset = int(m_entry.get("file_offset", -1))
        match = c_offset == m_offset
        if not match:
            all_match = False
        status = "✓" if match else "✗"
        print(f"{status} {c_name}: C={c_offset}, manifest={m_offset}")
        if not match:
            print(f"    Diff: {c_offset - m_offset}")

    if compared == 0:
        print("SKIP: no overlapping W_* defines matched manifest names")
    elif all_match:
        print("\n✓ All checked C defines match manifest offsets")
    else:
        print("\n✗ Some C defines differ from manifest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v7 layer-by-layer verification")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="v7 model directory (contains weights_manifest.json and weights.bump)",
    )
    parser.add_argument(
        "--token",
        type=int,
        default=100,
        help="Token id used for embedding/rmsnorm probe",
    )
    return parser.parse_args()


def initialize_from_args(args: argparse.Namespace) -> Optional[str]:
    global MODEL_DIR, TEST_TOKEN
    MODEL_DIR = args.model.expanduser().resolve()
    TEST_TOKEN = int(args.token)

    if not MODEL_DIR.exists():
        return f"Model directory does not exist: {MODEL_DIR}"
    if not (MODEL_DIR / "weights_manifest.json").exists():
        return f"weights_manifest.json missing in {MODEL_DIR}"
    if not (MODEL_DIR / "weights.bump").exists():
        return f"weights.bump missing in {MODEL_DIR}"

    return None


def main() -> int:
    args = parse_args()
    err = initialize_from_args(args)
    if err:
        print(f"ERROR: {err}")
        return 1

    print("=" * 60)
    print("V7 LAYER-BY-LAYER VERIFICATION")
    print("=" * 60)
    print(f"Model: {MODEL_DIR}")
    print(f"Token probe: {TEST_TOKEN}")

    test_v7_embedding()
    test_v7_rmsnorm()
    test_v7_layer0_weights()
    check_c_code_offsets()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("If these artifact checks pass but runtime still fails, investigate kernel execution with stop-op tracing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
