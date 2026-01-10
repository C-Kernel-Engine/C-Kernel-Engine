#!/usr/bin/env python3
"""
Convert any GGUF file to BUMP format
Automatically extracts metadata and creates config
"""

import argparse
import json
import os
import subprocess
import tempfile


def extract_metadata(gguf_path: str) -> dict:
    """Extract metadata from GGUF file"""
    # Run the --list command and parse output
    result = subprocess.run(
        ["python", "scripts/v4/convert_gguf_to_bump_v4.py", "--gguf", gguf_path, "--list"],
        capture_output=True,
        text=True,
    )

    output = result.stdout

    # Parse tensor list
    lines = output.split("\n")
    tensors = {}

    for line in lines:
        if "dims=" in line and ":" in line:
            # Parse: "  - tensor.name: TYPE dims=(D1, D2)"
            parts = line.split(":")
            if len(parts) >= 2:
                name = parts[0].strip().replace("- ", "")
                dims_part = parts[1].strip()

                # Extract dims
                if "dims=(" in dims_part:
                    dims_str = dims_part.split("dims=(")[1].split(")")[0]
                    dims = tuple(int(d.strip()) for d in dims_str.split(","))
                    tensors[name] = dims

    # Extract metadata from tensors
    meta = {}

    # Token embedding
    if "token_embd.weight" in tensors:
        embed_dim, vocab_size = tensors["token_embd.weight"]
        meta["hidden_size"] = embed_dim
        meta["vocab_size"] = vocab_size

    # Attention output
    for name in tensors:
        if "attn_output.weight" in name and "blk.0" in name:
            output_dim, hidden_dim = tensors[name]
            meta["attn_output_dim"] = output_dim
            meta["attn_hidden_dim"] = hidden_dim
            break

    # Count layers
    blocks = set()
    for name in tensors:
        if name.startswith("blk.") and ".attn" in name:
            block_num = name.split(".")[1]
            blocks.add(block_num)

    meta["num_layers"] = len(blocks)

    # Calculate heads
    if "attn_output_dim" in meta and "hidden_size" in meta:
        head_dim = meta["hidden_size"] // 32  # Devstral uses 32 heads
        meta["num_heads"] = meta["attn_output_dim"] // head_dim
        meta["head_dim"] = head_dim

    return meta


def convert_gguf_to_bump(gguf_path: str, output_path: str) -> bool:
    """Convert GGUF to BUMP with auto-extracted metadata"""

    print(f"📖 Extracting metadata from {gguf_path}...")

    meta = extract_metadata(gguf_path)

    if not meta:
        print("❌ Failed to extract metadata")
        return False

    print(f"   Architecture: Devstral/SmolLM (Mistral-based)")
    print(f"   Vocab size: {meta.get('vocab_size', 'N/A')}")
    print(f"   Hidden size: {meta.get('hidden_size', 'N/A')}")
    print(f"   Layers: {meta.get('num_layers', 'N/A')}")
    print(f"   Heads: {meta.get('num_heads', 'N/A')}")
    print(f"   Head dim: {meta.get('head_dim', 'N/A')}")

    # Create config
    config = {
        "vocab_size": meta.get("vocab_size", 131072),
        "context_window": 4096,
        "hidden_size": meta.get("hidden_size", 5120),
        "num_hidden_layers": meta.get("num_layers", 40),
        "num_attention_heads": meta.get("num_heads", 32),
    }

    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config_path = f.name
        json.dump(config, f, indent=2)

    try:
        print(f"\n🔄 Converting to BUMP...")

        # Run conversion
        result = subprocess.run(
            [
                "python",
                "scripts/v4/convert_gguf_to_bump_v4.py",
                "--gguf", gguf_path,
                "--output", output_path,
                "--config-out", config_path,
            ],
            capture_output=False,
            text=True,
        )

        return result.returncode == 0

    finally:
        # Cleanup
        if os.path.exists(config_path):
            os.unlink(config_path)


def main():
    parser = argparse.ArgumentParser(description="Convert GGUF to BUMP")
    parser.add_argument("gguf", help="Input GGUF file")
    parser.add_argument("output", nargs="?", help="Output BUMP file")
    args = parser.parse_args()

    if not os.path.exists(args.gguf):
        print(f"❌ File not found: {args.gguf}")
        return 1

    if not args.output:
        args.output = args.gguf.replace(".gguf", ".bump")

    success = convert_gguf_to_bump(args.gguf, args.output)

    if success:
        print(f"\n✅ Conversion complete: {args.output}")
        print(f"\nNext steps:")
        print(f"  ck_cli_v5 --model {args.output} --prompt 'Hello'")
        return 0
    else:
        print(f"\n❌ Conversion failed")
        return 1


if __name__ == "__main__":
    exit(main())
