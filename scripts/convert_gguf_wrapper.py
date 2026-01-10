#!/usr/bin/env python3
"""
Wrapper for GGUF to BUMP conversion with automatic metadata extraction
Handles Devstral, SmolLM, Qwen, and other architectures
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def extract_gguf_metadata(gguf_path: str) -> dict:
    """Extract metadata from GGUF file"""
    try:
        from gguf import GGUFReader
    except ImportError:
        print("❌ gguf library not found. Install with: pip install gguf")
        return {}

    reader = GGUFReader(gguf_path)
    meta = {}

    # Common fields
    meta["architecture"] = reader.get_value("general.architecture", "unknown")
    meta["vocab_size"] = reader.get_value("tokenizer.ggml.tokens", [])
    meta["vocab_size"] = len(meta["vocab_size"]) if meta["vocab_size"] else None

    # Attention parameters - try multiple keys
    for key in [
        "llama.attention.head_count",
        "mistral3.attention.head_count",
        "mistral.attention.head_count",
        "qwen2.attention.head_count",
    ]:
        val = reader.get_value(key)
        if val:
            meta["num_attention_heads"] = val
            break

    # KV heads
    for key in [
        "llama.attention.key_length",
        "mistral3.attention.key_length",
        "mistral.attention.key_length",
        "qwen2.attention.key_length",
    ]:
        val = reader.get_value(key)
        if val:
            meta["num_key_value_heads"] = val
            break

    # Other params
    meta["hidden_size"] = reader.get_value(f"{meta['architecture']}.embedding.layer_norm_rms_epsilon")
    meta["intermediate_size"] = reader.get_value(f"{meta['architecture']}.ffn_dim")

    # If not found in arch-specific, try generic
    if not meta["hidden_size"]:
        meta["hidden_size"] = reader.get_value("llama.embedding.layer_norm_rms_epsilon")

    # Calculate from tensors if needed
    if not meta.get("hidden_size"):
        # Extract from tensor shapes
        for name, info in reader.tensor_info.items():
            if "attn_output.weight" in name:
                # Shape is (hidden_size, hidden_size)
                meta["hidden_size"] = int(info.ne[0])
                break

    # Context length
    for key in [
        "llama.context_length",
        "mistral3.context_length",
        "mistral.context_length",
        "qwen2.context_length",
    ]:
        val = reader.get_value(key)
        if val:
            meta["context_length"] = val
            break

    if not meta.get("context_length"):
        meta["context_length"] = 4096  # Default

    # Layers
    for key in [
        "llama.block_count",
        "mistral3.block_count",
        "mistral.block_count",
        "qwen2.block_count",
    ]:
        val = reader.get_value(key)
        if val:
            meta["num_hidden_layers"] = val
            break

    return meta


def create_config_file(metadata: dict, output_path: str) -> str:
    """Create a config JSON file from metadata"""
    config = {
        "vocab_size": metadata.get("vocab_size"),
        "context_window": metadata.get("context_length"),
        "hidden_size": metadata.get("hidden_size"),
        "num_hidden_layers": metadata.get("num_hidden_layers"),
        "num_attention_heads": metadata.get("num_attention_heads"),
        "num_key_value_heads": metadata.get("num_key_value_heads"),
    }

    # Remove None values
    config = {k: v for k, v in config.items() if v is not None}

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    return output_path


def convert_gguf_to_bump(
    gguf_path: str,
    output_path: str,
    context: int = None,
    extra_args: list = None,
) -> bool:
    """Convert GGUF to BUMP with metadata handling"""
    print(f"📖 Extracting metadata from {gguf_path}...")

    # Extract metadata
    meta = extract_gguf_metadata(gguf_path)
    print(f"   Architecture: {meta.get('architecture', 'unknown')}")
    print(f"   Vocab size: {meta.get('vocab_size', 'N/A')}")
    print(f"   Hidden size: {meta.get('hidden_size', 'N/A')}")
    print(f"   Layers: {meta.get('num_hidden_layers', 'N/A')}")
    print(f"   Heads: {meta.get('num_attention_heads', 'N/A')}")

    # Create config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config_path = f.name
        create_config_file(meta, config_path)

    print(f"\n🔄 Converting GGUF to BUMP...")

    # Build command
    cmd = [
        "python",
        "scripts/v4/convert_gguf_to_bump_v4.py",
        "--gguf", gguf_path,
        "--output", output_path,
        "--config-out", config_path,
    ]

    if context:
        cmd.extend(["--context", str(context)])
    if extra_args:
        cmd.extend(extra_args)

    # Run conversion
    result = subprocess.run(cmd, capture_output=False)

    # Cleanup
    os.unlink(config_path)

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Convert GGUF to BUMP with automatic metadata handling"
    )
    parser.add_argument("gguf", help="Input GGUF file")
    parser.add_argument("output", nargs="?", help="Output BUMP file")
    parser.add_argument("--context", type=int, help="Context length override")
    parser.add_argument("--inspect", action="store_true", help="Inspect metadata only")
    parser.add_argument("--list", action="store_true", help="List tensors only")
    args = parser.parse_args()

    gguf_path = args.gguf
    if not os.path.exists(gguf_path):
        print(f"❌ File not found: {gguf_path}")
        return 1

    if not args.output:
        args.output = Path(gguf_path).with_suffix(".bump")

    if args.inspect or args.list:
        # Just list tensors
        subprocess.run(
            ["python", "scripts/v4/convert_gguf_to_bump_v4.py", "--gguf", gguf_path, "--list"]
        )
        return 0

    # Convert
    success = convert_gguf_to_bump(
        gguf_path,
        args.output,
        context=args.context,
    )

    if success:
        print(f"\n✅ Conversion complete: {args.output}")
        return 0
    else:
        print(f"\n❌ Conversion failed")
        return 1


if __name__ == "__main__":
    exit(main())
