#!/usr/bin/env python3
"""
Quick Patch: Add Devstral/SmolLM support to existing bump converter
Run this script to patch convert_hf_to_bump_v4.py with multi-architecture support
"""

import argparse
import re
from pathlib import Path


def patch_converter(converter_path: str, output_path: str = None) -> None:
    """Patch the existing converter to support Devstral/SmolLM"""

    converter_path = Path(converter_path)
    output_path = Path(output_path) if output_path else converter_path

    print(f"Patching: {converter_path}")
    print(f"Output: {output_path}\n")

    with open(converter_path, "r") as f:
        content = f.read()

    # Patch 1: Add architecture detection function
    patch1 = """
def detect_architecture(state_dict, cfg):
    \"\"\"Auto-detect if this is Devstral/SmolLM vs LLaMA\"\"\"
    # Check for Devstral/SmolLM patterns
    sample_keys = list(state_dict.keys())[:20]

    # Devstral/SmolLM might use model.embedding.weight
    if any("embedding.weight" in k for k in sample_keys):
        return "devstral"

    # Otherwise assume LLaMA
    return "llama"
"""

    # Find insertion point (after imports)
    insert_point = content.find("def build_dtype_table")
    if insert_point == -1:
        insert_point = content.find("def main()")
        if insert_point == -1:
            raise ValueError("Could not find insertion point")

    content = content[:insert_point] + patch1 + "\n" + content[insert_point:]

    # Patch 2: Update get_tensor calls to handle both architectures
    # Replace single weight lookups with flexible lookups

    # Embedding layer
    old_embed = '''tok = get_tensor(
            state_dict,
            "model.embed_tokens.weight",
            alt_keys=("model.tok_embeddings.weight",),
        ).detach().cpu().numpy()'''

    new_embed = '''# Flexible embedding layer detection
    embed_key = "model.embed_tokens.weight"
    if embed_key not in state_dict:
        if "model.embedding.weight" in state_dict:
            embed_key = "model.embedding.weight"
        elif "model.tok_embeddings.weight" in state_dict:
            embed_key = "model.tok_embeddings.weight"
        else:
            raise KeyError(f"Could not find embedding layer. Tried: model.embed_tokens.weight, model.embedding.weight, model.tok_embeddings.weight")

    tok = state_dict[embed_key].detach().cpu().numpy()'''

    content = content.replace(old_embed, new_embed)

    # LayerNorm layers
    old_ln1 = '''ln1 = get_tensor(state_dict, f"{prefix}.input_layernorm.weight").detach().cpu().numpy()'''
    new_ln1 = '''# Flexible LayerNorm detection
    ln1_key = f"{prefix}.input_layernorm.weight"
    if ln1_key not in state_dict:
        alt_key = f"{prefix}.layer_norm.weight"
        if alt_key in state_dict:
            ln1_key = alt_key
        else:
            raise KeyError(f"Could not find layer norm for layer {layer}")

    ln1 = state_dict[ln1_key].detach().cpu().numpy()'''

    content = content.replace(old_ln1, new_ln1)

    old_ln2 = '''ln2 = get_tensor(state_dict, f"{prefix}.post_attention_layernorm.weight").detach().cpu().numpy()'''
    new_ln2 = '''# Flexible post-attention LayerNorm detection
    ln2_key = f"{prefix}.post_attention_layernorm.weight"
    if ln2_key not in state_dict:
        alt_key = f"{prefix}.ffn_norm.weight"
        if alt_key in state_dict:
            ln2_key = alt_key
        else:
            raise KeyError(f"Could not find post-attention layer norm for layer {layer}")

    ln2 = state_dict[ln2_key].detach().cpu().numpy()'''

    content = content.replace(old_ln2, new_ln2)

    # Write patched version
    with open(output_path, "w") as f:
        f.write(content)

    print("✅ Converter patched successfully!")
    print("\nChanges made:")
    print("  - Added detect_architecture() function")
    print("  - Flexible embedding layer detection")
    print("  - Flexible LayerNorm detection")
    print("  - Supports LLaMA, Devstral, SmolLM")


def main():
    parser = argparse.ArgumentParser(description="Patch bump converter for Devstral/SmolLM support")
    parser.add_argument("--input", required=True, help="Input converter script")
    parser.add_argument("--output", help="Output path (defaults to input)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed")
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN - No changes will be made\n")
        print("This patch will:")
        print("  1. Add architecture detection function")
        print("  2. Make embedding layer lookup flexible")
        print("  3. Make LayerNorm lookup flexible")
        print("  4. Support LLaMA, Devstral, SmolLM out of the box")
        return

    patch_converter(args.input, args.output)


if __name__ == "__main__":
    main()
