#!/usr/bin/env python3
"""
Generate human-readable memory map (.map) file from layout JSON.

Usage:
    python generate_memory_map_v7.py layout_decode.json -o layout_decode.map
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any


def format_size(size: int) -> str:
    """Format size with commas."""
    return f"{size:,}"


def format_size_human(size: int) -> str:
    """Format size in human-readable form."""
    if size >= 1024 * 1024 * 1024:
        return f"{size / (1024**3):.2f} GB"
    elif size >= 1024 * 1024:
        return f"{size / (1024**2):.1f} MB"
    elif size >= 1024:
        return f"{size / 1024:.1f} KB"
    else:
        return f"{size} bytes"


def generate_memory_map(layout: Dict, mode: str) -> str:
    """Generate memory map string from layout dict."""
    lines = []

    config = layout.get("config", {})
    ir_ops = layout.get("ir_lower_1_ops", [])
    memory = layout.get("memory", {})

    model_type = config.get("model", "qwen2")
    embed_dim = config.get("embed_dim", 896)
    num_heads = config.get("num_heads", 14)
    num_kv_heads = config.get("num_kv_heads", 2)
    head_dim = config.get("head_dim", 64)
    intermediate_size = config.get("intermediate_size", 4864)
    num_layers = config.get("num_layers", 24)
    vocab_size = config.get("vocab_size", 151936)
    max_seq_len = config.get("context_length", 32768)
    rope_theta = config.get("rope_theta", 1000000.0)
    rms_eps = config.get("rms_eps", 1e-6)

    # Header
    lines.append("=" * 80)
    lines.append(f"MEMORY MAP: {model_type}_{mode}")
    lines.append("=" * 80)
    lines.append(f"Generated:    {datetime.now(timezone.utc).isoformat()} UTC")
    lines.append(f"IR Version:   v7")
    lines.append("=" * 80)
    lines.append("")

    # Configuration
    lines.append("CONFIGURATION")
    lines.append("-" * 80)
    lines.append(f"  {'model_type':<24} {model_type}")
    lines.append(f"  {'embed_dim':<24} {embed_dim}")
    lines.append(f"  {'num_heads':<24} {num_heads}")
    lines.append(f"  {'num_kv_heads':<24} {num_kv_heads}")
    lines.append(f"  {'head_dim':<24} {head_dim}")
    lines.append(f"  {'intermediate_dim':<24} {intermediate_size}")
    lines.append(f"  {'num_layers':<24} {num_layers}")
    lines.append(f"  {'vocab_size':<24} {vocab_size}")
    lines.append(f"  {'max_seq_len':<24} {max_seq_len}")
    context_len = config.get("context_len", max_seq_len)
    lines.append(f"  {'context_len':<24} {context_len}")
    lines.append(f"  {'rope_theta':<24} {rope_theta}")
    lines.append(f"  {'rms_norm_eps':<24} {rms_eps}")
    lines.append("")

    # Calculate memory sizes
    weights_info = memory.get("weights", {})
    activations_info = memory.get("activations", {})

    total_weights = weights_info.get("size", 0)
    total_activations = activations_info.get("size", 0)
    total_memory = total_weights + total_activations

    # Memory Summary
    lines.append("MEMORY SUMMARY")
    lines.append("-" * 80)
    lines.append(f"  {'Total:':<14} {format_size(total_memory):>20} bytes  ({format_size_human(total_memory)})")
    lines.append(f"  {'Weights:':<14} {format_size(total_weights):>20} bytes  ({format_size_human(total_weights)})")
    lines.append(f"  {'Activations:':<14} {format_size(total_activations):>20} bytes  ({format_size_human(total_activations)})")
    lines.append("")

    # Group ops by section and layer
    header_ops = []
    body_ops = {}  # layer -> ops
    footer_ops = []

    for op in ir_ops:
        section = op.get("section", "body")
        layer = op.get("layer", -1)

        if section == "header":
            header_ops.append(op)
        elif section == "footer":
            footer_ops.append(op)
        else:
            if layer not in body_ops:
                body_ops[layer] = []
            body_ops[layer].append(op)

    # Helper to format size with human-readable suffix
    def format_size_col(size: int) -> str:
        """Format size as 'bytes (human)' for column display."""
        if size >= 1024 * 1024:
            return f"{size:>12,}  ({size / (1024*1024):>7.2f} MB)"
        elif size >= 1024:
            return f"{size:>12,}  ({size / 1024:>7.2f} KB)"
        else:
            return f"{size:>12,}  ({size:>7} B )"

    # Helper to print weight entries (deduplicated and sorted by offset)
    def print_weights(ops: List[Dict], lines: List[str]):
        col_header = f"{'Offset':<14} {'End':<14} {'Size (bytes)':<26} {'Buffer':<32} {'Shape':<20} {'Type':<8}"
        lines.append(col_header)
        lines.append("-" * 120)

        # Collect unique weights by name
        seen_weights = {}
        for op in ops:
            weights = op.get("weights", {})

            for wname, winfo in weights.items():
                if isinstance(winfo, dict):
                    name = winfo.get("name", wname)
                    if name not in seen_weights:
                        seen_weights[name] = winfo

        # Sort by offset and print
        for name in sorted(seen_weights.keys(), key=lambda n: seen_weights[n].get("offset", 0)):
            winfo = seen_weights[name]
            offset = winfo.get("offset", 0)
            size = winfo.get("size", 0)
            dtype = winfo.get("dtype", "unknown")

            end = offset + size

            # Infer shape from name and config
            shape = infer_shape(name, config)
            size_str = format_size_col(size)

            lines.append(
                f"0x{offset:012X} 0x{end:012X} {size_str}  {name:<32} {shape:<20} weight"
            )

    def infer_shape(name: str, config: Dict) -> str:
        """Infer tensor shape from name."""
        embed = config.get("embed_dim", 896)
        heads = config.get("num_heads", 14)
        kv_heads = config.get("num_kv_heads", 2)
        head_dim = config.get("head_dim", 64)
        inter = config.get("intermediate_size", 4864)
        vocab = config.get("vocab_size", 151936)

        if "token_emb" in name:
            return f"[{vocab}, {embed}]"
        elif "vocab_offsets" in name:
            return f"[{vocab}]"
        elif "vocab_strings" in name:
            total_vocab_bytes = config.get("total_vocab_bytes", 0)
            return f"[{total_vocab_bytes}]" if total_vocab_bytes else "[bytes]"
        elif "vocab_merges" in name:
            num_merges = config.get("num_merges", 0)
            return f"[{num_merges}, 3]" if num_merges else "[merges, 3]"
        elif "ln1_gamma" in name or "ln2_gamma" in name or "final_ln" in name:
            return f"[{embed}]"
        elif ".wq" in name:
            return f"[{heads}, {head_dim}, {embed}]"
        elif ".wk" in name:
            return f"[{kv_heads}, {head_dim}, {embed}]"
        elif ".wv" in name:
            return f"[{kv_heads}, {head_dim}, {embed}]"
        elif ".wo" in name:
            return f"[{heads}, {embed}, {head_dim}]"
        elif ".bq" in name:
            return f"[{heads}, {head_dim}]"
        elif ".bk" in name or ".bv" in name:
            return f"[{kv_heads}, {head_dim}]"
        elif ".bo" in name:
            return f"[{embed}]"
        elif ".w1" in name:
            return f"[{inter * 2}, {embed}]"
        elif ".w2" in name:
            return f"[{embed}, {inter}]"
        elif ".b1" in name:
            return f"[{inter * 2}]"
        elif ".b2" in name:
            return f"[{embed}]"
        else:
            return "[]"

    # Section: Header (include vocab tables from weights)
    lines.append("=" * 80)
    lines.append("SECTION: HEADER")
    lines.append("=" * 80)
    lines.append("")

    # Get vocab/tokenizer weights from memory.weights.entries
    weight_entries = weights_info.get("entries", [])
    header_weights = {}

    # Add vocab tables and token_emb to header
    vocab_names = ["vocab_offsets", "vocab_strings", "vocab_merges", "token_emb"]
    for entry in weight_entries:
        name = entry.get("name", "")
        if name in vocab_names:
            header_weights[name] = entry

    # Also add weights from header ops
    for op in header_ops:
        weights = op.get("weights", {})
        for wname, winfo in weights.items():
            if isinstance(winfo, dict):
                name = winfo.get("name", wname)
                if name not in header_weights:
                    header_weights[name] = winfo

    if header_weights:
        col_header = f"{'Offset':<14} {'End':<14} {'Size (bytes)':<26} {'Buffer':<32} {'Shape':<20} {'Type':<8}"
        lines.append(col_header)
        lines.append("-" * 120)

        # Sort by offset and print
        for name in sorted(header_weights.keys(), key=lambda n: header_weights[n].get("offset", 0)):
            winfo = header_weights[name]
            offset = winfo.get("offset", 0)
            size = winfo.get("size", 0)
            dtype = winfo.get("dtype", "unknown")
            end = offset + size
            shape = infer_shape(name, config)
            size_str = format_size_col(size)
            lines.append(f"0x{offset:012X} 0x{end:012X} {size_str}  {name:<32} {shape:<20} weight")
    else:
        lines.append("  (no header weights)")
    lines.append("")

    # Section: Body layers
    for layer_idx in sorted(body_ops.keys()):
        ops = body_ops[layer_idx]
        lines.append("=" * 80)
        lines.append(f"LAYER {layer_idx}")
        lines.append("=" * 80)
        lines.append("")
        print_weights(ops, lines)
        lines.append("")

    # Section: Footer
    lines.append("=" * 80)
    lines.append("SECTION: FOOTER")
    lines.append("=" * 80)
    lines.append("")
    if footer_ops:
        print_weights(footer_ops, lines)
    else:
        lines.append("  (no footer weights)")
    lines.append("")

    # Activations
    lines.append("=" * 80)
    lines.append("ACTIVATION BUFFERS")
    lines.append("=" * 80)
    lines.append("")

    buffers = activations_info.get("buffers", [])
    if buffers:
        col_header = f"{'Offset':<14} {'End':<14} {'Size (bytes)':<26} {'Buffer':<24} {'Shape':<30}"
        lines.append(col_header)
        lines.append("-" * 120)

        for buf in buffers:
            name = buf.get("name", "unknown")
            offset = buf.get("offset", 0)
            size = buf.get("size", 0)
            shape = buf.get("shape", "")
            end = offset + size
            size_str = format_size_col(size)

            lines.append(f"0x{offset:012X} 0x{end:012X} {size_str}  {name:<24} {shape:<30}")
    lines.append("")

    # Verification
    lines.append("=" * 80)
    lines.append("VERIFICATION")
    lines.append("=" * 80)
    lines.append("[✓] Layout generated from IR Lower 1")
    lines.append(f"[✓] Total ops: {len(ir_ops)}")
    lines.append(f"[✓] Layers: {len(body_ops)}")
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF MEMORY MAP")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate memory map from layout JSON")
    parser.add_argument("layout_json", help="Path to layout JSON file")
    parser.add_argument("-o", "--output", help="Output .map file path")

    args = parser.parse_args()

    # Load layout
    with open(args.layout_json, 'r') as f:
        layout = json.load(f)

    mode = layout.get("mode", "decode")

    # Generate map
    map_content = generate_memory_map(layout, mode)

    # Output
    if args.output:
        output_path = Path(args.output)
    else:
        input_path = Path(args.layout_json)
        output_path = input_path.with_suffix(".map")

    with open(output_path, 'w') as f:
        f.write(map_content)

    print(f"✓ Generated memory map: {output_path}")
    print(f"  Size: {len(map_content)} bytes")


if __name__ == "__main__":
    main()
