#!/usr/bin/env python3
"""
bump_layout_v6_6.py - Generate explicit memory layout from lowered IR.

Creates:
1. Header file with offset structs (HeaderOffsets, LayerOffsets, FooterOffsets)
2. Static const arrays with precomputed offsets for each layer
3. Helper macros for pointer access

Usage:
    python bump_layout_v6_6.py --ir lowered_decode.json --output model_layout.h
"""

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple


@dataclass
class BufferInfo:
    """Information about a single buffer (weight or activation)."""
    name: str
    offset: int
    size: int
    dtype: str
    shape: List[int] = field(default_factory=list)
    role: str = "weight"  # weight, activation, scratch


@dataclass
class LayerLayout:
    """Memory layout for a single layer."""
    layer_id: int
    weights: Dict[str, BufferInfo] = field(default_factory=dict)
    activations: Dict[str, BufferInfo] = field(default_factory=dict)
    scratch: Dict[str, BufferInfo] = field(default_factory=dict)


@dataclass
class ModelLayout:
    """Complete memory layout for the model."""
    model_name: str
    config: Dict
    header: Dict[str, BufferInfo] = field(default_factory=dict)
    layers: List[LayerLayout] = field(default_factory=list)
    footer: Dict[str, BufferInfo] = field(default_factory=dict)
    total_weight_bytes: int = 0
    total_activation_bytes: int = 0


def extract_layout_from_ir(ir_data: Dict) -> ModelLayout:
    """Extract memory layout from lowered IR."""
    config = ir_data.get("config", {})
    memory = ir_data.get("memory", {})
    operations = ir_data.get("operations", [])

    layout = ModelLayout(
        model_name=config.get("model", "model"),
        config=config,
        total_weight_bytes=memory.get("weights", {}).get("size", 0),
        total_activation_bytes=memory.get("activations", {}).get("size", 0),
    )

    # Initialize layers
    num_layers = config.get("num_layers", 24)
    layout.layers = [LayerLayout(layer_id=i) for i in range(num_layers)]

    # Track which buffers we've seen
    seen_weights: Set[Tuple[str, int]] = set()
    seen_activations: Set[Tuple[str, int]] = set()

    # Process each operation
    for op in operations:
        layer = op.get("layer", -1)
        section = op.get("section", "body")

        # Process weights
        for name, w in op.get("weights", {}).items():
            key = (w.get("name", name), w.get("bump_offset", 0))
            if key in seen_weights:
                continue
            seen_weights.add(key)

            buf = BufferInfo(
                name=name,
                offset=w.get("bump_offset", 0),
                size=w.get("size", 0),
                dtype=w.get("dtype", "fp32"),
                role="weight"
            )

            if section == "header" or layer == -1:
                layout.header[name] = buf
            elif section == "footer":
                layout.footer[name] = buf
            elif 0 <= layer < num_layers:
                layout.layers[layer].weights[name] = buf

        # Process activations
        for name, a in op.get("activations", {}).items():
            key = (name, a.get("activation_offset", 0))
            if key in seen_activations:
                continue
            seen_activations.add(key)

            buf = BufferInfo(
                name=name,
                offset=a.get("activation_offset", 0),
                size=0,  # Size determined by shape
                dtype=a.get("dtype", "fp32"),
                role="activation"
            )

            if layer == -1:
                # Global activation
                pass  # We handle these separately
            elif 0 <= layer < num_layers:
                layout.layers[layer].activations[name] = buf

        # Process outputs
        for name, o in op.get("outputs", {}).items():
            buf = BufferInfo(
                name=name,
                offset=o.get("activation_offset", 0),
                size=0,
                dtype=o.get("dtype", "fp32"),
                role="activation"
            )
            if 0 <= layer < num_layers:
                layout.layers[layer].activations[f"out_{name}"] = buf

    return layout


def generate_header_file(layout: ModelLayout, mode: str) -> str:
    """Generate C header file with offset structs."""

    model_upper = layout.model_name.upper().replace("-", "_")
    mode_upper = mode.upper()
    prefix = f"{model_upper}_{mode_upper}"

    lines = [
        f"/**",
        f" * @file {layout.model_name}_{mode}_layout.h",
        f" * @brief AUTO-GENERATED: {layout.model_name} Memory Layout ({mode})",
        f" *",
        f" * Generated: {datetime.now().isoformat()} UTC",
        f" * Total Weights: {layout.total_weight_bytes:,} bytes",
        f" *",
        f" * DO NOT EDIT - Regenerate with bump_layout_v6_6.py",
        f" */",
        f"",
        f"#ifndef {prefix}_LAYOUT_H",
        f"#define {prefix}_LAYOUT_H",
        f"",
        f"#include <stddef.h>",
        f"#include <stdint.h>",
        f"",
        f"/* ============================================================================",
        f" * MODEL CONFIGURATION",
        f" * ============================================================================ */",
        f"",
    ]

    # Config defines
    config = layout.config
    lines.append(f"#define {prefix}_EMBED_DIM          {config.get('embed_dim', 0)}")
    lines.append(f"#define {prefix}_NUM_HEADS          {config.get('num_heads', 0)}")
    lines.append(f"#define {prefix}_NUM_KV_HEADS       {config.get('num_kv_heads', 0)}")
    lines.append(f"#define {prefix}_HEAD_DIM           {config.get('head_dim', 0)}")
    lines.append(f"#define {prefix}_INTERMEDIATE       {config.get('intermediate_size', 0)}")
    lines.append(f"#define {prefix}_NUM_LAYERS         {config.get('num_layers', 0)}")
    lines.append(f"#define {prefix}_VOCAB_SIZE         {config.get('vocab_size', 0)}")
    lines.append(f"#define {prefix}_MAX_SEQ_LEN        {config.get('context_length', 32768)}")
    lines.append(f"#define {prefix}_ROPE_THETA         {config.get('rope_theta', 10000.0)}f")
    lines.append(f"")
    lines.append(f"#define {prefix}_WEIGHT_BYTES       {layout.total_weight_bytes}ULL")
    lines.append(f"#define {prefix}_ACTIVATION_BYTES   {layout.total_activation_bytes}ULL")
    lines.append(f"")

    # Pointer macro
    lines.append(f"/* Pointer access macro */")
    lines.append(f"#define {prefix}_WEIGHT_PTR(model, offset) ((void*)((uint8_t*)(model)->bump_weights + (offset)))")
    lines.append(f"#define {prefix}_ACT_PTR(model, offset) ((float*)((uint8_t*)(model)->activations + (offset)))")
    lines.append(f"")

    # Header offsets struct
    if layout.header:
        lines.append(f"/* ============================================================================")
        lines.append(f" * HEADER OFFSETS (embedding, vocab, etc.)")
        lines.append(f" * ============================================================================ */")
        lines.append(f"")
        lines.append(f"typedef struct {{")
        for name, buf in sorted(layout.header.items()):
            lines.append(f"    size_t {name};  /* {buf.dtype}, {buf.size} bytes */")
        lines.append(f"}} {prefix}HeaderOffsets;")
        lines.append(f"")
        lines.append(f"static const {prefix}HeaderOffsets {prefix}_HEADER = {{")
        for name, buf in sorted(layout.header.items()):
            lines.append(f"    .{name} = {buf.offset},")
        lines.append(f"}};")
        lines.append(f"")

    # Collect all unique weight names across layers
    all_weight_names: Set[str] = set()
    for layer in layout.layers:
        all_weight_names.update(layer.weights.keys())

    # Layer offsets struct
    if all_weight_names:
        lines.append(f"/* ============================================================================")
        lines.append(f" * LAYER OFFSETS (per-layer weights)")
        lines.append(f" * ============================================================================ */")
        lines.append(f"")
        lines.append(f"typedef struct {{")
        for name in sorted(all_weight_names):
            # Find first layer with this weight to get dtype
            dtype = "fp32"
            for layer in layout.layers:
                if name in layer.weights:
                    dtype = layer.weights[name].dtype
                    break
            lines.append(f"    size_t {name};  /* {dtype} */")
        lines.append(f"}} {prefix}LayerOffsets;")
        lines.append(f"")

        # Layer offset arrays
        lines.append(f"static const {prefix}LayerOffsets {prefix}_LAYERS[{len(layout.layers)}] = {{")
        for layer in layout.layers:
            lines.append(f"    [{layer.layer_id}] = {{")
            for name in sorted(all_weight_names):
                offset = layer.weights.get(name, BufferInfo(name, 0, 0, "")).offset
                lines.append(f"        .{name} = {offset},")
            lines.append(f"    }},")
        lines.append(f"}};")
        lines.append(f"")

    # Footer offsets struct
    if layout.footer:
        lines.append(f"/* ============================================================================")
        lines.append(f" * FOOTER OFFSETS (final norm, output projection)")
        lines.append(f" * ============================================================================ */")
        lines.append(f"")
        lines.append(f"typedef struct {{")
        for name, buf in sorted(layout.footer.items()):
            lines.append(f"    size_t {name};  /* {buf.dtype}, {buf.size} bytes */")
        lines.append(f"}} {prefix}FooterOffsets;")
        lines.append(f"")
        lines.append(f"static const {prefix}FooterOffsets {prefix}_FOOTER = {{")
        for name, buf in sorted(layout.footer.items()):
            lines.append(f"    .{name} = {buf.offset},")
        lines.append(f"}};")
        lines.append(f"")

    lines.append(f"#endif /* {prefix}_LAYOUT_H */")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate memory layout from lowered IR")
    parser.add_argument("--ir", type=Path, required=True, help="Lowered IR JSON file")
    parser.add_argument("--output", type=Path, required=True, help="Output header file")
    parser.add_argument("--mode", type=str, default="decode", help="Mode (decode/prefill)")

    args = parser.parse_args()

    # Load IR
    with open(args.ir, 'r') as f:
        ir_data = json.load(f)

    # Extract layout
    layout = extract_layout_from_ir(ir_data)

    # Generate header
    header_content = generate_header_file(layout, args.mode)

    # Write output
    with open(args.output, 'w') as f:
        f.write(header_content)

    print(f"Generated {args.output}")
    print(f"  Header buffers: {len(layout.header)}")
    print(f"  Layers: {len(layout.layers)}")
    print(f"  Layer weights per layer: {len(layout.layers[0].weights) if layout.layers else 0}")
    print(f"  Footer buffers: {len(layout.footer)}")


if __name__ == "__main__":
    main()
