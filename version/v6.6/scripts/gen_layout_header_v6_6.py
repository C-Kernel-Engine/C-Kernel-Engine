#!/usr/bin/env python3
"""
gen_layout_header_v6_6.py - Generate C layout header from lowered IR.

This creates a clean C header file (like v6.5's ck-kernel-inference.h) that contains:
1. Model configuration #defines
2. Memory region offsets
3. HeaderOffsets struct (token_emb, final_ln, etc.)
4. LayerOffsets struct (per-layer weights)
5. ActivationOffsets struct (activation buffers)
6. Per-layer dtype arrays
7. Single allocation model struct
8. Accessor macros

Usage:
    python gen_layout_header_v6_6.py --lowered-ir=lowered_decode.json --output=model_layout.h
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def align_up(n: int, alignment: int = 64) -> int:
    """Align n up to the nearest multiple of alignment."""
    return (n + alignment - 1) & ~(alignment - 1)


def format_hex(offset: int) -> str:
    """Format offset as hex with proper width."""
    return f"0x{offset:08X}ULL"


def sanitize_name(name: str) -> str:
    """Convert model name to C identifier."""
    return name.upper().replace("-", "_").replace(".", "_").replace("/", "_")


def dtype_to_ck_enum(dtype: str) -> str:
    """Convert dtype string to CKDataType enum."""
    mapping = {
        "fp32": "CK_DT_FP32",
        "f32": "CK_DT_FP32",
        "fp16": "CK_DT_FP16",
        "bf16": "CK_DT_BF16",
        "q8_0": "CK_DT_Q8_0",
        "q5_0": "CK_DT_Q5_0",
        "q4_0": "CK_DT_Q4_0",
        "q4_k": "CK_DT_Q4_K",
        "q6_k": "CK_DT_Q6_K",
        "q8_k": "CK_DT_Q8_K",
    }
    return mapping.get(dtype.lower(), "CK_DT_FP32")


def generate_layout_header(lowered_ir: Dict, model_name: str = "model") -> str:
    """Generate complete C header from lowered IR."""

    config = lowered_ir.get("config", {})
    memory = lowered_ir.get("memory", {})
    operations = lowered_ir.get("operations", [])
    mode = lowered_ir.get("mode", "decode")

    # Extract config
    embed_dim = config.get("embed_dim", 896)
    num_heads = config.get("num_heads", 14)
    num_kv_heads = config.get("num_kv_heads", 2)
    head_dim = config.get("head_dim", 64)
    intermediate_size = config.get("intermediate_size", 4864)
    num_layers = config.get("num_layers", 24)
    vocab_size = config.get("vocab_size", 151936)
    context_length = config.get("context_length", 32768)
    rope_theta = config.get("rope_theta", 10000.0)
    rms_eps = config.get("rms_eps", 1e-6)

    # Extract memory info
    weights_info = memory.get("weights", {})
    activations_info = memory.get("activations", {})

    weights_size = weights_info.get("bump_size", weights_info.get("size", 0))
    activations_size = activations_info.get("size", 0)

    # Build weight offset lookup
    weight_entries = weights_info.get("entries", [])
    weight_lookup = {e["name"]: e for e in weight_entries}

    # Build activation buffer lookup
    activation_buffers = activations_info.get("buffers", [])
    activation_lookup = {b["name"]: b for b in activation_buffers}

    # Sanitize model name for C
    prefix = sanitize_name(model_name)
    prefix_mode = f"{prefix}_{mode.upper()}"

    # Calculate total size (single contiguous allocation)
    kv_cache_buf = activation_lookup.get("kv_cache", {})
    rope_cache_buf = activation_lookup.get("rope_cache", {})
    logits_buf = activation_lookup.get("logits", {})

    kv_cache_size = kv_cache_buf.get("size", num_layers * 2 * num_kv_heads * context_length * head_dim * 4)
    rope_cache_size = rope_cache_buf.get("size", context_length * head_dim * 4)
    logits_size = logits_buf.get("size", vocab_size * 4)

    total_size = weights_size + activations_size

    # Build layer weight info
    layer_weights = {}  # layer_id -> {wq: {offset, dtype, size}, ...}
    header_weights = {}  # name -> {offset, dtype, size}
    footer_weights = {}

    for name, entry in weight_lookup.items():
        offset = entry.get("offset", 0)
        dtype = entry.get("dtype", "fp32")
        size = entry.get("size", 0)

        if name.startswith("layer."):
            parts = name.split(".")
            if len(parts) >= 3:
                layer_id = int(parts[1])
                field = parts[2]
                if layer_id not in layer_weights:
                    layer_weights[layer_id] = {}
                layer_weights[layer_id][field] = {"offset": offset, "dtype": dtype, "size": size}
        elif name in ["token_emb", "vocab_offsets", "vocab_strings", "vocab_merges"]:
            header_weights[name] = {"offset": offset, "dtype": dtype, "size": size}
        elif name in ["final_ln_weight", "final_ln_bias", "output_weight"]:
            footer_weights[name] = {"offset": offset, "dtype": dtype, "size": size}

    lines = []

    def add(s=""):
        lines.append(s)

    # File header
    add(f"""/**
 * @file {model_name}_{mode}_layout.h
 * @brief AUTO-GENERATED Memory Layout for {model_name}
 *
 * Generated: {datetime.utcnow().isoformat()} UTC
 * Mode: {mode}
 * Total Memory: {total_size / 1e9:.2f} GB
 *
 * DO NOT EDIT - Regenerate with gen_layout_header_v6_6.py
 */

#ifndef {prefix_mode}_LAYOUT_H
#define {prefix_mode}_LAYOUT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif
""")

    # Configuration defines
    add(f"""/* ============================================================================
 * MODEL CONFIGURATION
 * ============================================================================ */

#define {prefix_mode}_EMBED_DIM          {embed_dim}
#define {prefix_mode}_NUM_HEADS          {num_heads}
#define {prefix_mode}_NUM_KV_HEADS       {num_kv_heads}
#define {prefix_mode}_HEAD_DIM           {head_dim}
#define {prefix_mode}_INTERMEDIATE       {intermediate_size}
#define {prefix_mode}_NUM_LAYERS         {num_layers}
#define {prefix_mode}_VOCAB_SIZE         {vocab_size}
#define {prefix_mode}_MAX_SEQ_LEN        {context_length}
#define {prefix_mode}_ROPE_THETA         {rope_theta}f
#define {prefix_mode}_RMS_EPS            {rms_eps}f

#define {prefix_mode}_TOTAL_BYTES        {total_size}ULL
#define {prefix_mode}_WEIGHT_BYTES       {weights_size}ULL
#define {prefix_mode}_ACTIVATION_BYTES   {activations_size}ULL
#define {prefix_mode}_KV_CACHE_BYTES     {kv_cache_size}ULL
""")

    # Header offsets struct
    add(f"""/* ============================================================================
 * HEADER OFFSETS (embedding, vocab)
 * ============================================================================ */

typedef struct {{""")

    for name in ["token_emb", "vocab_offsets", "vocab_strings", "vocab_merges"]:
        if name in header_weights:
            w = header_weights[name]
            add(f"    size_t {name};  /* offset={w['offset']}, size={w['size']}, dtype={w['dtype']} */")

    add(f"}} {prefix_mode}HeaderOffsets;")
    add("")
    add(f"static const {prefix_mode}HeaderOffsets {prefix_mode}_HEADER = {{")

    for name in ["token_emb", "vocab_offsets", "vocab_strings", "vocab_merges"]:
        if name in header_weights:
            add(f"    .{name} = {format_hex(header_weights[name]['offset'])},")

    add("};")
    add("")

    # Layer offsets struct
    add(f"""/* ============================================================================
 * LAYER OFFSETS (per-layer weights)
 * ============================================================================ */

typedef struct {{
    /* Attention weights */
    size_t ln1_gamma;   /* Layer norm 1 */
    size_t wq;          /* Q projection */
    size_t bq;          /* Q bias */
    size_t wk;          /* K projection */
    size_t bk;          /* K bias */
    size_t wv;          /* V projection */
    size_t bv;          /* V bias */
    size_t wo;          /* Output projection */
    size_t bo;          /* Output bias */

    /* MLP weights */
    size_t ln2_gamma;   /* Layer norm 2 */
    size_t w1;          /* Gate/Up projection */
    size_t b1;          /* Gate/Up bias */
    size_t w2;          /* Down projection */
    size_t b2;          /* Down bias */
}} {prefix_mode}LayerOffsets;

static const {prefix_mode}LayerOffsets {prefix_mode}_LAYERS[{num_layers}] = {{""")

    # Generate per-layer offsets
    layer_fields = ["ln1_gamma", "wq", "bq", "wk", "bk", "wv", "bv", "wo", "bo",
                    "ln2_gamma", "w1", "b1", "w2", "b2"]

    for layer_id in range(num_layers):
        lw = layer_weights.get(layer_id, {})
        add(f"    [{layer_id}] = {{")
        for field in layer_fields:
            if field in lw:
                add(f"        .{field} = {format_hex(lw[field]['offset'])},")
            else:
                add(f"        .{field} = 0,  /* not present */")
        add(f"    }},")

    add("};")
    add("")

    # Per-layer dtype arrays
    add(f"""/* ============================================================================
 * PER-LAYER DTYPE ARRAYS (for mixed quantization)
 * ============================================================================ */

typedef enum {{
    CK_DT_FP32 = 0,
    CK_DT_FP16,
    CK_DT_BF16,
    CK_DT_Q8_0,
    CK_DT_Q5_0,
    CK_DT_Q4_0,
    CK_DT_Q4_K,
    CK_DT_Q6_K,
    CK_DT_Q8_K,
}} CKDataType;
""")

    # Generate dtype arrays for each weight type
    for weight_name in ["wq", "wk", "wv", "wo", "w1", "w2"]:
        dtypes = []
        for layer_id in range(num_layers):
            lw = layer_weights.get(layer_id, {})
            if weight_name in lw:
                dtypes.append(dtype_to_ck_enum(lw[weight_name]["dtype"]))
            else:
                dtypes.append("CK_DT_FP32")

        add(f"static const CKDataType {prefix_mode}_LAYER_{weight_name.upper()}_DTYPE[] = {{")
        add("    " + ", ".join(dtypes))
        add("};")
        add("")

    # Footer offsets
    add(f"""/* ============================================================================
 * FOOTER OFFSETS (final norm, output projection)
 * ============================================================================ */

typedef struct {{
    size_t final_ln_weight;
    size_t final_ln_bias;
}} {prefix_mode}FooterOffsets;

static const {prefix_mode}FooterOffsets {prefix_mode}_FOOTER = {{""")

    if "final_ln_weight" in footer_weights:
        add(f"    .final_ln_weight = {format_hex(footer_weights['final_ln_weight']['offset'])},")
    else:
        add(f"    .final_ln_weight = 0,")
    if "final_ln_bias" in footer_weights:
        add(f"    .final_ln_bias = {format_hex(footer_weights['final_ln_bias']['offset'])},")
    else:
        add(f"    .final_ln_bias = 0,")

    add("};")
    add("")

    # Activation buffer offsets
    add(f"""/* ============================================================================
 * ACTIVATION BUFFER OFFSETS
 * ============================================================================ */

typedef struct {{""")

    for buf in activation_buffers:
        name = buf["name"]
        offset = buf["offset"]
        size = buf["size"]
        shape = buf.get("shape", "")
        add(f"    size_t {name};  /* {shape}, size={size} */")

    add(f"}} {prefix_mode}ActivationOffsets;")
    add("")
    add(f"static const {prefix_mode}ActivationOffsets {prefix_mode}_ACT = {{")

    for buf in activation_buffers:
        add(f"    .{buf['name']} = {format_hex(buf['offset'])},")

    add("};")
    add("")

    # Model struct with single allocation
    add(f"""/* ============================================================================
 * MODEL STRUCT (SINGLE contiguous allocation)
 * ============================================================================ */

typedef struct {{
    void *base;           /* Single contiguous allocation */
    size_t total_size;    /* Total bytes allocated */
    int pos;              /* Current sequence position */
}} {prefix_mode}Model;

/* ============================================================================
 * ACCESSOR MACROS
 * ============================================================================ */

/* Generic pointer access from base */
#define {prefix_mode}_PTR(model, offset) \\
    ((void*)((uint8_t*)(model)->base + (offset)))

/* Typed pointer access */
#define {prefix_mode}_FLOAT(model, offset) \\
    ((float*)((uint8_t*)(model)->base + (offset)))

#define {prefix_mode}_INT32(model, offset) \\
    ((int32_t*)((uint8_t*)(model)->base + (offset)))

#define {prefix_mode}_UINT8(model, offset) \\
    ((uint8_t*)((uint8_t*)(model)->base + (offset)))

/* Weight access (from base) */
#define {prefix_mode}_WEIGHT(model, offset) \\
    {prefix_mode}_PTR(model, offset)

#define {prefix_mode}_WEIGHT_F(model, offset) \\
    {prefix_mode}_FLOAT(model, offset)

/* Activation buffer access (weights_size + activation offset) */
#define {prefix_mode}_ACT_PTR(model, act_offset) \\
    {prefix_mode}_FLOAT(model, {prefix_mode}_WEIGHT_BYTES + (act_offset))

/* Layer weight access */
#define {prefix_mode}_LN1_GAMMA(model, layer) \\
    {prefix_mode}_WEIGHT_F(model, {prefix_mode}_LAYERS[layer].ln1_gamma)

#define {prefix_mode}_WQ(model, layer) \\
    {prefix_mode}_WEIGHT(model, {prefix_mode}_LAYERS[layer].wq)

#define {prefix_mode}_WK(model, layer) \\
    {prefix_mode}_WEIGHT(model, {prefix_mode}_LAYERS[layer].wk)

#define {prefix_mode}_WV(model, layer) \\
    {prefix_mode}_WEIGHT(model, {prefix_mode}_LAYERS[layer].wv)

#define {prefix_mode}_WO(model, layer) \\
    {prefix_mode}_WEIGHT(model, {prefix_mode}_LAYERS[layer].wo)

/* KV cache access */
#define {prefix_mode}_KV_K(model, layer) \\
    {prefix_mode}_ACT_PTR(model, {prefix_mode}_ACT.kv_cache + (layer) * 2 * {prefix_mode}_MAX_SEQ_LEN * {prefix_mode}_NUM_KV_HEADS * {prefix_mode}_HEAD_DIM * sizeof(float))

#define {prefix_mode}_KV_V(model, layer) \\
    {prefix_mode}_ACT_PTR(model, {prefix_mode}_ACT.kv_cache + ((layer) * 2 + 1) * {prefix_mode}_MAX_SEQ_LEN * {prefix_mode}_NUM_KV_HEADS * {prefix_mode}_HEAD_DIM * sizeof(float))

/* RoPE cache access */
#define {prefix_mode}_ROPE_COS(model, pos) \\
    ({prefix_mode}_ACT_PTR(model, {prefix_mode}_ACT.rope_cache) + (pos) * {prefix_mode}_HEAD_DIM)

#define {prefix_mode}_ROPE_SIN(model, pos) \\
    ({prefix_mode}_ACT_PTR(model, {prefix_mode}_ACT.rope_cache + {prefix_mode}_MAX_SEQ_LEN * {prefix_mode}_HEAD_DIM / 2 * sizeof(float)) + (pos) * {prefix_mode}_HEAD_DIM)

/* Logits output */
#define {prefix_mode}_LOGITS(model) \\
    {prefix_mode}_ACT_PTR(model, {prefix_mode}_ACT.logits)

#ifdef __cplusplus
}}
#endif

#endif /* {prefix_mode}_LAYOUT_H */
""")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate C layout header from lowered IR")
    parser.add_argument("--lowered-ir", type=Path, required=True, help="Path to lowered IR JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output header file path")
    parser.add_argument("--model-name", type=str, default="model", help="Model name for C identifiers")

    args = parser.parse_args()

    # Load lowered IR
    with open(args.lowered_ir, 'r') as f:
        lowered_ir = json.load(f)

    # Generate header
    header = generate_layout_header(lowered_ir, args.model_name)

    # Write output
    with open(args.output, 'w') as f:
        f.write(header)

    print(f"Generated: {args.output}")


if __name__ == "__main__":
    main()
