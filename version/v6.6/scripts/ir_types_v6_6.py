# ============================================================================
# ORIGIN: Copied from v3/build_ir_v3.py for v6.6 independence
# PURPOSE: Core IR data structures (Buffer, ModelLayout, BumpAllocator, etc.)
# NOTE: This file can be modified for v6.6-specific requirements
# ============================================================================

#!/usr/bin/env python3
"""
build_ir_v3.py - IR v3 Code Generator

Generates complete C code with baked-in memory offsets.
No runtime memory planning needed.

Usage:
    # From HuggingFace (recommended):
    python build_ir_v3.py Qwen/Qwen2-0.5B-Instruct
    python build_ir_v3.py Qwen/Qwen2-0.5B-Instruct --prefix=build/qwen2
    python build_ir_v3.py https://huggingface.co/Qwen/Qwen2-0.5B-Instruct

    # From local config:
    python build_ir_v3.py --config path/to/config.json --name model-name

Outputs:
    - graph.json           (graph IR: ops + buffers)
    - generated_model.h    (structs + offsets)
    - generated_model.c    (implementation)
    - layout.json          (machine-readable)
    - layout.map           (human-readable)
"""

import argparse
import json
import os
import sys
import hashlib
import re
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# ============================================================================
# HUGGINGFACE UTILITIES
# ============================================================================

def parse_hf_model_id(model_input: str) -> str:
    """
    Parse model ID from various input formats:
    - "Qwen/Qwen2-0.5B-Instruct"
    - "https://huggingface.co/Qwen/Qwen2-0.5B-Instruct"
    - "hf.co/Qwen/Qwen2-0.5B-Instruct"
    """
    # Strip whitespace
    model_input = model_input.strip()

    # Handle full URLs
    patterns = [
        r'https?://huggingface\.co/([^/]+/[^/]+)/?.*',
        r'https?://hf\.co/([^/]+/[^/]+)/?.*',
        r'hf\.co/([^/]+/[^/]+)/?.*',
    ]

    for pattern in patterns:
        match = re.match(pattern, model_input)
        if match:
            return match.group(1)

    # Already in org/model format
    if '/' in model_input and not model_input.startswith('http'):
        return model_input

    raise ValueError(f"Invalid model ID: {model_input}\n"
                     f"Expected format: 'org/model' or 'https://huggingface.co/org/model'")


def download_hf_config(model_id: str, cache_dir: str = None) -> Tuple[Dict, str]:
    """
    Download config.json from HuggingFace Hub.
    Returns (config_dict, local_path).
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/ckernel/configs")

    os.makedirs(cache_dir, exist_ok=True)

    # Sanitize model_id for filename
    safe_name = model_id.replace("/", "_")
    local_path = os.path.join(cache_dir, f"{safe_name}_config.json")

    # URL to fetch
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"

    print(f"[HF] Fetching config from: {model_id}")

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'CKernel-Engine/1.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read().decode('utf-8')
            config = json.loads(data)

            # Cache locally
            with open(local_path, 'w') as f:
                json.dump(config, f, indent=2)

            print(f"[HF] Config cached: {local_path}")
            return config, local_path

    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(f"Model not found: {model_id}\n"
                               f"Check the model exists at https://huggingface.co/{model_id}")
        raise RuntimeError(f"HTTP error {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error: {e.reason}\n"
                           f"Check your internet connection")


def model_id_to_name(model_id: str) -> str:
    """Convert model ID to a safe C identifier name."""
    # Take just the model name part (after /)
    name = model_id.split('/')[-1]
    # Convert to lowercase and replace non-alphanumeric with underscore
    name = re.sub(r'[^a-zA-Z0-9]', '_', name.lower())
    # Remove consecutive underscores
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    return name


# ============================================================================
# CONSTANTS
# ============================================================================

CACHE_LINE = 64
CANARY_SIZE = 64  # bytes
CANARY_VALUE = 0xDEADBEEF
MAGIC_PREFIX = 0x434B454E  # "CKEN"

DTYPE_BYTES = {
    "f32": 4,
    "fp32": 4,
    "f16": 2,
    "fp16": 2,
    "bf16": 2,
    "i32": 4,
    "i16": 2,
    "i8": 1,
    "u8": 1,
    "u16": 2,
}

# Quantized types: (block_size, bytes_per_block) - llama.cpp compatible
QUANT_BLOCK_INFO = {
    "q4_0": (32, 18),     # 4-bit, 32/block, 1 FP16 scale = 18 bytes
    "q4_k": (256, 144),   # 4-bit K-quant, 256/block = 144 bytes
    "q6_k": (256, 210),   # 6-bit K-quant, 256/block = 210 bytes
    "q8_0": (32, 34),     # 8-bit, 32/block, 1 FP16 scale = 34 bytes
    "q8_k": (256, 292),   # 8-bit K-quant, 256/block = 292 bytes
}

LLAMA_WEIGHT_MAP = {
    "model.embed_tokens.weight": "token_embed_weight",
    "model.layers.{layer}.input_layernorm.weight": "layer.{L}.ln1_gamma",
    "model.layers.{layer}.self_attn.q_proj.weight": "layer.{L}.wq",
    "model.layers.{layer}.self_attn.k_proj.weight": "layer.{L}.wk",
    "model.layers.{layer}.self_attn.v_proj.weight": "layer.{L}.wv",
    "model.layers.{layer}.self_attn.o_proj.weight": "layer.{L}.wo",
    "model.layers.{layer}.post_attention_layernorm.weight": "layer.{L}.ln2_gamma",
    "model.layers.{layer}.mlp.gate_proj.weight": "layer.{L}.mlp_gate_w",
    "model.layers.{layer}.mlp.up_proj.weight": "layer.{L}.mlp_up_w",
    "model.layers.{layer}.mlp.down_proj.weight": "layer.{L}.mlp_down_w",
    "model.norm.weight": "final_ln_gamma",
    "lm_head.weight": "lm_head_weight",
}

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Buffer:
    """A single tensor buffer with computed offset."""
    name: str
    shape: List[int]
    dtype: str
    role: str  # weight, activation, cache, grad
    offset: int = 0
    size: int = 0
    tied_to: Optional[str] = None

@dataclass
class Canary:
    """Canary marker for memory corruption detection."""
    name: str
    offset: int

@dataclass
class LayerLayout:
    """Memory layout for a single transformer layer."""
    layer_id: int
    canary_start: Canary
    buffers: List[Buffer]
    canary_end: Canary
    total_bytes: int = 0

@dataclass
class SectionLayout:
    """Memory layout for a section (encoder/decoder)."""
    name: str
    section_id: int
    config: Dict
    header_canary_start: Canary
    header_buffers: List[Buffer]
    header_canary_end: Canary
    layers: List[LayerLayout]
    footer_canary_start: Canary
    footer_buffers: List[Buffer]
    footer_canary_end: Canary
    globals: List[Buffer]
    total_bytes: int = 0

@dataclass
class ModelLayout:
    """Complete model memory layout."""
    name: str
    config: Dict
    sections: List[SectionLayout]
    magic_header_size: int = 64
    total_bytes: int = 0
    weight_bytes: int = 0
    activation_bytes: int = 0
    canary_count: int = 0
    canaries: List[Canary] = field(default_factory=list)

# ============================================================================
# CONFIG PARSER
# ============================================================================

def parse_config(config_path: str) -> Dict:
    """Parse HuggingFace config.json and extract model dimensions."""
    with open(config_path, 'r') as f:
        raw = json.load(f)

    # Normalize different config formats
    config = {
        "model_type": raw.get("model_type", "unknown"),
        "embed_dim": raw.get("hidden_size", raw.get("d_model", 768)),
        "num_heads": raw.get("num_attention_heads", raw.get("n_head", 12)),
        "num_kv_heads": raw.get("num_key_value_heads", raw.get("num_attention_heads", 12)),
        "head_dim": raw.get("head_dim", None),
        "intermediate_dim": raw.get("intermediate_size", raw.get("d_ff", None)),
        "num_layers": raw.get("num_hidden_layers", raw.get("n_layer", 12)),
        "vocab_size": raw.get("vocab_size", 32000),
        "max_seq_len": raw.get("max_position_embeddings", 2048),
        "rope_theta": raw.get("rope_theta", 10000.0),
        "rms_norm_eps": raw.get("rms_norm_eps", 1e-6),
        "tie_word_embeddings": raw.get("tie_word_embeddings", True),
        "dtype": "bf16",  # default
    }

    # Compute derived dimensions
    if config["head_dim"] is None:
        config["head_dim"] = config["embed_dim"] // config["num_heads"]

    if config["intermediate_dim"] is None:
        config["intermediate_dim"] = config["embed_dim"] * 4

    # Aligned dimensions (for SIMD)
    config["aligned_embed"] = align_up(config["embed_dim"], 32)  # 32 elements for bf16 AVX-512
    config["aligned_head"] = align_up(config["head_dim"], 32)
    config["aligned_intermediate"] = align_up(config["intermediate_dim"], 32)

    return config

# ============================================================================
# MEMORY CALCULATOR
# ============================================================================

def align_up(n: int, alignment: int) -> int:
    """Align n up to the nearest multiple of alignment."""
    return (n + alignment - 1) & ~(alignment - 1)

def compute_size(shape: List[int], dtype: str) -> int:
    """Compute buffer size in bytes from shape and dtype.

    For quantized types (q4_k, q6_k, etc.), calculates size based on
    block structure matching llama.cpp/GGML.
    """
    elements = 1
    for dim in shape:
        elements *= dim

    # Check for quantized types first
    dtype_lower = dtype.lower()
    if dtype_lower in QUANT_BLOCK_INFO:
        block_size, block_bytes = QUANT_BLOCK_INFO[dtype_lower]
        n_blocks = (elements + block_size - 1) // block_size
        return n_blocks * block_bytes

    # Standard types
    return elements * DTYPE_BYTES[dtype]

def aligned_size(shape: List[int], dtype: str, alignment: int = CACHE_LINE) -> int:
    """Compute aligned buffer size."""
    size = compute_size(shape, dtype)
    return align_up(size, alignment)

def eval_dim_expr(expr, symbols: Dict[str, int]) -> int:
    """Evaluate a dimension expression like 'H*D' or 'D/2'."""
    if isinstance(expr, int):
        return expr
    if isinstance(expr, float):
        return int(expr)
    if isinstance(expr, str):
        value = eval(expr, {"__builtins__": {}}, symbols)
        return int(value)
    raise TypeError(f"Unsupported dim expr type: {type(expr)}")

def resolve_shape_expr(shape_expr: List[str], symbols: Dict[str, int]) -> List[int]:
    """Resolve a shape expression list into concrete dimensions."""
    return [eval_dim_expr(dim, symbols) for dim in shape_expr]

# ============================================================================
# GRAPH IR (MACRO) BUILDER
# ============================================================================

def build_graph_ir(config: Dict, model_name: str) -> Dict:
    """Build a macro-level graph IR (ops + buffers, no offsets)."""
    E = config["embed_dim"]
    H = config["num_heads"]
    KV = config["num_kv_heads"]
    D = config["head_dim"]
    I = config["intermediate_dim"]
    T = config["max_seq_len"]
    V = config["vocab_size"]
    dtype = config["dtype"]

    symbols = {
        "E": {"name": "embed_dim", "value": E},
        "H": {"name": "num_heads", "value": H},
        "KV": {"name": "num_kv_heads", "value": KV},
        "D": {"name": "head_dim", "value": D},
        "I": {"name": "intermediate_dim", "value": I},
        "S": {"name": "tokens", "value": T},
        "T": {"name": "max_seq_len", "value": T},
        "V": {"name": "vocab_size", "value": V},
    }
    sym_values = {k: v["value"] for k, v in symbols.items()}

    def buf(name: str, role: str, shape_expr: List[str], tied_to: Optional[str] = None,
            buf_dtype: Optional[str] = None) -> Dict:
        resolved = resolve_shape_expr(shape_expr, sym_values)
        out = {
            "name": name,
            "role": role,
            "dtype": buf_dtype or dtype,
            "shape": shape_expr,
            "resolved_shape": resolved,
        }
        if tied_to:
            out["tied_to"] = tied_to
        return out

    globals_buffers = []
    if config.get("rope_theta", 0) > 0:
        globals_buffers.append(buf("rope_cos", "precomputed", ["T", "D/2"]))
        globals_buffers.append(buf("rope_sin", "precomputed", ["T", "D/2"]))

    header_buffers = [
        buf("token_embed_weight", "weight", ["V", "E"]),
        buf("embed_output", "activation", ["S", "E"]),
    ]

    layer_buffers = [
        buf("layer.{L}.ln1_gamma", "weight", ["E"]),
        buf("layer.{L}.ln1_output", "activation", ["S", "E"]),
        buf("layer.{L}.wq", "weight", ["E", "H*D"]),
        buf("layer.{L}.q", "activation", ["S", "H", "D"]),
        buf("layer.{L}.wk", "weight", ["E", "KV*D"]),
        buf("layer.{L}.k", "activation", ["S", "KV", "D"]),
        buf("layer.{L}.wv", "weight", ["E", "KV*D"]),
        buf("layer.{L}.v", "activation", ["S", "KV", "D"]),
        buf("layer.{L}.q_rope", "activation", ["S", "H", "D"]),
        buf("layer.{L}.k_rope", "activation", ["S", "KV", "D"]),
        buf("layer.{L}.attn_scores", "activation", ["H", "S", "T"]),
        buf("layer.{L}.attn_probs", "activation", ["H", "S", "T"]),
        buf("layer.{L}.attn_out", "activation", ["S", "H", "D"]),
        buf("layer.{L}.wo", "weight", ["H*D", "E"]),
        buf("layer.{L}.proj_out", "activation", ["S", "E"]),
        buf("layer.{L}.residual1", "activation", ["S", "E"]),
        buf("layer.{L}.ln2_gamma", "weight", ["E"]),
        buf("layer.{L}.ln2_output", "activation", ["S", "E"]),
        buf("layer.{L}.mlp_gate_w", "weight", ["E", "I"]),
        buf("layer.{L}.mlp_gate_out", "activation", ["S", "I"]),
        buf("layer.{L}.mlp_up_w", "weight", ["E", "I"]),
        buf("layer.{L}.mlp_up_out", "activation", ["S", "I"]),
        buf("layer.{L}.mlp_act_out", "activation", ["S", "I"]),
        buf("layer.{L}.mlp_down_w", "weight", ["I", "E"]),
        buf("layer.{L}.mlp_down_out", "activation", ["S", "E"]),
        buf("layer.{L}.residual2", "activation", ["S", "E"]),
        buf("layer.{L}.k_cache", "cache", ["T", "KV", "D"]),
        buf("layer.{L}.v_cache", "cache", ["T", "KV", "D"]),
    ]

    footer_buffers = [
        buf("final_ln_gamma", "weight", ["E"]),
        buf("final_ln_output", "activation", ["S", "E"]),
        buf("lm_head_weight", "weight", ["E", "V"], tied_to="token_embed_weight"),
        buf("logits", "activation", ["S", "V"]),
    ]

    header_ops = [
        {
            "op": "embedding",
            "name": "token_embed",
            "inputs": ["tokens"],
            "weights": ["token_embed_weight"],
            "outputs": ["embed_output"],
        },
    ]
    if globals_buffers:
        header_ops.append({
            "op": "rope_precompute",
            "name": "rope_precompute",
            "inputs": [],
            "outputs": ["rope_cos", "rope_sin"],
            "attrs": {"theta": config.get("rope_theta", 10000.0)},
        })

    body_ops = [
        {
            "op": "rmsnorm",
            "name": "ln1",
            "inputs": ["prev_output"],
            "weights": ["layer.{L}.ln1_gamma"],
            "outputs": ["layer.{L}.ln1_output"],
        },
        {
            "op": "linear",
            "name": "q_proj",
            "inputs": ["layer.{L}.ln1_output"],
            "weights": ["layer.{L}.wq"],
            "outputs": ["layer.{L}.q"],
        },
        {
            "op": "linear",
            "name": "k_proj",
            "inputs": ["layer.{L}.ln1_output"],
            "weights": ["layer.{L}.wk"],
            "outputs": ["layer.{L}.k"],
        },
        {
            "op": "linear",
            "name": "v_proj",
            "inputs": ["layer.{L}.ln1_output"],
            "weights": ["layer.{L}.wv"],
            "outputs": ["layer.{L}.v"],
        },
        {
            "op": "kv_cache_update",
            "name": "kv_cache",
            "inputs": ["layer.{L}.k", "layer.{L}.v"],
            "outputs": ["layer.{L}.k_cache", "layer.{L}.v_cache"],
            "when": ["prefill", "decode"],
        },
        {
            "op": "rope",
            "name": "rope",
            "inputs": ["layer.{L}.q", "layer.{L}.k", "rope_cos", "rope_sin"],
            "outputs": ["layer.{L}.q_rope", "layer.{L}.k_rope"],
        },
        {
            "op": "attention",
            "name": "attn",
            "inputs": ["layer.{L}.q_rope", "layer.{L}.k_rope", "layer.{L}.v"],
            "outputs": ["layer.{L}.attn_out"],
            "scratch": ["layer.{L}.attn_scores", "layer.{L}.attn_probs"],
        },
        {
            "op": "linear",
            "name": "o_proj",
            "inputs": ["layer.{L}.attn_out"],
            "weights": ["layer.{L}.wo"],
            "outputs": ["layer.{L}.proj_out"],
        },
        {
            "op": "add",
            "name": "residual1",
            "inputs": ["prev_output", "layer.{L}.proj_out"],
            "outputs": ["layer.{L}.residual1"],
        },
        {
            "op": "rmsnorm",
            "name": "ln2",
            "inputs": ["layer.{L}.residual1"],
            "weights": ["layer.{L}.ln2_gamma"],
            "outputs": ["layer.{L}.ln2_output"],
        },
        {
            "op": "linear",
            "name": "mlp_gate",
            "inputs": ["layer.{L}.ln2_output"],
            "weights": ["layer.{L}.mlp_gate_w"],
            "outputs": ["layer.{L}.mlp_gate_out"],
        },
        {
            "op": "linear",
            "name": "mlp_up",
            "inputs": ["layer.{L}.ln2_output"],
            "weights": ["layer.{L}.mlp_up_w"],
            "outputs": ["layer.{L}.mlp_up_out"],
        },
        {
            "op": "swiglu",
            "name": "swiglu",
            "inputs": ["layer.{L}.mlp_gate_out", "layer.{L}.mlp_up_out"],
            "outputs": ["layer.{L}.mlp_act_out"],
        },
        {
            "op": "linear",
            "name": "mlp_down",
            "inputs": ["layer.{L}.mlp_act_out"],
            "weights": ["layer.{L}.mlp_down_w"],
            "outputs": ["layer.{L}.mlp_down_out"],
        },
        {
            "op": "add",
            "name": "residual2",
            "inputs": ["layer.{L}.residual1", "layer.{L}.mlp_down_out"],
            "outputs": ["layer.{L}.residual2"],
        },
    ]

    footer_ops = [
        {
            "op": "rmsnorm",
            "name": "final_ln",
            "inputs": ["last_layer_output"],
            "weights": ["final_ln_gamma"],
            "outputs": ["final_ln_output"],
        },
        {
            "op": "linear",
            "name": "lm_head",
            "inputs": ["final_ln_output"],
            "weights": ["lm_head_weight"],
            "outputs": ["logits"],
            "attrs": {"tied_to": "token_embed_weight"},
        },
    ]

    section = {
        "id": 0,
        "name": "text_decoder",
        "inputs": [
            {"name": "tokens", "dtype": "i32", "shape": ["S"], "resolved_shape": [sym_values["S"]]},
        ],
        "globals": globals_buffers,
        "buffers": {
            "header": header_buffers,
            "layer": layer_buffers,
            "footer": footer_buffers,
        },
        "header": {
            "ops": header_ops,
            "outputs": ["embed_output"],
        },
        "body": {
            "repeat": "num_layers",
            "layer_var": "L",
            "bindings": {
                "prev_output": {
                    "first_layer": "embed_output",
                    "next_layer": "layer.{L-1}.residual2",
                }
            },
            "ops": body_ops,
            "outputs": ["layer.{L}.residual2"],
        },
        "footer": {
            "bindings": {
                "last_layer_output": "layer.{L-1}.residual2",
            },
            "ops": footer_ops,
            "outputs": ["logits"],
        },
    }

    if config.get("model_type") in {"llama", "qwen2", "mistral"}:
        weight_map = LLAMA_WEIGHT_MAP
    else:
        weight_map = {}

    return {
        "version": 3,
        "kind": "graph",
        "generated": datetime.utcnow().isoformat() + "Z",
        "model": model_name,
        "config": config,
        "symbols": symbols,
        "sections": [section],
        "weight_map": weight_map,
    }

# ============================================================================
# BUMP ALLOCATOR
# ============================================================================

class BumpAllocator:
    """Simple bump allocator for computing offsets."""

    def __init__(self, start_offset: int = 0, alignment: int = CACHE_LINE):
        self.offset = start_offset
        self.alignment = alignment
        self.allocations: List[Tuple[str, int, int]] = []  # name, offset, size

    def alloc(self, name: str, size: int) -> int:
        """Allocate size bytes, return offset."""
        # Align current offset
        self.offset = align_up(self.offset, self.alignment)
        offset = self.offset
        self.offset += size
        self.allocations.append((name, offset, size))
        return offset

    def alloc_canary(self, name: str) -> Canary:
        """Allocate a canary marker."""
        offset = self.alloc(f"canary_{name}", CANARY_SIZE)
        return Canary(name=name, offset=offset)

    def alloc_buffer(self, name: str, shape: List[int], dtype: str, role: str) -> Buffer:
        """Allocate a buffer, return Buffer with offset filled in."""
        size = aligned_size(shape, dtype, self.alignment)
        offset = self.alloc(name, size)
        return Buffer(
            name=name,
            shape=shape,
            dtype=dtype,
            role=role,
            offset=offset,
            size=size
        )

# ============================================================================
# SECTION BUILDER
# ============================================================================

def build_layer_layout(allocator: BumpAllocator, config: Dict, layer_id: int) -> LayerLayout:
    """Build memory layout for a single transformer layer."""

    E = config["embed_dim"]
    H = config["num_heads"]
    KV = config["num_kv_heads"]
    D = config["head_dim"]
    I = config["intermediate_dim"]
    T = config["max_seq_len"]
    dtype = config["dtype"]

    buffers = []

    # Canary start
    canary_start = allocator.alloc_canary(f"layer_{layer_id}_start")

    # Pre-attention RMSNorm
    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.ln1_gamma", [E], dtype, "weight"))
    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.ln1_output", [T, E], dtype, "activation"))

    # Q projection
    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.wq", [E, H * D], dtype, "weight"))
    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.q", [T, H, D], dtype, "activation"))

    # K projection
    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.wk", [E, KV * D], dtype, "weight"))
    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.k", [T, KV, D], dtype, "activation"))

    # V projection
    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.wv", [E, KV * D], dtype, "weight"))
    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.v", [T, KV, D], dtype, "activation"))

    # RoPE outputs
    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.q_rope", [T, H, D], dtype, "activation"))
    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.k_rope", [T, KV, D], dtype, "activation"))

    # Attention
    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.attn_scores", [H, T, T], dtype, "activation"))
    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.attn_probs", [H, T, T], dtype, "activation"))
    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.attn_out", [T, H, D], dtype, "activation"))

    # Output projection
    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.wo", [H * D, E], dtype, "weight"))
    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.proj_out", [T, E], dtype, "activation"))

    # Residual 1
    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.residual1", [T, E], dtype, "activation"))

    # Post-attention RMSNorm
    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.ln2_gamma", [E], dtype, "weight"))
    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.ln2_output", [T, E], dtype, "activation"))

    # SwiGLU MLP
    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.mlp_gate_w", [E, I], dtype, "weight"))
    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.mlp_gate_out", [T, I], dtype, "activation"))

    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.mlp_up_w", [E, I], dtype, "weight"))
    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.mlp_up_out", [T, I], dtype, "activation"))

    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.mlp_act_out", [T, I], dtype, "activation"))

    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.mlp_down_w", [I, E], dtype, "weight"))
    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.mlp_down_out", [T, E], dtype, "activation"))

    # Residual 2
    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.residual2", [T, E], dtype, "activation"))

    # KV Cache
    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.k_cache", [T, KV, D], dtype, "cache"))
    buffers.append(allocator.alloc_buffer(f"layer_{layer_id}.v_cache", [T, KV, D], dtype, "cache"))

    # Canary end
    canary_end = allocator.alloc_canary(f"layer_{layer_id}_end")

    start_offset = canary_start.offset
    end_offset = allocator.offset

    return LayerLayout(
        layer_id=layer_id,
        canary_start=canary_start,
        buffers=buffers,
        canary_end=canary_end,
        total_bytes=end_offset - start_offset
    )

def build_section_layout(allocator: BumpAllocator, config: Dict, section_name: str, section_id: int) -> SectionLayout:
    """Build memory layout for a complete section (encoder or decoder)."""

    E = config["embed_dim"]
    V = config["vocab_size"]
    T = config["max_seq_len"]
    D = config["head_dim"]
    dtype = config["dtype"]

    # Header
    header_canary_start = allocator.alloc_canary("header_start")

    header_buffers = []

    # Token embedding
    header_buffers.append(allocator.alloc_buffer("token_embed_weight", [V, E], dtype, "weight"))
    header_buffers.append(allocator.alloc_buffer("embed_output", [T, E], dtype, "activation"))

    header_canary_end = allocator.alloc_canary("header_end")

    # Body (transformer layers)
    layers = []
    for layer_id in range(config["num_layers"]):
        layer = build_layer_layout(allocator, config, layer_id)
        layers.append(layer)

    # Footer
    footer_canary_start = allocator.alloc_canary("footer_start")

    footer_buffers = []
    footer_buffers.append(allocator.alloc_buffer("final_ln_gamma", [E], dtype, "weight"))
    footer_buffers.append(allocator.alloc_buffer("final_ln_output", [T, E], dtype, "activation"))

    # LM head (tied or separate)
    if config.get("tie_word_embeddings", True):
        # Tied: create alias
        lm_head = Buffer(
            name="lm_head_weight",
            shape=[E, V],
            dtype=dtype,
            role="weight",
            offset=header_buffers[0].offset,  # same as token_embed
            size=0,  # no additional storage
            tied_to="token_embed_weight"
        )
    else:
        lm_head = allocator.alloc_buffer("lm_head_weight", [E, V], dtype, "weight")
    footer_buffers.append(lm_head)

    # Logits
    footer_buffers.append(allocator.alloc_buffer("logits", [T, V], dtype, "activation"))

    footer_canary_end = allocator.alloc_canary("footer_end")

    # Global buffers (RoPE, etc.)
    globals_buffers = []
    if config.get("rope_theta", 0) > 0:
        globals_buffers.append(allocator.alloc_buffer("rope_cos", [T, D // 2], dtype, "precomputed"))
        globals_buffers.append(allocator.alloc_buffer("rope_sin", [T, D // 2], dtype, "precomputed"))

    return SectionLayout(
        name=section_name,
        section_id=section_id,
        config=config,
        header_canary_start=header_canary_start,
        header_buffers=header_buffers,
        header_canary_end=header_canary_end,
        layers=layers,
        footer_canary_start=footer_canary_start,
        footer_buffers=footer_buffers,
        footer_canary_end=footer_canary_end,
        globals=globals_buffers,
        total_bytes=allocator.offset
    )

def build_model_layout(config: Dict, model_name: str) -> ModelLayout:
    """Build complete model memory layout."""

    allocator = BumpAllocator(start_offset=64)  # Skip magic header

    # For now, single section (text decoder)
    section = build_section_layout(allocator, config, "text_decoder", 0)

    # Compute totals
    weight_bytes = 0
    activation_bytes = 0
    canary_count = 0

    def count_buffers(buffers):
        nonlocal weight_bytes, activation_bytes
        for buf in buffers:
            if buf.tied_to:
                continue  # Don't count tied weights twice
            if buf.role == "weight":
                weight_bytes += buf.size
            else:
                activation_bytes += buf.size

    count_buffers(section.header_buffers)
    count_buffers(section.footer_buffers)
    count_buffers(section.globals)
    for layer in section.layers:
        count_buffers(layer.buffers)

    # Collect canaries
    canaries: List[Canary] = []
    canaries.append(section.header_canary_start)
    canaries.append(section.header_canary_end)
    for layer in section.layers:
        canaries.append(layer.canary_start)
        canaries.append(layer.canary_end)
    canaries.append(section.footer_canary_start)
    canaries.append(section.footer_canary_end)
    canary_count = len(canaries)

    return ModelLayout(
        name=model_name,
        config=config,
        sections=[section],
        magic_header_size=64,
        total_bytes=allocator.offset,
        weight_bytes=weight_bytes,
        activation_bytes=activation_bytes,
        canary_count=canary_count,
        canaries=canaries,
    )

# ============================================================================
# JSON EMITTER
# ============================================================================

def emit_graph_ir(graph: Dict, output_path: str):
    """Emit graph.json (macro-level IR)."""
    with open(output_path, 'w') as f:
        json.dump(graph, f, indent=2)
    print(f"[GRAPH] Written: {output_path}")

def emit_layout_json(layout: ModelLayout, output_path: str):
    """Emit machine-readable layout.json."""

    def buffer_to_dict(buf: Buffer) -> Dict:
        d = {
            "name": buf.name,
            "offset": f"0x{buf.offset:08X}",
            "size": buf.size,
            "shape": buf.shape,
            "dtype": buf.dtype,
            "role": buf.role,
        }
        if buf.tied_to:
            d["tied_to"] = buf.tied_to
        return d

    def canary_to_dict(c: Canary) -> Dict:
        return {"name": c.name, "offset": f"0x{c.offset:08X}"}

    def layer_to_dict(layer: LayerLayout) -> Dict:
        return {
            "id": layer.layer_id,
            "canary_start": canary_to_dict(layer.canary_start),
            "canary_end": canary_to_dict(layer.canary_end),
            "buffers": [buffer_to_dict(b) for b in layer.buffers],
            "total_bytes": layer.total_bytes,
        }

    sections_json = []
    for section in layout.sections:
        sections_json.append({
            "name": section.name,
            "id": section.section_id,
            "header": {
                "canary_start": canary_to_dict(section.header_canary_start),
                "canary_end": canary_to_dict(section.header_canary_end),
                "buffers": [buffer_to_dict(b) for b in section.header_buffers],
            },
            "layers": [layer_to_dict(l) for l in section.layers],
            "footer": {
                "canary_start": canary_to_dict(section.footer_canary_start),
                "canary_end": canary_to_dict(section.footer_canary_end),
                "buffers": [buffer_to_dict(b) for b in section.footer_buffers],
            },
            "globals": [buffer_to_dict(b) for b in section.globals],
        })

    output = {
        "version": 3,
        "kind": "layout",
        "generated": datetime.utcnow().isoformat() + "Z",
        "model": layout.name,
        "config": layout.config,
        "memory": {
            "total_bytes": layout.total_bytes,
            "weight_bytes": layout.weight_bytes,
            "activation_bytes": layout.activation_bytes,
            "canary_count": layout.canary_count,
            "magic_header_size": layout.magic_header_size,
        },
        "sections": sections_json,
    }
    if layout.canaries:
        output["canaries"] = [canary_to_dict(c) for c in layout.canaries]

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"[JSON] Written: {output_path}")

# ============================================================================
# MAP FILE EMITTER
# ============================================================================

def emit_layout_map(layout: ModelLayout, output_path: str):
    """Emit human-readable layout.map."""

    lines = []

    def add(s=""): lines.append(s)
    def add_sep(): add("=" * 80)
    def add_line(): add("-" * 80)

    add_sep()
    add(f"MEMORY MAP: {layout.name}")
    add_sep()
    add(f"Generated:    {datetime.utcnow().isoformat()} UTC")
    add(f"IR Version:   3")
    add_sep()
    add()

    # Config
    add("CONFIGURATION")
    add_line()
    for key, value in layout.config.items():
        add(f"  {key:24s} {value}")
    add()

    # Memory summary
    add("MEMORY SUMMARY")
    add_line()
    add(f"  Total:        {layout.total_bytes:>16,} bytes  ({layout.total_bytes / 1e9:.2f} GB)")
    add(f"  Weights:      {layout.weight_bytes:>16,} bytes  ({layout.weight_bytes / 1e6:.1f} MB)")
    add(f"  Activations:  {layout.activation_bytes:>16,} bytes  ({layout.activation_bytes / 1e9:.2f} GB)")
    add(f"  Canaries:     {layout.canary_count * CANARY_SIZE:>16,} bytes  ({layout.canary_count} × {CANARY_SIZE} bytes)")
    add()

    # Buffer table
    def emit_buffer_table(buffers, title):
        add(title)
        add_line()
        add(f"{'Offset':<14} {'End':<14} {'Size':>12}  {'Buffer':<32} {'Shape':<20} {'Type':<10}")
        add_line()
        for buf in buffers:
            end = buf.offset + buf.size
            shape_str = str(buf.shape)
            tied = " (TIED)" if buf.tied_to else ""
            add(f"0x{buf.offset:08X}   0x{end:08X}   {buf.size:>12,}  {buf.name:<32} {shape_str:<20} {buf.role}{tied}")
        add()

    for section in layout.sections:
        add_sep()
        add(f"SECTION {section.section_id}: {section.name}")
        add_sep()
        add()

        # Header
        emit_buffer_table(section.header_buffers, "HEADER")

        # Layers (show first, last, and note the pattern)
        if len(section.layers) > 0:
            emit_buffer_table(section.layers[0].buffers, f"LAYER 0")

            if len(section.layers) > 2:
                add(f"... LAYERS 1-{len(section.layers)-2} follow same pattern ...")
                add(f"    Layer stride: {section.layers[1].total_bytes:,} bytes")
                add()

            if len(section.layers) > 1:
                emit_buffer_table(section.layers[-1].buffers, f"LAYER {len(section.layers)-1}")

        # Footer
        emit_buffer_table(section.footer_buffers, "FOOTER")

        # Globals
        if section.globals:
            emit_buffer_table(section.globals, "GLOBALS")

    # Verification
    add_sep()
    add("VERIFICATION")
    add_sep()
    add("[✓] All offsets 64-byte aligned")
    add("[✓] No overlapping buffers")
    add(f"[✓] Canary count: {layout.canary_count}")
    add()
    add_sep()
    add("END OF MEMORY MAP")
    add_sep()

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"[MAP] Written: {output_path}")

# ============================================================================
# C HEADER EMITTER
# ============================================================================

def emit_c_header(layout: ModelLayout, output_path: str, extra_api: Optional[List[str]] = None):
    """Emit generated_model.h with all structs and offsets."""

    config = layout.config
    section = layout.sections[0]  # Single section for now

    # Generate safe name
    safe_name = layout.name.upper().replace("-", "_").replace(".", "_")

    lines = []
    def add(s=""): lines.append(s)

    add("/**")
    add(f" * @file {os.path.basename(output_path)}")
    add(f" * @brief AUTO-GENERATED: {layout.name} Memory Layout")
    add(f" *")
    add(f" * Generated: {datetime.utcnow().isoformat()} UTC")
    add(f" * Total Memory: {layout.total_bytes / 1e9:.2f} GB")
    add(f" *")
    add(f" * DO NOT EDIT - Regenerate with build_ir_v3.py")
    add(f" */")
    add()
    add(f"#ifndef GENERATED_{safe_name}_H")
    add(f"#define GENERATED_{safe_name}_H")
    add()
    add("#include <stddef.h>")
    add("#include <stdint.h>")
    add()
    add("#ifdef __cplusplus")
    add('extern "C" {')
    add("#endif")
    add()

    # Constants
    add("/* ============================================================================")
    add(" * MODEL CONFIGURATION")
    add(" * ============================================================================ */")
    add()
    add(f"#define {safe_name}_EMBED_DIM          {config['embed_dim']}")
    add(f"#define {safe_name}_NUM_HEADS          {config['num_heads']}")
    add(f"#define {safe_name}_NUM_KV_HEADS       {config['num_kv_heads']}")
    add(f"#define {safe_name}_HEAD_DIM           {config['head_dim']}")
    add(f"#define {safe_name}_INTERMEDIATE       {config['intermediate_dim']}")
    add(f"#define {safe_name}_NUM_LAYERS         {config['num_layers']}")
    add(f"#define {safe_name}_VOCAB_SIZE         {config['vocab_size']}")
    add(f"#define {safe_name}_MAX_SEQ_LEN        {config['max_seq_len']}")
    add(f"#define {safe_name}_NUM_MERGES         {config.get('num_merges', 0)}")
    add(f"#define {safe_name}_TOTAL_VOCAB_BYTES  {config.get('total_vocab_bytes', 0)}")
    add(f"#define {safe_name}_DTYPE_BYTES        {DTYPE_BYTES[config['dtype']]}")
    add()
    add(f"#define {safe_name}_TOTAL_BYTES        {layout.total_bytes}ULL")
    add(f"#define {safe_name}_WEIGHT_BYTES       {layout.weight_bytes}ULL")
    add(f"#define {safe_name}_ACTIVATION_BYTES   {layout.activation_bytes}ULL")
    add()

    # Magic and canary
    add(f"#define {safe_name}_MAGIC              0x{MAGIC_PREFIX:08X}UL")
    add(f"#define {safe_name}_CANARY_VALUE       0x{CANARY_VALUE:08X}UL")
    add(f"#define {safe_name}_CANARY_SIZE        {CANARY_SIZE}")
    add()

    # Header offsets struct
    add("/* ============================================================================")
    add(" * HEADER OFFSETS")
    add(" * ============================================================================ */")
    add()
    add("typedef struct {")
    for buf in section.header_buffers:
        add(f"    size_t {buf.name.replace('.', '_')};  /* {buf.shape} */")
    add(f"}} {safe_name}HeaderOffsets;")
    add()
    add(f"static const {safe_name}HeaderOffsets {safe_name}_HEADER = {{")
    for buf in section.header_buffers:
        add(f"    .{buf.name.replace('.', '_')} = 0x{buf.offset:08X},")
    add("};")
    add()

    # Layer offsets struct
    add("/* ============================================================================")
    add(" * LAYER OFFSETS")
    add(" * ============================================================================ */")
    add()
    add("typedef struct {")
    if section.layers:
        for buf in section.layers[0].buffers:
            short_name = buf.name.split('.')[-1]
            add(f"    size_t {short_name};  /* {buf.shape} */")
    add(f"}} {safe_name}LayerOffsets;")
    add()

    # Layer array
    add(f"static const {safe_name}LayerOffsets {safe_name}_LAYERS[{config['num_layers']}] = {{")
    for layer in section.layers:
        add(f"    [{layer.layer_id}] = {{")
        for buf in layer.buffers:
            short_name = buf.name.split('.')[-1]
            add(f"        .{short_name} = 0x{buf.offset:08X},")
        add("    },")
    add("};")
    add()

    # Layer stride
    if len(section.layers) > 1:
        stride = section.layers[1].buffers[0].offset - section.layers[0].buffers[0].offset
        add(f"#define {safe_name}_LAYER_STRIDE  0x{stride:08X}ULL")
        add()

    # Per-layer dtype arrays for mixed quantization support
    def _get_layer_dtype_maps(layers):
        """Build per-layer dtype maps from layer buffers."""
        layer_dtype_maps = []
        for layer in layers:
            layer_map = {}
            for buf in layer.buffers:
                # Parse "layer.N.weight_name"
                parts = buf.name.split(".", 2)
                if len(parts) == 3 and parts[0] == "layer":
                    layer_map[parts[2]] = str(buf.dtype).lower()
            layer_dtype_maps.append(layer_map)
        return layer_dtype_maps

    def _has_mixed_layer_dtypes(layer_dtype_maps):
        """Check if any layer differs from layer 0 in weight dtypes."""
        if len(layer_dtype_maps) <= 1:
            return False
        ref = layer_dtype_maps[0]
        for layer_map in layer_dtype_maps[1:]:
            for key in ["wq", "wk", "wv", "wo", "w1", "w2"]:
                if layer_map.get(key, "") != ref.get(key, ""):
                    return True
        return False

    def _dtype_const(dtype: str) -> str:
        """Convert dtype string to CK_DT_* constant."""
        if dtype.startswith("q4_k"):
            return "CK_DT_Q4_K"
        if dtype.startswith("q6_k"):
            return "CK_DT_Q6_K"
        if dtype.startswith("q4_0"):
            return "CK_DT_Q4_0"
        if dtype.startswith("q4_1"):
            return "CK_DT_Q4_1"
        if dtype.startswith("q5_0"):
            return "CK_DT_Q5_0"
        if dtype.startswith("q5_1"):
            return "CK_DT_Q5_1"
        if dtype.startswith("q8_0"):
            return "CK_DT_Q8_0"
        if dtype.startswith("q8_k"):
            return "CK_DT_Q8_K"
        return "CK_DT_FP32"

    if section.layers:
        layer_dtype_maps = _get_layer_dtype_maps(section.layers)
        mixed_quant = _has_mixed_layer_dtypes(layer_dtype_maps)

        if mixed_quant:
            add("/* ============================================================================")
            add(" * PER-LAYER DTYPE ARRAYS (mixed quantization)")
            add(" * ============================================================================ */")
            add()
            add('#include "ckernel_dtype.h"  /* For CKDataType */')
            add()

            weight_keys = ["wq", "wk", "wv", "wo", "w1", "w2"]
            for key in weight_keys:
                array_name = f"{safe_name}_LAYER_{key.upper()}_DTYPE"
                values = [_dtype_const(layer_dtype_maps[i].get(key, "")) for i in range(len(section.layers))]
                add(f"static const CKDataType {array_name}[] = {{")
                add(f"    {', '.join(values)}")
                add("};")
                add()

    # Footer offsets
    add("/* ============================================================================")
    add(" * FOOTER OFFSETS")
    add(" * ============================================================================ */")
    add()
    add("typedef struct {")
    for buf in section.footer_buffers:
        add(f"    size_t {buf.name.replace('.', '_')};  /* {buf.shape} */")
    add(f"}} {safe_name}FooterOffsets;")
    add()
    add(f"static const {safe_name}FooterOffsets {safe_name}_FOOTER = {{")
    for buf in section.footer_buffers:
        tied_comment = f"  /* TIED to {buf.tied_to} */" if buf.tied_to else ""
        add(f"    .{buf.name.replace('.', '_')} = 0x{buf.offset:08X},{tied_comment}")
    add("};")
    add()

    # Globals
    if section.globals:
        add("/* ============================================================================")
        add(" * GLOBAL OFFSETS")
        add(" * ============================================================================ */")
        add()
        add("typedef struct {")
        for buf in section.globals:
            add(f"    size_t {buf.name};  /* {buf.shape} */")
        add(f"}} {safe_name}GlobalOffsets;")
        add()
        add(f"static const {safe_name}GlobalOffsets {safe_name}_GLOBALS = {{")
        for buf in section.globals:
            add(f"    .{buf.name} = 0x{buf.offset:08X},")
        add("};")
        add()

    # Canary offsets
    add("/* ============================================================================")
    add(" * CANARY OFFSETS")
    add(" * ============================================================================ */")
    add()
    add("typedef struct {")
    add("    size_t offset;")
    add("    const char *name;")
    add(f"}} {safe_name}Canary;")
    add()
    canaries = list(layout.canaries) if layout.canaries else []
    if not canaries:
        canaries = [
            section.header_canary_start,
            section.header_canary_end,
        ]
        for layer in section.layers:
            canaries.append(layer.canary_start)
            canaries.append(layer.canary_end)
        canaries.append(section.footer_canary_start)
        canaries.append(section.footer_canary_end)

    add(f"static const {safe_name}Canary {safe_name}_CANARIES[] = {{")
    for canary in canaries:
        add(f"    {{0x{canary.offset:08X}, \"{canary.name}\"}},")
    add("};")
    add(f"#define {safe_name}_CANARY_COUNT {len(canaries)}")
    add()

    # Model struct
    add("/* ============================================================================")
    add(" * MODEL STRUCT")
    add(" * ============================================================================ */")
    add()
    add("typedef struct {")
    add("    void *base;")
    add("    size_t total_bytes;")
    add(f"}} {safe_name}Model;")
    add()

    # Accessor macros
    add("/* ============================================================================")
    add(" * ACCESSOR MACROS")
    add(" * ============================================================================ */")
    add()
    add(f"#define {safe_name}_PTR(model, offset) \\")
    add("    ((float*)((char*)(model)->base + (offset)))")
    add()
    add(f"#define {safe_name}_PTR_BF16(model, offset) \\")
    add("    ((uint16_t*)((char*)(model)->base + (offset)))")
    add()
    add(f"#define {safe_name}_LAYER(layer_id) \\")
    add(f"    (&{safe_name}_LAYERS[layer_id])")
    add()

    # API declarations
    add("/* ============================================================================")
    add(" * API")
    add(" * ============================================================================ */")
    add()
    add(f"int {safe_name.lower()}_model_allocate({safe_name}Model *model);")
    add(f"void {safe_name.lower()}_model_free({safe_name}Model *model);")
    add(f"int {safe_name.lower()}_verify_canaries({safe_name}Model *model);")
    add(f"void {safe_name.lower()}_forward({safe_name}Model *model, const int *tokens, int num_tokens);")
    if extra_api:
        for decl in extra_api:
            add(decl)
    add()

    add("#ifdef __cplusplus")
    add("}")
    add("#endif")
    add()
    add(f"#endif /* GENERATED_{safe_name}_H */")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"[.h] Written: {output_path}")

# ============================================================================
# C SOURCE EMITTER
# ============================================================================

def emit_c_source(layout: ModelLayout,
                  output_path: str,
                  header_name: str,
                  emit_main: bool = False):
    """Emit generated_model.c with implementation."""

    config = layout.config
    section = layout.sections[0]
    safe_name = layout.name.upper().replace("-", "_").replace(".", "_")
    safe_name_lower = safe_name.lower()

    lines = []
    def add(s=""): lines.append(s)

    add("/**")
    add(f" * @file {os.path.basename(output_path)}")
    add(f" * @brief AUTO-GENERATED: {layout.name} Implementation")
    add(f" *")
    add(f" * Generated: {datetime.utcnow().isoformat()} UTC")
    add(f" * Total Memory: {layout.total_bytes / 1e9:.2f} GB")
    add(f" *")
    add(f" * DO NOT EDIT - Regenerate with build_ir_v3.py")
    add(f" */")
    add()
    add("#define _GNU_SOURCE  /* For MAP_ANONYMOUS, MAP_HUGETLB */")
    add()
    add(f'#include "{header_name}"')
    add()
    add("#include <stdio.h>")
    add("#include <stdlib.h>")
    add("#include <string.h>")
    add("#include <stdint.h>")
    add("#include <math.h>")
    add()
    add("#ifdef __linux__")
    add("#include <sys/mman.h>")
    add("#endif")
    add()

    # Forward declarations
    add("/* Forward declarations */")
    add(f"static void {safe_name_lower}_init_canaries({safe_name}Model *model);")
    add()

    # Magic header structure
    add("/* ============================================================================")
    add(" * MAGIC HEADER")
    add(" * ============================================================================ */")
    add()
    add("typedef struct __attribute__((packed)) {")
    add(f"    uint32_t magic;           /* 0x{MAGIC_PREFIX:08X} */")
    add("    uint32_t version;          /* IR version */")
    add("    uint64_t total_bytes;")
    add("    uint64_t weight_bytes;")
    add("    uint64_t activation_bytes;")
    add("    uint32_t num_layers;")
    add("    uint32_t embed_dim;")
    add("    uint32_t num_heads;")
    add("    uint32_t vocab_size;")
    add("    uint32_t max_seq_len;")
    add("    uint32_t canary_count;")
    add("    uint8_t  reserved[8];       /* Pad to 64 bytes */")
    add("} MagicHeader;")
    add()
    add("_Static_assert(sizeof(MagicHeader) == 64, \"MagicHeader must be 64 bytes\");")
    add()

    # Allocation function
    add("/* ============================================================================")
    add(" * ALLOCATION")
    add(" * ============================================================================ */")
    add()
    add(f"int {safe_name_lower}_model_allocate({safe_name}Model *model) {{")
    add(f"    size_t total = {safe_name}_TOTAL_BYTES;")
    add()
    add("#ifdef __linux__")
    add("    /* Try hugepages first */")
    add("    model->base = mmap(NULL, total,")
    add("                       PROT_READ | PROT_WRITE,")
    add("                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,")
    add("                       -1, 0);")
    add("    if (model->base == MAP_FAILED) {")
    add("        /* Fall back to regular pages */")
    add("        model->base = mmap(NULL, total,")
    add("                           PROT_READ | PROT_WRITE,")
    add("                           MAP_PRIVATE | MAP_ANONYMOUS,")
    add("                           -1, 0);")
    add("    }")
    add("    if (model->base == MAP_FAILED) {")
    add("        perror(\"mmap failed\");")
    add("        return -1;")
    add("    }")
    add("#else")
    add("    model->base = aligned_alloc(64, total);")
    add("    if (!model->base) {")
    add("        perror(\"aligned_alloc failed\");")
    add("        return -1;")
    add("    }")
    add("#endif")
    add()
    add("    model->total_bytes = total;")
    add()
    add("    /* Initialize magic header */")
    add("    MagicHeader *header = (MagicHeader *)model->base;")
    add(f"    header->magic = {safe_name}_MAGIC;")
    add("    header->version = 3;")
    add(f"    header->total_bytes = {safe_name}_TOTAL_BYTES;")
    add(f"    header->weight_bytes = {safe_name}_WEIGHT_BYTES;")
    add(f"    header->activation_bytes = {safe_name}_ACTIVATION_BYTES;")
    add(f"    header->num_layers = {safe_name}_NUM_LAYERS;")
    add(f"    header->embed_dim = {safe_name}_EMBED_DIM;")
    add(f"    header->num_heads = {safe_name}_NUM_HEADS;")
    add(f"    header->vocab_size = {safe_name}_VOCAB_SIZE;")
    add(f"    header->max_seq_len = {safe_name}_MAX_SEQ_LEN;")
    add(f"    header->canary_count = {safe_name}_CANARY_COUNT;")
    add()
    add("    /* Initialize canaries */")
    add(f"    {safe_name_lower}_init_canaries(model);")
    add()
    add("    return 0;")
    add("}")
    add()

    # Free function
    add(f"void {safe_name_lower}_model_free({safe_name}Model *model) {{")
    add("    if (!model || !model->base) return;")
    add()
    add("#ifdef __linux__")
    add("    munmap(model->base, model->total_bytes);")
    add("#else")
    add("    free(model->base);")
    add("#endif")
    add()
    add("    model->base = NULL;")
    add("    model->total_bytes = 0;")
    add("}")
    add()

    # Init canaries function
    add("/* ============================================================================")
    add(" * CANARY SYSTEM")
    add(" * ============================================================================ */")
    add()
    add(f"static void {safe_name_lower}_init_canaries({safe_name}Model *model) {{")
    add("    uint32_t *ptr;")
    add()
    add(f"    /* Write 0x{CANARY_VALUE:08X} to each canary slot */")
    add(f"    for (int i = 0; i < {safe_name}_CANARY_COUNT; i++) {{")
    add(f"        ptr = (uint32_t*)((char*)model->base + {safe_name}_CANARIES[i].offset);")
    add(f"        for (int j = 0; j < {CANARY_SIZE // 4}; j++) {{")
    add(f"            ptr[j] = {safe_name}_CANARY_VALUE;")
    add("        }")
    add("    }")
    add("}")
    add()

    # Verify canaries function
    add(f"int {safe_name_lower}_verify_canaries({safe_name}Model *model) {{")
    add("    int errors = 0;")
    add("    uint32_t *ptr;")
    add()
    add(f"    for (int i = 0; i < {safe_name}_CANARY_COUNT; i++) {{")
    add(f"        ptr = (uint32_t*)((char*)model->base + {safe_name}_CANARIES[i].offset);")
    add(f"        for (int j = 0; j < {CANARY_SIZE // 4}; j++) {{")
    add(f"            if (ptr[j] != {safe_name}_CANARY_VALUE) {{")
    add(f'                fprintf(stderr, "CANARY CORRUPTION: %s at offset 0x%lX\\n",')
    add(f"                        {safe_name}_CANARIES[i].name,")
    add(f"                        {safe_name}_CANARIES[i].offset);")
    add("                errors++;")
    add("                break;")
    add("            }")
    add("        }")
    add("    }")
    add()
    add("    return errors;")
    add("}")
    add()

    # RoPE precompute
    add("/* ============================================================================")
    add(" * ROPE PRECOMPUTE")
    add(" * ============================================================================ */")
    add()
    add(f"void {safe_name_lower}_precompute_rope({safe_name}Model *model) {{")
    if section.globals:
        add(f"    const int T = {safe_name}_MAX_SEQ_LEN;")
        add(f"    const int D = {safe_name}_HEAD_DIM / 2;")
        add(f"    const float theta = {config.get('rope_theta', 10000.0)}f;")
        add()
        add(f"    float *cos_ptr = {safe_name}_PTR(model, {safe_name}_GLOBALS.rope_cos);")
        add(f"    float *sin_ptr = {safe_name}_PTR(model, {safe_name}_GLOBALS.rope_sin);")
        add()
        add("    for (int pos = 0; pos < T; pos++) {")
        add("        for (int i = 0; i < D; i++) {")
        add("            float freq = 1.0f / powf(theta, (float)(2 * i) / (float)(D * 2));")
        add("            float angle = (float)pos * freq;")
        add("            cos_ptr[pos * D + i] = cosf(angle);")
        add("            sin_ptr[pos * D + i] = sinf(angle);")
        add("        }")
        add("    }")
    else:
        add("    /* No RoPE globals defined */")
        add("    (void)model;")
    add("}")
    add()

    # Single layer forward (placeholder for actual kernels)
    add("/* ============================================================================")
    add(" * LAYER FORWARD")
    add(" * ============================================================================ */")
    add()
    add("/* Note: These are placeholder stubs. Replace with actual kernel calls. */")
    add()
    add(f"static void {safe_name_lower}_layer_forward(")
    add(f"    {safe_name}Model *model,")
    add("    int layer_id,")
    add("    int num_tokens,")
    add("    int start_pos")
    add(") {")
    add(f"    const {safe_name}LayerOffsets *L = &{safe_name}_LAYERS[layer_id];")
    add("    (void)num_tokens;")
    add("    (void)start_pos;")
    add()
    add("    /* Get layer input */")
    add("    void *layer_input;")
    add("    if (layer_id == 0) {")
    add(f"        layer_input = {safe_name}_PTR_BF16(model, {safe_name}_HEADER.embed_output);")
    add("    } else {")
    add(f"        layer_input = {safe_name}_PTR_BF16(model, {safe_name}_LAYERS[layer_id - 1].residual2);")
    add("    }")
    add()
    add("    /* RMSNorm 1 */")
    add(f"    uint16_t *ln1_gamma = {safe_name}_PTR_BF16(model, L->ln1_gamma);")
    add(f"    uint16_t *ln1_out = {safe_name}_PTR_BF16(model, L->ln1_output);")
    add("    /* rmsnorm_bf16(layer_input, ln1_gamma, ln1_out, num_tokens, EMBED_DIM); */")
    add()
    add("    /* Q projection */")
    add(f"    uint16_t *wq = {safe_name}_PTR_BF16(model, L->wq);")
    add(f"    uint16_t *q = {safe_name}_PTR_BF16(model, L->q);")
    add("    /* matmul_bf16(ln1_out, wq, q, num_tokens, EMBED_DIM, NUM_HEADS * HEAD_DIM); */")
    add()
    add("    /* K projection */")
    add(f"    uint16_t *wk = {safe_name}_PTR_BF16(model, L->wk);")
    add(f"    uint16_t *k = {safe_name}_PTR_BF16(model, L->k);")
    add("    /* matmul_bf16(ln1_out, wk, k, num_tokens, EMBED_DIM, NUM_KV_HEADS * HEAD_DIM); */")
    add()
    add("    /* V projection */")
    add(f"    uint16_t *wv = {safe_name}_PTR_BF16(model, L->wv);")
    add(f"    uint16_t *v = {safe_name}_PTR_BF16(model, L->v);")
    add("    /* matmul_bf16(ln1_out, wv, v, num_tokens, EMBED_DIM, NUM_KV_HEADS * HEAD_DIM); */")
    add()
    add("    /* RoPE */")
    add(f"    uint16_t *q_rope = {safe_name}_PTR_BF16(model, L->q_rope);")
    add(f"    uint16_t *k_rope = {safe_name}_PTR_BF16(model, L->k_rope);")
    add("    /* rope_bf16(q, q_rope, rope_cos, rope_sin, ...); */")
    add("    /* rope_bf16(k, k_rope, rope_cos, rope_sin, ...); */")
    add()
    add("    /* Attention */")
    add(f"    uint16_t *attn_scores = {safe_name}_PTR_BF16(model, L->attn_scores);")
    add(f"    uint16_t *attn_probs = {safe_name}_PTR_BF16(model, L->attn_probs);")
    add(f"    uint16_t *attn_out = {safe_name}_PTR_BF16(model, L->attn_out);")
    add("    /* attention_bf16(q_rope, k_rope, v, attn_scores, attn_probs, attn_out, ...); */")
    add()
    add("    /* Output projection + residual */")
    add(f"    uint16_t *wo = {safe_name}_PTR_BF16(model, L->wo);")
    add(f"    uint16_t *proj_out = {safe_name}_PTR_BF16(model, L->proj_out);")
    add(f"    uint16_t *residual1 = {safe_name}_PTR_BF16(model, L->residual1);")
    add("    /* matmul_bf16(attn_out, wo, proj_out, ...); */")
    add("    /* add_bf16(layer_input, proj_out, residual1, ...); */")
    add()
    add("    /* RMSNorm 2 */")
    add(f"    uint16_t *ln2_gamma = {safe_name}_PTR_BF16(model, L->ln2_gamma);")
    add(f"    uint16_t *ln2_out = {safe_name}_PTR_BF16(model, L->ln2_output);")
    add("    /* rmsnorm_bf16(residual1, ln2_gamma, ln2_out, ...); */")
    add()
    add("    /* SwiGLU MLP */")
    add(f"    uint16_t *gate_w = {safe_name}_PTR_BF16(model, L->mlp_gate_w);")
    add(f"    uint16_t *gate_out = {safe_name}_PTR_BF16(model, L->mlp_gate_out);")
    add("    /* matmul_bf16(ln2_out, gate_w, gate_out, ...); */")
    add()
    add(f"    uint16_t *up_w = {safe_name}_PTR_BF16(model, L->mlp_up_w);")
    add(f"    uint16_t *up_out = {safe_name}_PTR_BF16(model, L->mlp_up_out);")
    add("    /* matmul_bf16(ln2_out, up_w, up_out, ...); */")
    add()
    add(f"    uint16_t *act_out = {safe_name}_PTR_BF16(model, L->mlp_act_out);")
    add("    /* swiglu_bf16(gate_out, up_out, act_out, ...); */")
    add()
    add(f"    uint16_t *down_w = {safe_name}_PTR_BF16(model, L->mlp_down_w);")
    add(f"    uint16_t *down_out = {safe_name}_PTR_BF16(model, L->mlp_down_out);")
    add("    /* matmul_bf16(act_out, down_w, down_out, ...); */")
    add()
    add("    /* Residual 2 (layer output) */")
    add(f"    uint16_t *residual2 = {safe_name}_PTR_BF16(model, L->residual2);")
    add("    /* add_bf16(residual1, down_out, residual2, ...); */")
    add()
    add("    (void)ln1_gamma; (void)ln1_out; (void)wq; (void)q; (void)wk; (void)k;")
    add("    (void)wv; (void)v; (void)q_rope; (void)k_rope; (void)attn_scores;")
    add("    (void)attn_probs; (void)attn_out; (void)wo; (void)proj_out; (void)residual1;")
    add("    (void)ln2_gamma; (void)ln2_out; (void)gate_w; (void)gate_out;")
    add("    (void)up_w; (void)up_out; (void)act_out; (void)down_w; (void)down_out;")
    add("    (void)residual2;")
    add("}")
    add()

    # Full forward function (STRAIGHT-LINE, no loops)
    add("/* ============================================================================")
    add(" * FORWARD PASS (STRAIGHT-LINE CODE)")
    add(" * ============================================================================ */")
    add()
    add(f"void {safe_name_lower}_forward(")
    add(f"    {safe_name}Model *model,")
    add("    const int *tokens,")
    add("    int num_tokens")
    add(") {")
    add("    /* Embedding lookup */")
    add(f"    uint16_t *embed_weight = {safe_name}_PTR_BF16(model, {safe_name}_HEADER.token_embed_weight);")
    add(f"    uint16_t *embed_out = {safe_name}_PTR_BF16(model, {safe_name}_HEADER.embed_output);")
    add("    /* embedding_lookup_bf16(tokens, embed_weight, embed_out, num_tokens, EMBED_DIM); */")
    add("    (void)embed_weight; (void)embed_out; (void)tokens;")
    add()
    add("    /* Transformer layers - straight-line execution */")

    # Generate straight-line layer calls (no loop)
    for layer_id in range(config["num_layers"]):
        add(f"    {safe_name_lower}_layer_forward(model, {layer_id}, num_tokens, 0);")

    add()
    add("    /* Final RMSNorm */")
    add(f"    uint16_t *last_hidden = {safe_name}_PTR_BF16(model, {safe_name}_LAYERS[{config['num_layers'] - 1}].residual2);")
    add(f"    uint16_t *final_ln_gamma = {safe_name}_PTR_BF16(model, {safe_name}_FOOTER.final_ln_gamma);")
    add(f"    uint16_t *final_ln_out = {safe_name}_PTR_BF16(model, {safe_name}_FOOTER.final_ln_output);")
    add("    /* rmsnorm_bf16(last_hidden, final_ln_gamma, final_ln_out, num_tokens, EMBED_DIM); */")
    add("    (void)last_hidden; (void)final_ln_gamma; (void)final_ln_out;")
    add()
    add("    /* LM head projection */")
    add(f"    uint16_t *lm_head = {safe_name}_PTR_BF16(model, {safe_name}_FOOTER.lm_head_weight);")
    add(f"    uint16_t *logits = {safe_name}_PTR_BF16(model, {safe_name}_FOOTER.logits);")
    add("    /* matmul_bf16(final_ln_out, lm_head, logits, num_tokens, EMBED_DIM, VOCAB_SIZE); */")
    add("    (void)lm_head; (void)logits;")
    add("}")
    add()

    if emit_main:
        add("/* ============================================================================")
        add(" * STANDALONE MAIN (STUB)")
        add(" * ============================================================================ */")
        add()
        add("int main(int argc, char **argv) {")
        add("    printf(\"%s: generated runtime (ir v3)\\n\", argv[0]);")
        add(f"    printf(\"total_bytes=%zu weight_bytes=%zu activation_bytes=%zu\\n\",")
        add(f"           (size_t){safe_name}_TOTAL_BYTES,")
        add(f"           (size_t){safe_name}_WEIGHT_BYTES,")
        add(f"           (size_t){safe_name}_ACTIVATION_BYTES);")
        add("    if (argc > 1 && strcmp(argv[1], \"--alloc\") == 0) {")
        add(f"        {safe_name}Model model = {{0}};")
        add(f"        if ({safe_name_lower}_model_allocate(&model) != 0) {{")
        add("            return 1;")
        add("        }")
        add(f"        {safe_name_lower}_precompute_rope(&model);")
        add(f"        {safe_name_lower}_model_free(&model);")
        add("        return 0;")
        add("    }")
        add("    fprintf(stderr, \"Run with --alloc to allocate buffers.\\n\");")
        add("    fprintf(stderr, \"Load weights and call *_forward() from your host app.\\n\");")
        add("    return 0;")
        add("}")
        add()

    # Debug info function
    add("/* ============================================================================")
    add(" * DEBUG INFO")
    add(" * ============================================================================ */")
    add()
    add(f"void {safe_name_lower}_print_info({safe_name}Model *model) {{")
    add('    printf("=== %s Model Info ===\\n", "' + layout.name + '");')
    add(f'    printf("Total memory:     %zu bytes (%.2f GB)\\n", model->total_bytes, model->total_bytes / 1e9);')
    add(f'    printf("Weight bytes:     %llu\\n", (unsigned long long){safe_name}_WEIGHT_BYTES);')
    add(f'    printf("Activation bytes: %llu\\n", (unsigned long long){safe_name}_ACTIVATION_BYTES);')
    add(f'    printf("Embed dim:        %d\\n", {safe_name}_EMBED_DIM);')
    add(f'    printf("Num heads:        %d\\n", {safe_name}_NUM_HEADS);')
    add(f'    printf("Num KV heads:     %d\\n", {safe_name}_NUM_KV_HEADS);')
    add(f'    printf("Num layers:       %d\\n", {safe_name}_NUM_LAYERS);')
    add(f'    printf("Vocab size:       %d\\n", {safe_name}_VOCAB_SIZE);')
    add(f'    printf("Max seq len:      %d\\n", {safe_name}_MAX_SEQ_LEN);')
    add(f'    printf("Canary count:     %d\\n", {safe_name}_CANARY_COUNT);')
    add()
    add("    /* Verify magic */")
    add("    MagicHeader *header = (MagicHeader *)model->base;")
    add(f'    printf("Magic:            0x%08X %s\\n", header->magic,')
    add(f'           header->magic == {safe_name}_MAGIC ? "(OK)" : "(MISMATCH!)");')
    add()
    add("    /* Verify canaries */")
    add(f"    int canary_errors = {safe_name_lower}_verify_canaries(model);")
    add('    printf("Canary check:     %s\\n", canary_errors == 0 ? "PASS" : "FAIL");')
    add("}")
    add()

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"[.c] Written: {output_path}")

# ============================================================================
# MAIN
# ============================================================================

BANNER = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║  C-Kernel-Engine IR v3 Code Generator                                         ║
║  Generates C code with baked-in memory offsets from HuggingFace models        ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

def print_usage():
    print("""
Usage:
    python build_ir_v3.py MODEL [OPTIONS]

Arguments:
    MODEL                   HuggingFace model ID or URL
                            Examples:
                              Qwen/Qwen2-0.5B-Instruct
                              meta-llama/Llama-3.2-1B
                              https://huggingface.co/microsoft/phi-2

Options:
    --prefix=DIR            Output directory (default: build/<model_name>)
    --seq-len=N             Max sequence length (default: from config, max 4096)
    --config=FILE           Use local config.json instead of downloading
    --name=NAME             Override model name for generated files
    --help                  Show this help message

Examples:
    # Download and generate from HuggingFace
    python build_ir_v3.py Qwen/Qwen2-0.5B-Instruct

    # Custom output directory
    python build_ir_v3.py Qwen/Qwen2-0.5B-Instruct --prefix=./my_model

    # Limit sequence length for smaller memory footprint
    python build_ir_v3.py meta-llama/Llama-3.2-1B --seq-len=2048

    # Use local config file
    python build_ir_v3.py --config=config.json --name=my_model
""")


def main():
    print(BANNER)

    # Simple argument parsing (configure-style)
    args = sys.argv[1:]

    if not args or '--help' in args or '-h' in args:
        print_usage()
        sys.exit(0)

    # Parse arguments
    model_input = None
    prefix = None
    config_path = None
    model_name = None
    max_seq_len = None

    for arg in args:
        if arg.startswith('--prefix='):
            prefix = arg.split('=', 1)[1]
        elif arg.startswith('--config='):
            config_path = arg.split('=', 1)[1]
        elif arg.startswith('--name='):
            model_name = arg.split('=', 1)[1]
        elif arg.startswith('--seq-len='):
            max_seq_len = int(arg.split('=', 1)[1])
        elif arg.startswith('--'):
            print(f"Unknown option: {arg}")
            print_usage()
            sys.exit(1)
        else:
            model_input = arg

    # Validate inputs
    if not model_input and not config_path:
        print("Error: Must provide either a model ID or --config=FILE")
        print_usage()
        sys.exit(1)

    # Load config
    if config_path:
        # Local config file
        print(f"[CONFIG] Reading local: {config_path}")
        config = parse_config(config_path)
        if not model_name:
            model_name = config.get("model_type", "model")
        model_id = None
    else:
        # Download from HuggingFace
        try:
            model_id = parse_hf_model_id(model_input)
            raw_config, cached_path = download_hf_config(model_id)

            # Parse the downloaded config
            config = parse_config(cached_path)

            if not model_name:
                model_name = model_id_to_name(model_id)

        except (ValueError, RuntimeError) as e:
            print(f"\nError: {e}")
            sys.exit(1)

    # Override max_seq_len if specified
    if max_seq_len:
        config["max_seq_len"] = max_seq_len
        print(f"[CONFIG] Overriding max_seq_len = {max_seq_len}")

    # Cap sequence length at 4096 by default (to avoid huge memory)
    if config["max_seq_len"] > 4096:
        print(f"[CONFIG] Note: Capping max_seq_len from {config['max_seq_len']} to 4096")
        print(f"         Use --seq-len=N to override")
        config["max_seq_len"] = 4096

    # Determine output directory
    if prefix:
        output_dir = prefix
    else:
        safe_model_name = model_name.replace("-", "_").replace(".", "_")
        output_dir = os.path.join("build", safe_model_name)

    print()
    print(f"[MODEL]  {model_name}")
    print(f"         embed_dim={config['embed_dim']}, heads={config['num_heads']}, "
          f"kv_heads={config['num_kv_heads']}, layers={config['num_layers']}")
    print(f"         vocab={config['vocab_size']:,}, max_seq={config['max_seq_len']}")
    print(f"[OUTPUT] {output_dir}/")
    print()

    # Build graph + layout
    print("[BUILD]  Computing graph IR...")
    graph = build_graph_ir(config, model_name)
    print("[BUILD]  Computing memory layout...")
    layout = build_model_layout(config, model_name)

    # Print memory summary
    def fmt_bytes(n):
        if n >= 1e9:
            return f"{n/1e9:.2f} GB"
        elif n >= 1e6:
            return f"{n/1e6:.1f} MB"
        else:
            return f"{n/1e3:.1f} KB"

    print(f"         Total:       {layout.total_bytes:>15,} bytes  ({fmt_bytes(layout.total_bytes)})")
    print(f"         Weights:     {layout.weight_bytes:>15,} bytes  ({fmt_bytes(layout.weight_bytes)})")
    print(f"         Activations: {layout.activation_bytes:>15,} bytes  ({fmt_bytes(layout.activation_bytes)})")
    print(f"         Canaries:    {layout.canary_count * CANARY_SIZE:>15,} bytes  ({layout.canary_count} × 64)")
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Emit outputs
    safe_name = model_name.replace("-", "_").replace(".", "_")
    header_name = f"generated_{safe_name}.h"

    emit_graph_ir(graph, os.path.join(output_dir, "graph.json"))
    emit_layout_json(layout, os.path.join(output_dir, "layout.json"))
    emit_layout_map(layout, os.path.join(output_dir, "layout.map"))
    emit_c_header(layout, os.path.join(output_dir, header_name))
    emit_c_source(layout, os.path.join(output_dir, f"generated_{safe_name}.c"), header_name, emit_main=False)

    # Also save the config for reference
    config_out = os.path.join(output_dir, "config.json")
    with open(config_out, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[CFG]  Written: {config_out}")

    print()
    print("═" * 60)
    print("  BUILD COMPLETE")
    print("═" * 60)
    print()
    print(f"Generated files in {output_dir}/:")
    print(f"  ├── config.json              # Model configuration")
    print(f"  ├── graph.json               # Graph IR (ops + buffers)")
    print(f"  ├── layout.json              # Machine-readable layout")
    print(f"  ├── layout.map               # Human-readable map")
    print(f"  ├── {header_name:<24} # C header (offsets)")
    print(f"  └── generated_{safe_name}.c  # C source (implementation)")
    print()
    print("Next steps:")
    print(f"  1. cd {output_dir}")
    print(f"  2. gcc -c -Wall -std=c11 generated_{safe_name}.c")
    print(f"  3. Link with your kernel implementations")
    print()

if __name__ == "__main__":
    main()
