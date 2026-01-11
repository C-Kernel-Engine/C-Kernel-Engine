#!/usr/bin/env python3
"""
v6_ir_lowering.py - IR Lowering for v6 Pipeline

Handles memory layout computation with per-layer buffers for training/backprop support.

Key Design Decisions:
  - NO shared scratch buffers: Each layer has its own activations for backprop
  - Per-layer decode scratch buffers: Allocated in arena, not on stack
  - Training-compatible: All activations preserved for gradient computation
  - Zero-copy arena: Single mmap allocation with computed offsets

This module is the foundation for a proper inference AND training engine,
not just a llama.cpp clone.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

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
    "q4_k_m": (256, 144), # Same as q4_k (mixed quant uses per-weight dtypes)
    "q6_k": (256, 210),   # 6-bit K-quant, 256/block = 210 bytes
    "q8_0": (32, 34),     # 8-bit, 32/block, 1 FP16 scale = 34 bytes
    "q8_k": (256, 292),   # 8-bit K-quant, 256/block = 292 bytes
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Buffer:
    """A single tensor buffer with computed offset.

    Each buffer represents a contiguous memory region in the model arena.
    For training, activations are NOT shared between layers to preserve
    intermediate values for backpropagation.
    """
    name: str
    shape: List[int]
    dtype: str
    role: str  # weight, activation, cache, grad, scratch, optimizer_state
    offset: int = 0
    size: int = 0
    tied_to: Optional[str] = None
    layer_id: Optional[int] = None  # For per-layer buffers

    def __repr__(self):
        tied = f" -> {self.tied_to}" if self.tied_to else ""
        layer = f" [L{self.layer_id}]" if self.layer_id is not None else ""
        return f"Buffer({self.name}{layer}: {self.shape} {self.dtype} @ 0x{self.offset:08X}{tied})"


@dataclass
class Canary:
    """Canary marker for memory corruption detection."""
    name: str
    offset: int


@dataclass
class DecodeBuffers:
    """Per-layer decode scratch buffers.

    These buffers are used during decode (single-token generation) and are
    allocated in the arena, NOT on the stack. This enables:
    1. Zero-copy operation (no per-call allocation)
    2. Training support (activations preserved for backprop)
    3. Memory safety (arena bounds checking)
    """
    q_token: Buffer       # [H, 1, D] - single token Q after projection
    k_token: Buffer       # [KV, 1, D] - single token K after projection
    v_token: Buffer       # [KV, 1, D] - single token V after projection
    attn_out: Buffer      # [H, 1, D] - attention output
    fc1_out: Buffer       # [1, 2*I] - MLP gate+up output
    swiglu_out: Buffer    # [1, I] - SwiGLU activation output


@dataclass
class LayerLayout:
    """Memory layout for a single transformer layer.

    Each layer has its own complete set of buffers - no sharing between layers.
    This is essential for training where we need all intermediate activations
    for gradient computation during backpropagation.
    """
    layer_id: int
    canary_start: Canary
    buffers: List[Buffer]
    canary_end: Canary
    decode_buffers: Optional[DecodeBuffers] = None  # Per-layer decode scratch
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
    """Complete model memory layout.

    Represents the entire memory arena for a model, including:
    - Weights (quantized or full precision)
    - Activations (per-layer, not shared for training)
    - KV cache
    - Gradient buffers (for training)
    - Optimizer state (for training)
    - Decode scratch buffers (per-layer, in arena)
    """
    name: str
    config: Dict
    sections: List[SectionLayout]
    magic_header_size: int = 64
    total_bytes: int = 0
    weight_bytes: int = 0
    activation_bytes: int = 0
    gradient_bytes: int = 0
    optimizer_bytes: int = 0
    decode_scratch_bytes: int = 0
    canary_count: int = 0
    canaries: List[Canary] = field(default_factory=list)


# ============================================================================
# MEMORY CALCULATOR
# ============================================================================

def align_up(n: int, alignment: int) -> int:
    """Align n up to the nearest multiple of alignment."""
    return (n + alignment - 1) & ~(alignment - 1)


def is_quantized_dtype(dtype: str) -> bool:
    """Check if dtype is a quantized type."""
    return dtype.lower() in QUANT_BLOCK_INFO


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
    if dtype_lower not in DTYPE_BYTES:
        raise ValueError(f"Unknown dtype: {dtype}")
    return elements * DTYPE_BYTES[dtype_lower]


def aligned_size(shape: List[int], dtype: str, alignment: int = CACHE_LINE) -> int:
    """Compute aligned buffer size."""
    size = compute_size(shape, dtype)
    return align_up(size, alignment)


def compute_quantized_size(dtype: str, elements: int) -> int:
    """Compute size for quantized weights."""
    dtype_lower = dtype.lower()
    if dtype_lower in QUANT_BLOCK_INFO:
        block_size, block_bytes = QUANT_BLOCK_INFO[dtype_lower]
        n_blocks = (elements + block_size - 1) // block_size
        return n_blocks * block_bytes
    return elements * DTYPE_BYTES.get(dtype_lower, 4)


# ============================================================================
# BUMP ALLOCATOR
# ============================================================================

class BumpAllocator:
    """Simple bump allocator for computing memory offsets.

    Allocates memory sequentially from a starting offset, aligning each
    allocation to the specified alignment (default: cache line).

    This allocator is used to compute the memory layout at build time,
    not at runtime. The resulting offsets are baked into the generated C code.
    """

    def __init__(self, start_offset: int = 0, alignment: int = CACHE_LINE):
        self.offset = start_offset
        self.alignment = alignment
        self.allocations: List[Tuple[str, int, int]] = []  # name, offset, size
        self.stats = {
            "weight_bytes": 0,
            "activation_bytes": 0,
            "gradient_bytes": 0,
            "optimizer_bytes": 0,
            "scratch_bytes": 0,
            "cache_bytes": 0,
        }

    def alloc(self, name: str, size: int, role: str = "activation") -> int:
        """Allocate size bytes, return offset."""
        # Align current offset
        self.offset = align_up(self.offset, self.alignment)
        offset = self.offset
        self.offset += size
        self.allocations.append((name, offset, size))

        # Track stats by role
        if role == "weight":
            self.stats["weight_bytes"] += size
        elif role in ("activation", "input", "output"):
            self.stats["activation_bytes"] += size
        elif role in ("gradient", "weight_grad"):
            self.stats["gradient_bytes"] += size
        elif role == "optimizer_state":
            self.stats["optimizer_bytes"] += size
        elif role == "scratch":
            self.stats["scratch_bytes"] += size
        elif role == "cache":
            self.stats["cache_bytes"] += size

        return offset

    def alloc_canary(self, name: str) -> Canary:
        """Allocate a canary marker."""
        offset = self.alloc(f"canary_{name}", CANARY_SIZE, role="scratch")
        return Canary(name=name, offset=offset)

    def alloc_buffer(self, name: str, shape: List[int], dtype: str, role: str,
                     layer_id: Optional[int] = None) -> Buffer:
        """Allocate a buffer, return Buffer with offset filled in."""
        size = aligned_size(shape, dtype, self.alignment)
        offset = self.alloc(name, size, role)
        return Buffer(
            name=name,
            shape=shape,
            dtype=dtype,
            role=role,
            offset=offset,
            size=size,
            layer_id=layer_id,
        )

    def alloc_decode_buffers(self, layer_id: int, config: Dict) -> DecodeBuffers:
        """Allocate per-layer decode scratch buffers.

        These are allocated in the arena (not on stack) to support:
        1. Zero-copy operation
        2. Training (activations preserved for backprop)
        3. Memory safety
        """
        H = config["num_heads"]
        KV = config["num_kv_heads"]
        D = config.get("aligned_head", config["head_dim"])
        I = config.get("aligned_intermediate", config["intermediate_dim"])
        dtype = config["dtype"]

        prefix = f"layer.{layer_id}.decode"

        return DecodeBuffers(
            q_token=self.alloc_buffer(f"{prefix}.q", [H, 1, D], dtype, "scratch", layer_id),
            k_token=self.alloc_buffer(f"{prefix}.k", [KV, 1, D], dtype, "scratch", layer_id),
            v_token=self.alloc_buffer(f"{prefix}.v", [KV, 1, D], dtype, "scratch", layer_id),
            attn_out=self.alloc_buffer(f"{prefix}.attn", [H, 1, D], dtype, "scratch", layer_id),
            fc1_out=self.alloc_buffer(f"{prefix}.fc1", [1, 2 * I], dtype, "scratch", layer_id),
            swiglu_out=self.alloc_buffer(f"{prefix}.swiglu", [1, I], dtype, "scratch", layer_id),
        )

    def current_offset(self) -> int:
        """Return current allocation offset."""
        return self.offset

    def total_allocated(self) -> int:
        """Return total bytes allocated."""
        return self.offset

    def get_stats(self) -> Dict[str, int]:
        """Return allocation statistics by category."""
        return dict(self.stats)


# ============================================================================
# LAYOUT BUILDERS
# ============================================================================

def build_layer_layout(allocator: BumpAllocator, config: Dict, layer_id: int,
                       include_training: bool = False,
                       include_decode_scratch: bool = True) -> LayerLayout:
    """Build memory layout for a single transformer layer.

    Args:
        allocator: BumpAllocator for offset computation
        config: Model configuration
        layer_id: Layer index
        include_training: If True, include gradient and optimizer buffers
        include_decode_scratch: If True, include per-layer decode buffers

    Key Design: NO buffer sharing between layers. Each layer gets its own
    complete set of activation buffers. This is required for:
    1. Training: Need all activations for backpropagation
    2. Debugging: Can inspect any layer's state
    3. Correctness: No risk of data races or overwrites
    """
    E = config.get("aligned_embed", config["embed_dim"])
    H = config["num_heads"]
    KV = config["num_kv_heads"]
    D = config.get("aligned_head", config["head_dim"])
    I = config.get("aligned_intermediate", config["intermediate_dim"])
    T = config["max_seq_len"]
    dtype = config["dtype"]

    buffers = []

    # Canary start
    canary_start = allocator.alloc_canary(f"layer_{layer_id}_start")

    # -------------------------------------------------------------------------
    # Forward buffers (always needed)
    # -------------------------------------------------------------------------

    # Pre-attention RMSNorm
    buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.ln1_gamma", [E], dtype, "weight", layer_id))
    buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.ln1_out", [T, E], dtype, "activation", layer_id))

    # Attention weights
    buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.wq", [H, D, E], dtype, "weight", layer_id))
    buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.wk", [KV, D, E], dtype, "weight", layer_id))
    buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.wv", [KV, D, E], dtype, "weight", layer_id))
    buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.wo", [H, E, D], dtype, "weight", layer_id))

    # QKV projections (per-layer, not shared)
    buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.q", [H, T, D], dtype, "activation", layer_id))
    buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.k", [KV, T, D], dtype, "activation", layer_id))
    buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.v", [KV, T, D], dtype, "activation", layer_id))

    # Attention output
    buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.attn_scores", [H, T, T], dtype, "activation", layer_id))
    buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.attn_out", [H, T, D], dtype, "activation", layer_id))
    buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.proj_out", [T, E], dtype, "activation", layer_id))

    # Residual 1
    buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.residual1", [T, E], dtype, "activation", layer_id))

    # Post-attention RMSNorm
    buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.ln2_gamma", [E], dtype, "weight", layer_id))
    buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.ln2_out", [T, E], dtype, "activation", layer_id))

    # SwiGLU MLP weights (gate+up packed as w1, down as w2)
    buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.w1", [2 * I, E], dtype, "weight", layer_id))
    buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.w2", [E, I], dtype, "weight", layer_id))

    # MLP activations (per-layer, not shared)
    buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.fc1_out", [T, 2 * I], dtype, "activation", layer_id))
    buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.swiglu_out", [T, I], dtype, "activation", layer_id))
    buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.mlp_out", [T, E], dtype, "activation", layer_id))

    # Layer output (residual 2)
    buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.output", [T, E], dtype, "activation", layer_id))

    # KV Cache
    buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.k_cache", [T, KV, D], dtype, "cache", layer_id))
    buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.v_cache", [T, KV, D], dtype, "cache", layer_id))

    # -------------------------------------------------------------------------
    # Training buffers (gradients and optimizer state)
    # -------------------------------------------------------------------------

    if include_training:
        # Activation gradients
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.d_output", [T, E], dtype, "gradient", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.d_mlp_out", [T, E], dtype, "gradient", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.d_swiglu_out", [T, I], dtype, "gradient", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.d_fc1_out", [T, 2 * I], dtype, "gradient", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.d_ln2_out", [T, E], dtype, "gradient", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.d_residual1", [T, E], dtype, "gradient", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.d_proj_out", [T, E], dtype, "gradient", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.d_attn_out", [H, T, D], dtype, "gradient", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.d_q", [H, T, D], dtype, "gradient", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.d_k", [KV, T, D], dtype, "gradient", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.d_v", [KV, T, D], dtype, "gradient", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.d_ln1_out", [T, E], dtype, "gradient", layer_id))

        # Weight gradients
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.d_ln1_gamma", [E], dtype, "weight_grad", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.d_wq", [H, D, E], dtype, "weight_grad", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.d_wk", [KV, D, E], dtype, "weight_grad", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.d_wv", [KV, D, E], dtype, "weight_grad", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.d_wo", [H, E, D], dtype, "weight_grad", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.d_ln2_gamma", [E], dtype, "weight_grad", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.d_w1", [2 * I, E], dtype, "weight_grad", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.d_w2", [E, I], dtype, "weight_grad", layer_id))

        # Adam optimizer state (m, v for each weight) - stored in fp32 for stability
        opt_dtype = "f32"
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.m_ln1_gamma", [E], opt_dtype, "optimizer_state", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.v_ln1_gamma", [E], opt_dtype, "optimizer_state", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.m_wq", [H, D, E], opt_dtype, "optimizer_state", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.v_wq", [H, D, E], opt_dtype, "optimizer_state", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.m_wk", [KV, D, E], opt_dtype, "optimizer_state", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.v_wk", [KV, D, E], opt_dtype, "optimizer_state", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.m_wv", [KV, D, E], opt_dtype, "optimizer_state", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.v_wv", [KV, D, E], opt_dtype, "optimizer_state", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.m_wo", [H, E, D], opt_dtype, "optimizer_state", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.v_wo", [H, E, D], opt_dtype, "optimizer_state", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.m_ln2_gamma", [E], opt_dtype, "optimizer_state", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.v_ln2_gamma", [E], opt_dtype, "optimizer_state", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.m_w1", [2 * I, E], opt_dtype, "optimizer_state", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.v_w1", [2 * I, E], opt_dtype, "optimizer_state", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.m_w2", [E, I], opt_dtype, "optimizer_state", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.v_w2", [E, I], opt_dtype, "optimizer_state", layer_id))

        # RMSNorm rstd cache (needed for backward)
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.ln1_rstd", [T], dtype, "activation", layer_id))
        buffers.append(allocator.alloc_buffer(f"layer.{layer_id}.ln2_rstd", [T], dtype, "activation", layer_id))

    # -------------------------------------------------------------------------
    # Per-layer decode scratch buffers (in arena, not on stack)
    # -------------------------------------------------------------------------

    decode_bufs = None
    if include_decode_scratch:
        decode_bufs = allocator.alloc_decode_buffers(layer_id, config)

    # Canary end
    canary_end = allocator.alloc_canary(f"layer_{layer_id}_end")

    start_offset = canary_start.offset
    end_offset = allocator.current_offset()

    return LayerLayout(
        layer_id=layer_id,
        canary_start=canary_start,
        buffers=buffers,
        canary_end=canary_end,
        decode_buffers=decode_bufs,
        total_bytes=end_offset - start_offset,
    )


def build_model_layout(config: Dict, model_name: str,
                       include_training: bool = False,
                       include_decode_scratch: bool = True) -> ModelLayout:
    """Build complete model memory layout.

    Args:
        config: Model configuration
        model_name: Model identifier
        include_training: If True, include gradient and optimizer buffers
        include_decode_scratch: If True, include per-layer decode buffers

    Key Design Decisions:
    1. Single contiguous arena (one mmap allocation)
    2. Per-layer activation buffers (no sharing for training support)
    3. Decode scratch in arena (not stack) for zero-copy operation
    4. Canaries for memory corruption detection
    """
    allocator = BumpAllocator(start_offset=64)  # Skip magic header

    E = config.get("aligned_embed", config["embed_dim"])
    V = config["vocab_size"]
    T = config["max_seq_len"]
    D = config.get("aligned_head", config["head_dim"])
    dtype = config["dtype"]
    num_layers = config["num_layers"]

    canaries: List[Canary] = []

    # -------------------------------------------------------------------------
    # Header section (embeddings)
    # -------------------------------------------------------------------------

    header_canary_start = allocator.alloc_canary("header_start")
    canaries.append(header_canary_start)

    header_buffers = []
    header_buffers.append(allocator.alloc_buffer("token_emb", [V, E], dtype, "weight"))
    header_buffers.append(allocator.alloc_buffer("embedded_input", [T, E], dtype, "activation"))

    # Training: embedding gradients and optimizer state
    if include_training:
        header_buffers.append(allocator.alloc_buffer("d_embedded_input", [T, E], dtype, "gradient"))
        header_buffers.append(allocator.alloc_buffer("d_token_emb", [V, E], dtype, "weight_grad"))
        header_buffers.append(allocator.alloc_buffer("m_token_emb", [V, E], "f32", "optimizer_state"))
        header_buffers.append(allocator.alloc_buffer("v_token_emb", [V, E], "f32", "optimizer_state"))

    header_canary_end = allocator.alloc_canary("header_end")
    canaries.append(header_canary_end)

    # -------------------------------------------------------------------------
    # Transformer layers
    # -------------------------------------------------------------------------

    layers = []
    for layer_id in range(num_layers):
        layer = build_layer_layout(
            allocator, config, layer_id,
            include_training=include_training,
            include_decode_scratch=include_decode_scratch,
        )
        layers.append(layer)
        canaries.append(layer.canary_start)
        canaries.append(layer.canary_end)

    # -------------------------------------------------------------------------
    # Footer section (final LN + LM head)
    # -------------------------------------------------------------------------

    footer_canary_start = allocator.alloc_canary("footer_start")
    canaries.append(footer_canary_start)

    footer_buffers = []
    footer_buffers.append(allocator.alloc_buffer("final_ln_weight", [E], dtype, "weight"))
    footer_buffers.append(allocator.alloc_buffer("final_output", [T, E], dtype, "activation"))

    # LM head (possibly tied to token_emb)
    tie_embeddings = config.get("tie_word_embeddings", True)
    if tie_embeddings:
        # Tied: create alias pointing to token_emb
        lm_head = Buffer(
            name="lm_head_weight",
            shape=[E, V],
            dtype=dtype,
            role="weight",
            offset=header_buffers[0].offset,  # Same as token_emb
            size=0,  # No additional storage
            tied_to="token_emb",
        )
    else:
        lm_head = allocator.alloc_buffer("lm_head_weight", [E, V], dtype, "weight")
    footer_buffers.append(lm_head)

    # Logits output
    footer_buffers.append(allocator.alloc_buffer("logits", [T, V], dtype, "activation"))

    # Training buffers
    if include_training:
        footer_buffers.append(allocator.alloc_buffer("labels", [T], "i32", "input"))
        footer_buffers.append(allocator.alloc_buffer("loss", [1], "f32", "output"))
        footer_buffers.append(allocator.alloc_buffer("d_logits", [T, V], dtype, "gradient"))
        footer_buffers.append(allocator.alloc_buffer("d_final_output", [T, E], dtype, "gradient"))
        footer_buffers.append(allocator.alloc_buffer("d_final_ln_weight", [E], dtype, "weight_grad"))
        footer_buffers.append(allocator.alloc_buffer("final_ln_rstd", [T], dtype, "activation"))

        if not tie_embeddings:
            footer_buffers.append(allocator.alloc_buffer("d_lm_head_weight", [E, V], dtype, "weight_grad"))
            footer_buffers.append(allocator.alloc_buffer("m_lm_head_weight", [E, V], "f32", "optimizer_state"))
            footer_buffers.append(allocator.alloc_buffer("v_lm_head_weight", [E, V], "f32", "optimizer_state"))

        footer_buffers.append(allocator.alloc_buffer("m_final_ln_weight", [E], "f32", "optimizer_state"))
        footer_buffers.append(allocator.alloc_buffer("v_final_ln_weight", [E], "f32", "optimizer_state"))

    footer_canary_end = allocator.alloc_canary("footer_end")
    canaries.append(footer_canary_end)

    # -------------------------------------------------------------------------
    # Global buffers (RoPE, etc.)
    # -------------------------------------------------------------------------

    globals_buffers = []
    if config.get("rope_theta", 0) > 0:
        globals_buffers.append(allocator.alloc_buffer("rope_cos", [T, D // 2], dtype, "precomputed"))
        globals_buffers.append(allocator.alloc_buffer("rope_sin", [T, D // 2], dtype, "precomputed"))

    # -------------------------------------------------------------------------
    # Compute totals
    # -------------------------------------------------------------------------

    stats = allocator.get_stats()

    def count_buffers(buffers: List[Buffer]) -> Tuple[int, int, int, int]:
        weight_bytes = 0
        activation_bytes = 0
        gradient_bytes = 0
        optimizer_bytes = 0
        for buf in buffers:
            if buf.tied_to:
                continue  # Don't count tied weights twice
            if buf.role == "weight":
                weight_bytes += buf.size
            elif buf.role in ("activation", "cache", "input", "output", "precomputed"):
                activation_bytes += buf.size
            elif buf.role in ("gradient", "weight_grad"):
                gradient_bytes += buf.size
            elif buf.role == "optimizer_state":
                optimizer_bytes += buf.size
        return weight_bytes, activation_bytes, gradient_bytes, optimizer_bytes

    total_weight = 0
    total_activation = 0
    total_gradient = 0
    total_optimizer = 0
    total_decode_scratch = 0

    w, a, g, o = count_buffers(header_buffers)
    total_weight += w
    total_activation += a
    total_gradient += g
    total_optimizer += o

    w, a, g, o = count_buffers(footer_buffers)
    total_weight += w
    total_activation += a
    total_gradient += g
    total_optimizer += o

    w, a, g, o = count_buffers(globals_buffers)
    total_weight += w
    total_activation += a
    total_gradient += g
    total_optimizer += o

    for layer in layers:
        w, a, g, o = count_buffers(layer.buffers)
        total_weight += w
        total_activation += a
        total_gradient += g
        total_optimizer += o

        if layer.decode_buffers:
            db = layer.decode_buffers
            for buf in [db.q_token, db.k_token, db.v_token, db.attn_out, db.fc1_out, db.swiglu_out]:
                total_decode_scratch += buf.size

    section_layout = SectionLayout(
        name="text_decoder",
        section_id=0,
        config=config,
        header_canary_start=header_canary_start,
        header_buffers=header_buffers,
        header_canary_end=header_canary_end,
        layers=layers,
        footer_canary_start=footer_canary_start,
        footer_buffers=footer_buffers,
        footer_canary_end=footer_canary_end,
        globals=globals_buffers,
        total_bytes=allocator.current_offset(),
    )

    return ModelLayout(
        name=model_name,
        config=config,
        sections=[section_layout],
        magic_header_size=64,
        total_bytes=allocator.current_offset(),
        weight_bytes=total_weight,
        activation_bytes=total_activation,
        gradient_bytes=total_gradient,
        optimizer_bytes=total_optimizer,
        decode_scratch_bytes=total_decode_scratch,
        canary_count=len(canaries),
        canaries=canaries,
    )


# ============================================================================
# EMITTERS
# ============================================================================

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
        if buf.layer_id is not None:
            d["layer_id"] = buf.layer_id
        return d

    def canary_to_dict(c: Canary) -> Dict:
        return {"name": c.name, "offset": f"0x{c.offset:08X}"}

    def decode_buffers_to_dict(db: Optional[DecodeBuffers]) -> Optional[Dict]:
        if not db:
            return None
        return {
            "q_token": buffer_to_dict(db.q_token),
            "k_token": buffer_to_dict(db.k_token),
            "v_token": buffer_to_dict(db.v_token),
            "attn_out": buffer_to_dict(db.attn_out),
            "fc1_out": buffer_to_dict(db.fc1_out),
            "swiglu_out": buffer_to_dict(db.swiglu_out),
        }

    def layer_to_dict(layer: LayerLayout) -> Dict:
        d = {
            "id": layer.layer_id,
            "canary_start": canary_to_dict(layer.canary_start),
            "canary_end": canary_to_dict(layer.canary_end),
            "buffers": [buffer_to_dict(b) for b in layer.buffers],
            "total_bytes": layer.total_bytes,
        }
        if layer.decode_buffers:
            d["decode_buffers"] = decode_buffers_to_dict(layer.decode_buffers)
        return d

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
        "version": 6,
        "kind": "layout",
        "generated": datetime.utcnow().isoformat() + "Z",
        "model": layout.name,
        "config": layout.config,
        "memory": {
            "total_bytes": layout.total_bytes,
            "weight_bytes": layout.weight_bytes,
            "activation_bytes": layout.activation_bytes,
            "gradient_bytes": layout.gradient_bytes,
            "optimizer_bytes": layout.optimizer_bytes,
            "decode_scratch_bytes": layout.decode_scratch_bytes,
            "canary_count": layout.canary_count,
            "magic_header_size": layout.magic_header_size,
        },
        "sections": sections_json,
    }
    if layout.canaries:
        output["canaries"] = [canary_to_dict(c) for c in layout.canaries]

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"[LAYOUT] Written: {output_path}")


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
    add(f"IR Version:   6 (per-layer buffers, training-compatible)")
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
    add(f"  Total:            {layout.total_bytes:>16,} bytes  ({layout.total_bytes / 1e9:.2f} GB)")
    add(f"  Weights:          {layout.weight_bytes:>16,} bytes  ({layout.weight_bytes / 1e6:.1f} MB)")
    add(f"  Activations:      {layout.activation_bytes:>16,} bytes  ({layout.activation_bytes / 1e9:.2f} GB)")
    if layout.gradient_bytes > 0:
        add(f"  Gradients:        {layout.gradient_bytes:>16,} bytes  ({layout.gradient_bytes / 1e9:.2f} GB)")
    if layout.optimizer_bytes > 0:
        add(f"  Optimizer State:  {layout.optimizer_bytes:>16,} bytes  ({layout.optimizer_bytes / 1e9:.2f} GB)")
    if layout.decode_scratch_bytes > 0:
        add(f"  Decode Scratch:   {layout.decode_scratch_bytes:>16,} bytes  ({layout.decode_scratch_bytes / 1e6:.1f} MB)")
    add(f"  Canaries:         {layout.canary_count * CANARY_SIZE:>16,} bytes  ({layout.canary_count} × {CANARY_SIZE} bytes)")
    add()

    # Buffer table
    def emit_buffer_table(buffers, title):
        add(title)
        add_line()
        add(f"{'Offset':<14} {'End':<14} {'Size':>12}  {'Buffer':<40} {'Shape':<20} {'Type':<10}")
        add_line()
        for buf in buffers:
            end = buf.offset + buf.size
            shape_str = str(buf.shape)
            tied = " (TIED)" if buf.tied_to else ""
            add(f"0x{buf.offset:08X}   0x{end:08X}   {buf.size:>12,}  {buf.name:<40} {shape_str:<20} {buf.role}{tied}")
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

            # Show decode buffers for layer 0
            if section.layers[0].decode_buffers:
                add("  DECODE SCRATCH BUFFERS (per-layer, in arena):")
                db = section.layers[0].decode_buffers
                for name, buf in [("q_token", db.q_token), ("k_token", db.k_token),
                                  ("v_token", db.v_token), ("attn_out", db.attn_out),
                                  ("fc1_out", db.fc1_out), ("swiglu_out", db.swiglu_out)]:
                    add(f"    0x{buf.offset:08X}  {buf.size:>10,}  {buf.name:<36} {buf.shape}")
                add()

            if len(section.layers) > 2:
                add(f"... LAYERS 1-{len(section.layers)-2} follow same pattern ...")
                add(f"    Layer stride: {section.layers[0].total_bytes:,} bytes")
                add()

            if len(section.layers) > 1:
                emit_buffer_table(section.layers[-1].buffers, f"LAYER {len(section.layers)-1}")

        # Footer
        emit_buffer_table(section.footer_buffers, "FOOTER")

        # Globals
        if section.globals:
            emit_buffer_table(section.globals, "GLOBALS")

    # Design notes
    add_sep()
    add("DESIGN NOTES")
    add_sep()
    add("[✓] Per-layer activation buffers (no sharing between layers)")
    add("[✓] Per-layer decode scratch in arena (not stack)")
    add("[✓] Training-compatible: all activations preserved for backprop")
    add("[✓] All offsets 64-byte aligned")
    add("[✓] Canaries between sections for memory corruption detection")
    add()
    add_sep()
    add("END OF MEMORY MAP")
    add_sep()

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"[MAP] Written: {output_path}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_buffer_by_name(layout: ModelLayout, name: str) -> Optional[Buffer]:
    """Find a buffer by name in the layout."""
    section = layout.sections[0]

    for buf in section.header_buffers:
        if buf.name == name:
            return buf

    for buf in section.footer_buffers:
        if buf.name == name:
            return buf

    for buf in section.globals:
        if buf.name == name:
            return buf

    for layer in section.layers:
        for buf in layer.buffers:
            if buf.name == name:
                return buf
        if layer.decode_buffers:
            db = layer.decode_buffers
            for buf in [db.q_token, db.k_token, db.v_token, db.attn_out, db.fc1_out, db.swiglu_out]:
                if buf.name == name:
                    return buf

    return None


def get_layer_decode_buffers(layout: ModelLayout, layer_id: int) -> Optional[DecodeBuffers]:
    """Get decode scratch buffers for a specific layer."""
    section = layout.sections[0]
    if 0 <= layer_id < len(section.layers):
        return section.layers[layer_id].decode_buffers
    return None


def compute_memory_requirements(config: Dict, include_training: bool = False) -> Dict[str, int]:
    """Compute memory requirements without allocating.

    Useful for planning before committing to a layout.
    """
    layout = build_model_layout(config, "estimate",
                                include_training=include_training,
                                include_decode_scratch=True)
    return {
        "total_bytes": layout.total_bytes,
        "weight_bytes": layout.weight_bytes,
        "activation_bytes": layout.activation_bytes,
        "gradient_bytes": layout.gradient_bytes,
        "optimizer_bytes": layout.optimizer_bytes,
        "decode_scratch_bytes": layout.decode_scratch_bytes,
    }


# ============================================================================
# MODULE API
# ============================================================================

__all__ = [
    # Constants
    "CACHE_LINE", "CANARY_SIZE", "CANARY_VALUE", "MAGIC_PREFIX",
    "DTYPE_BYTES", "QUANT_BLOCK_INFO",

    # Data structures
    "Buffer", "Canary", "DecodeBuffers", "LayerLayout", "SectionLayout", "ModelLayout",

    # Functions
    "align_up", "is_quantized_dtype", "compute_size", "aligned_size", "compute_quantized_size",

    # Allocator
    "BumpAllocator",

    # Builders
    "build_layer_layout", "build_model_layout",

    # Emitters
    "emit_layout_json", "emit_layout_map",

    # Utilities
    "get_buffer_by_name", "get_layer_decode_buffers", "compute_memory_requirements",
]
