#!/usr/bin/env python3
"""
parallel_planner.py - OpenMP parallelization strategy planner for IR v4

Analyzes operation shapes and selects optimal parallelization strategies:
- M_parallel: Parallelize over output rows (batch/sequence)
- N_parallel: Parallelize over output columns (features)
- H_parallel: Parallelize over attention heads
- K_parallel: Parallelize over reduction dimension (rare, needs atomics)
- feature_parallel: Parallelize over intermediate features
- none: No parallelization (overhead > benefit)

Each strategy includes:
- dimension to parallelize
- schedule type (static, dynamic, guided)
- minimum size threshold
- pragma template for codegen
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Parallel Strategy Definitions
# ---------------------------------------------------------------------------

@dataclass
class ParallelStrategy:
    """Parallelization strategy for an operation."""
    strategy: str           # M_parallel, N_parallel, H_parallel, etc.
    dim_name: str          # Human-readable dimension name
    dim_value: int         # Actual dimension size
    schedule: str          # static, dynamic, guided
    chunk_size: Optional[int]  # Chunk size for schedule
    min_work: int          # Minimum work items to parallelize
    pragma: str            # OpenMP pragma template

    def to_dict(self) -> Dict:
        return {
            "strategy": self.strategy,
            "dim_name": self.dim_name,
            "dim_value": self.dim_value,
            "schedule": self.schedule,
            "chunk_size": self.chunk_size,
            "min_work": self.min_work,
            "pragma": self.pragma,
        }


# Minimum thresholds for parallelization to be worthwhile
MIN_PARALLEL_ROWS = 4       # Don't parallelize M if M < 4
MIN_PARALLEL_COLS = 64      # Don't parallelize N if N < 64
MIN_PARALLEL_HEADS = 2      # Don't parallelize H if H < 2
MIN_PARALLEL_FEATURES = 256 # Don't parallelize intermediate if < 256


# ---------------------------------------------------------------------------
# Strategy Selection Functions
# ---------------------------------------------------------------------------

def select_gemm_strategy(M: int, N: int, K: int, mode: str) -> ParallelStrategy:
    """
    Select parallelization strategy for GEMM operations.

    GEMM: C[M,N] = A[M,K] @ B[K,N]

    Options:
    - M_parallel: Good when M is large (prefill)
    - N_parallel: Good when M is small but N is large (decode)
    - none: When both are too small
    """
    if mode == "decode" or M < MIN_PARALLEL_ROWS:
        # Decode mode (M=1) or small M: parallelize over N (output features)
        if N >= MIN_PARALLEL_COLS:
            return ParallelStrategy(
                strategy="N_parallel",
                dim_name="output_features",
                dim_value=N,
                schedule="static",
                chunk_size=None,
                min_work=MIN_PARALLEL_COLS,
                pragma=f"#pragma omp parallel for schedule(static) if({N} >= {MIN_PARALLEL_COLS})"
            )
        else:
            return ParallelStrategy(
                strategy="none",
                dim_name="none",
                dim_value=0,
                schedule="none",
                chunk_size=None,
                min_work=0,
                pragma="// No parallelization (workload too small)"
            )
    else:
        # Prefill mode: parallelize over M (rows/tokens)
        return ParallelStrategy(
            strategy="M_parallel",
            dim_name="tokens",
            dim_value=M,
            schedule="static",
            chunk_size=None,
            min_work=MIN_PARALLEL_ROWS,
            pragma=f"#pragma omp parallel for schedule(static) if({M} >= {MIN_PARALLEL_ROWS})"
        )


def select_attention_strategy(num_heads: int, num_kv_heads: int,
                               seq_len: int, mode: str) -> ParallelStrategy:
    """
    Select parallelization strategy for attention operations.

    Options:
    - H_parallel: Parallelize over attention heads (most common)
    - MH_parallel: Parallelize over tokens × heads (prefill with large seq)
    """
    if num_heads >= MIN_PARALLEL_HEADS:
        return ParallelStrategy(
            strategy="H_parallel",
            dim_name="num_heads",
            dim_value=num_heads,
            schedule="static",
            chunk_size=None,
            min_work=MIN_PARALLEL_HEADS,
            pragma=f"#pragma omp parallel for schedule(static) if({num_heads} > 1)"
        )
    else:
        return ParallelStrategy(
            strategy="none",
            dim_name="none",
            dim_value=0,
            schedule="none",
            chunk_size=None,
            min_work=0,
            pragma="// Single head - no parallelization"
        )


def select_mlp_strategy(embed_dim: int, intermediate_dim: int,
                        seq_len: int, mode: str) -> ParallelStrategy:
    """
    Select parallelization strategy for MLP operations.

    MLP has two phases:
    1. Up projection: [S, E] -> [S, I] (parallelize over I)
    2. Down projection: [S, I] -> [S, E] (parallelize over E)

    For fused MLP, we typically parallelize over intermediate_dim (I)
    since it's larger and provides more work per thread.
    """
    if mode == "decode":
        # Decode: S=1, parallelize over intermediate features
        if intermediate_dim >= MIN_PARALLEL_FEATURES:
            return ParallelStrategy(
                strategy="feature_parallel",
                dim_name="intermediate_dim",
                dim_value=intermediate_dim,
                schedule="static",
                chunk_size=None,
                min_work=MIN_PARALLEL_FEATURES,
                pragma=f"#pragma omp parallel for schedule(static)"
            )
    else:
        # Prefill: parallelize over tokens if enough, else features
        if seq_len >= MIN_PARALLEL_ROWS:
            return ParallelStrategy(
                strategy="M_parallel",
                dim_name="tokens",
                dim_value=seq_len,
                schedule="static",
                chunk_size=None,
                min_work=MIN_PARALLEL_ROWS,
                pragma=f"#pragma omp parallel for schedule(static)"
            )
        elif intermediate_dim >= MIN_PARALLEL_FEATURES:
            return ParallelStrategy(
                strategy="feature_parallel",
                dim_name="intermediate_dim",
                dim_value=intermediate_dim,
                schedule="static",
                chunk_size=None,
                min_work=MIN_PARALLEL_FEATURES,
                pragma=f"#pragma omp parallel for schedule(static)"
            )

    return ParallelStrategy(
        strategy="none",
        dim_name="none",
        dim_value=0,
        schedule="none",
        chunk_size=None,
        min_work=0,
        pragma="// No parallelization"
    )


def select_norm_strategy(seq_len: int, embed_dim: int, mode: str) -> ParallelStrategy:
    """
    Select parallelization strategy for normalization (RMSNorm, LayerNorm).

    Norm operates on [S, E] - parallelize over S (tokens) if large enough.
    """
    if seq_len >= MIN_PARALLEL_ROWS:
        return ParallelStrategy(
            strategy="M_parallel",
            dim_name="tokens",
            dim_value=seq_len,
            schedule="static",
            chunk_size=None,
            min_work=MIN_PARALLEL_ROWS,
            pragma=f"#pragma omp parallel for schedule(static) if({seq_len} >= {MIN_PARALLEL_ROWS})"
        )
    else:
        return ParallelStrategy(
            strategy="none",
            dim_name="none",
            dim_value=0,
            schedule="none",
            chunk_size=None,
            min_work=0,
            pragma="// Single token - no parallelization"
        )


def select_elementwise_strategy(seq_len: int, embed_dim: int, mode: str) -> ParallelStrategy:
    """
    Select parallelization strategy for elementwise ops (add, swiglu, etc.)
    """
    total_elements = seq_len * embed_dim

    if seq_len >= MIN_PARALLEL_ROWS:
        return ParallelStrategy(
            strategy="M_parallel",
            dim_name="tokens",
            dim_value=seq_len,
            schedule="static",
            chunk_size=None,
            min_work=MIN_PARALLEL_ROWS,
            pragma=f"#pragma omp parallel for schedule(static)"
        )
    elif total_elements >= 1024:
        # Flatten and parallelize
        return ParallelStrategy(
            strategy="flat_parallel",
            dim_name="elements",
            dim_value=total_elements,
            schedule="static",
            chunk_size=None,
            min_work=1024,
            pragma=f"#pragma omp parallel for schedule(static)"
        )
    else:
        return ParallelStrategy(
            strategy="none",
            dim_name="none",
            dim_value=0,
            schedule="none",
            chunk_size=None,
            min_work=0,
            pragma="// No parallelization"
        )


# ---------------------------------------------------------------------------
# Op Type to Strategy Mapper
# ---------------------------------------------------------------------------

OP_STRATEGY_MAP = {
    # GEMM-based ops (forward)
    "linear": select_gemm_strategy,
    "qkv_project": select_gemm_strategy,
    "attn_proj": select_gemm_strategy,
    "mlp_up": select_gemm_strategy,
    "mlp_down": select_gemm_strategy,
    "lm_head": select_gemm_strategy,

    # Fused GEMM ops (forward)
    "gemm_swiglu": select_mlp_strategy,
    "fused_mlp": select_mlp_strategy,
    "gemm_gelu": select_gemm_strategy,
    "gemm_relu": select_gemm_strategy,

    # Attention (forward)
    "attention": select_attention_strategy,
    "attention_proj_fused": select_attention_strategy,

    # Normalization (forward)
    "rmsnorm": select_norm_strategy,
    "layernorm": select_norm_strategy,
    "residual_rmsnorm": select_norm_strategy,

    # Elementwise (forward)
    "residual_add": select_elementwise_strategy,
    "swiglu": select_elementwise_strategy,
    "gelu": select_elementwise_strategy,
    "relu": select_elementwise_strategy,
    "rope": select_elementwise_strategy,

    # GEMM-based ops (backward)
    "gemm_backward": select_gemm_strategy,
    "qkv_backward": select_gemm_strategy,

    # Fused backward ops
    "fused_mlp_backward": select_mlp_strategy,
    "gemm_swiglu_backward": select_mlp_strategy,

    # Attention (backward)
    "attention_backward": select_attention_strategy,
    "attention_qkv_backward": select_attention_strategy,

    # Normalization (backward)
    "rmsnorm_backward": select_norm_strategy,
    "layernorm_backward": select_norm_strategy,
    "rmsnorm_residual_backward": select_norm_strategy,

    # Elementwise (backward)
    "add_backward": select_elementwise_strategy,
    "swiglu_backward": select_elementwise_strategy,
    "gelu_backward": select_elementwise_strategy,
    "relu_backward": select_elementwise_strategy,
    "rope_backward": select_elementwise_strategy,
    "sigmoid_backward": select_elementwise_strategy,

    # Embedding (backward - scatter add)
    "embedding_backward": select_elementwise_strategy,
}


# ---------------------------------------------------------------------------
# Main Planner Function
# ---------------------------------------------------------------------------

def plan_op_parallelism(op: Dict, config: Dict, mode: str) -> Optional[ParallelStrategy]:
    """
    Plan parallelization strategy for a single op.

    Args:
        op: Operation dict from lowered IR
        config: Model config with dimensions
        mode: Execution mode (prefill/decode)

    Returns:
        ParallelStrategy or None if op type not recognized
    """
    op_type = op.get("op", "")

    # Get dimensions from config
    E = config.get("embed_dim", config.get("hidden_size", 768))
    H = config.get("num_heads", 12)
    KV = config.get("num_kv_heads", H)
    D = config.get("head_dim", E // H)
    I = config.get("intermediate_dim", E * 4)
    S = 1 if mode == "decode" else config.get("max_seq_len", 2048)

    # Select strategy based on op type
    if op_type in ("linear", "qkv_project", "attn_proj", "mlp_up", "mlp_down",
                   "lm_head", "gemm_gelu", "gemm_relu"):
        # Determine M, N, K based on op
        if op_type == "qkv_project":
            M, N, K = S, 3 * H * D, E
        elif op_type == "attn_proj":
            M, N, K = S, E, H * D
        elif op_type == "mlp_up":
            M, N, K = S, 2 * I, E  # Gate + Up combined
        elif op_type == "mlp_down":
            M, N, K = S, E, I
        elif op_type == "lm_head":
            M, N, K = S, config.get("vocab_size", 32000), E
        else:
            M, N, K = S, E, E  # Generic linear
        return select_gemm_strategy(M, N, K, mode)

    elif op_type in ("gemm_swiglu", "fused_mlp"):
        return select_mlp_strategy(E, I, S, mode)

    elif op_type in ("attention", "attention_proj_fused"):
        return select_attention_strategy(H, KV, S, mode)

    elif op_type in ("rmsnorm", "layernorm", "residual_rmsnorm"):
        return select_norm_strategy(S, E, mode)

    elif op_type in ("residual_add", "swiglu", "gelu", "relu", "rope"):
        return select_elementwise_strategy(S, E, mode)

    return None


def apply_parallel_planning(lowered: Dict, mode: str) -> Tuple[Dict, Dict]:
    """
    Apply parallelization planning to all ops in lowered IR.

    Args:
        lowered: Lowered IR dict
        mode: Execution mode

    Returns:
        (updated_lowered, parallel_report)
    """
    import copy
    result = copy.deepcopy(lowered)
    config = result["config"]

    stats = {
        "total_ops": 0,
        "parallelized_ops": 0,
        "strategies": {},
    }

    def process_ops(ops: List[Dict]) -> List[Dict]:
        for op in ops:
            stats["total_ops"] += 1
            strategy = plan_op_parallelism(op, config, mode)

            if strategy:
                op["parallel"] = strategy.to_dict()

                if strategy.strategy != "none":
                    stats["parallelized_ops"] += 1
                    strat_name = strategy.strategy
                    stats["strategies"][strat_name] = stats["strategies"].get(strat_name, 0) + 1

        return ops

    # Process header ops
    section = result["sections"][0]
    if "header" in section and "ops" in section["header"]:
        section["header"]["ops"] = process_ops(section["header"]["ops"])

    # Process layer ops
    for layer in section.get("layers", []):
        layer["ops"] = process_ops(layer["ops"])

    # Process footer ops
    if "footer" in section and "ops" in section["footer"]:
        section["footer"]["ops"] = process_ops(section["footer"]["ops"])

    return result, stats


def emit_parallel_report(stats: Dict, mode: str, path: str) -> None:
    """Emit parallelization report."""
    import json
    from datetime import datetime

    report = {
        "mode": mode,
        "generated": datetime.utcnow().isoformat() + "Z",
        **stats,
    }

    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[PARALLEL] Written: {path}")


# ---------------------------------------------------------------------------
# Buffer-Level Parallel Annotations
# ---------------------------------------------------------------------------

def get_buffer_parallel_hint(buffer: Dict, config: Dict, mode: str) -> Dict:
    """
    Determine how a buffer can be parallelized based on its role and shape.

    Returns parallel hint for the buffer including:
    - strategy: lookup_parallel, row_parallel, col_parallel, head_parallel, etc.
    - split_dim: which dimension to split across threads
    - min_chunk: minimum elements per thread
    """
    name = buffer.get("name", "")
    role = buffer.get("role", "activation")
    shape = buffer.get("resolved_shape", buffer.get("shape", []))
    dtype = buffer.get("dtype", "f32")

    if not shape:
        return {"strategy": "none", "reason": "no shape"}

    # Embedding table: lookup parallel (each thread handles different lookups)
    if role == "weight" and ("emb" in name or "embed" in name):
        return {
            "strategy": "lookup_parallel",
            "split_dim": 0,  # Split across vocab (rows)
            "dim_size": shape[0] if shape else 0,
            "pragma": "// Embedding: parallelize over batch tokens, each thread looks up different rows"
        }

    # QKV weights: can split across heads or output features
    if role == "weight" and any(x in name for x in ["wq", "wk", "wv"]):
        # Shape: [H, D, E] or [H*D, E]
        if len(shape) == 3:
            return {
                "strategy": "head_parallel",
                "split_dim": 0,  # Split across heads
                "dim_size": shape[0],
                "pragma": f"// QKV weight: parallelize over {shape[0]} heads"
            }
        else:
            return {
                "strategy": "row_parallel",
                "split_dim": 0,
                "dim_size": shape[0],
                "pragma": f"// Weight: parallelize over {shape[0]} output features"
            }

    # Output projection weight: column parallel for reduction
    if role == "weight" and "wo" in name:
        return {
            "strategy": "col_parallel",
            "split_dim": 1,  # Split across input features (reduction dim)
            "dim_size": shape[1] if len(shape) > 1 else shape[0],
            "pragma": "// Output projection: column parallel (partial sums need reduction)"
        }

    # MLP weights
    if role == "weight" and any(x in name for x in ["w1", "w2", "gate", "up", "down"]):
        return {
            "strategy": "row_parallel",
            "split_dim": 0,
            "dim_size": shape[0],
            "pragma": f"// MLP weight: parallelize over {shape[0]} output features"
        }

    # Activations: batch/token parallel in prefill, feature parallel in decode
    if role == "activation":
        if mode == "decode" and len(shape) >= 2:
            # Decode: S=1, parallelize over features
            return {
                "strategy": "feature_parallel",
                "split_dim": -1,  # Last dim (features)
                "dim_size": shape[-1],
                "pragma": f"// Activation (decode): parallelize over {shape[-1]} features"
            }
        elif len(shape) >= 2 and shape[0] > 1:
            # Prefill: parallelize over tokens
            return {
                "strategy": "batch_parallel",
                "split_dim": 0,
                "dim_size": shape[0],
                "pragma": f"// Activation (prefill): parallelize over {shape[0]} tokens"
            }

    # Attention scores: head parallel
    if "scores" in name or "attn" in name:
        if len(shape) >= 3:
            return {
                "strategy": "head_parallel",
                "split_dim": 0,
                "dim_size": shape[0],
                "pragma": f"// Attention: parallelize over {shape[0]} heads"
            }

    # KV cache: head parallel
    if "k_cache" in name or "v_cache" in name:
        if len(shape) >= 3:
            return {
                "strategy": "head_parallel",
                "split_dim": 0,
                "dim_size": shape[0],
                "pragma": f"// KV cache: parallelize over {shape[0]} heads"
            }

    # RoPE cache: precomputed, read-only, no parallelization needed
    if "rope" in name and "cache" in name:
        return {
            "strategy": "broadcast",
            "split_dim": -1,
            "dim_size": 0,
            "pragma": "// RoPE cache: broadcast (read-only, shared)"
        }

    # Norm weights: small, typically not worth parallelizing
    if role == "weight" and ("gamma" in name or "norm" in name):
        return {
            "strategy": "broadcast",
            "split_dim": -1,
            "dim_size": 0,
            "pragma": "// Norm weight: broadcast (small, shared)"
        }

    # Default: no specific parallel hint
    return {
        "strategy": "none",
        "split_dim": -1,
        "dim_size": 0,
        "pragma": "// No specific parallel strategy"
    }


def annotate_layout_buffers(layout_dict: Dict, config: Dict, mode: str) -> Dict:
    """
    Add parallel annotations to all buffers in a layout dict.
    """
    import copy
    result = copy.deepcopy(layout_dict)

    def annotate_buffers(buffers: List[Dict]) -> List[Dict]:
        for buf in buffers:
            hint = get_buffer_parallel_hint(buf, config, mode)
            buf["parallel"] = hint
        return buffers

    for section in result.get("sections", []):
        # Header buffers
        if "header" in section and "buffers" in section["header"]:
            annotate_buffers(section["header"]["buffers"])

        # Layer buffers
        for layer in section.get("layers", []):
            if "buffers" in layer:
                annotate_buffers(layer["buffers"])

        # Footer buffers
        if "footer" in section and "buffers" in section["footer"]:
            annotate_buffers(section["footer"]["buffers"])

        # Globals
        if "globals" in section:
            annotate_buffers(section["globals"])

    return result


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "ParallelStrategy",
    "plan_op_parallelism",
    "apply_parallel_planning",
    "emit_parallel_report",
    "get_buffer_parallel_hint",
    "annotate_layout_buffers",
    "MIN_PARALLEL_ROWS",
    "MIN_PARALLEL_COLS",
    "MIN_PARALLEL_HEADS",
    "MIN_PARALLEL_FEATURES",
]
