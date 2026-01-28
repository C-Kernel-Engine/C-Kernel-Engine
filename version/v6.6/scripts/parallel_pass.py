#!/usr/bin/env python3
"""
parallel_pass.py - Centralized OpenMP parallelization pass for IR v6.6

=============================================================================
ARCHITECTURE: WHY THIS EXISTS
=============================================================================

When looking at codegen output, you'll see many `for` loops that LOOK like
they could be parallelized with `#pragma omp parallel for`. It's tempting
to add pragmas directly in codegen. DON'T DO THIS.

Codegen is DUMB by design. It reads IR and emits exactly what IR says.
All intelligence about parallelization belongs HERE in the parallel pass.

Why centralized parallelization matters:

1. FALSE SHARING PREVENTION
   --------------------------
   If two ops write to adjacent memory and both are parallelized:

   Op A (parallel): writes output[0:512]    Thread 0: [0:256], Thread 1: [256:512]
   Op B (parallel): writes output[512:1024] Thread 0: [512:768], Thread 1: [768:1024]

   If output[256:512] and output[512:768] share a cache line (64 bytes),
   Thread 1 of Op A and Thread 0 of Op B thrash the same cache line.

   Solution: Only ONE of these ops should be parallelized, OR we need
   cache-line-aligned chunking.

2. THREAD OVER-SUBSCRIPTION
   --------------------------
   If Op A spawns 8 threads and Op B spawns 8 threads, and they're called
   back-to-back, we might have 16 threads competing for 8 cores.

   Solution: Parallel pass tracks total parallelism and throttles.

3. MEMORY BANDWIDTH SATURATION
   --------------------------
   Some ops are memory-bound (e.g., residual_add). Parallelizing them
   doesn't help - we're limited by DRAM bandwidth, not compute.
   Adding threads just adds overhead.

   Solution: Parallel pass knows which ops are compute vs memory bound.

4. DEPENDENCY ANALYSIS
   --------------------------
   If Op A's output is Op B's input, they can't run in parallel.
   But WITHIN each op, we can parallelize the loop.

   Solution: Parallel pass sees the full graph and knows dependencies.

=============================================================================
HOW IT WORKS
=============================================================================

Input:  lowered_*_call.json (from IR Lower 3)
Output: lowered_*_call.json with "parallel" field on each op

The "parallel" field contains:
{
    "enabled": true/false,
    "strategy": "token" | "head" | "feature" | "none",
    "pragma": "#pragma omp parallel for ...",  // EXACT string to emit
    "reason": "Human-readable explanation"
}

Codegen then BLINDLY emits op["parallel"]["pragma"] before the relevant loop.
No computation, no decisions, just emit what IR says.

=============================================================================
USAGE
=============================================================================

As part of build pipeline:
    python build_ir_v6_6.py ... --parallel

Standalone:
    python parallel_pass.py --input lowered_prefill_call.json --output lowered_prefill_parallel.json

=============================================================================
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# =============================================================================
# CONSTANTS
# =============================================================================

# Cache line size (bytes) - x86_64 standard
CACHE_LINE_BYTES = 64

# Minimum work items to justify parallelization overhead
MIN_PARALLEL_TOKENS = 4
MIN_PARALLEL_HEADS = 2
MIN_PARALLEL_FEATURES = 256

# Thread count (could be detected at runtime, but we plan conservatively)
DEFAULT_NUM_THREADS = 8


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ParallelDecision:
    """Parallelization decision for a single op."""
    enabled: bool
    strategy: str  # "token", "head", "feature", "none"
    pragma: str    # Exact pragma string to emit (or empty)
    reason: str    # Human-readable explanation
    loop_target: str = "outer"  # Which loop to parallelize

    def to_dict(self) -> Dict:
        return {
            "enabled": self.enabled,
            "strategy": self.strategy,
            "pragma": self.pragma,
            "reason": self.reason,
            "loop_target": self.loop_target,
        }


@dataclass
class ParallelContext:
    """Global context for parallelization decisions."""
    mode: str  # "prefill" or "decode"
    config: Dict
    num_threads: int = DEFAULT_NUM_THREADS

    # Track what's already parallelized to avoid conflicts
    parallelized_buffers: Set[str] = field(default_factory=set)
    total_parallel_ops: int = 0

    # Memory regions being written to (for false sharing detection)
    # Maps buffer_name -> list of (start, end) byte ranges
    write_regions: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)


# =============================================================================
# FALSE SHARING DETECTION
# =============================================================================

def check_false_sharing_risk(
    buffer_name: str,
    write_start: int,
    write_end: int,
    ctx: ParallelContext
) -> bool:
    """
    Check if parallelizing writes to this region risks false sharing.

    Returns True if there's a risk (should NOT parallelize).
    """
    if buffer_name not in ctx.write_regions:
        return False

    for (existing_start, existing_end) in ctx.write_regions[buffer_name]:
        # Check if regions are within same cache line
        existing_cache_start = existing_start // CACHE_LINE_BYTES
        existing_cache_end = existing_end // CACHE_LINE_BYTES
        new_cache_start = write_start // CACHE_LINE_BYTES
        new_cache_end = write_end // CACHE_LINE_BYTES

        # If cache lines overlap but byte ranges don't fully overlap, risk exists
        if (new_cache_start <= existing_cache_end and
            new_cache_end >= existing_cache_start):
            # Check if it's the SAME region (not a conflict)
            if not (write_start == existing_start and write_end == existing_end):
                return True

    return False


def register_parallel_write(
    buffer_name: str,
    write_start: int,
    write_end: int,
    ctx: ParallelContext
) -> None:
    """Register that we're parallelizing writes to this region."""
    if buffer_name not in ctx.write_regions:
        ctx.write_regions[buffer_name] = []
    ctx.write_regions[buffer_name].append((write_start, write_end))


# =============================================================================
# OP-SPECIFIC PARALLELIZATION STRATEGIES
# =============================================================================

def plan_transpose_parallel(op: Dict, ctx: ParallelContext) -> ParallelDecision:
    """
    Plan parallelization for transpose operations.

    Transpose ops have nested loops over tokens and heads.
    In prefill: parallelize over tokens (outer loop)
    In decode: usually T=1, so no parallelization benefit
    """
    op_type = op.get("op", "")
    layer = op.get("layer", 0)

    if ctx.mode == "decode":
        return ParallelDecision(
            enabled=False,
            strategy="none",
            pragma="",
            reason="Decode mode: T=1, no parallelization benefit"
        )

    # Prefill mode - parallelize over tokens
    num_heads = ctx.config.get("num_heads", 14)
    num_kv_heads = ctx.config.get("num_kv_heads", 2)

    # Determine which buffer we're writing to
    if "kv" in op_type.lower() or "k_scratch" in str(op) or "v_scratch" in str(op):
        heads = num_kv_heads
        buffer = f"kv_scratch_layer{layer}"
    else:
        heads = num_heads
        buffer = f"q_scratch_layer{layer}"

    # Check if another op already parallelized this buffer
    if buffer in ctx.parallelized_buffers:
        return ParallelDecision(
            enabled=False,
            strategy="none",
            pragma="",
            reason=f"Buffer {buffer} already parallelized by another op"
        )

    # Token parallelization for transpose
    ctx.parallelized_buffers.add(buffer)
    ctx.total_parallel_ops += 1

    return ParallelDecision(
        enabled=True,
        strategy="token",
        pragma=f"#pragma omp parallel for schedule(static) if(num_tokens >= {MIN_PARALLEL_TOKENS})",
        reason=f"Prefill transpose: parallelize over tokens, {heads} heads",
        loop_target="outer"
    )


def plan_gemm_parallel(op: Dict, ctx: ParallelContext) -> ParallelDecision:
    """
    Plan parallelization for GEMM operations.

    GEMM: C[M,N] = A[M,K] @ B[K,N]

    Prefill (M > 1): Parallelize over M (rows/tokens) - each row independent
    Decode (M = 1): Parallelize over N (output features) - but watch for false sharing
    """
    if ctx.mode == "prefill":
        return ParallelDecision(
            enabled=True,
            strategy="token",
            pragma=f"#pragma omp parallel for schedule(static) if(num_tokens >= {MIN_PARALLEL_TOKENS})",
            reason="GEMM prefill: parallelize over M (tokens)",
            loop_target="M_loop"
        )
    else:
        # Decode: M=1, could parallelize over N but need to be careful
        N = ctx.config.get("embed_dim", 896)

        # Check if N is cache-line aligned for clean partitioning
        elements_per_cache_line = CACHE_LINE_BYTES // 4  # float32
        if N % (ctx.num_threads * elements_per_cache_line) == 0:
            return ParallelDecision(
                enabled=True,
                strategy="feature",
                pragma=f"#pragma omp parallel for schedule(static)",
                reason=f"GEMM decode: parallelize over N={N} (cache-aligned)",
                loop_target="N_loop"
            )
        else:
            return ParallelDecision(
                enabled=False,
                strategy="none",
                pragma="",
                reason=f"GEMM decode: N={N} not cache-aligned, false sharing risk"
            )


def plan_attention_parallel(op: Dict, ctx: ParallelContext) -> ParallelDecision:
    """
    Plan parallelization for attention operations.

    Attention is naturally parallelizable over heads - each head is independent.
    """
    num_heads = ctx.config.get("num_heads", 14)

    if num_heads >= MIN_PARALLEL_HEADS:
        return ParallelDecision(
            enabled=True,
            strategy="head",
            pragma=f"#pragma omp parallel for schedule(static)",
            reason=f"Attention: parallelize over {num_heads} heads",
            loop_target="head_loop"
        )
    else:
        return ParallelDecision(
            enabled=False,
            strategy="none",
            pragma="",
            reason=f"Attention: only {num_heads} heads, not worth parallelizing"
        )


def plan_elementwise_parallel(op: Dict, ctx: ParallelContext) -> ParallelDecision:
    """
    Plan parallelization for elementwise ops (residual_add, swiglu, etc.)

    These are often MEMORY BOUND - parallelization may not help much.
    Only parallelize if we have enough elements AND are in prefill mode.
    """
    op_type = op.get("op", "")

    if ctx.mode == "decode":
        return ParallelDecision(
            enabled=False,
            strategy="none",
            pragma="",
            reason="Elementwise decode: memory-bound, parallelization adds overhead"
        )

    # Prefill: parallelize over tokens if enough
    return ParallelDecision(
        enabled=True,
        strategy="token",
        pragma=f"#pragma omp parallel for schedule(static) if(num_tokens >= {MIN_PARALLEL_TOKENS})",
        reason=f"Elementwise prefill: parallelize over tokens",
        loop_target="token_loop"
    )


def plan_kv_cache_parallel(op: Dict, ctx: ParallelContext) -> ParallelDecision:
    """
    Plan parallelization for KV cache operations.

    KV cache copy is parallelizable over heads - each head's data is independent.
    """
    num_kv_heads = ctx.config.get("num_kv_heads", 2)

    if num_kv_heads >= MIN_PARALLEL_HEADS:
        return ParallelDecision(
            enabled=True,
            strategy="head",
            pragma=f"#pragma omp parallel for schedule(static)",
            reason=f"KV cache: parallelize over {num_kv_heads} KV heads",
            loop_target="head_loop"
        )
    else:
        return ParallelDecision(
            enabled=False,
            strategy="none",
            pragma="",
            reason=f"KV cache: only {num_kv_heads} KV heads, not worth parallelizing"
        )


# =============================================================================
# MAIN PASS
# =============================================================================

# Map op types to planning functions
OP_PLANNERS = {
    # Transpose ops
    "transpose_kv_to_head_major": plan_transpose_parallel,
    "transpose_qkv_to_head_major": plan_transpose_parallel,
    "transpose_attn_out_to_token_major": plan_transpose_parallel,

    # GEMM ops (handled by kernel, but we annotate for future)
    "linear": plan_gemm_parallel,
    "qkv_project": plan_gemm_parallel,
    "attn_proj": plan_gemm_parallel,
    "out_proj": plan_gemm_parallel,
    "mlp_up": plan_gemm_parallel,
    "mlp_gate": plan_gemm_parallel,
    "mlp_down": plan_gemm_parallel,
    "lm_head": plan_gemm_parallel,

    # Attention
    "attention": plan_attention_parallel,
    "attention_decode": plan_attention_parallel,
    "attention_prefill": plan_attention_parallel,

    # Elementwise
    "residual_add": plan_elementwise_parallel,
    "swiglu": plan_elementwise_parallel,
    "silu": plan_elementwise_parallel,
    "gelu": plan_elementwise_parallel,

    # KV cache
    "kv_cache_store": plan_kv_cache_parallel,
    "kv_cache_batch_copy": plan_kv_cache_parallel,
}


def plan_op_parallel(op: Dict, ctx: ParallelContext) -> ParallelDecision:
    """Plan parallelization for a single op."""
    op_type = op.get("op", "")

    # Check if we have a specific planner
    planner = OP_PLANNERS.get(op_type)
    if planner:
        return planner(op, ctx)

    # Default: no parallelization for unknown ops
    return ParallelDecision(
        enabled=False,
        strategy="none",
        pragma="",
        reason=f"Unknown op type: {op_type}"
    )


def run_parallel_pass(lowered_ir: Dict, mode: str) -> Tuple[Dict, Dict]:
    """
    Run the parallelization pass on lowered IR.

    Args:
        lowered_ir: Lowered IR dict with "operations" list
        mode: "prefill" or "decode"

    Returns:
        (annotated_ir, stats)
    """
    result = copy.deepcopy(lowered_ir)
    config = result.get("config", {})

    ctx = ParallelContext(
        mode=mode,
        config=config,
    )

    stats = {
        "mode": mode,
        "total_ops": 0,
        "parallelized_ops": 0,
        "strategies": {},
        "skipped_reasons": {},
    }

    operations = result.get("operations", [])

    for op in operations:
        stats["total_ops"] += 1

        decision = plan_op_parallel(op, ctx)
        op["parallel"] = decision.to_dict()

        if decision.enabled:
            stats["parallelized_ops"] += 1
            stats["strategies"][decision.strategy] = \
                stats["strategies"].get(decision.strategy, 0) + 1
        else:
            stats["skipped_reasons"][decision.reason] = \
                stats["skipped_reasons"].get(decision.reason, 0) + 1

    return result, stats


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Parallel pass: annotate IR with OpenMP parallelization decisions"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Input lowered IR JSON")
    parser.add_argument("--output", "-o", required=True,
                        help="Output annotated IR JSON")
    parser.add_argument("--mode", choices=["prefill", "decode"], default="prefill",
                        help="Execution mode")
    parser.add_argument("--report", "-r",
                        help="Output parallelization report JSON")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    # Load input
    with open(args.input) as f:
        lowered_ir = json.load(f)

    # Detect mode from IR if not specified
    mode = args.mode
    if "mode" in lowered_ir:
        mode = lowered_ir["mode"]

    # Run pass
    annotated_ir, stats = run_parallel_pass(lowered_ir, mode)

    # Write output
    with open(args.output, "w") as f:
        json.dump(annotated_ir, f, indent=2)

    print(f"[PARALLEL PASS] {args.input} -> {args.output}")
    print(f"  Mode: {mode}")
    print(f"  Total ops: {stats['total_ops']}")
    print(f"  Parallelized: {stats['parallelized_ops']}")

    if args.verbose:
        print(f"  Strategies: {stats['strategies']}")
        print(f"  Skipped: {stats['skipped_reasons']}")

    # Write report if requested
    if args.report:
        with open(args.report, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Report: {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
