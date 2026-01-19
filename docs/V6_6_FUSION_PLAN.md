# V6.6 Fusion Integration Plan

## Goal
Make v6.6 faster than v6.5 by integrating mega-fused kernels into codegen, with unified buffer allocation.

## Architecture Vision

The correct data flow should be:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. BUMP FILE                                                            │
│    weights.bump + weights_manifest.json                                 │
│    (weights, quant types, sizes, offsets)                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. BUILD GRAPH IR                                                        │
│    graph.json - symbolic computation graph                              │
│    (ops: rmsnorm→linear_q→linear_k→linear_v→attention→...)             │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. LOWER IR + KERNEL SELECTION                                          │
│    lowered.json - concrete kernels with dtypes from manifest            │
│    Each op becomes: {"kernel": "gemv_q4_k_q8_k", "inputs": [...], ...}  │
│    Kernel registry provides: inputs, outputs, scratch requirements      │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. FUSION PASS                                                          │
│    Parse lowered.json, find fusable sequences, compress                 │
│    [rmsnorm, linear_q, linear_k, linear_v, attention, linear_o, add]   │
│    → [mega_fused_attention_prefill]                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. MEMORY PLANNER                                                       │
│    Walk fused IR, collect ALL buffer requirements from kernel registry  │
│    Allocate once for max_context_len (same for prefill & decode)        │
│    Output: memory_layout.json with exact offsets                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 6. CODEGEN                                                              │
│    Read fused IR + memory layout                                        │
│    Emit inference.c (prefill + decode), main.c                          │
│    All offsets hardcoded, all buffers pre-allocated                     │
└─────────────────────────────────────────────────────────────────────────┘
```

## Current State

### What Works (v6.5)
- `mega_fused_attention_prefill` - 1.45x speedup
- `mega_fused_outproj_mlp_prefill` - 1.08-1.14x speedup
- Unit tests pass in `unittest/fusion/`

### What's Broken (v6.6)
1. `codegen_v6_6.py` doesn't read fused ops from IR for prefill
2. `fusion_patterns.py` missing mega_fused patterns
3. Buffer allocation differs between prefill and decode (unnecessary complexity)
4. **Kernel registry lacks I/O specs** - doesn't say what buffers each kernel needs
5. **Memory planning before kernel selection** - wrong order
6. **Fusion pass disconnected** - patterns exist but never matched

## Phase 0: Kernel Registry with I/O Specs (FOUNDATION)

This is the missing piece. Every kernel must declare what buffers it needs.

### New File: `scripts/v6.6/kernel_registry.py`

```python
"""
Kernel Registry - Central definition of all kernels and their buffer requirements.

For each kernel:
- inputs: list of input buffer specs (name, dtype, shape)
- outputs: list of output buffer specs
- scratch: list of scratch buffer specs (temporary, can be reused)
- dims: dimension parameters the kernel needs

This is the SINGLE SOURCE OF TRUTH for memory planning.
Codegen reads this. Memory planner reads this. Fusion pass reads this.
"""

KERNEL_REGISTRY = {
    # =========================================================================
    # BASIC KERNELS
    # =========================================================================
    "rmsnorm_forward": {
        "inputs": [
            {"name": "input", "dtype": "fp32", "shape": ["T", "E"]},
            {"name": "gamma", "dtype": "fp32", "shape": ["E"]},
        ],
        "outputs": [
            {"name": "output", "dtype": "fp32", "shape": ["T", "E"]},
        ],
        "scratch": [
            {"name": "rstd", "dtype": "fp32", "shape": ["T"]},  # Optional, for backward
        ],
        "dims": ["T", "E"],
    },

    "gemm_nt_q4_k": {
        "inputs": [
            {"name": "A", "dtype": "fp32", "shape": ["M", "K"]},
            {"name": "B", "dtype": "q4_k", "shape": ["N", "K"]},  # Transposed
        ],
        "outputs": [
            {"name": "C", "dtype": "fp32", "shape": ["M", "N"]},
        ],
        "scratch": [],
        "dims": ["M", "N", "K"],
    },

    "gemv_q4_k_q8_k": {
        "inputs": [
            {"name": "A", "dtype": "q8_k", "shape": ["K"]},  # Quantized activation
            {"name": "B", "dtype": "q4_k", "shape": ["N", "K"]},
        ],
        "outputs": [
            {"name": "C", "dtype": "fp32", "shape": ["N"]},
        ],
        "scratch": [],
        "dims": ["N", "K"],
    },

    "quantize_row_q8_k": {
        "inputs": [
            {"name": "input", "dtype": "fp32", "shape": ["K"]},
        ],
        "outputs": [
            {"name": "output", "dtype": "q8_k", "shape": ["K"]},
        ],
        "scratch": [],
        "dims": ["K"],
    },

    # =========================================================================
    # FUSED KERNELS
    # =========================================================================
    "mega_fused_attention_prefill": {
        "inputs": [
            {"name": "input", "dtype": "fp32", "shape": ["T", "E"]},
            {"name": "residual", "dtype": "fp32", "shape": ["T", "E"]},
            {"name": "ln1_gamma", "dtype": "fp32", "shape": ["E"]},
            {"name": "wq", "dtype": "q5_0|q8_0", "shape": ["H*D", "E"]},
            {"name": "wk", "dtype": "q5_0|q8_0", "shape": ["KV*D", "E"]},
            {"name": "wv", "dtype": "q5_0|q8_0", "shape": ["KV*D", "E"]},
            {"name": "wo", "dtype": "q5_0|q8_0", "shape": ["E", "H*D"]},
        ],
        "outputs": [
            {"name": "output", "dtype": "fp32", "shape": ["T", "E"]},
        ],
        "scratch": [
            {"name": "q", "dtype": "fp32", "shape": ["H", "T", "D"]},
            {"name": "kv_cache_k", "dtype": "fp32", "shape": ["KV", "max_T", "D"]},
            {"name": "kv_cache_v", "dtype": "fp32", "shape": ["KV", "max_T", "D"]},
            {"name": "attn_out", "dtype": "fp32", "shape": ["H", "T", "D"]},
            {"name": "q8_scratch", "dtype": "q8_0", "shape": ["T", "E"]},
        ],
        "dims": ["T", "E", "H", "KV", "D", "max_T"],
        "fuses": ["rmsnorm", "linear_q", "linear_k", "linear_v", "attention", "linear_o", "residual_add"],
    },

    "mega_fused_outproj_mlp_prefill": {
        "inputs": [
            {"name": "attn_out", "dtype": "fp32", "shape": ["H", "T", "D"]},  # Head-major
            {"name": "residual", "dtype": "fp32", "shape": ["T", "E"]},
            {"name": "ln2_gamma", "dtype": "fp32", "shape": ["E"]},
            {"name": "wo", "dtype": "q5_0", "shape": ["E", "H*D"]},
            {"name": "w1", "dtype": "q5_0", "shape": ["2*I", "E"]},  # Gate+Up packed
            {"name": "w2", "dtype": "q4_k|q6_k", "shape": ["E", "I"]},
        ],
        "outputs": [
            {"name": "output", "dtype": "fp32", "shape": ["T", "E"]},
        ],
        "scratch": [
            {"name": "proj_out", "dtype": "fp32", "shape": ["T", "E"]},
            {"name": "ln2_out", "dtype": "fp32", "shape": ["T", "E"]},
            {"name": "fc1_out", "dtype": "fp32", "shape": ["T", "2*I"]},
            {"name": "swiglu_out", "dtype": "fp32", "shape": ["T", "I"]},
            {"name": "q8_scratch", "dtype": "q8_0", "shape": ["T", "E"]},
        ],
        "dims": ["T", "E", "H", "D", "I"],
        "fuses": ["linear_o", "residual_add", "rmsnorm", "linear_w1", "swiglu", "linear_w2", "residual_add"],
    },
}


def get_kernel_buffer_requirements(kernel_name: str, dims: dict) -> dict:
    """
    Given a kernel and concrete dimensions, compute exact buffer sizes.

    Args:
        kernel_name: e.g., "mega_fused_attention_prefill"
        dims: e.g., {"T": 512, "E": 2048, "H": 32, "D": 64, "I": 8192, "max_T": 2048}

    Returns:
        {"inputs": [...], "outputs": [...], "scratch": [...], "total_bytes": N}
    """
    spec = KERNEL_REGISTRY[kernel_name]

    def resolve_shape(shape_expr, dims):
        """Resolve ["T", "E"] → [512, 2048]"""
        result = []
        for s in shape_expr:
            if isinstance(s, int):
                result.append(s)
            elif isinstance(s, str):
                # Handle expressions like "H*D", "2*I"
                expr = s
                for dim_name, dim_val in dims.items():
                    expr = expr.replace(dim_name, str(dim_val))
                result.append(eval(expr))
        return result

    def compute_size(shape, dtype):
        """Compute buffer size in bytes."""
        elements = 1
        for dim in shape:
            elements *= dim
        # Simplified - real implementation would use QUANT_BLOCK_INFO
        if dtype.startswith("q"):
            return elements  # Placeholder
        return elements * 4  # FP32

    result = {"inputs": [], "outputs": [], "scratch": [], "total_bytes": 0}

    for category in ["inputs", "outputs", "scratch"]:
        for buf in spec[category]:
            shape = resolve_shape(buf["shape"], dims)
            size = compute_size(shape, buf["dtype"])
            result[category].append({
                "name": buf["name"],
                "dtype": buf["dtype"],
                "shape": shape,
                "size": size,
            })
            if category != "inputs":  # Inputs are weights, don't count
                result["total_bytes"] += size

    return result
```

### Why This Matters

1. **Memory planner** reads this to allocate buffers
2. **Fusion pass** reads this to know what fused kernel replaces
3. **Codegen** reads this to emit correct buffer pointers
4. **Validation** can check if all buffers are allocated

## Phase 1: Unified Buffer Allocation

### Problem
`codegen_v6_6.py` uses different memory strategies:
- **Prefill** (line 1094): Uses `proj_scratch` from IR layout
- **Decode** (line 1636): Stack arrays OR arena pointers

### Solution
Memory planner reads kernel registry, allocates ALL buffers for `max_context_len` upfront. Same buffers for prefill and decode.

### Files to Change

**1. `scripts/v6.6/memory_planner.py`** (NEW FILE)
```python
"""
Memory Planner - Allocates buffers based on kernel registry requirements.

Input: fused_ir.json (list of kernels per layer)
Output: memory_layout.json (buffer offsets)

Key principle: Allocate for max_context_len, use for both prefill and decode.
"""

from kernel_registry import KERNEL_REGISTRY, get_kernel_buffer_requirements

def plan_memory(fused_ir: dict, max_context_len: int) -> dict:
    """
    Walk fused IR, collect buffer requirements, allocate.
    """
    allocator = BumpAllocator()
    layout = {"layers": [], "total_bytes": 0}

    dims = {
        "T": max_context_len,  # Allocate for max
        "E": fused_ir["config"]["embed_dim"],
        "H": fused_ir["config"]["num_heads"],
        "KV": fused_ir["config"]["num_kv_heads"],
        "D": fused_ir["config"]["head_dim"],
        "I": fused_ir["config"]["intermediate_dim"],
        "max_T": max_context_len,
    }

    for layer in fused_ir["layers"]:
        layer_buffers = {}
        for op in layer["ops"]:
            kernel = op["kernel"]
            reqs = get_kernel_buffer_requirements(kernel, dims)

            # Allocate outputs and scratch (inputs are weights, already in bump)
            for buf in reqs["outputs"] + reqs["scratch"]:
                name = f"layer.{layer['id']}.{buf['name']}"
                if name not in layer_buffers:  # Avoid duplicates
                    offset = allocator.alloc(buf["size"])
                    layer_buffers[name] = {"offset": offset, "size": buf["size"]}

        layout["layers"].append({"id": layer["id"], "buffers": layer_buffers})

    layout["total_bytes"] = allocator.current_offset()
    return layout
```

**2. `scripts/v6.6/codegen_v6_6.py`**
- Remove decode stack array fallback (lines 1635-1643)
- Read memory_layout.json for buffer offsets
- Use same pointers for both prefill and decode

### Buffer Layout After Phase 1

```
Per-Layer Buffers (allocated once for max_context_len):
├── q_scratch      [H, max_T, D]     - Q projection output
├── k_cache        [KV, max_T, D]    - K cache (grows during generation)
├── v_cache        [KV, max_T, D]    - V cache (grows during generation)
├── attn_scratch   [H, max_T, D]     - Attention output
├── proj_scratch   [max_T, E]        - Out-projection + Q8 scratch
├── mlp_scratch    [max_T, 2*I]      - MLP intermediate (fc1_out)
└── layer_out      [max_T, E]        - Layer output

Runtime:
- Prefill: tokens = prompt_len (use first N rows of buffer)
- Decode: tokens = 1 (use first 1 row of buffer)
- Same buffers, different `tokens` parameter
```

## Phase 2: Add Mega-Fused Patterns to IR

### Files to Change

**1. `scripts/v6.6/fusion_patterns.py`**
Add two new patterns:

```python
# Pattern: mega_fused_attention_prefill
# Fuses: RMSNorm + QKV projection + Flash Attention + Out-projection + Residual
MEGA_FUSED_ATTENTION_PREFILL = FusionPattern(
    name="mega_fused_attention_prefill",
    ops=["rmsnorm", "linear_q", "linear_k", "linear_v", "attention", "linear_o", "residual_add"],
    constraints=[
        "prefill_mode",           # tokens > 1
        "q_weight_dtype in [q5_0, q8_0]",
        "kv_weight_dtype in [q5_0, q8_0]",
        "o_weight_dtype in [q5_0, q8_0]",
    ],
    kernel="mega_fused_attention_prefill",
    speedup_estimate=1.45,
)

# Pattern: mega_fused_outproj_mlp_prefill
# Fuses: Out-projection + Residual + RMSNorm + MLP (SwiGLU) + Residual
MEGA_FUSED_OUTPROJ_MLP_PREFILL = FusionPattern(
    name="mega_fused_outproj_mlp_prefill",
    ops=["linear_o", "residual_add", "rmsnorm", "linear_w1", "swiglu", "linear_w2", "residual_add"],
    constraints=[
        "prefill_mode",
        "o_weight_dtype == q5_0",
        "w1_weight_dtype == q5_0",
        "w2_weight_dtype in [q4_k, q6_k]",
    ],
    kernel="mega_fused_outproj_mlp_prefill",
    speedup_estimate=1.10,
)
```

**2. `scripts/v6.6/build_ir_v6_6.py`**
- Import new patterns
- Apply during IR construction with `--fusion=on`

## Phase 3: Codegen Reads IR Fusion Flags

### Current Problem
`codegen_v6_6.py` emits a fixed sequence of kernel calls regardless of IR fusion flags.

### Solution
Change `emit_prefill_layer()` to check IR for fused ops:

```python
def emit_prefill_layer(layer, ir_node, ...):
    # Check if this layer has a fused attention op in IR
    if ir_node.get("fused_op") == "mega_fused_attention_prefill":
        emit_mega_fused_attention_prefill(layer, ...)
    else:
        # Fallback: emit separate kernels
        emit_rmsnorm(...)
        emit_qkv_projection(...)
        emit_attention(...)
        emit_out_projection(...)
        emit_residual_add(...)
```

### Files to Change

**1. `scripts/v6.6/codegen_v6_6.py`**
- Add `emit_mega_fused_attention_prefill()` function
- Add `emit_mega_fused_outproj_mlp_prefill()` function
- Modify `emit_prefill_layer()` to check IR fusion flags
- Remove the hardcoded kernel sequence for prefill

## Phase 4: Validation

### Tests to Run
```bash
# Unit tests for fused kernels
make test-fusion

# Numerical parity vs llama.cpp
make llamacpp-fusion-test-full

# End-to-end v6.6 inference
python scripts/v6.6/test_v6_6_inference.py --model qwen2.5-1.5b

# Performance comparison
python scripts/bench_mega_fused_attention_prefill.py
```

### Success Criteria
1. All unit tests pass
2. Numerical parity with llama.cpp (max_diff < 1e-4)
3. v6.6 prefill faster than v6.5 by ~1.3-1.4x

## What Exists vs What Needs Building

| Component | v6.5 Status | v6.6 Status | Needs Work? |
|-----------|-------------|-------------|-------------|
| `convert_gguf_to_bump` | Works | Works | No |
| `weights_manifest.json` | Has dtypes, offsets | Has dtypes, offsets | No |
| `build_ir_v6_6.py` | N/A | Creates graph.json | Needs fusion pass integration |
| `v6_6_ir_lowering.py` | N/A | Allocates buffers | Needs to read kernel registry |
| **`kernel_registry.py`** | N/A | **DOESN'T EXIST** | **CREATE** |
| **`memory_planner.py`** | N/A | **DOESN'T EXIST** | **CREATE** |
| **`fusion_pass.py`** | N/A | Patterns exist, not connected | **FIX** |
| `codegen_v6_6.py` | N/A | Emits C, ignores fusion | Needs to read fused IR |
| Fused kernels (C) | mega_fused_* work | mega_fused_* work | No (already have) |
| Unit tests | Pass | Pass | No |

## New Data Flow (After Implementation)

```
weights.bump + manifest.json
         ↓
┌─────────────────────────────────────────┐
│ build_ir_v6_6.py                        │
│ → graph.json (symbolic ops)             │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│ lower_ir.py                             │
│ → lowered.json (concrete kernels)       │
│   Each op → kernel from registry        │
│   {"kernel": "gemm_nt_q4_k", ...}       │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│ fusion_pass.py (NEW)                    │
│ → fused.json (compressed ops)           │
│   [rmsnorm, qkv, attn, proj, add]       │
│   → [mega_fused_attention_prefill]      │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│ memory_planner.py (NEW)                 │
│ → memory_layout.json                    │
│   Reads kernel_registry for buffers     │
│   Allocates for max_context_len         │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│ codegen_v6_6.py                         │
│ → inference.c + main.c                  │
│   Reads fused.json + memory_layout      │
│   Emits unified prefill/decode          │
└─────────────────────────────────────────┘
```

## File Change Summary

| File | Status | Changes |
|------|--------|---------|
| `kernel_registry.py` | **NEW** | Define all kernels with I/O specs |
| `memory_planner.py` | **NEW** | Allocate buffers from kernel specs |
| `fusion_pass.py` | **NEW** | Pattern match and compress ops |
| `v6_6_ir_lowering.py` | Modify | Remove BumpAllocator (moved to memory_planner) |
| `fusion_patterns.py` | Modify | Add mega_fused patterns |
| `build_ir_v6_6.py` | Modify | Integrate fusion pass |
| `codegen_v6_6.py` | Modify | Read fused IR, remove stack arrays |

## Order of Implementation

1. **Phase 0: kernel_registry.py** - Create with all kernel I/O specs
2. **Phase 1: memory_planner.py** - Create, reads kernel registry
3. **Phase 2: fusion_pass.py** - Create, uses patterns from fusion_patterns.py
4. **Phase 3: Wire it together** - build_ir → lower → fuse → plan → codegen
5. **Phase 4: Validation** - Run existing tests

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Memory increase from unified buffers | Still smaller than training layout; user specifies max_context |
| Fusion pattern matching complexity | Start with exact match, no heuristics |
| Codegen complexity | Keep fallback to non-fused path |
| Breaking existing v6.6 | Keep old codegen path, add new one alongside |

## Estimated LOC

| File | Estimated Lines |
|------|-----------------|
| `kernel_registry.py` | ~300 (kernel specs) |
| `memory_planner.py` | ~150 (bump allocator) |
| `fusion_pass.py` | ~200 (pattern matching) |
| `codegen_v6_6.py` changes | ~100 (fused kernel emitters) |
| **Total new code** | **~750 lines** |
