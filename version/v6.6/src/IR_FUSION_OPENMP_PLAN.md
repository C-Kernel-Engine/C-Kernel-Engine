# v6.6 IR-Level Fusion & OpenMP Parallelism Plan

## Executive Summary

Extend the IR (Intermediate Representation) to:
1. **Detect fusion opportunities** - recognize patterns based on actual operations in the graph
2. **Emit fused kernel calls** - replace pattern with `mega_fused_attention_prefill()`
3. **Emit OpenMP parallel regions** - with semantic understanding of work distribution

**Key Principle:** IR is model-agnostic. It doesn't know "this is Qwen" - it just sees operations: `RMSNorm → QKV → Attention → OutProj → MLP → Add`. If that pattern exists, fuse it.

## How IR Decides Fusion (Model-Agnostic)

```
┌─────────────────────────────────────────────────────────────────┐
│  MODEL LOADING (Qwen / Llama / Mistral / Any Transformer)      │
│                                                                 │
│  Config: {hidden_dim: 896, num_heads: 14, num_layers: 24, ...} │
│                                                                 │
│  ↓                                                               │
│  IR GRAPH CREATED (operations, NOT model-specific)              │
│                                                                 │
│  Layer 0:                                                        │
│    nodes[0]  = CK_OP_RMSNORM    (from pre-layer norm)           │
│    nodes[1]  = CK_OP_LINEAR_QKV (Q, K, V projections)           │
│    nodes[2]  = CK_OP_ATTENTION  (causal attention)              │
│    nodes[3]  = CK_OP_LINEAR     (output projection)             │
│    nodes[4]  = CK_OP_SWIGLU     (MLP activation)                │
│    nodes[5]  = CK_OP_ADD        (residual)                      │
│                                                                 │
│  Layer 1: [same pattern]                                        │
│  Layer 2: [same pattern]                                        │
│  ...                                                             │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  FUSION PASS (IR looks at OPERATIONS, not model name)           │
│                                                                 │
│  Pattern Match: RMSNorm → QKV → Attn → OutProj → MLP → Add     │
│                                                                 │
│  IR says: "These operations match mega-fusion pattern!"         │
│  ↓                                                               │
│  Mark nodes: fusion_group_id = 1 (this layer is fused)          │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  CODEGEN (Emits what IR says)                                   │
│                                                                 │
│  // IR says fuse layer 0 → codegen emits:                       │
│  mega_fused_attention_prefill(                                   │
│      output, input, residual,                                   │
│      wq, wk, wv, wo,                                            │
│      kv_cache_k, kv_cache_v,                                    │
│      ...                                                        │
│  );                                                             │
│                                                                 │
│  // IR says parallelize → codegen emits:                        │
│  #pragma omp parallel num_threads(nth)                          │
│  {                                                              │
│      int ith = omp_get_thread_num();                            │
│      // kernel calls with ith, nth                              │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
```

## Current IR Structure

```
CKIRGraph
├── config (num_layers, hidden_size, etc.)
├── nodes[] (per-layer operations - MODEL-AGNOSTIC!)
│   ├── CK_OP_RMSNORM
│   ├── CK_OP_LINEAR_QKV
│   ├── CK_OP_ATTENTION
│   ├── CK_OP_ADD
│   ├── CK_OP_LINEAR (output proj)
│   └── CK_OP_SWIGLU
```

## Phase 1: Fusion Detection in IR

### 1.1 Add Fusion Opportunity Flags

```c
// In ckernel_ir.h - extend CKIRNode
typedef struct {
    CKKernelId id;
    CKOpType op;
    CKInputRef inputs[4];
    uint8_t n_inputs;
    uint8_t n_outputs;

    // NEW: Fusion metadata
    uint8_t can_fuse_prefetch;    // Can fuse with next attention
    uint8_t can_fuse_attention;   // Is attention that can be mega-fused
    uint8_t can_fuse_mlp;         // Can fuse with following MLP
    uint32_t fusion_group_id;     // ID of fusion group if merged
} CKIRNodeExtended;
```

### 1.2 Fusion Pattern Detection (Model-Agnostic)

```c
// In src/v6.6/ckernel_ir_fusion.c

/**
 * Detect mega-fusion opportunities by PATTERN MATCHING on operations.
 *
 * KEY: This function looks at CK_OP_* types, NOT model name!
 * It works for Qwen, Llama, Mistral, or any transformer that has
 * this operation sequence.
 *
 * Pattern: RMSNorm → QKV → Attention → OutProj → MLP → Add → [RMSNorm]
 *
 * Returns: fusion_group_id or 0 if no fusion
 */
int ck_ir_detect_mega_fusion(CKIRGraph *graph, int layer_start) {
    // Pattern we're looking for (operations, NOT model-specific!):
    // 1. CK_OP_RMSNORM (input preprocessing)
    // 2. CK_OP_LINEAR_QKV (Q, K, V projections)
    // 3. CK_OP_ATTENTION (causal attention)
    // 4. CK_OP_LINEAR (output projection)
    // 5. CK_OP_SWIGLU (MLP activation)
    // 6. CK_OP_ADD (residual connection)
    // 7. CK_OP_RMSNORM (next layer's norm, or end)

    // Scan nodes and look for the pattern
    int has_rmsnorm = 0, has_qkv = 0, has_attn = 0;
    int has_outproj = 0, has_swiglu = 0, has_add = 0;

    for (int i = layer_start; i < num_nodes; i++) {
        CKIRNode *n = &nodes[i];

        switch (n->op) {
            case CK_OP_RMSNORM:
                has_rmsnorm = 1;
                // Also check if this is pre-layer or post-layer
                n->can_fuse_prefetch = 1;  // Can be start of fusion
                break;

            case CK_OP_LINEAR_QKV:
                has_qkv = 1;
                n->can_fuse_attention = 1;  // QKV feeds into attention
                break;

            case CK_OP_ATTENTION:
                has_attn = 1;
                n->can_fuse_attention = 1;  // Core of mega fusion
                break;

            case CK_OP_LINEAR:  // Could be OutProj or MLP Down
                // Check if it follows attention (OutProj) or SwiGLU (Down)
                if (has_attn && !has_outproj) {
                    has_outproj = 1;
                    n->can_fuse_attention = 1;  // OutProj closes attention loop
                }
                break;

            case CK_OP_SWIGLU:
                has_swiglu = 1;
                n->can_fuse_mlp = 1;  // MLP fusion candidate
                break;

            case CK_OP_ADD:
                has_add = 1;
                n->can_fuse_mlp = 1;  // Add closes MLP loop
                break;

            default:
                break;
        }
    }

    // Check if we have the complete mega-fusion pattern
    int is_complete_pattern = has_rmsnorm && has_qkv && has_attn &&
                              has_outproj && has_swiglu && has_add;

    if (is_complete_pattern) {
        // Mark all nodes in this layer as part of fusion group 1
        for (int i = layer_start; i < num_nodes; i++) {
            nodes[i].fusion_group_id = 1;
        }
        return 1;  // Fusion group ID
    }

    return 0;  // No fusion
}
```

**This is model-agnostic!** It works for:
- **Qwen2**: RMSNorm → QKV(14→2) → Attn → OutProj → SwiGLU → Add
- **Llama7B**: RMSNorm → QKV(32→8) → Attn → OutProj → SwiGLU → Add
- **Mistral**: RMSNorm → QKV → Attn → OutProj → SwiGLU → Add

All match the same pattern → all get fused.

### 1.3 Fusion Decision Heuristics (Based on Operations + Hardware)

```c
/**
 * Decide whether to actually fuse based on:
 * 1. Pattern match (already detected above)
 * 2. Sequence length (fusion helps when intermediates > L2 cache)
 * 3. Hardware capabilities (cache size, thread count)
 *
 * NOT model-specific! Uses seq_len and hidden_dim which are
 * properties of any transformer, not specific to Qwen/Llama.
 */

/**
 * Decide whether to fuse based on:
 * 1. Sequence length (fusion helps more for longer sequences)
 * 2. Quantization format (Q5_0/Q8_0 fusion well-tested)
 * 3. Hardware (L2 cache size matters)
 */
typedef enum {
    CK_FUSION_NONE = 0,
    CK_FUSION_QKV_ONLY,           // Just RMSNorm + QKV
    CK_FUSION_ATTENTION_FULL,     // QKV + Attention + OutProj
    CK_FUSION_MEGA,               // Everything: RMSNorm + QKV + Attn + OutProj + MLP
    CK_FUSION_MLP_ONLY            // Just MLP portion
} CKFusionType;

CKFusionType ck_ir_decide_fusion(
    CKIRGraph *graph,
    int layer_idx,
    int seq_len,
    int hidden_dim
) {
    // Fusion only worthwhile if intermediates exceed L2 cache (~256KB)
    // x_norm for 128 tokens @ 896 dim = 459KB > L2
    // x_norm for 64 tokens @ 896 dim = 230KB < L2

    int x_norm_size = seq_len * hidden_dim * sizeof(float);  // 4 bytes per element

    if (x_norm_size > 256 * 1024) {
        // L2 cache exceeded - fusion helps
        if (graph_has_mlp_pattern) {
            return CK_FUSION_MEGA;  // Full fusion
        } else {
            return CK_FUSION_ATTENTION_FULL;
        }
    }

    // Short sequence - don't fuse (extra overhead not worth it)
    return CK_FUSION_NONE;
}
```

## Phase 2: IR Codegen for Fusion

### 2.1 Add Fusion Op Types

```c
// In ckernel_ir.h - extend CKOpType
typedef enum {
    // ... existing ops ...

    // NEW: Fusion operations
    CK_OP_MEGA_FUSED_ATTENTION_PREFILL,   // Full mega fusion for prefill
    CK_OP_MEGA_FUSED_ATTENTION_DECODE,    // Mega fusion for decode
    CK_OP_FUSED_QKV_RMSNORM,              // QKV + RMSNorm fusion
    CK_OP_FUSED_MLP,                      // Gate+Up → SwiGLU → Down fusion

    // NEW: Parallel region markers
    CK_OP_PARALLEL_START,                 // #pragma omp parallel
    CK_OP_PARALLEL_END,                   // #pragma omp barrier
    CK_OP_PARALLEL_FOR,                   // #pragma omp parallel for
} CKOpType;
```

### 2.2 Codegen for Fusion Kernels

```c
// In src/v6.6/ckernel_ir_codegen.c

/**
 * Emit fused attention prefill call.
 *
 * IR Node:
 *   inputs[0] = input (float*)
 *   inputs[1] = residual (float*)
 *   inputs[2] = ln_gamma (float*)
 *   weights: wq, wk, wv, wo (quantized)
 *   kv_cache_k, kv_cache_v
 *   rope_cos, rope_sin
 */
void ck_ir_emit_mega_fused_prefill(FILE *out, CKIRNode *node, CodegenContext *ctx) {
    // Emit function signature
    fprintf(out,
        "void mega_fused_attention_prefill(\n"
        "    float *output,           // %%%d\n"
        "    float *input,            // %%%d\n"
        "    float *residual,         // %%%d\n"
        "    float *ln_gamma,         // %%%d\n"
        "    void *wq, int wq_dt,     // Quantized weights\n"
        "    void *wk, int wk_dt,\n"
        "    void *wv, int wv_dt,\n"
        "    void *wo, int wo_dt,\n"
        "    float *kv_cache_k,\n"
        "    float *kv_cache_v,\n"
        "    float *rope_cos,\n"
        "    float *rope_sin,\n"
        "    int start_pos,\n"
        "    int tokens,\n"
        "    int embed_dim,\n"
        "    int aligned_embed_dim,\n"
        "    int num_heads,\n"
        "    int num_kv_heads,\n"
        "    int head_dim,\n"
        "    int aligned_head_dim,\n"
        "    float eps,\n"
        "    void *scratch\n"
        ");\n\n",
        node->outputs[0], node->inputs[0].producer, ...
    );

    // Emit call
    fprintf(out, "    // Mega-fused attention prefill\n");
    fprintf(out, "    mega_fused_attention_prefill(\n");
    fprintf(out, "        (float*)%%[%d],\n", node->outputs[0]);
    // ... emit all parameters ...
    fprintf(out, "    );\n\n");
}
```

### 2.3 Codegen for OpenMP Regions

```c
/**
 * Emit OpenMP parallel region with semantic understanding.
 *
 * Key insight: Parallelize at the ORCHESTRATION level, not inside kernels.
 * Each kernel receives ith/nth and processes only its slice.
 */
void ck_ir_emit_parallel_region(FILE *out, CKIRGraph *graph, int layer_idx) {
    fprintf(out, "    // ===== PARALLEL REGION START (Layer %d) =====\n", layer_idx);
    fprintf(out, "    {\n");
    fprintf(out, "        const int nth = omp_get_max_threads();\n");
    fprintf(out, "        #pragma omp parallel num_threads(nth)\n");
    fprintf(out, "        {\n");
    fprintf(out, "            const int ith = omp_get_thread_num();\n\n");

    // Emit all kernel calls with ith/nth parameters
    for (int node_idx = layer_start; node_idx < layer_end; node_idx++) {
        CKIRNode *node = &graph->nodes[node_idx];

        switch (node->op) {
            case CK_OP_RMSNORM:
                fprintf(out, "            // RMSNorm (parallel across hidden_dim)\n");
                fprintf(out, "            rmsnorm_threaded(%%[%d], gamma, tmp, D, eps, ith, nth);\n",
                    node->inputs[0], node->outputs[0]);
                break;

            case CK_OP_LINEAR_QKV:
                fprintf(out, "            // QKV projection (parallel across output rows)\n");
                fprintf(out, "            gemv_q4k_threaded(wq, tmp, q, q_dim, D, ith, nth);\n");
                fprintf(out, "            gemv_q4k_threaded(wk, tmp, k, kv_dim, D, ith, nth);\n");
                fprintf(out, "            gemv_q4k_threaded(wv, tmp, v, kv_dim, D, ith, nth);\n");
                break;

            case CK_OP_ATTENTION:
                fprintf(out, "            // Causal attention (parallel across heads)\n");
                fprintf(out, "            attention_decode_threaded(q, k_cache, v_cache, attn_out,\n");
                fprintf(out, "                                        n_heads, seq_len, head_dim, ith, nth);\n");
                break;

            case CK_OP_LINEAR:  // OutProj
                fprintf(out, "            // Output projection\n");
                fprintf(out, "            gemv_q4k_threaded(wo, attn_out, hidden_out, D, q_dim, ith, nth);\n");
                break;

            case CK_OP_SWIGLU:
                fprintf(out, "            // SwiGLU\n");
                fprintf(out, "            swiglu_threaded(gate, up, hidden_mlp, inter_dim, ith, nth);\n");
                break;

            case CK_OP_PARALLEL_END:
                fprintf(out, "            #pragma omp barrier\n");
                break;
        }
    }

    fprintf(out, "        }  // end parallel\n");
    fprintf(out, "    }  // end scope\n");
    fprintf(out, "    // ===== PARALLEL REGION END =====\n\n");
}
```

## Phase 3: Threaded Kernel Pattern

### 3.1 Kernel Signature Extension

```c
// In include/parallel_kernels.h (NEW FILE)

#ifndef CKERNEL_PARALLEL_KERNELS_H
#define CKERNEL_PARALLEL_KERNELS_H

#include <stdint.h>

/**
 * Threaded GEMV - processes only slice [r0, r1) of output rows.
 *
 * @param W      Quantized weight matrix (M x K)
 * @param x      Input vector (K)
 * @param out    Output vector (M) - only rows [r0, r1) written
 * @param M      Number of output rows
 * @param K      Input dimension
 * @param ith    Thread index (0 to nth-1)
 * @param nth    Total number of threads
 */
void gemv_q4k_threaded(
    const void *W,
    const float *x,
    float *out,
    int M, int K,
    int ith, int nth
);

void gemv_q8_0_threaded(
    const void *W,
    const float *x,
    float *out,
    int M, int K,
    int ith, int nth
);

/**
 * Threaded RMSNorm - parallel across hidden dimension.
 *
 * @param x      Input vector (D)
 * @param gamma  Scale parameter (D)
 * @param out    Output vector (D)
 * @param D      Hidden dimension
 * @param eps    epsilon for numerical stability
 * @param ith    Thread index
 * @param nth    Total threads
 */
void rmsnorm_threaded(
    const float *x,
    const float *gamma,
    float *out,
    int D,
    float eps,
    int ith, int nth
);

/**
 * Threaded SwiGLU - parallel across intermediate dimension.
 */
void swiglu_threaded(
    const float *gate,
    const float *up,
    float *out,
    int D,
    int ith, int nth
);

/**
 * Threaded attention decode - parallel across heads.
 *
 * For GQA: n_heads=14, n_kv=2, so each thread handles ~7 heads.
 */
void attention_decode_threaded(
    const float *q,      // [n_heads, head_dim]
    const float *k_cache,// [n_kv_heads, seq_len, head_dim]
    const float *v_cache,// [n_kv_heads, seq_len, head_dim]
    float *out,          // [n_heads, head_dim]
    int n_heads,
    int n_kv_heads,
    int seq_len,
    int head_dim,
    int ith, int nth
);

#endif
```

### 3.2 Implementation Pattern (Example)

```c
// In src/kernels/gemm_kernels_q4k.c

void gemv_q4k_threaded(const void *W, const float *x, float *out,
                       int M, int K, int ith, int nth) {
    // Compute work slice for this thread
    int dr = (M + nth - 1) / nth;  // rows per thread (ceiling)
    int r0 = dr * ith;             // start row
    int r1 = MIN(r0 + dr, M);      // end row (exclusive)

    // Only process our slice - no atomic ops needed!
    for (int row = r0; row < r1; row++) {
        gemv_q4k_row(W, x, &out[row], K, row);
    }
}
```

## Phase 4: IR Optimization Pipeline

### 4.1 Pass Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    IR OPTIMIZATION PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. PARSE          2. DETECT        3. DECIDE       4. TRANSFORM │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐   ┌───────────────┐│
│  │ Model     │───▶│ Fusion    │───▶│ Fusion    │──▶│ Fuse Nodes    ││
│  │ Config    │    │ Pattern   │    │ Decision  │   │ into Groups   ││
│  └───────────┘    └───────────┘    └───────────┘   └───────────────┘│
│                                                                  │
│                      5. PARALLEL         6. CODEGEN              │
│                   ┌───────────────┐   ┌───────────────────────┐  │
│                   │ Add OpenMP    │──▶│ Emit C code with      │  │
│                   │ Markers       │   │ fused calls + #pragma │  │
│                   └───────────────┘   └───────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 IR Transformation API

```c
/**
 * Run full IR optimization pipeline.
 *
 * @param graph         Input IR graph
 * @param seq_len       Sequence length (for fusion decision)
 * @param enable_openmp Whether to emit OpenMP parallel regions
 * @return Optimized graph
 */
CKIRGraph *ck_ir_optimize(
    CKIRGraph *graph,
    int seq_len,
    int enable_openmp
);

/**
 * Run fusion detection pass.
 * Marks nodes that can be fused and groups them.
 */
void ck_ir_pass_fusion_detect(CKIRGraph *graph);

/**
 * Run fusion decision pass.
 * Decides which groups to actually fuse based on heuristics.
 */
void ck_ir_pass_fusion_decide(CKIRGraph *graph, int seq_len, int hidden_dim);

/**
 * Run OpenMP parallelization pass.
 * Adds CK_OP_PARALLEL_START/END markers.
 */
void ck_ir_pass_parallelize(CKIRGraph *graph, int num_threads);
```

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `include/parallel_kernels.h` | Threaded kernel declarations |
| `src/kernels/parallel_kernels.c` | Threaded kernel implementations |
| `src/v6.6/ckernel_ir_fusion.c` | Fusion detection & decision |
| `src/v6.6/ckernel_ir_parallel.c` | OpenMP parallelization pass |

### Modified Files

| File | Change |
|------|--------|
| `include/ckernel_ir.h` | Add fusion op types, parallel markers |
| `src/v6.6/ckernel_ir_v6.6.c` | Add optimization pipeline |
| `src/v6.6/ckernel_ir_codegen.c` | Emit fused kernel calls, #pragma omp |
| `src/kernels/gemm_kernels_*.c` | Add `_threaded` variants |

## Testing Strategy

### Unit Tests

```c
// test_ir_fusion.c
void test_fusion_detection(void) {
    // Build IR for: RMSNorm → QKV → Attn → OutProj → MLP → Add
    CKIRGraph *graph = build_test_graph();

    ck_ir_pass_fusion_detect(graph);

    // Verify nodes are marked as fusion candidates
    assert(nodes[0].can_fuse_prefetch == 1);
    assert(nodes[1].can_fuse_attention == 1);
    assert(nodes[3].can_fuse_attention == 1);
}

void test_fusion_decision(void) {
    // Long sequence - should fuse
    CKIRGraph *graph = build_test_graph();
    ck_ir_pass_fusion_decide(graph, 128, 896);
    assert(graph->fusion_group_id != 0);

    // Short sequence - should NOT fuse
    CKIRGraph *graph2 = build_test_graph();
    ck_ir_pass_fusion_decide(graph2, 16, 896);
    assert(graph2->fusion_group_id == 0);
}

void test_parallel_codegen(void) {
    // Verify generated code has correct #pragma omp
}
```

### Integration Tests

```bash
# Run with Qwen model
./ck-engine --model Qwen2.5-0.5B --ir-fusion --openmp

# Compare output against reference
python3 compare_outputs.py --gold llama.cpp --test ck-engine

# Benchmark tok/sec
./ck-engine --model Qwen2.5-0.5B --timing
```

## Expected Performance

| Configuration | tok/sec | Notes |
|---------------|---------|-------|
| v6.5 (baseline) | ~8 | Single-threaded, no fusion |
| v6.6 (fusion only) | ~14 | 1.6-1.8x from fusion |
| v6.6 (fusion + OpenMP) | ~25-30 | Match llama.cpp |

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Thread overhead for small ops | Add minimum work threshold before parallelizing |
| Memory bandwidth saturation | Limit thread count (4T sweet spot per benchmarks) |
| Load imbalance (GQA) | Static work distribution: heads / nth |
| Regression in correctness | Compare against llama.cpp on test prompts |

## Timeline

| Week | Milestone |
|------|-----------|
| 1 | Add fusion op types, detection pass |
| 2 | Add threaded kernel implementations |
| 3 | Add OpenMP pass, codegen integration |
| 4 | Testing, bug fixes, tuning |
