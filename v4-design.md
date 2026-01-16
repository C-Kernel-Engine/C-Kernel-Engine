# IR v4 Design Document

IR v4 is the "bridge release" that unifies the best parts of v2 (lowering + memory planning) with v3 (deterministic layout + header/body/footer structure).

## Pipeline Overview

```
Input: config.json + safetensors header
           │
           ▼
┌─────────────────────────────────────┐
│   1. Template Selection             │  See ADR-0002
│   "qwen2" → templates/qwen2.yaml    │
└─────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│   2. Graph IR (High-level)          │  See ADR-0001
│   Ops: Embed, RMSNorm, GEMM, RoPE,  │
│        Attention, SwiGLU, Add       │
└─────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│   3. Lowering Phase                 │  See ADR-0003, ADR-0005
│   • Resolve modes (prefill/decode)  │
│   • Select kernel variants          │
│   • Allocate mode-specific buffers  │
└─────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│   4. Layout Phase                   │  See ADR-0004
│   • Compute deterministic offsets   │
│   • 64-byte align for SIMD          │
│   • Insert canary markers           │
└─────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│   5. Code Generation                │
│   • Emit C from lowered IR + layout │
│   • Header: magic, canary offsets   │
│   • Body: weight loading, kernels   │
│   • Footer: cleanup                 │
└─────────────────────────────────────┘
           │
           ▼
Output Files
```

## File Outputs

| File | Purpose |
|------|---------|
| `model_name.c` | Generated inference code with real kernel calls |
| `model_name.h` | Public API: init, forward, cleanup functions |
| `memory_layout.json` | All tensor offsets for debugging and visualization |
| `graph.json` | Graph IR representation for tooling |

## ADR References

| ADR | Title | Summary |
|-----|-------|---------|
| [0001](adr/0001-ir-v4-pipeline.md) | IR v4 Pipeline Split | Graph IR → Lowered IR → Layout → Codegen |
| [0002](adr/0002-templates-as-canonical.md) | Templates as Canonical | YAML templates define architecture |
| [0003](adr/0003-mode-specific-lowering.md) | Mode-Specific Lowering | Separate prefill/decode/backward |
| [0004](adr/0004-deterministic-layout-canaries.md) | Deterministic Layout | Fixed offsets + canary markers |
| [0005](adr/0005-kernel-selection-in-lowering.md) | Kernel Selection | Lowering picks concrete kernels |
| [0006](adr/0006-weight-mapping-via-template.md) | Weight Mapping | HF names → internal names via template |

## Key Design Principles

### 1. Separation of Concerns (ADR-0001)
- **Graph IR**: Architecture-agnostic, human-readable
- **Lowered IR**: Mode-specific, kernel-bound
- **Layout**: Memory offsets and alignment
- **Codegen**: C emission from resolved IR

### 2. Templates are Canonical (ADR-0002)
```yaml
# templates/qwen2.yaml
name: qwen2
config_mapping:
  hidden_size: embed_dim
  num_attention_heads: num_heads

layers:
  - name: embed
    op: Embed
    input: token_ids
    output: hidden_states
```

### 3. Mode-Specific Buffers (ADR-0003)
```
┌─────────────────────────────────────┐
│ Weights (shared, read-only)         │
├─────────────────────────────────────┤
│ KV Cache (persistent)               │
├─────────────────────────────────────┤
│ Prefill Buffers [max_seq, dim]      │
├─────────────────────────────────────┤
│ Decode Buffers [1, dim]             │
└─────────────────────────────────────┘
```

### 4. Deterministic Layout (ADR-0004)
- All offsets computed at codegen time
- 64-byte alignment for AVX-512
- Canary markers between tensors (debug builds)
- Exported as `memory_layout.json`

### 5. Kernel Selection in Lowering (ADR-0005)
```python
# Graph IR (abstract)
{"op": "GEMM", "inputs": ["x", "w"], "output": "y"}

# Prefill lowered (parallel GEMM)
{"kernel": "gemm_blocked_parallel_bf16", "args": [...]}

# Decode lowered (1x1 GEMM)
{"kernel": "gemm_1x1_bf16", "args": [...]}
```

### 6. Weight Mapping (ADR-0006)
```yaml
weight_mapping:
  "model.layers.{i}.self_attn.q_proj.weight": "layers.{i}.attn.wq"
  "model.layers.{i}.mlp.gate_proj.weight": "layers.{i}.mlp.gate"
```

## Example: Qwen2-0.5B

```bash
# Generate inference code from HuggingFace model
python scripts/build_ir_v4.py Qwen/Qwen2-0.5B-Instruct --prefix=build/qwen2

# Outputs:
# build/qwen2/qwen2_0_5b.c       - Inference code
# build/qwen2/qwen2_0_5b.h       - API header
# build/qwen2/memory_layout.json - Layout for debugging
# build/qwen2/graph.json         - Graph IR (optional)
```

## Generated Code Structure

```c
// qwen2_0_5b.c

/* === HEADER === */
typedef struct { ... } MagicHeader;
static const size_t TOTAL_BYTES = 1073741824;
#define OFFSET_EMBED_WEIGHT 64
#define OFFSET_LAYER_0_WQ   272629888
// ...

/* === BODY === */
void qwen2_prefill_layer_0(Model *m, int num_tokens) {
    // Pre-attention RMSNorm
    rmsnorm_forward_bf16(...);

    // QKV projections (parallel GEMM)
    gemm_blocked_parallel_bf16(...);

    // RoPE
    rope_forward_qk_bf16(...);

    // Attention
    attention_forward_causal_head_major_gqa_bf16(...);

    // Output projection
    gemm_blocked_parallel_bf16(...);

    // Residual
    add_forward_bf16(...);

    // Post-attention RMSNorm + MLP + Residual
    // ...
}

void qwen2_decode_layer_0(Model *m) {
    // Same ops but with decode kernels
    gemm_1x1_bf16(...);  // M=1 specialized
    attention_decode_single_query_bf16(...);
}

/* === FOOTER === */
void qwen2_cleanup(Model *m) { ... }
```

## Implementation Status

- [x] Native AVX-512 BF16 support in bf16_utils.h
- [x] Add kernel for residual connections (add_kernels_bf16.c)
- [x] ADR documentation complete
- [ ] Template YAML format specification
- [ ] Template → Graph IR compiler
- [ ] Lowering phase implementation
- [ ] Kernel registry and validation
- [ ] Update build_ir_v3.py → build_ir_v4.py
