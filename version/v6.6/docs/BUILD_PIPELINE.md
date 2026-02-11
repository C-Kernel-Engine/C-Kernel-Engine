# v6.6 Build Pipeline

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           v6.6 BUILD PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   GGUF /     │    │   Kernel     │    │   Fusion     │                   │
│  │ SafeTensors  │    │   Registry   │    │   Patterns   │                   │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                   │
│         │                   │                   │                            │
│         ▼                   ▼                   ▼                            │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                    1. VALIDATE MODEL                         │            │
│  │   - Check all required kernels exist in registry            │            │
│  │   - Map model quant types to kernel variants                │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                    2. BUILD IR                               │            │
│  │   - Create IR node for each layer operation                 │            │
│  │   - Sequence: embed → [layer × N] → final_norm → lm_head   │            │
│  │   - Each layer: attn_norm → QKV → RoPE → Attn → OutProj    │            │
│  │                  → residual → mlp_norm → MLP → residual     │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                    3. FUSION PASS                            │            │
│  │   - Scan IR for fusable sequences                           │            │
│  │   - Match patterns: rmsnorm→QKV→RoPE→Attn→OutProj→residual │            │
│  │   - Replace matched sequences with fused kernel nodes       │            │
│  │   - Track replacements for debugging                        │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                    4. LOWER IR                               │            │
│  │   - Resolve kernel variants (AVX2/AVX512/VNNI)             │            │
│  │   - Bind concrete function pointers                         │            │
│  │   - IR nodes now reference actual C functions               │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                    5. MEMORY PLANNING                        │            │
│  │   - Calculate buffer sizes from fused IR                    │            │
│  │   - KV cache: persistent per-layer                          │            │
│  │   - Scratch: shared across layers (max of all)             │            │
│  │   - Activations: double-buffer for prefill pipeline         │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                    6. CODEGEN                                │            │
│  │   - memory_layout.c: buffer offsets, sizes, allocation      │            │
│  │   - prefill.c: prefill path (T tokens)                      │            │
│  │   - decode.c: decode path (1 token)                         │            │
│  │   - inference.c: combined or dispatcher                     │            │
│  │   - main.c: CLI, weight loading, chat templates             │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                    7. COMPILE                                │            │
│  │   - Compile generated C with appropriate SIMD flags         │            │
│  │   - Link with kernel library                                │            │
│  │   - Output: model-specific inference binary                 │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Pipeline Scripts

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 0 | `gen_kernel_registry_from_maps.py` | `kernel_maps/*.json` | `KERNEL_REGISTRY.json` |
| 0b | `gen_kernel_registry.py` | `src/kernels/*.c` | `KERNEL_SOURCES.json` |
| 1 | `validate_model.py` | model + registry | validation report |
| 2 | `build_ir_v6_6.py` | model config | `ir.json` |
| 3 | `fusion_pass.py` | `ir.json` + patterns | `ir_fused.json` |
| 4 | `lower_ir.py` | `ir_fused.json` | `ir_lowered.json` |
| 5 | `memory_planner.py` | `ir_lowered.json` | `memory_layout.json` |
| 6 | `codegen_v6_6.py` | all above | `generated/*.c` |
| 7 | `make` / `cmake` | `generated/*.c` | binary |

## 1. Kernel Registry Generation (from kernel maps)

```bash
python3 version/v6.6/scripts/gen_kernel_registry_from_maps.py
```

Scans `version/v6.6/kernel_maps/` and generates `KERNEL_REGISTRY.json`:

```json
{
  "_meta": {"version": "v6.6", "counts": {...}},
  "kernels": [
    {"name": "gemv_q5_0_q8_0", "...": "..."},
    {"name": "mega_fused_attention_prefill", "...": "..."}
  ]
}
```

Optional source scan (sync check):

```bash
python3 version/v6.6/scripts/gen_kernel_registry.py
```

## 2. Model Validation

Before building IR, validate model has all required kernels:

```python
def validate_model(model_path: str, registry: dict) -> List[str]:
    """
    Check model quant types have matching kernels.
    Returns list of missing kernels.
    """
    model = load_model_metadata(model_path)  # GGUF or safetensors

    required = []
    for layer in model.layers:
        # Attention weights: Q5_0 → need gemv_q5_0_q8_0
        if layer.wq.dtype == "q5_0":
            required.append("gemv_q5_0_q8_0")
        # V weights: Q8_0 → need gemv_q8_0_q8_0
        if layer.wv.dtype == "q8_0":
            required.append("gemv_q8_0_q8_0")
        # etc.

    missing = [k for k in required if k not in registry]
    return missing
```

## 3. Fusion Pass - Sequence Matching

The fusion pass scans IR for sequences that match fusion patterns:

```python
FUSION_PATTERNS = {
    "mega_fused_attention_prefill": {
        "sequence": [
            "rmsnorm_forward",
            "gemv_q5_0_q8_0",  # Q proj
            "gemv_q5_0_q8_0",  # K proj
            "gemv_q8_0_q8_0",  # V proj
            "rope_forward_qk",
            "attention_forward_causal_head_major_gqa_flash_strided",
            "gemv_q5_0_fp32",  # Out proj
            "residual_add",
        ],
        "starts_with": "rmsnorm_forward",
        "speedup": 1.45,
    },
    "mega_fused_outproj_mlp_prefill": {
        "sequence": [
            "quantize_row_q8_0",  # quantize attention output
            "gemv_q5_0_q8_0",     # out_proj
            "residual_add",
            "rmsnorm_forward",
            "gemv_q4_k_q8_k",     # gate
            "gemv_q4_k_q8_k",     # up
            "swiglu_forward",
            "gemv_q4_k_q8_k",     # down
            "residual_add",
        ],
        "starts_with": "quantize_row_q8_0",
        "speedup": 1.1,
    },
}

def find_fusion_matches(ir_nodes: List[IRNode]) -> List[FusionMatch]:
    """Find all fusable sequences in IR."""
    matches = []

    for i, node in enumerate(ir_nodes):
        for fusion_name, pattern in FUSION_PATTERNS.items():
            if node.kernel == pattern["starts_with"]:
                # Check if full sequence matches
                seq_len = len(pattern["sequence"])
                if i + seq_len <= len(ir_nodes):
                    actual = [ir_nodes[i+j].kernel for j in range(seq_len)]
                    if actual == pattern["sequence"]:
                        matches.append(FusionMatch(
                            start_idx=i,
                            end_idx=i + seq_len,
                            fusion_kernel=fusion_name,
                            replaced=actual,
                        ))
    return matches
```

### Fusion Tracking (for debugging)

```json
{
  "layer_0": {
    "attention_block": {
      "fused_to": "mega_fused_attention_prefill",
      "replaced": [
        "rmsnorm_forward",
        "gemv_q5_0_q8_0",
        "gemv_q5_0_q8_0",
        "gemv_q8_0_q8_0",
        "rope_forward_qk",
        "attention_forward_causal_head_major_gqa_flash_strided",
        "gemv_q5_0_fp32",
        "residual_add"
      ],
      "speedup": "1.45x"
    }
  }
}
```

## 4. Memory Planning (Post-Fusion)

Memory layout is calculated AFTER fusion, not before:

```python
def plan_memory(fused_ir: List[IRNode], model_config: dict) -> MemoryLayout:
    """
    Calculate buffer sizes from fused IR.
    Must be called AFTER fusion pass.
    """
    layout = MemoryLayout()

    # KV cache: persistent, per-layer
    for layer_idx in range(model_config.num_layers):
        layout.kv_cache_k[layer_idx] = allocate(
            shape=[num_kv_heads, max_context, head_dim],
            dtype="fp32"
        )
        layout.kv_cache_v[layer_idx] = allocate(...)

    # Scratch: shared across layers, take max
    max_scratch = 0
    for node in fused_ir:
        if node.kernel in FUSED_KERNELS:
            scratch_size = get_scratch_size(node.kernel, model_config)
            max_scratch = max(max_scratch, scratch_size)
    layout.scratch = allocate(size=max_scratch)

    # Activations: double buffer for pipelining
    layout.act_buf_0 = allocate(shape=[max_tokens, aligned_embed_dim])
    layout.act_buf_1 = allocate(shape=[max_tokens, aligned_embed_dim])

    return layout
```

## 5. Codegen Output

### memory_layout.c
```c
// Auto-generated by codegen_v6_6.py
#include "memory_layout.h"

const CKMemoryLayout MEMORY_LAYOUT = {
    .total_bytes = 2147483648,  // 2GB
    .kv_cache_offset = 0,
    .kv_cache_size = 1073741824,
    .scratch_offset = 1073741824,
    .scratch_size = 268435456,
    .act_buf_0_offset = 1342177280,
    .act_buf_1_offset = 1476395008,
    // ...
};

void ck_allocate_memory(CKRuntime *rt) {
    rt->memory = aligned_alloc(64, MEMORY_LAYOUT.total_bytes);
    rt->kv_cache = rt->memory + MEMORY_LAYOUT.kv_cache_offset;
    rt->scratch = rt->memory + MEMORY_LAYOUT.scratch_offset;
    // ...
}
```

### prefill.c
```c
// Auto-generated prefill path
void ck_prefill(CKRuntime *rt, const int *tokens, int num_tokens, int start_pos) {
    float *act = rt->act_buf_0;
    float *act_next = rt->act_buf_1;

    // Embedding
    embedding_forward(tokens, rt->weights.embed, act, num_tokens, rt->config.embed_dim);

    // Layers (fused)
    for (int layer = 0; layer < rt->config.num_layers; layer++) {
        CKLayerWeights *w = &rt->weights.layers[layer];

        // Fused attention block (replaces 8 separate kernels)
        mega_fused_attention_prefill(
            act,                    // input
            act,                    // residual (same as input)
            w->attn_norm_gamma,
            w->wq, w->bq,
            w->wk, w->bk,
            w->wv, w->bv,
            w->wo, w->bo,
            rt->kv_cache_k[layer],
            rt->kv_cache_v[layer],
            rt->rope_cos,
            rt->rope_sin,
            act_next,               // output
            rt->scratch,
            start_pos, num_tokens, rt->config.max_context,
            rt->config.eps
        );

        // Swap buffers
        float *tmp = act; act = act_next; act_next = tmp;

        // Fused MLP block
        mega_fused_outproj_mlp_prefill(...);

        // Swap buffers
        tmp = act; act = act_next; act_next = tmp;
    }

    // Final norm + LM head
    rmsnorm_forward(act, rt->weights.final_norm_gamma, act, ...);
    gemv_q5_0_q8_0(act, rt->weights.lm_head, rt->logits, ...);
}
```

### main.c
```c
// Auto-generated CLI
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "memory_layout.h"
#include "inference.h"

typedef struct {
    char *model_path;
    char *prompt;
    int max_tokens;
    float temperature;
    float top_p;
    int top_k;
    char *system_prompt;
    int num_threads;
} CLIArgs;

// Chat template handling
const char *CHAT_TEMPLATES[] = {
    [TEMPLATE_LLAMA3] = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n%s<|eot_id|>"
                        "<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|>"
                        "<|start_header_id|>assistant<|end_header_id|>\n\n",
    [TEMPLATE_QWEN2]  = "<|im_start|>system\n%s<|im_end|>\n"
                        "<|im_start|>user\n%s<|im_end|>\n"
                        "<|im_start|>assistant\n",
    [TEMPLATE_CHATML] = "<|system|>\n%s</s>\n<|user|>\n%s</s>\n<|assistant|>\n",
};

// Special tokens
typedef struct {
    int bos_token;
    int eos_token;
    int pad_token;
    int *stop_tokens;
    int num_stop_tokens;
} TokenConfig;

int main(int argc, char **argv) {
    CLIArgs args = parse_args(argc, argv);

    // Load model
    CKRuntime *rt = ck_runtime_create();
    ck_allocate_memory(rt);
    ck_load_weights(rt, args.model_path);

    // Tokenize with chat template
    int *tokens = tokenize_with_template(args.prompt, args.system_prompt, rt->template);
    int num_tokens = count_tokens(tokens);

    // Prefill
    ck_prefill(rt, tokens, num_tokens, 0);

    // Decode loop
    int pos = num_tokens;
    while (pos < args.max_tokens) {
        int next_token = ck_decode_step(rt, pos);

        if (is_stop_token(next_token, rt->token_config)) {
            break;
        }

        // Handle special output (e.g., <think> tags)
        if (next_token == rt->token_config.think_start) {
            printf("[thinking...]\n");
        }

        printf("%s", rt->tokenizer.decode(next_token));
        fflush(stdout);
        pos++;
    }

    ck_runtime_free(rt);
    return 0;
}
```

## Directory Structure After Build

```
version/v6.6/
├── scripts/
│   ├── gen_kernel_registry_from_maps.py  # Step 0: build registry from maps
│   ├── gen_kernel_registry.py    # Step 0b: scan sources (sync check)
│   ├── validate_model.py         # Step 1: check model
│   ├── build_ir_v6_6.py          # Step 2: build IR
│   ├── fusion_pass.py            # Step 3: fuse operations
│   ├── lower_ir.py               # Step 4: resolve kernels
│   ├── memory_planner.py         # Step 5: plan memory
│   └── codegen_v6_6.py           # Step 6: generate C
├── kernel_maps/
│   ├── KERNEL_REGISTRY.json      # Generated kernel catalog
│   ├── KERNEL_SOURCES.json       # Source scan (optional)
│   ├── mega_fused_attention_prefill.json
│   └── ...
├── src/
│   └── generated/
│       ├── memory_layout.c       # Buffer allocation
│       ├── prefill.c             # Prefill path
│       ├── decode.c              # Decode path
│       ├── inference.c           # Combined
│       └── main.c                # CLI entry point
└── docs/
    └── BUILD_PIPELINE.md         # This document
```

## Running the Pipeline

```bash
# 1. Generate kernel registry from maps (once, or when maps change)
python3 version/v6.6/scripts/gen_kernel_registry_from_maps.py

# Optional: scan sources for sync checks
python3 version/v6.6/scripts/gen_kernel_registry.py

# 2. Full pipeline for a model
python3 version/v6.6/scripts/build_model.py \
    --model /path/to/model.gguf \
    --output version/v6.6/src/generated/ \
    --max-context 4096 \
    --threads 8

# 3. Compile
cd version/v6.6
make

# 4. Run
./ck-cli-v6.6 --model /path/to/model.gguf --prompt "Hello, world!"
```

## Fusion Debug Mode

```bash
# Show what would be fused without applying
python3 version/v6.6/scripts/fusion_pass.py --ir ir.json --dry-run

# Output:
# [fusion] layer 0 attention block:
#   WOULD FUSE: rmsnorm_forward → gemv_q5_0_q8_0 → ... (8 ops)
#   INTO: mega_fused_attention_prefill
#   SPEEDUP: 1.45x
#
# [fusion] layer 0 mlp block:
#   WOULD FUSE: quantize_row_q8_0 → gemv_q5_0_q8_0 → ... (9 ops)
#   INTO: mega_fused_outproj_mlp_prefill
#   SPEEDUP: 1.1x
```

## RoPE Scaling and rotary_dim Support

v6.6 supports RoPE scaling for extended context models and partial rotary dimensions.

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rotary_dim` | int | `head_dim` | Number of dimensions to rotate (subset of head_dim) |
| `rope_scaling_type` | string | `"none"` | Scaling type: `"none"`, `"linear"`, `"dynamic"`, `"yarn"` |
| `rope_scaling_factor` | float | `1.0` | Scaling factor (1.0 = no scaling) |

### GGUF Metadata Keys

RoPE parameters are extracted from GGUF metadata during conversion:

| GGUF Key | v6.6 Config | Notes |
|----------|------------|-------|
| `llama.rope.dim` | `rotary_dim` | Standard Llama key |
| `attention.rotary_dim` | `rotary_dim` | Alternative key |
| `qwen2.rotary_dim` | `rotary_dim` | Qwen-specific |
| `gemma.attention.key_length` | `rotary_dim` | Gemma uses key_length |
| `llama.rope.scaling.type` | `rope_scaling_type` | Scaling type |
| `rope.scaling.type` | `rope_scaling_type` | Generic key |
| `llama.rope.scaling.factor` | `rope_scaling_factor` | Scaling factor |

### Scaling Types

1. **`"none"`**: Standard RoPE, no scaling
2. **`"linear"`**: Scale positions by `1/scaling_factor` (extends context)
3. **`"dynamic"`**: NTK-aware dynamic scaling
4. **`"yarn"`**: YaRN (Yet another RoPE extensioN) scaling

### Memory Layout

```
RoPE Cache: [2, max_seq_len, rotary_dim/2]
  - cos_cache: [max_seq_len, rotary_dim/2]
  - sin_cache: [max_seq_len, rotary_dim/2]

# Codegen defines
#define ROTARY_DIM <from config>
#define ROPE_CACHE_SIZE (2 * MAX_SEQ_LEN * ROTARY_DIM / 2 * sizeof(float))
```

### Partial RoPE (rotary_dim < head_dim)

When `rotary_dim < head_dim`:
- Only the first `rotary_dim` channels are rotated
- Channels `[rotary_dim, head_dim)` pass through unchanged
- Cache is sized for `rotary_dim/2`, not `head_dim/2`

### Example: Llama 3.1 128K Context

```python
# From GGUF metadata:
rope_theta = 500000.0
rotary_dim = 128  # head_dim is 4096/32=128 for Llama 3.1
rope_scaling_type = "linear"
rope_scaling_factor = 16.0

# Generated defines:
#define HEAD_DIM 128
#define ROTARY_DIM 128
#define ROPE_SCALING linear
#define ROPE_SCALING_FACTOR 16.0f

# Cache size: 2 * 131072 * 64 * 4 = 67 MB
```

### Example: Gemma with Partial Rotation

```python
# Gemma 3 uses key_length for rotary_dim:
rotary_dim = 96  # attention.key_length
head_dim = 96     # hidden_size / num_heads

# Generated:
#define HEAD_DIM 96
#define ROTARY_DIM 96
#define ROPE_SCALING none
#define ROPE_SCALING_FACTOR 1.0f
```

