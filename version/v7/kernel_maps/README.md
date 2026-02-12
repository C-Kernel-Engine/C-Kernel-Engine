# v7 Kernel Maps

This directory is the single source of truth for kernel metadata in v7.
Each JSON file defines one concrete kernel variant (including quantization).

## Kernel Registry Overview

See **KERNEL_REGISTRY.json** for the complete categorized inventory of all 230+ kernels in `src/kernels/`.

| Category | Count | Description |
|----------|-------|-------------|
| Inference | 85 | Forward pass kernels (GEMM, attention, normalization, activation) |
| Training | 45 | Backward pass kernels for gradients |
| Optimizer | 14 | Adam, SGD, gradient utilities |
| Fusion | 25 | Multi-operation fused kernels (mega_fused_*, fused_*) |
| Quantization | 28 | Quantize, dequantize, vec_dot, FP16 conversion |
| Utility | 12 | AXPY, add, initialization |

### Key Fusion Kernels (Performance Critical)

| Kernel | Speedup | Operations Fused |
|--------|---------|------------------|
| `mega_fused_attention_prefill` | 1.45x | RMSNorm → QKV → RoPE → Attention → OutProj → Residual |
| `mega_fused_outproj_mlp_prefill` | 1.1x | Quantize → AttnProj → Residual → RMSNorm → MLP → Residual |

The kernel map is used by:
- IR lowering (op -> kernel selection)
- Fusion (pattern matching via "fuses")
- Memory planning (inputs/outputs/scratch buffers)
- Codegen (argument order and required sources)
- Validation and tests

## Design

- One file per kernel variant (recommended for explicit IO and quantization).
- File name should match `id` (C function name) where possible.
- Avoid embedding logic in JSON; use explicit fields per variant.
- Python tooling may generate JSON, but JSON is the source of truth.
- ISA/compile flags belong in `impl.variants`; avoid putting CPU features in
  `constraints.requires`.

## Single Source Of Truth

- Kernel maps (JSON) are the source of truth.
- Registry is generated from maps, not from code (`gen_kernel_registry_from_maps.py`).
- Source scan exists only to verify maps match real C functions
  (`gen_kernel_registry.py`, `check_kernel_map_sync.py`).

## Canonical Workflow

1) Write kernel code.
2) Write unit/parity tests (numpy/llama.cpp/pytorch).
3) Write kernel map JSON (inputs/outputs/scratch/quant/variants/tests).
4) Run validation (`validate_kernel_maps.py`, `check_kernel_map_sync.py`).
5) Generate registry from maps (`gen_kernel_registry_from_maps.py`).
6) IR + fusion + codegen consume registry + maps.

## Manual Step

Kernel map JSON is still manual; templates can be generated from function names,
but IO shapes/quant/tests need human confirmation.

## Useful Scripts

Inventory and validation:
- `python3 version/v7/scripts/gen_kernel_registry.py`  
  Source scan of `src/kernels/**/*.c` → `KERNEL_SOURCES.json` (counts + inventory).
- `python3 version/v7/scripts/gen_kernel_registry_from_maps.py`  
  Kernel maps → `KERNEL_REGISTRY.json` (what IR/codegen will consume).
- `python3 version/v7/scripts/check_kernel_map_sync.py`  
  Compare maps vs source functions (missing maps or missing symbols).
- `python3 version/v7/scripts/validate_kernel_maps.py`  
  Validate map schema and basic sanity rules.

## Required Fields

```json
{
  "id": "gemm_nt_q5_0_q8_0",
  "op": "gemm",
  "variant": "q5_0_w_q8_0_a",
  "modes": {"inference": true, "training": false, "backward": false},
  "quant": {
    "weight": "q5_0",
    "activation": "q8_0",
    "output": "fp32"
  },
  "inputs": [
    {"name": "A", "dtype": "q8_0", "shape": ["M", "K"]},
    {"name": "B", "dtype": "q5_0", "shape": ["N", "K"]},
    {"name": "bias", "dtype": "fp32", "shape": ["N"], "optional": true}
  ],
  "outputs": [
    {"name": "C", "dtype": "fp32", "shape": ["M", "N"]}
  ],
  "parallelization": {
    "supported": ["token", "feature"],
    "preferred": {"prefill": "token", "decode": "feature"},
    "strategies": [
      {
        "name": "token",
        "partition_dim": "M",
        "param_style": "range",
        "range": {"min_chunk": 1},
        "notes": "Split by rows/tokens; good for prefill."
      },
      {
        "name": "feature",
        "partition_dim": "N",
        "param_style": "range",
        "range": {"min_chunk": 16, "align_bytes": 64, "allow_tail": true},
        "notes": "Split by output features; good for decode."
      }
    ]
  },
  "scratch": [],
  "dims": ["M", "N", "K"],
  "constraints": {
    "alignment": {"K": 32}
  },
  "impl": {
    "function": "gemm_nt_q5_0_q8_0",
    "sources": ["src/kernels/gemm_kernels_q5_0.c"],
    "variants": [
      {"name": "avx512", "requires": ["avx512f", "fma"], "compile_flags": ["-mavx512f", "-mfma"]},
      {"name": "avx2", "requires": ["avx2", "fma"], "compile_flags": ["-mavx2", "-mfma"]},
      {"name": "avx", "requires": ["avx"], "compile_flags": ["-mavx"]},
      {"name": "sse4_1", "requires": ["sse4_1"], "compile_flags": ["-msse4.1"]},
      {"name": "ref", "requires": [], "compile_flags": []}
    ]
  },
  "tests": {
    "unit": ["unittest/test_gemm_q5_0_q8_0.py"],
    "bench": ["scripts/bench_gemm_q5_0_q8_0.py"],
    "parity": [
      {
        "kind": "llamacpp",
        "command": "make llamacpp-parity-full",
        "tolerance": "1e-5",
        "notes": "Compare logits/activations vs llama.cpp reference path."
      }
    ]
  }
}
```

## Training / Inference Metadata

Use `modes` to declare where a kernel is valid. Use `condition` and
`save_for_backward` on buffers to model training-only activations.

```json
{
  "id": "rmsnorm_forward",
  "op": "rmsnorm",
  "variant": "fp32",
  "modes": {"inference": true, "training": true, "backward": false},
  "inputs": [
    {"name": "input", "dtype": "fp32", "shape": ["T", "E"]},
    {"name": "gamma", "dtype": "fp32", "shape": ["E"]}
  ],
  "outputs": [
    {"name": "output", "dtype": "fp32", "shape": ["T", "E"]},
    {"name": "rstd", "dtype": "fp32", "shape": ["T"], "condition": "training_only", "save_for_backward": true}
  ],
  "scratch": [
    {"name": "mean", "dtype": "fp32", "shape": ["T"], "condition": "training_only"}
  ],
  "impl": {
    "function": "rmsnorm_forward",
    "sources": ["src/kernels/rmsnorm_kernels.c"]
  }
}
```

Guidance:
- If a kernel always writes an activation (even in inference), omit `condition`.
- If a kernel can skip storing activations in inference, mark them with
  `"condition": "training_only"`.
- `save_for_backward: true` means the memory planner must keep the buffer
  alive until the backward pass.

## Fused Kernels

Fused kernels declare which base kernels they replace:

```json
{
  "id": "mega_fused_attention_prefill",
  "op": "fused_attention_block",
  "variant": "q5_0_w_q8_0_a",
  "quant": {"weight": "q5_0|q8_0", "activation": "fp32", "output": "fp32"},
  "inputs": [ ... ],
  "outputs": [ ... ],
  "scratch": [ ... ],
  "dims": ["T", "E", "AE", "H", "KV", "D", "AD", "max_T"],
  "fuses": [
    "rmsnorm_forward",
    "gemm_nt_q5_0",
    "attention_forward_causal_head_major_gqa_flash_strided",
    "ck_residual_add_token_major"
  ],
  "impl": {
    "function": "mega_fused_attention_prefill",
    "sources": ["src/kernels/fused/mega_fused_attention_prefill.c"],
    "variants": [
      {"name": "default", "requires": [], "compile_flags": []}
    ]
  },
  "tests": {
    "unit": ["unittest/test_mega_fused_attention_prefill.py"],
    "bench": ["scripts/bench_mega_fused_attention_prefill.py"],
    "parity": [
      {
        "kind": "llamacpp",
        "command": "make fusion-test-full-with-lamacpp",
        "tolerance": "1e-5",
        "notes": "Parity check for fused attention vs llama.cpp reference."
      }
    ]
  }
}
```

## Dimension Symbols

- T: tokens (prefill seq_len; decode uses T=1)
- E: embed_dim
- AE: aligned_embed_dim
- H: num_heads
- KV: num_kv_heads
- D: head_dim
- AD: aligned_head_dim
- I: intermediate_dim
- AI: aligned_intermediate_dim
- max_T: max_context_len
- V: vocab_size

## Kernel Map Files

Each JSON file in this directory defines one concrete kernel variant. The following kernel maps are defined:

### Fusion Kernels (Multi-Operation)

| File | ID | Operation | Replaces |
|------|-----|-----------|----------|
| `mega_fused_attention_prefill.json` | `mega_fused_attention_prefill` | fused_attention_block | rmsnorm, gemv_q5_0, gemv_q8_0, attention, residual_add |
| `mega_fused_outproj_mlp_prefill.json` | `mega_fused_outproj_mlp_prefill` | fused_mlp_block | quantize, attn_proj, residual_add, rmsnorm, 3×gemm, swiglu |

### Single-Operation Kernels

| File | ID | Operation | Quantization |
|------|-----|-----------|--------------|
| `rmsnorm_fp32.json` | `rmsnorm_forward` | rmsnorm | fp32 → fp32 |
| `rope_forward_qk.json` | `rope_forward_qk` | rope | fp32 → fp32 |
| `swiglu_forward.json` | `swiglu_forward` | swiglu | fp32 + fp32 → fp32 |
| `ck_qkv_project_head_major.json` | `ck_qkv_project_head_major` | qkv_projection | fp32 → fp32 |
| `ck_attention_project_head_major.json` | `ck_attention_project_head_major` | attention_projection | fp32 → fp32 |
| `gemm_blocked_serial.json` | `gemm_blocked_serial` | gemm | fp32×fp32 → fp32 |
| `gemv_q5_0_q8_0.json` | `gemv_q5_0_q8_0` | gemv | Q5_0 × Q8_0 → FP32 |
| `gemv_q8_0_q8_0.json` | `gemv_q8_0_q8_0` | gemv | Q8_0 × Q8_0 → FP32 |
| `attention_forward_causal_head_major_gqa_flash_strided.json` | `attention_forward_causal_head_major_gqa_flash_strided` | attention | FP32 → FP32 |
| `ck_residual_add_token_major.json` | `ck_residual_add_token_major` | add | FP32 + FP32 → FP32 |

### Kernel Map Coverage

| Transformer Component | Kernels Defined |
|----------------------|-----------------|
| Attention Block | mega_fused_attention_prefill, attention, qkv_project, attn_proj |
| MLP Block | mega_fused_outproj_mlp_prefill, swiglu, gemm_blocked_serial |
| Normalization | rmsnorm, rope |
| Quantized GEMV | gemv_q5_0_q8_0, gemv_q8_0_q8_0 |
| Utilities | residual_add |

**Coverage Status:** Complete for v7 transformer block

### Adding New Kernels

To add a new kernel:

1. Create a JSON file named `<kernel_function>.json`
2. Fill in all required fields (id, op, variant, quant, inputs, outputs, dims, impl)
3. Add to `tests.unit` and `tests.bench` paths
4. Run `scripts/gen_kernel_specs.py` to regenerate `ckernel_kernel_specs.c`

### Dimension Symbols

- T: tokens (prefill seq_len; decode uses T=1)
- E: embed_dim
- AE: aligned_embed_dim
- H: num_heads
- KV: num_kv_heads
- D: head_dim
- AD: aligned_head_dim
- I: intermediate_dim
- AI: aligned_intermediate_dim
- max_T: max_context_len
- V: vocab_size

## Example: Creating a New Kernel

```json
{
  "id": "your_kernel_function",
  "op": "your_operation_type",
  "variant": "variant_description",
  "quant": {
    "weight": "q4_0",
    "activation": "fp32",
    "output": "fp32"
  },
  "inputs": [
    {"name": "input_name", "dtype": "fp32", "shape": ["T", "E"], "desc": "Description"}
  ],
  "outputs": [
    {"name": "output_name", "dtype": "fp32", "shape": ["T", "E"], "desc": "Description"}
  ],
  "scratch": [],
  "dims": ["T", "E"],
  "parallelization": {
    "supported": ["feature"],
    "preferred": {"prefill": "feature", "decode": "feature"},
    "strategies": [...]
  },
  "constraints": {
    "alignment": {"E": 64}
  },
  "impl": {
    "function": "your_kernel_function",
    "sources": ["src/kernels/your_kernel.c"],
    "variants": [
      {"name": "default", "requires": [], "compile_flags": []}
    ]
  },
  "tests": {
    "unit": ["unittest/test_your_kernel.py"],
    "bench": ["scripts/bench_your_kernel.py"]
  }
}
```
