# v6.6 Build Pipeline

```
┌─────────────────┐
│ Model Input     │
│ (GGUF/Safetensor│
│  HuggingFace)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ 1. GEN_REGISTRY │────▶│ KERNEL_REGISTRY │
│ Scan src/kernels│     │ All 230+ kernels│
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ 2. CHECK_MODEL  │────▶│ MISSING_KERNELS │
│ Match model ops │     │ Error if gaps   │
│ to registry     │     └─────────────────┘
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. BUILD_IR     │
│ Per-layer IR    │
│ (ops + buffers) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ 4. FUSION_PASS  │────▶│ FUSION_LOG      │
│ Pattern match   │     │ What replaced   │
│ Track replaces  │     │ For debugging   │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│ 5. LOWER_IR     │
│ Replace with    │
│ fused kernels   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ 6. MEM_LAYOUT   │────▶│ memory_layout.c │
│ Calc allocs     │     │ Scratch sizes   │
│ Scratch sizes   │     │ Tensor strides  │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ 7. CODEGEN      │────▶│ prefill.c       │
│ Per phase       │     │ decode.c        │
│                 │     │ or inference.c  │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│ 8. GENERIC_MAIN │
│ CLI args        │
│ Memory alloc    │
│ Weight load     │
│ Template handling│
│ (thinking, etc) │
└─────────────────┘
```

## Fusion Patterns

```python
# Patterns to detect (sequence matters)
FUSION_PATTERNS = [
    {
        "name": "mega_attention_prefill",
        "ops": ["rmsnorm", "qkv", "rope", "attention", "outproj", "residual_add"],
        "replaces": ["rmsnorm_forward", "gemv_*", "rope_forward_*", "attention_*", "ck_residual_add_*"]
    },
    {
        "name": "mega_mlp_prefill",
        "ops": ["quantize", "attn_proj", "residual", "rmsnorm", "gemm", "swiglu", "residual"],
        "replaces": ["quantize_*", "ck_attention_project_*", "ck_residual_*", "rmsnorm_*", "gemm_*", "swiglu_*"]
    },
    # More patterns...
]
```

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/gen_kernel_registry_from_maps.py` | Kernel maps → `KERNEL_REGISTRY.json` |
| `scripts/gen_kernel_registry.py` | Scan `src/kernels/**/*.c` → `KERNEL_SOURCES.json` |
| `scripts/check_model_coverage.py` | Model → required kernels → missing check |
| `scripts/build_ir.py` | Model → layer IR |
| `scripts/fusion_pass.py` | IR → fused IR + fusion_log.json |
| `scripts/lower_ir.py` | Fused IR → final IR |
| `scripts/gen_memory_layout.py` | IR → memory_layout.c |
| `scripts/codegen.py` | IR → prefill.c/decode.c |

## Output Artifacts

```
build/
├── kernel_registry.json      # From kernel maps
├── kernel_sources.json       # Optional source scan (sync check)
├── model_coverage.json       # From step 2
├── ir/                       # IR files
│   ├── layer_00.json
│   ├── layer_01.json
│   └── ...
├── fusion_log.json           # What fused, what replaced
├── memory_layout.c           # Allocations
└── inference.c               # Generated code
```
