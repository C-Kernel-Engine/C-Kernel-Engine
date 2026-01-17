# CK-Engine Kernel Status Table

Track compliance with kernel guidelines. Last updated: 2026-01-17

## Legend

| Symbol | Meaning |
|--------|---------|
| :white_check_mark: | Compliant |
| :warning: | Has violations (needs review) |
| :construction: | Legacy exception (documented) |
| :x: | Needs refactor |

## Kernel Rules

1. **NO malloc/free** - memory via bump allocator, pointers passed in
2. **NO OpenMP** - parallelization at orchestrator/codegen layer
3. **NO memcpy for layout** - use strided access, not copies
4. **API clarity** - define inputs, outputs, workspace, memory layouts
5. **Pure computation** - deterministic, no side effects

---

## Compliance Summary

| Category | Total | Compliant | Legacy | Needs Work |
|----------|-------|-----------|--------|------------|
| Core Ops | 8 | 8 | 0 | 0 |
| GEMM | 21 | 17 | 1 | 3 |
| Attention | 4 | 1 | 1 | 2 |
| MLP | 4 | 0 | 2 | 2 |
| BF16 Variants | 16 | 15 | 0 | 1 |
| Fused (new) | 7 | 0 | 0 | 7 |
| Other | 8 | 8 | 0 | 0 |
| **Total** | **68** | **49** | **4** | **15** |

---

## Core Operations

| Kernel | OpenMP | malloc | memcpy | Status | Notes |
|--------|--------|--------|--------|--------|-------|
| `rmsnorm_kernels.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `layernorm_kernels.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `softmax_kernels.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `gelu_kernels.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `swiglu_kernels.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `sigmoid_kernels.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `relu_kernels.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `rope_kernels.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |

## GEMM Kernels

| Kernel | OpenMP | malloc | memcpy | Status | Notes |
|--------|--------|--------|--------|--------|-------|
| `gemm_kernels.c` | :x: | :white_check_mark: | :white_check_mark: | :construction: | LEGACY EXCEPTION |
| `gemm_kernels_q4_0.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `gemm_kernels_q4_1.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `gemm_kernels_q4k.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `gemm_kernels_q4k_sse.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `gemm_kernels_q4k_avx.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `gemm_kernels_q4k_q8k.c` | :white_check_mark: | :white_check_mark: | :x: | :warning: | memcpy in qh unpack |
| `gemm_kernels_q4k_q8k_avx2.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `gemm_kernels_q4k_q8k_vnni.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `gemm_kernels_q5_0.c` | :white_check_mark: | :white_check_mark: | :x: | :warning: | memcpy for qh |
| `gemm_kernels_q5_0_sse.c` | :white_check_mark: | :white_check_mark: | :x: | :warning: | memcpy for qh |
| `gemm_kernels_q5_0_sse_v2.c` | :white_check_mark: | :white_check_mark: | :x: | :warning: | memcpy for qh |
| `gemm_kernels_q5_1.c` | :white_check_mark: | :white_check_mark: | :x: | :warning: | memcpy for qh |
| `gemm_kernels_q6k.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `gemm_kernels_q6k_sse.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `gemm_kernels_q6k_q8k.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `gemm_kernels_q8_0.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `gemm_kernels_bf16.c` | :x: | :white_check_mark: | :white_check_mark: | :warning: | Has OpenMP |
| `gemm_kernels_f16.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `gemm_kernels_amx.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `gemm_microkernel.c` | :x: | :white_check_mark: | :white_check_mark: | :warning: | Has OpenMP |
| `gemm_batch_int8.c` | :white_check_mark: | :white_check_mark: | :x: | :warning: | memcpy for packing |
| `gemm_fused_kernels.c` | :x: | :white_check_mark: | :white_check_mark: | :warning: | Has OpenMP |

## Attention Kernels

| Kernel | OpenMP | malloc | memcpy | Status | Notes |
|--------|--------|--------|--------|--------|-------|
| `attention_kernels.c` | :x: | :white_check_mark: | :white_check_mark: | :warning: | Has OpenMP |
| `attention_flash_true.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `attention_decode_fused.c` | :x: | :white_check_mark: | :white_check_mark: | :construction: | LEGACY v6/v6.5 |
| `kv_cache_kernels.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |

## MLP Kernels

| Kernel | OpenMP | malloc | memcpy | Status | Notes |
|--------|--------|--------|--------|--------|-------|
| `mlp_kernels.c` | :x: | :white_check_mark: | :white_check_mark: | :construction: | LEGACY EXCEPTION |
| `mlp_fused_decode.c` | :x: | :white_check_mark: | :x: | :construction: | LEGACY v6/v6.5, has memcpy |
| `mlp_kernels_bf16.c` | :x: | :white_check_mark: | :white_check_mark: | :warning: | Has OpenMP |

## BF16 Variant Kernels

| Kernel | OpenMP | malloc | memcpy | Status | Notes |
|--------|--------|--------|--------|--------|-------|
| `rmsnorm_kernels_bf16.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `layernorm_kernels_bf16.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `softmax_kernels_bf16.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `gelu_kernels_bf16.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `swiglu_kernels_bf16.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `sigmoid_kernels_bf16.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `relu_kernels_bf16.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `rope_kernels_bf16.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `add_kernels_bf16.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `embedding_kernels_bf16.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `loss_kernels_bf16.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `vision_kernels_bf16.c` | :white_check_mark: | :white_check_mark: | :x: | :warning: | memcpy in patch embed |
| `optimizer_kernels_bf16.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |

## INT Variant Kernels

| Kernel | OpenMP | malloc | memcpy | Status | Notes |
|--------|--------|--------|--------|--------|-------|
| `rmsnorm_kernels_int4.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `rmsnorm_kernels_int8.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |

## Other Kernels

| Kernel | OpenMP | malloc | memcpy | Status | Notes |
|--------|--------|--------|--------|--------|-------|
| `embedding_kernels.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `dequant_kernels.c` | :white_check_mark: | :white_check_mark: | :x: | :warning: | memcpy for block copy |
| `loss_kernels.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `topk_kernels.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `axpy_kernels.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `optimizer_kernels.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `vision_kernels.c` | :white_check_mark: | :white_check_mark: | :x: | :warning: | memcpy in patch embed |
| `quantize_row_q8_k_sse.c` | :white_check_mark: | :white_check_mark: | :x: | :warning: | memcpy in block packing |

## Fused Kernels (`fused/` directory)

| Kernel | OpenMP | malloc | memcpy | Status | Notes |
|--------|--------|--------|--------|--------|-------|
| `rmsnorm_q8_k_fused.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `fused_rmsnorm_linear.c` | :white_check_mark: | :x: | :x: | :x: | free() in test, memcpy |
| `rmsnorm_qkv.c` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `attention_mlp_fused.c` | :white_check_mark: | :white_check_mark: | :x: | :warning: | memcpy for layout |
| `mega_fused_attention_avx.c` | :white_check_mark: | :x: | :x: | :x: | malloc + memcpy |
| `mega_fused_attention_prefill.c` | :white_check_mark: | :white_check_mark: | :x: | :warning: | memcpy in flatten_head_major |
| `prefill_fused_gemm.c` | :x: | :white_check_mark: | :white_check_mark: | :warning: | Has OpenMP |

---

## Action Items

### Immediate (memcpy for layout conversion)

These use memcpy to reshape data - should use strided access instead:

1. **`fused/mega_fused_attention_prefill.c`** - `flatten_head_major()` copies head→token layout
2. **`fused/attention_mlp_fused.c`** - memcpy for layout changes
3. **`fused/mega_fused_attention_avx.c`** - malloc + memcpy (needs full refactor)
4. **`fused/fused_rmsnorm_linear.c`** - has free() and memcpy

### Acceptable memcpy (for struct unpacking)

Some memcpy usages are for unpacking packed structs (qh in Q5_0/Q5_1) - these are acceptable:

- `gemm_kernels_q5_0.c` - `memcpy(&qh_val, block->qh, 4)` for unaligned u32 read
- `gemm_kernels_q5_1.c` - same pattern
- `quantize_row_q8_k_sse.c` - block struct packing

### Review Required (OpenMP usage)

1. `gemm_kernels_bf16.c`
2. `gemm_microkernel.c`
3. `gemm_fused_kernels.c`
4. `attention_kernels.c`
5. `mlp_kernels_bf16.c`
6. `fused/prefill_fused_gemm.c`

---

## How to Update This Table

After modifying a kernel:

```bash
# Check for violations
grep -l "#pragma omp" src/kernels/*.c src/kernels/fused/*.c
grep -l "malloc\|calloc\|free(" src/kernels/*.c src/kernels/fused/*.c
grep -l "memcpy" src/kernels/*.c src/kernels/fused/*.c

# Run tests
make test && make llamacpp-parity-full
```

Update this table and commit with your changes.
