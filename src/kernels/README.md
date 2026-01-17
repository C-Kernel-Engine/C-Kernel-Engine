# CK-Engine Kernels

Kernels are the **bread and butter** of C-Kernel Engine. Each kernel is a
pure computational building block with clearly defined inputs and outputs.

## Core Principles

### 1. No Memory Management

Kernels must **NOT** allocate or free memory:

- NO `malloc()` / `calloc()` / `realloc()` / `free()`
- All memory is pre-allocated via bump allocator at orchestration layer
- Pointers passed to kernel API

### 2. No OpenMP Inside Kernels

Kernels must **NOT** contain parallelization directives:

- NO `#pragma omp parallel` or any OpenMP directives
- Parallelization is orchestrated at the codegen/orchestration layer
- Exceptions exist but must be explicitly documented and justified

### 3. Clearly Defined API

Every kernel function must explicitly define:

- **Inputs**: What pointers and dimensions are required
- **Outputs**: Where results are written and in what format/layout
- **Workspace**: Any intermediate memory needed (pre-allocated via bump allocator)

The bump allocator precomputes total memory requirements before execution.
Workspace memory is part of this allocation - not ad-hoc "scratch" buffers.

Example API pattern:

```c
/**
 * @param x       [in]  Input tensor [seq_len x hidden]
 * @param gamma   [in]  Scale weights [hidden]
 * @param W       [in]  Weight matrix [out_dim x hidden], row-major
 * @param out     [out] Output tensor [seq_len x out_dim]
 * @param work    [work] Workspace [seq_len x hidden] from bump allocator
 */
void fused_rmsnorm_gemm(
    const float *x,
    const float *gamma,
    const float *W,
    float *out,
    float *work,
    int seq_len,
    int hidden,
    int out_dim,
    float eps);

/** Returns workspace bytes needed for bump allocator planning */
size_t fused_rmsnorm_gemm_work_size(int seq_len, int hidden);
```

### 4. No memcpy for Layout Conversion

Kernels must **NOT** use `memcpy()` to reshape or reorder data:

- NO copying data just to change layout (head-major to token-major, etc.)
- Design APIs to accept strided access patterns directly
- CPUs handle strided access efficiently - use offsets, not copies
- If data needs a different layout, the orchestrator should allocate it correctly

**Bad:**
```c
// Copying to reshape - wastes cycles and cache
for (int h = 0; h < num_heads; h++) {
    memcpy(dst + h * head_dim, src + h * stride, head_dim * sizeof(float));
}
```

**Good:**
```c
// Process with stride directly - no copy needed
for (int h = 0; h < num_heads; h++) {
    process_head(src + h * stride, head_dim);
}
```

### 5. Pure Computation

Kernels are deterministic functions:

- Read from input pointers
- Write to output pointers
- Use SIMD intrinsics (SSE/AVX/AVX2/AVX512) for performance
- Handle alignment and tail elements
- NO side effects, NO global state

## Directory Structure

```
src/kernels/
├── fused/                   # New fused kernels (v7+)
│   ├── rmsnorm_qkv.c
│   └── attention_mlp_fused.c
├── *_kernels.c              # Core unfused kernels
├── *_kernels_bf16.c         # BF16 variants
├── *_kernels_q*.c           # Quantized variants (Q4_0, Q4_K, Q6_K, etc.)
├── attention_decode_fused.c # Legacy fused (v6/v6.5 - do not remove)
├── mlp_fused_decode.c       # Legacy fused (v6/v6.5 - do not remove)
└── prefill_fused_gemm.c     # Prefill fusion kernels
```

## Testing Workflow

After modifying or adding any kernel:

```bash
make test                    # Run unit tests
make llamacpp-parity-full    # Verify llama.cpp parity
```

This ensures existing kernels are not broken by changes.

## Legacy Notes

Files like `attention_decode_fused.c` and `mlp_fused_decode.c` in root
`src/kernels/` are legacy fused kernels from v6/v6.5. Keep them to maintain
backward compatibility. New fused kernels go in `src/kernels/fused/`.
