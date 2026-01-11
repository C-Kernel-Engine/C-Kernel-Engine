# Comprehensive Parity Testing Infrastructure: C-Kernel-Engine vs llama.cpp

## Overview

This document describes the parity testing infrastructure for validating C-Kernel-Engine (CK) against llama.cpp/ggml implementations.

### Two-Level Testing Strategy

1. **Level 1: Kernel Unit Tests** - Test individual CK kernels against ggml kernels
2. **Level 2: Full Model Runtime Parity** - Tensor-by-tensor comparison during inference

### User Preferences Applied
- **Integration:** Shared library + API
- **Scope:** All layers + per-kernel tests
- **Tolerance:** 1e-3 (normal)

---

## Files Created

| File | Purpose |
|------|---------|
| `llama.cpp/tests/test-kernel-parity.cpp` | GGML kernel wrappers for testing |
| `include/ck_parity_api.h` | CK parity testing API header |
| `src/ck_parity_api.c` | CK parity testing API implementation |
| `scripts/test_kernels_vs_llamacpp.py` | Python kernel-level test driver |
| `scripts/compare_runtime_parity.py` | Full model tensor comparison |
| `scripts/run_full_validation.sh` | Complete validation pipeline |
| `Makefile` (updated) | Build targets for parity testing |

---

## Phase 1: Kernel-Level Testing Library

### 1.1 llama.cpp Kernel Test Library (`test-kernel-parity.cpp`)

Exposes individual ggml ops for testing:

```cpp
extern "C" {
    // Dequantization
    void test_dequant_q4_k(const void *src, float *dst, int n);
    void test_dequant_q6_k(const void *src, float *dst, int n);
    void test_dequant_q4_0(const void *src, float *dst, int n);

    // Quantization
    void test_quantize_q8_k(const float *src, void *dst, int n);

    // GEMV/GEMM
    void test_gemv_q4_k(const void *weight, const float *input, float *output, int cols);
    void test_gemm_q4_k(const void *weight, const float *input, float *output,
                        int rows, int cols, int n_tokens);

    // Activation Kernels
    void test_rmsnorm(const float *input, const float *weight, float *output,
                      int n_tokens, int dim, float eps);
    void test_rope(float *q, float *k, int n_tokens, int n_heads, int n_heads_kv,
                   int head_dim, int pos_offset, float theta);
    void test_swiglu(const float *gate_up, float *output, int n_tokens, int dim);
    void test_softmax(const float *input, float *output, int n);
}
```

**Build:**
```bash
cd llama.cpp
g++ -shared -fPIC -o libggml_kernel_test.so \
    tests/test-kernel-parity.cpp \
    -I ggml/include -I ggml/src \
    -L build/bin -lggml -lggml-cpu -lggml-base -lm -lpthread
```

### 1.1.1 Real Weight Testing Strategy (Recommended)

Instead of generating random quantized data, use **actual model weights** for testing:

```bash
# Step 1: Convert GGUF to bump format (creates manifest with byte offsets)
python scripts/convert_gguf_to_bump.py \
    --gguf ./qwen2.5-3b-instruct-q4_k_m.gguf \
    --output ./weights.bump \
    --verify

# Step 2: Run kernel parity tests using real weights
python scripts/test_kernels_real_weights.py \
    --gguf ./qwen2.5-3b-instruct-q4_k_m.gguf \
    --bump ./weights.bump \
    --manifest ./weights_manifest.json
```

**Benefits:**
1. Weights are in exact format llama.cpp expects (from GGUF)
2. Same weights in bump format for CK (post-conversion)
3. Tests catch real kernel bugs immediately
4. No format mismatches from random data generation

**Sample test points from manifest:**
```
layer.0.wq   | blk.0.attn_q.weight   | Q4_K | offset: 0x1A6F0280 | 2.25 MB
layer.0.wk   | blk.0.attn_k.weight   | Q4_K | offset: 0x1A932280 | 288 KB
layer.18.wq  | blk.18.attn_q.weight  | Q4_K | mid-layer test
layer.35.wv  | blk.35.attn_v.weight  | Q4_K | last layer test
```

### 1.2 CK Parity API (`ck_parity_api.c/h`)

Mirrors the llama.cpp API:

```c
// Dequantization
void ck_test_dequant_q4_k(const void *src, float *dst, int n);
void ck_test_dequant_q6_k(const void *src, float *dst, int n);
void ck_test_dequant_q4_0(const void *src, float *dst, int n);

// Quantization
void ck_test_quantize_q8_k(const float *src, void *dst, int n);

// GEMV/GEMM
void ck_test_gemv_q4_k(const void *weight, const float *input, float *output, int cols);
void ck_test_gemm_q4_k(const void *weight, const float *input, float *output,
                       int rows, int cols, int n_tokens);

// Activation Kernels
void ck_test_rmsnorm(const float *input, const float *weight, float *output,
                     int n_tokens, int dim, float eps);
void ck_test_rope(float *q, float *k, int n_tokens, int n_heads, int n_heads_kv,
                  int head_dim, int pos_offset, float theta);
void ck_test_rope_interleaved(float *q, float *k, ...);  // For llama.cpp compatibility
void ck_test_swiglu(const float *gate_up, float *output, int n_tokens, int dim);
void ck_test_softmax(const float *input, float *output, int n);
```

**Build:**
```bash
make libck_parity.so
```

### 1.3 Python Test Driver (`test_kernels_vs_llamacpp.py`)

```bash
# Run all kernel tests
python scripts/test_kernels_vs_llamacpp.py --all

# Run specific kernel
python scripts/test_kernels_vs_llamacpp.py --kernel dequant_q4k

# Adjust tolerance
python scripts/test_kernels_vs_llamacpp.py --all --tol 1e-4
```

**Expected Output:**
```
======================================================================
KERNEL PARITY TESTS: C-Kernel-Engine vs llama.cpp/ggml
======================================================================

--- test_dequant_q4k (size=256) ---
[PASS] dequant_q4_k: max_diff=0.00e+00, mean=0.00e+00

--- test_gemv_q4k (cols=256) ---
[PASS] gemv_q4_k: max_diff=1.19e-07, mean=2.34e-08

--- test_rmsnorm (tokens=4, dim=256) ---
[PASS] rmsnorm: max_diff=2.38e-07, mean=4.56e-08

--- test_rope (tokens=4, heads=8, dim=64) ---
[PASS] rope_q: max_diff=1.19e-07, mean=3.21e-08
[PASS] rope_k: max_diff=9.54e-08, mean=2.87e-08

--- test_swiglu (tokens=4, inter=256) ---
[PASS] swiglu: max_diff=4.77e-07, mean=1.23e-07

======================================================================
KERNEL TEST SUMMARY
======================================================================
Passed: 8/8

All kernels match llama.cpp/ggml!
```

---

## Phase 2: Full Model Runtime Parity

### 2.1 Tensor Name Mapping

**llama.cpp tensor names** (runtime format):
- `attn_norm-{layer}`, `ffn_norm-{layer}`
- `Qcur-{layer}`, `Kcur-{layer}`, `Vcur-{layer}`
- `attn_out-{layer}`, `ffn_out-{layer}`
- `result_norm`, `result_output`

**CK tensor names** (parity dump format):
- `layer_{layer}_ln1_out`, `layer_{layer}_ln2_out`
- `layer_{layer}_q_proj`, `layer_{layer}_k_proj`, `layer_{layer}_v_proj`
- `layer_{layer}_attn_out`, `layer_{layer}_mlp_out`
- `final_ln`, `output`

### 2.2 Model Parity Script (`compare_runtime_parity.py`)

```bash
python scripts/compare_runtime_parity.py \
    --llama-dump llama_dump \
    --ck-dump parity \
    --manifest weights_manifest.json \
    --tol 1e-3
```

---

## Phase 3: Makefile Integration

### Build Targets Added

```makefile
# Build CK parity library
make libck_parity.so

# Build llama.cpp kernel test library
make llama_kernel_test

# Build both
make parity-libs

# Run kernel tests
make test-kernels

# Run specific kernel test
make test-kernel-dequant_q4k
make test-kernel-gemv_q4k
make test-kernel-rmsnorm
```

---

## Validation Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                    VALIDATION PIPELINE                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. KERNEL TESTS (one-time, for each kernel implementation)     │
│     ┌─────────────────┐     ┌─────────────────┐                 │
│     │ dequant_q4_k    │ vs  │ ggml dequant    │  → PASS/FAIL    │
│     │ gemv_q4_k       │ vs  │ ggml vec_dot    │  → PASS/FAIL    │
│     │ gemm_nt_q4_k    │ vs  │ ggml mul_mat    │  → PASS/FAIL    │
│     │ rmsnorm         │ vs  │ ggml rms_norm   │  → PASS/FAIL    │
│     │ rope            │ vs  │ ggml rope_ext   │  → PASS/FAIL    │
│     │ swiglu          │ vs  │ ggml silu*mul   │  → PASS/FAIL    │
│     └─────────────────┘     └─────────────────┘                 │
│                                                                  │
│  2. WEIGHT CONVERSION (per model)                               │
│     convert_gguf_to_bump.py → weights.bump + manifest           │
│                                                                  │
│  3. WEIGHT VALIDATION (per model)                               │
│     compare_layer_llamacpp.py → verify dequant matches          │
│                                                                  │
│  4. RUNTIME PARITY (per model, ALL layers)                      │
│     compare_runtime_parity.py → tensor-by-tensor comparison     │
│                                                                  │
│  5. END-TO-END OUTPUT (per model)                               │
│     Verify final logits match                                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Full Validation Script (`run_full_validation.sh`)

```bash
# Run everything
./scripts/run_full_validation.sh --gguf model.gguf

# Kernel tests only
./scripts/run_full_validation.sh --kernels-only

# Model parity only (assumes GGUF already converted)
./scripts/run_full_validation.sh --gguf model.gguf --model-only
```

---

## Key Technical Notes

### RoPE Format Difference

- **llama.cpp:** Interleaved format `(x0, x1) → (x0*cos - x1*sin, x0*sin + x1*cos)`
- **CK:** Rotate-half format (split first/second halves)

The `ck_test_rope_interleaved()` function provides llama.cpp-compatible output.

### Quantization Block Sizes

| Format | Block Size | Bytes/Block | Elements/Block |
|--------|------------|-------------|----------------|
| Q4_0   | QK4_0=32   | 18          | 32             |
| Q4_K   | QK_K=256   | 144         | 256            |
| Q6_K   | QK_K=256   | 210         | 256            |
| Q8_K   | QK_K=256   | 292         | 256            |

### GGML Tensor Naming (Runtime)

From `llama.cpp/src/models/llama.cpp`:
```cpp
cb(cur, "attn_norm", il);    // NOT blk.0.attn_norm!
cb(Qcur, "Qcur", il);
cb(cur, "attn_out", il);
cb(cur, "ffn_norm", il);
cb(cur, "ffn_out", il);
cb(cur, "result_norm", -1);
cb(cur, "result_output", -1);
```

---

## Quick Start

```bash
# 1. Build parity libraries
make parity-libs

# 2. Run kernel tests
make test-kernels

# 3. (Optional) Full model validation
./scripts/run_full_validation.sh --gguf path/to/model.gguf
```

---

## Critical: ggml Initialization Requirement

### The Problem

When calling ggml kernel functions directly (e.g., `ggml_vec_dot_q4_K_q8_K()`) from our
test library, they may return **0.0** even though the inputs are valid.

### Root Cause: Uninitialized FP16→FP32 Lookup Table

The K-quant formats (Q4_K, Q6_K, Q8_K) store scale factors as **FP16 (half-precision)**
to save memory. To use these values, ggml must convert them to FP32.

On x86 CPUs without F16C instruction support, ggml uses a **256KB lookup table** for
fast FP16→FP32 conversion:

```c
// llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c:77
float ggml_table_f32_f16[1 << 16];  // 65536 entries × 4 bytes = 256KB
```

The conversion function simply looks up the FP16 bit pattern:

```c
// llama.cpp/ggml/src/ggml-cpu/simd-mappings.h:123
inline static float ggml_lookup_fp16_to_fp32(ggml_fp16_t f) {
    uint16_t s;
    memcpy(&s, &f, sizeof(uint16_t));
    return ggml_table_f32_f16[s];  // Returns 0 if table not initialized!
}
```

### The Initialization Requirement

This table is populated by `ggml_cpu_init()`:

```c
// llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c:3687
for (int i = 0; i < 65536; i++) {
    ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
}
```

**When you use ggml through normal llama.cpp APIs**, this gets called automatically:
- `ggml_backend_cpu_init()` calls `ggml_cpu_init()`
- `ggml_graph_compute()` calls `ggml_cpu_init()`
- Loading a model triggers initialization

**But we bypass all of that** by calling the low-level kernel functions directly.

### The Chain of Failure

```
test_gemv_q4_k()
  └─> ggml_vec_dot_q4_K_q8_K()
        └─> GGML_CPU_FP16_TO_FP32(x[i].d)  // Convert Q4_K block's 'd' scale
              └─> ggml_lookup_fp16_to_fp32(0x3C00)  // 0x3C00 = 1.0 in FP16
                    └─> ggml_table_f32_f16[0x3C00]  // Returns 0.0 (uninitialized!)
                          └─> d = 0.0
                                └─> All products scaled by 0 = 0
```

### The Fix

In `test-kernel-parity.cpp`, the `test_init()` function MUST call `ggml_cpu_init()`:

```cpp
// patches/test-kernel-parity.cpp
#include "ggml-cpu.h"  // Required for ggml_cpu_init()

void test_init(void) {
    ggml_cpu_init();   // Initialize the FP16→FP32 lookup tables!
}
```

### Lesson Learned

> **CRITICAL:** When using ggml kernel functions directly (bypassing the normal
> graph/backend APIs), you MUST call `ggml_cpu_init()` first.

This applies to:
- Parity testing against ggml
- Benchmarking individual kernels
- Any direct kernel calls outside ggml's compute graph

### Symptoms of Missing Initialization

| Symptom | Cause |
|---------|-------|
| K-quant vec_dot returns 0.0 | FP16 scale factor converts to 0.0 |
| Q4_K/Q6_K GEMV returns 0 | Block `d` values are all zero |
| Quantization seems to work but dot products fail | Quantize uses FP32 `d`, but vec_dot reads FP16 `d` |

---

## Q5_0/Q8_0 Kernel Design Difference

### CK vs llama.cpp Approach

| Aspect | CK `gemv_q5_0` | llama.cpp `ggml_vec_dot_q5_0_q8_0` |
|--------|----------------|-----------------------------------|
| Input | **FP32 directly** | FP32 → **Quantize to Q8_0** |
| Computation | `dequant(W) × input_fp32` | `W_q5_0 × input_q8_0` (quantized dot) |
| Accuracy | Higher (FP32 input preserved) | Slightly lower (input quantized) |
| Speed | Slower (dequant on-the-fly) | Faster (integer math) |

**CK's Q5_0 kernel (gemm_kernels_q5_0.c:95-96):**
```c
// Dequantizes weight on-the-fly, multiplies with FP32 input
sum += d * (float)q0 * xp[j];  // dequantized weight × FP32 input
```

**llama.cpp's approach:**
```c
// Quantizes input first, then does quantized dot product
quantize_row_q8_0_ref(input_f32, q8_data, cols);  // Quantize input
ggml_vec_dot_q5_0_q8_0(W, q8_data);               // Q5_0 × Q8_0
```

### Why This Matters for Parity Testing

When comparing CK vs llama.cpp for Q5_0/Q8_0:
- The input quantization in llama.cpp introduces ~0.4% error per element
- This compounds through the dot product
- **Large parity differences are expected** if comparing different approaches

### Solution Options

1. **Option A:** Compare CK against dequant+FP32 reference (tests CK's accuracy)
2. **Option B:** Implement Q5_0×Q8_0 quantized kernel in CK (matches llama.cpp)

For full llama.cpp compatibility, Option B is recommended since:
- Q8_0 quantization introduces minimal error (~0.4%)
- LLMs are robust to small numerical noise
- If forward inference generates coherent text, the accuracy is fine
- llama.cpp uses this approach for all quantized GEMV/GEMM

---

## Troubleshooting

### ggml vec_dot returns 0.0
**Cause:** `ggml_cpu_init()` not called before using kernel functions.
**Fix:** Add `ggml_cpu_init()` call in `test_init()` function.
See "Critical: ggml Initialization Requirement" section above.

### "undefined symbol: dequantize_row_q4_0"
The llama.cpp kernel test library needs to link against `-lggml-cpu` in addition to `-lggml`.

### "undefined symbol: sigmoid_scalar"
The CK parity library needs `sigmoid_kernels.c` included in the build.

### Libraries not loading
Set `LD_LIBRARY_PATH` to include the llama.cpp build directory:
```bash
export LD_LIBRARY_PATH=llama.cpp/build/bin:$LD_LIBRARY_PATH
```

### Q5_0/Q8_0 tests show large differences
**Cause:** CK uses FP32 input directly, llama.cpp quantizes input to Q8_0 first.
**Fix:** Either compare against FP32 reference, or implement Q5_0×Q8_0 kernel in CK.
See "Q5_0/Q8_0 Kernel Design Difference" section above.
