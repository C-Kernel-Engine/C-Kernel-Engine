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

## Troubleshooting

### "undefined symbol: dequantize_row_q4_0"
The llama.cpp kernel test library needs to link against `-lggml-cpu` in addition to `-lggml`.

### "undefined symbol: sigmoid_scalar"
The CK parity library needs `sigmoid_kernels.c` included in the build.

### Libraries not loading
Set `LD_LIBRARY_PATH` to include the llama.cpp build directory:
```bash
export LD_LIBRARY_PATH=llama.cpp/build/bin:$LD_LIBRARY_PATH
```
