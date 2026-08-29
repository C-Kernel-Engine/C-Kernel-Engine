# Quantization Support Implementation Log
**Date:** 2025-12-27
**Goal:** Add Q4_K_M and F16 kernel support to C-Kernel-Engine

---

## Current State

### Existing Kernel Support
| Format | Kernels | Files |
|--------|---------|-------|
| FP32 | All 20 operations | Full coverage |
| BF16 | GELU, LayerNorm, RMSNorm, RoPE | `*_bf16.c` files |
| INT8 | RMSNorm only | `rmsnorm_kernels_int8.c` |
| INT4 | RMSNorm only (simple, no scales) | `rmsnorm_kernels_int4.c` |

### Target Formats (from user's GGUF models)
- **Q4_K_M** - All LLM weights (Devstral-24B, Qwen3VL-8B/4B, SmolLM, etc.)
- **F16** - Vision encoder projections (mmproj-*.gguf files)

---

## Tasks for Today

### Phase 1: Infrastructure
- [ ] Define Q4_K block structure in `include/ckernel_dtype.h`
- [ ] Define F16 conversion utilities in `include/bf16_utils.h` or new `fp16_utils.h`
- [ ] Add Q4_K_M and F16 to dtype enum

### Phase 2: Core Dequantization Kernels
- [ ] Create `src/kernels/dequant_kernels.c` - Q4_K → FP32 dequantization
- [ ] Create `src/kernels/dequant_kernels_f16.c` - F16 → FP32 conversion
- [ ] AVX-512 optimized implementations

### Phase 3: GEMM/GEMV with Quantized Weights
- [ ] Create `src/kernels/gemm_kernels_q4k.c` - Matrix multiply with Q4_K weights
- [ ] Fused dequant + FMA approach (no intermediate buffer)
- [ ] Support both GEMV (batch=1) and GEMM (batch>1)

### Phase 4: Extend Existing Kernels
- [ ] Add Q4_K variants for RMSNorm (replace simple INT4)
- [ ] Add F16 support for attention (for vision encoder)
- [ ] Update kernel registry with new types

### Phase 5: Testing
- [ ] Unit tests comparing Q4_K dequant vs reference
- [ ] GEMM accuracy tests (Q4_K weights vs FP32 reference)
- [ ] Performance benchmarks

---

## Q4_K_M Block Structure Reference

```c
// 256 weights per block = 144 bytes (4.5 bits/weight)
typedef struct {
    uint16_t d;            // 2B: super-block scale (FP16)
    uint16_t dmin;         // 2B: super-block minimum (FP16)
    uint8_t scales[12];    // 12B: 8 sub-block scales (6-bit packed)
    uint8_t qs[128];       // 128B: 256 x 4-bit weights
} block_q4_K;

// Dequantization formula:
// w_fp32 = q * (d * sub_scale) + dmin * sub_min
```

## F16 Format Reference

```c
// IEEE 754 half-precision: 1 sign + 5 exponent + 10 mantissa
// Range: ±65504, precision: ~3 decimal digits

// Conversion (no AVX-512 F16C on all CPUs, use software)
static inline float fp16_to_fp32(uint16_t h);
static inline uint16_t fp32_to_fp16(float f);
```

---

## File Structure Plan

```
src/kernels/
├── dequant_kernels.c        [NEW] Q4_K → FP32
├── dequant_kernels_f16.c    [NEW] F16 → FP32
├── gemm_kernels.c           [EXISTING] FP32 GEMM
├── gemm_kernels_q4k.c       [NEW] Q4_K × FP32 GEMM
├── gemm_kernels_f16.c       [NEW] F16 × FP32 GEMM
└── ...

include/
├── ckernel_dtype.h          [UPDATE] Add Q4_K, F16 types
├── ggml_quants.h            [NEW] Q4_K block definitions
└── fp16_utils.h             [NEW] F16 conversion utilities

unittest/
├── test_dequant_q4k.py      [NEW] Q4_K dequant tests
├── test_gemm_q4k.py         [NEW] Q4_K GEMM parity tests
└── test_f16_convert.py      [NEW] F16 conversion tests
```

---

## Progress Log

### Session Start: 2025-12-27

**Completed:**
- [x] Analyzed existing kernel support
- [x] Identified target formats from user's GGUF models
- [x] Created comprehensive quantization documentation page
- [x] Created SVG infographics (AMX pipeline, Q4_K structure, bump allocator)

**Completed Today:**
- [x] Created `include/ckernel_quant.h` - Q4_K, Q4_0, Q8_0 block structures + FP16 utils
- [x] Updated `include/ckernel_dtype.h` - Added CK_DT_Q4_0, CK_DT_Q4_K, CK_DT_Q8_0
- [x] Created `src/kernels/dequant_kernels.c` - Q4_0, Q4_K, Q8_0 dequantization (scalar + AVX-512)
- [x] Created `src/kernels/gemm_kernels_q4k.c` - GEMV/GEMM with Q4_K weights (fused dequant)
- [x] Created `src/kernels/gemm_kernels_f16.c` - GEMV/GEMM with F16 weights (for mmproj)
- [x] Updated Makefile with new kernel files and `libckernel_q4k.so` target
- [x] Created `unittest/test_q4k_kernels.py` - Unit tests for Q4_K dequant and GEMV
- [x] All unit tests passing:
  - Q4_K Dequantization: max_diff = 0.00 (exact match!)
  - Q4_K GEMV: max_diff = 2.44e-04 (within tolerance for 4-bit quantization)

**Completed (Phase 2):**
- [x] Created `src/kernels/gemm_kernels_q4_0.c` - Q4_0 forward + backward
- [x] Created `src/kernels/gemm_kernels_q8_0.c` - Q8_0 forward + backward
- [x] Added backward pass to Q4_K (`gemv_q4_k_backward`, `gemm_q4_k_backward`)
- [x] Added backward pass to F16 (`gemv_f16_backward`, `gemm_f16_backward`)
- [x] Updated kernel registry with `gemm_quant`, `gemv_quant`, `dequant` entries
- [x] Created comprehensive test suite `unittest/test_quant_kernels.py`
- [x] All 8 tests passing:
  - Q4_0 Dequant: 0.00
  - Q4_0 GEMV: 3.81e-06
  - Q8_0 Dequant: 0.00
  - Q8_0 GEMV: 3.05e-05
  - Q4_K Dequant: 0.00
  - Q4_K GEMV: 2.44e-04
  - Q4_K Backward: 3.05e-04
  - F16 GEMV: 2.86e-06

**Notes:**
- Renamed `ggml_quants.h` to `ckernel_quant.h` for vendor-neutral naming
- Q4_K_M, Q4_K_S, Q4_K_L all use the same `block_q4_K` structure
- The M/S/L suffix indicates quantization aggressiveness, not format differences
- Backward pass computes dL/dX only (weight gradients not computed - weights are frozen)
- For fine-tuning quantized models, use LoRA adapters (separate FP32 weights)

---

## Links to Existing Code

- Dtype definitions: `include/ckernel_dtype.h`
- BF16 utilities: `include/bf16_utils.h`
- Existing INT4 kernel: `src/kernels/rmsnorm_kernels_int4.c`
- GEMM kernels: `src/kernels/gemm_kernels.c`
- Kernel registry: `src/ckernel_kernel_specs.c`

---

## Notes

- Q4_K_M uses the same block structure as Q4_K, the "M" just indicates medium quantization aggressiveness during the quantization process
- F16 is needed for vision encoder projections (mmproj files)
- Activations stay FP32 - only weights are quantized
- Use fused dequant+compute pattern (dequant in registers, never write to RAM)
