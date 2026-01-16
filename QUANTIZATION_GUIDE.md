# Quantization Guide for C-Kernel-Engine

This guide explains quantization concepts, GGML/llama.cpp's approach, and how to integrate quantization with C-Kernel-Engine's bump allocator.

## Table of Contents
1. [BF16 (Brain Float 16)](#bf16-brain-float-16)
2. [What is Grouping?](#what-is-grouping)
3. [GGML Quantization Formats](#ggml-quantization-formats)
4. [Memory Layout and Cache Lines](#memory-layout-and-cache-lines)
5. [Bump Allocator Integration](#bump-allocator-integration)
6. [Kernel Dispatch](#kernel-dispatch)

---

## BF16 (Brain Float 16)

BF16 is a 16-bit floating-point format widely used in machine learning. It's the foundation of our reduced-precision compute path.

### Format Comparison

| Format | Sign | Exponent | Mantissa | Range | Precision |
|--------|------|----------|----------|-------|-----------|
| FP32   | 1    | 8        | 23       | ±3.4×10³⁸ | ~7 decimal digits |
| BF16   | 1    | 8        | 7        | ±3.4×10³⁸ | ~2 decimal digits |
| FP16   | 1    | 5        | 10       | ±65,504 | ~3 decimal digits |

### Why BF16 over FP16?

```
BF16:  [S][EEEEEEEE][MMMMMMM]    ← Same exponent as FP32!
FP16:  [S][EEEEE][MMMMMMMMMM]    ← Limited range, overflows in ML
FP32:  [S][EEEEEEEE][MMMMMMMMMMMMMMMMMMMMMMM]
```

**Key insight**: BF16 is literally the upper 16 bits of FP32. This means:
- Same dynamic range as FP32 (no overflow issues in gradients)
- Trivial conversion: just truncate/round the lower 16 bits
- 2× memory savings and bandwidth reduction

### BF16 ↔ FP32 Conversion

**BF16 to FP32** (lossless):
```c
float bf16_to_float(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;  // Place in upper 16 bits
    return *(float*)&bits;               // Zero-extend lower bits
}
```

**FP32 to BF16** (with rounding):
```c
uint16_t float_to_bf16(float f) {
    uint32_t bits = *(uint32_t*)&f;
    uint32_t lsb = (bits >> 16) & 1;      // LSB of result
    bits += 0x7FFF + lsb;                  // Round-to-nearest-even
    return (uint16_t)(bits >> 16);         // Truncate
}
```

### The Round-to-Nearest-Even Algorithm

Why not just truncate (`>> 16`)?

```
Simple truncation always rounds DOWN → systematic negative bias
                                     → errors compound through layers
                                     → training becomes unstable
```

**Round-to-nearest-even** (banker's rounding) eliminates bias:

| Fractional Part | Action | Bias |
|-----------------|--------|------|
| < 0.5 | Round down | None |
| > 0.5 | Round up | None |
| = 0.5 | Round to EVEN | Ties cancel out |

**The `0x7FFF + lsb` trick**:
- `0x7FFF` is "almost half" of the lower 16 bits
- If fraction < 0.5: adding 0x7FFF doesn't overflow → rounds down
- If fraction > 0.5: adding 0x7FFF overflows → rounds up
- If fraction = 0.5: `+lsb` breaks the tie → even result wins

See `bf16_format.svg` and `bf16_rounding.svg` for visual explanations.

### Compute Patterns in C-Kernel-Engine

**Pattern 1: Convert-Compute-Convert**
```c
// Simpler, good for debugging
bf16_tensor_to_float(input_bf16, tmp_fp32, count);
compute_in_fp32(tmp_fp32, output_fp32, count);
float_tensor_to_bf16(output_fp32, output_bf16, count);
```

**Pattern 2: Inline Conversion (preferred)**
```c
// No intermediate memory, used in optimized kernels
for (d = 0; d + 16 <= D; d += 16) {
    __m512 x = bf16_loadu_cvt_fp32(&input_bf16[d]);  // Load+convert
    __m512 y = compute_avx512(x);                    // FP32 compute
    fp32_cvt_storeu_bf16(&output_bf16[d], y);        // Convert+store
}
```

### AVX-512 BF16 Operations

Our BF16 kernels use AVX-512 for 16-wide SIMD:

```c
// Load 16 BF16 → 16 FP32
__m512 bf16_loadu_cvt_fp32(const uint16_t *ptr) {
    __m256i bf16_vec = _mm256_loadu_si256((const __m256i *)ptr);
    __m512i as_int = _mm512_cvtepu16_epi32(bf16_vec);  // Zero-extend
    __m512i shifted = _mm512_slli_epi32(as_int, 16);    // Shift to upper
    return _mm512_castsi512_ps(shifted);
}

// Store 16 FP32 → 16 BF16 (with rounding)
void fp32_cvt_storeu_bf16(uint16_t *ptr, __m512 fp32_vec) {
    __m512i as_int = _mm512_castps_si512(fp32_vec);
    __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(as_int, 16),
                                    _mm512_set1_epi32(1));
    __m512i rounding = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), lsb);
    __m512i rounded = _mm512_add_epi32(as_int, rounding);
    __m512i shifted = _mm512_srli_epi32(rounded, 16);
    __m256i bf16_vec = _mm512_cvtepi32_epi16(shifted);
    _mm256_storeu_si256((__m256i *)ptr, bf16_vec);
}
```

### BF16 Kernel Status

| Kernel | BF16 Support | SIMD |
|--------|-------------|------|
| GEMM | ✓ | AVX-512 |
| RMSNorm | ✓ | AVX-512 |
| Softmax | ✓ | AVX-512 (via tensor convert) |
| RoPE | ✓ | AVX-512 (via tensor convert) |
| GELU | ✓ | AVX-512 (via tensor convert) |
| SwiGLU | ✓ | AVX-512 |
| Attention | ✓ | AVX-512 |
| Embedding | ✓ | Scalar |
| Cross-Entropy | ✓ | Scalar |

---

## What is Grouping?

**The Problem**: A single scale for an entire weight matrix loses precision.

```
Weight matrix [4096 x 4096] with values ranging from -2.0 to +0.001:
- Single scale = 2.0/127 = 0.0157
- Small values like 0.001 become 0 after quantization!
```

**The Solution**: Divide weights into **groups** (blocks), each with its own scale.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GROUPING EXPLAINED                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  WITHOUT GROUPING (per-tensor scale):                                   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Weight Matrix [4096 × 4096]                                      │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │ range: [-2.0 to +0.001] → scale = 2.0/127 = 0.0157        │  │   │
│  │  │                                                             │  │   │
│  │  │ Problem: 0.001 / 0.0157 = 0.06 → rounds to 0 (lost!)      │  │   │
│  │  └────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  WITH GROUPING (per-block scale):                                       │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Weight Matrix [4096 × 4096] → 524,288 blocks of 32 weights      │   │
│  │                                                                    │   │
│  │  Block 0: [-2.0, -1.9, ...] → scale₀ = 2.0/127                   │   │
│  │  Block 1: [-0.1, 0.05, ...] → scale₁ = 0.1/127 (10× more precise)│   │
│  │  Block 2: [0.001, 0.002,..] → scale₂ = 0.01/127 (200× better!)   │   │
│  │  ...                                                               │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Group Size Trade-offs

| Group Size | Scales Stored | Memory Overhead | Precision |
|------------|---------------|-----------------|-----------|
| 32 (GGML)  | 1 per 32 weights | 12.5% | Best |
| 64         | 1 per 64 weights | 6.25% | Good |
| 128        | 1 per 128 weights | 3.1% | Medium |
| 256 (Q4_K) | Nested scales | ~4% | Very Good |

**Cache Line Consideration**: Groups are stored contiguously, so a 64-byte cache line sees:
- `[scale][32 packed weights]` for Q4_0/Q8_0
- NOT mixed types within a cache line

---

## GGML Quantization Formats

### Q4_0: Simple 4-bit (32 weights per block)

```c
#define QK4_0 32  // Group size

typedef struct {
    ggml_fp16_t d;       // 2 bytes: scale (stored as FP16)
    uint8_t qs[QK4_0/2]; // 16 bytes: 32 weights packed (2 per byte)
} block_q4_0;            // Total: 18 bytes per 32 weights

// Memory layout for 128 weights:
// ┌────────────────────────────────────────────────────────────────┐
// │ Block 0 (18B)     │ Block 1 (18B)     │ Block 2 (18B)    │ ...│
// │ [d₀][qs₀...qs₁₅]  │ [d₁][qs₀...qs₁₅]  │ [d₂][qs₀...qs₁₅] │    │
// └────────────────────────────────────────────────────────────────┘
//   ↑                   ↑
//   scale (FP16)        16 bytes = 32 nibbles
```

### Q4_1: 4-bit with min value (asymmetric)

```c
#define QK4_1 32

typedef struct {
    ggml_fp16_t d;       // 2 bytes: scale
    ggml_fp16_t m;       // 2 bytes: minimum value
    uint8_t qs[QK4_1/2]; // 16 bytes: packed weights
} block_q4_1;            // Total: 20 bytes per 32 weights

// Dequantization: value = d * q + m
// Allows representing [min, min+15*scale] instead of [-8*scale, 7*scale]
```

### Q8_0: 8-bit (higher precision)

```c
#define QK8_0 32

typedef struct {
    ggml_fp16_t d;       // 2 bytes: scale
    int8_t qs[QK8_0];    // 32 bytes: 32 INT8 weights
} block_q8_0;            // Total: 34 bytes per 32 weights

// Used for activations or when 4-bit is too lossy
```

### Q4_K: K-Quants (nested scales for better precision)

```c
#define QK_K 256  // Superblock size

typedef struct {
    ggml_fp16_t d;           // 2 bytes: super-scale
    ggml_fp16_t dmin;        // 2 bytes: super-minimum
    uint8_t scales[12];      // 12 bytes: 8 sub-block scales (6-bit each)
    uint8_t qs[QK_K/2];      // 128 bytes: 256 weights packed
} block_q4_K;                // Total: 144 bytes per 256 weights

// Structure:
// ┌─────────────────────────────────────────────────────────────────┐
// │                     SUPERBLOCK (256 weights)                     │
// ├─────────────────────────────────────────────────────────────────┤
// │  d (super-scale)  │  dmin (super-min)  │  scales[12]            │
// ├─────────────────────────────────────────────────────────────────┤
// │  Sub-block 0  │  Sub-block 1  │  ...  │  Sub-block 7            │
// │  (32 weights) │  (32 weights) │       │  (32 weights)           │
// │  scale₀       │  scale₁       │       │  scale₇                 │
// └─────────────────────────────────────────────────────────────────┘
//
// Each sub-block scale = scales[i] * d (two-level quantization)
```

---

## Memory Layout and Cache Lines

### Key Insight: No Type Mixing in Memory

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MEMORY LAYOUT (NO TYPE MIXING)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  WEIGHT REGION (all quantized, read-only during inference):             │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Addr 0x0000: [block_q4_0: d₀ + 16 bytes packed weights]          │   │
│  │ Addr 0x0012: [block_q4_0: d₁ + 16 bytes packed weights]          │   │
│  │ Addr 0x0024: [block_q4_0: d₂ + 16 bytes packed weights]          │   │
│  │ ...                                                               │   │
│  │ (All Q4_0 blocks contiguous - no FP32 mixed in)                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ACTIVATION REGION (all FP32 or BF16, reused each forward):             │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Addr 0x1000: [f32, f32, f32, f32, ...] hidden_state               │   │
│  │ Addr 0x2000: [f32, f32, f32, f32, ...] q_state                    │   │
│  │ Addr 0x3000: [f32, f32, f32, f32, ...] attention_output           │   │
│  │ ...                                                               │   │
│  │ (All FP32 contiguous - cache-friendly sequential access)         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Cache Line Access Pattern

```
64-byte cache line accessing Q4_0 blocks:

Cache Line 0 (bytes 0-63):
┌────────────────────────────────────────────────────────────────┐
│ block_q4_0[0]: 18B │ block_q4_0[1]: 18B │ block_q4_0[2]: 18B  │ 10B partial │
└────────────────────────────────────────────────────────────────┘
        ↑                     ↑                    ↑
    All same type! Sequential read = perfect prefetch.

Dequantization happens in REGISTERS, not memory:
1. Load packed Q4 bytes from weight region
2. Unpack nibbles → INT8 in registers
3. Multiply by scale → FP32 in registers
4. FMA with activation → FP32 accumulator
5. Write result to activation region (FP32)
```

---

## Bump Allocator Integration

### Arena Structure for C-Kernel-Engine

```c
typedef enum {
    DTYPE_F32,
    DTYPE_BF16,
    DTYPE_Q8_0,
    DTYPE_Q4_0,
    DTYPE_Q4_K,
} CKDType;

typedef struct {
    CKDType dtype;
    int     ndim;
    int     shape[4];
    size_t  offset;      // Offset into appropriate region
    void*   data;        // Direct pointer
} CKTensor;

typedef struct {
    // Separate regions by lifetime, not by dtype
    struct {
        uint8_t* base;
        size_t   capacity;
        size_t   used;
    } weights;           // Loaded once, read-only

    struct {
        uint8_t* base;
        size_t   capacity;
        size_t   used;
    } activations;       // Reset each forward pass

    struct {
        uint8_t* base;
        size_t   capacity;
        size_t   used;
    } scratch;           // Reset each layer

} CKArena;
```

### Size Calculation for Quantized Tensors

```c
static inline size_t ck_tensor_size(CKDType dtype, int n_elements) {
    switch (dtype) {
        case DTYPE_F32:
            return n_elements * sizeof(float);

        case DTYPE_BF16:
            return n_elements * sizeof(uint16_t);

        case DTYPE_Q8_0: {
            // 32 elements per block, 34 bytes per block
            int n_blocks = (n_elements + QK8_0 - 1) / QK8_0;
            return n_blocks * sizeof(block_q8_0);
        }

        case DTYPE_Q4_0: {
            // 32 elements per block, 18 bytes per block
            int n_blocks = (n_elements + QK4_0 - 1) / QK4_0;
            return n_blocks * sizeof(block_q4_0);
        }

        case DTYPE_Q4_K: {
            // 256 elements per superblock, 144 bytes per superblock
            int n_blocks = (n_elements + QK_K - 1) / QK_K;
            return n_blocks * sizeof(block_q4_K);
        }

        default:
            return 0;
    }
}

// Allocate weight tensor (cache-aligned)
CKTensor* ck_alloc_weight(CKArena* arena, CKDType dtype, int rows, int cols) {
    CKTensor* t = malloc(sizeof(CKTensor));
    t->dtype = dtype;
    t->ndim = 2;
    t->shape[0] = rows;
    t->shape[1] = cols;

    size_t nbytes = ck_tensor_size(dtype, rows * cols);
    nbytes = (nbytes + 63) & ~63;  // Align to cache line

    t->offset = arena->weights.used;
    t->data = arena->weights.base + arena->weights.used;
    arena->weights.used += nbytes;

    return t;
}

// Allocate activation tensor (always FP32 for simplicity)
CKTensor* ck_alloc_activation(CKArena* arena, int batch, int dim) {
    CKTensor* t = malloc(sizeof(CKTensor));
    t->dtype = DTYPE_F32;
    t->ndim = 2;
    t->shape[0] = batch;
    t->shape[1] = dim;

    size_t nbytes = batch * dim * sizeof(float);
    nbytes = (nbytes + 63) & ~63;  // Align to cache line

    t->offset = arena->activations.used;
    t->data = arena->activations.base + arena->activations.used;
    arena->activations.used += nbytes;

    return t;
}
```

### Loading Quantized Weights from GGUF

```c
int ck_load_gguf_weights(CKArena* arena, const char* path, CKModel* model) {
    // GGUF file structure:
    // 1. Header with metadata
    // 2. Tensor info (names, shapes, dtypes, offsets)
    // 3. Tensor data (aligned)

    FILE* f = fopen(path, "rb");
    GGUFHeader header;
    fread(&header, sizeof(header), 1, f);

    for (int i = 0; i < header.n_tensors; i++) {
        GGUFTensorInfo info;
        read_tensor_info(f, &info);

        // Map GGUF dtype to CK dtype
        CKDType dtype = gguf_to_ck_dtype(info.dtype);
        int n_elements = info.shape[0] * info.shape[1];

        // Allocate in weight region
        CKTensor* t = ck_alloc_weight(arena, dtype, info.shape[0], info.shape[1]);

        // Read directly into arena memory
        fseek(f, info.data_offset, SEEK_SET);
        fread(t->data, ck_tensor_size(dtype, n_elements), 1, f);

        // Store in model
        model->tensors[i] = t;
    }

    fclose(f);
    return 0;
}
```

---

## Kernel Dispatch

### Type-Based Dispatch (No Runtime Checking in Hot Path)

```c
void ck_matmul(CKTensor* out, CKTensor* act, CKTensor* wgt) {
    // Dispatch based on types - done ONCE per layer
    if (act->dtype == DTYPE_F32 && wgt->dtype == DTYPE_Q4_0) {
        ck_matmul_f32_q4_0(out, act, wgt);
    }
    else if (act->dtype == DTYPE_F32 && wgt->dtype == DTYPE_Q4_K) {
        ck_matmul_f32_q4_k(out, act, wgt);
    }
    else if (act->dtype == DTYPE_F32 && wgt->dtype == DTYPE_Q8_0) {
        ck_matmul_f32_q8_0(out, act, wgt);
    }
    else if (act->dtype == DTYPE_F32 && wgt->dtype == DTYPE_F32) {
        ck_matmul_f32_f32(out, act, wgt);
    }
    else if (act->dtype == DTYPE_BF16 && wgt->dtype == DTYPE_Q4_0) {
        ck_matmul_bf16_q4_0(out, act, wgt);
    }
    // ... etc
}
```

### Q4_0 GEMV Kernel (Register Dequantization)

```c
// Dequantization happens in registers - never touches memory
void ck_matmul_f32_q4_0(CKTensor* out, CKTensor* act, CKTensor* wgt) {
    const float* A = (const float*)act->data;
    const block_q4_0* B = (const block_q4_0*)wgt->data;
    float* C = (float*)out->data;

    int M = act->shape[0];
    int K = act->shape[1];
    int N = wgt->shape[0];
    int n_blocks = K / QK4_0;

    #pragma omp parallel for
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;

            for (int b = 0; b < n_blocks; b++) {
                const block_q4_0* blk = &B[n * n_blocks + b];
                float scale = GGML_FP16_TO_FP32(blk->d);

                // Dequantize 32 weights in this block
                for (int j = 0; j < QK4_0/2; j++) {
                    uint8_t packed = blk->qs[j];

                    // Low nibble
                    int8_t q0 = (packed & 0x0F) - 8;
                    float w0 = q0 * scale;
                    sum += A[m * K + b * QK4_0 + j*2] * w0;

                    // High nibble
                    int8_t q1 = (packed >> 4) - 8;
                    float w1 = q1 * scale;
                    sum += A[m * K + b * QK4_0 + j*2 + 1] * w1;
                }
            }
            C[m * N + n] = sum;
        }
    }
}

// AVX-512 optimized version
void ck_matmul_f32_q4_0_avx512(CKTensor* out, CKTensor* act, CKTensor* wgt) {
    // ... (vectorized dequantization in registers)
}
```

---

## Visual Summary

### BF16 Format and Conversion
- `bf16_format.svg` - BF16 vs FP32 bit layout comparison
- `bf16_rounding.svg` - Round-to-nearest-even algorithm flowchart

### Quantization (INT4/INT8)
- `quantization_infographic.svg` - Bit layouts and instruction mappings
- `quantization_grouping.svg` - Detailed grouping visualization

---

## References

- [GGML Quantization Types](https://github.com/ggerganov/ggml/blob/master/src/ggml-quants.h)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
