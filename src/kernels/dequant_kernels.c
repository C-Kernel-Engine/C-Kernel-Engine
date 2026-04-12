/**
 * @file dequant_kernels.c
 * @brief Dequantization kernels for GGML-compatible formats
 *
 * CK-ENGINE KERNEL RULES:
 * =======================
 * 1. NO malloc/free - memory via bump allocator, pointers passed in
 * 2. NO OpenMP - parallelization at orchestrator/codegen layer
 * 3. API must define: inputs, outputs, workspace, and memory layouts
 * 4. Pure computation - deterministic, no side effects
 *
 * After changes: make test && make llamacpp-parity-full
 *
 * Implements dequantization from Q4_0, Q5_0, Q5_1, Q4_K, Q6_K, Q8_0 to FP32.
 * These kernels are used as building blocks for quantized GEMM/GEMV.
 *
 * Key optimization: Dequantize into registers, use immediately in FMA,
 * never write intermediate FP32 values to memory.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif
#include "ckernel_quant.h"

/* ============================================================================
 * Q4_0 Dequantization
 * - 32 weights per block, 1 FP16 scale
 * - Weights stored as signed 4-bit (-8 to +7)
 * ============================================================================ */

/**
 * @brief Dequantize a single Q4_0 block to FP32
 * @param block Pointer to Q4_0 block (18 bytes)
 * @param output Output FP32 array (32 floats)
 */
void dequant_q4_0_block(const block_q4_0 *block, float *output)
{
    const float d = GGML_FP16_TO_FP32(block->d);

    for (int i = 0; i < QK4_0 / 2; i++) {
        const uint8_t packed = block->qs[i];

        /* Lower nibble: elements 0..15 */
        const int8_t q0 = (packed & 0x0F) - 8;
        /* Upper nibble: elements 16..31 */
        const int8_t q1 = (packed >> 4) - 8;

        output[i] = d * (float)q0;
        output[i + QK4_0 / 2] = d * (float)q1;
    }
}

/**
 * @brief Dequantize Q4_0 row (multiple blocks)
 * @param src Q4_0 data
 * @param dst FP32 output
 * @param n_elements Number of elements to dequantize
 */
void dequant_q4_0_row(const void *src, float *dst, size_t n_elements)
{
    const block_q4_0 *blocks = (const block_q4_0 *)src;
    const size_t n_blocks = n_elements / QK4_0;

    for (size_t b = 0; b < n_blocks; b++) {
        dequant_q4_0_block(&blocks[b], &dst[b * QK4_0]);
    }
}

#ifdef __AVX512F__
/**
 * @brief Dequantize Q4_0 block using AVX-512 (16 floats at a time)
 * @param block Pointer to Q4_0 block
 * @param out_lo Lower 16 floats (weights 0-15)
 * @param out_hi Upper 16 floats (weights 16-31)
 */
void dequant_q4_0_block_avx512(const block_q4_0 *block,
                                __m512 *out_lo, __m512 *out_hi)
{
    const __m512 scale = _mm512_set1_ps(GGML_FP16_TO_FP32(block->d));
    const __m512i offset = _mm512_set1_epi32(8);

    /* Load 16 bytes = 32 x 4-bit weights */
    __m128i packed = _mm_loadu_si128((const __m128i *)block->qs);

    /* Unpack lower nibbles (weights 0, 2, 4, ...) */
    __m512i lo_nibbles = _mm512_cvtepu8_epi32(packed);
    lo_nibbles = _mm512_and_epi32(lo_nibbles, _mm512_set1_epi32(0x0F));
    lo_nibbles = _mm512_sub_epi32(lo_nibbles, offset);

    /* Unpack upper nibbles (weights 1, 3, 5, ...) */
    __m512i hi_nibbles = _mm512_cvtepu8_epi32(packed);
    hi_nibbles = _mm512_srli_epi32(hi_nibbles, 4);
    hi_nibbles = _mm512_sub_epi32(hi_nibbles, offset);

    /* Convert to float and scale */
    *out_lo = _mm512_mul_ps(_mm512_cvtepi32_ps(lo_nibbles), scale);
    *out_hi = _mm512_mul_ps(_mm512_cvtepi32_ps(hi_nibbles), scale);

    /* Note: This gives interleaved output (0,2,4... and 1,3,5...)
     * For proper sequential order, would need shuffle/blend */
}
#endif /* __AVX512F__ */

/* ============================================================================
 * Q4_1 Dequantization
 * - 32 weights per block, 1 FP16 scale + 1 FP16 min
 * - Weights stored as unsigned 4-bit (0 to 15)
 * ============================================================================ */

/**
 * @brief Dequantize a single Q4_1 block to FP32
 * @param block Pointer to Q4_1 block (20 bytes)
 * @param output Output FP32 array (32 floats)
 */
void dequant_q4_1_block(const block_q4_1 *block, float *output)
{
    const float d = GGML_FP16_TO_FP32(block->d);
    const float m = GGML_FP16_TO_FP32(block->m);

    for (int i = 0; i < QK4_1 / 2; i++) {
        const uint8_t packed = block->qs[i];

        /* Lower nibble: unsigned 0-15 */
        const int q0 = (packed & 0x0F);
        /* Upper nibble: unsigned 0-15 */
        const int q1 = (packed >> 4);

        /* Dequantize: w = d * q + m */
        output[i] = d * (float)q0 + m;
        output[i + QK4_1 / 2] = d * (float)q1 + m;
    }
}

/**
 * @brief Dequantize Q4_1 row (multiple blocks)
 */
void dequant_q4_1_row(const void *src, float *dst, size_t n_elements)
{
    const block_q4_1 *blocks = (const block_q4_1 *)src;
    const size_t n_blocks = n_elements / QK4_1;

    for (size_t b = 0; b < n_blocks; b++) {
        dequant_q4_1_block(&blocks[b], &dst[b * QK4_1]);
    }
}

/* ============================================================================
 * Q5_0 Dequantization
 * - 32 weights per block, 1 FP16 scale
 * - Low 4 bits + 1 high bit packed separately
 * - Weights are 5-bit signed (-16 to +15)
 * ============================================================================ */

/**
 * @brief Dequantize a single Q5_0 block to FP32
 * @param block Pointer to Q5_0 block (22 bytes)
 * @param output Output FP32 array (32 floats)
 */
void dequant_q5_0_block(const block_q5_0 *block, float *output)
{
    const float d = GGML_FP16_TO_FP32(block->d);

    /* Get high bits as a 32-bit integer */
    uint32_t qh;
    memcpy(&qh, block->qh, sizeof(qh));

    /* llama.cpp Q5_0 layout:
     * - Weight j uses: low nibble of qs[j], high bit from qh bit j
     * - Weight j+16 uses: high nibble of qs[j], high bit from qh bit (j+12)
     */
    for (int j = 0; j < QK5_0 / 2; j++) {
        const uint8_t packed = block->qs[j];

        /* Extract low 4 bits for two weights */
        const int lo = (packed & 0x0F);
        const int hi = (packed >> 4);

        /* Extract high bits from qh - matches llama.cpp exactly */
        const int xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
        const int xh_1 = ((qh >> (j + 12))) & 0x10;

        /* Combine: 5-bit value, range 0-31, then subtract 16 */
        const int q0 = (lo | xh_0) - 16;
        const int q1 = (hi | xh_1) - 16;

        output[j] = d * (float)q0;
        output[j + 16] = d * (float)q1;
    }
}

/**
 * @brief Dequantize Q5_0 row (multiple blocks)
 */
void dequant_q5_0_row(const void *src, float *dst, size_t n_elements)
{
    const block_q5_0 *blocks = (const block_q5_0 *)src;
    const size_t n_blocks = n_elements / QK5_0;

    for (size_t b = 0; b < n_blocks; b++) {
        dequant_q5_0_block(&blocks[b], &dst[b * QK5_0]);
    }
}

/* ============================================================================
 * Q5_1 Dequantization
 * - 32 weights per block, 1 FP16 scale + 1 FP16 min
 * - Low 4 bits + 1 high bit packed separately
 * - Weights are unsigned 5-bit (0 to 31), scaled and offset by min
 * ============================================================================ */

/**
 * @brief Dequantize a single Q5_1 block to FP32
 * @param block Pointer to Q5_1 block (24 bytes)
 * @param output Output FP32 array (32 floats)
 */
void dequant_q5_1_block(const block_q5_1 *block, float *output)
{
    const float d = GGML_FP16_TO_FP32(block->d);
    const float m = GGML_FP16_TO_FP32(block->m);

    /* Get high bits as a 32-bit integer */
    uint32_t qh;
    memcpy(&qh, block->qh, sizeof(qh));

    /* llama.cpp Q5_1 layout (same as Q5_0):
     * - Weight j uses: low nibble of qs[j], high bit from qh bit j
     * - Weight j+16 uses: high nibble of qs[j], high bit from qh bit (j+12)
     */
    for (int j = 0; j < QK5_1 / 2; j++) {
        const uint8_t packed = block->qs[j];

        /* Extract low 4 bits for two weights */
        const int lo = (packed & 0x0F);
        const int hi = (packed >> 4);

        /* Extract high bits from qh - matches llama.cpp exactly */
        const int xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
        const int xh_1 = ((qh >> (j + 12))) & 0x10;

        /* Combine: 5-bit unsigned value, range 0-31 */
        const int q0 = (lo | xh_0);
        const int q1 = (hi | xh_1);

        /* Dequantize: w = d * q + m */
        output[j] = d * (float)q0 + m;
        output[j + 16] = d * (float)q1 + m;
    }
}

/**
 * @brief Dequantize Q5_1 row (multiple blocks)
 */
void dequant_q5_1_row(const void *src, float *dst, size_t n_elements)
{
    const block_q5_1 *blocks = (const block_q5_1 *)src;
    const size_t n_blocks = n_elements / QK5_1;

    for (size_t b = 0; b < n_blocks; b++) {
        dequant_q5_1_block(&blocks[b], &dst[b * QK5_1]);
    }
}

/* ============================================================================
 * Q8_0 Dequantization
 * - 32 weights per block, 1 FP16 scale
 * - Weights stored as signed 8-bit
 * ============================================================================ */

/**
 * @brief Dequantize a single Q8_0 block to FP32
 */
void dequant_q8_0_block(const block_q8_0 *block, float *output)
{
    const float d = GGML_FP16_TO_FP32(block->d);

    for (int i = 0; i < QK8_0; i++) {
        output[i] = d * (float)block->qs[i];
    }
}

/**
 * @brief Dequantize Q8_0 row (multiple blocks)
 */
void dequant_q8_0_row(const void *src, float *dst, size_t n_elements)
{
    const block_q8_0 *blocks = (const block_q8_0 *)src;
    const size_t n_blocks = n_elements / QK8_0;

    for (size_t b = 0; b < n_blocks; b++) {
        dequant_q8_0_block(&blocks[b], &dst[b * QK8_0]);
    }
}

#ifdef __AVX512F__
/**
 * @brief Dequantize Q8_0 block using AVX-512
 */
void dequant_q8_0_block_avx512(const block_q8_0 *block,
                                __m512 *out0, __m512 *out1)
{
    const __m512 scale = _mm512_set1_ps(GGML_FP16_TO_FP32(block->d));

    /* Load 32 x int8 as two __m128i */
    __m128i q0 = _mm_loadu_si128((const __m128i *)&block->qs[0]);
    __m128i q1 = _mm_loadu_si128((const __m128i *)&block->qs[16]);

    /* Sign-extend to 32-bit and convert to float */
    __m512i i0 = _mm512_cvtepi8_epi32(q0);
    __m512i i1 = _mm512_cvtepi8_epi32(q1);

    *out0 = _mm512_mul_ps(_mm512_cvtepi32_ps(i0), scale);
    *out1 = _mm512_mul_ps(_mm512_cvtepi32_ps(i1), scale);
}
#endif /* __AVX512F__ */

/* ============================================================================
 * Q4_K Dequantization (Primary Target for Q4_K_M)
 * - 256 weights per super-block
 * - 8 sub-blocks of 32 weights each
 * - Two-level scaling: super-block d/dmin + sub-block 6-bit scales
 * ============================================================================ */

/**
 * @brief Dequantize a single Q4_K block to FP32
 *
 * This matches llama.cpp's dequantize_row_q4_K exactly:
 * - Formula: weight = d * scale * q - dmin * m
 * - Layout: 4 iterations of 64 weights each
 *   - First 32: low nibbles of qs[0..31] with scale[2*iter], min[2*iter]
 *   - Next 32: high nibbles of qs[0..31] with scale[2*iter+1], min[2*iter+1]
 */
void dequant_q4_k_block(const block_q4_K *block, float *output)
{
    const float d = GGML_FP16_TO_FP32(block->d);
    const float dmin = GGML_FP16_TO_FP32(block->dmin);

    /* Unpack the 6-bit sub-block scales and mins */
    uint8_t sc[8], m[8];
    unpack_q4_k_scales(block->scales, sc, m);

    /* llama.cpp layout: 4 iterations of 64 weights each */
    for (int iter = 0; iter < 4; iter++) {
        const float d1 = d * (float)sc[2 * iter];
        const float m1 = dmin * (float)m[2 * iter];
        const float d2 = d * (float)sc[2 * iter + 1];
        const float m2 = dmin * (float)m[2 * iter + 1];

        const uint8_t *qs = &block->qs[iter * 32];
        float *out = &output[iter * 64];

        /* First 32 weights: low nibbles */
        for (int l = 0; l < 32; l++) {
            const int q = (qs[l] & 0x0F);
            out[l] = d1 * (float)q - m1;
        }

        /* Next 32 weights: high nibbles */
        for (int l = 0; l < 32; l++) {
            const int q = (qs[l] >> 4);
            out[32 + l] = d2 * (float)q - m2;
        }
    }
}

/**
 * @brief Dequantize Q4_K row (multiple blocks)
 */
void dequant_q4_k_row(const void *src, float *dst, size_t n_elements)
{
    const block_q4_K *blocks = (const block_q4_K *)src;
    const size_t n_blocks = n_elements / QK_K;

    for (size_t b = 0; b < n_blocks; b++) {
        dequant_q4_k_block(&blocks[b], &dst[b * QK_K]);
    }
}

/* ============================================================================
 * Q6_K Dequantization
 * - 256 weights per block
 * - 16 sub-blocks of 16 weights, int8 scales + FP16 super-scale
 * ============================================================================ */

/**
 * @brief Dequantize a single Q6_K block to FP32
 */
void dequant_q6_k_block(const block_q6_K *block, float *output)
{
    const float d = GGML_FP16_TO_FP32(block->d);
    const uint8_t *ql = block->ql;
    const uint8_t *qh = block->qh;
    const int8_t *sc = block->scales;
    float *y = output;

    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

            y[l + 0] = d * (float)sc[is + 0] * (float)q1;
            y[l + 32] = d * (float)sc[is + 2] * (float)q2;
            y[l + 64] = d * (float)sc[is + 4] * (float)q3;
            y[l + 96] = d * (float)sc[is + 6] * (float)q4;
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

/**
 * @brief Dequantize Q6_K row (multiple blocks)
 */
void dequant_q6_k_row(const void *src, float *dst, size_t n_elements)
{
    const block_q6_K *blocks = (const block_q6_K *)src;
    const size_t n_blocks = n_elements / QK_K;

    for (size_t b = 0; b < n_blocks; b++) {
        dequant_q6_k_block(&blocks[b], &dst[b * QK_K]);
    }
}

#ifdef __AVX512F__
/**
 * @brief Dequantize one Q4_K sub-block (32 weights) using AVX-512
 *
 * @param qs Pointer to 16 bytes of packed 4-bit weights
 * @param scale Pre-computed d * sub_scale
 * @param min_val Pre-computed dmin * sub_min
 * @param out0 Output: weights 0-15
 * @param out1 Output: weights 16-31
 */
/**
 * @brief Dequantize full Q4_K block using AVX-512
 *
 * This matches llama.cpp's dequantize_row_q4_K exactly:
 * - Formula: weight = d * scale * q - dmin * m
 * - Layout: 4 iterations of 64 weights each
 *   - First 32: low nibbles of qs[0..31] with scale[2*iter], min[2*iter]
 *   - Next 32: high nibbles of qs[0..31] with scale[2*iter+1], min[2*iter+1]
 */
void dequant_q4_k_block_avx512(const block_q4_K *block, float *output)
{
    const float d = GGML_FP16_TO_FP32(block->d);
    const float dmin = GGML_FP16_TO_FP32(block->dmin);

    uint8_t sc[8], m[8];
    unpack_q4_k_scales(block->scales, sc, m);

    const __m512i mask_lo = _mm512_set1_epi32(0x0F);

    /* llama.cpp layout: 4 iterations of 64 weights each */
    for (int iter = 0; iter < 4; iter++) {
        const float d1 = d * (float)sc[2 * iter];
        const float m1 = dmin * (float)m[2 * iter];
        const float d2 = d * (float)sc[2 * iter + 1];
        const float m2 = dmin * (float)m[2 * iter + 1];

        const __m512 vd1 = _mm512_set1_ps(d1);
        const __m512 vm1 = _mm512_set1_ps(m1);
        const __m512 vd2 = _mm512_set1_ps(d2);
        const __m512 vm2 = _mm512_set1_ps(m2);

        const uint8_t *qs = &block->qs[iter * 32];
        float *out = &output[iter * 64];

        /* Process first 32 weights (low nibbles) in two 16-float chunks */
        for (int chunk = 0; chunk < 2; chunk++) {
            __m128i packed = _mm_loadu_si128((const __m128i *)&qs[chunk * 16]);
            __m512i bytes = _mm512_cvtepu8_epi32(packed);
            __m512i lo = _mm512_and_epi32(bytes, mask_lo);
            /* w = d1 * q - m1: fnmadd computes -(a*b) + c = c - a*b = -m1 + d1*q */
            __m512 w = _mm512_fnmadd_ps(_mm512_set1_ps(1.0f), vm1,
                       _mm512_mul_ps(_mm512_cvtepi32_ps(lo), vd1));
            _mm512_storeu_ps(&out[chunk * 16], w);
        }

        /* Process next 32 weights (high nibbles) in two 16-float chunks */
        for (int chunk = 0; chunk < 2; chunk++) {
            __m128i packed = _mm_loadu_si128((const __m128i *)&qs[chunk * 16]);
            __m512i bytes = _mm512_cvtepu8_epi32(packed);
            __m512i hi = _mm512_srli_epi32(bytes, 4);
            /* w = d2 * q - m2 */
            __m512 w = _mm512_fnmadd_ps(_mm512_set1_ps(1.0f), vm2,
                       _mm512_mul_ps(_mm512_cvtepi32_ps(hi), vd2));
            _mm512_storeu_ps(&out[32 + chunk * 16], w);
        }
    }
}
#endif /* __AVX512F__ */

/* ============================================================================
 * Generic Dequantization Dispatch
 * ============================================================================ */

#include "ckernel_dtype.h"

/**
 * @brief Dequantize a row of quantized data to FP32
 * @param dtype Data type (must be quantized type)
 * @param src Source quantized data
 * @param dst Destination FP32 buffer
 * @param n_elements Number of elements
 */
void dequant_row(CKDataType dtype, const void *src, float *dst, size_t n_elements)
{
    switch (dtype) {
    case CK_DT_Q4_0:
        dequant_q4_0_row(src, dst, n_elements);
        break;
    case CK_DT_Q4_1:
        dequant_q4_1_row(src, dst, n_elements);
        break;
    case CK_DT_Q5_0:
        dequant_q5_0_row(src, dst, n_elements);
        break;
    case CK_DT_Q5_1:
        dequant_q5_1_row(src, dst, n_elements);
        break;
    case CK_DT_Q4_K:
        dequant_q4_k_row(src, dst, n_elements);
        break;
    case CK_DT_Q6_K:
        dequant_q6_k_row(src, dst, n_elements);
        break;
    case CK_DT_Q8_0:
        dequant_q8_0_row(src, dst, n_elements);
        break;
    default:
        /* Not a quantized type - no-op or error */
        break;
    }
}
