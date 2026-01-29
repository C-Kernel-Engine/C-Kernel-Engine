/**
 * @file gemv_fused_quant_bias.c
 * @brief Fused GEMV kernels with online quantization and bias
 *
 * These kernels fuse:
 *   1. Quantize FP32 input to Q8_0/Q8_K (no memory write)
 *   2. GEMV with quantized weights
 *   3. Bias add
 *
 * Benefits:
 *   - Eliminates memory traffic for quantized activations
 *   - Better cache utilization
 *   - Reduces total ops in IR from 3 to 1
 *
 * Kernel signature:
 *   gemv_fused_q5_0_bias(y, W, x, bias, M, K)
 *   - x: FP32 input [K]
 *   - W: Q5_0 weights [M, K]
 *   - bias: FP32 bias [M] (can be NULL)
 *   - y: FP32 output [M]
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include "ckernel_quant.h"

#if defined(__AVX2__) || defined(__AVX__) || defined(__SSE4_1__)
#include <immintrin.h>
#endif

/* ============================================================================
 * Scalar Helpers
 * ============================================================================ */

/**
 * @brief Round to nearest int, half away from zero (matches quantize_row_q8_0)
 */
static inline int ck_round_nearest(float v) {
    return (int)(v + (v >= 0.0f ? 0.5f : -0.5f));
}

/* ============================================================================
 * AVX + SSE SIMD Helpers
 * ============================================================================
 *
 * These are local copies of helpers from gemm_kernels_q5_0.c, needed for
 * the fused SIMD kernels. Kept local to avoid cross-file dependencies.
 */

#if defined(__AVX__)

/* Combine two __m128i into __m256i (AVX without AVX2) */
#ifndef MM256_SET_M128I_DEFINED
#define MM256_SET_M128I_DEFINED
#define MM256_SET_M128I(hi, lo) \
    _mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 1)
#endif

/**
 * @brief Spread 32 bits to 32 bytes using AVX
 * Returns __m256i with 0xFF where bit was set, 0x00 where not
 */
static inline __m256i fused_bytes_from_bits_32(const uint8_t *qh)
{
    uint32_t x32;
    memcpy(&x32, qh, sizeof(uint32_t));

    const __m128i shuf_maskl = _mm_set_epi64x(0x0101010101010101LL, 0x0000000000000000LL);
    const __m128i shuf_maskh = _mm_set_epi64x(0x0303030303030303LL, 0x0202020202020202LL);

    __m128i bytesl = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskl);
    __m128i bytesh = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskh);

    const __m128i bit_mask = _mm_set1_epi64x(0x7fbfdfeff7fbfdfeLL);

    bytesl = _mm_or_si128(bytesl, bit_mask);
    bytesh = _mm_or_si128(bytesh, bit_mask);

    bytesl = _mm_cmpeq_epi8(bytesl, _mm_set1_epi64x(-1LL));
    bytesh = _mm_cmpeq_epi8(bytesh, _mm_set1_epi64x(-1LL));

    return MM256_SET_M128I(bytesh, bytesl);
}

/**
 * @brief Multiply signed int8 pairs using sign trick (SSSE3)
 * Returns 4 int32 partial sums from 16 int8 pairs
 */
static inline __m128i fused_mul_sum_i8_pairs(__m128i x, __m128i y)
{
    const __m128i ax = _mm_sign_epi8(x, x);   /* abs(x) */
    const __m128i sy = _mm_sign_epi8(y, x);   /* y * sign(x) */
    const __m128i dot = _mm_maddubs_epi16(ax, sy);
    return _mm_madd_epi16(dot, _mm_set1_epi16(1));
}

/**
 * @brief Horizontal sum of 4 int32 in __m128i
 */
static inline int32_t fused_hsum_i32_sse(__m128i v)
{
    __m128i hi64 = _mm_unpackhi_epi64(v, v);
    __m128i sum64 = _mm_add_epi32(hi64, v);
    __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

/**
 * @brief Quantize 32 FP32 values to int8 in SSE registers (no memory write)
 *
 * Uses the same algorithm as quantize_row_q8_0 (AVX path) to ensure
 * numerical parity, but keeps results in registers instead of writing to memory.
 *
 * @param xp       Input: 32 FP32 values
 * @param qa_lo    Output: 16 int8 quantized values [0..15]
 * @param qa_hi    Output: 16 int8 quantized values [16..31]
 * @param out_d_x  Output: quantization scale (after FP16 round-trip)
 * @return amax (0 means all-zero input)
 */
static inline float fused_quantize_block_avx(
    const float *xp,
    __m128i *qa_lo,
    __m128i *qa_hi,
    float *out_d_x)
{
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);
    const __m256 v_half = _mm256_set1_ps(0.5f);
    const __m256 v_min = _mm256_set1_ps(-127.0f);
    const __m256 v_max = _mm256_set1_ps(127.0f);

    /* Load 32 FP32 values */
    __m256 vx0 = _mm256_loadu_ps(&xp[0]);
    __m256 vx1 = _mm256_loadu_ps(&xp[8]);
    __m256 vx2 = _mm256_loadu_ps(&xp[16]);
    __m256 vx3 = _mm256_loadu_ps(&xp[24]);

    /* Find max absolute value */
    __m256 max_abs = _mm256_andnot_ps(sign_bit, vx0);
    max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, vx1));
    max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, vx2));
    max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, vx3));

    /* Horizontal max: 256 -> 128 -> scalar */
    __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(max_abs, 1),
                             _mm256_castps256_ps128(max_abs));
    max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
    max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
    const float amax = _mm_cvtss_f32(max4);

    /* Compute scales */
    float d_x = amax / 127.0f;
    d_x = CK_FP16_TO_FP32(CK_FP32_TO_FP16(d_x)); /* FP16 round-trip for parity */
    *out_d_x = d_x;

    if (amax == 0.0f) {
        *qa_lo = _mm_setzero_si128();
        *qa_hi = _mm_setzero_si128();
        return 0.0f;
    }

    const float id_x = 127.0f / amax;
    const __m256 vmul = _mm256_set1_ps(id_x);

    /* Scale */
    vx0 = _mm256_mul_ps(vx0, vmul);
    vx1 = _mm256_mul_ps(vx1, vmul);
    vx2 = _mm256_mul_ps(vx2, vmul);
    vx3 = _mm256_mul_ps(vx3, vmul);

    /* Clamp to [-127, 127] */
    vx0 = _mm256_min_ps(_mm256_max_ps(vx0, v_min), v_max);
    vx1 = _mm256_min_ps(_mm256_max_ps(vx1, v_min), v_max);
    vx2 = _mm256_min_ps(_mm256_max_ps(vx2, v_min), v_max);
    vx3 = _mm256_min_ps(_mm256_max_ps(vx3, v_min), v_max);

    /* Round half away from zero: v + sign(v) * 0.5 */
    vx0 = _mm256_add_ps(vx0, _mm256_or_ps(_mm256_and_ps(vx0, sign_bit), v_half));
    vx1 = _mm256_add_ps(vx1, _mm256_or_ps(_mm256_and_ps(vx1, sign_bit), v_half));
    vx2 = _mm256_add_ps(vx2, _mm256_or_ps(_mm256_and_ps(vx2, sign_bit), v_half));
    vx3 = _mm256_add_ps(vx3, _mm256_or_ps(_mm256_and_ps(vx3, sign_bit), v_half));

    /* Convert to int32 (truncation after rounding) */
    __m256i i0 = _mm256_cvttps_epi32(vx0);
    __m256i i1 = _mm256_cvttps_epi32(vx1);
    __m256i i2 = _mm256_cvttps_epi32(vx2);
    __m256i i3 = _mm256_cvttps_epi32(vx3);

    /* Pack int32 -> int16 -> int8 (SSE, no AVX2 needed) */
#if defined(__AVX2__)
    /* AVX2: use 256-bit packing + permute */
    i0 = _mm256_packs_epi32(i0, i1);
    i2 = _mm256_packs_epi32(i2, i3);
    i0 = _mm256_packs_epi16(i0, i2);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    i0 = _mm256_permutevar8x32_epi32(i0, perm);
    *qa_lo = _mm256_castsi256_si128(i0);
    *qa_hi = _mm256_extractf128_si256(i0, 1);
#else
    /* AVX (no AVX2): extract 128-bit halves and pack manually */
    __m128i ni0 = _mm256_castsi256_si128(i0);
    __m128i ni1 = _mm256_extractf128_si256(i0, 1);
    __m128i ni2 = _mm256_castsi256_si128(i1);
    __m128i ni3 = _mm256_extractf128_si256(i1, 1);
    __m128i ni4 = _mm256_castsi256_si128(i2);
    __m128i ni5 = _mm256_extractf128_si256(i2, 1);
    __m128i ni6 = _mm256_castsi256_si128(i3);
    __m128i ni7 = _mm256_extractf128_si256(i3, 1);

    ni0 = _mm_packs_epi32(ni0, ni1);
    ni2 = _mm_packs_epi32(ni2, ni3);
    ni4 = _mm_packs_epi32(ni4, ni5);
    ni6 = _mm_packs_epi32(ni6, ni7);

    *qa_lo = _mm_packs_epi16(ni0, ni2);
    *qa_hi = _mm_packs_epi16(ni4, ni6);
#endif

    return amax;
}

/* ============================================================================
 * AVX Fused GEMV Kernels (works on Ivy Bridge and newer)
 * ============================================================================ */

/**
 * @brief AVX fused GEMV: FP32 → online Q8 → Q5_0 weights → FP32 + bias
 *
 * Uses AVX for float ops and SSE/SSSE3 for integer dot products.
 */
static void gemv_fused_q5_0_bias_avx(
    float *y,
    const void *W,
    const float *x,
    const float *bias,
    int M,
    int K)
{
    const block_q5_0 *blocks = (const block_q5_0 *)W;
    const int blocks_per_row = K / QK5_0;

    /* Pre-quantize input x ONCE (not per row) */
    float x_scales[blocks_per_row];
    int8_t x_qs[K];  /* 32 int8 values per block */

    for (int b = 0; b < blocks_per_row; b++) {
        __m128i qa_lo, qa_hi;
        float d_x;
        fused_quantize_block_avx(&x[b * QK5_0], &qa_lo, &qa_hi, &d_x);
        x_scales[b] = d_x;
        _mm_storeu_si128((__m128i *)&x_qs[b * 32], qa_lo);
        _mm_storeu_si128((__m128i *)&x_qs[b * 32 + 16], qa_hi);
    }

    const __m128i mask_0f = _mm_set1_epi8(0x0F);
    const __m128i mask_f0 = _mm_set1_epi8((char)0xF0);

    for (int row = 0; row < M; row++) {
        float sum = 0.0f;

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q5_0 *block = &blocks[row * blocks_per_row + b];
            const float d_w = CK_FP16_TO_FP32(block->d);
            const float d_x = x_scales[b];
            if (d_x == 0.0f) continue;

            const float d = d_w * d_x;

            /* Load pre-quantized input from buffer */
            __m128i qa_lo = _mm_loadu_si128((const __m128i *)&x_qs[b * 32]);
            __m128i qa_hi = _mm_loadu_si128((const __m128i *)&x_qs[b * 32 + 16]);

            /* Decode Q5_0 weights: extract nibbles */
            __m128i qs = _mm_loadu_si128((const __m128i *)block->qs);
            __m128i bx_lo = _mm_and_si128(qs, mask_0f);
            __m128i bx_hi = _mm_and_si128(_mm_srli_epi16(qs, 4), mask_0f);

            /* Spread 32 high bits to 32 bytes */
            __m256i bxhi256 = fused_bytes_from_bits_32(block->qh);
            __m128i bxhi_lo = _mm256_castsi256_si128(bxhi256);
            __m128i bxhi_hi = _mm256_extractf128_si256(bxhi256, 1);

            /* Apply encoding: (~bxhi) & 0xF0 */
            bxhi_lo = _mm_andnot_si128(bxhi_lo, mask_f0);
            bxhi_hi = _mm_andnot_si128(bxhi_hi, mask_f0);

            /* Combine: nibble | high_bit_contribution -> signed Q5_0 weight bytes */
            bx_lo = _mm_or_si128(bx_lo, bxhi_lo);
            bx_hi = _mm_or_si128(bx_hi, bxhi_hi);

            /* Dot product using sign trick */
            __m128i p_lo = fused_mul_sum_i8_pairs(bx_lo, qa_lo);
            __m128i p_hi = fused_mul_sum_i8_pairs(bx_hi, qa_hi);
            __m128i psum = _mm_add_epi32(p_lo, p_hi);

            int32_t sumi = fused_hsum_i32_sse(psum);
            sum += d * (float)sumi;
        }

        if (bias) sum += bias[row];
        y[row] = sum;
    }
}

/**
 * @brief AVX fused GEMV: FP32 → online Q8 → Q8_0 weights → FP32 + bias
 */
static void gemv_fused_q8_0_bias_avx(
    float *y,
    const void *W,
    const float *x,
    const float *bias,
    int M,
    int K)
{
    const block_q8_0 *blocks = (const block_q8_0 *)W;
    const int blocks_per_row = K / QK8_0;

    /* Pre-quantize input x ONCE (not per row) */
    float x_scales[blocks_per_row];
    int8_t x_qs[K];  /* 32 int8 values per block */

    for (int b = 0; b < blocks_per_row; b++) {
        __m128i qa_lo, qa_hi;
        float d_x;
        fused_quantize_block_avx(&x[b * QK8_0], &qa_lo, &qa_hi, &d_x);
        x_scales[b] = d_x;
        _mm_storeu_si128((__m128i *)&x_qs[b * 32], qa_lo);
        _mm_storeu_si128((__m128i *)&x_qs[b * 32 + 16], qa_hi);
    }

    for (int row = 0; row < M; row++) {
        float sum = 0.0f;

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q8_0 *block = &blocks[row * blocks_per_row + b];
            const float d_w = CK_FP16_TO_FP32(block->d);
            const float d_x = x_scales[b];
            if (d_x == 0.0f) continue;

            const float d = d_w * d_x;

            /* Load pre-quantized input from buffer */
            __m128i qa_lo = _mm_loadu_si128((const __m128i *)&x_qs[b * 32]);
            __m128i qa_hi = _mm_loadu_si128((const __m128i *)&x_qs[b * 32 + 16]);

            /* Load Q8_0 weights directly */
            __m128i qw_lo = _mm_loadu_si128((const __m128i *)block->qs);
            __m128i qw_hi = _mm_loadu_si128((const __m128i *)(block->qs + 16));

            /* Dot product using sign trick */
            __m128i p_lo = fused_mul_sum_i8_pairs(qa_lo, qw_lo);
            __m128i p_hi = fused_mul_sum_i8_pairs(qa_hi, qw_hi);
            __m128i psum = _mm_add_epi32(p_lo, p_hi);

            int32_t sumi = fused_hsum_i32_sse(psum);
            sum += d * (float)sumi;
        }

        if (bias) sum += bias[row];
        y[row] = sum;
    }
}

#endif /* __AVX__ */

/* ============================================================================
 * Scalar Reference Implementations
 * ============================================================================ */

/**
 * @brief Compute dot product of FP32 input with Q5_0 weight block, with online Q8 quantization
 */
static inline float dot_fp32_q5_0_block(const float *x, const block_q5_0 *block) {
    const float d_w = CK_FP16_TO_FP32(block->d);

    float amax = 0.0f;
    for (int j = 0; j < 32; j++) {
        float ax = x[j] >= 0 ? x[j] : -x[j];
        if (ax > amax) amax = ax;
    }

    float d_x = amax / 127.0f;
    d_x = CK_FP16_TO_FP32(CK_FP32_TO_FP16(d_x));
    const float id_x = (amax != 0.0f) ? 127.0f / amax : 0.0f;
    const float d = d_w * d_x;

    uint32_t qh;
    memcpy(&qh, block->qh, sizeof(qh));

    int32_t sumi = 0;
    for (int j = 0; j < 16; j++) {
        const uint8_t packed = block->qs[j];
        const int lo = (packed & 0x0F);
        const int hi = (packed >> 4);
        const int xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
        const int xh_1 = ((qh >> (j + 12))) & 0x10;
        const int w0 = (lo | xh_0) - 16;
        const int w1 = (hi | xh_1) - 16;

        float v0 = x[j] * id_x;
        float v1 = x[j + 16] * id_x;
        int q0 = ck_round_nearest(v0);
        int q1 = ck_round_nearest(v1);
        if (q0 > 127) q0 = 127; if (q0 < -127) q0 = -127;
        if (q1 > 127) q1 = 127; if (q1 < -127) q1 = -127;

        sumi += q0 * w0 + q1 * w1;
    }

    return d * (float)sumi;
}

/**
 * @brief Compute dot product of FP32 input with Q8_0 weight block, with online Q8 quantization
 */
static inline float dot_fp32_q8_0_block(const float *x, const block_q8_0 *block) {
    const float d_w = CK_FP16_TO_FP32(block->d);

    float amax = 0.0f;
    for (int j = 0; j < 32; j++) {
        float ax = x[j] >= 0 ? x[j] : -x[j];
        if (ax > amax) amax = ax;
    }

    float d_x = amax / 127.0f;
    d_x = CK_FP16_TO_FP32(CK_FP32_TO_FP16(d_x));
    const float id_x = (amax != 0.0f) ? 127.0f / amax : 0.0f;
    const float d = d_w * d_x;

    int32_t sumi = 0;
    for (int j = 0; j < 32; j++) {
        float v = x[j] * id_x;
        int q = ck_round_nearest(v);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        sumi += q * (int32_t)block->qs[j];
    }

    return d * (float)sumi;
}

/* ============================================================================
 * Scalar Fused GEMV Kernels (fallback)
 * ============================================================================ */

void gemv_fused_q5_0_bias(
    float *y,
    const void *W,
    const float *x,
    const float *bias,
    int M,
    int K)
{
    const block_q5_0 *blocks = (const block_q5_0 *)W;
    const int blocks_per_row = K / QK5_0;

    for (int row = 0; row < M; row++) {
        float sum = 0.0f;

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q5_0 *block = &blocks[row * blocks_per_row + b];
            const float *xp = &x[b * QK5_0];
            sum += dot_fp32_q5_0_block(xp, block);
        }

        if (bias) {
            sum += bias[row];
        }

        y[row] = sum;
    }
}

void gemv_fused_q8_0_bias(
    float *y,
    const void *W,
    const float *x,
    const float *bias,
    int M,
    int K)
{
    const block_q8_0 *blocks = (const block_q8_0 *)W;
    const int blocks_per_row = K / QK8_0;

    for (int row = 0; row < M; row++) {
        float sum = 0.0f;

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q8_0 *block = &blocks[row * blocks_per_row + b];
            const float *xp = &x[b * QK8_0];
            sum += dot_fp32_q8_0_block(xp, block);
        }

        if (bias) {
            sum += bias[row];
        }

        y[row] = sum;
    }
}

/* ============================================================================
 * Dispatch Functions
 * ============================================================================ */

void gemv_fused_q5_0_bias_dispatch(
    float *y,
    const void *W,
    const float *x,
    const float *bias,
    int M,
    int K)
{
#if defined(__AVX__)
    gemv_fused_q5_0_bias_avx(y, W, x, bias, M, K);
#else
    gemv_fused_q5_0_bias(y, W, x, bias, M, K);
#endif
}

void gemv_fused_q8_0_bias_dispatch(
    float *y,
    const void *W,
    const float *x,
    const float *bias,
    int M,
    int K)
{
#if defined(__AVX__)
    gemv_fused_q8_0_bias_avx(y, W, x, bias, M, K);
#else
    gemv_fused_q8_0_bias(y, W, x, bias, M, K);
#endif
}
