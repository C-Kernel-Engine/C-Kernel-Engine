/**
 * @file gemm_kernels_q5_0.c
 * @brief GEMM/GEMV kernels with Q5_0 quantized weights
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
 * Q5_0 Format:
 *   - 32 weights per block
 *   - 1 FP16 scale per block
 *   - Low 4-bits stored like Q4_0 (16 bytes)
 *   - High 1-bit packed separately (4 bytes)
 *   - 22 bytes per 32 weights = 5.5 bits/weight
 *
 * Dequantization: w = scale * (q5 - 16)
 * where q5 = low4bit | (highbit << 4), giving values 0-31, then subtract 16 for signed -16 to +15
 *
 * Operations:
 *   Forward:  Y = W @ X  (W is Q5_0, X and Y are FP32)
 *   Backward: dX = W^T @ dY (gradient w.r.t. input)
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdatomic.h>
#include "ckernel_quant.h"
#include "ck_features.h"

/* These checked callers can build the translation unit for a baseline ISA. */
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && \
    (!defined(__AVX__) || defined(__FMA__))
#define CK_Q5_RUNTIME_AVX2 1
#define CK_Q5_TARGET_AVX __attribute__((target("avx,ssse3")))
#define CK_Q5_TARGET_AVX2 __attribute__((target("avx2,fma,f16c")))
#endif
#ifndef CK_Q5_TARGET_AVX
#define CK_Q5_TARGET_AVX
#define CK_Q5_TARGET_AVX2
#endif

/* Include SIMD headers based on available extensions */
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__) || defined(__SSE4_1__) || defined(CK_Q5_RUNTIME_AVX2)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

/* Forward declarations for dequant functions (defined in dequant_kernels.c) */
void dequant_q5_0_block(const block_q5_0 *block, float *output);
void dequant_q5_0_row(const void *src, float *dst, size_t n_elements);

void gemm_nt_q5_0_sse_v2(const float *A,
                         const void *B,
                         const float *bias,
                         float *C,
                         int M, int N, int K);

/* ============================================================================
 * Forward Pass: GEMV y = W @ x
 * ============================================================================ */

/**
 * @brief Matrix-vector multiply with Q5_0 weights (scalar reference)
 *
 * @param y Output vector [M]
 * @param W Weight matrix in Q5_0 format [M x K]
 * @param x Input vector [K]
 * @param M Number of output rows
 * @param K Number of columns (must be multiple of 32)
 */
void gemv_q5_0_ref(float *y,
                   const void *W,
                   const float *x,
                   int M, int K)
{
    const block_q5_0 *blocks = (const block_q5_0 *)W;
    const int blocks_per_row = K / QK5_0;

    for (int row = 0; row < M; row++) {
        float sum = 0.0f;

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q5_0 *block = &blocks[row * blocks_per_row + b];
            const float d = CK_FP16_TO_FP32(block->d);
            const float *xp = &x[b * QK5_0];

            /* Get high bits as 32-bit integer */
            uint32_t qh;
            memcpy(&qh, block->qh, sizeof(qh));

            /* llama.cpp Q5_0 layout:
             * - Weight j uses: low nibble of qs[j], high bit from qh bit j
             * - Weight j+16 uses: high nibble of qs[j], high bit from qh bit (j+12)
             * Note: j+12 not j+16 for the high bit of the second weight!
             */
            for (int j = 0; j < QK5_0 / 2; j++) {
                const uint8_t packed = block->qs[j];

                /* Extract nibbles */
                const int lo = (packed & 0x0F);
                const int hi = (packed >> 4);

                /* Extract high bits - matches llama.cpp exactly */
                const int xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
                const int xh_1 = ((qh >> (j + 12))) & 0x10;

                /* Combine to 5-bit signed value */
                const int q0 = (lo | xh_0) - 16;
                const int q1 = (hi | xh_1) - 16;

                /* Weights at indices j and j+16 */
                sum += d * (float)q0 * xp[j];
                sum += d * (float)q1 * xp[j + 16];
            }
        }

        y[row] = sum;
    }
}

#ifdef __AVX512F__
/**
 * @brief Matrix-vector multiply with Q5_0 weights (AVX-512)
 */
void gemv_q5_0_avx512(float *y,
                      const void *W,
                      const float *x,
                      int M, int K)
{
    const block_q5_0 *blocks = (const block_q5_0 *)W;
    const int blocks_per_row = K / QK5_0;
    const __m512i offset = _mm512_set1_epi32(16);
    const __m512i mask_lo = _mm512_set1_epi32(0x0F);
    const __m512i one = _mm512_set1_epi32(1);

    for (int row = 0; row < M; row++) {
        __m512 acc = _mm512_setzero_ps();

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q5_0 *block = &blocks[row * blocks_per_row + b];
            const __m512 vscale = _mm512_set1_ps(CK_FP16_TO_FP32(block->d));
            const float *xp = &x[b * QK5_0];

            /* Load high bits */
            uint32_t qh;
            memcpy(&qh, block->qh, sizeof(qh));

            /* Load 16 bytes = 32 x 4-bit low weights */
            __m128i packed = _mm_loadu_si128((const __m128i *)block->qs);
            __m512i bytes = _mm512_cvtepu8_epi32(packed);

            /* Extract low nibbles */
            __m512i lo = _mm512_and_epi32(bytes, mask_lo);
            __m512i hi_shift = _mm512_srli_epi32(bytes, 4);

            /* llama.cpp Q5_0 layout:
             * - Weights 0-15: high bits from qh bits 0-15
             * - Weights 16-31: high bits from qh bits 12-27 (j+12 where j=0..15)
             */
            /* Build high bit contribution for first 16 weights (indices 0-15) */
            __m512i qh_lo = _mm512_set_epi32(
                ((qh >> 15) & 1) << 4, ((qh >> 14) & 1) << 4,
                ((qh >> 13) & 1) << 4, ((qh >> 12) & 1) << 4,
                ((qh >> 11) & 1) << 4, ((qh >> 10) & 1) << 4,
                ((qh >> 9) & 1) << 4, ((qh >> 8) & 1) << 4,
                ((qh >> 7) & 1) << 4, ((qh >> 6) & 1) << 4,
                ((qh >> 5) & 1) << 4, ((qh >> 4) & 1) << 4,
                ((qh >> 3) & 1) << 4, ((qh >> 2) & 1) << 4,
                ((qh >> 1) & 1) << 4, ((qh >> 0) & 1) << 4
            );

            /* Build high bit contribution for second 16 weights (indices 16-31)
             * Scalar code: xh_1 = ((qh >> (j + 12))) & 0x10 extracts bit (j+16)
             * (since 0x10 = bit 4, position is j+12+4 = j+16)
             * So weights 16-31 use qh bits 16-31 */
            __m512i qh_hi = _mm512_set_epi32(
                ((qh >> 31) & 1) << 4, ((qh >> 30) & 1) << 4,
                ((qh >> 29) & 1) << 4, ((qh >> 28) & 1) << 4,
                ((qh >> 27) & 1) << 4, ((qh >> 26) & 1) << 4,
                ((qh >> 25) & 1) << 4, ((qh >> 24) & 1) << 4,
                ((qh >> 23) & 1) << 4, ((qh >> 22) & 1) << 4,
                ((qh >> 21) & 1) << 4, ((qh >> 20) & 1) << 4,
                ((qh >> 19) & 1) << 4, ((qh >> 18) & 1) << 4,
                ((qh >> 17) & 1) << 4, ((qh >> 16) & 1) << 4
            );

            /* Combine low + high bits and subtract offset */
            __m512i q_lo = _mm512_sub_epi32(_mm512_or_epi32(lo, qh_lo), offset);
            __m512i q_hi = _mm512_sub_epi32(_mm512_or_epi32(hi_shift, qh_hi), offset);

            /* Dequantize */
            __m512 w_lo = _mm512_mul_ps(_mm512_cvtepi32_ps(q_lo), vscale);
            __m512 w_hi = _mm512_mul_ps(_mm512_cvtepi32_ps(q_hi), vscale);

            /* Load sequential input: x[0-15] and x[16-31] */
            __m512 x_first = _mm512_loadu_ps(&xp[0]);    /* x[0..15] */
            __m512 x_second = _mm512_loadu_ps(&xp[16]);  /* x[16..31] */

            acc = _mm512_fmadd_ps(w_lo, x_first, acc);
            acc = _mm512_fmadd_ps(w_hi, x_second, acc);
        }

        y[row] = _mm512_reduce_add_ps(acc);
    }
}
#endif

/* ============================================================================
 * AVX2 Implementation (Haswell+, 256-bit integer operations)
 *
 * AVX2 provides true 256-bit integer operations that AVX lacks.
 * This implementation uses:
 *   - _mm256_cvtepi8_epi32: Sign-extend 8 bytes to 8 int32s (AVX2)
 *   - _mm256_fmadd_ps: Fused multiply-add (FMA3, available with AVX2)
 *   - 256-bit integer shuffles and masks
 *
 * Processing: 32 weights per block, 8 at a time with AVX2 registers
 * ============================================================================ */

#if defined(__AVX2__) && !defined(__AVX512F__)

/* Helper: AVX2 horizontal sum of 8 floats */
static inline float hsum_avx2(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);  /* 4 floats */
    __m128 shuf = _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

/**
 * @brief Matrix-vector multiply with Q5_0 weights (AVX2 optimized)
 *
 * Uses AVX2 256-bit integer operations for efficient dequantization.
 * Processes 8 weights at a time with full 256-bit registers.
 */
void gemv_q5_0_avx2(float *y,
                    const void *W,
                    const float *x,
                    int M, int K)
{
    const block_q5_0 *blocks = (const block_q5_0 *)W;
    const int blocks_per_row = K / QK5_0;  /* QK5_0 = 32 */

    for (int row = 0; row < M; row++) {
        __m256 acc = _mm256_setzero_ps();

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q5_0 *block = &blocks[row * blocks_per_row + b];
            const float d = CK_FP16_TO_FP32(block->d);
            const __m256 vscale = _mm256_set1_ps(d);
            const float *xp = &x[b * QK5_0];

            /* Get high bits as 32-bit integer */
            uint32_t qh;
            memcpy(&qh, block->qh, sizeof(qh));

            /* Q5_0 layout: 32 weights per block
             * - Weights 0-15: low nibbles of qs[0-15], high bit from qh[0-15]
             * - Weights 16-31: high nibbles of qs[0-15], high bit from qh[16-31]
             *
             * Process in 4 groups of 8 for AVX2:
             */

            /* Group 0: weights 0-7 (low nibbles of qs[0-7], high bits qh[0-7]) */
            {
                __m128i qs8 = _mm_loadl_epi64((const __m128i *)block->qs);
                __m128i lo = _mm_and_si128(qs8, _mm_set1_epi8(0x0F));

                /* Build high bits for weights 0-7 */
                int8_t hb[8];
                for (int i = 0; i < 8; i++) {
                    hb[i] = ((qh >> i) << 4) & 0x10;
                }
                __m128i hi = _mm_loadl_epi64((const __m128i *)hb);

                /* Combine and subtract offset to get signed values */
                __m128i q5 = _mm_or_si128(lo, hi);
                __m128i offset = _mm_set1_epi8(16);
                __m128i q5_signed = _mm_sub_epi8(q5, offset);

                /* Sign-extend to 32-bit and convert to float (AVX2) */
                __m256i q32 = _mm256_cvtepi8_epi32(q5_signed);
                __m256 wf = _mm256_cvtepi32_ps(q32);
                wf = _mm256_mul_ps(wf, vscale);

                /* Load input and accumulate */
                __m256 xv = _mm256_loadu_ps(&xp[0]);
                acc = _mm256_fmadd_ps(wf, xv, acc);
            }

            /* Group 1: weights 8-15 (low nibbles of qs[8-15], high bits qh[8-15]) */
            {
                __m128i qs8 = _mm_loadl_epi64((const __m128i *)(block->qs + 8));
                __m128i lo = _mm_and_si128(qs8, _mm_set1_epi8(0x0F));

                int8_t hb[8];
                for (int i = 0; i < 8; i++) {
                    hb[i] = ((qh >> (8 + i)) << 4) & 0x10;
                }
                __m128i hi = _mm_loadl_epi64((const __m128i *)hb);

                __m128i q5 = _mm_or_si128(lo, hi);
                __m128i offset = _mm_set1_epi8(16);
                __m128i q5_signed = _mm_sub_epi8(q5, offset);

                __m256i q32 = _mm256_cvtepi8_epi32(q5_signed);
                __m256 wf = _mm256_cvtepi32_ps(q32);
                wf = _mm256_mul_ps(wf, vscale);

                __m256 xv = _mm256_loadu_ps(&xp[8]);
                acc = _mm256_fmadd_ps(wf, xv, acc);
            }

            /* Group 2: weights 16-23 (high nibbles of qs[0-7], high bits qh[16-23]) */
            {
                __m128i qs8 = _mm_loadl_epi64((const __m128i *)block->qs);
                __m128i hi_nib = _mm_and_si128(_mm_srli_epi16(qs8, 4), _mm_set1_epi8(0x0F));

                /* High bits for weights 16-23 come from qh bits 16-23 */
                int8_t hb[8];
                for (int i = 0; i < 8; i++) {
                    hb[i] = ((qh >> (16 + i)) & 1) << 4;
                }
                __m128i hi = _mm_loadl_epi64((const __m128i *)hb);

                __m128i q5 = _mm_or_si128(hi_nib, hi);
                __m128i offset = _mm_set1_epi8(16);
                __m128i q5_signed = _mm_sub_epi8(q5, offset);

                __m256i q32 = _mm256_cvtepi8_epi32(q5_signed);
                __m256 wf = _mm256_cvtepi32_ps(q32);
                wf = _mm256_mul_ps(wf, vscale);

                __m256 xv = _mm256_loadu_ps(&xp[16]);
                acc = _mm256_fmadd_ps(wf, xv, acc);
            }

            /* Group 3: weights 24-31 (high nibbles of qs[8-15], high bits qh[24-31]) */
            {
                __m128i qs8 = _mm_loadl_epi64((const __m128i *)(block->qs + 8));
                __m128i hi_nib = _mm_and_si128(_mm_srli_epi16(qs8, 4), _mm_set1_epi8(0x0F));

                int8_t hb[8];
                for (int i = 0; i < 8; i++) {
                    hb[i] = ((qh >> (24 + i)) & 1) << 4;
                }
                __m128i hi = _mm_loadl_epi64((const __m128i *)hb);

                __m128i q5 = _mm_or_si128(hi_nib, hi);
                __m128i offset = _mm_set1_epi8(16);
                __m128i q5_signed = _mm_sub_epi8(q5, offset);

                __m256i q32 = _mm256_cvtepi8_epi32(q5_signed);
                __m256 wf = _mm256_cvtepi32_ps(q32);
                wf = _mm256_mul_ps(wf, vscale);

                __m256 xv = _mm256_loadu_ps(&xp[24]);
                acc = _mm256_fmadd_ps(wf, xv, acc);
            }
        }

        y[row] = hsum_avx2(acc);
    }
}
#endif /* __AVX2__ && !__AVX512F__ */

/* ============================================================================
 * AVX Implementation with True SIMD Dequantization
 *
 * Q5_0 format: 32 weights per block
 *   - d: FP16 scale
 *   - qh: 4 bytes (32 high bits, one per weight)
 *   - qs: 16 bytes (low 4 bits, packed as pairs)
 *   - Dequant: w = d * ((lo | (highbit << 4)) - 16)
 *
 * This uses SSE for integer unpacking (Ivy Bridge doesn't have AVX2 for
 * 256-bit integer ops) and AVX for float accumulation.
 *
 * Key optimization: Instead of scalar dequant, we use SIMD to:
 *   1. Extract nibbles to bytes using SSE shuffle/shift
 *   2. Combine with high bits using SSE or/and
 *   3. Convert to float and scale
 * ============================================================================ */

#if defined(__AVX__) && !defined(__AVX2__) && !defined(__AVX512F__)

/* Helper: Extract low nibbles from 16 packed bytes to 16 bytes */
static inline __m128i extract_low_nibbles(__m128i packed) {
    return _mm_and_si128(packed, _mm_set1_epi8(0x0F));
}

/* Helper: Extract high nibbles from 16 packed bytes to 16 bytes */
static inline __m128i extract_high_nibbles(__m128i packed) {
    return _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(0x0F));
}

/* Helper: SSE horizontal sum of 4 floats */
static inline float hsum_sse(__m128 v) {
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

/* Helper: SSE dot product of 8 int8 values with 8 float values */
static inline float dot_int8_float8_sse(__m128i q8_lo, const float *x, float scale) {
    /* Sign-extend 8 int8s to 8 int32s (in two steps) */
    __m128i lo16 = _mm_cvtepi8_epi16(q8_lo);  /* 8 int8 -> 8 int16 */
    __m128i lo32_0 = _mm_cvtepi16_epi32(lo16);  /* low 4 int16 -> 4 int32 */
    __m128i lo32_1 = _mm_cvtepi16_epi32(_mm_srli_si128(lo16, 8));  /* high 4 int16 -> 4 int32 */

    /* Convert to float */
    __m128 w0 = _mm_cvtepi32_ps(lo32_0);
    __m128 w1 = _mm_cvtepi32_ps(lo32_1);

    /* Scale */
    __m128 vscale = _mm_set1_ps(scale);
    w0 = _mm_mul_ps(w0, vscale);
    w1 = _mm_mul_ps(w1, vscale);

    /* Load input and multiply */
    __m128 x0 = _mm_loadu_ps(x);
    __m128 x1 = _mm_loadu_ps(x + 4);

    __m128 prod0 = _mm_mul_ps(w0, x0);
    __m128 prod1 = _mm_mul_ps(w1, x1);

    /* Sum */
    __m128 sum = _mm_add_ps(prod0, prod1);
    return hsum_sse(sum);
}

/**
 * @brief Matrix-vector multiply with Q5_0 weights (AVX + SSE optimized)
 *
 * Uses SSE for integer dequantization, AVX for float accumulation.
 * ~3-5x faster than scalar reference on Ivy Bridge.
 */
void gemv_q5_0_avx(float *y,
                   const void *W,
                   const float *x,
                   int M, int K)
{
    const block_q5_0 *blocks = (const block_q5_0 *)W;
    const int blocks_per_row = K / QK5_0;  /* QK5_0 = 32 */

    const __m128i mask_0f = _mm_set1_epi8(0x0F);
    const __m128i mask_10 = _mm_set1_epi8(0x10);

    for (int row = 0; row < M; row++) {
        float sum = 0.0f;

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q5_0 *block = &blocks[row * blocks_per_row + b];
            const float d = CK_FP16_TO_FP32(block->d);
            const float *xp = &x[b * QK5_0];

            /* Load 16 packed bytes (32 nibbles) */
            __m128i qs = _mm_loadu_si128((const __m128i *)block->qs);

            /* Extract low and high nibbles */
            __m128i lo_nibbles = _mm_and_si128(qs, mask_0f);  /* 16 low nibbles */
            __m128i hi_nibbles = _mm_and_si128(_mm_srli_epi16(qs, 4), mask_0f);  /* 16 high nibbles */

            /* Get high bits as 32-bit integer */
            uint32_t qh;
            memcpy(&qh, block->qh, sizeof(qh));

            /* Q5_0 layout: weight j uses qs[j/2] nibble (low if j<16, high if j>=16)
             * plus high bit from qh:
             *   - weights 0-15: low nibbles of qs[0-15], high bit at qh[0-15]
             *   - weights 16-31: high nibbles of qs[0-15], high bit at qh[12-27]
             *
             * For efficiency, we process 32 weights in 4 groups of 8:
             */

            /* Group 0: weights 0-7 (low nibbles of qs[0-7], high bits from qh[0-7]) */
            {
                uint8_t w8[8];
                for (int i = 0; i < 8; i++) {
                    int lo = block->qs[i] & 0x0F;
                    int hb = ((qh >> i) << 4) & 0x10;
                    w8[i] = (lo | hb) - 16;  /* Signed -16 to +15 */
                }
                __m128i q8 = _mm_loadl_epi64((const __m128i *)w8);
                sum += dot_int8_float8_sse(q8, &xp[0], d);
            }

            /* Group 1: weights 8-15 (low nibbles of qs[8-15], high bits from qh[8-15]) */
            {
                uint8_t w8[8];
                for (int i = 0; i < 8; i++) {
                    int lo = block->qs[8 + i] & 0x0F;
                    int hb = ((qh >> (8 + i)) << 4) & 0x10;
                    w8[i] = (lo | hb) - 16;
                }
                __m128i q8 = _mm_loadl_epi64((const __m128i *)w8);
                sum += dot_int8_float8_sse(q8, &xp[8], d);
            }

            /* Group 2: weights 16-23 (high nibbles of qs[0-7], high bits from qh[12-19]) */
            {
                uint8_t w8[8];
                for (int i = 0; i < 8; i++) {
                    int hi = block->qs[i] >> 4;
                    int hb = (qh >> (12 + i)) & 0x10;
                    w8[i] = (hi | hb) - 16;
                }
                __m128i q8 = _mm_loadl_epi64((const __m128i *)w8);
                sum += dot_int8_float8_sse(q8, &xp[16], d);
            }

            /* Group 3: weights 24-31 (high nibbles of qs[8-15], high bits from qh[20-27]) */
            {
                uint8_t w8[8];
                for (int i = 0; i < 8; i++) {
                    int hi = block->qs[8 + i] >> 4;
                    int hb = (qh >> (20 + i)) & 0x10;
                    w8[i] = (hi | hb) - 16;
                }
                __m128i q8 = _mm_loadl_epi64((const __m128i *)w8);
                sum += dot_int8_float8_sse(q8, &xp[24], d);
            }
        }

        y[row] = sum;
    }
}
#endif /* __AVX__ && !__AVX512F__ */

/**
 * @brief Auto-dispatch GEMV for Q5_0 weights based on CPU features
 *
 * Dispatch priority (best available):
 *   1. AVX-512 (512-bit vectors) - Intel Skylake-X+
 *   2. AVX2+FMA (256-bit vectors) - Intel Haswell+
 *   3. AVX (256-bit vectors) - Intel Sandy Bridge+
 *   4. SSE4.1 (128-bit vectors) - Intel Nehalem+
 *   5. Reference (scalar) - Fallback
 *
 * Uses ck_features.h for standardized feature detection.
 *
 * @param y Output vector [M]
 * @param W Weight matrix in Q5_0 format [M x K]
 * @param x Input vector [K]
 * @param M Number of output rows
 * @param K Number of input columns (hidden dimension)
 */
void gemv_q5_0(float *y,
               const void *W,
               const float *x,
               int M, int K)
{
// Dispatch order: AVX512 > AVX2 > AVX > SSE > ref
#if defined(__AVX512F__)
    gemv_q5_0_avx512(y, W, x, M, K);
#elif defined(__AVX2__)
    gemv_q5_0_avx2(y, W, x, M, K);
#elif defined(__AVX__)
    gemv_q5_0_avx(y, W, x, M, K);
#elif defined(__SSE4_1__)
    gemv_q5_0_ref(y, W, x, M, K);
#else
    gemv_q5_0_ref(y, W, x, M, K);
#endif
}

/* ============================================================================
 * PARALLEL VERSIONS (for parallel orchestration)
 *
 * These receive ith (thread index) and nth (total threads) from orchestration.
 * OpenMP lives in orchestration layer, NOT here.
 * ============================================================================ */

/**
 * @brief Parallel reference GEMV for Q5_0 × FP32
 */
void gemv_q5_0_parallel(float *y,
                        const void *W,
                        const float *x,
                        int M, int K,
                        int ith, int nth)
{
    if (!y || !W || !x || M <= 0 || K <= 0) return;
    if (ith < 0 || nth <= 0 || ith >= nth) return;

    const int dr = (M + nth - 1) / nth;
    const int r0 = dr * ith;
    const int r1 = (r0 + dr < M) ? (r0 + dr) : M;

    if (r0 >= M) return;

    const block_q5_0 *blocks = (const block_q5_0 *)W;
    const int blocks_per_row = K / QK5_0;

    for (int row = r0; row < r1; row++) {
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            const block_q5_0 *block = &blocks[row * blocks_per_row + b];
            const float d = CK_FP16_TO_FP32(block->d);
            const float *xp = &x[b * QK5_0];

            uint32_t qh;
            memcpy(&qh, block->qh, sizeof(qh));

            for (int j = 0; j < QK5_0 / 2; j++) {
                const uint8_t packed = block->qs[j];
                const int lo = (packed & 0x0F);
                const int hi = (packed >> 4);
                const int xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
                const int xh_1 = ((qh >> (j + 12))) & 0x10;
                const int w0 = (lo | xh_0) - 16;
                const int w1 = (hi | xh_1) - 16;
                sum += d * (w0 * xp[j] + w1 * xp[j + QK5_0/2]);
            }
        }
        y[row] = sum;
    }
}

/**
 * @brief Parallel SIMD GEMV for Q5_0 × FP32 with prefetching
 */
void gemv_q5_0_parallel_simd(float *y,
                              const void *W,
                              const float *x,
                              int M, int K,
                              int ith, int nth)
{
    if (!y || !W || !x || M <= 0 || K <= 0) return;
    if (ith < 0 || nth <= 0 || ith >= nth) return;

    const int dr = (M + nth - 1) / nth;
    const int r0 = dr * ith;

    if (r0 >= M) return;

#if defined(__AVX__) || defined(__SSE4_1__)
    const int r1 = (r0 + dr < M) ? (r0 + dr) : M;
    const block_q5_0 *blocks = (const block_q5_0 *)W;
    const int blocks_per_row = K / QK5_0;
    /* Prefetch first few rows */
    const int PREFETCH_ROWS = 4;
    for (int p = 0; p < PREFETCH_ROWS && r0 + p < r1; ++p) {
        const char *row_ptr = (const char *)(blocks + (r0 + p) * blocks_per_row);
        _mm_prefetch(row_ptr, _MM_HINT_T0);
        _mm_prefetch(row_ptr + 64, _MM_HINT_T0);
    }

    for (int row = r0; row < r1; ++row) {
        /* Prefetch rows ahead */
        if (row + PREFETCH_ROWS < r1) {
            const char *prefetch_ptr = (const char *)(blocks + (row + PREFETCH_ROWS) * blocks_per_row);
            _mm_prefetch(prefetch_ptr, _MM_HINT_T0);
            _mm_prefetch(prefetch_ptr + 64, _MM_HINT_T0);
        }

        /* Use SIMD dot product for this row */
#if defined(__AVX512F__)
        /* Call single-row AVX512 implementation */
        gemv_q5_0_avx512(&y[row], (const char *)blocks + row * blocks_per_row * sizeof(block_q5_0), x, 1, K);
#elif defined(__AVX2__)
        gemv_q5_0_avx2(&y[row], (const char *)blocks + row * blocks_per_row * sizeof(block_q5_0), x, 1, K);
#elif defined(__AVX__)
        gemv_q5_0_avx(&y[row], (const char *)blocks + row * blocks_per_row * sizeof(block_q5_0), x, 1, K);
#else
        gemv_q5_0_ref(&y[row], (const char *)blocks + row * blocks_per_row * sizeof(block_q5_0), x, 1, K);
#endif
    }
#else
    /* Fallback to reference parallel */
    gemv_q5_0_parallel(y, W, x, M, K, ith, nth);
#endif
}

/* ============================================================================
 * Forward Pass: GEMM Y = W @ X
 * ============================================================================ */

/**
 * @brief Matrix-matrix multiply with Q5_0 weights
 */
void gemm_q5_0(float *Y,
               const void *W,
               const float *X,
               int M, int N, int K)
{
    for (int n = 0; n < N; n++) {
        gemv_q5_0(&Y[n * M], W, &X[n * K], M, K);
    }
}

/* ============================================================================
 * Backward Pass: Gradient w.r.t. Input
 * ============================================================================ */

/**
 * @brief Backward pass: compute input gradient
 *
 * @param dX Output gradient w.r.t. input [K]
 * @param W Weight matrix in Q5_0 format [M x K]
 * @param dY Gradient w.r.t. output [M]
 * @param M Number of output rows
 * @param K Number of columns (input dimension)
 */
void gemv_q5_0_backward_ref(float *dX,
                            const void *W,
                            const float *dY,
                            int M, int K)
{
    const block_q5_0 *blocks = (const block_q5_0 *)W;
    const int blocks_per_row = K / QK5_0;

    /* Zero output gradient */
    memset(dX, 0, K * sizeof(float));

    /* Accumulate: dX += W^T @ dY */
    for (int row = 0; row < M; row++) {
        const float dy = dY[row];

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q5_0 *block = &blocks[row * blocks_per_row + b];
            const float d = CK_FP16_TO_FP32(block->d);
            float *dxp = &dX[b * QK5_0];

            /* Get high bits */
            uint32_t qh;
            memcpy(&qh, block->qh, sizeof(qh));

            /* llama.cpp Q5_0 layout - note j+12 for second weight high bit */
            for (int j = 0; j < QK5_0 / 2; j++) {
                const uint8_t packed = block->qs[j];

                /* Extract and reconstruct 5-bit values */
                const int lo = (packed & 0x0F);
                const int hi = (packed >> 4);
                const int xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
                const int xh_1 = ((qh >> (j + 12))) & 0x10;
                const int q0 = (lo | xh_0) - 16;
                const int q1 = (hi | xh_1) - 16;

                dxp[j] += d * (float)q0 * dy;
                dxp[j + 16] += d * (float)q1 * dy;
            }
        }
    }
}

/**
 * @brief Auto-dispatch backward
 */
void gemv_q5_0_backward(float *dX,
                        const void *W,
                        const float *dY,
                        int M, int K)
{
    gemv_q5_0_backward_ref(dX, W, dY, M, K);
}

/**
 * @brief Batched backward pass
 */
void gemm_q5_0_backward(float *dX,
                        const void *W,
                        const float *dY,
                        int M, int N, int K)
{
    for (int n = 0; n < N; n++) {
        gemv_q5_0_backward(&dX[n * K], W, &dY[n * M], M, K);
    }
}

/* ============================================================================
 * GEMM NT (Non-Transpose A, Transpose B) - C = A @ B^T
 * For inference: A is activations [M x K], B is weights [N x K]
 * ============================================================================ */

/**
 * @brief GEMM with transposed Q5_0 weights: C = A @ B^T
 *
 * @param A Input activations [M x K], row-major FP32
 * @param B Weight matrix in Q5_0 format [N x K], row-major quantized
 * @param bias Optional bias [N], NULL if not used
 * @param C Output [M x N], row-major FP32
 * @param M Batch size (number of tokens)
 * @param N Output dimension (number of rows in B)
 * @param K Input dimension
 */
void gemm_nt_q5_0_ref(const float *A,
                      const void *B,
                      const float *bias,
                      float *C,
                      int M, int N, int K)
{
    const block_q5_0 *blocks = (const block_q5_0 *)B;
    const int blocks_per_row = K / QK5_0;

    for (int m = 0; m < M; m++) {
        const float *a_row = &A[m * K];

        for (int n = 0; n < N; n++) {
            float sum = 0.0f;

            for (int b = 0; b < blocks_per_row; b++) {
                const block_q5_0 *block = &blocks[n * blocks_per_row + b];
                const float d = CK_FP16_TO_FP32(block->d);
                const float *ap = &a_row[b * QK5_0];

                uint32_t qh;
                memcpy(&qh, block->qh, sizeof(qh));

                /* llama.cpp Q5_0 layout - note j+12 for second weight high bit */
                for (int j = 0; j < QK5_0 / 2; j++) {
                    const uint8_t packed = block->qs[j];
                    const int lo = (packed & 0x0F);
                    const int hi = (packed >> 4);
                    const int xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
                    const int xh_1 = ((qh >> (j + 12))) & 0x10;
                    const int q0 = (lo | xh_0) - 16;
                    const int q1 = (hi | xh_1) - 16;

                    sum += d * (float)q0 * ap[j];
                    sum += d * (float)q1 * ap[j + 16];
                }
            }

            C[m * N + n] = sum + (bias ? bias[n] : 0.0f);
        }
    }
}

void gemm_nt_q5_0(const float *A,
                  const void *B,
                  const float *bias,
                  float *C,
                  int M, int N, int K)
{
    /* For decode (M=1), use direct GEMV which has AVX optimization */
    if (M == 1) {
        /* gemm_q5_0 expects column-major output, but we need row-major
         * So we call gemv_q5_0 directly for each output element */
        gemv_q5_0(C, B, A, N, K);
        if (bias) {
            for (int n = 0; n < N; n++) {
                C[n] += bias[n];
            }
        }
        return;
    }

    /* For prefill (M>1), use GEMM which dispatches to GEMV with AVX/AVX512 */
    /* gemm_q5_0 produces Y as [batch x M_out]. Here:
     *   batch = M (tokens)
     *   M_out = N (output channels) */
    gemm_q5_0(C, B, A, /*M_out=*/N, /*N_batch=*/M, K);

    if (bias) {
        for (int m = 0; m < M; m++) {
            float *row = C + (size_t)m * (size_t)N;
            for (int n = 0; n < N; n++) {
                row[n] += bias[n];
            }
        }
    }
}

/* ============================================================================
 * Dot Product Utility
 * ============================================================================ */

float dot_q5_0(const void *w_q5_0, const float *x, int K)
{
    float result;
    gemv_q5_0(&result, w_q5_0, x, 1, K);
    return result;
}

/* ============================================================================
 * Quantized Dot Product: Q5_0 x Q8_0
 *
 * This matches llama.cpp's ggml_vec_dot_q5_0_q8_0 exactly.
 * Input is pre-quantized to Q8_0 format, enabling integer dot products.
 * Result: sum_blocks( (d_w * d_x) * sum_weights( w5 * x8 ) )
 *
 * Key difference from gemv_q5_0:
 *   - gemv_q5_0: Takes FP32 input, dequantizes weights to FP32, FP32 dot
 *   - vec_dot_q5_0_q8_0: Takes Q8_0 input, does integer dot, scales at end
 *
 * The quantized path is faster and matches llama.cpp for parity testing.
 * ============================================================================ */

/**
 * @brief Quantized dot product: Q5_0 weights x Q8_0 input (scalar reference)
 *
 * @param n Number of elements (must be multiple of 32)
 * @param s Output: scalar dot product result
 * @param vx Q5_0 quantized weights
 * @param vy Q8_0 quantized input
 */
void vec_dot_q5_0_q8_0_ref(int n, float *s, const void *vx, const void *vy)
{
    const int qk = QK5_0;  /* 32 */
    const int nb = n / qk;

    const block_q5_0 *x = (const block_q5_0 *)vx;
    const block_q8_0 *y = (const block_q8_0 *)vy;

    float sumf = 0.0f;

    for (int ib = 0; ib < nb; ib++) {
        /* Load high bits for this block */
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk / 2; j++) {
            /* Extract high bits - matches llama.cpp exactly */
            const uint8_t xh_0 = ((qh & (1u << (j + 0))) >> (j + 0)) << 4;
            const uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));

            /* Reconstruct 5-bit signed values (-16 to +15) */
            const int32_t x0 = (int8_t)(((x[ib].qs[j] & 0x0F) | xh_0) - 16);
            const int32_t x1 = (int8_t)(((x[ib].qs[j] >> 4) | xh_1) - 16);

            /* Integer dot product with Q8_0 values */
            sumi0 += x0 * y[ib].qs[j];
            sumi1 += x1 * y[ib].qs[j + qk / 2];
        }

        int sumi = sumi0 + sumi1;
        sumf += (CK_FP16_TO_FP32(x[ib].d) * CK_FP16_TO_FP32(y[ib].d)) * sumi;
    }

    *s = sumf;
}

#if defined(__ARM_NEON) || defined(__aarch64__)
static inline int32_t ck_hsum_s32x4(int32x4_t v)
{
    int32_t lanes[4];
    vst1q_s32(lanes, v);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}

void vec_dot_q5_0_q8_0_neon(int n, float *s, const void *vx, const void *vy)
{
    const int qk = QK5_0;
    const int nb = n / qk;

    const block_q5_0 *x = (const block_q5_0 *)vx;
    const block_q8_0 *y = (const block_q8_0 *)vy;

    float sumf = 0.0f;

    for (int ib = 0; ib < nb; ib++) {
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        int8_t wvals[QK5_0];
        for (int j = 0; j < qk / 2; j++) {
            const uint8_t packed = x[ib].qs[j];
            const uint8_t xh_0 = ((qh >> (j + 0)) & 1u) << 4;
            const uint8_t xh_1 = ((qh >> (j + 16)) & 1u) << 4;

            wvals[j] = (int8_t)(((packed & 0x0F) | xh_0) - 16);
            wvals[j + qk / 2] = (int8_t)(((packed >> 4) | xh_1) - 16);
        }

        const int8x16_t w0 = vld1q_s8(&wvals[0]);
        const int8x16_t w1 = vld1q_s8(&wvals[16]);
        const int8x16_t x0 = vld1q_s8(&y[ib].qs[0]);
        const int8x16_t x1 = vld1q_s8(&y[ib].qs[16]);

        int32x4_t acc = vdupq_n_s32(0);

        int16x8_t p0 = vmull_s8(vget_low_s8(w0), vget_low_s8(x0));
        int16x8_t p1 = vmull_s8(vget_high_s8(w0), vget_high_s8(x0));
        int16x8_t p2 = vmull_s8(vget_low_s8(w1), vget_low_s8(x1));
        int16x8_t p3 = vmull_s8(vget_high_s8(w1), vget_high_s8(x1));

        acc = vaddq_s32(acc, vpaddlq_s16(p0));
        acc = vaddq_s32(acc, vpaddlq_s16(p1));
        acc = vaddq_s32(acc, vpaddlq_s16(p2));
        acc = vaddq_s32(acc, vpaddlq_s16(p3));

        const float d = CK_FP16_TO_FP32(x[ib].d) * CK_FP16_TO_FP32(y[ib].d);
        sumf += d * (float)ck_hsum_s32x4(acc);
    }

    *s = sumf;
}

#if defined(__ARM_FEATURE_DOTPROD)
/* Q5_0 x Q8_0 dot using the AArch64 dot-product extension (SDOT). The 5-bit
 * weights are reconstructed entirely in vector form: low nibbles via
 * AND/shift, the 5th bit spread from qh with vdup_lane/vtstq (bit j of qh is
 * tested into byte lane j, matching the scalar reference layout), and the
 * -16 offset folded into the int8 lanes, so vdotq_s32 sees plain signed
 * int8 x int8. Only base-NEON intrinsics are used alongside vdotq_s32, so
 * nothing here is AArch64-only beyond the dot-product gate itself. Integer
 * arithmetic matches the scalar reference exactly; the FP32 per-block
 * epilogue is unchanged, so results are bit-identical to
 * vec_dot_q5_0_q8_0_ref.
 *
 * Gated on __ARM_FEATURE_DOTPROD rather than plain NEON: measured on DGX
 * Spark (GB10, Cortex-X925), the plain-NEON kernel above is 2.5-3.3x SLOWER
 * than the compiler's autovectorization of the scalar reference (its
 * per-block scalar unpack dominates), matching the Q4_K precedent, so
 * baseline armv8-a builds dispatch to the reference instead. The plain-NEON
 * kernel is kept compiled for the parity test's bit-exactness check. */
void vec_dot_q5_0_q8_0_neon_sdot(int n, float *s, const void *vx, const void *vy)
{
    const int qk = QK5_0;
    const int nb = n / qk;

    const block_q5_0 *x = (const block_q5_0 *)vx;
    const block_q8_0 *y = (const block_q8_0 *)vy;

    float sumf = 0.0f;

    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const uint8x16_t m10 = vdupq_n_u8(0x10);
    const int8x16_t off = vdupq_n_s8(16);
    /* Lane j selects bit (j % 8) of the replicated qh byte. */
    const uint8x16_t bits = {1, 2, 4, 8, 16, 32, 64, 128,
                             1, 2, 4, 8, 16, 32, 64, 128};

    for (int ib = 0; ib < nb; ib++) {
        uint32_t qh32;
        memcpy(&qh32, x[ib].qh, sizeof(qh32));
        const uint8x8_t qh8 = vcreate_u8((uint64_t)qh32);

        /* Spread bit j of qh to 0x10 in byte lane j: weights 0-15 use bits
         * 0-15 (qh bytes 0-1), weights 16-31 use bits 16-31 (qh bytes 2-3). */
        const uint8x16_t hb_lo = vandq_u8(
            vtstq_u8(vcombine_u8(vdup_lane_u8(qh8, 0), vdup_lane_u8(qh8, 1)),
                     bits), m10);
        const uint8x16_t hb_hi = vandq_u8(
            vtstq_u8(vcombine_u8(vdup_lane_u8(qh8, 2), vdup_lane_u8(qh8, 3)),
                     bits), m10);

        const uint8x16_t qs = vld1q_u8(x[ib].qs);
        const int8x16_t q5_lo = vsubq_s8(
            vreinterpretq_s8_u8(vorrq_u8(vandq_u8(qs, m4), hb_lo)), off);
        const int8x16_t q5_hi = vsubq_s8(
            vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(qs, 4), hb_hi)), off);

        int32x4_t acc = vdotq_s32(vdupq_n_s32(0), q5_lo,
                                  vld1q_s8(&y[ib].qs[0]));
        acc = vdotq_s32(acc, q5_hi, vld1q_s8(&y[ib].qs[16]));

        const float d = CK_FP16_TO_FP32(x[ib].d) * CK_FP16_TO_FP32(y[ib].d);
        sumf += d * (float)ck_hsum_s32x4(acc);
    }

    *s = sumf;
}
#endif /* __ARM_FEATURE_DOTPROD */
#endif

#ifdef __AVX512F__
/**
 * @brief Quantized dot product Q5_0 x Q8_0 (AVX-512)
 */
void vec_dot_q5_0_q8_0_avx512(int n, float *s, const void *vx, const void *vy)
{
    const int qk = QK5_0;
    const int nb = n / qk;

    const block_q5_0 *x = (const block_q5_0 *)vx;
    const block_q8_0 *y = (const block_q8_0 *)vy;

    float sumf = 0.0f;

    for (int ib = 0; ib < nb; ib++) {
        const float d = CK_FP16_TO_FP32(x[ib].d) * CK_FP16_TO_FP32(y[ib].d);

        /* Load high bits */
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        /* Load 16 packed bytes (32 nibbles) */
        __m128i qs = _mm_loadu_si128((const __m128i *)x[ib].qs);

        /* Process first 16 weights (low nibbles, high bits 0-15) */
        __m512i lo_nibbles = _mm512_cvtepu8_epi32(qs);
        lo_nibbles = _mm512_and_epi32(lo_nibbles, _mm512_set1_epi32(0x0F));

        /* Build high bit contribution for first 16 weights */
        __m512i qh_lo = _mm512_set_epi32(
            ((qh >> 15) & 1) << 4, ((qh >> 14) & 1) << 4,
            ((qh >> 13) & 1) << 4, ((qh >> 12) & 1) << 4,
            ((qh >> 11) & 1) << 4, ((qh >> 10) & 1) << 4,
            ((qh >> 9) & 1) << 4, ((qh >> 8) & 1) << 4,
            ((qh >> 7) & 1) << 4, ((qh >> 6) & 1) << 4,
            ((qh >> 5) & 1) << 4, ((qh >> 4) & 1) << 4,
            ((qh >> 3) & 1) << 4, ((qh >> 2) & 1) << 4,
            ((qh >> 1) & 1) << 4, ((qh >> 0) & 1) << 4
        );

        /* Combine and subtract 16 to get signed values */
        __m512i q5_lo = _mm512_sub_epi32(_mm512_or_epi32(lo_nibbles, qh_lo),
                                          _mm512_set1_epi32(16));

        /* Load Q8_0 values for first 16 */
        __m128i y8_lo = _mm_loadu_si128((const __m128i *)&y[ib].qs[0]);
        __m512i y32_lo = _mm512_cvtepi8_epi32(y8_lo);

        /* Integer multiply and accumulate */
        __m512i prod_lo = _mm512_mullo_epi32(q5_lo, y32_lo);

        /* Process second 16 weights (high nibbles, high bits 16-31 via j+12 mapping) */
        __m512i hi_nibbles = _mm512_cvtepu8_epi32(qs);
        hi_nibbles = _mm512_srli_epi32(hi_nibbles, 4);

        /* Build high bit contribution for second 16 weights (bits 16-31) */
        __m512i qh_hi = _mm512_set_epi32(
            ((qh >> 31) & 1) << 4, ((qh >> 30) & 1) << 4,
            ((qh >> 29) & 1) << 4, ((qh >> 28) & 1) << 4,
            ((qh >> 27) & 1) << 4, ((qh >> 26) & 1) << 4,
            ((qh >> 25) & 1) << 4, ((qh >> 24) & 1) << 4,
            ((qh >> 23) & 1) << 4, ((qh >> 22) & 1) << 4,
            ((qh >> 21) & 1) << 4, ((qh >> 20) & 1) << 4,
            ((qh >> 19) & 1) << 4, ((qh >> 18) & 1) << 4,
            ((qh >> 17) & 1) << 4, ((qh >> 16) & 1) << 4
        );

        __m512i q5_hi = _mm512_sub_epi32(_mm512_or_epi32(hi_nibbles, qh_hi),
                                          _mm512_set1_epi32(16));

        /* Load Q8_0 values for second 16 */
        __m128i y8_hi = _mm_loadu_si128((const __m128i *)&y[ib].qs[16]);
        __m512i y32_hi = _mm512_cvtepi8_epi32(y8_hi);

        __m512i prod_hi = _mm512_mullo_epi32(q5_hi, y32_hi);

        /* Sum all products */
        int sumi = _mm512_reduce_add_epi32(_mm512_add_epi32(prod_lo, prod_hi));

        /* Scale and accumulate - use scalar to avoid 16x broadcast bug */
        sumf += d * (float)sumi;
    }

    *s = sumf;
}
#endif

#if defined(__AVX2__) || defined(CK_Q5_RUNTIME_AVX2)
CK_Q5_TARGET_AVX static inline float hsum_float_8_avx(const __m256 x);

CK_Q5_TARGET_AVX2
static inline __m256i unpack_q5_0_avx2(const block_q5_0 *block)
{
    uint32_t high_bits;
    memcpy(&high_bits, block->qh, sizeof(high_bits));
    const __m256i byte_index = _mm256_set_epi64x(
        0x0303030303030303LL, 0x0202020202020202LL,
        0x0101010101010101LL, 0x0000000000000000LL);
    const __m256i bit_test = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfeLL);
    const __m256i bit_bytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int)high_bits), byte_index);
    const __m256i high_set = _mm256_cmpeq_epi8(_mm256_or_si256(bit_bytes, bit_test),
                                              _mm256_set1_epi8(-1));
    const __m128i packed = _mm_loadu_si128((const __m128i *)block->qs);
    const __m256i nibbles = _mm256_set_m128i(_mm_srli_epi16(packed, 4), packed);
    return _mm256_or_si256(_mm256_and_si256(nibbles, _mm256_set1_epi8(0x0f)),
                           _mm256_andnot_si256(high_set, _mm256_set1_epi8((char)0xf0)));
}

CK_Q5_TARGET_AVX2
static inline __m256 q5_0_q8_0_accumulate_avx2(__m256 accum, __m256i q5,
                                                float weight_scale, const block_q8_0 *x)
{
#if defined(CK_Q5_RUNTIME_AVX2)
    const __m256 d = _mm256_set1_ps(weight_scale * _cvtsh_ss(x->d));
#else
    const __m256 d = _mm256_set1_ps(weight_scale * CK_FP16_TO_FP32(x->d));
#endif
    const __m256i q8 = _mm256_loadu_si256((const __m256i *)x->qs);
    const __m256i products = _mm256_maddubs_epi16(
        _mm256_abs_epi8(q5), _mm256_sign_epi8(q8, q5));
    const __m256 q = _mm256_cvtepi32_ps(
        _mm256_madd_epi16(products, _mm256_set1_epi16(1)));
#if defined(__FMA__) || defined(CK_Q5_RUNTIME_AVX2)
    return _mm256_fmadd_ps(d, q, accum);
#else
    return _mm256_add_ps(_mm256_mul_ps(d, q), accum);
#endif
}

/**
 * @brief Quantized dot product Q5_0 x Q8_0 (AVX2/FMA)
 *
 * Mirrors the local llama.cpp AVX2 structure on machines like this one: unpack
 * Q5_0 into signed int8 lanes, multiply against Q8_0 int8 lanes, and fuse the
 * per-block scale into the FP32 accumulator.
 */
CK_Q5_TARGET_AVX2
void vec_dot_q5_0_q8_0_avx2(int n, float *s, const void *vx, const void *vy)
{
    const int qk = QK5_0;
    const int nb = n / qk;

    const block_q5_0 *x = (const block_q5_0 *)vx;
    const block_q8_0 *y = (const block_q8_0 *)vy;

    __m256 acc = _mm256_setzero_ps();

    for (int ib = 0; ib < nb; ++ib) {
#if defined(CK_Q5_RUNTIME_AVX2)
        const float scale = _cvtsh_ss(x[ib].d);
#else
        const float scale = CK_FP16_TO_FP32(x[ib].d);
#endif
        acc = q5_0_q8_0_accumulate_avx2(acc, unpack_q5_0_avx2(x + ib), scale, y + ib);
    }

    *s = hsum_float_8_avx(acc);
}
#endif

#if defined(__SSSE3__)
/**
 * @brief Spread 32 bits to 32 bytes { 0x00, 0xFF }
 * Adapted from llama.cpp bytes_from_bits_32 (AVX path)
 *
 * Uses shuffle to replicate each byte, then OR with bit_mask and compare.
 * Result: 0xFF where bit was set, 0x00 where bit was not set.
 */
static inline void bytes_from_bits_32_sse(__m128i *out_lo, __m128i *out_hi, const uint8_t *qh)
{
    uint32_t x32;
    memcpy(&x32, qh, sizeof(uint32_t));

    /* Shuffle masks: replicate byte j/8 of x32 to each position */
    const __m128i shuf_maskl = _mm_set_epi64x(0x0101010101010101LL, 0x0000000000000000LL);
    const __m128i shuf_maskh = _mm_set_epi64x(0x0303030303030303LL, 0x0202020202020202LL);

    __m128i bytes_lo = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskl);
    __m128i bytes_hi = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskh);

    /* Bit mask: pattern tests each bit position 0-7 within each byte.
     * 0x7fbfdfeff7fbfdfe in binary has bits 1,2,3,4,5,6,7,0 cleared per 8-byte cycle.
     * After OR, byte will be 0xFF if the corresponding bit was set. */
    const __m128i bit_mask = _mm_set1_epi64x(0x7fbfdfeff7fbfdfeLL);

    bytes_lo = _mm_or_si128(bytes_lo, bit_mask);
    bytes_hi = _mm_or_si128(bytes_hi, bit_mask);

    /* Compare with all 1s: 0xFF if bit was set, 0x00 if not */
    *out_lo = _mm_cmpeq_epi8(bytes_lo, _mm_set1_epi64x(-1LL));
    *out_hi = _mm_cmpeq_epi8(bytes_hi, _mm_set1_epi64x(-1LL));
}

/**
 * @brief Multiply signed int8 vectors using sign trick
 * Adapted from llama.cpp mul_sum_i8_pairs
 *
 * Uses: abs(x) * sign(y,x) = x * y for signed multiplication with maddubs
 */
static inline __m128i mul_sum_i8_pairs_sse(const __m128i x, const __m128i y)
{
    const __m128i ax = _mm_sign_epi8(x, x);   /* abs(x) */
    const __m128i sy = _mm_sign_epi8(y, x);   /* y * sign(x) */
    const __m128i dot = _mm_maddubs_epi16(ax, sy);  /* unsigned*signed pairs -> int16 */
    return _mm_madd_epi16(dot, _mm_set1_epi16(1));  /* sum pairs -> int32 */
}

/**
 * @brief Vectorized dot product Q5_0 x Q8_0 using SSSE3
 *
 * Based on llama.cpp ggml_vec_dot_q5_0_q8_0 AVX implementation.
 * Key insight: use shuffle-based bit spreading and sign trick.
 *
 * Q5_0 encoding: nibble | (high_bit ? 0 : 0xF0)
 * - When high bit SET: value = nibble (0-15, positive as signed)
 * - When high bit NOT SET: value = nibble | 0xF0 (negative as signed, -16 to -1)
 *
 * Sign trick handles signed*signed multiplication with unsigned*signed maddubs.
 */
void vec_dot_q5_0_q8_0_sse(int n, float *s, const void *vx, const void *vy)
{
    const int qk = QK5_0;  /* 32 */
    const int nb = n / qk;

    const block_q5_0 *x = (const block_q5_0 *)vx;
    const block_q8_0 *y = (const block_q8_0 *)vy;

    float sumf = 0.0f;

    const __m128i mask_0f = _mm_set1_epi8(0x0F);
    const __m128i mask_f0 = _mm_set1_epi8((char)0xF0);

    for (int ib = 0; ib < nb; ib++) {
        const float d = CK_FP16_TO_FP32(x[ib].d) * CK_FP16_TO_FP32(y[ib].d);

        /* Load 16 bytes of packed nibbles */
        __m128i qs = _mm_loadu_si128((const __m128i *)x[ib].qs);

        /* Extract nibbles: lo for indices 0-15, hi for indices 16-31 */
        __m128i bx_lo = _mm_and_si128(qs, mask_0f);
        __m128i bx_hi = _mm_and_si128(_mm_srli_epi16(qs, 4), mask_0f);

        /* Spread 32 high bits to 32 bytes (0xFF=set, 0x00=not set) */
        __m128i bxhi_lo, bxhi_hi;
        bytes_from_bits_32_sse(&bxhi_lo, &bxhi_hi, x[ib].qh);

        /* Apply encoding: (~bxhi) & 0xF0
         * When bit SET: bxhi=0xFF, result = 0x00 (value is positive 0-15)
         * When bit NOT SET: bxhi=0x00, result = 0xF0 (value is negative) */
        bxhi_lo = _mm_andnot_si128(bxhi_lo, mask_f0);
        bxhi_hi = _mm_andnot_si128(bxhi_hi, mask_f0);

        /* Combine: nibble | high_bit_contribution */
        bx_lo = _mm_or_si128(bx_lo, bxhi_lo);
        bx_hi = _mm_or_si128(bx_hi, bxhi_hi);

        /* Load Q8_0 values (32 signed int8) */
        __m128i by_lo = _mm_loadu_si128((const __m128i *)y[ib].qs);
        __m128i by_hi = _mm_loadu_si128((const __m128i *)(y[ib].qs + 16));

        /* Multiply using sign trick and sum to int32 */
        __m128i p_lo = mul_sum_i8_pairs_sse(bx_lo, by_lo);
        __m128i p_hi = mul_sum_i8_pairs_sse(bx_hi, by_hi);

        /* Sum the two halves */
        __m128i sum = _mm_add_epi32(p_lo, p_hi);

        /* Horizontal sum of 4 int32 values (avoiding hadd for better perf) */
        __m128i hi64 = _mm_unpackhi_epi64(sum, sum);
        __m128i sum64 = _mm_add_epi32(hi64, sum);
        __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
        int32_t sumi = _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));

        /* Scale and accumulate */
        sumf += d * (float)sumi;
    }

    *s = sumf;
}
#endif

#if defined(__AVX__) || defined(CK_Q5_RUNTIME_AVX2)

#if defined(__AVX__)
/* Combine two __m128i into __m256i (AVX without AVX2) */
#define MM256_SET_M128I(hi, lo) _mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 1)

/**
 * @brief Spread 32 bits to 32 bytes using AVX
 * Returns __m256i with 0xFF where bit was set, 0x00 where not
 */
CK_Q5_TARGET_AVX static inline __m256i bytes_from_bits_32_avx(const uint8_t *qh)
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
 * @brief Unpack 32 4-bit nibbles to 32 bytes using AVX
 */
CK_Q5_TARGET_AVX static inline __m256i bytes_from_nibbles_32_avx(const uint8_t *qs)
{
    __m128i tmpl = _mm_loadu_si128((const __m128i *)qs);
    __m128i tmph = _mm_srli_epi16(tmpl, 4);
    const __m128i lowMask = _mm_set1_epi8(0x0F);
    tmpl = _mm_and_si128(lowMask, tmpl);
    tmph = _mm_and_si128(lowMask, tmph);
    return MM256_SET_M128I(tmph, tmpl);
}

/**
 * @brief Multiply signed int8 pairs and return as float vector (AVX)
 * Uses 128-bit ops internally but returns 256-bit float result
 */
CK_Q5_TARGET_AVX static inline __m256 mul_sum_i8_pairs_float_avx(const __m256i x, const __m256i y)
{
    const __m128i xl = _mm256_castsi256_si128(x);
    const __m128i xh = _mm256_extractf128_si256(x, 1);
    const __m128i yl = _mm256_castsi256_si128(y);
    const __m128i yh = _mm256_extractf128_si256(y, 1);

    /* Get absolute values of x vectors */
    const __m128i axl = _mm_sign_epi8(xl, xl);
    const __m128i axh = _mm_sign_epi8(xh, xh);
    /* Sign the values of the y vectors */
    const __m128i syl = _mm_sign_epi8(yl, xl);
    const __m128i syh = _mm_sign_epi8(yh, xh);

    /* Perform multiplication and create 16-bit values */
    const __m128i dotl = _mm_maddubs_epi16(axl, syl);
    const __m128i doth = _mm_maddubs_epi16(axh, syh);

    /* Sum pairs to int32 */
    const __m128i ones = _mm_set1_epi16(1);
    const __m128i summed_pairsl = _mm_madd_epi16(ones, dotl);
    const __m128i summed_pairsh = _mm_madd_epi16(ones, doth);

    /* Convert to float */
    const __m256i summed_pairs = MM256_SET_M128I(summed_pairsh, summed_pairsl);
    return _mm256_cvtepi32_ps(summed_pairs);
}

/**
 * @brief Horizontal sum of 8 floats (AVX)
 */
#endif

CK_Q5_TARGET_AVX static inline float hsum_float_8_avx(const __m256 x)
{
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

#endif

#if defined(__AVX__)
/**
 * @brief Quantized dot product Q5_0 x Q8_0 (AVX) - Optimized with 2x unroll
 *
 * Based on llama.cpp ggml_vec_dot_q5_0_q8_0 AVX implementation.
 * Uses 256-bit accumulation and processes 32 values per block.
 *
 * Optimizations:
 *   - 2x loop unrolling to reduce loop overhead
 *   - Prefetching next blocks to hide memory latency
 *   - Interleaved operations for better instruction-level parallelism
 */
void vec_dot_q5_0_q8_0_avx(int n, float *s, const void *vx, const void *vy)
{
    const int qk = QK5_0;  /* 32 */
    const int nb = n / qk;

    const block_q5_0 *x = (const block_q5_0 *)vx;
    const block_q8_0 *y = (const block_q8_0 *)vy;

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    const __m128i mask = _mm_set1_epi8((char)0xF0);

    /* Process 2 blocks per iteration */
    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        /* Prefetch next blocks (2 cache lines ahead) */
        _mm_prefetch((const char *)&x[ib + 4], _MM_HINT_T0);
        _mm_prefetch((const char *)&y[ib + 4], _MM_HINT_T0);

        /* === Block 0 === */
        const __m256 d0 = _mm256_set1_ps(CK_FP16_TO_FP32(x[ib].d) * CK_FP16_TO_FP32(y[ib].d));

        /* Unpack nibbles to 32 bytes */
        __m256i bx0 = bytes_from_nibbles_32_avx(x[ib].qs);

        /* Spread high bits */
        const __m256i bxhi0 = bytes_from_bits_32_avx(x[ib].qh);
        __m128i bxhil0 = _mm256_castsi256_si128(bxhi0);
        __m128i bxhih0 = _mm256_extractf128_si256(bxhi0, 1);

        /* === Block 1 === (start while block 0 is in flight) */
        const __m256 d1 = _mm256_set1_ps(CK_FP16_TO_FP32(x[ib+1].d) * CK_FP16_TO_FP32(y[ib+1].d));

        __m256i bx1 = bytes_from_nibbles_32_avx(x[ib+1].qs);
        const __m256i bxhi1 = bytes_from_bits_32_avx(x[ib+1].qh);
        __m128i bxhil1 = _mm256_castsi256_si128(bxhi1);
        __m128i bxhih1 = _mm256_extractf128_si256(bxhi1, 1);

        /* === Finish Block 0 === */
        bxhil0 = _mm_andnot_si128(bxhil0, mask);
        bxhih0 = _mm_andnot_si128(bxhih0, mask);

        __m128i bxl0 = _mm256_castsi256_si128(bx0);
        __m128i bxh0 = _mm256_extractf128_si256(bx0, 1);
        bxl0 = _mm_or_si128(bxl0, bxhil0);
        bxh0 = _mm_or_si128(bxh0, bxhih0);
        bx0 = MM256_SET_M128I(bxh0, bxl0);

        const __m256i by0 = _mm256_loadu_si256((const __m256i *)y[ib].qs);
        const __m256 q0 = mul_sum_i8_pairs_float_avx(bx0, by0);
        acc0 = _mm256_add_ps(_mm256_mul_ps(d0, q0), acc0);

        /* === Finish Block 1 === */
        bxhil1 = _mm_andnot_si128(bxhil1, mask);
        bxhih1 = _mm_andnot_si128(bxhih1, mask);

        __m128i bxl1 = _mm256_castsi256_si128(bx1);
        __m128i bxh1 = _mm256_extractf128_si256(bx1, 1);
        bxl1 = _mm_or_si128(bxl1, bxhil1);
        bxh1 = _mm_or_si128(bxh1, bxhih1);
        bx1 = MM256_SET_M128I(bxh1, bxl1);

        const __m256i by1 = _mm256_loadu_si256((const __m256i *)y[ib+1].qs);
        const __m256 q1 = mul_sum_i8_pairs_float_avx(bx1, by1);
        acc1 = _mm256_add_ps(_mm256_mul_ps(d1, q1), acc1);
    }

    /* Handle remaining block if nb is odd */
    for (; ib < nb; ib++) {
        const __m256 d = _mm256_set1_ps(CK_FP16_TO_FP32(x[ib].d) * CK_FP16_TO_FP32(y[ib].d));

        __m256i bx_0 = bytes_from_nibbles_32_avx(x[ib].qs);
        const __m256i bxhi = bytes_from_bits_32_avx(x[ib].qh);
        __m128i bxhil = _mm256_castsi256_si128(bxhi);
        __m128i bxhih = _mm256_extractf128_si256(bxhi, 1);

        bxhil = _mm_andnot_si128(bxhil, mask);
        bxhih = _mm_andnot_si128(bxhih, mask);

        __m128i bxl = _mm256_castsi256_si128(bx_0);
        __m128i bxh = _mm256_extractf128_si256(bx_0, 1);
        bxl = _mm_or_si128(bxl, bxhil);
        bxh = _mm_or_si128(bxh, bxhih);
        bx_0 = MM256_SET_M128I(bxh, bxl);

        const __m256i by_0 = _mm256_loadu_si256((const __m256i *)y[ib].qs);
        const __m256 q = mul_sum_i8_pairs_float_avx(bx_0, by_0);
        acc0 = _mm256_add_ps(_mm256_mul_ps(d, q), acc0);
    }

    /* Combine accumulators and sum */
    acc0 = _mm256_add_ps(acc0, acc1);
    *s = hsum_float_8_avx(acc0);
}

/* ============================================================================
 * Unrolled GEMM: C[M,N] = A_q8[M,K] @ B_q5[N,K]^T + bias
 *
 * Key optimizations over the naive vec_dot-per-element approach:
 *   1. N-dimension 2x unroll: loads activation block ONCE, reuses for 2 weight rows
 *   2. Inlined Q5_0 unpack: eliminates function call overhead per block
 *   3. Block-level prefetching: hides DRAM latency for weight streaming
 *   4. F16C scale conversion: uses hardware vcvtsh2ss via CK_FP16_TO_FP32 macro
 *   5. Persistent __m256 accumulators: no per-block horizontal sum
 *
 * Expected speedup: 1.5-2x over naive dispatch loop on AVX (Ivy Bridge+)
 * ============================================================================ */

/**
 * @brief Batch GEMM with Q5_0 weights x Q8_0 activations — AVX unrolled
 *
 * @param A_q8   Input activations in Q8_0 format [M rows of K/32 blocks each]
 * @param B_q5   Weights in Q5_0 format [N rows of K/32 blocks each]
 * @param bias   Optional bias vector [N], NULL if not used
 * @param C      Output matrix [M x N], row-major FP32
 * @param M      Batch size (number of tokens)
 * @param N      Output dimension (number of output features)
 * @param K      Input dimension (must be multiple of 32)
 */
void gemm_nt_q5_0_q8_0_unroll_avx(
    const void *A_q8,
    const void *B_q5,
    const float *bias,
    float *C,
    int M, int N, int K)
{
    const int nb = K / QK5_0;
    const block_q8_0 *a_blocks = (const block_q8_0 *)A_q8;
    const block_q5_0 *b_blocks = (const block_q5_0 *)B_q5;
    const __m128i mask = _mm_set1_epi8((char)0xF0);

    for (int m = 0; m < M; m++) {
        const block_q8_0 *a_row = a_blocks + (size_t)m * nb;

        /* ---- 2x N-unrolled path: process 2 weight rows per iteration ---- */
        int n = 0;
        for (; n + 1 < N; n += 2) {
            const block_q5_0 *w0 = b_blocks + (size_t)(n + 0) * nb;
            const block_q5_0 *w1 = b_blocks + (size_t)(n + 1) * nb;

            __m256 acc_n0 = _mm256_setzero_ps();
            __m256 acc_n1 = _mm256_setzero_ps();

            for (int ib = 0; ib < nb; ib++) {
                /* Prefetch next weight blocks (1 cache line = ~2.9 Q5_0 blocks) */
                if (ib + 2 < nb) {
                    _mm_prefetch((const char *)&w0[ib + 2], _MM_HINT_T0);
                    _mm_prefetch((const char *)&w1[ib + 2], _MM_HINT_T0);
                }

                /* Load activation Q8_0 block ONCE — reused for both weight rows */
                const __m256i by = _mm256_loadu_si256((const __m256i *)a_row[ib].qs);
                const float da = CK_FP16_TO_FP32(a_row[ib].d);

                /* === Weight row 0: inline Q5_0 unpack + dot product === */
                {
                    const float d = da * CK_FP16_TO_FP32(w0[ib].d);

                    /* Unpack low 4-bit nibbles → 32 bytes */
                    __m256i bx = bytes_from_nibbles_32_avx(w0[ib].qs);

                    /* Spread 32 high bits → 32 byte mask (0xFF or 0x00) */
                    const __m256i bxhi = bytes_from_bits_32_avx(w0[ib].qh);
                    __m128i bxhil = _mm256_castsi256_si128(bxhi);
                    __m128i bxhih = _mm256_extractf128_si256(bxhi, 1);

                    /* Combine: nibble |= (highbit ? 0x10 : 0x00) */
                    bxhil = _mm_andnot_si128(bxhil, mask);
                    bxhih = _mm_andnot_si128(bxhih, mask);
                    __m128i bxl = _mm256_castsi256_si128(bx);
                    __m128i bxh = _mm256_extractf128_si256(bx, 1);
                    bxl = _mm_or_si128(bxl, bxhil);
                    bxh = _mm_or_si128(bxh, bxhih);
                    bx = MM256_SET_M128I(bxh, bxl);

                    /* Signed int8 dot product → 8 float partial sums */
                    const __m256 q = mul_sum_i8_pairs_float_avx(bx, by);
                    acc_n0 = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), q), acc_n0);
                }

                /* === Weight row 1: same activation data, different weights === */
                {
                    const float d = da * CK_FP16_TO_FP32(w1[ib].d);

                    __m256i bx = bytes_from_nibbles_32_avx(w1[ib].qs);
                    const __m256i bxhi = bytes_from_bits_32_avx(w1[ib].qh);
                    __m128i bxhil = _mm256_castsi256_si128(bxhi);
                    __m128i bxhih = _mm256_extractf128_si256(bxhi, 1);

                    bxhil = _mm_andnot_si128(bxhil, mask);
                    bxhih = _mm_andnot_si128(bxhih, mask);
                    __m128i bxl = _mm256_castsi256_si128(bx);
                    __m128i bxh = _mm256_extractf128_si256(bx, 1);
                    bxl = _mm_or_si128(bxl, bxhil);
                    bxh = _mm_or_si128(bxh, bxhih);
                    bx = MM256_SET_M128I(bxh, bxl);

                    const __m256 q = mul_sum_i8_pairs_float_avx(bx, by);
                    acc_n1 = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), q), acc_n1);
                }
            }

            /* Horizontal reduce and store with optional bias */
            float s0 = hsum_float_8_avx(acc_n0);
            float s1 = hsum_float_8_avx(acc_n1);
            if (bias) { s0 += bias[n]; s1 += bias[n + 1]; }
            C[(size_t)m * N + n]     = s0;
            C[(size_t)m * N + n + 1] = s1;
        }

        /* ---- Cleanup: remaining odd column ---- */
        for (; n < N; n++) {
            const block_q5_0 *w = b_blocks + (size_t)n * nb;
            __m256 acc = _mm256_setzero_ps();

            for (int ib = 0; ib < nb; ib++) {
                const __m256i by = _mm256_loadu_si256((const __m256i *)a_row[ib].qs);
                const float d = CK_FP16_TO_FP32(a_row[ib].d) * CK_FP16_TO_FP32(w[ib].d);

                __m256i bx = bytes_from_nibbles_32_avx(w[ib].qs);
                const __m256i bxhi = bytes_from_bits_32_avx(w[ib].qh);
                __m128i bxhil = _mm256_castsi256_si128(bxhi);
                __m128i bxhih = _mm256_extractf128_si256(bxhi, 1);
                bxhil = _mm_andnot_si128(bxhil, mask);
                bxhih = _mm_andnot_si128(bxhih, mask);
                __m128i bxl = _mm256_castsi256_si128(bx);
                __m128i bxh = _mm256_extractf128_si256(bx, 1);
                bxl = _mm_or_si128(bxl, bxhil);
                bxh = _mm_or_si128(bxh, bxhih);
                bx = MM256_SET_M128I(bxh, bxl);

                const __m256 q = mul_sum_i8_pairs_float_avx(bx, by);
                acc = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), q), acc);
            }

            float s = hsum_float_8_avx(acc);
            if (bias) s += bias[n];
            C[(size_t)m * N + n] = s;
        }
    }
}

#endif

/**
 * @brief Auto-dispatch quantized dot product Q5_0 x Q8_0
 *
 * Dispatch priority:
 *   1. AVX512 (best performance on modern Intel/AMD)
 *   2. AVX (256-bit float ops, works on Sandy/Ivy Bridge and newer)
 *   3. SSSE3 (128-bit fallback)
 *   4. NEON with SDOT (AArch64 dot-product extension)
 *   5. Reference scalar (last resort; also the baseline armv8-a default -
 *      plain NEON measured slower than autovectorized scalar, see the SDOT
 *      kernel comment above)
 *
 * CK_DEBUG_Q5_0_Q8_0_REF=1 forces the scalar reference (A/B escape hatch,
 * mirrors CK_DEBUG_Q4K_Q8_REF).
 */
static int ck_q5_0_q8_0_force_ref(void)
{
    /* Concurrent first use from parallel dispatch threads must not race on
     * the lazy init: keep the cache in an atomic (idempotent value, relaxed
     * ordering is sufficient). */
    static atomic_int cached = -1;
    int v = atomic_load_explicit(&cached, memory_order_relaxed);
    if (v < 0) {
        const char *env = getenv("CK_DEBUG_Q5_0_Q8_0_REF");
        v = (env && env[0] && env[0] != '0') ? 1 : 0;
        atomic_store_explicit(&cached, v, memory_order_relaxed);
    }
    return v;
}

void vec_dot_q5_0_q8_0(int n, float *s, const void *vx, const void *vy)
{
    if (ck_q5_0_q8_0_force_ref()) {
        vec_dot_q5_0_q8_0_ref(n, s, vx, vy);
        return;
    }
#if defined(__AVX2__)
    /* llama.cpp uses the packed AVX2 dot on AVX-512 hosts as well. It keeps
     * Q5/Q8 data in byte lanes and avoids the per-block 32-bit lane expansion
     * overhead of the baseline AVX-512 path. */
    vec_dot_q5_0_q8_0_avx2(n, s, vx, vy);
#elif defined(__AVX512F__)
    vec_dot_q5_0_q8_0_avx512(n, s, vx, vy);
#elif defined(__ARM_FEATURE_DOTPROD)
    /* SDOT NEON is integer-exact against the scalar reference (identical
     * block arithmetic and FP32 epilogue); CK_DEBUG_Q5_0_Q8_0_REF=1 still
     * forces the reference above. Baseline armv8-a builds (no dot-product
     * extension) keep the scalar reference: the plain-NEON kernel measured
     * 2.5-3.3x slower than gcc's autovectorized ref on Cortex-X925 (DGX
     * Spark), so it is not dispatched (same finding as Q4_K). */
    vec_dot_q5_0_q8_0_neon_sdot(n, s, vx, vy);
#elif defined(__AVX__)
    /* AVX for 256-bit float ops (works on Ivy Bridge and newer) */
    vec_dot_q5_0_q8_0_avx(n, s, vx, vy);
#elif defined(__SSSE3__)
    /* SSSE3 - most efficient on older CPUs */
    vec_dot_q5_0_q8_0_sse(n, s, vx, vy);
#else
    vec_dot_q5_0_q8_0_ref(n, s, vx, vy);
#endif
}

void gemv_q5_0_q8_0_checked_cpu(float *out, const void *weights, const void *input, int rows, int width)
{
    const size_t blocks = (size_t)width / QK5_0;
    void (*dot)(int, float *, const void *, const void *) = vec_dot_q5_0_q8_0_ref;
#if defined(CK_Q5_RUNTIME_AVX2)
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
        __builtin_cpu_supports("f16c")) dot = vec_dot_q5_0_q8_0_avx2;
#endif
    for (int i = 0; i < rows; ++i)
        dot(width, out + i, (const block_q5_0 *)weights + (size_t)i * blocks, input);
}

#if defined(CK_Q5_RUNTIME_AVX2)
CK_Q5_TARGET_AVX2
static void gemv_q5_0_q8_0_pair_avx2(float *out0, float *out1,
                                     const block_q5_0 *weights, const block_q8_0 *x0,
                                     const block_q8_0 *x1, int rows, size_t blocks)
{
    for (int row = 0; row < rows; ++row) {
        const block_q5_0 *w = weights + (size_t)row * blocks;
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        for (size_t ib = 0; ib < blocks; ++ib) {
            const __m256i q5 = unpack_q5_0_avx2(w + ib);
            const float scale = _cvtsh_ss(w[ib].d);
            acc0 = q5_0_q8_0_accumulate_avx2(acc0, q5, scale, x0 + ib);
            acc1 = q5_0_q8_0_accumulate_avx2(acc1, q5, scale, x1 + ib);
        }
        out0[row] = hsum_float_8_avx(acc0);
        out1[row] = hsum_float_8_avx(acc1);
    }
}
#endif

void gemv_q5_0_q8_0_pair_checked_cpu(float *out0, float *out1, const void *weights,
                                     const void *x0, const void *x1, int rows, int width)
{
#if defined(CK_Q5_RUNTIME_AVX2)
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
        __builtin_cpu_supports("f16c")) {
        gemv_q5_0_q8_0_pair_avx2(out0, out1, weights, x0, x1, rows, (size_t)width / QK5_0);
        return;
    }
#endif
    gemv_q5_0_q8_0_checked_cpu(out0, weights, x0, rows, width);
    gemv_q5_0_q8_0_checked_cpu(out1, weights, x1, rows, width);
}

/* Truthful provider-name helper (same pattern as ck_q4_k_q8_k_provider_name):
 * reports the provider the dispatcher above actually selects, including the
 * CK_DEBUG_Q5_0_Q8_0_REF escape hatch. */
const char *ck_q5_0_q8_0_provider_name(void)
{
    if (ck_q5_0_q8_0_force_ref()) {
        return "q5_0_q8_0_ref";
    }
#if defined(__AVX2__)
    return "q5_0_q8_0_avx2";
#elif defined(__AVX512F__)
    return "q5_0_q8_0_avx512";
#elif defined(__ARM_FEATURE_DOTPROD)
    return "q5_0_q8_0_neon_sdot";
#elif defined(__AVX__)
    return "q5_0_q8_0_avx";
#elif defined(__SSSE3__)
    return "q5_0_q8_0_sse";
#else
    return "q5_0_q8_0_ref";
#endif
}

/* ============================================================================
 * Quantized GEMV: y = W @ x where W is Q5_0 and x is Q8_0
 *
 * This is the quantized equivalent of gemv_q5_0, but takes pre-quantized
 * input in Q8_0 format. Used for parity testing with llama.cpp.
 * ============================================================================ */

/**
 * @brief Matrix-vector multiply with Q5_0 weights and Q8_0 input
 *
 * @param y Output vector [M]
 * @param W Weight matrix in Q5_0 format [M x K]
 * @param x_q8 Input vector in Q8_0 format [K]
 * @param M Number of output rows
 * @param K Number of columns (must be multiple of 32)
 */
void gemv_q5_0_q8_0(float *y,
                     const void *W,
                     const void *x_q8,
                     int M, int K)
{
    const block_q5_0 *w_blocks = (const block_q5_0 *)W;
    const block_q8_0 *x_blocks = (const block_q8_0 *)x_q8;
    const int blocks_per_row = K / QK5_0;

    for (int row = 0; row < M; row++) {
        vec_dot_q5_0_q8_0(K, &y[row],
                          &w_blocks[row * blocks_per_row],
                          x_blocks);
    }
}

/**
 * @brief Parallel SIMD GEMV for Q5_0 x Q8_0 with prefetching
 *
 * Each thread processes rows [r0, r1) where r0 = ith * ceil(M/nth).
 * Uses vec_dot_q5_0_q8_0 dispatch (auto-selects AVX512/AVX/SSE/scalar).
 */
void gemv_q5_0_q8_0_parallel_simd(float *y,
                                   const void *W,
                                   const void *x_q8,
                                   int M, int K,
                                   int ith, int nth)
{
    if (!y || !W || !x_q8 || M <= 0 || K <= 0) return;
    if (ith < 0 || nth <= 0 || ith >= nth) return;

    const int dr = (M + nth - 1) / nth;
    const int r0 = dr * ith;
    const int r1 = (r0 + dr < M) ? (r0 + dr) : M;

    if (r0 >= M) return;

    const block_q5_0 *w_blocks = (const block_q5_0 *)W;
    const block_q8_0 *x_blocks = (const block_q8_0 *)x_q8;
    const int blocks_per_row = K / QK5_0;

#if defined(__AVX__) || defined(__SSE4_1__)
    const int PREFETCH_ROWS = 4;
    for (int p = 0; p < PREFETCH_ROWS && r0 + p < r1; ++p) {
        const char *row_ptr = (const char *)(w_blocks + (r0 + p) * blocks_per_row);
        _mm_prefetch(row_ptr, _MM_HINT_T0);
        _mm_prefetch(row_ptr + 64, _MM_HINT_T0);
    }

    for (int row = r0; row < r1; ++row) {
        if (row + PREFETCH_ROWS < r1) {
            const char *pf = (const char *)(w_blocks + (row + PREFETCH_ROWS) * blocks_per_row);
            _mm_prefetch(pf, _MM_HINT_T0);
            _mm_prefetch(pf + 64, _MM_HINT_T0);
        }

        vec_dot_q5_0_q8_0(K, &y[row],
                          &w_blocks[row * blocks_per_row],
                          x_blocks);
    }
#else
    for (int row = r0; row < r1; row++) {
        vec_dot_q5_0_q8_0(K, &y[row],
                          &w_blocks[row * blocks_per_row],
                          x_blocks);
    }
#endif
}

/**
 * @brief Batch GEMM with Q5_0 weights and Q8_0 activations for prefill
 *
 * Computes C = A @ B^T + bias where:
 *   A: [M x K] Q8_0 quantized activations (M tokens, K features)
 *   B: [N x K] Q5_0 quantized weights (N outputs, K features)
 *   C: [M x N] FP32 output
 *
 * This is the INT8 batch kernel for prefill, using pre-quantized activations
 * to avoid FP32->Q8_0 conversion overhead per operation.
 *
 * @param A_q8   Input activations in Q8_0 format [M rows of K/32 blocks each]
 * @param B_q5   Weights in Q5_0 format [N rows of K/32 blocks each]
 * @param bias   Optional bias vector [N], NULL if not used
 * @param C      Output matrix [M x N], row-major FP32
 * @param M      Batch size (number of tokens)
 * @param N      Output dimension (number of output features)
 * @param K      Input dimension (must be multiple of 32)
 */
void gemm_nt_q5_0_q8_0(
    const void *A_q8,
    const void *B_q5,
    const float *bias,
    float *C,
    int M,
    int N,
    int K)
{
    const block_q5_0 *weights = (const block_q5_0 *)B_q5;
    const block_q8_0 *inputs = (const block_q8_0 *)A_q8;
    const int blocks_per_row = K / QK5_0;

    for (int m = 0; m < M; m++) {
        const block_q8_0 *input_row = &inputs[m * blocks_per_row];

        for (int n = 0; n < N; n++) {
            const block_q5_0 *weight_row = &weights[n * blocks_per_row];
            float *out = &C[m * N + n];

            /* Dispatches to vec_dot_q5_0_q8_0_avx (2x block unrolled) on AVX */
            vec_dot_q5_0_q8_0(K, out, weight_row, input_row);

            if (bias) {
                *out += bias[n];
            }
        }
    }
}

/*
 * Two-token by four-output Q5_0 provider.
 *
 * Q5 unpacking is independent of the activation row. Keep the certified
 * eight-lane accumulator and horizontal reduction for every output while
 * applying each unpacked weight block to two token rows. This removes one
 * complete Q5 reconstruction per token pair without coupling reductions.
 */
void gemm_nt_q5_0_q8_0_m2n4_tile(
    const void *A_q8,
    const void *B_q5,
    const float *bias,
    float *C,
    int M,
    int N,
    int K,
    int ldc)
{
#if defined(__AVX2__)
    if (!A_q8 || !B_q5 || !C || M <= 0 || N <= 0 || K <= 0 ||
        (K % QK5_0) != 0 || ldc < N) {
        return;
    }

    const block_q8_0 *a = (const block_q8_0 *)A_q8;
    const block_q5_0 *w = (const block_q5_0 *)B_q5;
    const int nb = K / QK5_0;
    int m = 0;

    for (; m + 1 < M; m += 2) {
        const block_q8_0 *a0 = a + (size_t)(m + 0) * (size_t)nb;
        const block_q8_0 *a1 = a + (size_t)(m + 1) * (size_t)nb;
        int n = 0;

        for (; n + 3 < N; n += 4) {
            const block_q5_0 *w0 = w + (size_t)(n + 0) * (size_t)nb;
            const block_q5_0 *w1 = w + (size_t)(n + 1) * (size_t)nb;
            const block_q5_0 *w2 = w + (size_t)(n + 2) * (size_t)nb;
            const block_q5_0 *w3 = w + (size_t)(n + 3) * (size_t)nb;
            __m256 acc00 = _mm256_setzero_ps();
            __m256 acc01 = _mm256_setzero_ps();
            __m256 acc02 = _mm256_setzero_ps();
            __m256 acc03 = _mm256_setzero_ps();
            __m256 acc10 = _mm256_setzero_ps();
            __m256 acc11 = _mm256_setzero_ps();
            __m256 acc12 = _mm256_setzero_ps();
            __m256 acc13 = _mm256_setzero_ps();

            for (int ib = 0; ib < nb; ++ib) {
                __m256i qw0 = bytes_from_nibbles_32_avx(w0[ib].qs);
                __m256i qw1 = bytes_from_nibbles_32_avx(w1[ib].qs);
                __m256i qw2 = bytes_from_nibbles_32_avx(w2[ib].qs);
                __m256i qw3 = bytes_from_nibbles_32_avx(w3[ib].qs);
                const __m256i sign = _mm256_set1_epi8((char)0xF0);
                qw0 = _mm256_or_si256(qw0, _mm256_andnot_si256(bytes_from_bits_32_avx(w0[ib].qh), sign));
                qw1 = _mm256_or_si256(qw1, _mm256_andnot_si256(bytes_from_bits_32_avx(w1[ib].qh), sign));
                qw2 = _mm256_or_si256(qw2, _mm256_andnot_si256(bytes_from_bits_32_avx(w2[ib].qh), sign));
                qw3 = _mm256_or_si256(qw3, _mm256_andnot_si256(bytes_from_bits_32_avx(w3[ib].qh), sign));

                const __m256i qa0 = _mm256_loadu_si256((const __m256i *)a0[ib].qs);
                const __m256i qa1 = _mm256_loadu_si256((const __m256i *)a1[ib].qs);
                const float da0 = CK_FP16_TO_FP32(a0[ib].d);
                const float da1 = CK_FP16_TO_FP32(a1[ib].d);
                const float dw0 = CK_FP16_TO_FP32(w0[ib].d);
                const float dw1 = CK_FP16_TO_FP32(w1[ib].d);
                const float dw2 = CK_FP16_TO_FP32(w2[ib].d);
                const float dw3 = CK_FP16_TO_FP32(w3[ib].d);
                const __m256 p00 = mul_sum_i8_pairs_float_avx(qw0, qa0);
                const __m256 p01 = mul_sum_i8_pairs_float_avx(qw1, qa0);
                const __m256 p02 = mul_sum_i8_pairs_float_avx(qw2, qa0);
                const __m256 p03 = mul_sum_i8_pairs_float_avx(qw3, qa0);
                const __m256 p10 = mul_sum_i8_pairs_float_avx(qw0, qa1);
                const __m256 p11 = mul_sum_i8_pairs_float_avx(qw1, qa1);
                const __m256 p12 = mul_sum_i8_pairs_float_avx(qw2, qa1);
                const __m256 p13 = mul_sum_i8_pairs_float_avx(qw3, qa1);
#if defined(__FMA__)
                acc00 = _mm256_fmadd_ps(_mm256_set1_ps(dw0 * da0), p00, acc00);
                acc01 = _mm256_fmadd_ps(_mm256_set1_ps(dw1 * da0), p01, acc01);
                acc02 = _mm256_fmadd_ps(_mm256_set1_ps(dw2 * da0), p02, acc02);
                acc03 = _mm256_fmadd_ps(_mm256_set1_ps(dw3 * da0), p03, acc03);
                acc10 = _mm256_fmadd_ps(_mm256_set1_ps(dw0 * da1), p10, acc10);
                acc11 = _mm256_fmadd_ps(_mm256_set1_ps(dw1 * da1), p11, acc11);
                acc12 = _mm256_fmadd_ps(_mm256_set1_ps(dw2 * da1), p12, acc12);
                acc13 = _mm256_fmadd_ps(_mm256_set1_ps(dw3 * da1), p13, acc13);
#else
                acc00 = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(dw0 * da0), p00), acc00);
                acc01 = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(dw1 * da0), p01), acc01);
                acc02 = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(dw2 * da0), p02), acc02);
                acc03 = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(dw3 * da0), p03), acc03);
                acc10 = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(dw0 * da1), p10), acc10);
                acc11 = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(dw1 * da1), p11), acc11);
                acc12 = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(dw2 * da1), p12), acc12);
                acc13 = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(dw3 * da1), p13), acc13);
#endif
            }

            C[(size_t)(m + 0) * (size_t)ldc + n + 0] = hsum_float_8_avx(acc00) + (bias ? bias[n + 0] : 0.0f);
            C[(size_t)(m + 0) * (size_t)ldc + n + 1] = hsum_float_8_avx(acc01) + (bias ? bias[n + 1] : 0.0f);
            C[(size_t)(m + 0) * (size_t)ldc + n + 2] = hsum_float_8_avx(acc02) + (bias ? bias[n + 2] : 0.0f);
            C[(size_t)(m + 0) * (size_t)ldc + n + 3] = hsum_float_8_avx(acc03) + (bias ? bias[n + 3] : 0.0f);
            C[(size_t)(m + 1) * (size_t)ldc + n + 0] = hsum_float_8_avx(acc10) + (bias ? bias[n + 0] : 0.0f);
            C[(size_t)(m + 1) * (size_t)ldc + n + 1] = hsum_float_8_avx(acc11) + (bias ? bias[n + 1] : 0.0f);
            C[(size_t)(m + 1) * (size_t)ldc + n + 2] = hsum_float_8_avx(acc12) + (bias ? bias[n + 2] : 0.0f);
            C[(size_t)(m + 1) * (size_t)ldc + n + 3] = hsum_float_8_avx(acc13) + (bias ? bias[n + 3] : 0.0f);
        }

        for (; n < N; ++n) {
            const block_q5_0 *wn = w + (size_t)n * (size_t)nb;
            vec_dot_q5_0_q8_0(K, &C[(size_t)(m + 0) * (size_t)ldc + n], wn, a0);
            vec_dot_q5_0_q8_0(K, &C[(size_t)(m + 1) * (size_t)ldc + n], wn, a1);
            if (bias) {
                C[(size_t)(m + 0) * (size_t)ldc + n] += bias[n];
                C[(size_t)(m + 1) * (size_t)ldc + n] += bias[n];
            }
        }
    }

    if (m < M) {
        gemm_nt_q5_0_q8_0(
            a + (size_t)m * (size_t)nb, w, bias,
            C + (size_t)m * (size_t)ldc, 1, N, K);
    }
#else
    if (ldc == N) {
        gemm_nt_q5_0_q8_0(A_q8, B_q5, bias, C, M, N, K);
    } else {
        const int nb = K / QK5_0;
        const block_q8_0 *a = (const block_q8_0 *)A_q8;
        for (int m = 0; m < M; ++m) {
            gemm_nt_q5_0_q8_0(a + (size_t)m * (size_t)nb, B_q5, bias,
                              C + (size_t)m * (size_t)ldc, 1, N, K);
        }
    }
#endif
}

void gemm_nt_q5_0_q8_0_m2n4(
    const void *A_q8,
    const void *B_q5,
    const float *bias,
    float *C,
    int M,
    int N,
    int K)
{
    gemm_nt_q5_0_q8_0_m2n4_tile(A_q8, B_q5, bias, C, M, N, K, N);
}

/* Four-token by two-output companion to m2n4. It keeps the same eight
 * independent accumulators but amortizes Q5 reconstruction over four rows. */
void gemm_nt_q5_0_q8_0_m4n2_tile(
    const void *A_q8,
    const void *B_q5,
    const float *bias,
    float *C,
    int M,
    int N,
    int K,
    int ldc)
{
#if defined(__AVX2__)
    if (!A_q8 || !B_q5 || !C || M <= 0 || N <= 0 || K <= 0 ||
        (K % QK5_0) != 0 || ldc < N) {
        return;
    }

    const block_q8_0 *a = (const block_q8_0 *)A_q8;
    const block_q5_0 *w = (const block_q5_0 *)B_q5;
    const int nb = K / QK5_0;
    int m = 0;

    for (; m + 3 < M; m += 4) {
        const block_q8_0 *a0 = a + (size_t)(m + 0) * (size_t)nb;
        const block_q8_0 *a1 = a + (size_t)(m + 1) * (size_t)nb;
        const block_q8_0 *a2 = a + (size_t)(m + 2) * (size_t)nb;
        const block_q8_0 *a3 = a + (size_t)(m + 3) * (size_t)nb;
        int n = 0;

        for (; n + 1 < N; n += 2) {
            const block_q5_0 *w0 = w + (size_t)(n + 0) * (size_t)nb;
            const block_q5_0 *w1 = w + (size_t)(n + 1) * (size_t)nb;
            __m256 acc00 = _mm256_setzero_ps();
            __m256 acc01 = _mm256_setzero_ps();
            __m256 acc10 = _mm256_setzero_ps();
            __m256 acc11 = _mm256_setzero_ps();
            __m256 acc20 = _mm256_setzero_ps();
            __m256 acc21 = _mm256_setzero_ps();
            __m256 acc30 = _mm256_setzero_ps();
            __m256 acc31 = _mm256_setzero_ps();

            for (int ib = 0; ib < nb; ++ib) {
                __m256i qw0 = bytes_from_nibbles_32_avx(w0[ib].qs);
                __m256i qw1 = bytes_from_nibbles_32_avx(w1[ib].qs);
                const __m256i sign = _mm256_set1_epi8((char)0xF0);
                qw0 = _mm256_or_si256(qw0, _mm256_andnot_si256(bytes_from_bits_32_avx(w0[ib].qh), sign));
                qw1 = _mm256_or_si256(qw1, _mm256_andnot_si256(bytes_from_bits_32_avx(w1[ib].qh), sign));

                const __m256i qa0 = _mm256_loadu_si256((const __m256i *)a0[ib].qs);
                const __m256i qa1 = _mm256_loadu_si256((const __m256i *)a1[ib].qs);
                const __m256i qa2 = _mm256_loadu_si256((const __m256i *)a2[ib].qs);
                const __m256i qa3 = _mm256_loadu_si256((const __m256i *)a3[ib].qs);
                const float dw0 = CK_FP16_TO_FP32(w0[ib].d);
                const float dw1 = CK_FP16_TO_FP32(w1[ib].d);
                const float da0 = CK_FP16_TO_FP32(a0[ib].d);
                const float da1 = CK_FP16_TO_FP32(a1[ib].d);
                const float da2 = CK_FP16_TO_FP32(a2[ib].d);
                const float da3 = CK_FP16_TO_FP32(a3[ib].d);
#if defined(__FMA__)
                acc00 = _mm256_fmadd_ps(_mm256_set1_ps(dw0 * da0), mul_sum_i8_pairs_float_avx(qw0, qa0), acc00);
                acc01 = _mm256_fmadd_ps(_mm256_set1_ps(dw1 * da0), mul_sum_i8_pairs_float_avx(qw1, qa0), acc01);
                acc10 = _mm256_fmadd_ps(_mm256_set1_ps(dw0 * da1), mul_sum_i8_pairs_float_avx(qw0, qa1), acc10);
                acc11 = _mm256_fmadd_ps(_mm256_set1_ps(dw1 * da1), mul_sum_i8_pairs_float_avx(qw1, qa1), acc11);
                acc20 = _mm256_fmadd_ps(_mm256_set1_ps(dw0 * da2), mul_sum_i8_pairs_float_avx(qw0, qa2), acc20);
                acc21 = _mm256_fmadd_ps(_mm256_set1_ps(dw1 * da2), mul_sum_i8_pairs_float_avx(qw1, qa2), acc21);
                acc30 = _mm256_fmadd_ps(_mm256_set1_ps(dw0 * da3), mul_sum_i8_pairs_float_avx(qw0, qa3), acc30);
                acc31 = _mm256_fmadd_ps(_mm256_set1_ps(dw1 * da3), mul_sum_i8_pairs_float_avx(qw1, qa3), acc31);
#else
                acc00 = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(dw0 * da0), mul_sum_i8_pairs_float_avx(qw0, qa0)), acc00);
                acc01 = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(dw1 * da0), mul_sum_i8_pairs_float_avx(qw1, qa0)), acc01);
                acc10 = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(dw0 * da1), mul_sum_i8_pairs_float_avx(qw0, qa1)), acc10);
                acc11 = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(dw1 * da1), mul_sum_i8_pairs_float_avx(qw1, qa1)), acc11);
                acc20 = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(dw0 * da2), mul_sum_i8_pairs_float_avx(qw0, qa2)), acc20);
                acc21 = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(dw1 * da2), mul_sum_i8_pairs_float_avx(qw1, qa2)), acc21);
                acc30 = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(dw0 * da3), mul_sum_i8_pairs_float_avx(qw0, qa3)), acc30);
                acc31 = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(dw1 * da3), mul_sum_i8_pairs_float_avx(qw1, qa3)), acc31);
#endif
            }

            C[(size_t)(m + 0) * (size_t)ldc + n + 0] = hsum_float_8_avx(acc00) + (bias ? bias[n + 0] : 0.0f);
            C[(size_t)(m + 0) * (size_t)ldc + n + 1] = hsum_float_8_avx(acc01) + (bias ? bias[n + 1] : 0.0f);
            C[(size_t)(m + 1) * (size_t)ldc + n + 0] = hsum_float_8_avx(acc10) + (bias ? bias[n + 0] : 0.0f);
            C[(size_t)(m + 1) * (size_t)ldc + n + 1] = hsum_float_8_avx(acc11) + (bias ? bias[n + 1] : 0.0f);
            C[(size_t)(m + 2) * (size_t)ldc + n + 0] = hsum_float_8_avx(acc20) + (bias ? bias[n + 0] : 0.0f);
            C[(size_t)(m + 2) * (size_t)ldc + n + 1] = hsum_float_8_avx(acc21) + (bias ? bias[n + 1] : 0.0f);
            C[(size_t)(m + 3) * (size_t)ldc + n + 0] = hsum_float_8_avx(acc30) + (bias ? bias[n + 0] : 0.0f);
            C[(size_t)(m + 3) * (size_t)ldc + n + 1] = hsum_float_8_avx(acc31) + (bias ? bias[n + 1] : 0.0f);
        }

        if (n < N) {
            const block_q5_0 *wn = w + (size_t)n * (size_t)nb;
            const block_q8_0 *rows[4] = {a0, a1, a2, a3};
            for (int r = 0; r < 4; ++r) {
                float *out = &C[(size_t)(m + r) * (size_t)ldc + n];
                vec_dot_q5_0_q8_0(K, out, wn, rows[r]);
                if (bias) *out += bias[n];
            }
        }
    }

    if (m < M) {
        gemm_nt_q5_0_q8_0_m2n4_tile(
            a + (size_t)m * (size_t)nb, w, bias,
            C + (size_t)m * (size_t)ldc, M - m, N, K, ldc);
    }
#else
    gemm_nt_q5_0_q8_0_m2n4_tile(A_q8, B_q5, bias, C, M, N, K, ldc);
#endif
}

void gemm_nt_q5_0_q8_0_m4n2(
    const void *A_q8,
    const void *B_q5,
    const float *bias,
    float *C,
    int M,
    int N,
    int K)
{
    gemm_nt_q5_0_q8_0_m4n2_tile(A_q8, B_q5, bias, C, M, N, K, N);
}
