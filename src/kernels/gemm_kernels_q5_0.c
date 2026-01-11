/**
 * @file gemm_kernels_q5_0.c
 * @brief GEMM/GEMV kernels with Q5_0 quantized weights
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
#include <string.h>
#include <stdio.h>
#include "ckernel_quant.h"

/* Include SIMD headers based on available extensions */
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__) || defined(__SSE4_1__)
#include <immintrin.h>
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
             * Uses bits 12-27 (j+12 where j=0..15), NOT bits 16-31 */
            __m512i qh_hi = _mm512_set_epi32(
                ((qh >> 27) & 1) << 4, ((qh >> 26) & 1) << 4,
                ((qh >> 25) & 1) << 4, ((qh >> 24) & 1) << 4,
                ((qh >> 23) & 1) << 4, ((qh >> 22) & 1) << 4,
                ((qh >> 21) & 1) << 4, ((qh >> 20) & 1) << 4,
                ((qh >> 19) & 1) << 4, ((qh >> 18) & 1) << 4,
                ((qh >> 17) & 1) << 4, ((qh >> 16) & 1) << 4,
                ((qh >> 15) & 1) << 4, ((qh >> 14) & 1) << 4,
                ((qh >> 13) & 1) << 4, ((qh >> 12) & 1) << 4
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

#if defined(__AVX__) && !defined(__AVX512F__)

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
 * @brief Auto-dispatch GEMV
 */
void gemv_q5_0(float *y,
               const void *W,
               const float *x,
               int M, int K)
{
#ifdef __AVX512F__
    gemv_q5_0_avx512(y, W, x, M, K);
#elif defined(__AVX__)
    gemv_q5_0_avx(y, W, x, M, K);
#else
    gemv_q5_0_ref(y, W, x, M, K);
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

#if defined(__SSE4_1__)
    gemm_nt_q5_0_sse_v2(A, B, bias, C, M, N, K);
    return;
#endif
    gemm_nt_q5_0_ref(A, B, bias, C, M, N, K);
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
