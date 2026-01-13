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
#include "ck_features.h"

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
             * Scalar code: xh_1 = ((qh >> (j+12))) & 0x10 extracts bit (j+16)
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
#if defined(__AVX512F__)
    gemv_q5_0_avx512(y, W, x, M, K);
#elif defined(__AVX2__) && defined(__FMA__)
    gemv_q5_0_avx2(y, W, x, M, K);
#elif defined(__AVX__)
    gemv_q5_0_avx(y, W, x, M, K);
#elif defined(__SSE4_1__)
    gemv_q5_0_sse_v2(y, W, x, M, K);
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

    __m512 acc = _mm512_setzero_ps();

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

        /* Build high bit contribution for second 16 weights (uses bits 12-27) */
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

        __m512i q5_hi = _mm512_sub_epi32(_mm512_or_epi32(hi_nibbles, qh_hi),
                                          _mm512_set1_epi32(16));

        /* Load Q8_0 values for second 16 */
        __m128i y8_hi = _mm_loadu_si128((const __m128i *)&y[ib].qs[16]);
        __m512i y32_hi = _mm512_cvtepi8_epi32(y8_hi);

        __m512i prod_hi = _mm512_mullo_epi32(q5_hi, y32_hi);

        /* Sum all products */
        int sumi = _mm512_reduce_add_epi32(_mm512_add_epi32(prod_lo, prod_hi));

        /* Scale and accumulate */
        acc = _mm512_add_ps(acc, _mm512_set1_ps(d * (float)sumi));
    }

    *s = _mm512_reduce_add_ps(acc);
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

#if defined(__AVX__) && !defined(__AVX512F__)

/* Combine two __m128i into __m256i (AVX without AVX2) */
#define MM256_SET_M128I(hi, lo) _mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 1)

/**
 * @brief Spread 32 bits to 32 bytes using AVX
 * Returns __m256i with 0xFF where bit was set, 0x00 where not
 */
static inline __m256i bytes_from_bits_32_avx(const uint8_t *qh)
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
static inline __m256i bytes_from_nibbles_32_avx(const uint8_t *qs)
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
static inline __m256 mul_sum_i8_pairs_float_avx(const __m256i x, const __m256i y)
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
static inline float hsum_float_8_avx(const __m256 x)
{
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

/**
 * @brief Quantized dot product Q5_0 x Q8_0 (AVX)
 *
 * Based on llama.cpp ggml_vec_dot_q5_0_q8_0 AVX implementation.
 * Uses 256-bit accumulation and processes 32 values per block.
 */
void vec_dot_q5_0_q8_0_avx(int n, float *s, const void *vx, const void *vy)
{
    const int qk = QK5_0;  /* 32 */
    const int nb = n / qk;

    const block_q5_0 *x = (const block_q5_0 *)vx;
    const block_q8_0 *y = (const block_q8_0 *)vy;

    __m256 acc = _mm256_setzero_ps();
    __m128i mask = _mm_set1_epi8((char)0xF0);

    for (int ib = 0; ib < nb; ib++) {
        /* Compute combined scale for the block */
        const __m256 d = _mm256_set1_ps(CK_FP16_TO_FP32(x[ib].d) * CK_FP16_TO_FP32(y[ib].d));

        /* Unpack nibbles to 32 bytes */
        __m256i bx_0 = bytes_from_nibbles_32_avx(x[ib].qs);

        /* Spread high bits */
        const __m256i bxhi = bytes_from_bits_32_avx(x[ib].qh);
        __m128i bxhil = _mm256_castsi256_si128(bxhi);
        __m128i bxhih = _mm256_extractf128_si256(bxhi, 1);

        /* Apply encoding: (~bxhi) & 0xF0 */
        bxhil = _mm_andnot_si128(bxhil, mask);
        bxhih = _mm_andnot_si128(bxhih, mask);

        /* Combine with nibbles */
        __m128i bxl = _mm256_castsi256_si128(bx_0);
        __m128i bxh = _mm256_extractf128_si256(bx_0, 1);
        bxl = _mm_or_si128(bxl, bxhil);
        bxh = _mm_or_si128(bxh, bxhih);
        bx_0 = MM256_SET_M128I(bxh, bxl);

        /* Load Q8_0 values (32 signed int8) */
        const __m256i by_0 = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        /* Multiply and sum to float */
        const __m256 q = mul_sum_i8_pairs_float_avx(bx_0, by_0);

        /* Multiply q with scale and accumulate */
        acc = _mm256_add_ps(_mm256_mul_ps(d, q), acc);
    }

    *s = hsum_float_8_avx(acc);
}
#endif

/**
 * @brief Auto-dispatch quantized dot product Q5_0 x Q8_0
 *
 * Dispatch priority:
 *   1. AVX512 (best performance on modern Intel/AMD)
 *   2. AVX2 (full 256-bit integer support)
 *   3. SSSE3 (128-bit, most efficient on older AVX CPUs like Sandy/Ivy Bridge)
 *   4. Reference scalar (last resort)
 *
 * Note: AVX without AVX2 is actually slower due to AVX-SSE transitions,
 * so we prefer SSSE3 on those CPUs.
 */
void vec_dot_q5_0_q8_0(int n, float *s, const void *vx, const void *vy)
{
#if defined(__AVX512F__)
    vec_dot_q5_0_q8_0_avx512(n, s, vx, vy);
#elif defined(__AVX2__) && defined(__FMA__)
    /* AVX2 with FMA for true 256-bit integer and float ops */
    vec_dot_q5_0_q8_0_avx(n, s, vx, vy);
#elif defined(__SSSE3__)
    /* SSSE3 - most efficient on older CPUs */
    vec_dot_q5_0_q8_0_sse(n, s, vx, vy);
#else
    vec_dot_q5_0_q8_0_ref(n, s, vx, vy);
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
    const int blocks_per_row = K / QK5_0;  /* QK5_0 == QK8_0 == 32 */

    for (int m = 0; m < M; m++) {
        const block_q8_0 *input_row = &inputs[m * blocks_per_row];

        for (int n = 0; n < N; n++) {
            const block_q5_0 *weight_row = &weights[n * blocks_per_row];
            float *out = &C[m * N + n];

            /* Use portable dispatch (selects AVX512/AVX/SSE/scalar) */
            vec_dot_q5_0_q8_0(K, out, weight_row, input_row);

            if (bias) {
                *out += bias[n];
            }
        }
    }
}
