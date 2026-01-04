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
#include "ckernel_quant.h"

#ifdef __AVX512F__
#include <immintrin.h>
#endif

/* Forward declarations for dequant functions (defined in dequant_kernels.c) */
void dequant_q5_0_block(const block_q5_0 *block, float *output);
void dequant_q5_0_row(const void *src, float *dst, size_t n_elements);

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

            for (int i = 0; i < QK5_0 / 2; i++) {
                const uint8_t packed = block->qs[i];

                /* Extract low 4 bits */
                const int lo0 = (packed & 0x0F);
                const int lo1 = (packed >> 4);

                /* Extract high bits */
                const int hi0 = ((qh >> (2 * i + 0)) & 1) << 4;
                const int hi1 = ((qh >> (2 * i + 1)) & 1) << 4;

                /* Combine to 5-bit signed value */
                const int q0 = (lo0 | hi0) - 16;
                const int q1 = (lo1 | hi1) - 16;

                sum += d * (float)q0 * xp[2 * i + 0];
                sum += d * (float)q1 * xp[2 * i + 1];
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

            /* Build high bit contribution for first 16 weights (indices 0,2,4,...,30) */
            __m512i qh_lo = _mm512_set_epi32(
                ((qh >> 30) & 1) << 4, ((qh >> 28) & 1) << 4,
                ((qh >> 26) & 1) << 4, ((qh >> 24) & 1) << 4,
                ((qh >> 22) & 1) << 4, ((qh >> 20) & 1) << 4,
                ((qh >> 18) & 1) << 4, ((qh >> 16) & 1) << 4,
                ((qh >> 14) & 1) << 4, ((qh >> 12) & 1) << 4,
                ((qh >> 10) & 1) << 4, ((qh >> 8) & 1) << 4,
                ((qh >> 6) & 1) << 4, ((qh >> 4) & 1) << 4,
                ((qh >> 2) & 1) << 4, ((qh >> 0) & 1) << 4
            );

            /* Build high bit contribution for second 16 weights (indices 1,3,5,...,31) */
            __m512i qh_hi = _mm512_set_epi32(
                ((qh >> 31) & 1) << 4, ((qh >> 29) & 1) << 4,
                ((qh >> 27) & 1) << 4, ((qh >> 25) & 1) << 4,
                ((qh >> 23) & 1) << 4, ((qh >> 21) & 1) << 4,
                ((qh >> 19) & 1) << 4, ((qh >> 17) & 1) << 4,
                ((qh >> 15) & 1) << 4, ((qh >> 13) & 1) << 4,
                ((qh >> 11) & 1) << 4, ((qh >> 9) & 1) << 4,
                ((qh >> 7) & 1) << 4, ((qh >> 5) & 1) << 4,
                ((qh >> 3) & 1) << 4, ((qh >> 1) & 1) << 4
            );

            /* Combine low + high bits and subtract offset */
            __m512i q_lo = _mm512_sub_epi32(_mm512_or_epi32(lo, qh_lo), offset);
            __m512i q_hi = _mm512_sub_epi32(_mm512_or_epi32(hi_shift, qh_hi), offset);

            /* Dequantize */
            __m512 w_lo = _mm512_mul_ps(_mm512_cvtepi32_ps(q_lo), vscale);
            __m512 w_hi = _mm512_mul_ps(_mm512_cvtepi32_ps(q_hi), vscale);

            /* Load interleaved input */
            __m512 x_even = _mm512_set_ps(
                xp[30], xp[28], xp[26], xp[24], xp[22], xp[20], xp[18], xp[16],
                xp[14], xp[12], xp[10], xp[8], xp[6], xp[4], xp[2], xp[0]);
            __m512 x_odd = _mm512_set_ps(
                xp[31], xp[29], xp[27], xp[25], xp[23], xp[21], xp[19], xp[17],
                xp[15], xp[13], xp[11], xp[9], xp[7], xp[5], xp[3], xp[1]);

            acc = _mm512_fmadd_ps(w_lo, x_even, acc);
            acc = _mm512_fmadd_ps(w_hi, x_odd, acc);
        }

        y[row] = _mm512_reduce_add_ps(acc);
    }
}
#endif

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

            for (int i = 0; i < QK5_0 / 2; i++) {
                const uint8_t packed = block->qs[i];

                /* Extract and reconstruct 5-bit values */
                const int lo0 = (packed & 0x0F);
                const int lo1 = (packed >> 4);
                const int hi0 = ((qh >> (2 * i + 0)) & 1) << 4;
                const int hi1 = ((qh >> (2 * i + 1)) & 1) << 4;
                const int q0 = (lo0 | hi0) - 16;
                const int q1 = (lo1 | hi1) - 16;

                dxp[2 * i + 0] += d * (float)q0 * dy;
                dxp[2 * i + 1] += d * (float)q1 * dy;
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
void gemm_nt_q5_0(const float *A,
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

                for (int i = 0; i < QK5_0 / 2; i++) {
                    const uint8_t packed = block->qs[i];
                    const int lo0 = (packed & 0x0F);
                    const int lo1 = (packed >> 4);
                    const int hi0 = ((qh >> (2 * i + 0)) & 1) << 4;
                    const int hi1 = ((qh >> (2 * i + 1)) & 1) << 4;
                    const int q0 = (lo0 | hi0) - 16;
                    const int q1 = (lo1 | hi1) - 16;

                    sum += d * (float)q0 * ap[2 * i + 0];
                    sum += d * (float)q1 * ap[2 * i + 1];
                }
            }

            C[m * N + n] = sum + (bias ? bias[n] : 0.0f);
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
