/**
 * @file gemm_kernels_q4_1.c
 * @brief GEMM/GEMV kernels with Q4_1 quantized weights
 *
 * Q4_1 Format:
 *   - 32 weights per block
 *   - 1 FP16 scale (d) per block
 *   - 1 FP16 minimum (m) per block
 *   - 20 bytes per 32 weights = 5.0 bits/weight
 *
 * Dequantization: w = d * q + m
 * where q is the 4-bit unsigned value (0-15)
 *
 * Operations:
 *   Forward:  Y = W @ X  (W is Q4_1, X and Y are FP32)
 *   Backward: dX = W^T @ dY (gradient w.r.t. input)
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "ckernel_quant.h"

#ifdef __AVX512F__
#include <immintrin.h>
#endif

/* ============================================================================
 * Forward Pass: GEMV y = W @ x
 * ============================================================================ */

/**
 * @brief Matrix-vector multiply with Q4_1 weights (scalar reference)
 *
 * @param y Output vector [M]
 * @param W Weight matrix in Q4_1 format [M x K]
 * @param x Input vector [K]
 * @param M Number of output rows
 * @param K Number of columns (must be multiple of 32)
 */
void gemv_q4_1_ref(float *y,
                   const void *W,
                   const float *x,
                   int M, int K)
{
    const block_q4_1 *blocks = (const block_q4_1 *)W;
    const int blocks_per_row = K / QK4_1;

    for (int row = 0; row < M; row++) {
        float sum = 0.0f;

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q4_1 *block = &blocks[row * blocks_per_row + b];
            const float d = CK_FP16_TO_FP32(block->d);
            const float m = CK_FP16_TO_FP32(block->m);
            const float *xp = &x[b * QK4_1];

            for (int i = 0; i < QK4_1 / 2; i++) {
                const uint8_t packed = block->qs[i];
                const int q0 = (packed & 0x0F);
                const int q1 = (packed >> 4);

                /* Dequantize: w = d * q + m */
                const float w0 = d * (float)q0 + m;
                const float w1 = d * (float)q1 + m;

                sum += w0 * xp[2*i + 0];
                sum += w1 * xp[2*i + 1];
            }
        }

        y[row] = sum;
    }
}

#ifdef __AVX512F__
/**
 * @brief Matrix-vector multiply with Q4_1 weights (AVX-512)
 */
void gemv_q4_1_avx512(float *y,
                      const void *W,
                      const float *x,
                      int M, int K)
{
    const block_q4_1 *blocks = (const block_q4_1 *)W;
    const int blocks_per_row = K / QK4_1;
    const __m512i mask_lo = _mm512_set1_epi32(0x0F);

    for (int row = 0; row < M; row++) {
        __m512 acc = _mm512_setzero_ps();

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q4_1 *block = &blocks[row * blocks_per_row + b];
            const __m512 vscale = _mm512_set1_ps(CK_FP16_TO_FP32(block->d));
            const __m512 vmin = _mm512_set1_ps(CK_FP16_TO_FP32(block->m));
            const float *xp = &x[b * QK4_1];

            /* Load 16 bytes = 32 x 4-bit weights */
            __m128i packed = _mm_loadu_si128((const __m128i *)block->qs);
            __m512i bytes = _mm512_cvtepu8_epi32(packed);

            /* Extract nibbles */
            __m512i lo = _mm512_and_epi32(bytes, mask_lo);
            __m512i hi = _mm512_srli_epi32(bytes, 4);

            /* Dequantize: w = d * q + m */
            __m512 w_lo = _mm512_fmadd_ps(_mm512_cvtepi32_ps(lo), vscale, vmin);
            __m512 w_hi = _mm512_fmadd_ps(_mm512_cvtepi32_ps(hi), vscale, vmin);

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
void gemv_q4_1(float *y,
               const void *W,
               const float *x,
               int M, int K)
{
#ifdef __AVX512F__
    gemv_q4_1_avx512(y, W, x, M, K);
#else
    gemv_q4_1_ref(y, W, x, M, K);
#endif
}

/* ============================================================================
 * Forward Pass: GEMM Y = W @ X
 * ============================================================================ */

/**
 * @brief Matrix-matrix multiply with Q4_1 weights
 */
void gemm_q4_1(float *Y,
               const void *W,
               const float *X,
               int M, int N, int K)
{
    for (int n = 0; n < N; n++) {
        gemv_q4_1(&Y[n * M], W, &X[n * K], M, K);
    }
}

/* ============================================================================
 * Backward Pass: Gradient w.r.t. Input
 * ============================================================================ */

/**
 * @brief Backward pass: compute input gradient
 *
 * @param dX Output gradient w.r.t. input [K]
 * @param W Weight matrix in Q4_1 format [M x K]
 * @param dY Gradient w.r.t. output [M]
 * @param M Number of output rows
 * @param K Number of columns (input dimension)
 */
void gemv_q4_1_backward_ref(float *dX,
                            const void *W,
                            const float *dY,
                            int M, int K)
{
    const block_q4_1 *blocks = (const block_q4_1 *)W;
    const int blocks_per_row = K / QK4_1;

    /* Zero output gradient */
    memset(dX, 0, K * sizeof(float));

    /* Accumulate: dX += W^T @ dY */
    for (int row = 0; row < M; row++) {
        const float dy = dY[row];

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q4_1 *block = &blocks[row * blocks_per_row + b];
            const float d = CK_FP16_TO_FP32(block->d);
            const float m = CK_FP16_TO_FP32(block->m);
            float *dxp = &dX[b * QK4_1];

            for (int i = 0; i < QK4_1 / 2; i++) {
                const uint8_t packed = block->qs[i];
                const int q0 = (packed & 0x0F);
                const int q1 = (packed >> 4);

                const float w0 = d * (float)q0 + m;
                const float w1 = d * (float)q1 + m;

                dxp[2*i + 0] += w0 * dy;
                dxp[2*i + 1] += w1 * dy;
            }
        }
    }
}

/**
 * @brief Auto-dispatch backward
 */
void gemv_q4_1_backward(float *dX,
                        const void *W,
                        const float *dY,
                        int M, int K)
{
    gemv_q4_1_backward_ref(dX, W, dY, M, K);
}

/**
 * @brief Batched backward pass
 */
void gemm_q4_1_backward(float *dX,
                        const void *W,
                        const float *dY,
                        int M, int N, int K)
{
    for (int n = 0; n < N; n++) {
        gemv_q4_1_backward(&dX[n * K], W, &dY[n * M], M, K);
    }
}

/* ============================================================================
 * GEMM NT (Non-Transpose A, Transpose B) - C = A @ B^T
 * ============================================================================ */

/**
 * @brief GEMM with transposed Q4_1 weights: C = A @ B^T
 *
 * @param A Input activations [M x K], row-major FP32
 * @param B Weight matrix in Q4_1 format [N x K], row-major quantized
 * @param bias Optional bias [N], NULL if not used
 * @param C Output [M x N], row-major FP32
 * @param M Batch size (number of tokens)
 * @param N Output dimension
 * @param K Input dimension
 */
void gemm_nt_q4_1(const float *A,
                  const void *B,
                  const float *bias,
                  float *C,
                  int M, int N, int K)
{
    const block_q4_1 *blocks = (const block_q4_1 *)B;
    const int blocks_per_row = K / QK4_1;

    for (int m = 0; m < M; m++) {
        const float *a_row = &A[m * K];

        for (int n = 0; n < N; n++) {
            float sum = 0.0f;

            for (int b = 0; b < blocks_per_row; b++) {
                const block_q4_1 *block = &blocks[n * blocks_per_row + b];
                const float d = CK_FP16_TO_FP32(block->d);
                const float min = CK_FP16_TO_FP32(block->m);
                const float *ap = &a_row[b * QK4_1];

                for (int i = 0; i < QK4_1 / 2; i++) {
                    const uint8_t packed = block->qs[i];
                    const int q0 = (packed & 0x0F);
                    const int q1 = (packed >> 4);

                    const float w0 = d * (float)q0 + min;
                    const float w1 = d * (float)q1 + min;

                    sum += w0 * ap[2 * i + 0];
                    sum += w1 * ap[2 * i + 1];
                }
            }

            C[m * N + n] = sum + (bias ? bias[n] : 0.0f);
        }
    }
}

/* ============================================================================
 * Dot Product Utility
 * ============================================================================ */

float dot_q4_1(const void *w_q4_1, const float *x, int K)
{
    float result;
    gemv_q4_1(&result, w_q4_1, x, 1, K);
    return result;
}
