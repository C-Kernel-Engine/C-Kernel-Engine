/**
 * @file gemm_kernels_q4_0.c
 * @brief GEMM/GEMV kernels with Q4_0 quantized weights
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
 * Q4_0 Format:
 *   - 32 weights per block
 *   - 1 FP16 scale per block
 *   - 18 bytes per 32 weights = 4.5 bits/weight
 *
 * Operations:
 *   Forward:  Y = W @ X  (W is Q4_0, X and Y are FP32)
 *   Backward: dX = W^T @ dY (gradient w.r.t. input)
 *
 * Note: Weight gradients are not computed for quantized weights.
 * For fine-tuning, use LoRA adapters which maintain FP32 gradients separately.
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
 * @brief Matrix-vector multiply with Q4_0 weights (scalar reference)
 *
 * @param y Output vector [M]
 * @param W Weight matrix in Q4_0 format [M x K]
 * @param x Input vector [K]
 * @param M Number of output rows
 * @param K Number of columns (must be multiple of 32)
 */
void gemv_q4_0_ref(float *y,
                   const void *W,
                   const float *x,
                   int M, int K)
{
    const block_q4_0 *blocks = (const block_q4_0 *)W;
    const int blocks_per_row = K / QK4_0;

    for (int row = 0; row < M; row++) {
        float sum = 0.0f;

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q4_0 *block = &blocks[row * blocks_per_row + b];
            const float d = CK_FP16_TO_FP32(block->d);
            const float *xp = &x[b * QK4_0];

            for (int i = 0; i < QK4_0 / 2; i++) {
                const uint8_t packed = block->qs[i];
                const int8_t q0 = (packed & 0x0F) - 8;
                const int8_t q1 = (packed >> 4) - 8;

                sum += d * (float)q0 * xp[2*i + 0];
                sum += d * (float)q1 * xp[2*i + 1];
            }
        }

        y[row] = sum;
    }
}

#ifdef __AVX512F__
/**
 * @brief Matrix-vector multiply with Q4_0 weights (AVX-512)
 */
void gemv_q4_0_avx512(float *y,
                      const void *W,
                      const float *x,
                      int M, int K)
{
    const block_q4_0 *blocks = (const block_q4_0 *)W;
    const int blocks_per_row = K / QK4_0;
    const __m512i offset = _mm512_set1_epi32(8);
    const __m512i mask_lo = _mm512_set1_epi32(0x0F);

    for (int row = 0; row < M; row++) {
        __m512 acc = _mm512_setzero_ps();

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q4_0 *block = &blocks[row * blocks_per_row + b];
            const __m512 vscale = _mm512_set1_ps(CK_FP16_TO_FP32(block->d));
            const float *xp = &x[b * QK4_0];

            /* Load 16 bytes = 32 x 4-bit weights */
            __m128i packed = _mm_loadu_si128((const __m128i *)block->qs);
            __m512i bytes = _mm512_cvtepu8_epi32(packed);

            /* Extract and dequantize */
            __m512i lo = _mm512_sub_epi32(_mm512_and_epi32(bytes, mask_lo), offset);
            __m512i hi = _mm512_sub_epi32(_mm512_srli_epi32(bytes, 4), offset);

            __m512 w_lo = _mm512_mul_ps(_mm512_cvtepi32_ps(lo), vscale);
            __m512 w_hi = _mm512_mul_ps(_mm512_cvtepi32_ps(hi), vscale);

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
void gemv_q4_0(float *y,
               const void *W,
               const float *x,
               int M, int K)
{
#ifdef __AVX512F__
    gemv_q4_0_avx512(y, W, x, M, K);
#else
    gemv_q4_0_ref(y, W, x, M, K);
#endif
}

/* ============================================================================
 * Forward Pass: GEMM Y = W @ X
 * ============================================================================ */

/**
 * @brief Matrix-matrix multiply with Q4_0 weights
 */
void gemm_q4_0(float *Y,
               const void *W,
               const float *X,
               int M, int N, int K)
{
    for (int n = 0; n < N; n++) {
        gemv_q4_0(&Y[n * M], W, &X[n * K], M, K);
    }
}

/* ============================================================================
 * GEMM NT: C = A @ B^T + bias  (B stored as N rows of K elements)
 * ============================================================================ */

/**
 * @brief Matrix-matrix multiply: C[M,N] = A[M,K] @ B[N,K]^T + bias
 *
 * @param A Input matrix [M x K], row-major FP32
 * @param B Weight matrix in Q4_0 format, [N x K] stored row-major
 * @param bias Optional bias [N], NULL if not used
 * @param C Output [M x N], row-major FP32
 * @param M Batch size (number of tokens)
 * @param N Output dimension (number of rows in B)
 * @param K Input dimension
 */
void gemm_nt_q4_0(const float *A,
                  const void *B,
                  const float *bias,
                  float *C,
                  int M, int N, int K)
{
    const block_q4_0 *blocks = (const block_q4_0 *)B;
    const int blocks_per_row = K / QK4_0;

    for (int m = 0; m < M; m++) {
        const float *a_row = &A[m * K];

        for (int n = 0; n < N; n++) {
            float sum = 0.0f;

            for (int b = 0; b < blocks_per_row; b++) {
                const block_q4_0 *block = &blocks[n * blocks_per_row + b];
                const float d = CK_FP16_TO_FP32(block->d);
                const float *ap = &a_row[b * QK4_0];

                for (int i = 0; i < QK4_0 / 2; i++) {
                    const uint8_t packed = block->qs[i];
                    const int q0 = (packed & 0x0F) - 8;
                    const int q1 = (packed >> 4) - 8;

                    sum += d * (float)q0 * ap[2 * i + 0];
                    sum += d * (float)q1 * ap[2 * i + 1];
                }
            }

            C[m * N + n] = sum + (bias ? bias[n] : 0.0f);
        }
    }
}

/* ============================================================================
 * Backward Pass: Gradient w.r.t. Input
 *
 * Given: dL/dY (gradient of loss w.r.t. output)
 * Compute: dL/dX = W^T @ dL/dY
 *
 * For quantized weights, we dequantize on-the-fly during backprop.
 * Weight gradients are NOT computed (weights are frozen).
 * ============================================================================ */

/**
 * @brief Backward pass: compute input gradient
 *
 * @param dX Output gradient w.r.t. input [K]
 * @param W Weight matrix in Q4_0 format [M x K]
 * @param dY Gradient w.r.t. output [M]
 * @param M Number of output rows
 * @param K Number of columns (input dimension)
 */
void gemv_q4_0_backward_ref(float *dX,
                            const void *W,
                            const float *dY,
                            int M, int K)
{
    const block_q4_0 *blocks = (const block_q4_0 *)W;
    const int blocks_per_row = K / QK4_0;

    /* Zero output gradient */
    memset(dX, 0, K * sizeof(float));

    /* Accumulate: dX += W^T @ dY */
    for (int row = 0; row < M; row++) {
        const float dy = dY[row];

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q4_0 *block = &blocks[row * blocks_per_row + b];
            const float d = CK_FP16_TO_FP32(block->d);
            float *dxp = &dX[b * QK4_0];

            for (int i = 0; i < QK4_0 / 2; i++) {
                const uint8_t packed = block->qs[i];
                const int8_t q0 = (packed & 0x0F) - 8;
                const int8_t q1 = (packed >> 4) - 8;

                dxp[2*i + 0] += d * (float)q0 * dy;
                dxp[2*i + 1] += d * (float)q1 * dy;
            }
        }
    }
}

#ifdef __AVX512F__
/**
 * @brief Backward pass with AVX-512
 */
void gemv_q4_0_backward_avx512(float *dX,
                               const void *W,
                               const float *dY,
                               int M, int K)
{
    const block_q4_0 *blocks = (const block_q4_0 *)W;
    const int blocks_per_row = K / QK4_0;
    const __m512i offset = _mm512_set1_epi32(8);
    const __m512i mask_lo = _mm512_set1_epi32(0x0F);

    /* Zero output */
    memset(dX, 0, K * sizeof(float));

    for (int row = 0; row < M; row++) {
        const __m512 vdy = _mm512_set1_ps(dY[row]);

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q4_0 *block = &blocks[row * blocks_per_row + b];
            const __m512 vscale = _mm512_set1_ps(CK_FP16_TO_FP32(block->d));
            float *dxp = &dX[b * QK4_0];

            /* Dequantize weights */
            __m128i packed = _mm_loadu_si128((const __m128i *)block->qs);
            __m512i bytes = _mm512_cvtepu8_epi32(packed);

            __m512i lo = _mm512_sub_epi32(_mm512_and_epi32(bytes, mask_lo), offset);
            __m512i hi = _mm512_sub_epi32(_mm512_srli_epi32(bytes, 4), offset);

            __m512 w_lo = _mm512_mul_ps(_mm512_cvtepi32_ps(lo), vscale);
            __m512 w_hi = _mm512_mul_ps(_mm512_cvtepi32_ps(hi), vscale);

            /* Compute gradients: dX += W * dY */
            __m512 grad_lo = _mm512_mul_ps(w_lo, vdy);
            __m512 grad_hi = _mm512_mul_ps(w_hi, vdy);

            /* Scatter to interleaved positions (simplified - actual impl needs gather/scatter) */
            float grad_lo_arr[16], grad_hi_arr[16];
            _mm512_storeu_ps(grad_lo_arr, grad_lo);
            _mm512_storeu_ps(grad_hi_arr, grad_hi);

            for (int i = 0; i < 16; i++) {
                dxp[2*i + 0] += grad_lo_arr[i];
                dxp[2*i + 1] += grad_hi_arr[i];
            }
        }
    }
}
#endif

/**
 * @brief Auto-dispatch backward
 */
void gemv_q4_0_backward(float *dX,
                        const void *W,
                        const float *dY,
                        int M, int K)
{
#ifdef __AVX512F__
    gemv_q4_0_backward_avx512(dX, W, dY, M, K);
#else
    gemv_q4_0_backward_ref(dX, W, dY, M, K);
#endif
}

/**
 * @brief Batched backward pass
 */
void gemm_q4_0_backward(float *dX,
                        const void *W,
                        const float *dY,
                        int M, int N, int K)
{
    for (int n = 0; n < N; n++) {
        gemv_q4_0_backward(&dX[n * K], W, &dY[n * M], M, K);
    }
}

/* ============================================================================
 * Dot Product Utility
 * ============================================================================ */

float dot_q4_0(const void *w_q4_0, const float *x, int K)
{
    float result;
    gemv_q4_0(&result, w_q4_0, x, 1, K);
    return result;
}
