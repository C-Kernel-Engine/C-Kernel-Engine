/**
 * @file gemm_kernels_q8_0.c
 * @brief GEMM/GEMV kernels with Q8_0 quantized weights
 *
 * Q8_0 Format:
 *   - 32 weights per block
 *   - 1 FP16 scale per block
 *   - 34 bytes per 32 weights = 8.5 bits/weight
 *   - Weights stored as signed 8-bit integers
 *
 * Operations:
 *   Forward:  Y = W @ X  (W is Q8_0, X and Y are FP32)
 *   Backward: dX = W^T @ dY (gradient w.r.t. input)
 *
 * Note: Q8_0 is often used for activation quantization or as an
 * intermediate format. Higher precision than Q4_0/Q4_K.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "ckernel_quant.h"

void quantize_row_q8_k(const float *x, void *vy, int k);

#ifdef __AVX512F__
#include <immintrin.h>
#endif

/* ============================================================================
 * Forward Pass: GEMV y = W @ x
 * ============================================================================ */

/**
 * @brief Matrix-vector multiply with Q8_0 weights (scalar reference)
 *
 * @param y Output vector [M]
 * @param W Weight matrix in Q8_0 format [M x K]
 * @param x Input vector [K]
 * @param M Number of output rows
 * @param K Number of columns (must be multiple of 32)
 */
void gemv_q8_0_ref(float *y,
                   const void *W,
                   const float *x,
                   int M, int K)
{
    const block_q8_0 *blocks = (const block_q8_0 *)W;
    const int blocks_per_row = K / QK8_0;

    for (int row = 0; row < M; row++) {
        float sum = 0.0f;

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q8_0 *block = &blocks[row * blocks_per_row + b];
            const float d = CK_FP16_TO_FP32(block->d);
            const float *xp = &x[b * QK8_0];

            for (int i = 0; i < QK8_0; i++) {
                sum += d * (float)block->qs[i] * xp[i];
            }
        }

        y[row] = sum;
    }
}

#ifdef __AVX512F__
/**
 * @brief Matrix-vector multiply with Q8_0 weights (AVX-512)
 */
void gemv_q8_0_avx512(float *y,
                      const void *W,
                      const float *x,
                      int M, int K)
{
    const block_q8_0 *blocks = (const block_q8_0 *)W;
    const int blocks_per_row = K / QK8_0;

    for (int row = 0; row < M; row++) {
        __m512 acc = _mm512_setzero_ps();

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q8_0 *block = &blocks[row * blocks_per_row + b];
            const __m512 vscale = _mm512_set1_ps(CK_FP16_TO_FP32(block->d));
            const float *xp = &x[b * QK8_0];

            /* Process 32 weights in two batches of 16 */
            for (int chunk = 0; chunk < 2; chunk++) {
                /* Load 16 x int8 weights */
                __m128i q8 = _mm_loadu_si128((const __m128i *)&block->qs[chunk * 16]);

                /* Sign-extend to 32-bit */
                __m512i q32 = _mm512_cvtepi8_epi32(q8);

                /* Convert to float and scale */
                __m512 w = _mm512_mul_ps(_mm512_cvtepi32_ps(q32), vscale);

                /* Load input */
                __m512 x_vec = _mm512_loadu_ps(&xp[chunk * 16]);

                /* FMA */
                acc = _mm512_fmadd_ps(w, x_vec, acc);
            }
        }

        y[row] = _mm512_reduce_add_ps(acc);
    }
}
#endif

#if defined(__SSE4_1__)
#include <immintrin.h>
void gemv_q8_0_sse(float *y,
                   const void *W,
                   const float *x,
                   int M, int K)
{
    const block_q8_0 *blocks = (const block_q8_0 *)W;
    const int blocks_per_row = K / QK8_0;

    for (int row = 0; row < M; row++) {
        __m128 acc = _mm_setzero_ps();

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q8_0 *block = &blocks[row * blocks_per_row + b];
            const float d_val = CK_FP16_TO_FP32(block->d);
            const float *xp = &x[b * QK8_0];

            // Load 32 weights (signed 8-bit)
            __m128i q8_0 = _mm_loadu_si128((const __m128i *)&block->qs[0]);
            __m128i q8_1 = _mm_loadu_si128((const __m128i *)&block->qs[16]);

            // Process in chunks of 4 floats for input
            for (int i=0; i<32; i+=4) {
                __m128 vx = _mm_loadu_ps(&xp[i]);
                
                // Extract 4 weights, convert to float
                __m128i qw;
                if (i < 16) {
                    qw = _mm_cvtepi8_epi32(_mm_srli_si128(q8_0, i));
                } else {
                    qw = _mm_cvtepi8_epi32(_mm_srli_si128(q8_1, i - 16));
                }
                __m128 vw = _mm_cvtepi32_ps(qw);
                acc = _mm_add_ps(acc, _mm_mul_ps(_mm_mul_ps(vw, vx), _mm_set1_ps(d_val)));
            }
        }
        
        // Hsum
        acc = _mm_add_ps(acc, _mm_shuffle_ps(acc, acc, _MM_SHUFFLE(1, 0, 3, 2)));
        acc = _mm_add_ps(acc, _mm_shuffle_ps(acc, acc, _MM_SHUFFLE(0, 1, 0, 1)));
        _mm_store_ss(&y[row], acc);
    }
}
#endif

/**
 * @brief Auto-dispatch GEMV
 */
void gemv_q8_0(float *y,
               const void *W,
               const float *x,
               int M, int K)
{
#ifdef __AVX512F__
    gemv_q8_0_avx512(y, W, x, M, K);
#elif defined(__SSE4_1__)
    gemv_q8_0_sse(y, W, x, M, K);
#else
    gemv_q8_0_ref(y, W, x, M, K);
#endif
}

/* ============================================================================
 * Forward Pass: GEMM Y = W @ X
 * ============================================================================ */

/**
 * @brief Matrix-matrix multiply with Q8_0 weights
 */
void gemm_q8_0(float *Y,
               const void *W,
               const float *X,
               int M, int N, int K)
{
    for (int n = 0; n < N; n++) {
        gemv_q8_0(&Y[n * M], W, &X[n * K], M, K);
    }
}

/* ============================================================================
 * GEMM NT: C = A @ B^T + bias  (B stored as N rows of K elements)
 * ============================================================================ */

#if defined(__SSE4_1__)
void gemm_nt_q8_0_sse(const float *A,
                      const void *B,
                      const float *bias,
                      float *C,
                      int M, int N, int K)
{
    if (K % QK_K != 0) {
        // Fallback for odd sizes (though Q8_0 weights usually aligned)
        for (int m = 0; m < M; m++) {
            gemv_q8_0_sse(&C[m * N], B, &A[m * K], N, K);
            if (bias) for (int n=0; n<N; n++) C[m*N + n] += bias[n];
        }
        return;
    }

    size_t q8_size = (K / QK_K) * sizeof(block_q8_K);
    block_q8_K *A_q8 = (block_q8_K *)alloca(q8_size);

    const block_q8_0 *weights = (const block_q8_0 *)B;
    const int blocks_per_row = K / 32;

    for (int m = 0; m < M; m++) {
        quantize_row_q8_k(&A[m * K], A_q8, K);

        for (int n = 0; n < N; n++) {
            float sumf = 0.0f;
            const block_q8_0 *w_row = weights + n * blocks_per_row;

            for (int b = 0; b < blocks_per_row; b++) {
                __m128i acc_i32 = _mm_setzero_si128();
                int q8_block_idx = (b * 32) / QK_K;
                int q8_offset = (b * 32) % QK_K;
                
                __m128i vw0 = _mm_loadu_si128((const __m128i *)&w_row[b].qs[0]);
                __m128i vw1 = _mm_loadu_si128((const __m128i *)&w_row[b].qs[16]);
                __m128i va0 = _mm_loadu_si128((const __m128i *)&A_q8[q8_block_idx].qs[q8_offset]);
                __m128i va1 = _mm_loadu_si128((const __m128i *)&A_q8[q8_block_idx].qs[q8_offset + 16]);

                // dot(q8_w, q8_a)
                __m128i vw0_lo = _mm_cvtepi8_epi16(vw0);
                __m128i vw0_hi = _mm_cvtepi8_epi16(_mm_srli_si128(vw0, 8));
                __m128i va0_lo = _mm_cvtepi8_epi16(va0);
                __m128i va0_hi = _mm_cvtepi8_epi16(_mm_srli_si128(va0, 8));
                
                acc_i32 = _mm_add_epi32(acc_i32, _mm_madd_epi16(vw0_lo, va0_lo));
                acc_i32 = _mm_add_epi32(acc_i32, _mm_madd_epi16(vw0_hi, va0_hi));

                __m128i vw1_lo = _mm_cvtepi8_epi16(vw1);
                __m128i vw1_hi = _mm_cvtepi8_epi16(_mm_srli_si128(vw1, 8));
                __m128i va1_lo = _mm_cvtepi8_epi16(va1);
                __m128i va1_hi = _mm_cvtepi8_epi16(_mm_srli_si128(va1, 8));

                acc_i32 = _mm_add_epi32(acc_i32, _mm_madd_epi16(vw1_lo, va1_lo));
                acc_i32 = _mm_add_epi32(acc_i32, _mm_madd_epi16(vw1_hi, va1_hi));
                
                // Scale by per-block deltas
                // Note: Q8_0 has per-block d, Q8_K has per-block d.
                // For Q8_0, each block is 32 elements.
                // For Q8_K, each block is 256 elements.
                
                // Horizontal sum of acc_i32
                __m128i shuf = _mm_shuffle_epi32(acc_i32, _MM_SHUFFLE(1, 0, 3, 2));
                __m128i sums = _mm_add_epi32(acc_i32, shuf);
                shuf = _mm_shuffle_epi32(sums, _MM_SHUFFLE(2, 3, 0, 1));
                sums = _mm_add_epi32(sums, shuf);
                int32_t dot_wa = _mm_cvtsi128_si32(sums);

                sumf += (float)dot_wa * CK_FP16_TO_FP32(w_row[b].d) * A_q8[q8_block_idx].d;
            }
            C[m * N + n] = sumf + (bias ? bias[n] : 0.0f);
        }
    }
}
#endif

/**
 * @brief Matrix-matrix multiply: C[M,N] = A[M,K] @ B[N,K]^T + bias
 *
 * @param A Input matrix [M x K], row-major FP32
 * @param B Weight matrix in Q8_0 format, [N x K] stored row-major
 * @param bias Optional bias [N], NULL if not used
 * @param C Output [M x N], row-major FP32
 * @param M Batch size (number of tokens)
 * @param N Output dimension (number of rows in B)
 * @param K Input dimension
 */
void gemm_nt_q8_0(const float *A,
                  const void *B,
                  const float *bias,
                  float *C,
                  int M, int N, int K)
{
#if defined(__SSE4_1__)
    gemm_nt_q8_0_sse(A, B, bias, C, M, N, K);
    return;
#endif

    const block_q8_0 *blocks = (const block_q8_0 *)B;
    const int blocks_per_row = K / QK8_0;

    for (int m = 0; m < M; m++) {
        const float *a_row = &A[m * K];

        for (int n = 0; n < N; n++) {
            float sum = 0.0f;

            for (int b = 0; b < blocks_per_row; b++) {
                const block_q8_0 *block = &blocks[n * blocks_per_row + b];
                const float d = CK_FP16_TO_FP32(block->d);
                const float *ap = &a_row[b * QK8_0];

                for (int i = 0; i < QK8_0; i++) {
                    sum += d * (float)block->qs[i] * ap[i];
                }
            }

            C[m * N + n] = sum + (bias ? bias[n] : 0.0f);
        }
    }
}

/* ============================================================================
 * Backward Pass: Gradient w.r.t. Input
 * ============================================================================ */

/**
 * @brief Backward pass: compute input gradient (scalar reference)
 *
 * @param dX Output gradient w.r.t. input [K]
 * @param W Weight matrix in Q8_0 format [M x K]
 * @param dY Gradient w.r.t. output [M]
 * @param M Number of output rows
 * @param K Number of columns (input dimension)
 */
void gemv_q8_0_backward_ref(float *dX,
                            const void *W,
                            const float *dY,
                            int M, int K)
{
    const block_q8_0 *blocks = (const block_q8_0 *)W;
    const int blocks_per_row = K / QK8_0;

    /* Zero output gradient */
    memset(dX, 0, K * sizeof(float));

    /* Accumulate: dX += W^T @ dY */
    for (int row = 0; row < M; row++) {
        const float dy = dY[row];

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q8_0 *block = &blocks[row * blocks_per_row + b];
            const float d = CK_FP16_TO_FP32(block->d);
            float *dxp = &dX[b * QK8_0];

            for (int i = 0; i < QK8_0; i++) {
                dxp[i] += d * (float)block->qs[i] * dy;
            }
        }
    }
}

#ifdef __AVX512F__
/**
 * @brief Backward pass with AVX-512
 */
void gemv_q8_0_backward_avx512(float *dX,
                               const void *W,
                               const float *dY,
                               int M, int K)
{
    const block_q8_0 *blocks = (const block_q8_0 *)W;
    const int blocks_per_row = K / QK8_0;

    /* Zero output */
    memset(dX, 0, K * sizeof(float));

    for (int row = 0; row < M; row++) {
        const __m512 vdy = _mm512_set1_ps(dY[row]);

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q8_0 *block = &blocks[row * blocks_per_row + b];
            const __m512 vscale = _mm512_set1_ps(CK_FP16_TO_FP32(block->d));
            float *dxp = &dX[b * QK8_0];

            /* Process 32 weights in two batches of 16 */
            for (int chunk = 0; chunk < 2; chunk++) {
                /* Load and dequantize weights */
                __m128i q8 = _mm_loadu_si128((const __m128i *)&block->qs[chunk * 16]);
                __m512i q32 = _mm512_cvtepi8_epi32(q8);
                __m512 w = _mm512_mul_ps(_mm512_cvtepi32_ps(q32), vscale);

                /* Compute gradient */
                __m512 grad = _mm512_mul_ps(w, vdy);

                /* Accumulate */
                __m512 dx_cur = _mm512_loadu_ps(&dxp[chunk * 16]);
                _mm512_storeu_ps(&dxp[chunk * 16], _mm512_add_ps(dx_cur, grad));
            }
        }
    }
}
#endif

/**
 * @brief Auto-dispatch backward
 */
void gemv_q8_0_backward(float *dX,
                        const void *W,
                        const float *dY,
                        int M, int K)
{
#ifdef __AVX512F__
    gemv_q8_0_backward_avx512(dX, W, dY, M, K);
#else
    gemv_q8_0_backward_ref(dX, W, dY, M, K);
#endif
}

/**
 * @brief Batched backward pass
 */
void gemm_q8_0_backward(float *dX,
                        const void *W,
                        const float *dY,
                        int M, int N, int K)
{
    for (int n = 0; n < N; n++) {
        gemv_q8_0_backward(&dX[n * K], W, &dY[n * M], M, K);
    }
}

/* ============================================================================
 * Dot Product Utility
 * ============================================================================ */

float dot_q8_0(const void *w_q8_0, const float *x, int K)
{
    float result;
    gemv_q8_0(&result, w_q8_0, x, 1, K);
    return result;
}
