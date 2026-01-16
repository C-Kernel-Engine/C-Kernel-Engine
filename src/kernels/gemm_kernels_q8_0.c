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
#include "ck_features.h"

void quantize_row_q8_k(const float *x, void *vy, int k);

/* ============================================================================
 * Q8_0 Quantization
 *
 * Quantizes FP32 values to Q8_0 format (32 elements per block).
 * Each block has:
 *   - 1 FP16 scale (computed as max abs value / 127)
 *   - 32 int8 quantized values
 *
 * This matches llama.cpp's quantize_row_q8_0.
 * ============================================================================ */

/**
 * @brief Quantize FP32 to Q8_0 format (scalar reference)
 *
 * @param x Input FP32 values
 * @param vy Output Q8_0 blocks
 * @param k Number of elements (must be multiple of 32)
 */
void quantize_row_q8_0(const float *x, void *vy, int k)
{
    block_q8_0 *y = (block_q8_0 *)vy;
    const int nb = k / QK8_0;  /* QK8_0 = 32 */

    for (int i = 0; i < nb; i++) {
        const float *xb = x + i * QK8_0;

        /* Find max absolute value in block */
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            float av = xb[j] >= 0 ? xb[j] : -xb[j];
            if (av > amax) amax = av;
        }

        /* Compute scale: d = max / 127 */
        float d = amax / 127.0f;
        float id = d != 0.0f ? 127.0f / amax : 0.0f;

        /* Store scale as FP16 */
        y[i].d = CK_FP32_TO_FP16(d);

        /* Quantize values */
        for (int j = 0; j < QK8_0; j++) {
            float v = xb[j] * id;
            /* Round to nearest int and clamp to [-127, 127] */
            int q = (int)(v + (v >= 0 ? 0.5f : -0.5f));
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            y[i].qs[j] = (int8_t)q;
        }
    }
}

/* Include SIMD headers based on available extensions */
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__) || defined(__SSE4_1__)
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

/* ============================================================================
 * AVX Implementation with True SIMD (256-bit float + 128-bit integer)
 *
 * Q8_0 format: 32 signed int8 weights per block
 *   - d: FP16 scale
 *   - qs: 32 int8 weights
 *   - Dequant: w = d * q
 *
 * This is much simpler than Q5_0 since weights are already in int8 format.
 * We use SSE for integer-to-float conversion and AVX for accumulation.
 * ============================================================================ */

#if defined(__AVX__) && !defined(__AVX512F__)

/* Helper: SSE horizontal sum of 4 floats */
static inline float hsum_sse_q8(__m128 v) {
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

/**
 * @brief Matrix-vector multiply with Q8_0 weights (AVX + SSE optimized)
 *
 * Uses full SIMD: SSE for int8->float conversion, SSE/AVX for dot product.
 * ~4-6x faster than scalar reference on Ivy Bridge.
 */
void gemv_q8_0_avx(float *y,
                   const void *W,
                   const float *x,
                   int M, int K)
{
    const block_q8_0 *blocks = (const block_q8_0 *)W;
    const int blocks_per_row = K / QK8_0;  /* QK8_0 = 32 */

    for (int row = 0; row < M; row++) {
        /* Use 4 SSE accumulators for ILP */
        __m128 acc0 = _mm_setzero_ps();
        __m128 acc1 = _mm_setzero_ps();
        __m128 acc2 = _mm_setzero_ps();
        __m128 acc3 = _mm_setzero_ps();

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q8_0 *block = &blocks[row * blocks_per_row + b];
            const float d = CK_FP16_TO_FP32(block->d);
            const float *xp = &x[b * QK8_0];
            const __m128 vscale = _mm_set1_ps(d);

            /* Load 32 int8 weights in 2 SSE loads of 16 bytes each */
            __m128i q8_0 = _mm_loadu_si128((const __m128i *)&block->qs[0]);
            __m128i q8_1 = _mm_loadu_si128((const __m128i *)&block->qs[16]);

            /* Process first 16 weights: convert int8 -> int16 -> int32 -> float */
            /* Chunk 0: weights 0-3 */
            {
                __m128i q16 = _mm_cvtepi8_epi16(q8_0);  /* 8 int8 -> 8 int16 */
                __m128i q32 = _mm_cvtepi16_epi32(q16);  /* 4 int16 -> 4 int32 */
                __m128 w = _mm_mul_ps(_mm_cvtepi32_ps(q32), vscale);
                __m128 vx = _mm_loadu_ps(&xp[0]);
                acc0 = _mm_add_ps(acc0, _mm_mul_ps(w, vx));
            }

            /* Chunk 1: weights 4-7 */
            {
                __m128i q16 = _mm_cvtepi8_epi16(q8_0);
                __m128i q32 = _mm_cvtepi16_epi32(_mm_srli_si128(q16, 8));
                __m128 w = _mm_mul_ps(_mm_cvtepi32_ps(q32), vscale);
                __m128 vx = _mm_loadu_ps(&xp[4]);
                acc1 = _mm_add_ps(acc1, _mm_mul_ps(w, vx));
            }

            /* Chunk 2: weights 8-11 */
            {
                __m128i q8_shifted = _mm_srli_si128(q8_0, 8);  /* shift right 8 bytes */
                __m128i q16 = _mm_cvtepi8_epi16(q8_shifted);
                __m128i q32 = _mm_cvtepi16_epi32(q16);
                __m128 w = _mm_mul_ps(_mm_cvtepi32_ps(q32), vscale);
                __m128 vx = _mm_loadu_ps(&xp[8]);
                acc2 = _mm_add_ps(acc2, _mm_mul_ps(w, vx));
            }

            /* Chunk 3: weights 12-15 */
            {
                __m128i q8_shifted = _mm_srli_si128(q8_0, 8);
                __m128i q16 = _mm_cvtepi8_epi16(q8_shifted);
                __m128i q32 = _mm_cvtepi16_epi32(_mm_srli_si128(q16, 8));
                __m128 w = _mm_mul_ps(_mm_cvtepi32_ps(q32), vscale);
                __m128 vx = _mm_loadu_ps(&xp[12]);
                acc3 = _mm_add_ps(acc3, _mm_mul_ps(w, vx));
            }

            /* Process second 16 weights (16-31) */
            /* Chunk 4: weights 16-19 */
            {
                __m128i q16 = _mm_cvtepi8_epi16(q8_1);
                __m128i q32 = _mm_cvtepi16_epi32(q16);
                __m128 w = _mm_mul_ps(_mm_cvtepi32_ps(q32), vscale);
                __m128 vx = _mm_loadu_ps(&xp[16]);
                acc0 = _mm_add_ps(acc0, _mm_mul_ps(w, vx));
            }

            /* Chunk 5: weights 20-23 */
            {
                __m128i q16 = _mm_cvtepi8_epi16(q8_1);
                __m128i q32 = _mm_cvtepi16_epi32(_mm_srli_si128(q16, 8));
                __m128 w = _mm_mul_ps(_mm_cvtepi32_ps(q32), vscale);
                __m128 vx = _mm_loadu_ps(&xp[20]);
                acc1 = _mm_add_ps(acc1, _mm_mul_ps(w, vx));
            }

            /* Chunk 6: weights 24-27 */
            {
                __m128i q8_shifted = _mm_srli_si128(q8_1, 8);
                __m128i q16 = _mm_cvtepi8_epi16(q8_shifted);
                __m128i q32 = _mm_cvtepi16_epi32(q16);
                __m128 w = _mm_mul_ps(_mm_cvtepi32_ps(q32), vscale);
                __m128 vx = _mm_loadu_ps(&xp[24]);
                acc2 = _mm_add_ps(acc2, _mm_mul_ps(w, vx));
            }

            /* Chunk 7: weights 28-31 */
            {
                __m128i q8_shifted = _mm_srli_si128(q8_1, 8);
                __m128i q16 = _mm_cvtepi8_epi16(q8_shifted);
                __m128i q32 = _mm_cvtepi16_epi32(_mm_srli_si128(q16, 8));
                __m128 w = _mm_mul_ps(_mm_cvtepi32_ps(q32), vscale);
                __m128 vx = _mm_loadu_ps(&xp[28]);
                acc3 = _mm_add_ps(acc3, _mm_mul_ps(w, vx));
            }
        }

        /* Combine accumulators and reduce */
        __m128 sum01 = _mm_add_ps(acc0, acc1);
        __m128 sum23 = _mm_add_ps(acc2, acc3);
        __m128 sum = _mm_add_ps(sum01, sum23);

        y[row] = hsum_sse_q8(sum);
    }
}
#endif /* __AVX__ && !__AVX512F__ */

#if defined(__SSE4_1__)
#include <immintrin.h>

/* Helper macro: extract 4 int8 weights at byte offset, convert to float, multiply with x */
#define SSE_Q8_BLOCK(q8_reg, offset, xp, d_val, acc) do { \
    __m128 vx = _mm_loadu_ps(&(xp)[offset]); \
    __m128i qw = _mm_cvtepi8_epi32(_mm_srli_si128(q8_reg, offset)); \
    __m128 vw = _mm_cvtepi32_ps(qw); \
    acc = _mm_add_ps(acc, _mm_mul_ps(_mm_mul_ps(vw, vx), _mm_set1_ps(d_val))); \
} while(0)

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

            /* Load 32 weights (signed 8-bit) in two 16-byte chunks */
            __m128i q8_0 = _mm_loadu_si128((const __m128i *)&block->qs[0]);
            __m128i q8_1 = _mm_loadu_si128((const __m128i *)&block->qs[16]);

            /* Process first 16 weights (q8_0) - unrolled with compile-time constants */
            SSE_Q8_BLOCK(q8_0, 0, xp, d_val, acc);
            SSE_Q8_BLOCK(q8_0, 4, xp, d_val, acc);
            SSE_Q8_BLOCK(q8_0, 8, xp, d_val, acc);
            SSE_Q8_BLOCK(q8_0, 12, xp, d_val, acc);

            /* Process second 16 weights (q8_1) - offset xp by 16 */
            const float *xp1 = xp + 16;
            SSE_Q8_BLOCK(q8_1, 0, xp1, d_val, acc);
            SSE_Q8_BLOCK(q8_1, 4, xp1, d_val, acc);
            SSE_Q8_BLOCK(q8_1, 8, xp1, d_val, acc);
            SSE_Q8_BLOCK(q8_1, 12, xp1, d_val, acc);
        }

        /* Horizontal sum */
        acc = _mm_add_ps(acc, _mm_shuffle_ps(acc, acc, _MM_SHUFFLE(1, 0, 3, 2)));
        acc = _mm_add_ps(acc, _mm_shuffle_ps(acc, acc, _MM_SHUFFLE(0, 1, 0, 1)));
        _mm_store_ss(&y[row], acc);
    }
}

#undef SSE_Q8_BLOCK
#endif

/**
 * @brief Auto-dispatch GEMV for Q8_0 weights based on CPU features
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
 * @param W Weight matrix in Q8_0 format [M x K]
 * @param x Input vector [K]
 * @param M Number of output rows
 * @param K Number of input columns (hidden dimension)
 */
void gemv_q8_0(float *y,
               const void *W,
               const float *x,
               int M, int K)
{
// Dispatch order: AVX512 > AVX2 > AVX > SSE > ref
// TODO: Implement proper gemv_q8_0_avx2 for AVX2 machines
// For now, AVX2 uses SSE path (more stable at large dimensions)
#if defined(__AVX512F__)
    gemv_q8_0_avx512(y, W, x, M, K);
#elif defined(__AVX2__)
    // AVX2 machines use SSE until proper AVX2 kernel is implemented
    // (AVX path has precision issues at large dims on AVX2)
    gemv_q8_0_sse(y, W, x, M, K);
#elif defined(__AVX__)
    gemv_q8_0_avx(y, W, x, M, K);
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
    /* Use GEMV dispatch which selects AVX/SSE/scalar based on CPU */
    for (int m = 0; m < M; m++) {
        gemv_q8_0(&C[m * N], B, &A[m * K], N, K);
        if (bias) {
            for (int n = 0; n < N; n++) C[m * N + n] += bias[n];
        }
    }
    return;

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

/* ============================================================================
 * Quantized Dot Product: Q8_0 x Q8_0
 *
 * This matches llama.cpp's ggml_vec_dot_q8_0_q8_0 exactly.
 * Both weights and input are in Q8_0 format, enabling pure integer dot products.
 * Result: sum_blocks( (d_w * d_x) * sum_weights( w8 * x8 ) )
 *
 * Key difference from gemv_q8_0:
 *   - gemv_q8_0: Takes FP32 input, dequantizes weights to FP32, FP32 dot
 *   - vec_dot_q8_0_q8_0: Takes Q8_0 input, does integer dot, scales at end
 *
 * The quantized path is faster and matches llama.cpp for parity testing.
 * ============================================================================ */

/**
 * @brief Quantized dot product: Q8_0 weights x Q8_0 input (scalar reference)
 *
 * @param n Number of elements (must be multiple of 32)
 * @param s Output: scalar dot product result
 * @param vx Q8_0 quantized weights
 * @param vy Q8_0 quantized input
 */
void vec_dot_q8_0_q8_0_ref(int n, float *s, const void *vx, const void *vy)
{
    const int qk = QK8_0;  /* 32 */
    const int nb = n / qk;

    const block_q8_0 *x = (const block_q8_0 *)vx;
    const block_q8_0 *y = (const block_q8_0 *)vy;

    float sumf = 0.0f;

    for (int ib = 0; ib < nb; ib++) {
        int sumi = 0;

        for (int j = 0; j < qk; j++) {
            sumi += x[ib].qs[j] * y[ib].qs[j];
        }

        sumf += sumi * (CK_FP16_TO_FP32(x[ib].d) * CK_FP16_TO_FP32(y[ib].d));
    }

    *s = sumf;
}

#ifdef __AVX512F__
/**
 * @brief Quantized dot product Q8_0 x Q8_0 (AVX-512)
 */
void vec_dot_q8_0_q8_0_avx512(int n, float *s, const void *vx, const void *vy)
{
    const int qk = QK8_0;
    const int nb = n / qk;

    const block_q8_0 *x = (const block_q8_0 *)vx;
    const block_q8_0 *y = (const block_q8_0 *)vy;

    float sumf = 0.0f;

    for (int ib = 0; ib < nb; ib++) {
        const float d = CK_FP16_TO_FP32(x[ib].d) * CK_FP16_TO_FP32(y[ib].d);

        /* Load 32 int8 weights in two batches of 16 */
        __m128i x8_lo = _mm_loadu_si128((const __m128i *)&x[ib].qs[0]);
        __m128i x8_hi = _mm_loadu_si128((const __m128i *)&x[ib].qs[16]);
        __m128i y8_lo = _mm_loadu_si128((const __m128i *)&y[ib].qs[0]);
        __m128i y8_hi = _mm_loadu_si128((const __m128i *)&y[ib].qs[16]);

        /* Sign-extend to 32-bit */
        __m512i x32_lo = _mm512_cvtepi8_epi32(x8_lo);
        __m512i x32_hi = _mm512_cvtepi8_epi32(x8_hi);
        __m512i y32_lo = _mm512_cvtepi8_epi32(y8_lo);
        __m512i y32_hi = _mm512_cvtepi8_epi32(y8_hi);

        /* Integer multiply */
        __m512i prod_lo = _mm512_mullo_epi32(x32_lo, y32_lo);
        __m512i prod_hi = _mm512_mullo_epi32(x32_hi, y32_hi);

        /* Sum all products */
        int sumi = _mm512_reduce_add_epi32(_mm512_add_epi32(prod_lo, prod_hi));

        /* Scale and accumulate - use scalar to avoid 16x broadcast bug */
        sumf += d * (float)sumi;
    }

    *s = sumf;
}
#endif

#if defined(__AVX__) && !defined(__AVX512F__)
/**
 * @brief Quantized dot product Q8_0 x Q8_0 (AVX + SSE)
 */
void vec_dot_q8_0_q8_0_avx(int n, float *s, const void *vx, const void *vy)
{
    const int qk = QK8_0;
    const int nb = n / qk;

    const block_q8_0 *x = (const block_q8_0 *)vx;
    const block_q8_0 *y = (const block_q8_0 *)vy;

    float sumf = 0.0f;

    for (int ib = 0; ib < nb; ib++) {
        const float d = CK_FP16_TO_FP32(x[ib].d) * CK_FP16_TO_FP32(y[ib].d);

        int sumi = 0;

        /* Simple loop - compiler should vectorize */
        for (int j = 0; j < qk; j++) {
            sumi += x[ib].qs[j] * y[ib].qs[j];
        }

        sumf += d * (float)sumi;
    }

    *s = sumf;
}
#endif

#if defined(__SSE4_1__) && !defined(__AVX__)
/**
 * @brief Quantized dot product Q8_0 x Q8_0 (SSE4.1)
 */
void vec_dot_q8_0_q8_0_sse(int n, float *s, const void *vx, const void *vy)
{
    const int qk = QK8_0;
    const int nb = n / qk;

    const block_q8_0 *x = (const block_q8_0 *)vx;
    const block_q8_0 *y = (const block_q8_0 *)vy;

    float sumf = 0.0f;

    for (int ib = 0; ib < nb; ib++) {
        const float d = CK_FP16_TO_FP32(x[ib].d) * CK_FP16_TO_FP32(y[ib].d);

        __m128i acc_lo = _mm_setzero_si128();
        __m128i acc_hi = _mm_setzero_si128();

        /* Process 32 elements in 4 groups of 8 */
        for (int j = 0; j < 32; j += 8) {
            /* Load 8 int8 values from each */
            __m128i x8 = _mm_loadl_epi64((const __m128i *)&x[ib].qs[j]);
            __m128i y8 = _mm_loadl_epi64((const __m128i *)&y[ib].qs[j]);

            /* Sign-extend to 16-bit */
            __m128i x16 = _mm_cvtepi8_epi16(x8);
            __m128i y16 = _mm_cvtepi8_epi16(y8);

            /* Multiply and add horizontally: (a0*b0 + a1*b1, a2*b2 + a3*b3, ...) */
            __m128i prod = _mm_madd_epi16(x16, y16);

            /* Accumulate */
            acc_lo = _mm_add_epi32(acc_lo, prod);
        }

        /* Horizontal sum */
        acc_lo = _mm_add_epi32(acc_lo, _mm_shuffle_epi32(acc_lo, _MM_SHUFFLE(1, 0, 3, 2)));
        acc_lo = _mm_add_epi32(acc_lo, _mm_shuffle_epi32(acc_lo, _MM_SHUFFLE(0, 1, 0, 1)));
        int sumi = _mm_extract_epi32(acc_lo, 0);

        sumf += d * (float)sumi;
    }

    *s = sumf;
}
#endif

/**
 * @brief Auto-dispatch quantized dot product Q8_0 x Q8_0
 */
void vec_dot_q8_0_q8_0(int n, float *s, const void *vx, const void *vy)
{
#ifdef __AVX512F__
    vec_dot_q8_0_q8_0_avx512(n, s, vx, vy);
#elif defined(__AVX__)
    vec_dot_q8_0_q8_0_avx(n, s, vx, vy);
#elif defined(__SSE4_1__)
    vec_dot_q8_0_q8_0_sse(n, s, vx, vy);
#else
    vec_dot_q8_0_q8_0_ref(n, s, vx, vy);
#endif
}

/* ============================================================================
 * Quantized GEMV: y = W @ x where W is Q8_0 and x is Q8_0
 *
 * This is the quantized equivalent of gemv_q8_0, but takes pre-quantized
 * input in Q8_0 format. Used for parity testing with llama.cpp.
 * ============================================================================ */

/**
 * @brief Matrix-vector multiply with Q8_0 weights and Q8_0 input
 *
 * @param y Output vector [M]
 * @param W Weight matrix in Q8_0 format [M x K]
 * @param x_q8 Input vector in Q8_0 format [K]
 * @param M Number of output rows
 * @param K Number of columns (must be multiple of 32)
 */
void gemv_q8_0_q8_0(float *y,
                     const void *W,
                     const void *x_q8,
                     int M, int K)
{
    const block_q8_0 *w_blocks = (const block_q8_0 *)W;
    const block_q8_0 *x_blocks = (const block_q8_0 *)x_q8;
    const int blocks_per_row = K / QK8_0;

    for (int row = 0; row < M; row++) {
        vec_dot_q8_0_q8_0(K, &y[row],
                          &w_blocks[row * blocks_per_row],
                          x_blocks);
    }
}