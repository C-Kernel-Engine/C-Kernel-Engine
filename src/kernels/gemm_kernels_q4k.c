/**
 * @file gemm_kernels_q4k.c
 * @brief GEMM/GEMV kernels with Q4_K quantized weights
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
 * Implements matrix multiplication where:
 *   - Activations (input): FP32
 *   - Weights: Q4_K (4.5 bits/weight, nested scales)
 *   - Output: FP32
 *
 * Key optimization: Fused dequantization - weights are dequantized in
 * registers and immediately used in FMA, never written to memory.
 *
 * Operations:
 *   - gemv_q4_k: Matrix-vector multiply (batch=1, token generation)
 *   - gemm_q4_k: Matrix-matrix multiply (batch>1, prefill)
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "ckernel_quant.h"

/* Include SIMD headers based on available extensions */
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__) || defined(__SSE4_1__)
#include <immintrin.h>
#endif

/* ============================================================================
 * GEMV: y = W @ x  (W is Q4_K, x and y are FP32)
 *
 * For token generation (batch=1), this is the critical path.
 * Memory-bound: we're loading ~4GB of weights for a 7B model per token.
 * ============================================================================ */

/**
 * @brief Matrix-vector multiply with Q4_K weights (scalar reference)
 *
 * @param y Output vector [M]
 * @param W Weight matrix in Q4_K format [M x K], stored row-major
 * @param x Input vector [K]
 * @param M Number of output rows
 * @param K Number of columns (must be multiple of 256)
 */
void gemv_q4_k_ref(float *y,
                   const void *W,
                   const float *x,
                   int M, int K)
{
    const block_q4_K *blocks = (const block_q4_K *)W;
    const int blocks_per_row = K / QK_K;  /* QK_K = 256 */

    for (int row = 0; row < M; row++) {
        float sum = 0.0f;

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q4_K *block = &blocks[row * blocks_per_row + b];
            const float d = GGML_FP16_TO_FP32(block->d);
            const float dmin = GGML_FP16_TO_FP32(block->dmin);

            /* Unpack sub-block scales */
            uint8_t sc[8], m[8];
            unpack_q4_k_scales(block->scales, sc, m);

            /* llama.cpp Q4_K layout: 4 iterations of 64 weights each
             * Each iteration uses 32 bytes of qs and 2 scales:
             *   - First 32 weights (indices 0-31): low nibbles with scale[2*iter]
             *   - Next 32 weights (indices 32-63): high nibbles with scale[2*iter+1]
             */
            for (int iter = 0; iter < 4; iter++) {
                const float d1 = d * (float)sc[2*iter];
                const float m1 = dmin * (float)m[2*iter];
                const float d2 = d * (float)sc[2*iter + 1];
                const float m2 = dmin * (float)m[2*iter + 1];
                const uint8_t *qs = &block->qs[iter * 32];
                const float *xp = &x[b * QK_K + iter * 64];

                /* First 32 weights: low nibbles of qs[0..31] */
                for (int l = 0; l < 32; l++) {
                    const int8_t q = (qs[l] & 0x0F);
                    sum += (d1 * (float)q - m1) * xp[l];
                }
                /* Next 32 weights: high nibbles of qs[0..31] */
                for (int l = 0; l < 32; l++) {
                    const int8_t q = (qs[l] >> 4);
                    sum += (d2 * (float)q - m2) * xp[l + 32];
                }
            }
        }

        y[row] = sum;
    }
}

#ifdef __AVX512F__
/**
 * @brief Matrix-vector multiply with Q4_K weights (AVX-512 optimized)
 *
 * Fused dequant + FMA: weights dequantized in ZMM registers, never touch RAM.
 */
void gemv_q4_k_avx512(float *y,
                      const void *W,
                      const float *x,
                      int M, int K)
{
    const block_q4_K *blocks = (const block_q4_K *)W;
    const int blocks_per_row = K / QK_K;

    for (int row = 0; row < M; row++) {
        __m512 acc = _mm512_setzero_ps();

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q4_K *block = &blocks[row * blocks_per_row + b];
            const float d = GGML_FP16_TO_FP32(block->d);
            const float dmin = GGML_FP16_TO_FP32(block->dmin);

            uint8_t sc[8], m_arr[8];
            unpack_q4_k_scales(block->scales, sc, m_arr);

            const __m512i mask_lo = _mm512_set1_epi32(0x0F);

            /* llama.cpp Q4_K layout: 4 iterations of 64 weights each
             * Formula: w = d * q - m (NOT d * (q-8) + m)
             */
            for (int iter = 0; iter < 4; iter++) {
                const float d1 = d * (float)sc[2*iter];
                const float m1 = dmin * (float)m_arr[2*iter];
                const float d2 = d * (float)sc[2*iter + 1];
                const float m2 = dmin * (float)m_arr[2*iter + 1];

                const __m512 vscale1 = _mm512_set1_ps(d1);
                const __m512 vmin1 = _mm512_set1_ps(m1);
                const __m512 vscale2 = _mm512_set1_ps(d2);
                const __m512 vmin2 = _mm512_set1_ps(m2);

                const uint8_t *qs = &block->qs[iter * 32];
                const float *xp = &x[b * QK_K + iter * 64];

                /* Process first 32 weights (low nibbles) */
                /* Load 16 bytes at a time */
                for (int chunk = 0; chunk < 2; chunk++) {
                    __m128i packed = _mm_loadu_si128((const __m128i *)&qs[chunk * 16]);
                    __m512i bytes = _mm512_cvtepu8_epi32(packed);
                    __m512i lo = _mm512_and_epi32(bytes, mask_lo);
                    /* w = d * q - m: use fnmadd (negative m) */
                    __m512 w = _mm512_fnmadd_ps(_mm512_set1_ps(1.0f), vmin1,
                               _mm512_mul_ps(_mm512_cvtepi32_ps(lo), vscale1));
                    __m512 x_vec = _mm512_loadu_ps(&xp[chunk * 16]);
                    acc = _mm512_fmadd_ps(w, x_vec, acc);
                }

                /* Process next 32 weights (high nibbles) */
                for (int chunk = 0; chunk < 2; chunk++) {
                    __m128i packed = _mm_loadu_si128((const __m128i *)&qs[chunk * 16]);
                    __m512i bytes = _mm512_cvtepu8_epi32(packed);
                    __m512i hi = _mm512_srli_epi32(bytes, 4);
                    /* w = d * q - m: use fnmadd (negative m) */
                    __m512 w = _mm512_fnmadd_ps(_mm512_set1_ps(1.0f), vmin2,
                               _mm512_mul_ps(_mm512_cvtepi32_ps(hi), vscale2));
                    __m512 x_vec = _mm512_loadu_ps(&xp[32 + chunk * 16]);
                    acc = _mm512_fmadd_ps(w, x_vec, acc);
                }
            }
        }

        /* Horizontal sum */
        y[row] = _mm512_reduce_add_ps(acc);
    }
}
#endif /* __AVX512F__ */

/* ============================================================================
 * AVX Implementation (256-bit, works on Sandy Bridge and later)
 *
 * This is critical for CPUs that have AVX but not AVX-512.
 * Processes 8 floats per iteration using 256-bit registers.
 * NOTE: Uses separate mul+add (no FMA) for Ivy Bridge compatibility.
 * ============================================================================ */

#if defined(__AVX__) && !defined(__AVX512F__)
/**
 * @brief Matrix-vector multiply with Q4_K weights (AVX optimized)
 *
 * Processes 8 floats at a time. No FMA required (works on Ivy Bridge).
 * About 4-8x faster than scalar reference.
 */
void gemv_q4_k_avx(float *y,
                   const void *W,
                   const float *x,
                   int M, int K)
{
    const block_q4_K *blocks = (const block_q4_K *)W;
    const int blocks_per_row = K / QK_K;  /* QK_K = 256 */

    for (int row = 0; row < M; row++) {
        /* Use 4 accumulators for better instruction-level parallelism */
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q4_K *block = &blocks[row * blocks_per_row + b];
            const float d = GGML_FP16_TO_FP32(block->d);
            const float dmin = GGML_FP16_TO_FP32(block->dmin);

            /* Unpack sub-block scales */
            uint8_t sc[8], m_arr[8];
            unpack_q4_k_scales(block->scales, sc, m_arr);

            /* Process 256 weights in 4 iterations of 64 weights each */
            for (int iter = 0; iter < 4; iter++) {
                const float d1 = d * (float)sc[2*iter];
                const float m1 = dmin * (float)m_arr[2*iter];
                const float d2 = d * (float)sc[2*iter + 1];
                const float m2 = dmin * (float)m_arr[2*iter + 1];
                const uint8_t *qs = &block->qs[iter * 32];
                const float *xp = &x[b * QK_K + iter * 64];

                /* Broadcast scale and min values */
                __m256 vd1 = _mm256_set1_ps(d1);
                __m256 vm1 = _mm256_set1_ps(m1);
                __m256 vd2 = _mm256_set1_ps(d2);
                __m256 vm2 = _mm256_set1_ps(m2);

                /* Process first 32 weights (low nibbles) in 4 groups of 8 */
                for (int g = 0; g < 4; g++) {
                    /* Dequantize 8 weights from low nibbles */
                    float dq[8];
                    for (int i = 0; i < 8; i++) {
                        dq[i] = d1 * (float)(qs[g*8 + i] & 0x0F) - m1;
                    }
                    __m256 vw = _mm256_loadu_ps(dq);
                    __m256 vx = _mm256_loadu_ps(&xp[g*8]);

                    /* acc0 += vw * vx (using mul+add, no FMA needed) */
                    __m256 prod = _mm256_mul_ps(vw, vx);
                    acc0 = _mm256_add_ps(acc0, prod);
                }

                /* Process next 32 weights (high nibbles) in 4 groups of 8 */
                for (int g = 0; g < 4; g++) {
                    /* Dequantize 8 weights from high nibbles */
                    float dq[8];
                    for (int i = 0; i < 8; i++) {
                        dq[i] = d2 * (float)(qs[g*8 + i] >> 4) - m2;
                    }
                    __m256 vw = _mm256_loadu_ps(dq);
                    __m256 vx = _mm256_loadu_ps(&xp[32 + g*8]);

                    __m256 prod = _mm256_mul_ps(vw, vx);
                    acc1 = _mm256_add_ps(acc1, prod);
                }
            }
        }

        /* Combine accumulators */
        __m256 sum01 = _mm256_add_ps(acc0, acc1);
        __m256 sum23 = _mm256_add_ps(acc2, acc3);
        __m256 sum = _mm256_add_ps(sum01, sum23);

        /* Horizontal sum of 8 floats */
        __m128 hi = _mm256_extractf128_ps(sum, 1);
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 sum128 = _mm_add_ps(hi, lo);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);

        y[row] = _mm_cvtss_f32(sum128);
    }
}
#endif /* __AVX__ && !__AVX512F__ */

/**
 * @brief Auto-dispatch GEMV based on available SIMD
 */
void gemv_q4_k(float *y,
               const void *W,
               const float *x,
               int M, int K)
{
#ifdef __AVX512F__
    gemv_q4_k_avx512(y, W, x, M, K);
#elif defined(__AVX__)
    gemv_q4_k_avx(y, W, x, M, K);
#else
    gemv_q4_k_ref(y, W, x, M, K);
#endif
}

/* ============================================================================
 * GEMM: Y = W @ X  (W is Q4_K, X and Y are FP32)
 *
 * For prefill (batch > 1), we can amortize weight loading across batch.
 * More compute-bound than GEMV.
 * ============================================================================ */

/**
 * @brief Matrix-matrix multiply with Q4_K weights (scalar reference)
 *
 * @param Y Output matrix [M x N]
 * @param W Weight matrix in Q4_K format [M x K]
 * @param X Input matrix [K x N] (column-major for cache efficiency)
 * @param M Number of output rows
 * @param N Batch size (number of columns)
 * @param K Hidden dimension
 */
void gemm_q4_k_ref(float *Y,
                   const void *W,
                   const float *X,
                   int M, int N, int K)
{
    /* For each column in batch, use the dispatching gemv_q4_k
     * which automatically selects AVX/AVX-512/scalar based on CPU */
    for (int n = 0; n < N; n++) {
        gemv_q4_k(&Y[n * M], W, &X[n * K], M, K);
    }
}

#ifdef __AVX512F__
/**
 * @brief Matrix-matrix multiply with Q4_K weights (AVX-512)
 *
 * Processes multiple batch elements to improve weight reuse.
 */
void gemm_q4_k_avx512(float *Y,
                      const void *W,
                      const float *X,
                      int M, int N, int K)
{
    const block_q4_K *blocks = (const block_q4_K *)W;
    const int blocks_per_row = K / QK_K;

    /* Process 4 batch elements at a time for better register utilization */
    const int N4 = N / 4 * 4;

    for (int row = 0; row < M; row++) {
        /* Batch of 4 */
        for (int n = 0; n < N4; n += 4) {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            for (int b = 0; b < blocks_per_row; b++) {
                const block_q4_K *block = &blocks[row * blocks_per_row + b];
                const float d = GGML_FP16_TO_FP32(block->d);
                const float dmin = GGML_FP16_TO_FP32(block->dmin);

                uint8_t sc[8], m_arr[8];
                unpack_q4_k_scales(block->scales, sc, m_arr);

                for (int sub = 0; sub < 8; sub++) {
                    const float scale = d * (float)sc[sub];
                    const float min_val = dmin * (float)m_arr[sub];
                    const __m512 vscale = _mm512_set1_ps(scale);
                    const __m512 vmin = _mm512_set1_ps(min_val);
                    const __m512i offset = _mm512_set1_epi32(8);
                    const __m512i mask_lo = _mm512_set1_epi32(0x0F);

                    const uint8_t *qs = &block->qs[sub * 16];
                    const int x_offset = b * QK_K + sub * 32;

                    /* Dequantize weights (same for all batch elements) */
                    __m128i packed = _mm_loadu_si128((const __m128i *)qs);
                    __m512i bytes = _mm512_cvtepu8_epi32(packed);

                    __m512i lo = _mm512_sub_epi32(_mm512_and_epi32(bytes, mask_lo), offset);
                    __m512i hi = _mm512_sub_epi32(_mm512_srli_epi32(bytes, 4), offset);

                    __m512 w_lo = _mm512_fmadd_ps(_mm512_cvtepi32_ps(lo), vscale, vmin);
                    __m512 w_hi = _mm512_fmadd_ps(_mm512_cvtepi32_ps(hi), vscale, vmin);

                    /* Load inputs for 4 batch elements and accumulate */
                    /* (simplified - full impl would handle interleaving) */
                    for (int bn = 0; bn < 4; bn++) {
                        const float *xp = &X[(n + bn) * K + x_offset];

                        __m512 x_even = _mm512_set_ps(
                            xp[30], xp[28], xp[26], xp[24], xp[22], xp[20], xp[18], xp[16],
                            xp[14], xp[12], xp[10], xp[8], xp[6], xp[4], xp[2], xp[0]);
                        __m512 x_odd = _mm512_set_ps(
                            xp[31], xp[29], xp[27], xp[25], xp[23], xp[21], xp[19], xp[17],
                            xp[15], xp[13], xp[11], xp[9], xp[7], xp[5], xp[3], xp[1]);

                        __m512 *acc = (bn == 0) ? &acc0 : (bn == 1) ? &acc1 :
                                      (bn == 2) ? &acc2 : &acc3;
                        *acc = _mm512_fmadd_ps(w_lo, x_even, *acc);
                        *acc = _mm512_fmadd_ps(w_hi, x_odd, *acc);
                    }
                }
            }

            Y[(n + 0) * M + row] = _mm512_reduce_add_ps(acc0);
            Y[(n + 1) * M + row] = _mm512_reduce_add_ps(acc1);
            Y[(n + 2) * M + row] = _mm512_reduce_add_ps(acc2);
            Y[(n + 3) * M + row] = _mm512_reduce_add_ps(acc3);
        }

        /* Remainder */
        for (int n = N4; n < N; n++) {
            __m512 acc = _mm512_setzero_ps();

            for (int b = 0; b < blocks_per_row; b++) {
                const block_q4_K *block = &blocks[row * blocks_per_row + b];
                const float d = GGML_FP16_TO_FP32(block->d);
                const float dmin = GGML_FP16_TO_FP32(block->dmin);

                uint8_t sc[8], m_arr[8];
                unpack_q4_k_scales(block->scales, sc, m_arr);

                for (int sub = 0; sub < 8; sub++) {
                    const float scale = d * (float)sc[sub];
                    const float min_val = dmin * (float)m_arr[sub];
                    const __m512 vscale = _mm512_set1_ps(scale);
                    const __m512 vmin = _mm512_set1_ps(min_val);
                    const __m512i offset = _mm512_set1_epi32(8);
                    const __m512i mask_lo = _mm512_set1_epi32(0x0F);

                    const uint8_t *qs = &block->qs[sub * 16];
                    const float *xp = &X[n * K + b * QK_K + sub * 32];

                    __m128i packed = _mm_loadu_si128((const __m128i *)qs);
                    __m512i bytes = _mm512_cvtepu8_epi32(packed);

                    __m512i lo = _mm512_sub_epi32(_mm512_and_epi32(bytes, mask_lo), offset);
                    __m512i hi = _mm512_sub_epi32(_mm512_srli_epi32(bytes, 4), offset);

                    __m512 w_lo = _mm512_fmadd_ps(_mm512_cvtepi32_ps(lo), vscale, vmin);
                    __m512 w_hi = _mm512_fmadd_ps(_mm512_cvtepi32_ps(hi), vscale, vmin);

                    __m512 x_even = _mm512_set_ps(
                        xp[30], xp[28], xp[26], xp[24], xp[22], xp[20], xp[18], xp[16],
                        xp[14], xp[12], xp[10], xp[8], xp[6], xp[4], xp[2], xp[0]);
                    __m512 x_odd = _mm512_set_ps(
                        xp[31], xp[29], xp[27], xp[25], xp[23], xp[21], xp[19], xp[17],
                        xp[15], xp[13], xp[11], xp[9], xp[7], xp[5], xp[3], xp[1]);

                    acc = _mm512_fmadd_ps(w_lo, x_even, acc);
                    acc = _mm512_fmadd_ps(w_hi, x_odd, acc);
                }
            }

            Y[n * M + row] = _mm512_reduce_add_ps(acc);
        }
    }
}
#endif /* __AVX512F__ */

/**
 * @brief Auto-dispatch GEMM based on available SIMD
 */
void gemm_q4_k(float *Y,
               const void *W,
               const float *X,
               int M, int N, int K)
{
    /* Use reference implementation for correctness
     * TODO: Fix AVX-512 version to match llama.cpp layout */
    gemm_q4_k_ref(Y, W, X, M, N, K);
}

/* ============================================================================
 * Dot Product: Single row dot product with Q4_K weights
 * Used internally and for testing.
 * ============================================================================ */

/**
 * @brief Compute dot product of Q4_K row with FP32 vector
 *
 * @param w_q4k Q4_K blocks for one row
 * @param x FP32 input vector
 * @param K Vector length (must be multiple of 256)
 * @return Dot product result
 */
float dot_q4_k(const void *w_q4k, const float *x, int K)
{
    float result;
    gemv_q4_k(&result, w_q4k, x, 1, K);
    return result;
}

/* ============================================================================
 * Backward Pass: Gradient w.r.t. Input
 *
 * Given: dL/dY (gradient of loss w.r.t. output)
 * Compute: dL/dX = W^T @ dL/dY
 *
 * For quantized weights, we dequantize on-the-fly during backprop.
 * Weight gradients are NOT computed (weights are frozen).
 * For fine-tuning, use LoRA adapters which maintain FP32 gradients separately.
 * ============================================================================ */

/**
 * @brief Backward pass: compute input gradient (scalar reference)
 *
 * @param dX Output gradient w.r.t. input [K]
 * @param W Weight matrix in Q4_K format [M x K]
 * @param dY Gradient w.r.t. output [M]
 * @param M Number of output rows
 * @param K Number of columns (input dimension)
 */
void gemv_q4_k_backward_ref(float *dX,
                            const void *W,
                            const float *dY,
                            int M, int K)
{
    const block_q4_K *blocks = (const block_q4_K *)W;
    const int blocks_per_row = K / QK_K;

    /* Zero output gradient */
    memset(dX, 0, K * sizeof(float));

    /* Accumulate: dX += W^T @ dY
     * Uses llama.cpp layout: 4 iterations of 64 weights each */
    for (int row = 0; row < M; row++) {
        const float dy = dY[row];

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q4_K *block = &blocks[row * blocks_per_row + b];
            const float d = CK_FP16_TO_FP32(block->d);
            const float dmin = CK_FP16_TO_FP32(block->dmin);

            uint8_t sc[8], m[8];
            unpack_q4_k_scales(block->scales, sc, m);

            /* llama.cpp layout: 4 iterations of 64 weights each */
            for (int iter = 0; iter < 4; iter++) {
                const float d1 = d * (float)sc[2 * iter];
                const float m1 = dmin * (float)m[2 * iter];
                const float d2 = d * (float)sc[2 * iter + 1];
                const float m2 = dmin * (float)m[2 * iter + 1];

                const uint8_t *qs = &block->qs[iter * 32];
                float *dxp = &dX[b * QK_K + iter * 64];

                /* First 32 weights: low nibbles */
                for (int l = 0; l < 32; l++) {
                    const int q = (qs[l] & 0x0F);
                    const float w = d1 * (float)q - m1;
                    dxp[l] += w * dy;
                }

                /* Next 32 weights: high nibbles */
                for (int l = 0; l < 32; l++) {
                    const int q = (qs[l] >> 4);
                    const float w = d2 * (float)q - m2;
                    dxp[32 + l] += w * dy;
                }
            }
        }
    }
}

#ifdef __AVX512F__
/**
 * @brief Backward pass with AVX-512
 *
 * Uses llama.cpp layout: 4 iterations of 64 weights each
 */
void gemv_q4_k_backward_avx512(float *dX,
                               const void *W,
                               const float *dY,
                               int M, int K)
{
    const block_q4_K *blocks = (const block_q4_K *)W;
    const int blocks_per_row = K / QK_K;
    const __m512i mask_lo = _mm512_set1_epi32(0x0F);

    /* Zero output */
    memset(dX, 0, K * sizeof(float));

    for (int row = 0; row < M; row++) {
        const __m512 vdy = _mm512_set1_ps(dY[row]);

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q4_K *block = &blocks[row * blocks_per_row + b];
            const float d = CK_FP16_TO_FP32(block->d);
            const float dmin = CK_FP16_TO_FP32(block->dmin);

            uint8_t sc[8], m_arr[8];
            unpack_q4_k_scales(block->scales, sc, m_arr);

            /* llama.cpp layout: 4 iterations of 64 weights each */
            for (int iter = 0; iter < 4; iter++) {
                const float d1 = d * (float)sc[2 * iter];
                const float m1 = dmin * (float)m_arr[2 * iter];
                const float d2 = d * (float)sc[2 * iter + 1];
                const float m2 = dmin * (float)m_arr[2 * iter + 1];

                const __m512 vd1 = _mm512_set1_ps(d1);
                const __m512 vm1 = _mm512_set1_ps(m1);
                const __m512 vd2 = _mm512_set1_ps(d2);
                const __m512 vm2 = _mm512_set1_ps(m2);

                const uint8_t *qs = &block->qs[iter * 32];
                float *dxp = &dX[b * QK_K + iter * 64];

                /* Process first 32 weights (low nibbles) */
                for (int chunk = 0; chunk < 2; chunk++) {
                    __m128i packed = _mm_loadu_si128((const __m128i *)&qs[chunk * 16]);
                    __m512i bytes = _mm512_cvtepu8_epi32(packed);
                    __m512i lo = _mm512_and_epi32(bytes, mask_lo);
                    /* w = d1 * q - m1 */
                    __m512 w = _mm512_fnmadd_ps(_mm512_set1_ps(1.0f), vm1,
                               _mm512_mul_ps(_mm512_cvtepi32_ps(lo), vd1));
                    __m512 grad = _mm512_mul_ps(w, vdy);
                    __m512 existing = _mm512_loadu_ps(&dxp[chunk * 16]);
                    _mm512_storeu_ps(&dxp[chunk * 16], _mm512_add_ps(existing, grad));
                }

                /* Process next 32 weights (high nibbles) */
                for (int chunk = 0; chunk < 2; chunk++) {
                    __m128i packed = _mm_loadu_si128((const __m128i *)&qs[chunk * 16]);
                    __m512i bytes = _mm512_cvtepu8_epi32(packed);
                    __m512i hi = _mm512_srli_epi32(bytes, 4);
                    /* w = d2 * q - m2 */
                    __m512 w = _mm512_fnmadd_ps(_mm512_set1_ps(1.0f), vm2,
                               _mm512_mul_ps(_mm512_cvtepi32_ps(hi), vd2));
                    __m512 grad = _mm512_mul_ps(w, vdy);
                    __m512 existing = _mm512_loadu_ps(&dxp[32 + chunk * 16]);
                    _mm512_storeu_ps(&dxp[32 + chunk * 16], _mm512_add_ps(existing, grad));
                }
            }
        }
    }
}
#endif

/**
 * @brief Auto-dispatch backward
 */
void gemv_q4_k_backward(float *dX,
                        const void *W,
                        const float *dY,
                        int M, int K)
{
#ifdef __AVX512F__
    gemv_q4_k_backward_avx512(dX, W, dY, M, K);
#else
    gemv_q4_k_backward_ref(dX, W, dY, M, K);
#endif
}

/**
 * @brief Batched backward pass
 */
void gemm_q4_k_backward(float *dX,
                        const void *W,
                        const float *dY,
                        int M, int N, int K)
{
    for (int n = 0; n < N; n++) {
        gemv_q4_k_backward(&dX[n * K], W, &dY[n * M], M, K);
    }
}

/* ============================================================================
 * Engine-compatible wrapper: GEMM_NT with Q4_K weights
 *
 * The core q4_k kernels in this file use the convention:
 *   - W: [M_out x K] (quantized row-major)
 *   - X: [N_batch x K] (fp32)
 *   - Y: [N_batch x M_out] (fp32)
 *
 * The C-Kernel-Engine convention for NN weights uses:
 *   - A: [M_tokens x K] (fp32)
 *   - B: [N_out x K] (quantized row-major, transposed layout)
 *   - C: [M_tokens x N_out] (fp32)
 *
 * This wrapper swaps (M_out, N_batch) to match the engine layout and applies
 * an optional bias.
 * ============================================================================ */

void gemm_nt_q4_k(const float *A,
                  const void *B,
                  const float *bias,
                  float *C,
                  int M, int N, int K)
{
    if (!A || !B || !C) {
        return;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    /* gemm_q4_k produces Y as [batch x M_out]. Here:
     *   batch = M (tokens)
     *   M_out = N (output channels) */
    gemm_q4_k(C, B, A, /*M_out=*/N, /*N_batch=*/M, K);

    if (!bias) {
        return;
    }

    for (int i = 0; i < M; ++i) {
        float *row = C + (size_t)i * (size_t)N;
        for (int j = 0; j < N; ++j) {
            row[j] += bias[j];
        }
    }
}
