/**
 * @file gemm_kernels_q6k.c
 * @brief GEMM/GEMV kernels with Q6_K quantized weights
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
 *   - Weights: Q6_K (6-bit k-quant, int8 scales)
 *   - Output: FP32
 *
 * Q6_K Format (256 weights per block):
 *   - d: FP16 super-block scale
 *   - ql: 128 bytes (low 4 bits of each weight)
 *   - qh: 64 bytes (high 2 bits of each weight)
 *   - scales: 16 int8 sub-block scales
 */

#include <stdint.h>
#include <stddef.h>
#include "ckernel_quant.h"

/* Include SIMD headers based on available extensions */
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__) || defined(__SSE4_1__)
#include <immintrin.h>
#endif

/* ============================================================================
 * GEMV: y = W @ x  (W is Q6_K, x and y are FP32)
 * ============================================================================ */

static float dot_q6_k_ref(const block_q6_K *w,
                          const float *x,
                          int K)
{
    const int blocks_per_row = K / QK_K;
    float sum = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const block_q6_K *block = &w[b];
        const float d = GGML_FP16_TO_FP32(block->d);

        const uint8_t *ql = block->ql;
        const uint8_t *qh = block->qh;
        const int8_t *sc = block->scales;
        const float *xp = x + (size_t)b * QK_K;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                sum += (d * (float)sc[is + 0] * (float)q1) * xp[l + 0];
                sum += (d * (float)sc[is + 2] * (float)q2) * xp[l + 32];
                sum += (d * (float)sc[is + 4] * (float)q3) * xp[l + 64];
                sum += (d * (float)sc[is + 6] * (float)q4) * xp[l + 96];
            }
            xp += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }

    return sum;
}

/* ============================================================================
 * AVX Implementation (256-bit)
 *
 * Q6_K is complex: 6-bit weights packed as 4+2 bits, with 16 sub-block scales.
 * Uses AVX for the final accumulation while keeping scalar dequantization
 * for simplicity and correctness.
 * ============================================================================ */

#if defined(__AVX__) && !defined(__AVX512F__)
static float dot_q6_k_avx(const block_q6_K *w,
                          const float *x,
                          int K)
{
    const int blocks_per_row = K / QK_K;

    /* Use 4 accumulators for better ILP */
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    for (int b = 0; b < blocks_per_row; ++b) {
        const block_q6_K *block = &w[b];
        const float d = GGML_FP16_TO_FP32(block->d);

        const uint8_t *ql = block->ql;
        const uint8_t *qh = block->qh;
        const int8_t *sc = block->scales;
        const float *xp = x + (size_t)b * QK_K;

        /* Process 256 weights in 2 iterations of 128 */
        for (int n = 0; n < QK_K; n += 128) {
            /* Process 32 elements at a time in groups of 8 for AVX */
            for (int l = 0; l < 32; l += 8) {
                const int is = l / 16;

                /* Dequantize 8 weights for each of the 4 streams */
                float dq0[8], dq1[8], dq2[8], dq3[8];

                for (int i = 0; i < 8; i++) {
                    int idx = l + i;
                    const int8_t q1 = (int8_t)((ql[idx + 0] & 0xF) | (((qh[idx] >> 0) & 3) << 4)) - 32;
                    const int8_t q2 = (int8_t)((ql[idx + 32] & 0xF) | (((qh[idx] >> 2) & 3) << 4)) - 32;
                    const int8_t q3 = (int8_t)((ql[idx + 0] >> 4) | (((qh[idx] >> 4) & 3) << 4)) - 32;
                    const int8_t q4 = (int8_t)((ql[idx + 32] >> 4) | (((qh[idx] >> 6) & 3) << 4)) - 32;

                    const int is_i = (l + i) / 16;
                    dq0[i] = d * (float)sc[is_i + 0] * (float)q1;
                    dq1[i] = d * (float)sc[is_i + 2] * (float)q2;
                    dq2[i] = d * (float)sc[is_i + 4] * (float)q3;
                    dq3[i] = d * (float)sc[is_i + 6] * (float)q4;
                }

                __m256 vw0 = _mm256_loadu_ps(dq0);
                __m256 vw1 = _mm256_loadu_ps(dq1);
                __m256 vw2 = _mm256_loadu_ps(dq2);
                __m256 vw3 = _mm256_loadu_ps(dq3);

                __m256 vx0 = _mm256_loadu_ps(&xp[l + 0]);
                __m256 vx1 = _mm256_loadu_ps(&xp[l + 32]);
                __m256 vx2 = _mm256_loadu_ps(&xp[l + 64]);
                __m256 vx3 = _mm256_loadu_ps(&xp[l + 96]);

                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(vw0, vx0));
                acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(vw1, vx1));
                acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(vw2, vx2));
                acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(vw3, vx3));
            }
            xp += 128;
            ql += 64;
            qh += 32;
            sc += 8;
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

    return _mm_cvtss_f32(sum128);
}
#endif /* __AVX__ && !__AVX512F__ */

void gemv_q6_k(float *y,
               const void *W,
               const float *x,
               int M, int K)
{
    if (!y || !W || !x) {
        return;
    }
    if (M <= 0 || K <= 0) {
        return;
    }
    // TEMPORARILY DISABLE NEW AVX KERNELS - USE REFERENCE ONLY

    const block_q6_K *blocks = (const block_q6_K *)W;
    const int blocks_per_row = K / QK_K;

    for (int row = 0; row < M; ++row) {
        const block_q6_K *w_row = blocks + (size_t)row * (size_t)blocks_per_row;
#if defined(__AVX__) && !defined(__AVX512F__)
        y[row] = dot_q6_k_avx(w_row, x, K);
#else
        y[row] = dot_q6_k_ref(w_row, x, K);
#endif
    }
}

void gemm_q6_k(float *Y,
               const void *W,
               const float *X,
               int M, int N, int K)
{
    if (!Y || !W || !X) {
        return;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    for (int n = 0; n < N; ++n) {
        gemv_q6_k(&Y[n * M], W, &X[n * K], M, K);
    }
}

void gemm_nt_q6_k(const float *A,
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

    /* gemm_q6_k produces Y as [batch x M_out] where:
     *   batch = M (tokens)
     *   M_out = N (output channels) */
    gemm_q6_k(C, B, A, /*M_out=*/N, /*N_batch=*/M, K);

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

/* Reference implementation - used as fallback from SSE when K not aligned */
void gemm_nt_q6_k_ref(const float *A,
                      const void *B,
                      const float *bias,
                      float *C,
                      int M, int N, int K)
{
    gemm_nt_q6_k(A, B, bias, C, M, N, K);
}