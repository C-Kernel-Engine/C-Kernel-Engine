/**
 * @file fused_rmsnorm_linear.c
 * @brief Fused RMSNorm + Linear (GEMV) kernel
 *
 * FUSION BENEFIT:
 * ===============
 * Unfused:
 *   RMSNorm(x) → [DRAM write: norm_out] → Quantize → [DRAM write: q8] → GEMV
 *   Total DRAM: 2 writes + 2 reads = 4 * hidden_size bytes
 *
 * Fused:
 *   RMSNorm(x) → [registers] → Quantize → [stack/L1: q8] → GEMV
 *   Total DRAM: 0 intermediate writes/reads
 *
 * Expected: 2-4x memory traffic reduction for this operation
 */

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "ckernel_quant.h"

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

/* Forward declarations */
void gemv_q4_k_q8_k(float *y, const void *W, const void *x_q8, int M, int K);

/* Inline quantization helper - same as quantize_row_q8_k but operates on
 * normalized values that may still be in registers/cache */
static inline int ck_nearest_int_fused(float fval) {
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

#if defined(__AVX__) && !defined(__AVX512F__)
static inline float hsum256_ps_fused(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}
#endif

/**
 * @brief Fused RMSNorm + Q4_K Linear projection
 *
 * Computes: y = Linear(RMSNorm(x))
 * where Linear uses Q4_K weights and Q8_K activations internally.
 *
 * The key optimization is that the normalized values never touch DRAM -
 * they go directly from RMSNorm computation to Q8_K quantization to GEMV.
 *
 * @param y         Output (FP32), shape [M]
 * @param x         Input hidden state (FP32), shape [K]
 * @param gamma     RMSNorm scale weights (FP32), shape [K]
 * @param W_q4k     Linear weights in Q4_K format, shape [M, K]
 * @param M         Output dimension (e.g., 3 * hidden for QKV)
 * @param K         Input dimension (hidden_size)
 * @param eps       RMSNorm epsilon (typically 1e-5 or 1e-6)
 */
void fused_rmsnorm_linear_q4k(float *y,
                               const float *x,
                               const float *gamma,
                               const void *W_q4k,
                               int M, int K,
                               float eps)
{
    if (!y || !x || !gamma || !W_q4k || M <= 0 || K <= 0) {
        return;
    }

    assert(K % QK_K == 0);
    const int nb = K / QK_K;  /* Number of Q8_K blocks */

    /* Stack-allocated Q8_K buffer - stays in L1/L2 cache */
    /* Max supported K = 8192 (8 blocks of 256) */
    block_q8_K q8_buffer[32];  /* 32 * ~260 bytes = ~8KB on stack */
    assert(nb <= 32 && "K too large for stack buffer");

    /* ================================================================
     * PHASE 1: Compute RMSNorm and quantize to Q8_K
     *          Result stays in stack (L1/L2), never touches DRAM
     * ================================================================ */

#if defined(__AVX512F__)
    /* AVX-512: Compute sum of squares */
    __m512 sum_sq_vec = _mm512_setzero_ps();
    int d = 0;
    for (; d + 16 <= K; d += 16) {
        __m512 xv = _mm512_loadu_ps(&x[d]);
        sum_sq_vec = _mm512_fmadd_ps(xv, xv, sum_sq_vec);
    }
    float sum_sq = _mm512_reduce_add_ps(sum_sq_vec);
    for (; d < K; ++d) {
        sum_sq += x[d] * x[d];
    }

#elif defined(__AVX__)
    /* AVX: Compute sum of squares */
    __m256 sum_sq_vec = _mm256_setzero_ps();
    int d = 0;
    for (; d + 8 <= K; d += 8) {
        __m256 xv = _mm256_loadu_ps(&x[d]);
        __m256 xv_sq = _mm256_mul_ps(xv, xv);
        sum_sq_vec = _mm256_add_ps(sum_sq_vec, xv_sq);
    }
    float sum_sq = hsum256_ps_fused(sum_sq_vec);
    for (; d < K; ++d) {
        sum_sq += x[d] * x[d];
    }

#else
    /* Scalar fallback */
    double sum_sq = 0.0;
    for (int d = 0; d < K; ++d) {
        double v = (double)x[d];
        sum_sq += v * v;
    }
#endif

    float mean_sq = (float)sum_sq / (float)K;
    float rstd = 1.0f / sqrtf(mean_sq + eps);

    /* ================================================================
     * PHASE 2: Apply RMSNorm and quantize to Q8_K in one pass
     *          Normalized values go directly to Q8_K blocks
     * ================================================================ */

    for (int i = 0; i < nb; ++i) {
        const float *x_block = x + i * QK_K;
        const float *g_block = gamma + i * QK_K;

        /* Find max absolute value for this block's normalized output */
        float max_val = 0.0f;
        float amax = 0.0f;

#if defined(__AVX512F__)
        __m512 rstd_vec = _mm512_set1_ps(rstd);
        __m512 max_vec = _mm512_setzero_ps();
        __m512 sign_mask = _mm512_set1_ps(-0.0f);

        for (int j = 0; j < QK_K; j += 16) {
            __m512 xv = _mm512_loadu_ps(&x_block[j]);
            __m512 gv = _mm512_loadu_ps(&g_block[j]);
            __m512 norm = _mm512_mul_ps(_mm512_mul_ps(xv, rstd_vec), gv);
            __m512 abs_norm = _mm512_andnot_ps(sign_mask, norm);
            max_vec = _mm512_max_ps(max_vec, abs_norm);

            /* Track max with sign for scale computation */
            __mmask16 gt_mask = _mm512_cmp_ps_mask(abs_norm, _mm512_set1_ps(amax), _CMP_GT_OQ);
            if (gt_mask) {
                float temp_amax = _mm512_reduce_max_ps(abs_norm);
                if (temp_amax > amax) {
                    amax = temp_amax;
                    /* Find the actual max value with sign */
                    for (int k = 0; k < 16; ++k) {
                        float v = x_block[j + k] * rstd * g_block[j + k];
                        if (fabsf(v) >= amax - 1e-6f) {
                            max_val = v;
                            break;
                        }
                    }
                }
            }
        }
        amax = _mm512_reduce_max_ps(max_vec);

#elif defined(__AVX__)
        __m256 rstd_vec = _mm256_set1_ps(rstd);

        for (int j = 0; j < QK_K; j += 8) {
            __m256 xv = _mm256_loadu_ps(&x_block[j]);
            __m256 gv = _mm256_loadu_ps(&g_block[j]);
            __m256 norm = _mm256_mul_ps(_mm256_mul_ps(xv, rstd_vec), gv);

            /* Check each element for max */
            float norm_arr[8];
            _mm256_storeu_ps(norm_arr, norm);
            for (int k = 0; k < 8; ++k) {
                float av = fabsf(norm_arr[k]);
                if (av > amax) {
                    amax = av;
                    max_val = norm_arr[k];
                }
            }
        }

#else
        for (int j = 0; j < QK_K; ++j) {
            float norm = x_block[j] * rstd * g_block[j];
            float av = fabsf(norm);
            if (av > amax) {
                amax = av;
                max_val = norm;
            }
        }
#endif

        /* Handle zero block */
        if (amax < 1e-10f) {
            q8_buffer[i].d = 0.0f;
            memset(q8_buffer[i].qs, 0, sizeof(q8_buffer[i].qs));
            memset(q8_buffer[i].bsums, 0, sizeof(q8_buffer[i].bsums));
            continue;
        }

        /* Compute scale and quantize */
        const float iscale = -127.0f / max_val;
        q8_buffer[i].d = 1.0f / iscale;

        /* Quantize and compute bsums */
        for (int j = 0; j < QK_K; ++j) {
            float norm = x_block[j] * rstd * g_block[j];
            int v = ck_nearest_int_fused(iscale * norm);
            v = (v > 127) ? 127 : ((v < -128) ? -128 : v);
            q8_buffer[i].qs[j] = (int8_t)v;
        }

        /* Compute block sums (16 elements each) */
        for (int j = 0; j < QK_K / 16; ++j) {
            int sum = 0;
            const int8_t *qs = &q8_buffer[i].qs[j * 16];
            for (int k = 0; k < 16; ++k) {
                sum += qs[k];
            }
            q8_buffer[i].bsums[j] = (int16_t)sum;
        }
    }

    /* ================================================================
     * PHASE 3: GEMV with Q4_K weights and Q8_K activations
     *          Q8_K data is in stack (L1/L2), not DRAM
     * ================================================================ */

    gemv_q4_k_q8_k(y, W_q4k, q8_buffer, M, K);
}

/**
 * @brief Reference (unfused) implementation for correctness testing
 *
 * This is the SLOW version that does separate RMSNorm and GEMV calls,
 * with intermediate results going to DRAM.
 */
void unfused_rmsnorm_linear_q4k_ref(float *y,
                                     const float *x,
                                     const float *gamma,
                                     const void *W_q4k,
                                     int M, int K,
                                     float eps)
{
    if (!y || !x || !gamma || !W_q4k || M <= 0 || K <= 0) {
        return;
    }

    assert(K % QK_K == 0);
    const int nb = K / QK_K;

    /* Allocate intermediate buffer (this hits DRAM!) */
    float *norm_out = (float *)malloc(K * sizeof(float));
    block_q8_K *q8_buffer = (block_q8_K *)malloc(nb * sizeof(block_q8_K));

    if (!norm_out || !q8_buffer) {
        free(norm_out);
        free(q8_buffer);
        return;
    }

    /* Step 1: RMSNorm (writes to DRAM) */
    double sum_sq = 0.0;
    for (int d = 0; d < K; ++d) {
        sum_sq += (double)x[d] * (double)x[d];
    }
    float rstd = 1.0f / sqrtf((float)(sum_sq / K) + eps);

    for (int d = 0; d < K; ++d) {
        norm_out[d] = x[d] * rstd * gamma[d];  /* DRAM WRITE */
    }

    /* Step 2: Quantize (reads DRAM, writes DRAM) */
    extern void quantize_row_q8_k(const float *x, void *vy, int k);
    quantize_row_q8_k(norm_out, q8_buffer, K);  /* DRAM READ + WRITE */

    /* Step 3: GEMV (reads Q8_K from DRAM) */
    gemv_q4_k_q8_k(y, W_q4k, q8_buffer, M, K);  /* DRAM READ */

    free(norm_out);
    free(q8_buffer);
}

#ifdef FUSED_KERNEL_TEST
/* Simple correctness test */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
    const int K = 512;   /* Hidden size */
    const int M = 1536;  /* Output size (3 * hidden for QKV) */
    const int nb = K / QK_K;

    printf("Fused RMSNorm+Linear Test\n");
    printf("K=%d, M=%d, blocks=%d\n", K, M, nb);

    /* Allocate test data */
    float *x = (float *)aligned_alloc(64, K * sizeof(float));
    float *gamma = (float *)aligned_alloc(64, K * sizeof(float));
    float *y_fused = (float *)aligned_alloc(64, M * sizeof(float));
    float *y_unfused = (float *)aligned_alloc(64, M * sizeof(float));

    /* Initialize with random data */
    srand(42);
    for (int i = 0; i < K; ++i) {
        x[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        gamma[i] = (float)rand() / RAND_MAX * 0.5f + 0.75f;
    }

    /* Create dummy Q4_K weights (in real usage, these come from model) */
    block_q4_K *W = (block_q4_K *)aligned_alloc(64, M * nb * sizeof(block_q4_K));
    memset(W, 0, M * nb * sizeof(block_q4_K));
    for (int i = 0; i < M * nb; ++i) {
        W[i].d = 0x3C00;  /* 1.0 in FP16 */
        W[i].dmin = 0x0000;
    }

    /* Run both versions */
    printf("Running fused version...\n");
    fused_rmsnorm_linear_q4k(y_fused, x, gamma, W, M, K, 1e-5f);

    printf("Running unfused version...\n");
    unfused_rmsnorm_linear_q4k_ref(y_unfused, x, gamma, W, M, K, 1e-5f);

    /* Compare results */
    float max_diff = 0.0f;
    for (int i = 0; i < M; ++i) {
        float diff = fabsf(y_fused[i] - y_unfused[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("Max difference: %e\n", max_diff);
    printf("Test %s\n", max_diff < 1e-3f ? "PASSED" : "FAILED");

    free(x);
    free(gamma);
    free(y_fused);
    free(y_unfused);
    free(W);

    return max_diff < 1e-3f ? 0 : 1;
}
#endif
