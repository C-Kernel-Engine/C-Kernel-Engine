/*
 * rmsnorm_qkv.c - Fused RMSNorm + QKV Projection
 *
 * Part of C-Kernel-Engine v6.6 Fusion Kernels
 *
 * PROBLEM:
 *   Non-fused version does 4 DRAM round-trips for 'normed' buffer:
 *     rmsnorm(x, weight, normed);    // Write normed to DRAM
 *     gemv(wq, normed, q);           // Read normed from DRAM
 *     gemv(wk, normed, k);           // Read normed from DRAM
 *     gemv(wv, normed, v);           // Read normed from DRAM
 *
 * SOLUTION:
 *   Fused version keeps 'normed' in registers/L1, zero DRAM access:
 *     rmsnorm_qkv_fused(x, weight, wq, wk, wv, q, k, v);
 *
 * EXPECTED SPEEDUP: 1.5-2x for this operation
 */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <string.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "ckernel_quant.h"

/* ============================================================================
 * HELPER: RMSNorm computation (inline, result stays in registers)
 * ============================================================================ */

static inline float compute_rms_scale(const float *x, int n, float eps) {
    float sum_sq = 0.0f;

#ifdef __AVX2__
    __m256 vsum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        vsum = _mm256_fmadd_ps(vx, vx, vsum);
    }
    // Horizontal sum
    __m128 vlow = _mm256_castps256_ps128(vsum);
    __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    vlow = _mm_hadd_ps(vlow, vlow);
    vlow = _mm_hadd_ps(vlow, vlow);
    sum_sq = _mm_cvtss_f32(vlow);
    // Remainder
    for (; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
#else
    for (int i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
#endif

    float rms = sqrtf(sum_sq / (float)n + eps);
    return 1.0f / rms;
}

/* ============================================================================
 * FUSED KERNEL: RMSNorm + QKV Projection (FP32 weights)
 * ============================================================================ */

void rmsnorm_qkv_fp32_fused(
    const float *x,           /* [embed_dim] input hidden state */
    const float *rms_weight,  /* [embed_dim] RMSNorm gamma */
    const float *wq,          /* [q_dim, embed_dim] Q projection */
    const float *wk,          /* [kv_dim, embed_dim] K projection */
    const float *wv,          /* [kv_dim, embed_dim] V projection */
    float *q_out,             /* [q_dim] output Q */
    float *k_out,             /* [kv_dim] output K */
    float *v_out,             /* [kv_dim] output V */
    int embed_dim,            /* Hidden dimension */
    int q_dim,                /* Q output dimension (num_heads * head_dim) */
    int kv_dim,               /* KV output dimension (num_kv_heads * head_dim) */
    float eps                 /* RMSNorm epsilon (typically 1e-6) */
) {
    /* Step 1: Compute RMS scale factor (stays in register) */
    float scale = compute_rms_scale(x, embed_dim, eps);

    /* Step 2: Fused normalize + project
     *
     * Key insight: We compute normed[i] = x[i] * rms_weight[i] * scale
     * on-the-fly during the GEMV, never storing the full normed vector.
     *
     * For each output element:
     *   q[j] = sum_i( wq[j,i] * x[i] * rms_weight[i] * scale )
     *        = scale * sum_i( wq[j,i] * x[i] * rms_weight[i] )
     */

    /* Q projection */
    for (int j = 0; j < q_dim; j++) {
        float sum = 0.0f;
        const float *wq_row = wq + j * embed_dim;

#ifdef __AVX2__
        __m256 vsum = _mm256_setzero_ps();
        int i = 0;
        for (; i + 7 < embed_dim; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vrms = _mm256_loadu_ps(rms_weight + i);
            __m256 vw = _mm256_loadu_ps(wq_row + i);
            __m256 vnormed = _mm256_mul_ps(vx, vrms);
            vsum = _mm256_fmadd_ps(vw, vnormed, vsum);
        }
        /* Horizontal sum */
        __m128 vlow = _mm256_castps256_ps128(vsum);
        __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
        vlow = _mm_add_ps(vlow, vhigh);
        vlow = _mm_hadd_ps(vlow, vlow);
        vlow = _mm_hadd_ps(vlow, vlow);
        sum = _mm_cvtss_f32(vlow);
        /* Remainder */
        for (; i < embed_dim; i++) {
            sum += wq_row[i] * x[i] * rms_weight[i];
        }
#else
        for (int i = 0; i < embed_dim; i++) {
            sum += wq_row[i] * x[i] * rms_weight[i];
        }
#endif
        q_out[j] = sum * scale;
    }

    /* K projection */
    for (int j = 0; j < kv_dim; j++) {
        float sum = 0.0f;
        const float *wk_row = wk + j * embed_dim;

#ifdef __AVX2__
        __m256 vsum = _mm256_setzero_ps();
        int i = 0;
        for (; i + 7 < embed_dim; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vrms = _mm256_loadu_ps(rms_weight + i);
            __m256 vw = _mm256_loadu_ps(wk_row + i);
            __m256 vnormed = _mm256_mul_ps(vx, vrms);
            vsum = _mm256_fmadd_ps(vw, vnormed, vsum);
        }
        __m128 vlow = _mm256_castps256_ps128(vsum);
        __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
        vlow = _mm_add_ps(vlow, vhigh);
        vlow = _mm_hadd_ps(vlow, vlow);
        vlow = _mm_hadd_ps(vlow, vlow);
        sum = _mm_cvtss_f32(vlow);
        for (; i < embed_dim; i++) {
            sum += wk_row[i] * x[i] * rms_weight[i];
        }
#else
        for (int i = 0; i < embed_dim; i++) {
            sum += wk_row[i] * x[i] * rms_weight[i];
        }
#endif
        k_out[j] = sum * scale;
    }

    /* V projection */
    for (int j = 0; j < kv_dim; j++) {
        float sum = 0.0f;
        const float *wv_row = wv + j * embed_dim;

#ifdef __AVX2__
        __m256 vsum = _mm256_setzero_ps();
        int i = 0;
        for (; i + 7 < embed_dim; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vrms = _mm256_loadu_ps(rms_weight + i);
            __m256 vw = _mm256_loadu_ps(wv_row + i);
            __m256 vnormed = _mm256_mul_ps(vx, vrms);
            vsum = _mm256_fmadd_ps(vw, vnormed, vsum);
        }
        __m128 vlow = _mm256_castps256_ps128(vsum);
        __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
        vlow = _mm_add_ps(vlow, vhigh);
        vlow = _mm_hadd_ps(vlow, vlow);
        vlow = _mm_hadd_ps(vlow, vlow);
        sum = _mm_cvtss_f32(vlow);
        for (; i < embed_dim; i++) {
            sum += wv_row[i] * x[i] * rms_weight[i];
        }
#else
        for (int i = 0; i < embed_dim; i++) {
            sum += wv_row[i] * x[i] * rms_weight[i];
        }
#endif
        v_out[j] = sum * scale;
    }
}

/* ============================================================================
 * FUSED KERNEL: RMSNorm + QKV Projection (Q4_K quantized weights)
 *
 * This is the production version - weights are Q4_K quantized.
 * We dequantize on-the-fly during the fused operation.
 * ============================================================================ */

void rmsnorm_qkv_q4k_fused(
    const float *x,           /* [embed_dim] input hidden state */
    const float *rms_weight,  /* [embed_dim] RMSNorm gamma */
    const void *wq,           /* Q4_K quantized Q projection */
    const void *wk,           /* Q4_K quantized K projection */
    const void *wv,           /* Q4_K quantized V projection */
    float *q_out,             /* [q_dim] output Q */
    float *k_out,             /* [kv_dim] output K */
    float *v_out,             /* [kv_dim] output V */
    int embed_dim,            /* Hidden dimension */
    int q_dim,                /* Q output dimension */
    int kv_dim,               /* KV output dimension */
    float eps                 /* RMSNorm epsilon */
) {
    /* Step 1: Compute RMS scale */
    float scale = compute_rms_scale(x, embed_dim, eps);

    /* Step 2: Compute normalized input (we need this for Q4_K dequant fusion)
     *
     * For Q4_K, we can't easily fuse the normalization into the dequant loop
     * because the block structure is complex. So we compute normed[] first,
     * but keep it small enough to fit in L1 cache.
     *
     * TODO: For maximum performance, implement a true fused Q4_K GEMV
     * that dequantizes and multiplies by normed[i] in the same loop.
     */

    /* Allocate normed on stack (fits in L1 for typical embed_dim <= 4096) */
    float normed[4096];  /* Max supported embed_dim */
    if (embed_dim > 4096) {
        /* Fallback for very large models */
        return;  /* TODO: heap allocation */
    }

    /* Compute normed = x * rms_weight * scale */
#ifdef __AVX2__
    __m256 vscale = _mm256_set1_ps(scale);
    int i = 0;
    for (; i + 7 < embed_dim; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vrms = _mm256_loadu_ps(rms_weight + i);
        __m256 vn = _mm256_mul_ps(_mm256_mul_ps(vx, vrms), vscale);
        _mm256_storeu_ps(normed + i, vn);
    }
    for (; i < embed_dim; i++) {
        normed[i] = x[i] * rms_weight[i] * scale;
    }
#else
    for (int i = 0; i < embed_dim; i++) {
        normed[i] = x[i] * rms_weight[i] * scale;
    }
#endif

    /* Step 3: Q4_K GEMV with normed input
     *
     * Call existing Q4_K GEMV kernel with normed[] as input.
     * The normed[] buffer is in L1 cache, so this is still fast.
     *
     * Key insight: normed[] never leaves L1 cache because we use it
     * immediately in the next 3 GEMVs. This eliminates the DRAM write
     * that would happen in the non-fused version.
     */

    /* Declare external GEMV function */
    extern void gemv_q4_k(float *y, const void *W, const float *x, int M, int K);

    /* Q projection: q_out[q_dim] = wq[q_dim, embed_dim] @ normed[embed_dim] */
    gemv_q4_k(q_out, wq, normed, q_dim, embed_dim);

    /* K projection: k_out[kv_dim] = wk[kv_dim, embed_dim] @ normed[embed_dim] */
    gemv_q4_k(k_out, wk, normed, kv_dim, embed_dim);

    /* V projection: v_out[kv_dim] = wv[kv_dim, embed_dim] @ normed[embed_dim] */
    gemv_q4_k(v_out, wv, normed, kv_dim, embed_dim);
}

/* ============================================================================
 * NON-FUSED REFERENCE: For benchmarking comparison
 *
 * This is what we're comparing against. Call rmsnorm + 3x GEMV separately.
 * ============================================================================ */

void rmsnorm_qkv_separate_fp32(
    const float *x,
    const float *rms_weight,
    const float *wq,
    const float *wk,
    const float *wv,
    float *normed,            /* [embed_dim] intermediate buffer - DRAM write! */
    float *q_out,
    float *k_out,
    float *v_out,
    int embed_dim,
    int q_dim,
    int kv_dim,
    float eps
) {
    /* Step 1: RMSNorm - writes normed to DRAM */
    float scale = compute_rms_scale(x, embed_dim, eps);
    for (int i = 0; i < embed_dim; i++) {
        normed[i] = x[i] * rms_weight[i] * scale;
    }

    /* Step 2: Q projection - reads normed from DRAM */
    for (int j = 0; j < q_dim; j++) {
        float sum = 0.0f;
        for (int i = 0; i < embed_dim; i++) {
            sum += wq[j * embed_dim + i] * normed[i];
        }
        q_out[j] = sum;
    }

    /* Step 3: K projection - reads normed from DRAM */
    for (int j = 0; j < kv_dim; j++) {
        float sum = 0.0f;
        for (int i = 0; i < embed_dim; i++) {
            sum += wk[j * embed_dim + i] * normed[i];
        }
        k_out[j] = sum;
    }

    /* Step 4: V projection - reads normed from DRAM */
    for (int j = 0; j < kv_dim; j++) {
        float sum = 0.0f;
        for (int i = 0; i < embed_dim; i++) {
            sum += wv[j * embed_dim + i] * normed[i];
        }
        v_out[j] = sum;
    }
}
