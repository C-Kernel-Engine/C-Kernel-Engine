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
 * TRUE SIMD FUSION: RMSNorm + QKV (FP32 weights) - VARIATION 2
 *
 * KEY INSIGHT: Process OUTPUT cache-line by cache-line.
 * For each output cache line:
 *   - Keep accumulators in YMM/ZMM registers
 *   - For each INPUT cache line:
 *     - Compute normed chunk IN REGISTER (never stored to memory!)
 *     - Use immediately for all output accumulators via FMADD
 *   - Only store when output cache line is complete
 *
 * This is TRUE register-level fusion:
 *   - normed[] NEVER touches L1 cache
 *   - Each input cache line loaded ONCE, used for multiple outputs
 *   - Memory traffic: input + weights + output (no intermediate!)
 *
 * Expected speedup: 1.5-2x over separate kernels
 * ============================================================================ */

#ifdef __AVX2__
static inline float hsum256_ps(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    vlow = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, vlow);
    vlow = _mm_add_ss(vlow, shuf);
    return _mm_cvtss_f32(vlow);
}
#endif

void rmsnorm_qkv_fp32_fused_v2(
    const float *x,           /* [embed_dim] input hidden state */
    const float *rms_weight,  /* [embed_dim] RMSNorm gamma */
    const float *wq,          /* [q_dim, embed_dim] Q projection (row-major) */
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
    /* Step 1: Compute RMS scale (requires full pass - unavoidable) */
    float scale = compute_rms_scale(x, embed_dim, eps);

#ifdef __AVX2__
    __m256 vscale = _mm256_set1_ps(scale);

    /* ═══════════════════════════════════════════════════════════════════════
     * Q PROJECTION: Process 8 outputs at a time (one cache line)
     *
     * For each output cache line [j:j+8]:
     *   acc[0..7] = 0
     *   For each input cache line [i:i+8]:
     *     normed = x[i:i+8] * rms_weight[i:i+8] * scale  ← IN REGISTER!
     *     acc[k] += W[j+k, i:i+8] · normed              ← FMADD
     *   Store q_out[j:j+8]
     * ═══════════════════════════════════════════════════════════════════════ */

    for (int j = 0; j < q_dim; j += 8) {
        /* 8 accumulators for 8 output elements - all in YMM registers */
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps();
        __m256 acc7 = _mm256_setzero_ps();

        /* Process input in cache-line chunks */
        int i = 0;
        for (; i + 7 < embed_dim; i += 8) {
            /* Load input cache line and normalize IN REGISTER */
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vrms = _mm256_loadu_ps(rms_weight + i);
            __m256 normed = _mm256_mul_ps(_mm256_mul_ps(vx, vrms), vscale);
            /* normed is now in YMM register - NEVER touches memory! */

            /* Load 8 weight rows and accumulate */
            /* Each row is at wq[(j+k)*embed_dim + i] */
            if (j + 0 < q_dim) {
                __m256 w0 = _mm256_loadu_ps(wq + (j+0)*embed_dim + i);
                acc0 = _mm256_fmadd_ps(w0, normed, acc0);
            }
            if (j + 1 < q_dim) {
                __m256 w1 = _mm256_loadu_ps(wq + (j+1)*embed_dim + i);
                acc1 = _mm256_fmadd_ps(w1, normed, acc1);
            }
            if (j + 2 < q_dim) {
                __m256 w2 = _mm256_loadu_ps(wq + (j+2)*embed_dim + i);
                acc2 = _mm256_fmadd_ps(w2, normed, acc2);
            }
            if (j + 3 < q_dim) {
                __m256 w3 = _mm256_loadu_ps(wq + (j+3)*embed_dim + i);
                acc3 = _mm256_fmadd_ps(w3, normed, acc3);
            }
            if (j + 4 < q_dim) {
                __m256 w4 = _mm256_loadu_ps(wq + (j+4)*embed_dim + i);
                acc4 = _mm256_fmadd_ps(w4, normed, acc4);
            }
            if (j + 5 < q_dim) {
                __m256 w5 = _mm256_loadu_ps(wq + (j+5)*embed_dim + i);
                acc5 = _mm256_fmadd_ps(w5, normed, acc5);
            }
            if (j + 6 < q_dim) {
                __m256 w6 = _mm256_loadu_ps(wq + (j+6)*embed_dim + i);
                acc6 = _mm256_fmadd_ps(w6, normed, acc6);
            }
            if (j + 7 < q_dim) {
                __m256 w7 = _mm256_loadu_ps(wq + (j+7)*embed_dim + i);
                acc7 = _mm256_fmadd_ps(w7, normed, acc7);
            }
        }

        /* Handle remainder (scalar, rare for aligned dims) */
        for (; i < embed_dim; i++) {
            float normed_scalar = x[i] * rms_weight[i] * scale;
            if (j + 0 < q_dim) acc0 = _mm256_add_ps(acc0, _mm256_set1_ps(wq[(j+0)*embed_dim + i] * normed_scalar));
            if (j + 1 < q_dim) acc1 = _mm256_add_ps(acc1, _mm256_set1_ps(wq[(j+1)*embed_dim + i] * normed_scalar));
            if (j + 2 < q_dim) acc2 = _mm256_add_ps(acc2, _mm256_set1_ps(wq[(j+2)*embed_dim + i] * normed_scalar));
            if (j + 3 < q_dim) acc3 = _mm256_add_ps(acc3, _mm256_set1_ps(wq[(j+3)*embed_dim + i] * normed_scalar));
            if (j + 4 < q_dim) acc4 = _mm256_add_ps(acc4, _mm256_set1_ps(wq[(j+4)*embed_dim + i] * normed_scalar));
            if (j + 5 < q_dim) acc5 = _mm256_add_ps(acc5, _mm256_set1_ps(wq[(j+5)*embed_dim + i] * normed_scalar));
            if (j + 6 < q_dim) acc6 = _mm256_add_ps(acc6, _mm256_set1_ps(wq[(j+6)*embed_dim + i] * normed_scalar));
            if (j + 7 < q_dim) acc7 = _mm256_add_ps(acc7, _mm256_set1_ps(wq[(j+7)*embed_dim + i] * normed_scalar));
        }

        /* Horizontal sum and store output cache line */
        if (j + 0 < q_dim) q_out[j+0] = hsum256_ps(acc0);
        if (j + 1 < q_dim) q_out[j+1] = hsum256_ps(acc1);
        if (j + 2 < q_dim) q_out[j+2] = hsum256_ps(acc2);
        if (j + 3 < q_dim) q_out[j+3] = hsum256_ps(acc3);
        if (j + 4 < q_dim) q_out[j+4] = hsum256_ps(acc4);
        if (j + 5 < q_dim) q_out[j+5] = hsum256_ps(acc5);
        if (j + 6 < q_dim) q_out[j+6] = hsum256_ps(acc6);
        if (j + 7 < q_dim) q_out[j+7] = hsum256_ps(acc7);
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * K PROJECTION: Same pattern, smaller output
     * ═══════════════════════════════════════════════════════════════════════ */

    for (int j = 0; j < kv_dim; j += 8) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps();
        __m256 acc7 = _mm256_setzero_ps();

        int i = 0;
        for (; i + 7 < embed_dim; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vrms = _mm256_loadu_ps(rms_weight + i);
            __m256 normed = _mm256_mul_ps(_mm256_mul_ps(vx, vrms), vscale);

            if (j + 0 < kv_dim) acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(wk + (j+0)*embed_dim + i), normed, acc0);
            if (j + 1 < kv_dim) acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(wk + (j+1)*embed_dim + i), normed, acc1);
            if (j + 2 < kv_dim) acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(wk + (j+2)*embed_dim + i), normed, acc2);
            if (j + 3 < kv_dim) acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(wk + (j+3)*embed_dim + i), normed, acc3);
            if (j + 4 < kv_dim) acc4 = _mm256_fmadd_ps(_mm256_loadu_ps(wk + (j+4)*embed_dim + i), normed, acc4);
            if (j + 5 < kv_dim) acc5 = _mm256_fmadd_ps(_mm256_loadu_ps(wk + (j+5)*embed_dim + i), normed, acc5);
            if (j + 6 < kv_dim) acc6 = _mm256_fmadd_ps(_mm256_loadu_ps(wk + (j+6)*embed_dim + i), normed, acc6);
            if (j + 7 < kv_dim) acc7 = _mm256_fmadd_ps(_mm256_loadu_ps(wk + (j+7)*embed_dim + i), normed, acc7);
        }

        for (; i < embed_dim; i++) {
            float normed_scalar = x[i] * rms_weight[i] * scale;
            if (j + 0 < kv_dim) acc0 = _mm256_add_ps(acc0, _mm256_set1_ps(wk[(j+0)*embed_dim + i] * normed_scalar));
            if (j + 1 < kv_dim) acc1 = _mm256_add_ps(acc1, _mm256_set1_ps(wk[(j+1)*embed_dim + i] * normed_scalar));
            if (j + 2 < kv_dim) acc2 = _mm256_add_ps(acc2, _mm256_set1_ps(wk[(j+2)*embed_dim + i] * normed_scalar));
            if (j + 3 < kv_dim) acc3 = _mm256_add_ps(acc3, _mm256_set1_ps(wk[(j+3)*embed_dim + i] * normed_scalar));
            if (j + 4 < kv_dim) acc4 = _mm256_add_ps(acc4, _mm256_set1_ps(wk[(j+4)*embed_dim + i] * normed_scalar));
            if (j + 5 < kv_dim) acc5 = _mm256_add_ps(acc5, _mm256_set1_ps(wk[(j+5)*embed_dim + i] * normed_scalar));
            if (j + 6 < kv_dim) acc6 = _mm256_add_ps(acc6, _mm256_set1_ps(wk[(j+6)*embed_dim + i] * normed_scalar));
            if (j + 7 < kv_dim) acc7 = _mm256_add_ps(acc7, _mm256_set1_ps(wk[(j+7)*embed_dim + i] * normed_scalar));
        }

        if (j + 0 < kv_dim) k_out[j+0] = hsum256_ps(acc0);
        if (j + 1 < kv_dim) k_out[j+1] = hsum256_ps(acc1);
        if (j + 2 < kv_dim) k_out[j+2] = hsum256_ps(acc2);
        if (j + 3 < kv_dim) k_out[j+3] = hsum256_ps(acc3);
        if (j + 4 < kv_dim) k_out[j+4] = hsum256_ps(acc4);
        if (j + 5 < kv_dim) k_out[j+5] = hsum256_ps(acc5);
        if (j + 6 < kv_dim) k_out[j+6] = hsum256_ps(acc6);
        if (j + 7 < kv_dim) k_out[j+7] = hsum256_ps(acc7);
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * V PROJECTION: Same pattern
     * ═══════════════════════════════════════════════════════════════════════ */

    for (int j = 0; j < kv_dim; j += 8) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps();
        __m256 acc7 = _mm256_setzero_ps();

        int i = 0;
        for (; i + 7 < embed_dim; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vrms = _mm256_loadu_ps(rms_weight + i);
            __m256 normed = _mm256_mul_ps(_mm256_mul_ps(vx, vrms), vscale);

            if (j + 0 < kv_dim) acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(wv + (j+0)*embed_dim + i), normed, acc0);
            if (j + 1 < kv_dim) acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(wv + (j+1)*embed_dim + i), normed, acc1);
            if (j + 2 < kv_dim) acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(wv + (j+2)*embed_dim + i), normed, acc2);
            if (j + 3 < kv_dim) acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(wv + (j+3)*embed_dim + i), normed, acc3);
            if (j + 4 < kv_dim) acc4 = _mm256_fmadd_ps(_mm256_loadu_ps(wv + (j+4)*embed_dim + i), normed, acc4);
            if (j + 5 < kv_dim) acc5 = _mm256_fmadd_ps(_mm256_loadu_ps(wv + (j+5)*embed_dim + i), normed, acc5);
            if (j + 6 < kv_dim) acc6 = _mm256_fmadd_ps(_mm256_loadu_ps(wv + (j+6)*embed_dim + i), normed, acc6);
            if (j + 7 < kv_dim) acc7 = _mm256_fmadd_ps(_mm256_loadu_ps(wv + (j+7)*embed_dim + i), normed, acc7);
        }

        for (; i < embed_dim; i++) {
            float normed_scalar = x[i] * rms_weight[i] * scale;
            if (j + 0 < kv_dim) acc0 = _mm256_add_ps(acc0, _mm256_set1_ps(wv[(j+0)*embed_dim + i] * normed_scalar));
            if (j + 1 < kv_dim) acc1 = _mm256_add_ps(acc1, _mm256_set1_ps(wv[(j+1)*embed_dim + i] * normed_scalar));
            if (j + 2 < kv_dim) acc2 = _mm256_add_ps(acc2, _mm256_set1_ps(wv[(j+2)*embed_dim + i] * normed_scalar));
            if (j + 3 < kv_dim) acc3 = _mm256_add_ps(acc3, _mm256_set1_ps(wv[(j+3)*embed_dim + i] * normed_scalar));
            if (j + 4 < kv_dim) acc4 = _mm256_add_ps(acc4, _mm256_set1_ps(wv[(j+4)*embed_dim + i] * normed_scalar));
            if (j + 5 < kv_dim) acc5 = _mm256_add_ps(acc5, _mm256_set1_ps(wv[(j+5)*embed_dim + i] * normed_scalar));
            if (j + 6 < kv_dim) acc6 = _mm256_add_ps(acc6, _mm256_set1_ps(wv[(j+6)*embed_dim + i] * normed_scalar));
            if (j + 7 < kv_dim) acc7 = _mm256_add_ps(acc7, _mm256_set1_ps(wv[(j+7)*embed_dim + i] * normed_scalar));
        }

        if (j + 0 < kv_dim) v_out[j+0] = hsum256_ps(acc0);
        if (j + 1 < kv_dim) v_out[j+1] = hsum256_ps(acc1);
        if (j + 2 < kv_dim) v_out[j+2] = hsum256_ps(acc2);
        if (j + 3 < kv_dim) v_out[j+3] = hsum256_ps(acc3);
        if (j + 4 < kv_dim) v_out[j+4] = hsum256_ps(acc4);
        if (j + 5 < kv_dim) v_out[j+5] = hsum256_ps(acc5);
        if (j + 6 < kv_dim) v_out[j+6] = hsum256_ps(acc6);
        if (j + 7 < kv_dim) v_out[j+7] = hsum256_ps(acc7);
    }

#else
    /* Scalar fallback - same logic, no SIMD */
    for (int j = 0; j < q_dim; j++) {
        float sum = 0.0f;
        for (int i = 0; i < embed_dim; i++) {
            float normed = x[i] * rms_weight[i] * scale;
            sum += wq[j * embed_dim + i] * normed;
        }
        q_out[j] = sum;
    }
    for (int j = 0; j < kv_dim; j++) {
        float sum = 0.0f;
        for (int i = 0; i < embed_dim; i++) {
            float normed = x[i] * rms_weight[i] * scale;
            sum += wk[j * embed_dim + i] * normed;
        }
        k_out[j] = sum;
    }
    for (int j = 0; j < kv_dim; j++) {
        float sum = 0.0f;
        for (int i = 0; i < embed_dim; i++) {
            float normed = x[i] * rms_weight[i] * scale;
            sum += wv[j * embed_dim + i] * normed;
        }
        v_out[j] = sum;
    }
#endif
}

/* ============================================================================
 * TRUE SIMD FUSION V3: RMSNorm + QKV (FP32 weights)
 *
 * KEY FIX: Process Q, K, V SIMULTANEOUSLY in one pass through input!
 *
 * Previous versions had a flaw:
 *   v1: Recomputes normed for each Q row, then each K row, then each V row
 *       = 3 * q_dim * embed_dim FMA operations (tripled work!)
 *   v2: Same issue, just with 8-at-a-time grouping
 *
 * v3 approach:
 *   For each OUTPUT index j (0 to max(q_dim, kv_dim)):
 *     q_acc = k_acc = v_acc = 0
 *     For each INPUT chunk [i:i+8]:
 *       normed = x[i:i+8] * rms_weight[i:i+8] * scale  (computed ONCE!)
 *       q_acc += wq[j,i:i+8] · normed
 *       k_acc += wk[j,i:i+8] · normed  (if j < kv_dim)
 *       v_acc += wv[j,i:i+8] · normed  (if j < kv_dim)
 *     Store q_out[j], k_out[j], v_out[j]
 *
 * Benefits:
 *   - normed computed ONCE per input chunk, used 3x
 *   - Sequential weight access (good prefetch)
 *   - Minimal register pressure (3 accumulators + 1 normed)
 * ============================================================================ */

void rmsnorm_qkv_fp32_fused_v3(
    const float *x,           /* [embed_dim] input hidden state */
    const float *rms_weight,  /* [embed_dim] RMSNorm gamma */
    const float *wq,          /* [q_dim, embed_dim] Q projection (row-major) */
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
    /* Step 1: Compute RMS scale (requires full pass - unavoidable) */
    float scale = compute_rms_scale(x, embed_dim, eps);

#ifdef __AVX2__
    __m256 vscale = _mm256_set1_ps(scale);

    /* ═══════════════════════════════════════════════════════════════════════
     * Phase 1: Process Q outputs that have corresponding K,V outputs
     *          (j < kv_dim: compute Q, K, V together)
     * ═══════════════════════════════════════════════════════════════════════ */
    for (int j = 0; j < kv_dim; j++) {
        __m256 q_acc = _mm256_setzero_ps();
        __m256 k_acc = _mm256_setzero_ps();
        __m256 v_acc = _mm256_setzero_ps();

        int i = 0;
        for (; i + 7 < embed_dim; i += 8) {
            /* Load input and normalize - computed ONCE, used THREE times! */
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vrms = _mm256_loadu_ps(rms_weight + i);
            __m256 normed = _mm256_mul_ps(_mm256_mul_ps(vx, vrms), vscale);

            /* Load weight rows - sequential access per row */
            __m256 wq_row = _mm256_loadu_ps(wq + j * embed_dim + i);
            __m256 wk_row = _mm256_loadu_ps(wk + j * embed_dim + i);
            __m256 wv_row = _mm256_loadu_ps(wv + j * embed_dim + i);

            /* Accumulate - normed stays in register! */
            q_acc = _mm256_fmadd_ps(wq_row, normed, q_acc);
            k_acc = _mm256_fmadd_ps(wk_row, normed, k_acc);
            v_acc = _mm256_fmadd_ps(wv_row, normed, v_acc);
        }

        /* Handle remainder (scalar) */
        float q_sum = hsum256_ps(q_acc);
        float k_sum = hsum256_ps(k_acc);
        float v_sum = hsum256_ps(v_acc);

        for (; i < embed_dim; i++) {
            float normed = x[i] * rms_weight[i] * scale;
            q_sum += wq[j * embed_dim + i] * normed;
            k_sum += wk[j * embed_dim + i] * normed;
            v_sum += wv[j * embed_dim + i] * normed;
        }

        q_out[j] = q_sum;
        k_out[j] = k_sum;
        v_out[j] = v_sum;
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * Phase 2: Process remaining Q outputs (j >= kv_dim: Q only)
     *          This handles GQA where q_dim > kv_dim
     * ═══════════════════════════════════════════════════════════════════════ */
    for (int j = kv_dim; j < q_dim; j++) {
        __m256 q_acc = _mm256_setzero_ps();

        int i = 0;
        for (; i + 7 < embed_dim; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vrms = _mm256_loadu_ps(rms_weight + i);
            __m256 normed = _mm256_mul_ps(_mm256_mul_ps(vx, vrms), vscale);

            __m256 wq_row = _mm256_loadu_ps(wq + j * embed_dim + i);
            q_acc = _mm256_fmadd_ps(wq_row, normed, q_acc);
        }

        float q_sum = hsum256_ps(q_acc);
        for (; i < embed_dim; i++) {
            float normed = x[i] * rms_weight[i] * scale;
            q_sum += wq[j * embed_dim + i] * normed;
        }

        q_out[j] = q_sum;
    }

#else
    /* Scalar fallback - same simultaneous Q,K,V approach */
    for (int j = 0; j < kv_dim; j++) {
        float q_sum = 0.0f, k_sum = 0.0f, v_sum = 0.0f;
        for (int i = 0; i < embed_dim; i++) {
            float normed = x[i] * rms_weight[i] * scale;
            q_sum += wq[j * embed_dim + i] * normed;
            k_sum += wk[j * embed_dim + i] * normed;
            v_sum += wv[j * embed_dim + i] * normed;
        }
        q_out[j] = q_sum;
        k_out[j] = k_sum;
        v_out[j] = v_sum;
    }
    for (int j = kv_dim; j < q_dim; j++) {
        float q_sum = 0.0f;
        for (int i = 0; i < embed_dim; i++) {
            float normed = x[i] * rms_weight[i] * scale;
            q_sum += wq[j * embed_dim + i] * normed;
        }
        q_out[j] = q_sum;
    }
#endif
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
