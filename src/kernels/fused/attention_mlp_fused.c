/**
 * @file attention_mlp_fused.c
 * @brief Mega-Fused Attention + MLP Block
 *
 * CK-ENGINE KERNEL RULES:
 * =======================
 * 1. NO malloc/free - memory via bump allocator, pointers passed in
 * 2. NO OpenMP - parallelization at orchestrator/codegen layer
 * 3. NO memcpy for layout - use strided access, not copies
 * 4. API must define: inputs, outputs, workspace, and memory layouts
 * 5. Pure computation - deterministic, no side effects
 *
 * After changes: make test && make llamacpp-parity-full
 *
 * VIOLATION: Uses memcpy for layout conversion. TODO: Use strided access.
 *
 * Part of C-Kernel-Engine v6.6 Fusion Kernels
 *
 * FUSES THE ENTIRE BLOCK from Attention output to next layer input:
 *
 *   Attention(Q, K_cache, V_cache)
 *        │
 *        ▼
 *   Output Projection (attn @ Wo)
 *        │
 *        ▼
 *   + residual_1
 *        │
 *        ▼
 *   RMSNorm
 *        │
 *        ▼
 *   MLP: gate ──► SwiGLU ◄── up
 *        │
 *        ▼
 *      down
 *        │
 *        ▼
 *   + residual_2
 *        │
 *        ▼
 *   hidden_out (ready for next layer)
 *
 * NON-FUSED version writes these buffers to DRAM:
 *   - attn_output [embed_dim]
 *   - projected [embed_dim]
 *   - hidden_after_attn [embed_dim]
 *   - normed [embed_dim]
 *   - gate [intermediate_dim]
 *   - up [intermediate_dim]
 *   - swiglu [intermediate_dim]
 *   - mlp_out [embed_dim]
 *   = 8 DRAM round-trips!
 *
 * FUSED version: ALL intermediates stay in L1/L2, ZERO DRAM writes
 *
 * EXPECTED SPEEDUP: 2-3x for this block
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "ckernel_quant.h"

/* ============================================================================
 * HELPER: RMSNorm computation (inline, result stays in registers)
 * ============================================================================ */

static inline float compute_rms_scale_internal(const float *x, int n, float eps) {
    float sum_sq = 0.0f;

#ifdef __AVX2__
    __m256 vsum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        vsum = _mm256_fmadd_ps(vx, vx, vsum);
    }
    __m128 vlow = _mm256_castps256_ps128(vsum);
    __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    vlow = _mm_hadd_ps(vlow, vlow);
    vlow = _mm_hadd_ps(vlow, vlow);
    sum_sq = _mm_cvtss_f32(vlow);
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
 * HELPER: SiLU activation (x * sigmoid(x))
 * ============================================================================ */

static inline float silu_scalar(float x) {
    return x / (1.0f + expf(-x));
}

#ifdef __AVX2__
static inline __m256 silu_avx2(__m256 x) {
    float lanes[8];
    _mm256_storeu_ps(lanes, x);
    for (int i = 0; i < 8; i++) {
        lanes[i] = silu_scalar(lanes[i]);
    }
    return _mm256_loadu_ps(lanes);
}
#endif

/* ============================================================================
 * HELPER: Softmax with online computation (for attention)
 * ============================================================================ */

static void softmax_inplace(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        x[i] *= inv_sum;
    }
}

/* ============================================================================
 * MEGA-FUSED KERNEL: Attention + Output + RMSNorm + MLP
 *
 * This fuses the entire block from attention to MLP output.
 * All intermediates stay in L1/L2 cache.
 * ============================================================================ */

void attention_mlp_fused_fp32(
    /* Attention inputs */
    const float *q,               /* [num_heads * head_dim] query vector */
    const float *k_cache,         /* [seq_len, num_kv_heads * head_dim] K cache */
    const float *v_cache,         /* [seq_len, num_kv_heads * head_dim] V cache */
    int seq_len,                  /* Current sequence length */
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float attn_scale,             /* 1/sqrt(head_dim) */

    /* Output projection */
    const float *wo,              /* [embed_dim, num_heads * head_dim] */

    /* Residual input */
    const float *residual_1,      /* [embed_dim] input to attention block */

    /* RMSNorm */
    const float *rms_weight,      /* [embed_dim] */
    float eps,

    /* MLP weights (FP32 for this version) */
    const float *w_gate,          /* [intermediate_dim, embed_dim] */
    const float *w_up,            /* [intermediate_dim, embed_dim] */
    const float *w_down,          /* [embed_dim, intermediate_dim] */

    /* Residual 2 input (usually same as after attention residual) */
    /* If NULL, uses the hidden_after_attn */

    /* Dimensions */
    int embed_dim,
    int intermediate_dim,

    /* Output */
    float *hidden_out             /* [embed_dim] output for next layer */
) {
    const int heads_per_kv = num_heads / num_kv_heads;
    const int q_dim = num_heads * head_dim;
    const int kv_dim = num_kv_heads * head_dim;

    /* Stack buffers - all stay in L1/L2 */
    float attn_out[4096];         /* Attention output per head, then combined */
    float hidden_after_attn[4096];
    float normed[4096];
    float gate_out[16384];        /* Intermediate dim (e.g., 4864 for Qwen2) */
    float up_out[16384];

    if (embed_dim > 4096 || intermediate_dim > 16384) {
        return; /* TODO: heap allocation for large models */
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 1: Multi-Head Attention (Q @ K^T -> softmax -> @ V)
     * ═══════════════════════════════════════════════════════════════════════ */

    memset(attn_out, 0, q_dim * sizeof(float));

    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / heads_per_kv;  /* GQA: map query head to KV head */

        const float *q_head = q + h * head_dim;
        float *out_head = attn_out + h * head_dim;

        /* Compute attention scores: Q @ K^T */
        float scores[8192];  /* Max seq_len */
        if (seq_len > 8192) return;

        for (int t = 0; t < seq_len; t++) {
            const float *k_t = k_cache + t * kv_dim + kv_h * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q_head[d] * k_t[d];
            }
            scores[t] = score * attn_scale;
        }

        /* Softmax */
        softmax_inplace(scores, seq_len);

        /* Weighted sum of V: scores @ V */
        for (int t = 0; t < seq_len; t++) {
            const float *v_t = v_cache + t * kv_dim + kv_h * head_dim;
            float w = scores[t];
            for (int d = 0; d < head_dim; d++) {
                out_head[d] += w * v_t[d];
            }
        }
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 2: Output Projection (attn_out @ Wo) + Residual
     * ═══════════════════════════════════════════════════════════════════════ */

    for (int i = 0; i < embed_dim; i++) {
        float sum = 0.0f;
        const float *wo_row = wo + i * q_dim;
        for (int j = 0; j < q_dim; j++) {
            sum += wo_row[j] * attn_out[j];
        }
        hidden_after_attn[i] = sum + residual_1[i];  /* Residual add */
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 3: RMSNorm
     * ═══════════════════════════════════════════════════════════════════════ */

    float rms_scale = compute_rms_scale_internal(hidden_after_attn, embed_dim, eps);

#ifdef __AVX2__
    __m256 vscale = _mm256_set1_ps(rms_scale);
    int i = 0;
    for (; i + 7 < embed_dim; i += 8) {
        __m256 vh = _mm256_loadu_ps(hidden_after_attn + i);
        __m256 vw = _mm256_loadu_ps(rms_weight + i);
        __m256 vn = _mm256_mul_ps(_mm256_mul_ps(vh, vw), vscale);
        _mm256_storeu_ps(normed + i, vn);
    }
    for (; i < embed_dim; i++) {
        normed[i] = hidden_after_attn[i] * rms_weight[i] * rms_scale;
    }
#else
    for (int i = 0; i < embed_dim; i++) {
        normed[i] = hidden_after_attn[i] * rms_weight[i] * rms_scale;
    }
#endif

    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 4: MLP Gate + Up projections (can be parallelized)
     * ═══════════════════════════════════════════════════════════════════════ */

    /* Gate projection: gate_out = normed @ W_gate^T */
    for (int i = 0; i < intermediate_dim; i++) {
        float sum = 0.0f;
        const float *wg_row = w_gate + i * embed_dim;
        for (int j = 0; j < embed_dim; j++) {
            sum += wg_row[j] * normed[j];
        }
        gate_out[i] = sum;
    }

    /* Up projection: up_out = normed @ W_up^T */
    for (int i = 0; i < intermediate_dim; i++) {
        float sum = 0.0f;
        const float *wu_row = w_up + i * embed_dim;
        for (int j = 0; j < embed_dim; j++) {
            sum += wu_row[j] * normed[j];
        }
        up_out[i] = sum;
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 5: SwiGLU activation: silu(gate) * up
     * ═══════════════════════════════════════════════════════════════════════ */

#ifdef __AVX2__
    i = 0;
    for (; i + 7 < intermediate_dim; i += 8) {
        __m256 vg = _mm256_loadu_ps(gate_out + i);
        __m256 vu = _mm256_loadu_ps(up_out + i);
        __m256 vsilu = silu_avx2(vg);
        __m256 vswiglu = _mm256_mul_ps(vsilu, vu);
        _mm256_storeu_ps(gate_out + i, vswiglu);  /* Reuse gate_out buffer */
    }
    for (; i < intermediate_dim; i++) {
        gate_out[i] = silu_scalar(gate_out[i]) * up_out[i];
    }
#else
    for (int i = 0; i < intermediate_dim; i++) {
        gate_out[i] = silu_scalar(gate_out[i]) * up_out[i];
    }
#endif

    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 6: Down projection + Final Residual
     * ═══════════════════════════════════════════════════════════════════════ */

    for (int i = 0; i < embed_dim; i++) {
        float sum = 0.0f;
        const float *wd_row = w_down + i * intermediate_dim;
        for (int j = 0; j < intermediate_dim; j++) {
            sum += wd_row[j] * gate_out[j];  /* gate_out now holds SwiGLU output */
        }
        hidden_out[i] = sum + hidden_after_attn[i];  /* Final residual */
    }
}

/* ============================================================================
 * V2: MLP-ONLY FUSED KERNEL with SIMD GEMV
 *
 * Key optimizations over v1:
 * 1. AVX2 SIMD for ALL GEMVs (not just RMSNorm/SwiGLU)
 * 2. Gate + Up computed TOGETHER (one pass through normed)
 * 3. Horizontal sums done efficiently
 *
 * This isolates the MLP portion for benchmarking.
 * ============================================================================ */

#ifdef __AVX2__
/* Inline SIMD GEMV helper - processes one output row */
static inline float gemv_fp32_row_avx2(
    const float *row,      /* [K] weight row */
    const float *x,        /* [K] input vector */
    int K
) {
    __m256 acc = _mm256_setzero_ps();
    int k = 0;

    for (; k + 7 < K; k += 8) {
        __m256 vw = _mm256_loadu_ps(row + k);
        __m256 vx = _mm256_loadu_ps(x + k);
        acc = _mm256_fmadd_ps(vw, vx, acc);
    }

    /* Horizontal sum */
    __m128 vlow = _mm256_castps256_ps128(acc);
    __m128 vhigh = _mm256_extractf128_ps(acc, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    vlow = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, vlow);
    vlow = _mm_add_ss(vlow, shuf);
    float sum = _mm_cvtss_f32(vlow);

    /* Remainder */
    for (; k < K; k++) {
        sum += row[k] * x[k];
    }

    return sum;
}
#endif

void mlp_fused_fp32_v2(
    /* Input (after attention + residual) */
    const float *hidden_in,       /* [embed_dim] */

    /* RMSNorm */
    const float *rms_weight,      /* [embed_dim] */
    float eps,

    /* MLP weights (FP32) */
    const float *w_gate,          /* [intermediate_dim, embed_dim] */
    const float *w_up,            /* [intermediate_dim, embed_dim] */
    const float *w_down,          /* [embed_dim, intermediate_dim] */

    /* Dimensions */
    int embed_dim,
    int intermediate_dim,

    /* Output */
    float *hidden_out             /* [embed_dim] */
) {
    /* Stack buffers - sized for typical models */
    float normed[4096];
    float swiglu[16384];  /* intermediate_dim */

    if (embed_dim > 4096 || intermediate_dim > 16384) {
        return; /* TODO: handle larger models */
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 1: RMSNorm (SIMD)
     * ═══════════════════════════════════════════════════════════════════════ */

    float rms_scale = compute_rms_scale_internal(hidden_in, embed_dim, eps);

#ifdef __AVX2__
    __m256 vscale = _mm256_set1_ps(rms_scale);
    int i = 0;
    for (; i + 7 < embed_dim; i += 8) {
        __m256 vh = _mm256_loadu_ps(hidden_in + i);
        __m256 vw = _mm256_loadu_ps(rms_weight + i);
        __m256 vn = _mm256_mul_ps(_mm256_mul_ps(vh, vw), vscale);
        _mm256_storeu_ps(normed + i, vn);
    }
    for (; i < embed_dim; i++) {
        normed[i] = hidden_in[i] * rms_weight[i] * rms_scale;
    }
#else
    for (int i = 0; i < embed_dim; i++) {
        normed[i] = hidden_in[i] * rms_weight[i] * rms_scale;
    }
#endif

    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 2: Gate + Up projections with TRUE FUSION + SwiGLU
     *
     * Key insight: Compute gate[i] and up[i] together, then immediately
     * apply SwiGLU. This eliminates separate gate_out and up_out buffers.
     * ═══════════════════════════════════════════════════════════════════════ */

#ifdef __AVX2__
    for (int j = 0; j < intermediate_dim; j++) {
        /* Compute gate and up for output j using SIMD GEMV */
        const float *wg_row = w_gate + j * embed_dim;
        const float *wu_row = w_up + j * embed_dim;

        __m256 gate_acc = _mm256_setzero_ps();
        __m256 up_acc = _mm256_setzero_ps();

        int k = 0;
        for (; k + 7 < embed_dim; k += 8) {
            __m256 vn = _mm256_loadu_ps(normed + k);
            __m256 vwg = _mm256_loadu_ps(wg_row + k);
            __m256 vwu = _mm256_loadu_ps(wu_row + k);

            gate_acc = _mm256_fmadd_ps(vwg, vn, gate_acc);
            up_acc = _mm256_fmadd_ps(vwu, vn, up_acc);
        }

        /* Horizontal sums */
        __m128 glow = _mm256_castps256_ps128(gate_acc);
        __m128 ghigh = _mm256_extractf128_ps(gate_acc, 1);
        glow = _mm_add_ps(glow, ghigh);
        __m128 gshuf = _mm_movehdup_ps(glow);
        glow = _mm_add_ps(glow, gshuf);
        gshuf = _mm_movehl_ps(gshuf, glow);
        glow = _mm_add_ss(glow, gshuf);
        float gate_val = _mm_cvtss_f32(glow);

        __m128 ulow = _mm256_castps256_ps128(up_acc);
        __m128 uhigh = _mm256_extractf128_ps(up_acc, 1);
        ulow = _mm_add_ps(ulow, uhigh);
        __m128 ushuf = _mm_movehdup_ps(ulow);
        ulow = _mm_add_ps(ulow, ushuf);
        ushuf = _mm_movehl_ps(ushuf, ulow);
        ulow = _mm_add_ss(ulow, ushuf);
        float up_val = _mm_cvtss_f32(ulow);

        /* Remainder */
        for (; k < embed_dim; k++) {
            gate_val += wg_row[k] * normed[k];
            up_val += wu_row[k] * normed[k];
        }

        /* Fused SwiGLU: silu(gate) * up */
        swiglu[j] = silu_scalar(gate_val) * up_val;
    }
#else
    for (int j = 0; j < intermediate_dim; j++) {
        const float *wg_row = w_gate + j * embed_dim;
        const float *wu_row = w_up + j * embed_dim;
        float gate_val = 0.0f, up_val = 0.0f;

        for (int k = 0; k < embed_dim; k++) {
            gate_val += wg_row[k] * normed[k];
            up_val += wu_row[k] * normed[k];
        }

        swiglu[j] = silu_scalar(gate_val) * up_val;
    }
#endif

    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 3: Down projection + Residual (SIMD GEMV)
     * ═══════════════════════════════════════════════════════════════════════ */

#ifdef __AVX2__
    for (int j = 0; j < embed_dim; j++) {
        float sum = gemv_fp32_row_avx2(w_down + j * intermediate_dim, swiglu, intermediate_dim);
        hidden_out[j] = sum + hidden_in[j];  /* Residual */
    }
#else
    for (int j = 0; j < embed_dim; j++) {
        float sum = 0.0f;
        const float *wd_row = w_down + j * intermediate_dim;
        for (int k = 0; k < intermediate_dim; k++) {
            sum += wd_row[k] * swiglu[k];
        }
        hidden_out[j] = sum + hidden_in[j];
    }
#endif
}


/* ============================================================================
 * V3: MLP with SIMD GEMV but SEQUENTIAL weight access
 *
 * Key insight from v2 benchmark: fusing gate+up HURTS performance because
 * interleaved weight loading destroys cache prefetch patterns.
 *
 * v3 approach:
 * 1. Use SIMD GEMV for all projections
 * 2. Keep SEQUENTIAL weight access (gate first, then up)
 * 3. Still fuse SwiGLU immediately after projections
 *
 * This should be faster than v2 AND faster than scalar separate.
 * ============================================================================ */

void mlp_fused_fp32_v3(
    const float *hidden_in,
    const float *rms_weight,
    float eps,
    const float *w_gate,
    const float *w_up,
    const float *w_down,
    int embed_dim,
    int intermediate_dim,
    float *hidden_out
) {
    /* Stack buffers */
    float normed[4096];
    float gate_out[16384];
    float swiglu[16384];

    if (embed_dim > 4096 || intermediate_dim > 16384) {
        return;
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 1: RMSNorm (SIMD)
     * ═══════════════════════════════════════════════════════════════════════ */

    float rms_scale = compute_rms_scale_internal(hidden_in, embed_dim, eps);

#ifdef __AVX2__
    __m256 vscale = _mm256_set1_ps(rms_scale);
    int i = 0;
    for (; i + 7 < embed_dim; i += 8) {
        __m256 vh = _mm256_loadu_ps(hidden_in + i);
        __m256 vw = _mm256_loadu_ps(rms_weight + i);
        __m256 vn = _mm256_mul_ps(_mm256_mul_ps(vh, vw), vscale);
        _mm256_storeu_ps(normed + i, vn);
    }
    for (; i < embed_dim; i++) {
        normed[i] = hidden_in[i] * rms_weight[i] * rms_scale;
    }
#else
    for (int i = 0; i < embed_dim; i++) {
        normed[i] = hidden_in[i] * rms_weight[i] * rms_scale;
    }
#endif

    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 2: Gate projection (SIMD GEMV, sequential weight access)
     * ═══════════════════════════════════════════════════════════════════════ */

#ifdef __AVX2__
    for (int j = 0; j < intermediate_dim; j++) {
        gate_out[j] = gemv_fp32_row_avx2(w_gate + j * embed_dim, normed, embed_dim);
    }
#else
    for (int j = 0; j < intermediate_dim; j++) {
        float sum = 0.0f;
        const float *wg_row = w_gate + j * embed_dim;
        for (int k = 0; k < embed_dim; k++) {
            sum += wg_row[k] * normed[k];
        }
        gate_out[j] = sum;
    }
#endif

    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 3: Up projection + FUSED SwiGLU (SIMD GEMV, sequential access)
     *
     * Key: compute up[j], then immediately apply SwiGLU with gate[j].
     * This avoids storing the full up_out buffer.
     * ═══════════════════════════════════════════════════════════════════════ */

#ifdef __AVX2__
    for (int j = 0; j < intermediate_dim; j++) {
        float up_val = gemv_fp32_row_avx2(w_up + j * embed_dim, normed, embed_dim);
        /* Fused SwiGLU: silu(gate) * up */
        swiglu[j] = silu_scalar(gate_out[j]) * up_val;
    }
#else
    for (int j = 0; j < intermediate_dim; j++) {
        float up_val = 0.0f;
        const float *wu_row = w_up + j * embed_dim;
        for (int k = 0; k < embed_dim; k++) {
            up_val += wu_row[k] * normed[k];
        }
        swiglu[j] = silu_scalar(gate_out[j]) * up_val;
    }
#endif

    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 4: Down projection + Residual (SIMD GEMV)
     * ═══════════════════════════════════════════════════════════════════════ */

#ifdef __AVX2__
    for (int j = 0; j < embed_dim; j++) {
        float sum = gemv_fp32_row_avx2(w_down + j * intermediate_dim, swiglu, intermediate_dim);
        hidden_out[j] = sum + hidden_in[j];
    }
#else
    for (int j = 0; j < embed_dim; j++) {
        float sum = 0.0f;
        const float *wd_row = w_down + j * intermediate_dim;
        for (int k = 0; k < intermediate_dim; k++) {
            sum += wd_row[k] * swiglu[k];
        }
        hidden_out[j] = sum + hidden_in[j];
    }
#endif
}


/* ============================================================================
 * SEPARATE MLP (for benchmarking comparison)
 *
 * Same operations but as separate function calls.
 * ============================================================================ */

void mlp_separate_fp32(
    const float *hidden_in,
    const float *rms_weight,
    float eps,
    const float *w_gate,
    const float *w_up,
    const float *w_down,
    float *normed_buf,        /* [embed_dim] caller-provided */
    float *gate_buf,          /* [intermediate_dim] caller-provided */
    float *up_buf,            /* [intermediate_dim] caller-provided */
    int embed_dim,
    int intermediate_dim,
    float *hidden_out
) {
    /* Step 1: RMSNorm */
    float rms_scale = compute_rms_scale_internal(hidden_in, embed_dim, eps);
    for (int i = 0; i < embed_dim; i++) {
        normed_buf[i] = hidden_in[i] * rms_weight[i] * rms_scale;
    }

    /* Step 2: Gate projection */
    for (int j = 0; j < intermediate_dim; j++) {
        float sum = 0.0f;
        const float *wg_row = w_gate + j * embed_dim;
        for (int k = 0; k < embed_dim; k++) {
            sum += wg_row[k] * normed_buf[k];
        }
        gate_buf[j] = sum;
    }

    /* Step 3: Up projection */
    for (int j = 0; j < intermediate_dim; j++) {
        float sum = 0.0f;
        const float *wu_row = w_up + j * embed_dim;
        for (int k = 0; k < embed_dim; k++) {
            sum += wu_row[k] * normed_buf[k];
        }
        up_buf[j] = sum;
    }

    /* Step 4: SwiGLU */
    for (int j = 0; j < intermediate_dim; j++) {
        gate_buf[j] = silu_scalar(gate_buf[j]) * up_buf[j];
    }

    /* Step 5: Down projection + Residual */
    for (int j = 0; j < embed_dim; j++) {
        float sum = 0.0f;
        const float *wd_row = w_down + j * intermediate_dim;
        for (int k = 0; k < intermediate_dim; k++) {
            sum += wd_row[k] * gate_buf[k];
        }
        hidden_out[j] = sum + hidden_in[j];
    }
}


/* ============================================================================
 * Q4_K VERSION: Attention + Output + RMSNorm + MLP with quantized weights
 *
 * All MLP weights are Q4_K quantized.
 * ============================================================================ */

void attention_mlp_fused_q4k(
    /* Attention inputs */
    const float *q,               /* [num_heads * head_dim] */
    const float *k_cache,         /* [seq_len, num_kv_heads * head_dim] */
    const float *v_cache,         /* [seq_len, num_kv_heads * head_dim] */
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float attn_scale,

    /* Output projection (Q4_K) */
    const void *wo,

    /* Residual */
    const float *residual_1,

    /* RMSNorm */
    const float *rms_weight,
    float eps,

    /* MLP weights (Q4_K) */
    const void *w_gate,
    const void *w_up,
    const void *w_down,

    /* Dimensions */
    int embed_dim,
    int intermediate_dim,

    /* Output */
    float *hidden_out
) {
    const int heads_per_kv = num_heads / num_kv_heads;
    const int q_dim = num_heads * head_dim;
    const int kv_dim = num_kv_heads * head_dim;

    /* Stack buffers */
    float attn_out[4096];
    float hidden_after_attn[4096];
    float normed[4096];
    float mlp_out[4096];

    if (embed_dim > 4096) return;

    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 1: Multi-Head Attention (same as FP32 version)
     * ═══════════════════════════════════════════════════════════════════════ */

    memset(attn_out, 0, q_dim * sizeof(float));

    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / heads_per_kv;
        const float *q_head = q + h * head_dim;
        float *out_head = attn_out + h * head_dim;

        float scores[8192];
        if (seq_len > 8192) return;

        for (int t = 0; t < seq_len; t++) {
            const float *k_t = k_cache + t * kv_dim + kv_h * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q_head[d] * k_t[d];
            }
            scores[t] = score * attn_scale;
        }

        softmax_inplace(scores, seq_len);

        for (int t = 0; t < seq_len; t++) {
            const float *v_t = v_cache + t * kv_dim + kv_h * head_dim;
            float w = scores[t];
            for (int d = 0; d < head_dim; d++) {
                out_head[d] += w * v_t[d];
            }
        }
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 2: Output Projection (Q4_K) + Residual
     * ═══════════════════════════════════════════════════════════════════════ */

    extern void gemv_q4_k(float *y, const void *W, const float *x, int M, int K);

    gemv_q4_k(hidden_after_attn, wo, attn_out, embed_dim, q_dim);

    /* Add residual */
    for (int i = 0; i < embed_dim; i++) {
        hidden_after_attn[i] += residual_1[i];
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 3: RMSNorm (same as before)
     * ═══════════════════════════════════════════════════════════════════════ */

    float rms_scale = compute_rms_scale_internal(hidden_after_attn, embed_dim, eps);

    for (int i = 0; i < embed_dim; i++) {
        normed[i] = hidden_after_attn[i] * rms_weight[i] * rms_scale;
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 4-6: MLP with Q4_K weights (inline implementation)
     *
     * gate_out = normed @ W_gate
     * up_out = normed @ W_up
     * swiglu = silu(gate_out) * up_out
     * mlp_out = swiglu @ W_down
     * ═══════════════════════════════════════════════════════════════════════ */

    float gate_out[16384];
    float up_out[16384];

    if (intermediate_dim > 16384) return;

    /* Gate projection */
    gemv_q4_k(gate_out, w_gate, normed, intermediate_dim, embed_dim);

    /* Up projection */
    gemv_q4_k(up_out, w_up, normed, intermediate_dim, embed_dim);

    /* SwiGLU: silu(gate) * up */
#ifdef __AVX2__
    int i = 0;
    for (; i + 7 < intermediate_dim; i += 8) {
        __m256 vg = _mm256_loadu_ps(gate_out + i);
        __m256 vu = _mm256_loadu_ps(up_out + i);
        __m256 vsilu = silu_avx2(vg);
        __m256 vswiglu = _mm256_mul_ps(vsilu, vu);
        _mm256_storeu_ps(gate_out + i, vswiglu);
    }
    for (; i < intermediate_dim; i++) {
        gate_out[i] = silu_scalar(gate_out[i]) * up_out[i];
    }
#else
    for (int i = 0; i < intermediate_dim; i++) {
        gate_out[i] = silu_scalar(gate_out[i]) * up_out[i];
    }
#endif

    /* Down projection */
    gemv_q4_k(mlp_out, w_down, gate_out, embed_dim, intermediate_dim);

    /* Final residual add */
    for (int i = 0; i < embed_dim; i++) {
        hidden_out[i] = mlp_out[i] + hidden_after_attn[i];
    }
}

/* ============================================================================
 * COMPLETE LAYER FUSION: Attention → MLP → Next Layer's QKV
 *
 * This is the TRUE mega-fusion: from one layer's attention output all the
 * way to the next layer's Q (ready for attention) + K,V written to cache.
 *
 * The hidden state NEVER touches DRAM between layers!
 * ============================================================================ */

void layer_fused_attn_mlp_qkv_q4k(
    /* === CURRENT LAYER ATTENTION INPUTS === */
    const float *q,               /* [num_heads * head_dim] query for this layer */
    const float *k_cache,         /* [seq_len, num_kv_heads * head_dim] */
    const float *v_cache,         /* [seq_len, num_kv_heads * head_dim] */
    int seq_len,
    float attn_scale,

    /* === CURRENT LAYER WEIGHTS (Q4_K) === */
    const void *wo,               /* Output projection */
    const float *rms_weight_mlp,  /* RMSNorm before MLP */
    const void *w_gate,           /* MLP gate */
    const void *w_up,             /* MLP up */
    const void *w_down,           /* MLP down */

    /* === NEXT LAYER WEIGHTS (Q4_K) === */
    const float *rms_weight_attn, /* RMSNorm before next attention */
    const void *wq_next,          /* Next layer Q projection */
    const void *wk_next,          /* Next layer K projection */
    const void *wv_next,          /* Next layer V projection */

    /* === RESIDUAL INPUT === */
    const float *residual_in,     /* [embed_dim] input to this layer */

    /* === DIMENSIONS === */
    int embed_dim,
    int intermediate_dim,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float eps,

    /* === OUTPUTS === */
    float *q_next,                /* [num_heads * head_dim] Q for next layer */
    float *k_next,                /* [num_kv_heads * head_dim] K to write to cache */
    float *v_next,                /* [num_kv_heads * head_dim] V to write to cache */
    float *hidden_out             /* [embed_dim] hidden state (for final layer) */
) {
    extern void gemv_q4_k(float *y, const void *W, const float *x, int M, int K);

    const int heads_per_kv = num_heads / num_kv_heads;
    const int q_dim = num_heads * head_dim;
    const int kv_dim = num_kv_heads * head_dim;

    /* All intermediate buffers on stack - stay in L1/L2
     * hidden_out is the final output buffer - we write to it directly! */
    float attn_out[4096];
    float hidden_after_attn[4096];
    float normed_mlp[4096];
    float gate_out[16384];
    float up_out[16384];
    /* NOTE: No hidden_after_mlp buffer - we output directly to hidden_out */
    float normed_attn[4096];

    if (embed_dim > 4096 || intermediate_dim > 16384) return;

    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 1: Multi-Head Attention
     * ═══════════════════════════════════════════════════════════════════════ */

    memset(attn_out, 0, q_dim * sizeof(float));

    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / heads_per_kv;
        const float *q_head = q + h * head_dim;
        float *out_head = attn_out + h * head_dim;

        float scores[8192];
        if (seq_len > 8192) return;

        for (int t = 0; t < seq_len; t++) {
            const float *k_t = k_cache + t * kv_dim + kv_h * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q_head[d] * k_t[d];
            }
            scores[t] = score * attn_scale;
        }

        /* Softmax */
        float max_score = scores[0];
        for (int t = 1; t < seq_len; t++) {
            if (scores[t] > max_score) max_score = scores[t];
        }
        float sum_exp = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum_exp += scores[t];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int t = 0; t < seq_len; t++) {
            scores[t] *= inv_sum;
        }

        /* Weighted sum of V */
        for (int t = 0; t < seq_len; t++) {
            const float *v_t = v_cache + t * kv_dim + kv_h * head_dim;
            float w = scores[t];
            for (int d = 0; d < head_dim; d++) {
                out_head[d] += w * v_t[d];
            }
        }
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 2: Output Projection (Q4_K) + Residual
     * ═══════════════════════════════════════════════════════════════════════ */

    gemv_q4_k(hidden_after_attn, wo, attn_out, embed_dim, q_dim);

    for (int i = 0; i < embed_dim; i++) {
        hidden_after_attn[i] += residual_in[i];
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 3: RMSNorm (for MLP)
     * ═══════════════════════════════════════════════════════════════════════ */

    float sum_sq = 0.0f;
    for (int i = 0; i < embed_dim; i++) {
        sum_sq += hidden_after_attn[i] * hidden_after_attn[i];
    }
    float rms_scale = 1.0f / sqrtf(sum_sq / embed_dim + eps);

    for (int i = 0; i < embed_dim; i++) {
        normed_mlp[i] = hidden_after_attn[i] * rms_weight_mlp[i] * rms_scale;
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 4-6: MLP (gate + up + SwiGLU + down)
     * ═══════════════════════════════════════════════════════════════════════ */

    gemv_q4_k(gate_out, w_gate, normed_mlp, intermediate_dim, embed_dim);
    gemv_q4_k(up_out, w_up, normed_mlp, intermediate_dim, embed_dim);

    /* SwiGLU: silu(gate) * up */
    for (int i = 0; i < intermediate_dim; i++) {
        float g = gate_out[i];
        float silu_g = g / (1.0f + expf(-g));
        gate_out[i] = silu_g * up_out[i];
    }

    /* Down projection - output DIRECTLY to hidden_out (no intermediate buffer!) */
    gemv_q4_k(hidden_out, w_down, gate_out, embed_dim, intermediate_dim);

    /* MLP residual - hidden_out now contains the final hidden state */
    for (int i = 0; i < embed_dim; i++) {
        hidden_out[i] += hidden_after_attn[i];
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 7: RMSNorm (for NEXT layer's attention)
     * Read from hidden_out (already contains final hidden state)
     * ═══════════════════════════════════════════════════════════════════════ */

    sum_sq = 0.0f;
    for (int i = 0; i < embed_dim; i++) {
        sum_sq += hidden_out[i] * hidden_out[i];
    }
    rms_scale = 1.0f / sqrtf(sum_sq / embed_dim + eps);

    for (int i = 0; i < embed_dim; i++) {
        normed_attn[i] = hidden_out[i] * rms_weight_attn[i] * rms_scale;
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 8: NEXT LAYER's Q, K, V Projections
     *
     * Q goes to caller (for attention computation)
     * K, V go to KV cache (DRAM write - this is intentional!)
     * ═══════════════════════════════════════════════════════════════════════ */

    gemv_q4_k(q_next, wq_next, normed_attn, q_dim, embed_dim);
    gemv_q4_k(k_next, wk_next, normed_attn, kv_dim, embed_dim);
    gemv_q4_k(v_next, wv_next, normed_attn, kv_dim, embed_dim);

    /* hidden_out already contains the final hidden state - no memcpy needed! */
}

/* ============================================================================
 * NON-FUSED REFERENCE: For benchmarking comparison
 * ============================================================================ */

void attention_mlp_separate_fp32(
    const float *q, const float *k_cache, const float *v_cache,
    int seq_len, int num_heads, int num_kv_heads, int head_dim,
    float attn_scale,
    const float *wo, const float *residual_1,
    const float *rms_weight, float eps,
    const float *w_gate, const float *w_up, const float *w_down,
    int embed_dim, int intermediate_dim,
    /* Intermediate buffers - DRAM traffic! */
    float *attn_out_buf,
    float *hidden_after_attn_buf,
    float *normed_buf,
    float *gate_buf,
    float *up_buf,
    float *mlp_out_buf,
    /* Output */
    float *hidden_out
) {
    /* This version writes all intermediates to the provided buffers,
     * simulating non-fused execution with DRAM traffic */

    const int heads_per_kv = num_heads / num_kv_heads;
    const int q_dim = num_heads * head_dim;
    const int kv_dim = num_kv_heads * head_dim;

    /* Step 1: Attention */
    memset(attn_out_buf, 0, q_dim * sizeof(float));

    /* Stack-allocated scores buffer (no malloc!) */
    float scores[8192];  /* Max seq_len supported */
    if (seq_len > 8192) return;

    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / heads_per_kv;
        const float *q_head = q + h * head_dim;
        float *out_head = attn_out_buf + h * head_dim;

        for (int t = 0; t < seq_len; t++) {
            const float *k_t = k_cache + t * kv_dim + kv_h * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q_head[d] * k_t[d];
            }
            scores[t] = score * attn_scale;
        }

        softmax_inplace(scores, seq_len);

        for (int t = 0; t < seq_len; t++) {
            const float *v_t = v_cache + t * kv_dim + kv_h * head_dim;
            float w = scores[t];
            for (int d = 0; d < head_dim; d++) {
                out_head[d] += w * v_t[d];
            }
        }
    }

    /* Step 2: Output projection + residual -> DRAM write */
    for (int i = 0; i < embed_dim; i++) {
        float sum = 0.0f;
        const float *wo_row = wo + i * q_dim;
        for (int j = 0; j < q_dim; j++) {
            sum += wo_row[j] * attn_out_buf[j];
        }
        hidden_after_attn_buf[i] = sum + residual_1[i];
    }

    /* Step 3: RMSNorm -> DRAM write */
    float rms_scale = compute_rms_scale_internal(hidden_after_attn_buf, embed_dim, eps);
    for (int i = 0; i < embed_dim; i++) {
        normed_buf[i] = hidden_after_attn_buf[i] * rms_weight[i] * rms_scale;
    }

    /* Step 4: Gate projection -> DRAM write */
    for (int i = 0; i < intermediate_dim; i++) {
        float sum = 0.0f;
        const float *wg_row = w_gate + i * embed_dim;
        for (int j = 0; j < embed_dim; j++) {
            sum += wg_row[j] * normed_buf[j];
        }
        gate_buf[i] = sum;
    }

    /* Step 5: Up projection -> DRAM write */
    for (int i = 0; i < intermediate_dim; i++) {
        float sum = 0.0f;
        const float *wu_row = w_up + i * embed_dim;
        for (int j = 0; j < embed_dim; j++) {
            sum += wu_row[j] * normed_buf[j];
        }
        up_buf[i] = sum;
    }

    /* Step 6: SwiGLU (in-place in gate_buf) */
    for (int i = 0; i < intermediate_dim; i++) {
        gate_buf[i] = silu_scalar(gate_buf[i]) * up_buf[i];
    }

    /* Step 7: Down projection -> DRAM write */
    for (int i = 0; i < embed_dim; i++) {
        float sum = 0.0f;
        const float *wd_row = w_down + i * intermediate_dim;
        for (int j = 0; j < intermediate_dim; j++) {
            sum += wd_row[j] * gate_buf[j];
        }
        mlp_out_buf[i] = sum;
    }

    /* Step 8: Final residual */
    for (int i = 0; i < embed_dim; i++) {
        hidden_out[i] = mlp_out_buf[i] + hidden_after_attn_buf[i];
    }
}
