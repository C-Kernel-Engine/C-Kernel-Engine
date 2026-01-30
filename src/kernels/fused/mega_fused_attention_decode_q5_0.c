/**
 * @file mega_fused_attention_decode_q5_0.c
 * @brief Mega-fused attention decode with Q5_0 weights
 *
 * STATUS: Serial kernel complete and correct. Parallel variant is a prototype
 *         that requires threadpool barrier support (not yet available).
 *         The non-fused decode path (ck_parallel_decode.h) already parallelizes
 *         each GEMV via row-splitting, so this fused kernel is not on the
 *         critical path. It can be enabled once threadpool parallelization
 *         is resolved (see PARALLELIZATION NOTES below).
 *
 * FUSION: Combines 9 operations to minimize memory traffic.
 * All intermediate data stays in scratch buffer (L1/L2 cache).
 *
 * Operations fused:
 *   1. RMSNorm
 *   2. Q projection (Q5_0) with bias
 *   3. K projection (Q5_0) with bias
 *   4. V projection (Q8_0) with bias
 *   5. RoPE application
 *   6. KV cache store
 *   7. Flash attention decode (GQA-aware)
 *   8. O projection (Q5_0) with bias
 *   9. Residual add
 *
 * PARALLELIZATION NOTES:
 *   The parallel_simd variant below documents the intended threading model
 *   but cannot run with the current threadpool (single dispatch, no mid-dispatch
 *   barrier). Three approaches were evaluated:
 *
 *   (A) Multi-dispatch (RECOMMENDED):
 *       Break into 3 ck_threadpool_dispatch() calls per layer:
 *         Dispatch 1: Row-split Q proj across threads.
 *                     Thread 0 also does RMSNorm, K/V proj, RoPE, KV store
 *                     (small ops that fit within Q proj wall time).
 *         Dispatch 2: Split attention across heads (h_start..h_end per thread).
 *         Dispatch 3: Row-split O proj across threads.
 *                     Thread 0 does residual add after its rows.
 *       Cost: ~1us total for 2 extra barrier round-trips (negligible vs ~100us GEMV).
 *       Intermediates stay in shared scratch — cache benefit preserved.
 *
 *   (B) Redundant compute (single dispatch, no barrier):
 *       All threads redundantly compute RMSNorm + K/V proj + RoPE (~4us wasted
 *       per thread). Avoids barrier but wastes cycles on small ops.
 *       Only viable if Q/O proj dominate (true for short contexts).
 *
 *   (C) Skip fusion, use existing parallel GEMV:
 *       The non-fused decode path already parallelizes each GEMV call via
 *       ck_parallel_decode.h. For decode (M=1), intermediates are small
 *       (~3.5KB), so DRAM bandwidth savings from fusion are minimal.
 *       This is the current production path.
 *
 * TESTING:
 *   make test-mega-fused-parity    # Numerical parity
 *   make test-mega-fused-speed     # Performance benchmark
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "ckernel_quant.h"

/* Declare functions from other kernel files */
extern void rmsnorm_forward(const float *input, const float *gamma, float *output,
                            float *rstd, int T, int D, int AD, float eps);
extern void attention_forward_decode_head_major_gqa_flash(
    const float *q_token, const float *k_cache, const float *v_cache,
    float *out_token, int num_heads, int num_kv_heads, int kv_tokens,
    int cache_capacity, int head_dim, int aligned_head_dim);
extern void quantize_row_q8_0(const float *x, void *vy, int k);
extern void vec_dot_q5_0_q8_0(int n, float *s, const void *vx, const void *vy);
extern void vec_dot_q8_0_q8_0(int n, float *s, const void *vx, const void *vy);

/* ============================================================================
 * Q5_0 GEMV with inline processing
 *
 * For true fusion: quantize input to Q8_0, then use efficient vec_dot
 * Uses scratch buffer instead of malloc (kernel rule)
 * ============================================================================ */

static inline void gemv_q5_0_from_fp32(
    float *out,           /* Output [M] */
    const void *W_q5_0,   /* Q5_0 weights [M, K] */
    const float *x_fp32,  /* FP32 input [K] */
    const float *bias,    /* Bias [M] or NULL */
    int M,
    int K,
    block_q8_0 *x_q8_scratch)  /* Scratch buffer for quantized input */
{
    const block_q5_0 *w_blocks = (const block_q5_0 *)W_q5_0;
    const int blocks_per_row = K / QK5_0;

    /* Quantize input to Q8_0 (reuse existing kernel, scratch buffer) */
    quantize_row_q8_0(x_fp32, x_q8_scratch, K);

    /* Compute dot products using optimized kernel */
    for (int row = 0; row < M; row++) {
        float dot;
        vec_dot_q5_0_q8_0(K, &dot, &w_blocks[row * blocks_per_row], x_q8_scratch);
        out[row] = dot + (bias ? bias[row] : 0.0f);
    }
}

/* Q8_0 GEMV - input is already FP32, weights are Q8_0 */
static inline void gemv_q8_0_from_fp32(
    float *out,
    const void *W_q8_0,
    const float *x_fp32,
    const float *bias,
    int M,
    int K,
    block_q8_0 *x_q8_scratch)
{
    const block_q8_0 *w_blocks = (const block_q8_0 *)W_q8_0;
    const int blocks_per_row = K / QK8_0;

    /* Quantize input to Q8_0 (reuse existing kernel, scratch buffer) */
    quantize_row_q8_0(x_fp32, x_q8_scratch, K);

    /* Compute dot products */
    for (int row = 0; row < M; row++) {
        float dot;
        vec_dot_q8_0_q8_0(K, &dot, &w_blocks[row * blocks_per_row], x_q8_scratch);
        out[row] = dot + (bias ? bias[row] : 0.0f);
    }
}

/* ============================================================================
 * RoPE application (inline)
 * ============================================================================ */

static inline void apply_rope_inline(
    float *q,
    float *k,
    const float *rope_cos,
    const float *rope_sin,
    int pos,
    int H,
    int KV,
    int AD)
{
    const int D = AD / 2;
    const float *cos_row = &rope_cos[pos * D];
    const float *sin_row = &rope_sin[pos * D];

    /* Q heads */
    for (int h = 0; h < H; h++) {
        float *q_head = &q[h * AD];
        for (int d = 0; d < D; d++) {
            float q0 = q_head[d];
            float q1 = q_head[d + D];
            q_head[d] = q0 * cos_row[d] - q1 * sin_row[d];
            q_head[d + D] = q0 * sin_row[d] + q1 * cos_row[d];
        }
    }

    /* K heads */
    for (int kv = 0; kv < KV; kv++) {
        float *k_head = &k[kv * AD];
        for (int d = 0; d < D; d++) {
            float k0 = k_head[d];
            float k1 = k_head[d + D];
            k_head[d] = k0 * cos_row[d] - k1 * sin_row[d];
            k_head[d + D] = k0 * sin_row[d] + k1 * cos_row[d];
        }
    }
}

/* ============================================================================
 * Calculate scratch size needed (with all required parameters)
 * ============================================================================ */

int mega_fused_attention_decode_scratch_size(int AE, int H, int KV, int AD) {
    /* Need: 1x AE for RMSNorm output
            1x AE for RMSNorm rstd (avoid VLA)
            1x H*AD for Q
            1x KV*AD for K
            1x KV*AD for V
            1x H*AD for attention output
            1x max(AE, H*AD)/QK8_0 * sizeof(block_q8_0) for GEMV scratch
    */
    int max_input_dim = (AE > H * AD) ? AE : H * AD;
    int q8_blocks = (max_input_dim + QK8_0 - 1) / QK8_0;
    return (int)(sizeof(float) * (AE + AE + H * AD + 2 * KV * AD + H * AD)
                 + q8_blocks * sizeof(block_q8_0));
}

/* ============================================================================
 * MAIN KERNEL
 *
 * @param output Output [AE] (final result, after residual add)
 * @param input Input activation [AE]
 * @param residual Residual input for add [AE]
 * @param wq_q5_0 Q projection weights [H*AD, AE] Q5_0
 * @param wk_q5_0 K projection weights [KV*AD, AE] Q5_0
 * @param wv_q8_0 V projection weights [KV*AD, AE] Q8_0
 * @param wo_q5_0 O projection weights [AE, H*AD] Q5_0 (row e has H*AD elements)
 * @param ln_gamma RMSNorm gamma [AE]
 * @param bq Q bias [H*AD] or NULL
 * @param bk K bias [KV*AD] or NULL
 * @param bv V bias [KV*AD] or NULL
 * @param bo O bias [AE] or NULL
 * @param kv_cache_k K cache [KV, max_T, AD]
 * @param kv_cache_v V cache [KV, max_T, AD]
 * @param rope_cos RoPE cos [max_T, D]
 * @param rope_sin RoPE sin [max_T, D]
 * @param pos Current position
 * @param embed_dim Original embed dim E
 * @param aligned_embed_dim Aligned embed dim AE (multiple of 64)
 * @param num_heads H
 * @param num_kv_heads KV
 * @param head_dim AD
 * @param aligned_head_dim AAD (multiple of 64)
 * @param cache_capacity max_T
 * @param eps RMSNorm epsilon
 * @param scratch Scratch buffer (must be >= scratch_size bytes)
 * ============================================================================ */

void mega_fused_attention_decode_q5_0(
    float *output,
    const float *input,
    const float *residual,
    const void *wq_q5_0,
    const void *wk_q5_0,
    const void *wv_q8_0,
    const void *wo_q5_0,
    const float *ln_gamma,
    const float *bq,
    const float *bk,
    const float *bv,
    const float *bo,
    float *kv_cache_k,
    float *kv_cache_v,
    const float *rope_cos,
    const float *rope_sin,
    int pos,
    int embed_dim,
    int aligned_embed_dim,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int aligned_head_dim,
    int cache_capacity,
    float eps,
    void *scratch)
{
    const int H = num_heads;
    const int KV = num_kv_heads;
    const int AD = head_dim;
    const int AE = aligned_embed_dim;
    (void)embed_dim;  /* Unused but kept for API consistency */

    /* Parse scratch buffer - all allocations from scratch, no VLAs */
    float *scratch_ptr = (float *)scratch;

    float *rmsnorm_out = scratch_ptr;
    scratch_ptr += AE;

    float *rstd_scratch = scratch_ptr;  /* For rmsnorm rstd output - avoids VLA */
    scratch_ptr += AE;

    float *q = scratch_ptr;
    scratch_ptr += H * AD;

    float *k = scratch_ptr;
    scratch_ptr += KV * AD;

    float *v = scratch_ptr;
    scratch_ptr += KV * AD;

    float *attn_out = scratch_ptr;
    scratch_ptr += H * AD;

    block_q8_0 *x_q8_scratch = (block_q8_0 *)scratch_ptr;

    const int q_size = H * AD;
    const int k_size = KV * AD;
    const int v_size = KV * AD;

    /* ========================================================================
     * STEP 1: RMSNorm
     * Correct signature: rmsnorm_forward(in, gamma, out, rstd, T, D, AD, eps)
     * T=1 (single token), D=AE (full embed dim for norm)
     * ======================================================================== */
    rmsnorm_forward(input, ln_gamma, rmsnorm_out, rstd_scratch, 1, AE, AD, eps);

    /* ========================================================================
     * STEP 2-4: Q, K, V projections (fused with quantization)
     * Use scratch buffer for quantized input
     * ======================================================================== */
    gemv_q5_0_from_fp32(q, wq_q5_0, rmsnorm_out, bq, q_size, AE, x_q8_scratch);
    gemv_q5_0_from_fp32(k, wk_q5_0, rmsnorm_out, bk, k_size, AE, x_q8_scratch);
    gemv_q8_0_from_fp32(v, wv_q8_0, rmsnorm_out, bv, v_size, AE, x_q8_scratch);

    /* ========================================================================
     * STEP 5: Apply RoPE
     * ======================================================================== */
    apply_rope_inline(q, k, rope_cos, rope_sin, pos, H, KV, AD);

    /* ========================================================================
     * STEP 6: Store K and V to cache
     * Cache layout: [KV, cache_capacity, AD]
     * ======================================================================== */
    const size_t kv_stride = (size_t)cache_capacity * AD;
    for (int kv = 0; kv < KV; kv++) {
        float *k_cache = &kv_cache_k[kv * kv_stride];
        float *v_cache = &kv_cache_v[kv * kv_stride];
        const float *k_src = &k[kv * AD];
        const float *v_src = &v[kv * AD];
        const int offset = pos * AD;
        for (int d = 0; d < AD; d++) {
            k_cache[offset + d] = k_src[d];
            v_cache[offset + d] = v_src[d];
        }
    }

    /* ========================================================================
     * STEP 7: Flash attention decode (GQA-aware variant)
     * attention_forward_decode_head_major_gqa_flash handles H != KV correctly
     * It maps each of H heads to one of KV KV heads via: kv_head = h * KV / H
     * ======================================================================== */
    attention_forward_decode_head_major_gqa_flash(
        q, kv_cache_k, kv_cache_v,
        attn_out, H, KV, pos + 1, cache_capacity, AD, aligned_head_dim);

    /* ========================================================================
     * STEP 8: O projection (Q5_0 weights) with bias and residual add
     *
     * attn_out layout: [H * AD] flattened
     * wo_q5_0 layout: [AE, H*AD] - row e has H*AD input features
     *
     * O projection: output[e] = dot(wo[e], attn_out) + bias[e] + residual[e]
     *
     * Using vec_dot_q5_0_q8_0 for efficient quantized dot product.
     * ======================================================================== */

    /* Quantize attention output to Q8_0 for GEMV */
    quantize_row_q8_0(attn_out, x_q8_scratch, H * AD);

    const block_q5_0 *wo = (const block_q5_0 *)wo_q5_0;
    const int blocks_per_row = (H * AD) / QK5_0;

    for (int e = 0; e < AE; e++) {
        float dot;
        vec_dot_q5_0_q8_0(H * AD, &dot, &wo[e * blocks_per_row], x_q8_scratch);
        output[e] = dot + (bo ? bo[e] : 0.0f) + residual[e];
    }
}

/* ============================================================================
 * PARALLEL SIMD VARIANT (threadpool-aware)
 *
 * Parallelizes across attention heads using (ith, nth) pattern.
 * Each thread processes a subset of heads.
 *
 * IMPORTANT: Caller must ensure barrier sync between phases:
 *   Phase 1 (ith==0 only): RMSNorm, Q/K/V projection, RoPE, KV cache store
 *   -- BARRIER --
 *   Phase 2 (all threads): Attention for assigned heads
 *   -- BARRIER --
 *   Phase 3 (ith==0 only): O projection and residual add
 * ======================================================================== */

void mega_fused_attention_decode_q5_0_parallel_simd(
    float *output,
    const float *input,
    const float *residual,
    const void *wq_q5_0,
    const void *wk_q5_0,
    const void *wv_q8_0,
    const void *wo_q5_0,
    const float *ln_gamma,
    const float *bq,
    const float *bk,
    const float *bv,
    const float *bo,
    float *kv_cache_k,
    float *kv_cache_v,
    const float *rope_cos,
    const float *rope_sin,
    int pos,
    int embed_dim,
    int aligned_embed_dim,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int aligned_head_dim,
    int cache_capacity,
    float eps,
    void *scratch,
    int ith,
    int nth)
{
    const int H = num_heads;
    const int KV = num_kv_heads;
    const int AD = head_dim;
    const int AE = aligned_embed_dim;
    (void)embed_dim;

    /* Each thread handles a subset of heads */
    const int heads_per_thread = (H + nth - 1) / nth;
    const int h_start = ith * heads_per_thread;
    const int h_end = (h_start + heads_per_thread < H) ? h_start + heads_per_thread : H;
    const int my_heads = h_end - h_start;

    if (h_start >= H) return;

    /* Parse scratch buffer (shared across threads) */
    float *scratch_ptr = (float *)scratch;

    float *rmsnorm_out = scratch_ptr;
    scratch_ptr += AE;

    float *rstd_scratch = scratch_ptr;
    scratch_ptr += AE;

    float *q = scratch_ptr;
    scratch_ptr += H * AD;

    float *k = scratch_ptr;
    scratch_ptr += KV * AD;

    float *v = scratch_ptr;
    scratch_ptr += KV * AD;

    float *attn_out = scratch_ptr;
    scratch_ptr += H * AD;

    block_q8_0 *x_q8_scratch = (block_q8_0 *)scratch_ptr;

    /* ========================================================================
     * PHASE 1: Only thread 0 does RMSNorm and K/V projections
     * These are shared across all heads.
     * CALLER MUST BARRIER AFTER THIS PHASE.
     * ======================================================================== */
    if (ith == 0) {
        rmsnorm_forward(input, ln_gamma, rmsnorm_out, rstd_scratch, 1, AE, AD, eps);

        gemv_q5_0_from_fp32(q, wq_q5_0, rmsnorm_out, bq, H * AD, AE, x_q8_scratch);
        gemv_q5_0_from_fp32(k, wk_q5_0, rmsnorm_out, bk, KV * AD, AE, x_q8_scratch);
        gemv_q8_0_from_fp32(v, wv_q8_0, rmsnorm_out, bv, KV * AD, AE, x_q8_scratch);

        apply_rope_inline(q, k, rope_cos, rope_sin, pos, H, KV, AD);

        /* Store K/V to cache */
        const size_t kv_stride = (size_t)cache_capacity * AD;
        for (int kv_idx = 0; kv_idx < KV; kv_idx++) {
            float *k_cache = &kv_cache_k[kv_idx * kv_stride];
            float *v_cache = &kv_cache_v[kv_idx * kv_stride];
            const int offset = pos * AD;
            for (int d = 0; d < AD; d++) {
                k_cache[offset + d] = k[kv_idx * AD + d];
                v_cache[offset + d] = v[kv_idx * AD + d];
            }
        }
    }

    /* ========================================================================
     * CALLER MUST BARRIER HERE
     * All threads need to wait for thread 0 to finish projections
     * ======================================================================== */

    /* ========================================================================
     * PHASE 2: Each thread does attention for its heads only
     * attention_forward_decode_head_major_gqa_flash expects:
     *   - q_token: pointer to start of Q for these heads
     *   - out_token: pointer to start of output for these heads
     *   - num_heads: number of heads THIS THREAD is processing
     * ======================================================================== */
    if (my_heads > 0) {
        attention_forward_decode_head_major_gqa_flash(
            &q[h_start * AD],           /* Q for this thread's heads */
            kv_cache_k, kv_cache_v,
            &attn_out[h_start * AD],    /* Output for this thread's heads */
            my_heads,                   /* Only my_heads, not H */
            KV,                         /* Still need all KV heads for GQA */
            pos + 1, cache_capacity, AD, aligned_head_dim);
    }

    /* ========================================================================
     * CALLER MUST BARRIER HERE
     * Thread 0 needs all threads to finish attention before O projection
     * ======================================================================== */

    /* ========================================================================
     * PHASE 3: Thread 0 does O projection and residual add
     * ======================================================================== */
    if (ith == 0) {
        /* Quantize full attention output for O projection */
        quantize_row_q8_0(attn_out, x_q8_scratch, H * AD);

        const block_q5_0 *wo = (const block_q5_0 *)wo_q5_0;
        const int blocks_per_row = (H * AD) / QK5_0;

        for (int e = 0; e < AE; e++) {
            float dot;
            vec_dot_q5_0_q8_0(H * AD, &dot, &wo[e * blocks_per_row], x_q8_scratch);
            output[e] = dot + (bo ? bo[e] : 0.0f) + residual[e];
        }
    }
}
