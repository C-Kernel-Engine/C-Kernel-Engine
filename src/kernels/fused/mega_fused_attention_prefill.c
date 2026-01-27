/**
 * @file mega_fused_attention_prefill.c
 * @brief Mega-fused prefill attention kernel
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
 * RMSNorm → QKV → RoPE → Flash Attention → OutProj + Residual
 * Writes K/V directly into the KV cache layout (stride = cache_capacity).
 *
 * PERFORMANCE OPTIMIZATION:
 * =========================
 * Uses ck_gemm_nt_head_major_*() to read head-major attention output
 * directly with strided access, eliminating the flatten_head_major()
 * memcpy bottleneck (448 memcpy calls for 32 tokens × 14 heads)
 *
.* TESTING 
 * =======
 *  python3 scripts/bench_mega_fused_attention_prefill.py --q8-outproj --seq-lens 32,64 --iters 3 --warmup 1   
 *
 */

#include "ckernel_engine.h"
#include "ckernel_orchestration.h"
#include "ckernel_quant.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static size_t align_up_size(size_t value, size_t align) {
    return (value + align - 1) & ~(align - 1);
}

static void flatten_head_major(const float *attn_out,
                               float *dst,
                               int tokens,
                               int aligned_embed_dim,
                               int num_heads,
                               int aligned_head_dim)
{
    const size_t head_in_stride = (size_t)tokens * (size_t)aligned_head_dim;
    for (int t = 0; t < tokens; ++t) {
        float *out_row = dst + (size_t)t * (size_t)aligned_embed_dim;
        for (int h = 0; h < num_heads; ++h) {
            const float *src = attn_out + (size_t)h * head_in_stride +
                               (size_t)t * (size_t)aligned_head_dim;
            memcpy(out_row + (size_t)h * (size_t)aligned_head_dim,
                   src,
                   (size_t)aligned_head_dim * sizeof(float));
        }
    }
}

static int ck_q8_0_outproj_enabled(void)
{
    static int cached = -2;
    if (cached != -2) {
        return cached;
    }

    const char *env = getenv("CK_Q8_0_OUTPROJ");
    if (!env || !env[0]) {
        cached = 0;
        return cached;
    }
    if (env[0] == '0' || env[0] == 'n' || env[0] == 'N' ||
        env[0] == 'f' || env[0] == 'F') {
        cached = 0;
    } else {
        cached = 1;
    }
    return cached;
}

static void quantize_attn_out_head_major_q8_0(const float *attn_out,
                                              uint8_t *dst,
                                              int tokens,
                                              int num_heads,
                                              int aligned_head_dim)
{
    const size_t q8_row_bytes = ck_dtype_row_bytes(CK_DT_Q8_0,
                                                   (size_t)aligned_head_dim);
    const size_t head_stride = (size_t)tokens * (size_t)aligned_head_dim;
    for (int h = 0; h < num_heads; ++h) {
        const float *head = attn_out + (size_t)h * head_stride;
        for (int t = 0; t < tokens; ++t) {
            const float *row = head + (size_t)t * (size_t)aligned_head_dim;
            uint8_t *out = dst + ((size_t)h * (size_t)tokens + (size_t)t) *
                                  q8_row_bytes;
            quantize_row_q8_0(row, out, aligned_head_dim);
        }
    }
}

static void out_proj_head_major_q5_0_q8_0(const uint8_t *attn_q8,
                                          const void *wo,
                                          const float *bias,
                                          float *output,
                                          int tokens,
                                          int aligned_embed_dim,
                                          int num_heads,
                                          int aligned_head_dim)
{
    const size_t q8_row_bytes = ck_dtype_row_bytes(CK_DT_Q8_0,
                                                   (size_t)aligned_head_dim);
    const int blocks_per_head = aligned_head_dim / QK5_0;
    const int blocks_per_row = aligned_embed_dim / QK5_0;
    const block_q5_0 *weights = (const block_q5_0 *)wo;

    for (int t = 0; t < tokens; ++t) {
        float *out_row = output + (size_t)t * (size_t)aligned_embed_dim;
        for (int n = 0; n < aligned_embed_dim; ++n) {
            float sum = bias ? bias[n] : 0.0f;
            const block_q5_0 *w_row = weights + (size_t)n * (size_t)blocks_per_row;

            for (int h = 0; h < num_heads; ++h) {
                const uint8_t *a_row = attn_q8 +
                                       ((size_t)h * (size_t)tokens + (size_t)t) *
                                       q8_row_bytes;
                const block_q5_0 *w_head = w_row + (size_t)h * (size_t)blocks_per_head;
                float partial = 0.0f;
                vec_dot_q5_0_q8_0(aligned_head_dim, &partial, w_head, a_row);
                sum += partial;
            }
            out_row[n] = sum;
        }
    }
}

size_t mega_fused_attention_prefill_scratch_size(int tokens,
                                                 int aligned_embed_dim,
                                                 int num_heads,
                                                 int aligned_head_dim)
{
    if (tokens <= 0 || aligned_embed_dim <= 0 || num_heads <= 0 || aligned_head_dim <= 0) {
        return 0;
    }

    const size_t q_bytes = (size_t)num_heads * (size_t)tokens *
                           (size_t)aligned_head_dim * sizeof(float);
    const size_t attn_bytes = q_bytes;
    const size_t proj_bytes = (size_t)tokens * (size_t)aligned_embed_dim * sizeof(float);
    const size_t qkv_scratch = fused_rmsnorm_qkv_prefill_head_major_quant_scratch_size(aligned_embed_dim);

    return align_up_size(q_bytes, 64) +
           align_up_size(attn_bytes, 64) +
           align_up_size(proj_bytes, 64) +
           align_up_size(qkv_scratch, 64);
}

void mega_fused_attention_prefill(
    float *output,
    const float *input,
    const float *residual,
    const float *ln1_gamma,
    const void *wq, const float *bq, CKDataType wq_dt,
    const void *wk, const float *bk, CKDataType wk_dt,
    const void *wv, const float *bv, CKDataType wv_dt,
    const void *wo, const float *bo, CKDataType wo_dt,
    float *kv_cache_k,
    float *kv_cache_v,
    const float *rope_cos,
    const float *rope_sin,
    int start_pos,
    int tokens,
    int cache_capacity,
    int embed_dim,
    int aligned_embed_dim,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int aligned_head_dim,
    float eps,
    void *scratch)
{
    if (!output || !input || !ln1_gamma || !wq || !wk || !wv || !wo ||
        !kv_cache_k || !kv_cache_v || !scratch) {
        return;
    }
    if (tokens <= 0 || cache_capacity <= 0 || embed_dim <= 0 || aligned_embed_dim <= 0 ||
        head_dim <= 0 || aligned_head_dim <= 0 || num_heads <= 0 || num_kv_heads <= 0) {
        return;
    }
    if (aligned_embed_dim < embed_dim || aligned_head_dim < head_dim) {
        return;
    }
    if (start_pos < 0 || start_pos + tokens > cache_capacity) {
        return;
    }

    const size_t q_bytes = (size_t)num_heads * (size_t)tokens *
                           (size_t)aligned_head_dim * sizeof(float);
    const size_t attn_bytes = q_bytes;
    const size_t proj_bytes = (size_t)tokens * (size_t)aligned_embed_dim * sizeof(float);
    const size_t qkv_scratch_bytes = fused_rmsnorm_qkv_prefill_head_major_quant_scratch_size(aligned_embed_dim);

    uint8_t *scratch_bytes = (uint8_t *)scratch;
    float *q = (float *)scratch_bytes;
    scratch_bytes += align_up_size(q_bytes, 64);
    float *attn_out = (float *)scratch_bytes;
    scratch_bytes += align_up_size(attn_bytes, 64);
    float *proj_scratch = (float *)scratch_bytes;
    scratch_bytes += align_up_size(proj_bytes, 64);
    void *qkv_scratch = (void *)scratch_bytes;
    (void)qkv_scratch_bytes;

    float *k_ptr = kv_cache_k + (size_t)start_pos * (size_t)aligned_head_dim;
    float *v_ptr = kv_cache_v + (size_t)start_pos * (size_t)aligned_head_dim;

    if (wq_dt == CK_DT_FP32 && wk_dt == CK_DT_FP32 && wv_dt == CK_DT_FP32) {
        fused_rmsnorm_qkv_prefill_head_major(input,
                                             ln1_gamma,
                                             (const float *)wq, bq,
                                             (const float *)wk, bk,
                                             (const float *)wv, bv,
                                             q,
                                             k_ptr,
                                             v_ptr,
                                             tokens,
                                             embed_dim,
                                             aligned_embed_dim,
                                             num_heads,
                                             num_kv_heads,
                                             head_dim,
                                             aligned_head_dim,
                                             cache_capacity,
                                             eps,
                                             qkv_scratch);
    } else {
        fused_rmsnorm_qkv_prefill_head_major_quant(input,
                                                   ln1_gamma,
                                                   wq, bq, wq_dt,
                                                   wk, bk, wk_dt,
                                                   wv, bv, wv_dt,
                                                   q,
                                                   k_ptr,
                                                   v_ptr,
                                                   tokens,
                                                   embed_dim,
                                                   aligned_embed_dim,
                                                   num_heads,
                                                   num_kv_heads,
                                                   head_dim,
                                                   aligned_head_dim,
                                                   cache_capacity,
                                                   eps,
                                                   qkv_scratch);
    }

    if (rope_cos && rope_sin) {
        rope_forward_qk_strided(q,
                                k_ptr,
                                rope_cos,
                                rope_sin,
                                num_heads,
                                num_kv_heads,
                                tokens,
                                head_dim,
                                aligned_head_dim,
                                start_pos,
                                tokens,
                                cache_capacity);
    }

    if (start_pos == 0) {
        attention_forward_causal_head_major_gqa_flash_strided(q,
                                                             k_ptr,
                                                             v_ptr,
                                                             attn_out,
                                                             num_heads,
                                                             num_kv_heads,
                                                             tokens,
                                                             head_dim,
                                                             aligned_head_dim,
                                                             cache_capacity);
    } else {
        const float scale = 1.0f / sqrtf((float)head_dim);
        const size_t q_head_stride = (size_t)tokens * (size_t)aligned_head_dim;
        const size_t kv_head_stride = (size_t)cache_capacity * (size_t)aligned_head_dim;

        for (int h = 0; h < num_heads; ++h) {
            int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
            const float *k_head = kv_cache_k + (size_t)kv_head * kv_head_stride;
            const float *v_head = kv_cache_v + (size_t)kv_head * kv_head_stride;

            for (int i = 0; i < tokens; ++i) {
                const float *q_vec = q + (size_t)h * q_head_stride + (size_t)i * (size_t)aligned_head_dim;
                float *out_vec = attn_out + (size_t)h * q_head_stride + (size_t)i * (size_t)aligned_head_dim;
                attention_flash_decode(out_vec,
                                       q_vec,
                                       k_head,
                                       v_head,
                                       1,
                                       start_pos + i + 1,
                                       1,
                                       aligned_head_dim,
                                       scale);
            }
        }
    }

    if ((num_heads * aligned_head_dim) != aligned_embed_dim) {
        return;
    }

    if (wo_dt == CK_DT_Q5_0 &&
        ck_q8_0_outproj_enabled() &&
        (aligned_head_dim % QK5_0) == 0 &&
        (aligned_embed_dim % QK5_0) == 0) {
        /* Quantized activations path: Q8_0 attn_out + Q5_0 weights. */
        uint8_t *attn_q8 = (uint8_t *)q;
        quantize_attn_out_head_major_q8_0(attn_out,
                                          attn_q8,
                                          tokens,
                                          num_heads,
                                          aligned_head_dim);
        out_proj_head_major_q5_0_q8_0(attn_q8,
                                      wo,
                                      bo,
                                      output,
                                      tokens,
                                      aligned_embed_dim,
                                      num_heads,
                                      aligned_head_dim);
    } else if (wo_dt == CK_DT_Q5_0 &&
               (aligned_head_dim % QK5_0) == 0 &&
               (aligned_embed_dim % QK5_0) == 0) {
        /* Head-major output projection with Q5_0 weights - no flatten needed */
        ck_gemm_nt_head_major_q5_0(attn_out,
                                    wo,
                                    bo,
                                    output,
                                    tokens,
                                    aligned_embed_dim,
                                    num_heads,
                                    aligned_head_dim);
    } else if (wo_dt == CK_DT_Q8_0 &&
               (aligned_head_dim % QK8_0) == 0 &&
               (aligned_embed_dim % QK8_0) == 0) {
        /* Head-major output projection with Q8_0 weights - no flatten needed */
        ck_gemm_nt_head_major_q8_0(attn_out,
                                    wo,
                                    bo,
                                    output,
                                    tokens,
                                    aligned_embed_dim,
                                    num_heads,
                                    aligned_head_dim);
    } else {
        /* Fallback: flatten then GEMM (slow path) */
        flatten_head_major(attn_out,
                           proj_scratch,
                           tokens,
                           aligned_embed_dim,
                           num_heads,
                           aligned_head_dim);

        ck_gemm_nt_quant(proj_scratch,
                         wo,
                         bo,
                         output,
                         tokens,
                         aligned_embed_dim,
                         aligned_embed_dim,
                         wo_dt);
    }

    if (residual) {
        ck_residual_add_token_major(residual,
                                    output,
                                    output,
                                    tokens,
                                    aligned_embed_dim);
    }

}
