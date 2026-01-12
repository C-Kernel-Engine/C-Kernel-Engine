/**
 * @file ck-kernel-inference.c
 * @brief AUTO-GENERATED: qwen2_0.5b_decode Implementation (IR v6 - Explicit Unrolled)
 *
 * Generated: 2026-01-12T04:06:36.662558 UTC
 * Total Memory: 3.57 GB
 * Mode: decode
 * Layers: 24 (fully unrolled)
 *
 * Per-layer quant types:
 *   Layer 0: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 *   Layer 1: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 *   Layer 2: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 *   ... (21 more layers)
 *
 * DO NOT EDIT - Regenerate with build_ir_v6.py or codegen_v6.py
 */

#define _GNU_SOURCE  /* For MAP_ANONYMOUS, MAP_HUGETLB */

#include "ck-kernel-inference.h"

#include "ckernel_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#ifdef __linux__
#include <sys/mman.h>
#endif

#if QWEN2_0_5B_DECODE_DTYPE_BYTES != 4
#error "qwen2_0.5b_decode: v6 codegen currently supports fp32 only. Use --dtype=fp32."
#endif

/* ============================================================================
 * LOCAL HELPERS (no orchestration dependency)
 * ============================================================================ */

static void qwen2_0_5b_decode_residual_add_token_major(
    const float *a,
    const float *b,
    float *out,
    int tokens,
    int aligned_embed_dim
) {
    if (!a || !b || !out) {
        return;
    }
    for (int t = 0; t < tokens; ++t) {
        const float *pa = a + (size_t)t * (size_t)aligned_embed_dim;
        const float *pb = b + (size_t)t * (size_t)aligned_embed_dim;
        float *pc = out + (size_t)t * (size_t)aligned_embed_dim;
        for (int d = 0; d < aligned_embed_dim; ++d) {
            pc[d] = pa[d] + pb[d];
        }
    }
}

/* ============================================================================
 * MAGIC HEADER
 * ============================================================================ */

typedef struct __attribute__((packed)) {
    uint32_t magic;           /* 0x434B454E */
    uint32_t version;          /* IR version */
    uint64_t total_bytes;
    uint64_t weight_bytes;
    uint64_t activation_bytes;
    uint32_t num_layers;
    uint32_t embed_dim;
    uint32_t num_heads;
    uint32_t vocab_size;
    uint32_t max_seq_len;
    uint32_t canary_count;
    uint8_t  reserved[8];       /* Pad to 64 bytes */
} MagicHeader;

_Static_assert(sizeof(MagicHeader) == 64, "MagicHeader must be 64 bytes");

/* ============================================================================
 * ALLOCATION
 * ============================================================================ */

int qwen2_0_5b_decode_model_allocate(QWEN2_0_5B_DECODEModel *model) {
    size_t total = QWEN2_0_5B_DECODE_TOTAL_BYTES;

#ifdef __linux__
    model->base = mmap(NULL, total,
                       PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                       -1, 0);
    if (model->base == MAP_FAILED) {
        model->base = mmap(NULL, total,
                           PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS,
                           -1, 0);
    }
    if (model->base == MAP_FAILED) {
        perror("mmap failed");
        return -1;
    }
#else
    model->base = aligned_alloc(64, total);
    if (!model->base) {
        perror("aligned_alloc failed");
        return -1;
    }
#endif

    model->total_bytes = total;

    /* Initialize magic header */
    MagicHeader *header = (MagicHeader *)model->base;
    header->magic = QWEN2_0_5B_DECODE_MAGIC;
    header->version = 5;
    header->total_bytes = QWEN2_0_5B_DECODE_TOTAL_BYTES;
    header->weight_bytes = QWEN2_0_5B_DECODE_WEIGHT_BYTES;
    header->activation_bytes = QWEN2_0_5B_DECODE_ACTIVATION_BYTES;
    header->num_layers = QWEN2_0_5B_DECODE_NUM_LAYERS;
    header->embed_dim = QWEN2_0_5B_DECODE_EMBED_DIM;
    header->num_heads = QWEN2_0_5B_DECODE_NUM_HEADS;
    header->vocab_size = QWEN2_0_5B_DECODE_VOCAB_SIZE;
    header->max_seq_len = QWEN2_0_5B_DECODE_MAX_SEQ_LEN;
    header->canary_count = QWEN2_0_5B_DECODE_CANARY_COUNT;

    /* Initialize canary guards */
    for (int i = 0; i < QWEN2_0_5B_DECODE_CANARY_COUNT; i++) {
        uint32_t *ptr = (uint32_t*)((char*)model->base + QWEN2_0_5B_DECODE_CANARIES[i].offset);
        for (int j = 0; j < (QWEN2_0_5B_DECODE_CANARY_SIZE / 4); j++) {
            ptr[j] = QWEN2_0_5B_DECODE_CANARY_VALUE;
        }
    }

    return 0;
}

void qwen2_0_5b_decode_model_free(QWEN2_0_5B_DECODEModel *model) {
    if (!model || !model->base) return;
#ifdef __linux__
    munmap(model->base, model->total_bytes);
#else
    free(model->base);
#endif
    model->base = NULL;
    model->total_bytes = 0;
}

int qwen2_0_5b_decode_verify_canaries(QWEN2_0_5B_DECODEModel *model) {
    int errors = 0;
    uint32_t *ptr;

    for (int i = 0; i < QWEN2_0_5B_DECODE_CANARY_COUNT; i++) {
        ptr = (uint32_t*)((char*)model->base + QWEN2_0_5B_DECODE_CANARIES[i].offset);
        for (int j = 0; j < 4; j++) {
            if (ptr[j] != QWEN2_0_5B_DECODE_CANARY_VALUE) {
                fprintf(stderr, "CANARY CORRUPTION: %s at offset 0x%lX\n",
                        QWEN2_0_5B_DECODE_CANARIES[i].name,
                        QWEN2_0_5B_DECODE_CANARIES[i].offset);
                errors++;
                break;
            }
        }
    }

    return errors;
}

/* ============================================================================
 * ALIGNMENT HELPERS
 * ============================================================================ */

static int qwen2_0_5b_decode_align_elems(int elems, int elem_bytes, int align_bytes) {
    int bytes = elems * elem_bytes;
    int aligned = (bytes + align_bytes - 1) / align_bytes * align_bytes;
    return aligned / elem_bytes;
}

/* ============================================================================
 * ROPE PRECOMPUTE
 * ============================================================================ */

void qwen2_0_5b_decode_precompute_rope(QWEN2_0_5B_DECODEModel *model) {
    const int T = QWEN2_0_5B_DECODE_MAX_SEQ_LEN;
    const int D = QWEN2_0_5B_DECODE_HEAD_DIM / 2;
    const float theta = 1000000.0f;

    float *cos_ptr = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *sin_ptr = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    for (int pos = 0; pos < T; pos++) {
        for (int i = 0; i < D; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / (float)(D * 2));
            float angle = (float)pos * freq;
            cos_ptr[pos * D + i] = cosf(angle);
            sin_ptr[pos * D + i] = sinf(angle);
        }
    }
}

/* ============================================================================
 * EXPLICIT PER-LAYER PREFILL FUNCTIONS (v6 unrolled)
 * ============================================================================ */

/*
 * Layer 0: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_0_prefill(
    QWEN2_0_5B_DECODEModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[0];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_HEADER.embedded_input);
    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *q = QWEN2_0_5B_DECODE_PTR(model, L->q);
    float *k = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *attn_out = QWEN2_0_5B_DECODE_PTR(model, L->attn_out);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *fc1_out = QWEN2_0_5B_DECODE_PTR(model, L->fc1_out);
    float *swiglu_out = QWEN2_0_5B_DECODE_PTR(model, L->swiglu_out);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Q projection (head-major) */
    const size_t wq_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WQ_bytes = (const uint8_t *)WQ;
    for (int h = 0; h < H; ++h) {
        const void *wq_h = (const void *)(WQ_bytes + (size_t)h * wq_head_bytes);
        const float *bq_h = BQ ? (BQ + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;
        gemm_nt_q4_k(ln1_out, wq_h, bq_h, q_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* K projection (head-major) */
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        const float *bk_h = BK ? (BK + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wk_h, bk_h, k_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* V projection (head-major) */
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        const float *bv_h = BV ? (BV + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *v_h = v + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wv_h, bv_h, v_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* RoPE */
    rope_forward_qk(q,
                    k,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    num_tokens,
                    head_dim,
                    aligned_head_dim,
                    0);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash(q,
                                                   k,
                                                   v,
                                                   attn_out,
                                                   H,
                                                   H_kv,
                                                   num_tokens,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Output projection (flatten head-major to token-major) */
    const int K = H * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }
    const float *proj_in = attn_out;
    if (H > 1) {
        if (!proj_scratch) {
            return;
        }
        for (int t = 0; t < num_tokens; ++t) {
            float *dst = proj_scratch + (size_t)t * (size_t)aligned_embed_dim;
            for (int h = 0; h < H; ++h) {
                const float *src = attn_out + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;
                memcpy(dst + (size_t)h * (size_t)aligned_head_dim,
                       src,
                       (size_t)aligned_head_dim * sizeof(float));
            }
        }
        proj_in = proj_scratch;
    }
    gemm_nt_q4_k(proj_in, WO, BO, proj_tmp, num_tokens, aligned_embed_dim, K);

    /* Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 1: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_1_prefill(
    QWEN2_0_5B_DECODEModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[1];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[0].output);
    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *q = QWEN2_0_5B_DECODE_PTR(model, L->q);
    float *k = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *attn_out = QWEN2_0_5B_DECODE_PTR(model, L->attn_out);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *fc1_out = QWEN2_0_5B_DECODE_PTR(model, L->fc1_out);
    float *swiglu_out = QWEN2_0_5B_DECODE_PTR(model, L->swiglu_out);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Q projection (head-major) */
    const size_t wq_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WQ_bytes = (const uint8_t *)WQ;
    for (int h = 0; h < H; ++h) {
        const void *wq_h = (const void *)(WQ_bytes + (size_t)h * wq_head_bytes);
        const float *bq_h = BQ ? (BQ + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;
        gemm_nt_q4_k(ln1_out, wq_h, bq_h, q_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* K projection (head-major) */
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        const float *bk_h = BK ? (BK + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wk_h, bk_h, k_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* V projection (head-major) */
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        const float *bv_h = BV ? (BV + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *v_h = v + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wv_h, bv_h, v_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* RoPE */
    rope_forward_qk(q,
                    k,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    num_tokens,
                    head_dim,
                    aligned_head_dim,
                    0);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash(q,
                                                   k,
                                                   v,
                                                   attn_out,
                                                   H,
                                                   H_kv,
                                                   num_tokens,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Output projection (flatten head-major to token-major) */
    const int K = H * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }
    const float *proj_in = attn_out;
    if (H > 1) {
        if (!proj_scratch) {
            return;
        }
        for (int t = 0; t < num_tokens; ++t) {
            float *dst = proj_scratch + (size_t)t * (size_t)aligned_embed_dim;
            for (int h = 0; h < H; ++h) {
                const float *src = attn_out + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;
                memcpy(dst + (size_t)h * (size_t)aligned_head_dim,
                       src,
                       (size_t)aligned_head_dim * sizeof(float));
            }
        }
        proj_in = proj_scratch;
    }
    gemm_nt_q4_k(proj_in, WO, BO, proj_tmp, num_tokens, aligned_embed_dim, K);

    /* Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 2: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_2_prefill(
    QWEN2_0_5B_DECODEModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[2];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[1].output);
    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *q = QWEN2_0_5B_DECODE_PTR(model, L->q);
    float *k = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *attn_out = QWEN2_0_5B_DECODE_PTR(model, L->attn_out);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *fc1_out = QWEN2_0_5B_DECODE_PTR(model, L->fc1_out);
    float *swiglu_out = QWEN2_0_5B_DECODE_PTR(model, L->swiglu_out);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Q projection (head-major) */
    const size_t wq_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WQ_bytes = (const uint8_t *)WQ;
    for (int h = 0; h < H; ++h) {
        const void *wq_h = (const void *)(WQ_bytes + (size_t)h * wq_head_bytes);
        const float *bq_h = BQ ? (BQ + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;
        gemm_nt_q4_k(ln1_out, wq_h, bq_h, q_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* K projection (head-major) */
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        const float *bk_h = BK ? (BK + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wk_h, bk_h, k_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* V projection (head-major) */
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        const float *bv_h = BV ? (BV + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *v_h = v + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wv_h, bv_h, v_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* RoPE */
    rope_forward_qk(q,
                    k,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    num_tokens,
                    head_dim,
                    aligned_head_dim,
                    0);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash(q,
                                                   k,
                                                   v,
                                                   attn_out,
                                                   H,
                                                   H_kv,
                                                   num_tokens,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Output projection (flatten head-major to token-major) */
    const int K = H * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }
    const float *proj_in = attn_out;
    if (H > 1) {
        if (!proj_scratch) {
            return;
        }
        for (int t = 0; t < num_tokens; ++t) {
            float *dst = proj_scratch + (size_t)t * (size_t)aligned_embed_dim;
            for (int h = 0; h < H; ++h) {
                const float *src = attn_out + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;
                memcpy(dst + (size_t)h * (size_t)aligned_head_dim,
                       src,
                       (size_t)aligned_head_dim * sizeof(float));
            }
        }
        proj_in = proj_scratch;
    }
    gemm_nt_q4_k(proj_in, WO, BO, proj_tmp, num_tokens, aligned_embed_dim, K);

    /* Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 3: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_3_prefill(
    QWEN2_0_5B_DECODEModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[3];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[2].output);
    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *q = QWEN2_0_5B_DECODE_PTR(model, L->q);
    float *k = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *attn_out = QWEN2_0_5B_DECODE_PTR(model, L->attn_out);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *fc1_out = QWEN2_0_5B_DECODE_PTR(model, L->fc1_out);
    float *swiglu_out = QWEN2_0_5B_DECODE_PTR(model, L->swiglu_out);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Q projection (head-major) */
    const size_t wq_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WQ_bytes = (const uint8_t *)WQ;
    for (int h = 0; h < H; ++h) {
        const void *wq_h = (const void *)(WQ_bytes + (size_t)h * wq_head_bytes);
        const float *bq_h = BQ ? (BQ + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;
        gemm_nt_q4_k(ln1_out, wq_h, bq_h, q_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* K projection (head-major) */
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        const float *bk_h = BK ? (BK + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wk_h, bk_h, k_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* V projection (head-major) */
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        const float *bv_h = BV ? (BV + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *v_h = v + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wv_h, bv_h, v_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* RoPE */
    rope_forward_qk(q,
                    k,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    num_tokens,
                    head_dim,
                    aligned_head_dim,
                    0);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash(q,
                                                   k,
                                                   v,
                                                   attn_out,
                                                   H,
                                                   H_kv,
                                                   num_tokens,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Output projection (flatten head-major to token-major) */
    const int K = H * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }
    const float *proj_in = attn_out;
    if (H > 1) {
        if (!proj_scratch) {
            return;
        }
        for (int t = 0; t < num_tokens; ++t) {
            float *dst = proj_scratch + (size_t)t * (size_t)aligned_embed_dim;
            for (int h = 0; h < H; ++h) {
                const float *src = attn_out + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;
                memcpy(dst + (size_t)h * (size_t)aligned_head_dim,
                       src,
                       (size_t)aligned_head_dim * sizeof(float));
            }
        }
        proj_in = proj_scratch;
    }
    gemm_nt_q4_k(proj_in, WO, BO, proj_tmp, num_tokens, aligned_embed_dim, K);

    /* Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 4: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_4_prefill(
    QWEN2_0_5B_DECODEModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[4];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[3].output);
    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *q = QWEN2_0_5B_DECODE_PTR(model, L->q);
    float *k = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *attn_out = QWEN2_0_5B_DECODE_PTR(model, L->attn_out);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *fc1_out = QWEN2_0_5B_DECODE_PTR(model, L->fc1_out);
    float *swiglu_out = QWEN2_0_5B_DECODE_PTR(model, L->swiglu_out);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Q projection (head-major) */
    const size_t wq_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WQ_bytes = (const uint8_t *)WQ;
    for (int h = 0; h < H; ++h) {
        const void *wq_h = (const void *)(WQ_bytes + (size_t)h * wq_head_bytes);
        const float *bq_h = BQ ? (BQ + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;
        gemm_nt_q4_k(ln1_out, wq_h, bq_h, q_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* K projection (head-major) */
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        const float *bk_h = BK ? (BK + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wk_h, bk_h, k_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* V projection (head-major) */
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        const float *bv_h = BV ? (BV + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *v_h = v + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wv_h, bv_h, v_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* RoPE */
    rope_forward_qk(q,
                    k,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    num_tokens,
                    head_dim,
                    aligned_head_dim,
                    0);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash(q,
                                                   k,
                                                   v,
                                                   attn_out,
                                                   H,
                                                   H_kv,
                                                   num_tokens,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Output projection (flatten head-major to token-major) */
    const int K = H * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }
    const float *proj_in = attn_out;
    if (H > 1) {
        if (!proj_scratch) {
            return;
        }
        for (int t = 0; t < num_tokens; ++t) {
            float *dst = proj_scratch + (size_t)t * (size_t)aligned_embed_dim;
            for (int h = 0; h < H; ++h) {
                const float *src = attn_out + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;
                memcpy(dst + (size_t)h * (size_t)aligned_head_dim,
                       src,
                       (size_t)aligned_head_dim * sizeof(float));
            }
        }
        proj_in = proj_scratch;
    }
    gemm_nt_q4_k(proj_in, WO, BO, proj_tmp, num_tokens, aligned_embed_dim, K);

    /* Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 5: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_5_prefill(
    QWEN2_0_5B_DECODEModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[5];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[4].output);
    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *q = QWEN2_0_5B_DECODE_PTR(model, L->q);
    float *k = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *attn_out = QWEN2_0_5B_DECODE_PTR(model, L->attn_out);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *fc1_out = QWEN2_0_5B_DECODE_PTR(model, L->fc1_out);
    float *swiglu_out = QWEN2_0_5B_DECODE_PTR(model, L->swiglu_out);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Q projection (head-major) */
    const size_t wq_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WQ_bytes = (const uint8_t *)WQ;
    for (int h = 0; h < H; ++h) {
        const void *wq_h = (const void *)(WQ_bytes + (size_t)h * wq_head_bytes);
        const float *bq_h = BQ ? (BQ + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;
        gemm_nt_q4_k(ln1_out, wq_h, bq_h, q_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* K projection (head-major) */
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        const float *bk_h = BK ? (BK + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wk_h, bk_h, k_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* V projection (head-major) */
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        const float *bv_h = BV ? (BV + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *v_h = v + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wv_h, bv_h, v_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* RoPE */
    rope_forward_qk(q,
                    k,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    num_tokens,
                    head_dim,
                    aligned_head_dim,
                    0);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash(q,
                                                   k,
                                                   v,
                                                   attn_out,
                                                   H,
                                                   H_kv,
                                                   num_tokens,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Output projection (flatten head-major to token-major) */
    const int K = H * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }
    const float *proj_in = attn_out;
    if (H > 1) {
        if (!proj_scratch) {
            return;
        }
        for (int t = 0; t < num_tokens; ++t) {
            float *dst = proj_scratch + (size_t)t * (size_t)aligned_embed_dim;
            for (int h = 0; h < H; ++h) {
                const float *src = attn_out + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;
                memcpy(dst + (size_t)h * (size_t)aligned_head_dim,
                       src,
                       (size_t)aligned_head_dim * sizeof(float));
            }
        }
        proj_in = proj_scratch;
    }
    gemm_nt_q4_k(proj_in, WO, BO, proj_tmp, num_tokens, aligned_embed_dim, K);

    /* Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 6: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_6_prefill(
    QWEN2_0_5B_DECODEModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[6];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[5].output);
    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *q = QWEN2_0_5B_DECODE_PTR(model, L->q);
    float *k = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *attn_out = QWEN2_0_5B_DECODE_PTR(model, L->attn_out);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *fc1_out = QWEN2_0_5B_DECODE_PTR(model, L->fc1_out);
    float *swiglu_out = QWEN2_0_5B_DECODE_PTR(model, L->swiglu_out);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Q projection (head-major) */
    const size_t wq_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WQ_bytes = (const uint8_t *)WQ;
    for (int h = 0; h < H; ++h) {
        const void *wq_h = (const void *)(WQ_bytes + (size_t)h * wq_head_bytes);
        const float *bq_h = BQ ? (BQ + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;
        gemm_nt_q4_k(ln1_out, wq_h, bq_h, q_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* K projection (head-major) */
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        const float *bk_h = BK ? (BK + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wk_h, bk_h, k_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* V projection (head-major) */
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        const float *bv_h = BV ? (BV + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *v_h = v + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wv_h, bv_h, v_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* RoPE */
    rope_forward_qk(q,
                    k,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    num_tokens,
                    head_dim,
                    aligned_head_dim,
                    0);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash(q,
                                                   k,
                                                   v,
                                                   attn_out,
                                                   H,
                                                   H_kv,
                                                   num_tokens,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Output projection (flatten head-major to token-major) */
    const int K = H * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }
    const float *proj_in = attn_out;
    if (H > 1) {
        if (!proj_scratch) {
            return;
        }
        for (int t = 0; t < num_tokens; ++t) {
            float *dst = proj_scratch + (size_t)t * (size_t)aligned_embed_dim;
            for (int h = 0; h < H; ++h) {
                const float *src = attn_out + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;
                memcpy(dst + (size_t)h * (size_t)aligned_head_dim,
                       src,
                       (size_t)aligned_head_dim * sizeof(float));
            }
        }
        proj_in = proj_scratch;
    }
    gemm_nt_q4_k(proj_in, WO, BO, proj_tmp, num_tokens, aligned_embed_dim, K);

    /* Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 7: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_7_prefill(
    QWEN2_0_5B_DECODEModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[7];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[6].output);
    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *q = QWEN2_0_5B_DECODE_PTR(model, L->q);
    float *k = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *attn_out = QWEN2_0_5B_DECODE_PTR(model, L->attn_out);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *fc1_out = QWEN2_0_5B_DECODE_PTR(model, L->fc1_out);
    float *swiglu_out = QWEN2_0_5B_DECODE_PTR(model, L->swiglu_out);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Q projection (head-major) */
    const size_t wq_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WQ_bytes = (const uint8_t *)WQ;
    for (int h = 0; h < H; ++h) {
        const void *wq_h = (const void *)(WQ_bytes + (size_t)h * wq_head_bytes);
        const float *bq_h = BQ ? (BQ + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;
        gemm_nt_q4_k(ln1_out, wq_h, bq_h, q_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* K projection (head-major) */
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        const float *bk_h = BK ? (BK + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wk_h, bk_h, k_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* V projection (head-major) */
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        const float *bv_h = BV ? (BV + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *v_h = v + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wv_h, bv_h, v_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* RoPE */
    rope_forward_qk(q,
                    k,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    num_tokens,
                    head_dim,
                    aligned_head_dim,
                    0);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash(q,
                                                   k,
                                                   v,
                                                   attn_out,
                                                   H,
                                                   H_kv,
                                                   num_tokens,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Output projection (flatten head-major to token-major) */
    const int K = H * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }
    const float *proj_in = attn_out;
    if (H > 1) {
        if (!proj_scratch) {
            return;
        }
        for (int t = 0; t < num_tokens; ++t) {
            float *dst = proj_scratch + (size_t)t * (size_t)aligned_embed_dim;
            for (int h = 0; h < H; ++h) {
                const float *src = attn_out + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;
                memcpy(dst + (size_t)h * (size_t)aligned_head_dim,
                       src,
                       (size_t)aligned_head_dim * sizeof(float));
            }
        }
        proj_in = proj_scratch;
    }
    gemm_nt_q4_k(proj_in, WO, BO, proj_tmp, num_tokens, aligned_embed_dim, K);

    /* Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 8: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_8_prefill(
    QWEN2_0_5B_DECODEModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[8];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[7].output);
    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *q = QWEN2_0_5B_DECODE_PTR(model, L->q);
    float *k = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *attn_out = QWEN2_0_5B_DECODE_PTR(model, L->attn_out);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *fc1_out = QWEN2_0_5B_DECODE_PTR(model, L->fc1_out);
    float *swiglu_out = QWEN2_0_5B_DECODE_PTR(model, L->swiglu_out);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Q projection (head-major) */
    const size_t wq_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WQ_bytes = (const uint8_t *)WQ;
    for (int h = 0; h < H; ++h) {
        const void *wq_h = (const void *)(WQ_bytes + (size_t)h * wq_head_bytes);
        const float *bq_h = BQ ? (BQ + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;
        gemm_nt_q4_k(ln1_out, wq_h, bq_h, q_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* K projection (head-major) */
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        const float *bk_h = BK ? (BK + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wk_h, bk_h, k_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* V projection (head-major) */
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        const float *bv_h = BV ? (BV + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *v_h = v + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wv_h, bv_h, v_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* RoPE */
    rope_forward_qk(q,
                    k,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    num_tokens,
                    head_dim,
                    aligned_head_dim,
                    0);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash(q,
                                                   k,
                                                   v,
                                                   attn_out,
                                                   H,
                                                   H_kv,
                                                   num_tokens,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Output projection (flatten head-major to token-major) */
    const int K = H * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }
    const float *proj_in = attn_out;
    if (H > 1) {
        if (!proj_scratch) {
            return;
        }
        for (int t = 0; t < num_tokens; ++t) {
            float *dst = proj_scratch + (size_t)t * (size_t)aligned_embed_dim;
            for (int h = 0; h < H; ++h) {
                const float *src = attn_out + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;
                memcpy(dst + (size_t)h * (size_t)aligned_head_dim,
                       src,
                       (size_t)aligned_head_dim * sizeof(float));
            }
        }
        proj_in = proj_scratch;
    }
    gemm_nt_q4_k(proj_in, WO, BO, proj_tmp, num_tokens, aligned_embed_dim, K);

    /* Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 9: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_9_prefill(
    QWEN2_0_5B_DECODEModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[9];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[8].output);
    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *q = QWEN2_0_5B_DECODE_PTR(model, L->q);
    float *k = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *attn_out = QWEN2_0_5B_DECODE_PTR(model, L->attn_out);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *fc1_out = QWEN2_0_5B_DECODE_PTR(model, L->fc1_out);
    float *swiglu_out = QWEN2_0_5B_DECODE_PTR(model, L->swiglu_out);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Q projection (head-major) */
    const size_t wq_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WQ_bytes = (const uint8_t *)WQ;
    for (int h = 0; h < H; ++h) {
        const void *wq_h = (const void *)(WQ_bytes + (size_t)h * wq_head_bytes);
        const float *bq_h = BQ ? (BQ + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;
        gemm_nt_q4_k(ln1_out, wq_h, bq_h, q_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* K projection (head-major) */
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        const float *bk_h = BK ? (BK + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wk_h, bk_h, k_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* V projection (head-major) */
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        const float *bv_h = BV ? (BV + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *v_h = v + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wv_h, bv_h, v_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* RoPE */
    rope_forward_qk(q,
                    k,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    num_tokens,
                    head_dim,
                    aligned_head_dim,
                    0);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash(q,
                                                   k,
                                                   v,
                                                   attn_out,
                                                   H,
                                                   H_kv,
                                                   num_tokens,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Output projection (flatten head-major to token-major) */
    const int K = H * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }
    const float *proj_in = attn_out;
    if (H > 1) {
        if (!proj_scratch) {
            return;
        }
        for (int t = 0; t < num_tokens; ++t) {
            float *dst = proj_scratch + (size_t)t * (size_t)aligned_embed_dim;
            for (int h = 0; h < H; ++h) {
                const float *src = attn_out + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;
                memcpy(dst + (size_t)h * (size_t)aligned_head_dim,
                       src,
                       (size_t)aligned_head_dim * sizeof(float));
            }
        }
        proj_in = proj_scratch;
    }
    gemm_nt_q4_k(proj_in, WO, BO, proj_tmp, num_tokens, aligned_embed_dim, K);

    /* Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 10: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_10_prefill(
    QWEN2_0_5B_DECODEModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[10];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[9].output);
    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *q = QWEN2_0_5B_DECODE_PTR(model, L->q);
    float *k = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *attn_out = QWEN2_0_5B_DECODE_PTR(model, L->attn_out);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *fc1_out = QWEN2_0_5B_DECODE_PTR(model, L->fc1_out);
    float *swiglu_out = QWEN2_0_5B_DECODE_PTR(model, L->swiglu_out);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Q projection (head-major) */
    const size_t wq_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WQ_bytes = (const uint8_t *)WQ;
    for (int h = 0; h < H; ++h) {
        const void *wq_h = (const void *)(WQ_bytes + (size_t)h * wq_head_bytes);
        const float *bq_h = BQ ? (BQ + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;
        gemm_nt_q4_k(ln1_out, wq_h, bq_h, q_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* K projection (head-major) */
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        const float *bk_h = BK ? (BK + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wk_h, bk_h, k_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* V projection (head-major) */
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        const float *bv_h = BV ? (BV + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *v_h = v + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wv_h, bv_h, v_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* RoPE */
    rope_forward_qk(q,
                    k,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    num_tokens,
                    head_dim,
                    aligned_head_dim,
                    0);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash(q,
                                                   k,
                                                   v,
                                                   attn_out,
                                                   H,
                                                   H_kv,
                                                   num_tokens,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Output projection (flatten head-major to token-major) */
    const int K = H * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }
    const float *proj_in = attn_out;
    if (H > 1) {
        if (!proj_scratch) {
            return;
        }
        for (int t = 0; t < num_tokens; ++t) {
            float *dst = proj_scratch + (size_t)t * (size_t)aligned_embed_dim;
            for (int h = 0; h < H; ++h) {
                const float *src = attn_out + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;
                memcpy(dst + (size_t)h * (size_t)aligned_head_dim,
                       src,
                       (size_t)aligned_head_dim * sizeof(float));
            }
        }
        proj_in = proj_scratch;
    }
    gemm_nt_q4_k(proj_in, WO, BO, proj_tmp, num_tokens, aligned_embed_dim, K);

    /* Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 11: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_11_prefill(
    QWEN2_0_5B_DECODEModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[11];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[10].output);
    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *q = QWEN2_0_5B_DECODE_PTR(model, L->q);
    float *k = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *attn_out = QWEN2_0_5B_DECODE_PTR(model, L->attn_out);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *fc1_out = QWEN2_0_5B_DECODE_PTR(model, L->fc1_out);
    float *swiglu_out = QWEN2_0_5B_DECODE_PTR(model, L->swiglu_out);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Q projection (head-major) */
    const size_t wq_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WQ_bytes = (const uint8_t *)WQ;
    for (int h = 0; h < H; ++h) {
        const void *wq_h = (const void *)(WQ_bytes + (size_t)h * wq_head_bytes);
        const float *bq_h = BQ ? (BQ + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;
        gemm_nt_q4_k(ln1_out, wq_h, bq_h, q_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* K projection (head-major) */
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        const float *bk_h = BK ? (BK + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wk_h, bk_h, k_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* V projection (head-major) */
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        const float *bv_h = BV ? (BV + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *v_h = v + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wv_h, bv_h, v_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* RoPE */
    rope_forward_qk(q,
                    k,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    num_tokens,
                    head_dim,
                    aligned_head_dim,
                    0);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash(q,
                                                   k,
                                                   v,
                                                   attn_out,
                                                   H,
                                                   H_kv,
                                                   num_tokens,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Output projection (flatten head-major to token-major) */
    const int K = H * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }
    const float *proj_in = attn_out;
    if (H > 1) {
        if (!proj_scratch) {
            return;
        }
        for (int t = 0; t < num_tokens; ++t) {
            float *dst = proj_scratch + (size_t)t * (size_t)aligned_embed_dim;
            for (int h = 0; h < H; ++h) {
                const float *src = attn_out + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;
                memcpy(dst + (size_t)h * (size_t)aligned_head_dim,
                       src,
                       (size_t)aligned_head_dim * sizeof(float));
            }
        }
        proj_in = proj_scratch;
    }
    gemm_nt_q4_k(proj_in, WO, BO, proj_tmp, num_tokens, aligned_embed_dim, K);

    /* Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 12: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_12_prefill(
    QWEN2_0_5B_DECODEModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[12];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[11].output);
    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *q = QWEN2_0_5B_DECODE_PTR(model, L->q);
    float *k = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *attn_out = QWEN2_0_5B_DECODE_PTR(model, L->attn_out);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *fc1_out = QWEN2_0_5B_DECODE_PTR(model, L->fc1_out);
    float *swiglu_out = QWEN2_0_5B_DECODE_PTR(model, L->swiglu_out);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Q projection (head-major) */
    const size_t wq_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WQ_bytes = (const uint8_t *)WQ;
    for (int h = 0; h < H; ++h) {
        const void *wq_h = (const void *)(WQ_bytes + (size_t)h * wq_head_bytes);
        const float *bq_h = BQ ? (BQ + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;
        gemm_nt_q4_k(ln1_out, wq_h, bq_h, q_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* K projection (head-major) */
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        const float *bk_h = BK ? (BK + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wk_h, bk_h, k_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* V projection (head-major) */
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        const float *bv_h = BV ? (BV + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *v_h = v + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wv_h, bv_h, v_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* RoPE */
    rope_forward_qk(q,
                    k,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    num_tokens,
                    head_dim,
                    aligned_head_dim,
                    0);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash(q,
                                                   k,
                                                   v,
                                                   attn_out,
                                                   H,
                                                   H_kv,
                                                   num_tokens,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Output projection (flatten head-major to token-major) */
    const int K = H * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }
    const float *proj_in = attn_out;
    if (H > 1) {
        if (!proj_scratch) {
            return;
        }
        for (int t = 0; t < num_tokens; ++t) {
            float *dst = proj_scratch + (size_t)t * (size_t)aligned_embed_dim;
            for (int h = 0; h < H; ++h) {
                const float *src = attn_out + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;
                memcpy(dst + (size_t)h * (size_t)aligned_head_dim,
                       src,
                       (size_t)aligned_head_dim * sizeof(float));
            }
        }
        proj_in = proj_scratch;
    }
    gemm_nt_q4_k(proj_in, WO, BO, proj_tmp, num_tokens, aligned_embed_dim, K);

    /* Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 13: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_13_prefill(
    QWEN2_0_5B_DECODEModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[13];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[12].output);
    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *q = QWEN2_0_5B_DECODE_PTR(model, L->q);
    float *k = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *attn_out = QWEN2_0_5B_DECODE_PTR(model, L->attn_out);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *fc1_out = QWEN2_0_5B_DECODE_PTR(model, L->fc1_out);
    float *swiglu_out = QWEN2_0_5B_DECODE_PTR(model, L->swiglu_out);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Q projection (head-major) */
    const size_t wq_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WQ_bytes = (const uint8_t *)WQ;
    for (int h = 0; h < H; ++h) {
        const void *wq_h = (const void *)(WQ_bytes + (size_t)h * wq_head_bytes);
        const float *bq_h = BQ ? (BQ + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;
        gemm_nt_q4_k(ln1_out, wq_h, bq_h, q_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* K projection (head-major) */
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        const float *bk_h = BK ? (BK + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wk_h, bk_h, k_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* V projection (head-major) */
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        const float *bv_h = BV ? (BV + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *v_h = v + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wv_h, bv_h, v_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* RoPE */
    rope_forward_qk(q,
                    k,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    num_tokens,
                    head_dim,
                    aligned_head_dim,
                    0);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash(q,
                                                   k,
                                                   v,
                                                   attn_out,
                                                   H,
                                                   H_kv,
                                                   num_tokens,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Output projection (flatten head-major to token-major) */
    const int K = H * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }
    const float *proj_in = attn_out;
    if (H > 1) {
        if (!proj_scratch) {
            return;
        }
        for (int t = 0; t < num_tokens; ++t) {
            float *dst = proj_scratch + (size_t)t * (size_t)aligned_embed_dim;
            for (int h = 0; h < H; ++h) {
                const float *src = attn_out + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;
                memcpy(dst + (size_t)h * (size_t)aligned_head_dim,
                       src,
                       (size_t)aligned_head_dim * sizeof(float));
            }
        }
        proj_in = proj_scratch;
    }
    gemm_nt_q4_k(proj_in, WO, BO, proj_tmp, num_tokens, aligned_embed_dim, K);

    /* Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 14: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_14_prefill(
    QWEN2_0_5B_DECODEModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[14];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[13].output);
    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *q = QWEN2_0_5B_DECODE_PTR(model, L->q);
    float *k = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *attn_out = QWEN2_0_5B_DECODE_PTR(model, L->attn_out);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *fc1_out = QWEN2_0_5B_DECODE_PTR(model, L->fc1_out);
    float *swiglu_out = QWEN2_0_5B_DECODE_PTR(model, L->swiglu_out);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Q projection (head-major) */
    const size_t wq_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WQ_bytes = (const uint8_t *)WQ;
    for (int h = 0; h < H; ++h) {
        const void *wq_h = (const void *)(WQ_bytes + (size_t)h * wq_head_bytes);
        const float *bq_h = BQ ? (BQ + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;
        gemm_nt_q4_k(ln1_out, wq_h, bq_h, q_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* K projection (head-major) */
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        const float *bk_h = BK ? (BK + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wk_h, bk_h, k_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* V projection (head-major) */
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        const float *bv_h = BV ? (BV + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *v_h = v + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wv_h, bv_h, v_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* RoPE */
    rope_forward_qk(q,
                    k,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    num_tokens,
                    head_dim,
                    aligned_head_dim,
                    0);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash(q,
                                                   k,
                                                   v,
                                                   attn_out,
                                                   H,
                                                   H_kv,
                                                   num_tokens,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Output projection (flatten head-major to token-major) */
    const int K = H * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }
    const float *proj_in = attn_out;
    if (H > 1) {
        if (!proj_scratch) {
            return;
        }
        for (int t = 0; t < num_tokens; ++t) {
            float *dst = proj_scratch + (size_t)t * (size_t)aligned_embed_dim;
            for (int h = 0; h < H; ++h) {
                const float *src = attn_out + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;
                memcpy(dst + (size_t)h * (size_t)aligned_head_dim,
                       src,
                       (size_t)aligned_head_dim * sizeof(float));
            }
        }
        proj_in = proj_scratch;
    }
    gemm_nt_q4_k(proj_in, WO, BO, proj_tmp, num_tokens, aligned_embed_dim, K);

    /* Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 15: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_15_prefill(
    QWEN2_0_5B_DECODEModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[15];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[14].output);
    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *q = QWEN2_0_5B_DECODE_PTR(model, L->q);
    float *k = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *attn_out = QWEN2_0_5B_DECODE_PTR(model, L->attn_out);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *fc1_out = QWEN2_0_5B_DECODE_PTR(model, L->fc1_out);
    float *swiglu_out = QWEN2_0_5B_DECODE_PTR(model, L->swiglu_out);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Q projection (head-major) */
    const size_t wq_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WQ_bytes = (const uint8_t *)WQ;
    for (int h = 0; h < H; ++h) {
        const void *wq_h = (const void *)(WQ_bytes + (size_t)h * wq_head_bytes);
        const float *bq_h = BQ ? (BQ + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;
        gemm_nt_q4_k(ln1_out, wq_h, bq_h, q_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* K projection (head-major) */
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        const float *bk_h = BK ? (BK + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wk_h, bk_h, k_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* V projection (head-major) */
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        const float *bv_h = BV ? (BV + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *v_h = v + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wv_h, bv_h, v_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* RoPE */
    rope_forward_qk(q,
                    k,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    num_tokens,
                    head_dim,
                    aligned_head_dim,
                    0);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash(q,
                                                   k,
                                                   v,
                                                   attn_out,
                                                   H,
                                                   H_kv,
                                                   num_tokens,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Output projection (flatten head-major to token-major) */
    const int K = H * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }
    const float *proj_in = attn_out;
    if (H > 1) {
        if (!proj_scratch) {
            return;
        }
        for (int t = 0; t < num_tokens; ++t) {
            float *dst = proj_scratch + (size_t)t * (size_t)aligned_embed_dim;
            for (int h = 0; h < H; ++h) {
                const float *src = attn_out + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;
                memcpy(dst + (size_t)h * (size_t)aligned_head_dim,
                       src,
                       (size_t)aligned_head_dim * sizeof(float));
            }
        }
        proj_in = proj_scratch;
    }
    gemm_nt_q4_k(proj_in, WO, BO, proj_tmp, num_tokens, aligned_embed_dim, K);

    /* Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 16: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_16_prefill(
    QWEN2_0_5B_DECODEModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[16];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[15].output);
    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *q = QWEN2_0_5B_DECODE_PTR(model, L->q);
    float *k = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *attn_out = QWEN2_0_5B_DECODE_PTR(model, L->attn_out);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *fc1_out = QWEN2_0_5B_DECODE_PTR(model, L->fc1_out);
    float *swiglu_out = QWEN2_0_5B_DECODE_PTR(model, L->swiglu_out);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Q projection (head-major) */
    const size_t wq_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WQ_bytes = (const uint8_t *)WQ;
    for (int h = 0; h < H; ++h) {
        const void *wq_h = (const void *)(WQ_bytes + (size_t)h * wq_head_bytes);
        const float *bq_h = BQ ? (BQ + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;
        gemm_nt_q4_k(ln1_out, wq_h, bq_h, q_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* K projection (head-major) */
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        const float *bk_h = BK ? (BK + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wk_h, bk_h, k_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* V projection (head-major) */
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        const float *bv_h = BV ? (BV + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *v_h = v + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wv_h, bv_h, v_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* RoPE */
    rope_forward_qk(q,
                    k,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    num_tokens,
                    head_dim,
                    aligned_head_dim,
                    0);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash(q,
                                                   k,
                                                   v,
                                                   attn_out,
                                                   H,
                                                   H_kv,
                                                   num_tokens,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Output projection (flatten head-major to token-major) */
    const int K = H * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }
    const float *proj_in = attn_out;
    if (H > 1) {
        if (!proj_scratch) {
            return;
        }
        for (int t = 0; t < num_tokens; ++t) {
            float *dst = proj_scratch + (size_t)t * (size_t)aligned_embed_dim;
            for (int h = 0; h < H; ++h) {
                const float *src = attn_out + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;
                memcpy(dst + (size_t)h * (size_t)aligned_head_dim,
                       src,
                       (size_t)aligned_head_dim * sizeof(float));
            }
        }
        proj_in = proj_scratch;
    }
    gemm_nt_q4_k(proj_in, WO, BO, proj_tmp, num_tokens, aligned_embed_dim, K);

    /* Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 17: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_17_prefill(
    QWEN2_0_5B_DECODEModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[17];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[16].output);
    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *q = QWEN2_0_5B_DECODE_PTR(model, L->q);
    float *k = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *attn_out = QWEN2_0_5B_DECODE_PTR(model, L->attn_out);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *fc1_out = QWEN2_0_5B_DECODE_PTR(model, L->fc1_out);
    float *swiglu_out = QWEN2_0_5B_DECODE_PTR(model, L->swiglu_out);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Q projection (head-major) */
    const size_t wq_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WQ_bytes = (const uint8_t *)WQ;
    for (int h = 0; h < H; ++h) {
        const void *wq_h = (const void *)(WQ_bytes + (size_t)h * wq_head_bytes);
        const float *bq_h = BQ ? (BQ + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;
        gemm_nt_q4_k(ln1_out, wq_h, bq_h, q_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* K projection (head-major) */
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        const float *bk_h = BK ? (BK + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wk_h, bk_h, k_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* V projection (head-major) */
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        const float *bv_h = BV ? (BV + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *v_h = v + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wv_h, bv_h, v_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* RoPE */
    rope_forward_qk(q,
                    k,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    num_tokens,
                    head_dim,
                    aligned_head_dim,
                    0);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash(q,
                                                   k,
                                                   v,
                                                   attn_out,
                                                   H,
                                                   H_kv,
                                                   num_tokens,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Output projection (flatten head-major to token-major) */
    const int K = H * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }
    const float *proj_in = attn_out;
    if (H > 1) {
        if (!proj_scratch) {
            return;
        }
        for (int t = 0; t < num_tokens; ++t) {
            float *dst = proj_scratch + (size_t)t * (size_t)aligned_embed_dim;
            for (int h = 0; h < H; ++h) {
                const float *src = attn_out + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;
                memcpy(dst + (size_t)h * (size_t)aligned_head_dim,
                       src,
                       (size_t)aligned_head_dim * sizeof(float));
            }
        }
        proj_in = proj_scratch;
    }
    gemm_nt_q4_k(proj_in, WO, BO, proj_tmp, num_tokens, aligned_embed_dim, K);

    /* Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 18: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_18_prefill(
    QWEN2_0_5B_DECODEModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[18];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[17].output);
    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *q = QWEN2_0_5B_DECODE_PTR(model, L->q);
    float *k = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *attn_out = QWEN2_0_5B_DECODE_PTR(model, L->attn_out);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *fc1_out = QWEN2_0_5B_DECODE_PTR(model, L->fc1_out);
    float *swiglu_out = QWEN2_0_5B_DECODE_PTR(model, L->swiglu_out);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Q projection (head-major) */
    const size_t wq_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WQ_bytes = (const uint8_t *)WQ;
    for (int h = 0; h < H; ++h) {
        const void *wq_h = (const void *)(WQ_bytes + (size_t)h * wq_head_bytes);
        const float *bq_h = BQ ? (BQ + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;
        gemm_nt_q4_k(ln1_out, wq_h, bq_h, q_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* K projection (head-major) */
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        const float *bk_h = BK ? (BK + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wk_h, bk_h, k_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* V projection (head-major) */
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        const float *bv_h = BV ? (BV + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *v_h = v + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wv_h, bv_h, v_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* RoPE */
    rope_forward_qk(q,
                    k,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    num_tokens,
                    head_dim,
                    aligned_head_dim,
                    0);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash(q,
                                                   k,
                                                   v,
                                                   attn_out,
                                                   H,
                                                   H_kv,
                                                   num_tokens,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Output projection (flatten head-major to token-major) */
    const int K = H * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }
    const float *proj_in = attn_out;
    if (H > 1) {
        if (!proj_scratch) {
            return;
        }
        for (int t = 0; t < num_tokens; ++t) {
            float *dst = proj_scratch + (size_t)t * (size_t)aligned_embed_dim;
            for (int h = 0; h < H; ++h) {
                const float *src = attn_out + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;
                memcpy(dst + (size_t)h * (size_t)aligned_head_dim,
                       src,
                       (size_t)aligned_head_dim * sizeof(float));
            }
        }
        proj_in = proj_scratch;
    }
    gemm_nt_q4_k(proj_in, WO, BO, proj_tmp, num_tokens, aligned_embed_dim, K);

    /* Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 19: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_19_prefill(
    QWEN2_0_5B_DECODEModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[19];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[18].output);
    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *q = QWEN2_0_5B_DECODE_PTR(model, L->q);
    float *k = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *attn_out = QWEN2_0_5B_DECODE_PTR(model, L->attn_out);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *fc1_out = QWEN2_0_5B_DECODE_PTR(model, L->fc1_out);
    float *swiglu_out = QWEN2_0_5B_DECODE_PTR(model, L->swiglu_out);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Q projection (head-major) */
    const size_t wq_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WQ_bytes = (const uint8_t *)WQ;
    for (int h = 0; h < H; ++h) {
        const void *wq_h = (const void *)(WQ_bytes + (size_t)h * wq_head_bytes);
        const float *bq_h = BQ ? (BQ + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;
        gemm_nt_q4_k(ln1_out, wq_h, bq_h, q_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* K projection (head-major) */
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        const float *bk_h = BK ? (BK + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wk_h, bk_h, k_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* V projection (head-major) */
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        const float *bv_h = BV ? (BV + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *v_h = v + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wv_h, bv_h, v_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* RoPE */
    rope_forward_qk(q,
                    k,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    num_tokens,
                    head_dim,
                    aligned_head_dim,
                    0);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash(q,
                                                   k,
                                                   v,
                                                   attn_out,
                                                   H,
                                                   H_kv,
                                                   num_tokens,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Output projection (flatten head-major to token-major) */
    const int K = H * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }
    const float *proj_in = attn_out;
    if (H > 1) {
        if (!proj_scratch) {
            return;
        }
        for (int t = 0; t < num_tokens; ++t) {
            float *dst = proj_scratch + (size_t)t * (size_t)aligned_embed_dim;
            for (int h = 0; h < H; ++h) {
                const float *src = attn_out + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;
                memcpy(dst + (size_t)h * (size_t)aligned_head_dim,
                       src,
                       (size_t)aligned_head_dim * sizeof(float));
            }
        }
        proj_in = proj_scratch;
    }
    gemm_nt_q4_k(proj_in, WO, BO, proj_tmp, num_tokens, aligned_embed_dim, K);

    /* Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 20: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_20_prefill(
    QWEN2_0_5B_DECODEModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[20];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[19].output);
    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *q = QWEN2_0_5B_DECODE_PTR(model, L->q);
    float *k = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *attn_out = QWEN2_0_5B_DECODE_PTR(model, L->attn_out);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *fc1_out = QWEN2_0_5B_DECODE_PTR(model, L->fc1_out);
    float *swiglu_out = QWEN2_0_5B_DECODE_PTR(model, L->swiglu_out);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Q projection (head-major) */
    const size_t wq_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WQ_bytes = (const uint8_t *)WQ;
    for (int h = 0; h < H; ++h) {
        const void *wq_h = (const void *)(WQ_bytes + (size_t)h * wq_head_bytes);
        const float *bq_h = BQ ? (BQ + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;
        gemm_nt_q4_k(ln1_out, wq_h, bq_h, q_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* K projection (head-major) */
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        const float *bk_h = BK ? (BK + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wk_h, bk_h, k_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* V projection (head-major) */
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        const float *bv_h = BV ? (BV + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *v_h = v + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wv_h, bv_h, v_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* RoPE */
    rope_forward_qk(q,
                    k,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    num_tokens,
                    head_dim,
                    aligned_head_dim,
                    0);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash(q,
                                                   k,
                                                   v,
                                                   attn_out,
                                                   H,
                                                   H_kv,
                                                   num_tokens,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Output projection (flatten head-major to token-major) */
    const int K = H * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }
    const float *proj_in = attn_out;
    if (H > 1) {
        if (!proj_scratch) {
            return;
        }
        for (int t = 0; t < num_tokens; ++t) {
            float *dst = proj_scratch + (size_t)t * (size_t)aligned_embed_dim;
            for (int h = 0; h < H; ++h) {
                const float *src = attn_out + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;
                memcpy(dst + (size_t)h * (size_t)aligned_head_dim,
                       src,
                       (size_t)aligned_head_dim * sizeof(float));
            }
        }
        proj_in = proj_scratch;
    }
    gemm_nt_q4_k(proj_in, WO, BO, proj_tmp, num_tokens, aligned_embed_dim, K);

    /* Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 21: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_21_prefill(
    QWEN2_0_5B_DECODEModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[21];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[20].output);
    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *q = QWEN2_0_5B_DECODE_PTR(model, L->q);
    float *k = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *attn_out = QWEN2_0_5B_DECODE_PTR(model, L->attn_out);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *fc1_out = QWEN2_0_5B_DECODE_PTR(model, L->fc1_out);
    float *swiglu_out = QWEN2_0_5B_DECODE_PTR(model, L->swiglu_out);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Q projection (head-major) */
    const size_t wq_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WQ_bytes = (const uint8_t *)WQ;
    for (int h = 0; h < H; ++h) {
        const void *wq_h = (const void *)(WQ_bytes + (size_t)h * wq_head_bytes);
        const float *bq_h = BQ ? (BQ + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;
        gemm_nt_q4_k(ln1_out, wq_h, bq_h, q_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* K projection (head-major) */
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        const float *bk_h = BK ? (BK + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wk_h, bk_h, k_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* V projection (head-major) */
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        const float *bv_h = BV ? (BV + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *v_h = v + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wv_h, bv_h, v_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* RoPE */
    rope_forward_qk(q,
                    k,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    num_tokens,
                    head_dim,
                    aligned_head_dim,
                    0);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash(q,
                                                   k,
                                                   v,
                                                   attn_out,
                                                   H,
                                                   H_kv,
                                                   num_tokens,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Output projection (flatten head-major to token-major) */
    const int K = H * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }
    const float *proj_in = attn_out;
    if (H > 1) {
        if (!proj_scratch) {
            return;
        }
        for (int t = 0; t < num_tokens; ++t) {
            float *dst = proj_scratch + (size_t)t * (size_t)aligned_embed_dim;
            for (int h = 0; h < H; ++h) {
                const float *src = attn_out + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;
                memcpy(dst + (size_t)h * (size_t)aligned_head_dim,
                       src,
                       (size_t)aligned_head_dim * sizeof(float));
            }
        }
        proj_in = proj_scratch;
    }
    gemm_nt_q4_k(proj_in, WO, BO, proj_tmp, num_tokens, aligned_embed_dim, K);

    /* Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 22: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_22_prefill(
    QWEN2_0_5B_DECODEModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[22];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[21].output);
    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *q = QWEN2_0_5B_DECODE_PTR(model, L->q);
    float *k = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *attn_out = QWEN2_0_5B_DECODE_PTR(model, L->attn_out);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *fc1_out = QWEN2_0_5B_DECODE_PTR(model, L->fc1_out);
    float *swiglu_out = QWEN2_0_5B_DECODE_PTR(model, L->swiglu_out);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Q projection (head-major) */
    const size_t wq_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WQ_bytes = (const uint8_t *)WQ;
    for (int h = 0; h < H; ++h) {
        const void *wq_h = (const void *)(WQ_bytes + (size_t)h * wq_head_bytes);
        const float *bq_h = BQ ? (BQ + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;
        gemm_nt_q4_k(ln1_out, wq_h, bq_h, q_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* K projection (head-major) */
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        const float *bk_h = BK ? (BK + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wk_h, bk_h, k_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* V projection (head-major) */
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        const float *bv_h = BV ? (BV + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *v_h = v + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wv_h, bv_h, v_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* RoPE */
    rope_forward_qk(q,
                    k,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    num_tokens,
                    head_dim,
                    aligned_head_dim,
                    0);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash(q,
                                                   k,
                                                   v,
                                                   attn_out,
                                                   H,
                                                   H_kv,
                                                   num_tokens,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Output projection (flatten head-major to token-major) */
    const int K = H * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }
    const float *proj_in = attn_out;
    if (H > 1) {
        if (!proj_scratch) {
            return;
        }
        for (int t = 0; t < num_tokens; ++t) {
            float *dst = proj_scratch + (size_t)t * (size_t)aligned_embed_dim;
            for (int h = 0; h < H; ++h) {
                const float *src = attn_out + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;
                memcpy(dst + (size_t)h * (size_t)aligned_head_dim,
                       src,
                       (size_t)aligned_head_dim * sizeof(float));
            }
        }
        proj_in = proj_scratch;
    }
    gemm_nt_q4_k(proj_in, WO, BO, proj_tmp, num_tokens, aligned_embed_dim, K);

    /* Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 23: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_23_prefill(
    QWEN2_0_5B_DECODEModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[23];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[22].output);
    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *q = QWEN2_0_5B_DECODE_PTR(model, L->q);
    float *k = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *attn_out = QWEN2_0_5B_DECODE_PTR(model, L->attn_out);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *fc1_out = QWEN2_0_5B_DECODE_PTR(model, L->fc1_out);
    float *swiglu_out = QWEN2_0_5B_DECODE_PTR(model, L->swiglu_out);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Q projection (head-major) */
    const size_t wq_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WQ_bytes = (const uint8_t *)WQ;
    for (int h = 0; h < H; ++h) {
        const void *wq_h = (const void *)(WQ_bytes + (size_t)h * wq_head_bytes);
        const float *bq_h = BQ ? (BQ + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;
        gemm_nt_q4_k(ln1_out, wq_h, bq_h, q_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* K projection (head-major) */
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        const float *bk_h = BK ? (BK + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wk_h, bk_h, k_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* V projection (head-major) */
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        const float *bv_h = BV ? (BV + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *v_h = v + (size_t)h * kv_head_stride;
        gemm_nt_q4_k(ln1_out, wv_h, bv_h, v_h, num_tokens, aligned_head_dim, aligned_embed_dim);
    }

    /* RoPE */
    rope_forward_qk(q,
                    k,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    num_tokens,
                    head_dim,
                    aligned_head_dim,
                    0);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash(q,
                                                   k,
                                                   v,
                                                   attn_out,
                                                   H,
                                                   H_kv,
                                                   num_tokens,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Output projection (flatten head-major to token-major) */
    const int K = H * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }
    const float *proj_in = attn_out;
    if (H > 1) {
        if (!proj_scratch) {
            return;
        }
        for (int t = 0; t < num_tokens; ++t) {
            float *dst = proj_scratch + (size_t)t * (size_t)aligned_embed_dim;
            for (int h = 0; h < H; ++h) {
                const float *src = attn_out + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;
                memcpy(dst + (size_t)h * (size_t)aligned_head_dim,
                       src,
                       (size_t)aligned_head_dim * sizeof(float));
            }
        }
        proj_in = proj_scratch;
    }
    gemm_nt_q4_k(proj_in, WO, BO, proj_tmp, num_tokens, aligned_embed_dim, K);

    /* Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/* ============================================================================
 * FORWARD PASS (PREFILL)
 * ============================================================================ */

static void qwen2_0_5b_decode_forward_prefill_impl(
    QWEN2_0_5B_DECODEModel *model,
    const int *tokens,
    int num_tokens
) {
    if (!model || !tokens || num_tokens <= 0) {
        return;
    }

    const int elem_bytes = QWEN2_0_5B_DECODE_DTYPE_BYTES;
    const int aligned_embed_dim = 1024;
    const int aligned_head_dim = 64;
    const int aligned_intermediate_dim = 4864;
    const int aligned_context_window = 131072;

    float *embed_out = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_HEADER.embedded_input);
    const void *embed_weight = (const void *)QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_HEADER.token_emb);
    embedding_forward_q4_k((const int32_t *)tokens,
                          num_tokens,
                          QWEN2_0_5B_DECODE_VOCAB_SIZE,
                          embed_weight,
                          NULL,
                          embed_out,
                          QWEN2_0_5B_DECODE_EMBED_DIM,
                          aligned_embed_dim,
                          num_tokens,
                          0);

    qwen2_0_5b_decode_layer_0_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[0].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[0].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_1_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[1].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[1].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_2_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[2].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[2].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_3_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[3].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[3].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_4_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[4].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[4].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_5_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[5].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[5].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_6_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[6].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[6].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_7_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[7].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[7].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_8_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[8].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[8].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_9_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[9].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[9].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_10_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[10].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[10].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_11_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[11].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[11].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_12_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[12].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[12].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_13_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[13].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[13].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_14_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[14].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[14].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_15_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[15].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[15].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_16_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[16].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[16].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_17_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[17].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[17].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_18_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[18].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[18].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_19_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[19].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[19].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_20_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[20].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[20].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_21_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[21].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[21].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_22_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[22].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[22].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_23_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[23].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[23].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    float *last_hidden = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[QWEN2_0_5B_DECODE_NUM_LAYERS - 1].output);
    float *final_ln_weight = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_FOOTER.final_ln_weight);
    float *final_out = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_FOOTER.final_output);
    rmsnorm_forward(last_hidden,
                   final_ln_weight,
                   final_out,
                   NULL,
                   num_tokens,
                   QWEN2_0_5B_DECODE_EMBED_DIM,
                   aligned_embed_dim,
                   1e-06f);

    float *logits = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_FOOTER.logits);
    const void *lm_head = (const void *)QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_FOOTER.lm_head_weight);
    const size_t q8_bytes = ck_dtype_row_bytes(CK_DT_Q8_K, (size_t)aligned_embed_dim);
    for (int t = 0; t < num_tokens; ++t) {
        uint8_t q8_buf[q8_bytes];
        const float *row = final_out + (size_t)t * (size_t)aligned_embed_dim;
        float *logits_row = logits + (size_t)t * (size_t)QWEN2_0_5B_DECODE_VOCAB_SIZE;
        quantize_row_q8_k(row, q8_buf, aligned_embed_dim);
        gemm_nt_q4_k_q8_k(q8_buf,
                          lm_head,
                          NULL,
                          logits_row,
                          1,
                          QWEN2_0_5B_DECODE_VOCAB_SIZE,
                          aligned_embed_dim);
    }
}

/* ============================================================================
 * EXPLICIT PER-LAYER DECODE FUNCTIONS (v6 unrolled)
 * ============================================================================ */

/*
 * Layer 0: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_0_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[0];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_HEADER.embedded_input);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 0) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WQ, NULL, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WK, NULL, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WV, NULL, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 1: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_1_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[1];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[0].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 1) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WQ, NULL, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WK, NULL, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WV, NULL, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 2: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_2_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[2];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[1].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 2) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WQ, NULL, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WK, NULL, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WV, NULL, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 3: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_3_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[3];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[2].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 3) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WQ, NULL, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WK, NULL, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WV, NULL, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 4: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_4_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[4];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[3].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 4) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WQ, NULL, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WK, NULL, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WV, NULL, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 5: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_5_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[5];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[4].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 5) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WQ, NULL, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WK, NULL, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WV, NULL, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 6: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_6_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[6];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[5].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 6) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WQ, NULL, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WK, NULL, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WV, NULL, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 7: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_7_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[7];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[6].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 7) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WQ, NULL, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WK, NULL, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WV, NULL, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 8: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_8_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[8];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[7].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 8) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WQ, NULL, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WK, NULL, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WV, NULL, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 9: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_9_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[9];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[8].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 9) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WQ, NULL, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WK, NULL, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WV, NULL, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 10: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_10_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[10];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[9].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 10) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WQ, NULL, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WK, NULL, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WV, NULL, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 11: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_11_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[11];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[10].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 11) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WQ, NULL, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WK, NULL, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WV, NULL, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 12: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_12_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[12];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[11].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 12) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WQ, NULL, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WK, NULL, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WV, NULL, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 13: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_13_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[13];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[12].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 13) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WQ, NULL, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WK, NULL, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WV, NULL, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 14: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_14_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[14];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[13].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 14) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WQ, NULL, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WK, NULL, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WV, NULL, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 15: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_15_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[15];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[14].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 15) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WQ, NULL, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WK, NULL, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WV, NULL, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 16: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_16_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[16];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[15].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 16) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WQ, NULL, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WK, NULL, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WV, NULL, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 17: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_17_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[17];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[16].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 17) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WQ, NULL, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WK, NULL, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WV, NULL, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 18: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_18_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[18];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[17].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 18) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WQ, NULL, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WK, NULL, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WV, NULL, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 19: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_19_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[19];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[18].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 19) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WQ, NULL, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WK, NULL, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WV, NULL, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 20: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_20_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[20];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[19].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 20) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WQ, NULL, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WK, NULL, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WV, NULL, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 21: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_21_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[21];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[20].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 21) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WQ, NULL, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WK, NULL, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WV, NULL, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 22: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_22_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[22];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[21].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 22) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WQ, NULL, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WK, NULL, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WV, NULL, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 23: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void qwen2_0_5b_decode_layer_23_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[23];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[22].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 23) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WQ, NULL, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WK, NULL, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln1_out, WV, NULL, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/* ============================================================================
 * DECODE TOKEN (calls each layer explicitly)
 * ============================================================================ */

static void qwen2_0_5b_decode_decode_token(
    QWEN2_0_5B_DECODEModel *model,
    const int *token,
    int token_index
) {
    if (!model || !token) return;

    const int aligned_embed_dim = 1024;
    const int aligned_head_dim = 64;
    const int aligned_intermediate_dim = 4864;
    const int aligned_context_window = 131072;

    if (token_index < 0 || token_index >= aligned_context_window) return;

    /* Embedding lookup */
    float *embed_out = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_HEADER.embedded_input);
    const void *embed_weight = (const void *)QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_HEADER.token_emb);
    /* Embedding: Q4_K -> embedding_forward_q4_k */
    embedding_forward_q4_k((const int32_t *)token,
                          1,
                          QWEN2_0_5B_DECODE_VOCAB_SIZE,
                          embed_weight,
                          NULL,
                          embed_out,
                          QWEN2_0_5B_DECODE_EMBED_DIM,
                          aligned_embed_dim,
                          1,
                          0);

    /* Process each layer explicitly */
    qwen2_0_5b_decode_layer_0_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_1_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_2_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_3_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_4_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_5_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_6_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_7_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_8_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_9_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_10_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_11_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_12_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_13_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_14_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_15_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_16_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_17_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_18_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_19_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_20_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_21_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_22_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_23_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);

    /* Final RMSNorm */
    float *last_hidden = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[23].output);
    float *final_ln_weight = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_FOOTER.final_ln_weight);
    float *final_out = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_FOOTER.final_output);
    rmsnorm_forward(last_hidden,
                    final_ln_weight,
                    final_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* LM head projection */
    float *logits = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_FOOTER.logits);
    const void *lm_head = (const void *)QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_FOOTER.lm_head_weight);
    /* LM head: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(final_out, lm_head, NULL, logits, 1, QWEN2_0_5B_DECODE_VOCAB_SIZE, aligned_embed_dim);
}

/* ============================================================================
 * PUBLIC API
 * ============================================================================ */

void qwen2_0_5b_decode_forward(
    QWEN2_0_5B_DECODEModel *model,
    const int *tokens,
    int num_tokens
) {
    if (!model || !tokens || num_tokens <= 0) return;
    qwen2_0_5b_decode_forward_prefill_impl(model, tokens, num_tokens);
}

void qwen2_0_5b_decode_decode(QWEN2_0_5B_DECODEModel *model, const int *token, int token_index) {
    qwen2_0_5b_decode_decode_token(model, token, token_index);
}
