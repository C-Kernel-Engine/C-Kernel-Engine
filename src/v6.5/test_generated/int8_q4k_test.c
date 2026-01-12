/**
 * @file int8_q4k_test.c
 * @brief AUTO-GENERATED: model Implementation (IR v6.5 - Explicit Unrolled)
 *
 * Generated: 2026-01-12T04:06:43.069656 UTC
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
 * DO NOT EDIT - Regenerate with build_ir_v6.5.py or codegen_v6.5.py
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

#if MODEL_DTYPE_BYTES != 4
#error "model: v6.5 codegen currently supports fp32 only. Use --dtype=fp32."
#endif

/* ============================================================================
 * LOCAL HELPERS (no orchestration dependency)
 * ============================================================================ */

static void model_residual_add_token_major(
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

int model_model_allocate(MODELModel *model) {
    size_t total = MODEL_TOTAL_BYTES;

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
    header->magic = MODEL_MAGIC;
    header->version = 5;
    header->total_bytes = MODEL_TOTAL_BYTES;
    header->weight_bytes = MODEL_WEIGHT_BYTES;
    header->activation_bytes = MODEL_ACTIVATION_BYTES;
    header->num_layers = MODEL_NUM_LAYERS;
    header->embed_dim = MODEL_EMBED_DIM;
    header->num_heads = MODEL_NUM_HEADS;
    header->vocab_size = MODEL_VOCAB_SIZE;
    header->max_seq_len = MODEL_MAX_SEQ_LEN;
    header->canary_count = MODEL_CANARY_COUNT;

    /* Initialize canary guards */
    for (int i = 0; i < MODEL_CANARY_COUNT; i++) {
        uint32_t *ptr = (uint32_t*)((char*)model->base + MODEL_CANARIES[i].offset);
        for (int j = 0; j < (MODEL_CANARY_SIZE / 4); j++) {
            ptr[j] = MODEL_CANARY_VALUE;
        }
    }

    return 0;
}

void model_model_free(MODELModel *model) {
    if (!model || !model->base) return;
#ifdef __linux__
    munmap(model->base, model->total_bytes);
#else
    free(model->base);
#endif
    model->base = NULL;
    model->total_bytes = 0;
}

int model_verify_canaries(MODELModel *model) {
    int errors = 0;
    uint32_t *ptr;

    for (int i = 0; i < MODEL_CANARY_COUNT; i++) {
        ptr = (uint32_t*)((char*)model->base + MODEL_CANARIES[i].offset);
        for (int j = 0; j < 4; j++) {
            if (ptr[j] != MODEL_CANARY_VALUE) {
                fprintf(stderr, "CANARY CORRUPTION: %s at offset 0x%lX\n",
                        MODEL_CANARIES[i].name,
                        MODEL_CANARIES[i].offset);
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

static int model_align_elems(int elems, int elem_bytes, int align_bytes) {
    int bytes = elems * elem_bytes;
    int aligned = (bytes + align_bytes - 1) / align_bytes * align_bytes;
    return aligned / elem_bytes;
}

/* ============================================================================
 * ROPE PRECOMPUTE
 * ============================================================================ */

void model_precompute_rope(MODELModel *model) {
    const int T = MODEL_MAX_SEQ_LEN;
    const int D = MODEL_HEAD_DIM / 2;
    const float theta = 1000000.0f;

    float *cos_ptr = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *sin_ptr = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

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
 * EXPLICIT PER-LAYER PREFILL FUNCTIONS (v6.5 unrolled)
 * ============================================================================ */

/*
 * Layer 0: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_0_prefill(
    MODELModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[0];

    float *input = MODEL_PTR(model, MODEL_HEADER.embedded_input);
    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *q = MODEL_PTR(model, L->q);
    float *k = MODEL_PTR(model, L->k);
    float *v = MODEL_PTR(model, L->v);
    float *attn_out = MODEL_PTR(model, L->attn_out);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *fc1_out = MODEL_PTR(model, L->fc1_out);
    float *swiglu_out = MODEL_PTR(model, L->swiglu_out);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    const void *WQ = (const void *)MODEL_PTR(model, L->wq);
    const void *WK = (const void *)MODEL_PTR(model, L->wk);
    const void *WV = (const void *)MODEL_PTR(model, L->wv);
    const void *WO = (const void *)MODEL_PTR(model, L->wo);
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
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
    rope_forward_qk_strided(q,
                            k,
                            rope_cos,
                            rope_sin,
                            H,
                            H_kv,
                            num_tokens,
                            head_dim,
                            aligned_head_dim,
                            0,
                            num_tokens,
                            aligned_context_window);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash_strided(q,
                                                           k,
                                                           v,
                                                           attn_out,
                                                           H,
                                                           H_kv,
                                                           num_tokens,
                                                           head_dim,
                                                           aligned_head_dim,
                                                           aligned_context_window);

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
    model_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 1: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_1_prefill(
    MODELModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[1];

    float *input = MODEL_PTR(model, MODEL_LAYERS[0].output);
    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *q = MODEL_PTR(model, L->q);
    float *k = MODEL_PTR(model, L->k);
    float *v = MODEL_PTR(model, L->v);
    float *attn_out = MODEL_PTR(model, L->attn_out);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *fc1_out = MODEL_PTR(model, L->fc1_out);
    float *swiglu_out = MODEL_PTR(model, L->swiglu_out);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    const void *WQ = (const void *)MODEL_PTR(model, L->wq);
    const void *WK = (const void *)MODEL_PTR(model, L->wk);
    const void *WV = (const void *)MODEL_PTR(model, L->wv);
    const void *WO = (const void *)MODEL_PTR(model, L->wo);
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
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
    rope_forward_qk_strided(q,
                            k,
                            rope_cos,
                            rope_sin,
                            H,
                            H_kv,
                            num_tokens,
                            head_dim,
                            aligned_head_dim,
                            0,
                            num_tokens,
                            aligned_context_window);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash_strided(q,
                                                           k,
                                                           v,
                                                           attn_out,
                                                           H,
                                                           H_kv,
                                                           num_tokens,
                                                           head_dim,
                                                           aligned_head_dim,
                                                           aligned_context_window);

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
    model_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 2: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_2_prefill(
    MODELModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[2];

    float *input = MODEL_PTR(model, MODEL_LAYERS[1].output);
    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *q = MODEL_PTR(model, L->q);
    float *k = MODEL_PTR(model, L->k);
    float *v = MODEL_PTR(model, L->v);
    float *attn_out = MODEL_PTR(model, L->attn_out);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *fc1_out = MODEL_PTR(model, L->fc1_out);
    float *swiglu_out = MODEL_PTR(model, L->swiglu_out);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    const void *WQ = (const void *)MODEL_PTR(model, L->wq);
    const void *WK = (const void *)MODEL_PTR(model, L->wk);
    const void *WV = (const void *)MODEL_PTR(model, L->wv);
    const void *WO = (const void *)MODEL_PTR(model, L->wo);
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
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
    rope_forward_qk_strided(q,
                            k,
                            rope_cos,
                            rope_sin,
                            H,
                            H_kv,
                            num_tokens,
                            head_dim,
                            aligned_head_dim,
                            0,
                            num_tokens,
                            aligned_context_window);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash_strided(q,
                                                           k,
                                                           v,
                                                           attn_out,
                                                           H,
                                                           H_kv,
                                                           num_tokens,
                                                           head_dim,
                                                           aligned_head_dim,
                                                           aligned_context_window);

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
    model_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 3: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_3_prefill(
    MODELModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[3];

    float *input = MODEL_PTR(model, MODEL_LAYERS[2].output);
    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *q = MODEL_PTR(model, L->q);
    float *k = MODEL_PTR(model, L->k);
    float *v = MODEL_PTR(model, L->v);
    float *attn_out = MODEL_PTR(model, L->attn_out);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *fc1_out = MODEL_PTR(model, L->fc1_out);
    float *swiglu_out = MODEL_PTR(model, L->swiglu_out);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    const void *WQ = (const void *)MODEL_PTR(model, L->wq);
    const void *WK = (const void *)MODEL_PTR(model, L->wk);
    const void *WV = (const void *)MODEL_PTR(model, L->wv);
    const void *WO = (const void *)MODEL_PTR(model, L->wo);
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
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
    rope_forward_qk_strided(q,
                            k,
                            rope_cos,
                            rope_sin,
                            H,
                            H_kv,
                            num_tokens,
                            head_dim,
                            aligned_head_dim,
                            0,
                            num_tokens,
                            aligned_context_window);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash_strided(q,
                                                           k,
                                                           v,
                                                           attn_out,
                                                           H,
                                                           H_kv,
                                                           num_tokens,
                                                           head_dim,
                                                           aligned_head_dim,
                                                           aligned_context_window);

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
    model_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 4: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_4_prefill(
    MODELModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[4];

    float *input = MODEL_PTR(model, MODEL_LAYERS[3].output);
    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *q = MODEL_PTR(model, L->q);
    float *k = MODEL_PTR(model, L->k);
    float *v = MODEL_PTR(model, L->v);
    float *attn_out = MODEL_PTR(model, L->attn_out);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *fc1_out = MODEL_PTR(model, L->fc1_out);
    float *swiglu_out = MODEL_PTR(model, L->swiglu_out);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    const void *WQ = (const void *)MODEL_PTR(model, L->wq);
    const void *WK = (const void *)MODEL_PTR(model, L->wk);
    const void *WV = (const void *)MODEL_PTR(model, L->wv);
    const void *WO = (const void *)MODEL_PTR(model, L->wo);
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
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
    rope_forward_qk_strided(q,
                            k,
                            rope_cos,
                            rope_sin,
                            H,
                            H_kv,
                            num_tokens,
                            head_dim,
                            aligned_head_dim,
                            0,
                            num_tokens,
                            aligned_context_window);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash_strided(q,
                                                           k,
                                                           v,
                                                           attn_out,
                                                           H,
                                                           H_kv,
                                                           num_tokens,
                                                           head_dim,
                                                           aligned_head_dim,
                                                           aligned_context_window);

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
    model_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 5: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_5_prefill(
    MODELModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[5];

    float *input = MODEL_PTR(model, MODEL_LAYERS[4].output);
    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *q = MODEL_PTR(model, L->q);
    float *k = MODEL_PTR(model, L->k);
    float *v = MODEL_PTR(model, L->v);
    float *attn_out = MODEL_PTR(model, L->attn_out);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *fc1_out = MODEL_PTR(model, L->fc1_out);
    float *swiglu_out = MODEL_PTR(model, L->swiglu_out);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    const void *WQ = (const void *)MODEL_PTR(model, L->wq);
    const void *WK = (const void *)MODEL_PTR(model, L->wk);
    const void *WV = (const void *)MODEL_PTR(model, L->wv);
    const void *WO = (const void *)MODEL_PTR(model, L->wo);
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
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
    rope_forward_qk_strided(q,
                            k,
                            rope_cos,
                            rope_sin,
                            H,
                            H_kv,
                            num_tokens,
                            head_dim,
                            aligned_head_dim,
                            0,
                            num_tokens,
                            aligned_context_window);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash_strided(q,
                                                           k,
                                                           v,
                                                           attn_out,
                                                           H,
                                                           H_kv,
                                                           num_tokens,
                                                           head_dim,
                                                           aligned_head_dim,
                                                           aligned_context_window);

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
    model_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 6: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_6_prefill(
    MODELModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[6];

    float *input = MODEL_PTR(model, MODEL_LAYERS[5].output);
    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *q = MODEL_PTR(model, L->q);
    float *k = MODEL_PTR(model, L->k);
    float *v = MODEL_PTR(model, L->v);
    float *attn_out = MODEL_PTR(model, L->attn_out);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *fc1_out = MODEL_PTR(model, L->fc1_out);
    float *swiglu_out = MODEL_PTR(model, L->swiglu_out);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    const void *WQ = (const void *)MODEL_PTR(model, L->wq);
    const void *WK = (const void *)MODEL_PTR(model, L->wk);
    const void *WV = (const void *)MODEL_PTR(model, L->wv);
    const void *WO = (const void *)MODEL_PTR(model, L->wo);
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
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
    rope_forward_qk_strided(q,
                            k,
                            rope_cos,
                            rope_sin,
                            H,
                            H_kv,
                            num_tokens,
                            head_dim,
                            aligned_head_dim,
                            0,
                            num_tokens,
                            aligned_context_window);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash_strided(q,
                                                           k,
                                                           v,
                                                           attn_out,
                                                           H,
                                                           H_kv,
                                                           num_tokens,
                                                           head_dim,
                                                           aligned_head_dim,
                                                           aligned_context_window);

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
    model_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 7: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_7_prefill(
    MODELModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[7];

    float *input = MODEL_PTR(model, MODEL_LAYERS[6].output);
    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *q = MODEL_PTR(model, L->q);
    float *k = MODEL_PTR(model, L->k);
    float *v = MODEL_PTR(model, L->v);
    float *attn_out = MODEL_PTR(model, L->attn_out);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *fc1_out = MODEL_PTR(model, L->fc1_out);
    float *swiglu_out = MODEL_PTR(model, L->swiglu_out);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    const void *WQ = (const void *)MODEL_PTR(model, L->wq);
    const void *WK = (const void *)MODEL_PTR(model, L->wk);
    const void *WV = (const void *)MODEL_PTR(model, L->wv);
    const void *WO = (const void *)MODEL_PTR(model, L->wo);
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
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
    rope_forward_qk_strided(q,
                            k,
                            rope_cos,
                            rope_sin,
                            H,
                            H_kv,
                            num_tokens,
                            head_dim,
                            aligned_head_dim,
                            0,
                            num_tokens,
                            aligned_context_window);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash_strided(q,
                                                           k,
                                                           v,
                                                           attn_out,
                                                           H,
                                                           H_kv,
                                                           num_tokens,
                                                           head_dim,
                                                           aligned_head_dim,
                                                           aligned_context_window);

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
    model_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 8: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_8_prefill(
    MODELModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[8];

    float *input = MODEL_PTR(model, MODEL_LAYERS[7].output);
    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *q = MODEL_PTR(model, L->q);
    float *k = MODEL_PTR(model, L->k);
    float *v = MODEL_PTR(model, L->v);
    float *attn_out = MODEL_PTR(model, L->attn_out);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *fc1_out = MODEL_PTR(model, L->fc1_out);
    float *swiglu_out = MODEL_PTR(model, L->swiglu_out);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    const void *WQ = (const void *)MODEL_PTR(model, L->wq);
    const void *WK = (const void *)MODEL_PTR(model, L->wk);
    const void *WV = (const void *)MODEL_PTR(model, L->wv);
    const void *WO = (const void *)MODEL_PTR(model, L->wo);
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
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
    rope_forward_qk_strided(q,
                            k,
                            rope_cos,
                            rope_sin,
                            H,
                            H_kv,
                            num_tokens,
                            head_dim,
                            aligned_head_dim,
                            0,
                            num_tokens,
                            aligned_context_window);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash_strided(q,
                                                           k,
                                                           v,
                                                           attn_out,
                                                           H,
                                                           H_kv,
                                                           num_tokens,
                                                           head_dim,
                                                           aligned_head_dim,
                                                           aligned_context_window);

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
    model_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 9: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_9_prefill(
    MODELModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[9];

    float *input = MODEL_PTR(model, MODEL_LAYERS[8].output);
    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *q = MODEL_PTR(model, L->q);
    float *k = MODEL_PTR(model, L->k);
    float *v = MODEL_PTR(model, L->v);
    float *attn_out = MODEL_PTR(model, L->attn_out);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *fc1_out = MODEL_PTR(model, L->fc1_out);
    float *swiglu_out = MODEL_PTR(model, L->swiglu_out);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    const void *WQ = (const void *)MODEL_PTR(model, L->wq);
    const void *WK = (const void *)MODEL_PTR(model, L->wk);
    const void *WV = (const void *)MODEL_PTR(model, L->wv);
    const void *WO = (const void *)MODEL_PTR(model, L->wo);
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
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
    rope_forward_qk_strided(q,
                            k,
                            rope_cos,
                            rope_sin,
                            H,
                            H_kv,
                            num_tokens,
                            head_dim,
                            aligned_head_dim,
                            0,
                            num_tokens,
                            aligned_context_window);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash_strided(q,
                                                           k,
                                                           v,
                                                           attn_out,
                                                           H,
                                                           H_kv,
                                                           num_tokens,
                                                           head_dim,
                                                           aligned_head_dim,
                                                           aligned_context_window);

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
    model_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 10: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_10_prefill(
    MODELModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[10];

    float *input = MODEL_PTR(model, MODEL_LAYERS[9].output);
    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *q = MODEL_PTR(model, L->q);
    float *k = MODEL_PTR(model, L->k);
    float *v = MODEL_PTR(model, L->v);
    float *attn_out = MODEL_PTR(model, L->attn_out);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *fc1_out = MODEL_PTR(model, L->fc1_out);
    float *swiglu_out = MODEL_PTR(model, L->swiglu_out);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    const void *WQ = (const void *)MODEL_PTR(model, L->wq);
    const void *WK = (const void *)MODEL_PTR(model, L->wk);
    const void *WV = (const void *)MODEL_PTR(model, L->wv);
    const void *WO = (const void *)MODEL_PTR(model, L->wo);
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
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
    rope_forward_qk_strided(q,
                            k,
                            rope_cos,
                            rope_sin,
                            H,
                            H_kv,
                            num_tokens,
                            head_dim,
                            aligned_head_dim,
                            0,
                            num_tokens,
                            aligned_context_window);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash_strided(q,
                                                           k,
                                                           v,
                                                           attn_out,
                                                           H,
                                                           H_kv,
                                                           num_tokens,
                                                           head_dim,
                                                           aligned_head_dim,
                                                           aligned_context_window);

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
    model_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 11: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_11_prefill(
    MODELModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[11];

    float *input = MODEL_PTR(model, MODEL_LAYERS[10].output);
    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *q = MODEL_PTR(model, L->q);
    float *k = MODEL_PTR(model, L->k);
    float *v = MODEL_PTR(model, L->v);
    float *attn_out = MODEL_PTR(model, L->attn_out);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *fc1_out = MODEL_PTR(model, L->fc1_out);
    float *swiglu_out = MODEL_PTR(model, L->swiglu_out);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    const void *WQ = (const void *)MODEL_PTR(model, L->wq);
    const void *WK = (const void *)MODEL_PTR(model, L->wk);
    const void *WV = (const void *)MODEL_PTR(model, L->wv);
    const void *WO = (const void *)MODEL_PTR(model, L->wo);
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
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
    rope_forward_qk_strided(q,
                            k,
                            rope_cos,
                            rope_sin,
                            H,
                            H_kv,
                            num_tokens,
                            head_dim,
                            aligned_head_dim,
                            0,
                            num_tokens,
                            aligned_context_window);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash_strided(q,
                                                           k,
                                                           v,
                                                           attn_out,
                                                           H,
                                                           H_kv,
                                                           num_tokens,
                                                           head_dim,
                                                           aligned_head_dim,
                                                           aligned_context_window);

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
    model_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 12: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_12_prefill(
    MODELModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[12];

    float *input = MODEL_PTR(model, MODEL_LAYERS[11].output);
    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *q = MODEL_PTR(model, L->q);
    float *k = MODEL_PTR(model, L->k);
    float *v = MODEL_PTR(model, L->v);
    float *attn_out = MODEL_PTR(model, L->attn_out);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *fc1_out = MODEL_PTR(model, L->fc1_out);
    float *swiglu_out = MODEL_PTR(model, L->swiglu_out);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    const void *WQ = (const void *)MODEL_PTR(model, L->wq);
    const void *WK = (const void *)MODEL_PTR(model, L->wk);
    const void *WV = (const void *)MODEL_PTR(model, L->wv);
    const void *WO = (const void *)MODEL_PTR(model, L->wo);
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
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
    rope_forward_qk_strided(q,
                            k,
                            rope_cos,
                            rope_sin,
                            H,
                            H_kv,
                            num_tokens,
                            head_dim,
                            aligned_head_dim,
                            0,
                            num_tokens,
                            aligned_context_window);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash_strided(q,
                                                           k,
                                                           v,
                                                           attn_out,
                                                           H,
                                                           H_kv,
                                                           num_tokens,
                                                           head_dim,
                                                           aligned_head_dim,
                                                           aligned_context_window);

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
    model_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 13: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_13_prefill(
    MODELModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[13];

    float *input = MODEL_PTR(model, MODEL_LAYERS[12].output);
    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *q = MODEL_PTR(model, L->q);
    float *k = MODEL_PTR(model, L->k);
    float *v = MODEL_PTR(model, L->v);
    float *attn_out = MODEL_PTR(model, L->attn_out);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *fc1_out = MODEL_PTR(model, L->fc1_out);
    float *swiglu_out = MODEL_PTR(model, L->swiglu_out);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    const void *WQ = (const void *)MODEL_PTR(model, L->wq);
    const void *WK = (const void *)MODEL_PTR(model, L->wk);
    const void *WV = (const void *)MODEL_PTR(model, L->wv);
    const void *WO = (const void *)MODEL_PTR(model, L->wo);
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
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
    rope_forward_qk_strided(q,
                            k,
                            rope_cos,
                            rope_sin,
                            H,
                            H_kv,
                            num_tokens,
                            head_dim,
                            aligned_head_dim,
                            0,
                            num_tokens,
                            aligned_context_window);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash_strided(q,
                                                           k,
                                                           v,
                                                           attn_out,
                                                           H,
                                                           H_kv,
                                                           num_tokens,
                                                           head_dim,
                                                           aligned_head_dim,
                                                           aligned_context_window);

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
    model_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 14: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_14_prefill(
    MODELModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[14];

    float *input = MODEL_PTR(model, MODEL_LAYERS[13].output);
    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *q = MODEL_PTR(model, L->q);
    float *k = MODEL_PTR(model, L->k);
    float *v = MODEL_PTR(model, L->v);
    float *attn_out = MODEL_PTR(model, L->attn_out);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *fc1_out = MODEL_PTR(model, L->fc1_out);
    float *swiglu_out = MODEL_PTR(model, L->swiglu_out);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    const void *WQ = (const void *)MODEL_PTR(model, L->wq);
    const void *WK = (const void *)MODEL_PTR(model, L->wk);
    const void *WV = (const void *)MODEL_PTR(model, L->wv);
    const void *WO = (const void *)MODEL_PTR(model, L->wo);
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
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
    rope_forward_qk_strided(q,
                            k,
                            rope_cos,
                            rope_sin,
                            H,
                            H_kv,
                            num_tokens,
                            head_dim,
                            aligned_head_dim,
                            0,
                            num_tokens,
                            aligned_context_window);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash_strided(q,
                                                           k,
                                                           v,
                                                           attn_out,
                                                           H,
                                                           H_kv,
                                                           num_tokens,
                                                           head_dim,
                                                           aligned_head_dim,
                                                           aligned_context_window);

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
    model_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 15: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_15_prefill(
    MODELModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[15];

    float *input = MODEL_PTR(model, MODEL_LAYERS[14].output);
    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *q = MODEL_PTR(model, L->q);
    float *k = MODEL_PTR(model, L->k);
    float *v = MODEL_PTR(model, L->v);
    float *attn_out = MODEL_PTR(model, L->attn_out);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *fc1_out = MODEL_PTR(model, L->fc1_out);
    float *swiglu_out = MODEL_PTR(model, L->swiglu_out);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    const void *WQ = (const void *)MODEL_PTR(model, L->wq);
    const void *WK = (const void *)MODEL_PTR(model, L->wk);
    const void *WV = (const void *)MODEL_PTR(model, L->wv);
    const void *WO = (const void *)MODEL_PTR(model, L->wo);
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
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
    rope_forward_qk_strided(q,
                            k,
                            rope_cos,
                            rope_sin,
                            H,
                            H_kv,
                            num_tokens,
                            head_dim,
                            aligned_head_dim,
                            0,
                            num_tokens,
                            aligned_context_window);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash_strided(q,
                                                           k,
                                                           v,
                                                           attn_out,
                                                           H,
                                                           H_kv,
                                                           num_tokens,
                                                           head_dim,
                                                           aligned_head_dim,
                                                           aligned_context_window);

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
    model_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 16: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_16_prefill(
    MODELModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[16];

    float *input = MODEL_PTR(model, MODEL_LAYERS[15].output);
    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *q = MODEL_PTR(model, L->q);
    float *k = MODEL_PTR(model, L->k);
    float *v = MODEL_PTR(model, L->v);
    float *attn_out = MODEL_PTR(model, L->attn_out);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *fc1_out = MODEL_PTR(model, L->fc1_out);
    float *swiglu_out = MODEL_PTR(model, L->swiglu_out);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    const void *WQ = (const void *)MODEL_PTR(model, L->wq);
    const void *WK = (const void *)MODEL_PTR(model, L->wk);
    const void *WV = (const void *)MODEL_PTR(model, L->wv);
    const void *WO = (const void *)MODEL_PTR(model, L->wo);
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
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
    rope_forward_qk_strided(q,
                            k,
                            rope_cos,
                            rope_sin,
                            H,
                            H_kv,
                            num_tokens,
                            head_dim,
                            aligned_head_dim,
                            0,
                            num_tokens,
                            aligned_context_window);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash_strided(q,
                                                           k,
                                                           v,
                                                           attn_out,
                                                           H,
                                                           H_kv,
                                                           num_tokens,
                                                           head_dim,
                                                           aligned_head_dim,
                                                           aligned_context_window);

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
    model_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 17: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_17_prefill(
    MODELModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[17];

    float *input = MODEL_PTR(model, MODEL_LAYERS[16].output);
    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *q = MODEL_PTR(model, L->q);
    float *k = MODEL_PTR(model, L->k);
    float *v = MODEL_PTR(model, L->v);
    float *attn_out = MODEL_PTR(model, L->attn_out);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *fc1_out = MODEL_PTR(model, L->fc1_out);
    float *swiglu_out = MODEL_PTR(model, L->swiglu_out);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    const void *WQ = (const void *)MODEL_PTR(model, L->wq);
    const void *WK = (const void *)MODEL_PTR(model, L->wk);
    const void *WV = (const void *)MODEL_PTR(model, L->wv);
    const void *WO = (const void *)MODEL_PTR(model, L->wo);
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
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
    rope_forward_qk_strided(q,
                            k,
                            rope_cos,
                            rope_sin,
                            H,
                            H_kv,
                            num_tokens,
                            head_dim,
                            aligned_head_dim,
                            0,
                            num_tokens,
                            aligned_context_window);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash_strided(q,
                                                           k,
                                                           v,
                                                           attn_out,
                                                           H,
                                                           H_kv,
                                                           num_tokens,
                                                           head_dim,
                                                           aligned_head_dim,
                                                           aligned_context_window);

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
    model_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 18: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_18_prefill(
    MODELModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[18];

    float *input = MODEL_PTR(model, MODEL_LAYERS[17].output);
    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *q = MODEL_PTR(model, L->q);
    float *k = MODEL_PTR(model, L->k);
    float *v = MODEL_PTR(model, L->v);
    float *attn_out = MODEL_PTR(model, L->attn_out);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *fc1_out = MODEL_PTR(model, L->fc1_out);
    float *swiglu_out = MODEL_PTR(model, L->swiglu_out);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    const void *WQ = (const void *)MODEL_PTR(model, L->wq);
    const void *WK = (const void *)MODEL_PTR(model, L->wk);
    const void *WV = (const void *)MODEL_PTR(model, L->wv);
    const void *WO = (const void *)MODEL_PTR(model, L->wo);
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
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
    rope_forward_qk_strided(q,
                            k,
                            rope_cos,
                            rope_sin,
                            H,
                            H_kv,
                            num_tokens,
                            head_dim,
                            aligned_head_dim,
                            0,
                            num_tokens,
                            aligned_context_window);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash_strided(q,
                                                           k,
                                                           v,
                                                           attn_out,
                                                           H,
                                                           H_kv,
                                                           num_tokens,
                                                           head_dim,
                                                           aligned_head_dim,
                                                           aligned_context_window);

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
    model_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 19: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_19_prefill(
    MODELModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[19];

    float *input = MODEL_PTR(model, MODEL_LAYERS[18].output);
    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *q = MODEL_PTR(model, L->q);
    float *k = MODEL_PTR(model, L->k);
    float *v = MODEL_PTR(model, L->v);
    float *attn_out = MODEL_PTR(model, L->attn_out);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *fc1_out = MODEL_PTR(model, L->fc1_out);
    float *swiglu_out = MODEL_PTR(model, L->swiglu_out);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    const void *WQ = (const void *)MODEL_PTR(model, L->wq);
    const void *WK = (const void *)MODEL_PTR(model, L->wk);
    const void *WV = (const void *)MODEL_PTR(model, L->wv);
    const void *WO = (const void *)MODEL_PTR(model, L->wo);
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
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
    rope_forward_qk_strided(q,
                            k,
                            rope_cos,
                            rope_sin,
                            H,
                            H_kv,
                            num_tokens,
                            head_dim,
                            aligned_head_dim,
                            0,
                            num_tokens,
                            aligned_context_window);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash_strided(q,
                                                           k,
                                                           v,
                                                           attn_out,
                                                           H,
                                                           H_kv,
                                                           num_tokens,
                                                           head_dim,
                                                           aligned_head_dim,
                                                           aligned_context_window);

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
    model_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 20: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_20_prefill(
    MODELModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[20];

    float *input = MODEL_PTR(model, MODEL_LAYERS[19].output);
    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *q = MODEL_PTR(model, L->q);
    float *k = MODEL_PTR(model, L->k);
    float *v = MODEL_PTR(model, L->v);
    float *attn_out = MODEL_PTR(model, L->attn_out);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *fc1_out = MODEL_PTR(model, L->fc1_out);
    float *swiglu_out = MODEL_PTR(model, L->swiglu_out);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    const void *WQ = (const void *)MODEL_PTR(model, L->wq);
    const void *WK = (const void *)MODEL_PTR(model, L->wk);
    const void *WV = (const void *)MODEL_PTR(model, L->wv);
    const void *WO = (const void *)MODEL_PTR(model, L->wo);
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
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
    rope_forward_qk_strided(q,
                            k,
                            rope_cos,
                            rope_sin,
                            H,
                            H_kv,
                            num_tokens,
                            head_dim,
                            aligned_head_dim,
                            0,
                            num_tokens,
                            aligned_context_window);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash_strided(q,
                                                           k,
                                                           v,
                                                           attn_out,
                                                           H,
                                                           H_kv,
                                                           num_tokens,
                                                           head_dim,
                                                           aligned_head_dim,
                                                           aligned_context_window);

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
    model_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 21: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_21_prefill(
    MODELModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[21];

    float *input = MODEL_PTR(model, MODEL_LAYERS[20].output);
    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *q = MODEL_PTR(model, L->q);
    float *k = MODEL_PTR(model, L->k);
    float *v = MODEL_PTR(model, L->v);
    float *attn_out = MODEL_PTR(model, L->attn_out);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *fc1_out = MODEL_PTR(model, L->fc1_out);
    float *swiglu_out = MODEL_PTR(model, L->swiglu_out);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    const void *WQ = (const void *)MODEL_PTR(model, L->wq);
    const void *WK = (const void *)MODEL_PTR(model, L->wk);
    const void *WV = (const void *)MODEL_PTR(model, L->wv);
    const void *WO = (const void *)MODEL_PTR(model, L->wo);
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
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
    rope_forward_qk_strided(q,
                            k,
                            rope_cos,
                            rope_sin,
                            H,
                            H_kv,
                            num_tokens,
                            head_dim,
                            aligned_head_dim,
                            0,
                            num_tokens,
                            aligned_context_window);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash_strided(q,
                                                           k,
                                                           v,
                                                           attn_out,
                                                           H,
                                                           H_kv,
                                                           num_tokens,
                                                           head_dim,
                                                           aligned_head_dim,
                                                           aligned_context_window);

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
    model_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 22: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_22_prefill(
    MODELModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[22];

    float *input = MODEL_PTR(model, MODEL_LAYERS[21].output);
    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *q = MODEL_PTR(model, L->q);
    float *k = MODEL_PTR(model, L->k);
    float *v = MODEL_PTR(model, L->v);
    float *attn_out = MODEL_PTR(model, L->attn_out);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *fc1_out = MODEL_PTR(model, L->fc1_out);
    float *swiglu_out = MODEL_PTR(model, L->swiglu_out);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    const void *WQ = (const void *)MODEL_PTR(model, L->wq);
    const void *WK = (const void *)MODEL_PTR(model, L->wk);
    const void *WV = (const void *)MODEL_PTR(model, L->wv);
    const void *WO = (const void *)MODEL_PTR(model, L->wo);
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
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
    rope_forward_qk_strided(q,
                            k,
                            rope_cos,
                            rope_sin,
                            H,
                            H_kv,
                            num_tokens,
                            head_dim,
                            aligned_head_dim,
                            0,
                            num_tokens,
                            aligned_context_window);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash_strided(q,
                                                           k,
                                                           v,
                                                           attn_out,
                                                           H,
                                                           H_kv,
                                                           num_tokens,
                                                           head_dim,
                                                           aligned_head_dim,
                                                           aligned_context_window);

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
    model_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/*
 * Layer 23: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_23_prefill(
    MODELModel *model,
    int num_tokens,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[23];

    float *input = MODEL_PTR(model, MODEL_LAYERS[22].output);
    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *q = MODEL_PTR(model, L->q);
    float *k = MODEL_PTR(model, L->k);
    float *v = MODEL_PTR(model, L->v);
    float *attn_out = MODEL_PTR(model, L->attn_out);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *fc1_out = MODEL_PTR(model, L->fc1_out);
    float *swiglu_out = MODEL_PTR(model, L->swiglu_out);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    const void *WQ = (const void *)MODEL_PTR(model, L->wq);
    const void *WK = (const void *)MODEL_PTR(model, L->wk);
    const void *WV = (const void *)MODEL_PTR(model, L->wv);
    const void *WO = (const void *)MODEL_PTR(model, L->wo);
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);
    const float *BQ = NULL;
    const float *BK = NULL;
    const float *BV = NULL;
    const float *BO = NULL;
    const float *B1 = NULL;
    const float *B2 = NULL;

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
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
    rope_forward_qk_strided(q,
                            k,
                            rope_cos,
                            rope_sin,
                            H,
                            H_kv,
                            num_tokens,
                            head_dim,
                            aligned_head_dim,
                            0,
                            num_tokens,
                            aligned_context_window);

    /* Attention (prefill, causal) */
    attention_forward_causal_head_major_gqa_flash_strided(q,
                                                           k,
                                                           v,
                                                           attn_out,
                                                           H,
                                                           H_kv,
                                                           num_tokens,
                                                           head_dim,
                                                           aligned_head_dim,
                                                           aligned_context_window);

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
    model_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);

    /* RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    num_tokens,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* MLP (SwiGLU) */
    gemm_nt_q4_k(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);
    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);
    gemm_nt_q4_k(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);

    /* Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);
}

/* ============================================================================
 * FORWARD PASS (PREFILL)
 * ============================================================================ */

static void model_forward_prefill_impl(
    MODELModel *model,
    const int *tokens,
    int num_tokens
) {
    if (!model || !tokens || num_tokens <= 0) {
        return;
    }

    const int elem_bytes = MODEL_DTYPE_BYTES;
    const int aligned_embed_dim = 1024;
    const int aligned_head_dim = 64;
    const int aligned_intermediate_dim = 4864;
    const int aligned_context_window = 131072;

    float *embed_out = MODEL_PTR(model, MODEL_HEADER.embedded_input);
    const void *embed_weight = (const void *)MODEL_PTR(model, MODEL_HEADER.token_emb);
    embedding_forward_q4_k((const int32_t *)tokens,
                          num_tokens,
                          MODEL_VOCAB_SIZE,
                          embed_weight,
                          NULL,
                          embed_out,
                          MODEL_EMBED_DIM,
                          aligned_embed_dim,
                          num_tokens,
                          0);

    model_layer_0_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    model_layer_1_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    model_layer_2_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    model_layer_3_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    model_layer_4_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    model_layer_5_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    model_layer_6_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    model_layer_7_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    model_layer_8_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    model_layer_9_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    model_layer_10_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    model_layer_11_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    model_layer_12_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    model_layer_13_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    model_layer_14_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    model_layer_15_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    model_layer_16_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    model_layer_17_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    model_layer_18_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    model_layer_19_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    model_layer_20_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    model_layer_21_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    model_layer_22_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    model_layer_23_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    float *last_hidden = MODEL_PTR(model, MODEL_LAYERS[MODEL_NUM_LAYERS - 1].output);
    float *final_ln_weight = MODEL_PTR(model, MODEL_FOOTER.final_ln_weight);
    float *final_out = MODEL_PTR(model, MODEL_FOOTER.final_output);
    rmsnorm_forward(last_hidden,
                   final_ln_weight,
                   final_out,
                   NULL,
                   num_tokens,
                   MODEL_EMBED_DIM,
                   aligned_embed_dim,
                   1e-06f);

    float *logits = MODEL_PTR(model, MODEL_FOOTER.logits);
    const void *lm_head = (const void *)MODEL_PTR(model, MODEL_FOOTER.lm_head_weight);
    const size_t q8_bytes = ck_dtype_row_bytes(CK_DT_Q8_K, (size_t)aligned_embed_dim);
    for (int t = 0; t < num_tokens; ++t) {
        uint8_t q8_buf[q8_bytes];
        const float *row = final_out + (size_t)t * (size_t)aligned_embed_dim;
        float *logits_row = logits + (size_t)t * (size_t)MODEL_VOCAB_SIZE;
        quantize_row_q8_k(row, q8_buf, aligned_embed_dim);
        gemm_nt_q4_k_q8_k(q8_buf,
                          lm_head,
                          NULL,
                          logits_row,
                          1,
                          MODEL_VOCAB_SIZE,
                          aligned_embed_dim);
    }
}

/* ============================================================================
 * EXPLICIT PER-LAYER DECODE FUNCTIONS (v6.5 unrolled)
 * ============================================================================ */

/*
 * Layer 0: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_0_decode(
    MODELModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[0];

    float *input = MODEL_PTR(model, MODEL_HEADER.embedded_input);

    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *k_cache = MODEL_PTR(model, L->k);
    float *v_cache = MODEL_PTR(model, L->v);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    /* Weights (explicit types for layer 0) */
    const void *WQ = (const void *)MODEL_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)MODEL_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)MODEL_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)MODEL_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;

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
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* Step 2: QKV projection */
    /* Q projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln1_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln1_q8[ln1_q8_bytes];
    quantize_row_q8_k(ln1_out, ln1_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(q_token, WQ, ln1_q8, H * head_dim, aligned_embed_dim);
    if (aligned_head_dim > head_dim) {
        for (int h = 0; h < H; ++h) {
            float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                q_head[d] = 0.0f;
            }
        }
    }

    /* K projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wk_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wk_head_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(k_head, wk_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            k_head[d] = 0.0f;
        }
    }

    /* V projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wv_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wv_head_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        float *v_head = v_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(v_head, wv_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            v_head[d] = 0.0f;
        }
    }

    /* Step 3: RoPE */
    rope_forward(q_token,
                 rope_cos,
                 rope_sin,
                 H,
                 1,
                 head_dim,
                 aligned_head_dim,
                 token_index);
    for (int h = 0; h < H_kv; ++h) {
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        rope_forward(k_head,
                     rope_cos,
                     rope_sin,
                     1,
                     1,
                     head_dim,
                     aligned_head_dim,
                     token_index);
    }

    /* Step 4: KV cache write (direct-to-cache) */

    /* Step 5: Attention (decode, flash) */
    attention_forward_decode_head_major_gqa_flash(q_token,
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
    /* WO projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t attn_q8_bytes = ((((H * head_dim) + 255) / 256) * 292);
    uint8_t attn_q8[attn_q8_bytes];
    quantize_row_q8_k(attn_token, attn_q8, H * head_dim);
    gemv_q4_k_q8_k(proj_tmp, WO, attn_q8, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    model_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln2_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln2_q8[ln2_q8_bytes];
    quantize_row_q8_k(ln2_out, ln2_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(fc1_out, W1, ln2_q8, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t swiglu_q8_bytes = ((((aligned_intermediate_dim) + 255) / 256) * 292);
    uint8_t swiglu_q8[swiglu_q8_bytes];
    quantize_row_q8_k(swiglu_out, swiglu_q8, aligned_intermediate_dim);
    gemv_q4_k_q8_k(mlp_out, W2, swiglu_q8, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 1: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_1_decode(
    MODELModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[1];

    float *input = MODEL_PTR(model, MODEL_LAYERS[0].output);

    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *k_cache = MODEL_PTR(model, L->k);
    float *v_cache = MODEL_PTR(model, L->v);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    /* Weights (explicit types for layer 1) */
    const void *WQ = (const void *)MODEL_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)MODEL_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)MODEL_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)MODEL_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;

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
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* Step 2: QKV projection */
    /* Q projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln1_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln1_q8[ln1_q8_bytes];
    quantize_row_q8_k(ln1_out, ln1_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(q_token, WQ, ln1_q8, H * head_dim, aligned_embed_dim);
    if (aligned_head_dim > head_dim) {
        for (int h = 0; h < H; ++h) {
            float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                q_head[d] = 0.0f;
            }
        }
    }

    /* K projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wk_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wk_head_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(k_head, wk_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            k_head[d] = 0.0f;
        }
    }

    /* V projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wv_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wv_head_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        float *v_head = v_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(v_head, wv_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            v_head[d] = 0.0f;
        }
    }

    /* Step 3: RoPE */
    rope_forward(q_token,
                 rope_cos,
                 rope_sin,
                 H,
                 1,
                 head_dim,
                 aligned_head_dim,
                 token_index);
    for (int h = 0; h < H_kv; ++h) {
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        rope_forward(k_head,
                     rope_cos,
                     rope_sin,
                     1,
                     1,
                     head_dim,
                     aligned_head_dim,
                     token_index);
    }

    /* Step 4: KV cache write (direct-to-cache) */

    /* Step 5: Attention (decode, flash) */
    attention_forward_decode_head_major_gqa_flash(q_token,
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
    /* WO projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t attn_q8_bytes = ((((H * head_dim) + 255) / 256) * 292);
    uint8_t attn_q8[attn_q8_bytes];
    quantize_row_q8_k(attn_token, attn_q8, H * head_dim);
    gemv_q4_k_q8_k(proj_tmp, WO, attn_q8, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    model_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln2_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln2_q8[ln2_q8_bytes];
    quantize_row_q8_k(ln2_out, ln2_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(fc1_out, W1, ln2_q8, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t swiglu_q8_bytes = ((((aligned_intermediate_dim) + 255) / 256) * 292);
    uint8_t swiglu_q8[swiglu_q8_bytes];
    quantize_row_q8_k(swiglu_out, swiglu_q8, aligned_intermediate_dim);
    gemv_q4_k_q8_k(mlp_out, W2, swiglu_q8, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 2: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_2_decode(
    MODELModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[2];

    float *input = MODEL_PTR(model, MODEL_LAYERS[1].output);

    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *k_cache = MODEL_PTR(model, L->k);
    float *v_cache = MODEL_PTR(model, L->v);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    /* Weights (explicit types for layer 2) */
    const void *WQ = (const void *)MODEL_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)MODEL_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)MODEL_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)MODEL_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;

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
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* Step 2: QKV projection */
    /* Q projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln1_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln1_q8[ln1_q8_bytes];
    quantize_row_q8_k(ln1_out, ln1_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(q_token, WQ, ln1_q8, H * head_dim, aligned_embed_dim);
    if (aligned_head_dim > head_dim) {
        for (int h = 0; h < H; ++h) {
            float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                q_head[d] = 0.0f;
            }
        }
    }

    /* K projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wk_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wk_head_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(k_head, wk_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            k_head[d] = 0.0f;
        }
    }

    /* V projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wv_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wv_head_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        float *v_head = v_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(v_head, wv_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            v_head[d] = 0.0f;
        }
    }

    /* Step 3: RoPE */
    rope_forward(q_token,
                 rope_cos,
                 rope_sin,
                 H,
                 1,
                 head_dim,
                 aligned_head_dim,
                 token_index);
    for (int h = 0; h < H_kv; ++h) {
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        rope_forward(k_head,
                     rope_cos,
                     rope_sin,
                     1,
                     1,
                     head_dim,
                     aligned_head_dim,
                     token_index);
    }

    /* Step 4: KV cache write (direct-to-cache) */

    /* Step 5: Attention (decode, flash) */
    attention_forward_decode_head_major_gqa_flash(q_token,
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
    /* WO projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t attn_q8_bytes = ((((H * head_dim) + 255) / 256) * 292);
    uint8_t attn_q8[attn_q8_bytes];
    quantize_row_q8_k(attn_token, attn_q8, H * head_dim);
    gemv_q4_k_q8_k(proj_tmp, WO, attn_q8, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    model_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln2_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln2_q8[ln2_q8_bytes];
    quantize_row_q8_k(ln2_out, ln2_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(fc1_out, W1, ln2_q8, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t swiglu_q8_bytes = ((((aligned_intermediate_dim) + 255) / 256) * 292);
    uint8_t swiglu_q8[swiglu_q8_bytes];
    quantize_row_q8_k(swiglu_out, swiglu_q8, aligned_intermediate_dim);
    gemv_q4_k_q8_k(mlp_out, W2, swiglu_q8, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 3: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_3_decode(
    MODELModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[3];

    float *input = MODEL_PTR(model, MODEL_LAYERS[2].output);

    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *k_cache = MODEL_PTR(model, L->k);
    float *v_cache = MODEL_PTR(model, L->v);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    /* Weights (explicit types for layer 3) */
    const void *WQ = (const void *)MODEL_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)MODEL_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)MODEL_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)MODEL_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;

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
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* Step 2: QKV projection */
    /* Q projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln1_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln1_q8[ln1_q8_bytes];
    quantize_row_q8_k(ln1_out, ln1_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(q_token, WQ, ln1_q8, H * head_dim, aligned_embed_dim);
    if (aligned_head_dim > head_dim) {
        for (int h = 0; h < H; ++h) {
            float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                q_head[d] = 0.0f;
            }
        }
    }

    /* K projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wk_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wk_head_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(k_head, wk_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            k_head[d] = 0.0f;
        }
    }

    /* V projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wv_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wv_head_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        float *v_head = v_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(v_head, wv_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            v_head[d] = 0.0f;
        }
    }

    /* Step 3: RoPE */
    rope_forward(q_token,
                 rope_cos,
                 rope_sin,
                 H,
                 1,
                 head_dim,
                 aligned_head_dim,
                 token_index);
    for (int h = 0; h < H_kv; ++h) {
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        rope_forward(k_head,
                     rope_cos,
                     rope_sin,
                     1,
                     1,
                     head_dim,
                     aligned_head_dim,
                     token_index);
    }

    /* Step 4: KV cache write (direct-to-cache) */

    /* Step 5: Attention (decode, flash) */
    attention_forward_decode_head_major_gqa_flash(q_token,
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
    /* WO projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t attn_q8_bytes = ((((H * head_dim) + 255) / 256) * 292);
    uint8_t attn_q8[attn_q8_bytes];
    quantize_row_q8_k(attn_token, attn_q8, H * head_dim);
    gemv_q4_k_q8_k(proj_tmp, WO, attn_q8, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    model_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln2_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln2_q8[ln2_q8_bytes];
    quantize_row_q8_k(ln2_out, ln2_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(fc1_out, W1, ln2_q8, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t swiglu_q8_bytes = ((((aligned_intermediate_dim) + 255) / 256) * 292);
    uint8_t swiglu_q8[swiglu_q8_bytes];
    quantize_row_q8_k(swiglu_out, swiglu_q8, aligned_intermediate_dim);
    gemv_q4_k_q8_k(mlp_out, W2, swiglu_q8, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 4: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_4_decode(
    MODELModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[4];

    float *input = MODEL_PTR(model, MODEL_LAYERS[3].output);

    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *k_cache = MODEL_PTR(model, L->k);
    float *v_cache = MODEL_PTR(model, L->v);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    /* Weights (explicit types for layer 4) */
    const void *WQ = (const void *)MODEL_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)MODEL_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)MODEL_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)MODEL_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;

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
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* Step 2: QKV projection */
    /* Q projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln1_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln1_q8[ln1_q8_bytes];
    quantize_row_q8_k(ln1_out, ln1_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(q_token, WQ, ln1_q8, H * head_dim, aligned_embed_dim);
    if (aligned_head_dim > head_dim) {
        for (int h = 0; h < H; ++h) {
            float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                q_head[d] = 0.0f;
            }
        }
    }

    /* K projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wk_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wk_head_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(k_head, wk_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            k_head[d] = 0.0f;
        }
    }

    /* V projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wv_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wv_head_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        float *v_head = v_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(v_head, wv_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            v_head[d] = 0.0f;
        }
    }

    /* Step 3: RoPE */
    rope_forward(q_token,
                 rope_cos,
                 rope_sin,
                 H,
                 1,
                 head_dim,
                 aligned_head_dim,
                 token_index);
    for (int h = 0; h < H_kv; ++h) {
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        rope_forward(k_head,
                     rope_cos,
                     rope_sin,
                     1,
                     1,
                     head_dim,
                     aligned_head_dim,
                     token_index);
    }

    /* Step 4: KV cache write (direct-to-cache) */

    /* Step 5: Attention (decode, flash) */
    attention_forward_decode_head_major_gqa_flash(q_token,
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
    /* WO projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t attn_q8_bytes = ((((H * head_dim) + 255) / 256) * 292);
    uint8_t attn_q8[attn_q8_bytes];
    quantize_row_q8_k(attn_token, attn_q8, H * head_dim);
    gemv_q4_k_q8_k(proj_tmp, WO, attn_q8, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    model_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln2_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln2_q8[ln2_q8_bytes];
    quantize_row_q8_k(ln2_out, ln2_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(fc1_out, W1, ln2_q8, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t swiglu_q8_bytes = ((((aligned_intermediate_dim) + 255) / 256) * 292);
    uint8_t swiglu_q8[swiglu_q8_bytes];
    quantize_row_q8_k(swiglu_out, swiglu_q8, aligned_intermediate_dim);
    gemv_q4_k_q8_k(mlp_out, W2, swiglu_q8, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 5: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_5_decode(
    MODELModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[5];

    float *input = MODEL_PTR(model, MODEL_LAYERS[4].output);

    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *k_cache = MODEL_PTR(model, L->k);
    float *v_cache = MODEL_PTR(model, L->v);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    /* Weights (explicit types for layer 5) */
    const void *WQ = (const void *)MODEL_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)MODEL_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)MODEL_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)MODEL_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;

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
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* Step 2: QKV projection */
    /* Q projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln1_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln1_q8[ln1_q8_bytes];
    quantize_row_q8_k(ln1_out, ln1_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(q_token, WQ, ln1_q8, H * head_dim, aligned_embed_dim);
    if (aligned_head_dim > head_dim) {
        for (int h = 0; h < H; ++h) {
            float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                q_head[d] = 0.0f;
            }
        }
    }

    /* K projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wk_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wk_head_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(k_head, wk_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            k_head[d] = 0.0f;
        }
    }

    /* V projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wv_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wv_head_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        float *v_head = v_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(v_head, wv_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            v_head[d] = 0.0f;
        }
    }

    /* Step 3: RoPE */
    rope_forward(q_token,
                 rope_cos,
                 rope_sin,
                 H,
                 1,
                 head_dim,
                 aligned_head_dim,
                 token_index);
    for (int h = 0; h < H_kv; ++h) {
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        rope_forward(k_head,
                     rope_cos,
                     rope_sin,
                     1,
                     1,
                     head_dim,
                     aligned_head_dim,
                     token_index);
    }

    /* Step 4: KV cache write (direct-to-cache) */

    /* Step 5: Attention (decode, flash) */
    attention_forward_decode_head_major_gqa_flash(q_token,
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
    /* WO projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t attn_q8_bytes = ((((H * head_dim) + 255) / 256) * 292);
    uint8_t attn_q8[attn_q8_bytes];
    quantize_row_q8_k(attn_token, attn_q8, H * head_dim);
    gemv_q4_k_q8_k(proj_tmp, WO, attn_q8, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    model_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln2_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln2_q8[ln2_q8_bytes];
    quantize_row_q8_k(ln2_out, ln2_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(fc1_out, W1, ln2_q8, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t swiglu_q8_bytes = ((((aligned_intermediate_dim) + 255) / 256) * 292);
    uint8_t swiglu_q8[swiglu_q8_bytes];
    quantize_row_q8_k(swiglu_out, swiglu_q8, aligned_intermediate_dim);
    gemv_q4_k_q8_k(mlp_out, W2, swiglu_q8, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 6: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_6_decode(
    MODELModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[6];

    float *input = MODEL_PTR(model, MODEL_LAYERS[5].output);

    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *k_cache = MODEL_PTR(model, L->k);
    float *v_cache = MODEL_PTR(model, L->v);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    /* Weights (explicit types for layer 6) */
    const void *WQ = (const void *)MODEL_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)MODEL_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)MODEL_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)MODEL_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;

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
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* Step 2: QKV projection */
    /* Q projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln1_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln1_q8[ln1_q8_bytes];
    quantize_row_q8_k(ln1_out, ln1_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(q_token, WQ, ln1_q8, H * head_dim, aligned_embed_dim);
    if (aligned_head_dim > head_dim) {
        for (int h = 0; h < H; ++h) {
            float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                q_head[d] = 0.0f;
            }
        }
    }

    /* K projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wk_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wk_head_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(k_head, wk_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            k_head[d] = 0.0f;
        }
    }

    /* V projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wv_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wv_head_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        float *v_head = v_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(v_head, wv_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            v_head[d] = 0.0f;
        }
    }

    /* Step 3: RoPE */
    rope_forward(q_token,
                 rope_cos,
                 rope_sin,
                 H,
                 1,
                 head_dim,
                 aligned_head_dim,
                 token_index);
    for (int h = 0; h < H_kv; ++h) {
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        rope_forward(k_head,
                     rope_cos,
                     rope_sin,
                     1,
                     1,
                     head_dim,
                     aligned_head_dim,
                     token_index);
    }

    /* Step 4: KV cache write (direct-to-cache) */

    /* Step 5: Attention (decode, flash) */
    attention_forward_decode_head_major_gqa_flash(q_token,
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
    /* WO projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t attn_q8_bytes = ((((H * head_dim) + 255) / 256) * 292);
    uint8_t attn_q8[attn_q8_bytes];
    quantize_row_q8_k(attn_token, attn_q8, H * head_dim);
    gemv_q4_k_q8_k(proj_tmp, WO, attn_q8, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    model_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln2_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln2_q8[ln2_q8_bytes];
    quantize_row_q8_k(ln2_out, ln2_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(fc1_out, W1, ln2_q8, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t swiglu_q8_bytes = ((((aligned_intermediate_dim) + 255) / 256) * 292);
    uint8_t swiglu_q8[swiglu_q8_bytes];
    quantize_row_q8_k(swiglu_out, swiglu_q8, aligned_intermediate_dim);
    gemv_q4_k_q8_k(mlp_out, W2, swiglu_q8, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 7: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_7_decode(
    MODELModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[7];

    float *input = MODEL_PTR(model, MODEL_LAYERS[6].output);

    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *k_cache = MODEL_PTR(model, L->k);
    float *v_cache = MODEL_PTR(model, L->v);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    /* Weights (explicit types for layer 7) */
    const void *WQ = (const void *)MODEL_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)MODEL_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)MODEL_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)MODEL_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;

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
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* Step 2: QKV projection */
    /* Q projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln1_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln1_q8[ln1_q8_bytes];
    quantize_row_q8_k(ln1_out, ln1_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(q_token, WQ, ln1_q8, H * head_dim, aligned_embed_dim);
    if (aligned_head_dim > head_dim) {
        for (int h = 0; h < H; ++h) {
            float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                q_head[d] = 0.0f;
            }
        }
    }

    /* K projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wk_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wk_head_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(k_head, wk_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            k_head[d] = 0.0f;
        }
    }

    /* V projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wv_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wv_head_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        float *v_head = v_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(v_head, wv_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            v_head[d] = 0.0f;
        }
    }

    /* Step 3: RoPE */
    rope_forward(q_token,
                 rope_cos,
                 rope_sin,
                 H,
                 1,
                 head_dim,
                 aligned_head_dim,
                 token_index);
    for (int h = 0; h < H_kv; ++h) {
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        rope_forward(k_head,
                     rope_cos,
                     rope_sin,
                     1,
                     1,
                     head_dim,
                     aligned_head_dim,
                     token_index);
    }

    /* Step 4: KV cache write (direct-to-cache) */

    /* Step 5: Attention (decode, flash) */
    attention_forward_decode_head_major_gqa_flash(q_token,
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
    /* WO projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t attn_q8_bytes = ((((H * head_dim) + 255) / 256) * 292);
    uint8_t attn_q8[attn_q8_bytes];
    quantize_row_q8_k(attn_token, attn_q8, H * head_dim);
    gemv_q4_k_q8_k(proj_tmp, WO, attn_q8, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    model_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln2_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln2_q8[ln2_q8_bytes];
    quantize_row_q8_k(ln2_out, ln2_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(fc1_out, W1, ln2_q8, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t swiglu_q8_bytes = ((((aligned_intermediate_dim) + 255) / 256) * 292);
    uint8_t swiglu_q8[swiglu_q8_bytes];
    quantize_row_q8_k(swiglu_out, swiglu_q8, aligned_intermediate_dim);
    gemv_q4_k_q8_k(mlp_out, W2, swiglu_q8, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 8: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_8_decode(
    MODELModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[8];

    float *input = MODEL_PTR(model, MODEL_LAYERS[7].output);

    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *k_cache = MODEL_PTR(model, L->k);
    float *v_cache = MODEL_PTR(model, L->v);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    /* Weights (explicit types for layer 8) */
    const void *WQ = (const void *)MODEL_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)MODEL_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)MODEL_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)MODEL_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;

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
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* Step 2: QKV projection */
    /* Q projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln1_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln1_q8[ln1_q8_bytes];
    quantize_row_q8_k(ln1_out, ln1_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(q_token, WQ, ln1_q8, H * head_dim, aligned_embed_dim);
    if (aligned_head_dim > head_dim) {
        for (int h = 0; h < H; ++h) {
            float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                q_head[d] = 0.0f;
            }
        }
    }

    /* K projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wk_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wk_head_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(k_head, wk_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            k_head[d] = 0.0f;
        }
    }

    /* V projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wv_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wv_head_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        float *v_head = v_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(v_head, wv_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            v_head[d] = 0.0f;
        }
    }

    /* Step 3: RoPE */
    rope_forward(q_token,
                 rope_cos,
                 rope_sin,
                 H,
                 1,
                 head_dim,
                 aligned_head_dim,
                 token_index);
    for (int h = 0; h < H_kv; ++h) {
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        rope_forward(k_head,
                     rope_cos,
                     rope_sin,
                     1,
                     1,
                     head_dim,
                     aligned_head_dim,
                     token_index);
    }

    /* Step 4: KV cache write (direct-to-cache) */

    /* Step 5: Attention (decode, flash) */
    attention_forward_decode_head_major_gqa_flash(q_token,
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
    /* WO projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t attn_q8_bytes = ((((H * head_dim) + 255) / 256) * 292);
    uint8_t attn_q8[attn_q8_bytes];
    quantize_row_q8_k(attn_token, attn_q8, H * head_dim);
    gemv_q4_k_q8_k(proj_tmp, WO, attn_q8, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    model_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln2_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln2_q8[ln2_q8_bytes];
    quantize_row_q8_k(ln2_out, ln2_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(fc1_out, W1, ln2_q8, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t swiglu_q8_bytes = ((((aligned_intermediate_dim) + 255) / 256) * 292);
    uint8_t swiglu_q8[swiglu_q8_bytes];
    quantize_row_q8_k(swiglu_out, swiglu_q8, aligned_intermediate_dim);
    gemv_q4_k_q8_k(mlp_out, W2, swiglu_q8, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 9: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_9_decode(
    MODELModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[9];

    float *input = MODEL_PTR(model, MODEL_LAYERS[8].output);

    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *k_cache = MODEL_PTR(model, L->k);
    float *v_cache = MODEL_PTR(model, L->v);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    /* Weights (explicit types for layer 9) */
    const void *WQ = (const void *)MODEL_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)MODEL_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)MODEL_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)MODEL_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;

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
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* Step 2: QKV projection */
    /* Q projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln1_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln1_q8[ln1_q8_bytes];
    quantize_row_q8_k(ln1_out, ln1_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(q_token, WQ, ln1_q8, H * head_dim, aligned_embed_dim);
    if (aligned_head_dim > head_dim) {
        for (int h = 0; h < H; ++h) {
            float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                q_head[d] = 0.0f;
            }
        }
    }

    /* K projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wk_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wk_head_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(k_head, wk_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            k_head[d] = 0.0f;
        }
    }

    /* V projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wv_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wv_head_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        float *v_head = v_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(v_head, wv_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            v_head[d] = 0.0f;
        }
    }

    /* Step 3: RoPE */
    rope_forward(q_token,
                 rope_cos,
                 rope_sin,
                 H,
                 1,
                 head_dim,
                 aligned_head_dim,
                 token_index);
    for (int h = 0; h < H_kv; ++h) {
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        rope_forward(k_head,
                     rope_cos,
                     rope_sin,
                     1,
                     1,
                     head_dim,
                     aligned_head_dim,
                     token_index);
    }

    /* Step 4: KV cache write (direct-to-cache) */

    /* Step 5: Attention (decode, flash) */
    attention_forward_decode_head_major_gqa_flash(q_token,
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
    /* WO projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t attn_q8_bytes = ((((H * head_dim) + 255) / 256) * 292);
    uint8_t attn_q8[attn_q8_bytes];
    quantize_row_q8_k(attn_token, attn_q8, H * head_dim);
    gemv_q4_k_q8_k(proj_tmp, WO, attn_q8, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    model_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln2_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln2_q8[ln2_q8_bytes];
    quantize_row_q8_k(ln2_out, ln2_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(fc1_out, W1, ln2_q8, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t swiglu_q8_bytes = ((((aligned_intermediate_dim) + 255) / 256) * 292);
    uint8_t swiglu_q8[swiglu_q8_bytes];
    quantize_row_q8_k(swiglu_out, swiglu_q8, aligned_intermediate_dim);
    gemv_q4_k_q8_k(mlp_out, W2, swiglu_q8, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 10: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_10_decode(
    MODELModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[10];

    float *input = MODEL_PTR(model, MODEL_LAYERS[9].output);

    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *k_cache = MODEL_PTR(model, L->k);
    float *v_cache = MODEL_PTR(model, L->v);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    /* Weights (explicit types for layer 10) */
    const void *WQ = (const void *)MODEL_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)MODEL_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)MODEL_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)MODEL_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;

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
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* Step 2: QKV projection */
    /* Q projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln1_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln1_q8[ln1_q8_bytes];
    quantize_row_q8_k(ln1_out, ln1_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(q_token, WQ, ln1_q8, H * head_dim, aligned_embed_dim);
    if (aligned_head_dim > head_dim) {
        for (int h = 0; h < H; ++h) {
            float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                q_head[d] = 0.0f;
            }
        }
    }

    /* K projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wk_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wk_head_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(k_head, wk_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            k_head[d] = 0.0f;
        }
    }

    /* V projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wv_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wv_head_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        float *v_head = v_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(v_head, wv_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            v_head[d] = 0.0f;
        }
    }

    /* Step 3: RoPE */
    rope_forward(q_token,
                 rope_cos,
                 rope_sin,
                 H,
                 1,
                 head_dim,
                 aligned_head_dim,
                 token_index);
    for (int h = 0; h < H_kv; ++h) {
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        rope_forward(k_head,
                     rope_cos,
                     rope_sin,
                     1,
                     1,
                     head_dim,
                     aligned_head_dim,
                     token_index);
    }

    /* Step 4: KV cache write (direct-to-cache) */

    /* Step 5: Attention (decode, flash) */
    attention_forward_decode_head_major_gqa_flash(q_token,
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
    /* WO projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t attn_q8_bytes = ((((H * head_dim) + 255) / 256) * 292);
    uint8_t attn_q8[attn_q8_bytes];
    quantize_row_q8_k(attn_token, attn_q8, H * head_dim);
    gemv_q4_k_q8_k(proj_tmp, WO, attn_q8, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    model_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln2_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln2_q8[ln2_q8_bytes];
    quantize_row_q8_k(ln2_out, ln2_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(fc1_out, W1, ln2_q8, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t swiglu_q8_bytes = ((((aligned_intermediate_dim) + 255) / 256) * 292);
    uint8_t swiglu_q8[swiglu_q8_bytes];
    quantize_row_q8_k(swiglu_out, swiglu_q8, aligned_intermediate_dim);
    gemv_q4_k_q8_k(mlp_out, W2, swiglu_q8, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 11: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_11_decode(
    MODELModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[11];

    float *input = MODEL_PTR(model, MODEL_LAYERS[10].output);

    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *k_cache = MODEL_PTR(model, L->k);
    float *v_cache = MODEL_PTR(model, L->v);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    /* Weights (explicit types for layer 11) */
    const void *WQ = (const void *)MODEL_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)MODEL_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)MODEL_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)MODEL_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;

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
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* Step 2: QKV projection */
    /* Q projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln1_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln1_q8[ln1_q8_bytes];
    quantize_row_q8_k(ln1_out, ln1_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(q_token, WQ, ln1_q8, H * head_dim, aligned_embed_dim);
    if (aligned_head_dim > head_dim) {
        for (int h = 0; h < H; ++h) {
            float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                q_head[d] = 0.0f;
            }
        }
    }

    /* K projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wk_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wk_head_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(k_head, wk_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            k_head[d] = 0.0f;
        }
    }

    /* V projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wv_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wv_head_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        float *v_head = v_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(v_head, wv_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            v_head[d] = 0.0f;
        }
    }

    /* Step 3: RoPE */
    rope_forward(q_token,
                 rope_cos,
                 rope_sin,
                 H,
                 1,
                 head_dim,
                 aligned_head_dim,
                 token_index);
    for (int h = 0; h < H_kv; ++h) {
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        rope_forward(k_head,
                     rope_cos,
                     rope_sin,
                     1,
                     1,
                     head_dim,
                     aligned_head_dim,
                     token_index);
    }

    /* Step 4: KV cache write (direct-to-cache) */

    /* Step 5: Attention (decode, flash) */
    attention_forward_decode_head_major_gqa_flash(q_token,
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
    /* WO projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t attn_q8_bytes = ((((H * head_dim) + 255) / 256) * 292);
    uint8_t attn_q8[attn_q8_bytes];
    quantize_row_q8_k(attn_token, attn_q8, H * head_dim);
    gemv_q4_k_q8_k(proj_tmp, WO, attn_q8, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    model_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln2_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln2_q8[ln2_q8_bytes];
    quantize_row_q8_k(ln2_out, ln2_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(fc1_out, W1, ln2_q8, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t swiglu_q8_bytes = ((((aligned_intermediate_dim) + 255) / 256) * 292);
    uint8_t swiglu_q8[swiglu_q8_bytes];
    quantize_row_q8_k(swiglu_out, swiglu_q8, aligned_intermediate_dim);
    gemv_q4_k_q8_k(mlp_out, W2, swiglu_q8, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 12: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_12_decode(
    MODELModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[12];

    float *input = MODEL_PTR(model, MODEL_LAYERS[11].output);

    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *k_cache = MODEL_PTR(model, L->k);
    float *v_cache = MODEL_PTR(model, L->v);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    /* Weights (explicit types for layer 12) */
    const void *WQ = (const void *)MODEL_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)MODEL_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)MODEL_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)MODEL_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;

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
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* Step 2: QKV projection */
    /* Q projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln1_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln1_q8[ln1_q8_bytes];
    quantize_row_q8_k(ln1_out, ln1_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(q_token, WQ, ln1_q8, H * head_dim, aligned_embed_dim);
    if (aligned_head_dim > head_dim) {
        for (int h = 0; h < H; ++h) {
            float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                q_head[d] = 0.0f;
            }
        }
    }

    /* K projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wk_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wk_head_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(k_head, wk_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            k_head[d] = 0.0f;
        }
    }

    /* V projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wv_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wv_head_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        float *v_head = v_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(v_head, wv_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            v_head[d] = 0.0f;
        }
    }

    /* Step 3: RoPE */
    rope_forward(q_token,
                 rope_cos,
                 rope_sin,
                 H,
                 1,
                 head_dim,
                 aligned_head_dim,
                 token_index);
    for (int h = 0; h < H_kv; ++h) {
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        rope_forward(k_head,
                     rope_cos,
                     rope_sin,
                     1,
                     1,
                     head_dim,
                     aligned_head_dim,
                     token_index);
    }

    /* Step 4: KV cache write (direct-to-cache) */

    /* Step 5: Attention (decode, flash) */
    attention_forward_decode_head_major_gqa_flash(q_token,
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
    /* WO projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t attn_q8_bytes = ((((H * head_dim) + 255) / 256) * 292);
    uint8_t attn_q8[attn_q8_bytes];
    quantize_row_q8_k(attn_token, attn_q8, H * head_dim);
    gemv_q4_k_q8_k(proj_tmp, WO, attn_q8, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    model_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln2_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln2_q8[ln2_q8_bytes];
    quantize_row_q8_k(ln2_out, ln2_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(fc1_out, W1, ln2_q8, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t swiglu_q8_bytes = ((((aligned_intermediate_dim) + 255) / 256) * 292);
    uint8_t swiglu_q8[swiglu_q8_bytes];
    quantize_row_q8_k(swiglu_out, swiglu_q8, aligned_intermediate_dim);
    gemv_q4_k_q8_k(mlp_out, W2, swiglu_q8, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 13: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_13_decode(
    MODELModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[13];

    float *input = MODEL_PTR(model, MODEL_LAYERS[12].output);

    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *k_cache = MODEL_PTR(model, L->k);
    float *v_cache = MODEL_PTR(model, L->v);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    /* Weights (explicit types for layer 13) */
    const void *WQ = (const void *)MODEL_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)MODEL_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)MODEL_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)MODEL_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;

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
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* Step 2: QKV projection */
    /* Q projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln1_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln1_q8[ln1_q8_bytes];
    quantize_row_q8_k(ln1_out, ln1_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(q_token, WQ, ln1_q8, H * head_dim, aligned_embed_dim);
    if (aligned_head_dim > head_dim) {
        for (int h = 0; h < H; ++h) {
            float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                q_head[d] = 0.0f;
            }
        }
    }

    /* K projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wk_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wk_head_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(k_head, wk_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            k_head[d] = 0.0f;
        }
    }

    /* V projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wv_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wv_head_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        float *v_head = v_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(v_head, wv_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            v_head[d] = 0.0f;
        }
    }

    /* Step 3: RoPE */
    rope_forward(q_token,
                 rope_cos,
                 rope_sin,
                 H,
                 1,
                 head_dim,
                 aligned_head_dim,
                 token_index);
    for (int h = 0; h < H_kv; ++h) {
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        rope_forward(k_head,
                     rope_cos,
                     rope_sin,
                     1,
                     1,
                     head_dim,
                     aligned_head_dim,
                     token_index);
    }

    /* Step 4: KV cache write (direct-to-cache) */

    /* Step 5: Attention (decode, flash) */
    attention_forward_decode_head_major_gqa_flash(q_token,
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
    /* WO projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t attn_q8_bytes = ((((H * head_dim) + 255) / 256) * 292);
    uint8_t attn_q8[attn_q8_bytes];
    quantize_row_q8_k(attn_token, attn_q8, H * head_dim);
    gemv_q4_k_q8_k(proj_tmp, WO, attn_q8, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    model_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln2_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln2_q8[ln2_q8_bytes];
    quantize_row_q8_k(ln2_out, ln2_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(fc1_out, W1, ln2_q8, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t swiglu_q8_bytes = ((((aligned_intermediate_dim) + 255) / 256) * 292);
    uint8_t swiglu_q8[swiglu_q8_bytes];
    quantize_row_q8_k(swiglu_out, swiglu_q8, aligned_intermediate_dim);
    gemv_q4_k_q8_k(mlp_out, W2, swiglu_q8, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 14: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_14_decode(
    MODELModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[14];

    float *input = MODEL_PTR(model, MODEL_LAYERS[13].output);

    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *k_cache = MODEL_PTR(model, L->k);
    float *v_cache = MODEL_PTR(model, L->v);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    /* Weights (explicit types for layer 14) */
    const void *WQ = (const void *)MODEL_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)MODEL_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)MODEL_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)MODEL_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;

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
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* Step 2: QKV projection */
    /* Q projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln1_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln1_q8[ln1_q8_bytes];
    quantize_row_q8_k(ln1_out, ln1_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(q_token, WQ, ln1_q8, H * head_dim, aligned_embed_dim);
    if (aligned_head_dim > head_dim) {
        for (int h = 0; h < H; ++h) {
            float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                q_head[d] = 0.0f;
            }
        }
    }

    /* K projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wk_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wk_head_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(k_head, wk_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            k_head[d] = 0.0f;
        }
    }

    /* V projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wv_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wv_head_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        float *v_head = v_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(v_head, wv_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            v_head[d] = 0.0f;
        }
    }

    /* Step 3: RoPE */
    rope_forward(q_token,
                 rope_cos,
                 rope_sin,
                 H,
                 1,
                 head_dim,
                 aligned_head_dim,
                 token_index);
    for (int h = 0; h < H_kv; ++h) {
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        rope_forward(k_head,
                     rope_cos,
                     rope_sin,
                     1,
                     1,
                     head_dim,
                     aligned_head_dim,
                     token_index);
    }

    /* Step 4: KV cache write (direct-to-cache) */

    /* Step 5: Attention (decode, flash) */
    attention_forward_decode_head_major_gqa_flash(q_token,
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
    /* WO projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t attn_q8_bytes = ((((H * head_dim) + 255) / 256) * 292);
    uint8_t attn_q8[attn_q8_bytes];
    quantize_row_q8_k(attn_token, attn_q8, H * head_dim);
    gemv_q4_k_q8_k(proj_tmp, WO, attn_q8, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    model_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln2_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln2_q8[ln2_q8_bytes];
    quantize_row_q8_k(ln2_out, ln2_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(fc1_out, W1, ln2_q8, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t swiglu_q8_bytes = ((((aligned_intermediate_dim) + 255) / 256) * 292);
    uint8_t swiglu_q8[swiglu_q8_bytes];
    quantize_row_q8_k(swiglu_out, swiglu_q8, aligned_intermediate_dim);
    gemv_q4_k_q8_k(mlp_out, W2, swiglu_q8, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 15: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_15_decode(
    MODELModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[15];

    float *input = MODEL_PTR(model, MODEL_LAYERS[14].output);

    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *k_cache = MODEL_PTR(model, L->k);
    float *v_cache = MODEL_PTR(model, L->v);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    /* Weights (explicit types for layer 15) */
    const void *WQ = (const void *)MODEL_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)MODEL_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)MODEL_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)MODEL_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;

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
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* Step 2: QKV projection */
    /* Q projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln1_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln1_q8[ln1_q8_bytes];
    quantize_row_q8_k(ln1_out, ln1_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(q_token, WQ, ln1_q8, H * head_dim, aligned_embed_dim);
    if (aligned_head_dim > head_dim) {
        for (int h = 0; h < H; ++h) {
            float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                q_head[d] = 0.0f;
            }
        }
    }

    /* K projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wk_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wk_head_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(k_head, wk_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            k_head[d] = 0.0f;
        }
    }

    /* V projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wv_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wv_head_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        float *v_head = v_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(v_head, wv_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            v_head[d] = 0.0f;
        }
    }

    /* Step 3: RoPE */
    rope_forward(q_token,
                 rope_cos,
                 rope_sin,
                 H,
                 1,
                 head_dim,
                 aligned_head_dim,
                 token_index);
    for (int h = 0; h < H_kv; ++h) {
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        rope_forward(k_head,
                     rope_cos,
                     rope_sin,
                     1,
                     1,
                     head_dim,
                     aligned_head_dim,
                     token_index);
    }

    /* Step 4: KV cache write (direct-to-cache) */

    /* Step 5: Attention (decode, flash) */
    attention_forward_decode_head_major_gqa_flash(q_token,
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
    /* WO projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t attn_q8_bytes = ((((H * head_dim) + 255) / 256) * 292);
    uint8_t attn_q8[attn_q8_bytes];
    quantize_row_q8_k(attn_token, attn_q8, H * head_dim);
    gemv_q4_k_q8_k(proj_tmp, WO, attn_q8, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    model_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln2_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln2_q8[ln2_q8_bytes];
    quantize_row_q8_k(ln2_out, ln2_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(fc1_out, W1, ln2_q8, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t swiglu_q8_bytes = ((((aligned_intermediate_dim) + 255) / 256) * 292);
    uint8_t swiglu_q8[swiglu_q8_bytes];
    quantize_row_q8_k(swiglu_out, swiglu_q8, aligned_intermediate_dim);
    gemv_q4_k_q8_k(mlp_out, W2, swiglu_q8, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 16: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_16_decode(
    MODELModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[16];

    float *input = MODEL_PTR(model, MODEL_LAYERS[15].output);

    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *k_cache = MODEL_PTR(model, L->k);
    float *v_cache = MODEL_PTR(model, L->v);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    /* Weights (explicit types for layer 16) */
    const void *WQ = (const void *)MODEL_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)MODEL_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)MODEL_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)MODEL_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;

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
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* Step 2: QKV projection */
    /* Q projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln1_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln1_q8[ln1_q8_bytes];
    quantize_row_q8_k(ln1_out, ln1_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(q_token, WQ, ln1_q8, H * head_dim, aligned_embed_dim);
    if (aligned_head_dim > head_dim) {
        for (int h = 0; h < H; ++h) {
            float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                q_head[d] = 0.0f;
            }
        }
    }

    /* K projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wk_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wk_head_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(k_head, wk_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            k_head[d] = 0.0f;
        }
    }

    /* V projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wv_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wv_head_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        float *v_head = v_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(v_head, wv_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            v_head[d] = 0.0f;
        }
    }

    /* Step 3: RoPE */
    rope_forward(q_token,
                 rope_cos,
                 rope_sin,
                 H,
                 1,
                 head_dim,
                 aligned_head_dim,
                 token_index);
    for (int h = 0; h < H_kv; ++h) {
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        rope_forward(k_head,
                     rope_cos,
                     rope_sin,
                     1,
                     1,
                     head_dim,
                     aligned_head_dim,
                     token_index);
    }

    /* Step 4: KV cache write (direct-to-cache) */

    /* Step 5: Attention (decode, flash) */
    attention_forward_decode_head_major_gqa_flash(q_token,
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
    /* WO projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t attn_q8_bytes = ((((H * head_dim) + 255) / 256) * 292);
    uint8_t attn_q8[attn_q8_bytes];
    quantize_row_q8_k(attn_token, attn_q8, H * head_dim);
    gemv_q4_k_q8_k(proj_tmp, WO, attn_q8, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    model_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln2_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln2_q8[ln2_q8_bytes];
    quantize_row_q8_k(ln2_out, ln2_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(fc1_out, W1, ln2_q8, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t swiglu_q8_bytes = ((((aligned_intermediate_dim) + 255) / 256) * 292);
    uint8_t swiglu_q8[swiglu_q8_bytes];
    quantize_row_q8_k(swiglu_out, swiglu_q8, aligned_intermediate_dim);
    gemv_q4_k_q8_k(mlp_out, W2, swiglu_q8, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 17: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_17_decode(
    MODELModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[17];

    float *input = MODEL_PTR(model, MODEL_LAYERS[16].output);

    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *k_cache = MODEL_PTR(model, L->k);
    float *v_cache = MODEL_PTR(model, L->v);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    /* Weights (explicit types for layer 17) */
    const void *WQ = (const void *)MODEL_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)MODEL_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)MODEL_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)MODEL_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;

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
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* Step 2: QKV projection */
    /* Q projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln1_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln1_q8[ln1_q8_bytes];
    quantize_row_q8_k(ln1_out, ln1_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(q_token, WQ, ln1_q8, H * head_dim, aligned_embed_dim);
    if (aligned_head_dim > head_dim) {
        for (int h = 0; h < H; ++h) {
            float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                q_head[d] = 0.0f;
            }
        }
    }

    /* K projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wk_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wk_head_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(k_head, wk_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            k_head[d] = 0.0f;
        }
    }

    /* V projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wv_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wv_head_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        float *v_head = v_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(v_head, wv_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            v_head[d] = 0.0f;
        }
    }

    /* Step 3: RoPE */
    rope_forward(q_token,
                 rope_cos,
                 rope_sin,
                 H,
                 1,
                 head_dim,
                 aligned_head_dim,
                 token_index);
    for (int h = 0; h < H_kv; ++h) {
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        rope_forward(k_head,
                     rope_cos,
                     rope_sin,
                     1,
                     1,
                     head_dim,
                     aligned_head_dim,
                     token_index);
    }

    /* Step 4: KV cache write (direct-to-cache) */

    /* Step 5: Attention (decode, flash) */
    attention_forward_decode_head_major_gqa_flash(q_token,
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
    /* WO projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t attn_q8_bytes = ((((H * head_dim) + 255) / 256) * 292);
    uint8_t attn_q8[attn_q8_bytes];
    quantize_row_q8_k(attn_token, attn_q8, H * head_dim);
    gemv_q4_k_q8_k(proj_tmp, WO, attn_q8, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    model_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln2_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln2_q8[ln2_q8_bytes];
    quantize_row_q8_k(ln2_out, ln2_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(fc1_out, W1, ln2_q8, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t swiglu_q8_bytes = ((((aligned_intermediate_dim) + 255) / 256) * 292);
    uint8_t swiglu_q8[swiglu_q8_bytes];
    quantize_row_q8_k(swiglu_out, swiglu_q8, aligned_intermediate_dim);
    gemv_q4_k_q8_k(mlp_out, W2, swiglu_q8, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 18: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_18_decode(
    MODELModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[18];

    float *input = MODEL_PTR(model, MODEL_LAYERS[17].output);

    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *k_cache = MODEL_PTR(model, L->k);
    float *v_cache = MODEL_PTR(model, L->v);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    /* Weights (explicit types for layer 18) */
    const void *WQ = (const void *)MODEL_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)MODEL_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)MODEL_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)MODEL_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;

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
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* Step 2: QKV projection */
    /* Q projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln1_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln1_q8[ln1_q8_bytes];
    quantize_row_q8_k(ln1_out, ln1_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(q_token, WQ, ln1_q8, H * head_dim, aligned_embed_dim);
    if (aligned_head_dim > head_dim) {
        for (int h = 0; h < H; ++h) {
            float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                q_head[d] = 0.0f;
            }
        }
    }

    /* K projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wk_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wk_head_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(k_head, wk_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            k_head[d] = 0.0f;
        }
    }

    /* V projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wv_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wv_head_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        float *v_head = v_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(v_head, wv_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            v_head[d] = 0.0f;
        }
    }

    /* Step 3: RoPE */
    rope_forward(q_token,
                 rope_cos,
                 rope_sin,
                 H,
                 1,
                 head_dim,
                 aligned_head_dim,
                 token_index);
    for (int h = 0; h < H_kv; ++h) {
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        rope_forward(k_head,
                     rope_cos,
                     rope_sin,
                     1,
                     1,
                     head_dim,
                     aligned_head_dim,
                     token_index);
    }

    /* Step 4: KV cache write (direct-to-cache) */

    /* Step 5: Attention (decode, flash) */
    attention_forward_decode_head_major_gqa_flash(q_token,
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
    /* WO projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t attn_q8_bytes = ((((H * head_dim) + 255) / 256) * 292);
    uint8_t attn_q8[attn_q8_bytes];
    quantize_row_q8_k(attn_token, attn_q8, H * head_dim);
    gemv_q4_k_q8_k(proj_tmp, WO, attn_q8, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    model_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln2_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln2_q8[ln2_q8_bytes];
    quantize_row_q8_k(ln2_out, ln2_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(fc1_out, W1, ln2_q8, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t swiglu_q8_bytes = ((((aligned_intermediate_dim) + 255) / 256) * 292);
    uint8_t swiglu_q8[swiglu_q8_bytes];
    quantize_row_q8_k(swiglu_out, swiglu_q8, aligned_intermediate_dim);
    gemv_q4_k_q8_k(mlp_out, W2, swiglu_q8, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 19: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_19_decode(
    MODELModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[19];

    float *input = MODEL_PTR(model, MODEL_LAYERS[18].output);

    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *k_cache = MODEL_PTR(model, L->k);
    float *v_cache = MODEL_PTR(model, L->v);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    /* Weights (explicit types for layer 19) */
    const void *WQ = (const void *)MODEL_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)MODEL_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)MODEL_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)MODEL_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;

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
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* Step 2: QKV projection */
    /* Q projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln1_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln1_q8[ln1_q8_bytes];
    quantize_row_q8_k(ln1_out, ln1_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(q_token, WQ, ln1_q8, H * head_dim, aligned_embed_dim);
    if (aligned_head_dim > head_dim) {
        for (int h = 0; h < H; ++h) {
            float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                q_head[d] = 0.0f;
            }
        }
    }

    /* K projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wk_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wk_head_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(k_head, wk_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            k_head[d] = 0.0f;
        }
    }

    /* V projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wv_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wv_head_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        float *v_head = v_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(v_head, wv_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            v_head[d] = 0.0f;
        }
    }

    /* Step 3: RoPE */
    rope_forward(q_token,
                 rope_cos,
                 rope_sin,
                 H,
                 1,
                 head_dim,
                 aligned_head_dim,
                 token_index);
    for (int h = 0; h < H_kv; ++h) {
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        rope_forward(k_head,
                     rope_cos,
                     rope_sin,
                     1,
                     1,
                     head_dim,
                     aligned_head_dim,
                     token_index);
    }

    /* Step 4: KV cache write (direct-to-cache) */

    /* Step 5: Attention (decode, flash) */
    attention_forward_decode_head_major_gqa_flash(q_token,
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
    /* WO projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t attn_q8_bytes = ((((H * head_dim) + 255) / 256) * 292);
    uint8_t attn_q8[attn_q8_bytes];
    quantize_row_q8_k(attn_token, attn_q8, H * head_dim);
    gemv_q4_k_q8_k(proj_tmp, WO, attn_q8, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    model_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln2_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln2_q8[ln2_q8_bytes];
    quantize_row_q8_k(ln2_out, ln2_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(fc1_out, W1, ln2_q8, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t swiglu_q8_bytes = ((((aligned_intermediate_dim) + 255) / 256) * 292);
    uint8_t swiglu_q8[swiglu_q8_bytes];
    quantize_row_q8_k(swiglu_out, swiglu_q8, aligned_intermediate_dim);
    gemv_q4_k_q8_k(mlp_out, W2, swiglu_q8, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 20: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_20_decode(
    MODELModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[20];

    float *input = MODEL_PTR(model, MODEL_LAYERS[19].output);

    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *k_cache = MODEL_PTR(model, L->k);
    float *v_cache = MODEL_PTR(model, L->v);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    /* Weights (explicit types for layer 20) */
    const void *WQ = (const void *)MODEL_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)MODEL_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)MODEL_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)MODEL_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;

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
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* Step 2: QKV projection */
    /* Q projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln1_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln1_q8[ln1_q8_bytes];
    quantize_row_q8_k(ln1_out, ln1_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(q_token, WQ, ln1_q8, H * head_dim, aligned_embed_dim);
    if (aligned_head_dim > head_dim) {
        for (int h = 0; h < H; ++h) {
            float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                q_head[d] = 0.0f;
            }
        }
    }

    /* K projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wk_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wk_head_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(k_head, wk_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            k_head[d] = 0.0f;
        }
    }

    /* V projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wv_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wv_head_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        float *v_head = v_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(v_head, wv_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            v_head[d] = 0.0f;
        }
    }

    /* Step 3: RoPE */
    rope_forward(q_token,
                 rope_cos,
                 rope_sin,
                 H,
                 1,
                 head_dim,
                 aligned_head_dim,
                 token_index);
    for (int h = 0; h < H_kv; ++h) {
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        rope_forward(k_head,
                     rope_cos,
                     rope_sin,
                     1,
                     1,
                     head_dim,
                     aligned_head_dim,
                     token_index);
    }

    /* Step 4: KV cache write (direct-to-cache) */

    /* Step 5: Attention (decode, flash) */
    attention_forward_decode_head_major_gqa_flash(q_token,
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
    /* WO projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t attn_q8_bytes = ((((H * head_dim) + 255) / 256) * 292);
    uint8_t attn_q8[attn_q8_bytes];
    quantize_row_q8_k(attn_token, attn_q8, H * head_dim);
    gemv_q4_k_q8_k(proj_tmp, WO, attn_q8, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    model_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln2_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln2_q8[ln2_q8_bytes];
    quantize_row_q8_k(ln2_out, ln2_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(fc1_out, W1, ln2_q8, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t swiglu_q8_bytes = ((((aligned_intermediate_dim) + 255) / 256) * 292);
    uint8_t swiglu_q8[swiglu_q8_bytes];
    quantize_row_q8_k(swiglu_out, swiglu_q8, aligned_intermediate_dim);
    gemv_q4_k_q8_k(mlp_out, W2, swiglu_q8, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 21: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_21_decode(
    MODELModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[21];

    float *input = MODEL_PTR(model, MODEL_LAYERS[20].output);

    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *k_cache = MODEL_PTR(model, L->k);
    float *v_cache = MODEL_PTR(model, L->v);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    /* Weights (explicit types for layer 21) */
    const void *WQ = (const void *)MODEL_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)MODEL_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)MODEL_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)MODEL_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;

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
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* Step 2: QKV projection */
    /* Q projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln1_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln1_q8[ln1_q8_bytes];
    quantize_row_q8_k(ln1_out, ln1_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(q_token, WQ, ln1_q8, H * head_dim, aligned_embed_dim);
    if (aligned_head_dim > head_dim) {
        for (int h = 0; h < H; ++h) {
            float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                q_head[d] = 0.0f;
            }
        }
    }

    /* K projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wk_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wk_head_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(k_head, wk_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            k_head[d] = 0.0f;
        }
    }

    /* V projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wv_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wv_head_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        float *v_head = v_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(v_head, wv_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            v_head[d] = 0.0f;
        }
    }

    /* Step 3: RoPE */
    rope_forward(q_token,
                 rope_cos,
                 rope_sin,
                 H,
                 1,
                 head_dim,
                 aligned_head_dim,
                 token_index);
    for (int h = 0; h < H_kv; ++h) {
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        rope_forward(k_head,
                     rope_cos,
                     rope_sin,
                     1,
                     1,
                     head_dim,
                     aligned_head_dim,
                     token_index);
    }

    /* Step 4: KV cache write (direct-to-cache) */

    /* Step 5: Attention (decode, flash) */
    attention_forward_decode_head_major_gqa_flash(q_token,
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
    /* WO projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t attn_q8_bytes = ((((H * head_dim) + 255) / 256) * 292);
    uint8_t attn_q8[attn_q8_bytes];
    quantize_row_q8_k(attn_token, attn_q8, H * head_dim);
    gemv_q4_k_q8_k(proj_tmp, WO, attn_q8, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    model_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln2_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln2_q8[ln2_q8_bytes];
    quantize_row_q8_k(ln2_out, ln2_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(fc1_out, W1, ln2_q8, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t swiglu_q8_bytes = ((((aligned_intermediate_dim) + 255) / 256) * 292);
    uint8_t swiglu_q8[swiglu_q8_bytes];
    quantize_row_q8_k(swiglu_out, swiglu_q8, aligned_intermediate_dim);
    gemv_q4_k_q8_k(mlp_out, W2, swiglu_q8, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 22: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_22_decode(
    MODELModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[22];

    float *input = MODEL_PTR(model, MODEL_LAYERS[21].output);

    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *k_cache = MODEL_PTR(model, L->k);
    float *v_cache = MODEL_PTR(model, L->v);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    /* Weights (explicit types for layer 22) */
    const void *WQ = (const void *)MODEL_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)MODEL_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)MODEL_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)MODEL_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;

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
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* Step 2: QKV projection */
    /* Q projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln1_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln1_q8[ln1_q8_bytes];
    quantize_row_q8_k(ln1_out, ln1_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(q_token, WQ, ln1_q8, H * head_dim, aligned_embed_dim);
    if (aligned_head_dim > head_dim) {
        for (int h = 0; h < H; ++h) {
            float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                q_head[d] = 0.0f;
            }
        }
    }

    /* K projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wk_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wk_head_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(k_head, wk_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            k_head[d] = 0.0f;
        }
    }

    /* V projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wv_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wv_head_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        float *v_head = v_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(v_head, wv_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            v_head[d] = 0.0f;
        }
    }

    /* Step 3: RoPE */
    rope_forward(q_token,
                 rope_cos,
                 rope_sin,
                 H,
                 1,
                 head_dim,
                 aligned_head_dim,
                 token_index);
    for (int h = 0; h < H_kv; ++h) {
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        rope_forward(k_head,
                     rope_cos,
                     rope_sin,
                     1,
                     1,
                     head_dim,
                     aligned_head_dim,
                     token_index);
    }

    /* Step 4: KV cache write (direct-to-cache) */

    /* Step 5: Attention (decode, flash) */
    attention_forward_decode_head_major_gqa_flash(q_token,
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
    /* WO projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t attn_q8_bytes = ((((H * head_dim) + 255) / 256) * 292);
    uint8_t attn_q8[attn_q8_bytes];
    quantize_row_q8_k(attn_token, attn_q8, H * head_dim);
    gemv_q4_k_q8_k(proj_tmp, WO, attn_q8, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    model_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln2_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln2_q8[ln2_q8_bytes];
    quantize_row_q8_k(ln2_out, ln2_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(fc1_out, W1, ln2_q8, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t swiglu_q8_bytes = ((((aligned_intermediate_dim) + 255) / 256) * 292);
    uint8_t swiglu_q8[swiglu_q8_bytes];
    quantize_row_q8_k(swiglu_out, swiglu_q8, aligned_intermediate_dim);
    gemv_q4_k_q8_k(mlp_out, W2, swiglu_q8, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 23: wq=q4_k wk=q4_k wv=q4_k wo=q4_k w1=q4_k w2=q4_k
 */
static void model_layer_23_decode(
    MODELModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const MODELLayerOffsets *L = &MODEL_LAYERS[23];

    float *input = MODEL_PTR(model, MODEL_LAYERS[22].output);

    float *ln1_gamma = MODEL_PTR(model, L->ln1_gamma);
    float *ln1_out = MODEL_PTR(model, L->ln1_out);
    float *ln2_gamma = MODEL_PTR(model, L->ln2_gamma);
    float *ln2_out = MODEL_PTR(model, L->ln2_out);
    float *k_cache = MODEL_PTR(model, L->k);
    float *v_cache = MODEL_PTR(model, L->v);
    float *proj_tmp = MODEL_PTR(model, L->proj_tmp);
    float *proj_scratch = MODEL_PTR(model, L->proj_scratch);
    float *residual1 = MODEL_PTR(model, L->residual1);
    float *mlp_out = MODEL_PTR(model, L->mlp_out);
    float *output = MODEL_PTR(model, L->output);

    /* Weights (explicit types for layer 23) */
    const void *WQ = (const void *)MODEL_PTR(model, L->wq);  /* Q4_K */
    const void *WK = (const void *)MODEL_PTR(model, L->wk);  /* Q4_K */
    const void *WV = (const void *)MODEL_PTR(model, L->wv);  /* Q4_K */
    const void *WO = (const void *)MODEL_PTR(model, L->wo);  /* Q4_K */
    const void *W1 = (const void *)MODEL_PTR(model, L->w1);  /* Q4_K (gate+up) */
    const void *W2 = (const void *)MODEL_PTR(model, L->w2);  /* Q4_K (down) */

    float *rope_cos = MODEL_PTR(model, MODEL_GLOBALS.rope_cos_cache);
    float *rope_sin = MODEL_PTR(model, MODEL_GLOBALS.rope_sin_cache);

    const int H = MODEL_NUM_HEADS;
    const int H_kv = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;

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
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    const size_t kv_head_stride = (size_t)aligned_context_window * (size_t)aligned_head_dim;

    /* Step 2: QKV projection */
    /* Q projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln1_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln1_q8[ln1_q8_bytes];
    quantize_row_q8_k(ln1_out, ln1_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(q_token, WQ, ln1_q8, H * head_dim, aligned_embed_dim);
    if (aligned_head_dim > head_dim) {
        for (int h = 0; h < H; ++h) {
            float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                q_head[d] = 0.0f;
            }
        }
    }

    /* K projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wk_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wk_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wk_head_elems);
    const uint8_t *WK_bytes = (const uint8_t *)WK;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(k_head, wk_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            k_head[d] = 0.0f;
        }
    }

    /* V projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k (direct-to-cache) */
    const size_t wv_head_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wv_head_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, wv_head_elems);
    const uint8_t *WV_bytes = (const uint8_t *)WV;
    /* ln1_q8 already quantized above */
    for (int h = 0; h < H_kv; ++h) {
        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);
        float *v_head = v_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        gemv_q4_k_q8_k(v_head, wv_h, ln1_q8, head_dim, aligned_embed_dim);
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            v_head[d] = 0.0f;
        }
    }

    /* Step 3: RoPE */
    rope_forward(q_token,
                 rope_cos,
                 rope_sin,
                 H,
                 1,
                 head_dim,
                 aligned_head_dim,
                 token_index);
    for (int h = 0; h < H_kv; ++h) {
        float *k_head = k_cache + (size_t)h * kv_head_stride + (size_t)token_index * (size_t)aligned_head_dim;
        rope_forward(k_head,
                     rope_cos,
                     rope_sin,
                     1,
                     1,
                     head_dim,
                     aligned_head_dim,
                     token_index);
    }

    /* Step 4: KV cache write (direct-to-cache) */

    /* Step 5: Attention (decode, flash) */
    attention_forward_decode_head_major_gqa_flash(q_token,
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
    /* WO projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t attn_q8_bytes = ((((H * head_dim) + 255) / 256) * 292);
    uint8_t attn_q8[attn_q8_bytes];
    quantize_row_q8_k(attn_token, attn_q8, H * head_dim);
    gemv_q4_k_q8_k(proj_tmp, WO, attn_q8, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    model_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t ln2_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t ln2_q8[ln2_q8_bytes];
    quantize_row_q8_k(ln2_out, ln2_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(fc1_out, W1, ln2_q8, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t swiglu_q8_bytes = ((((aligned_intermediate_dim) + 255) / 256) * 292);
    uint8_t swiglu_q8[swiglu_q8_bytes];
    quantize_row_q8_k(swiglu_out, swiglu_q8, aligned_intermediate_dim);
    gemv_q4_k_q8_k(mlp_out, W2, swiglu_q8, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    model_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/* ============================================================================
 * DECODE TOKEN (calls each layer explicitly)
 * ============================================================================ */

static void model_decode_token(
    MODELModel *model,
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
    float *embed_out = MODEL_PTR(model, MODEL_HEADER.embedded_input);
    const void *embed_weight = (const void *)MODEL_PTR(model, MODEL_HEADER.token_emb);
    /* Embedding: Q4_K -> embedding_forward_q4_k */
    embedding_forward_q4_k((const int32_t *)token,
                          1,
                          MODEL_VOCAB_SIZE,
                          embed_weight,
                          NULL,
                          embed_out,
                          MODEL_EMBED_DIM,
                          aligned_embed_dim,
                          1,
                          0);

    /* Process each layer explicitly */
    model_layer_0_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    model_layer_1_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    model_layer_2_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    model_layer_3_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    model_layer_4_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    model_layer_5_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    model_layer_6_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    model_layer_7_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    model_layer_8_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    model_layer_9_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    model_layer_10_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    model_layer_11_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    model_layer_12_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    model_layer_13_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    model_layer_14_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    model_layer_15_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    model_layer_16_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    model_layer_17_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    model_layer_18_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    model_layer_19_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    model_layer_20_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    model_layer_21_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    model_layer_22_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    model_layer_23_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);

    /* Final RMSNorm */
    float *last_hidden = MODEL_PTR(model, MODEL_LAYERS[23].output);
    float *final_ln_weight = MODEL_PTR(model, MODEL_FOOTER.final_ln_weight);
    float *final_out = MODEL_PTR(model, MODEL_FOOTER.final_output);
    rmsnorm_forward(last_hidden,
                    final_ln_weight,
                    final_out,
                    NULL,
                    1,
                    MODEL_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* LM head projection */
    float *logits = MODEL_PTR(model, MODEL_FOOTER.logits);
    const void *lm_head = (const void *)MODEL_PTR(model, MODEL_FOOTER.lm_head_weight);
    /* LM head (INT8): Q4_K x Q8_K -> gemv_q4_k_q8_k */
    const size_t final_q8_bytes = ((((aligned_embed_dim) + 255) / 256) * 292);
    uint8_t final_q8[final_q8_bytes];
    quantize_row_q8_k(final_out, final_q8, aligned_embed_dim);
    gemv_q4_k_q8_k(logits, lm_head, final_q8, MODEL_VOCAB_SIZE, aligned_embed_dim);
}

/* ============================================================================
 * PUBLIC API
 * ============================================================================ */

void model_forward(
    MODELModel *model,
    const int *tokens,
    int num_tokens
) {
    if (!model || !tokens || num_tokens <= 0) return;
    model_forward_prefill_impl(model, tokens, num_tokens);
}

void model_decode(MODELModel *model, const int *token, int token_index) {
    model_decode_token(model, token, token_index);
}
