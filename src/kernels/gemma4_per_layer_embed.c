#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "ck_threadpool.h"
#include "ckernel_quant.h"

typedef struct {
    ck_half d;
    ck_half dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
} ck_gemma4_block_q5_K;

static inline float ck_bf16_to_f32(uint16_t v)
{
    uint32_t bits = ((uint32_t)v) << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static inline float ck_gemma4_gelu(float x)
{
    const float c0 = 0.7978845608028654f;
    const float c1 = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(c0 * x * (1.0f + c1 * x * x)));
}

static inline void ck_gemma4_unpack_q5_k_scales(const uint8_t *scales, uint8_t *sc, uint8_t *m)
{
    sc[0] = scales[0] & 0x3F;
    sc[1] = scales[1] & 0x3F;
    sc[2] = scales[2] & 0x3F;
    sc[3] = scales[3] & 0x3F;

    m[0] = scales[4] & 0x3F;
    m[1] = scales[5] & 0x3F;
    m[2] = scales[6] & 0x3F;
    m[3] = scales[7] & 0x3F;

    sc[4] = (scales[8]  & 0x0F) | ((scales[0] >> 6) << 4);
    sc[5] = (scales[9]  & 0x0F) | ((scales[1] >> 6) << 4);
    sc[6] = (scales[10] & 0x0F) | ((scales[2] >> 6) << 4);
    sc[7] = (scales[11] & 0x0F) | ((scales[3] >> 6) << 4);

    m[4] = (scales[8]  >> 4) | ((scales[4] >> 6) << 4);
    m[5] = (scales[9]  >> 4) | ((scales[5] >> 6) << 4);
    m[6] = (scales[10] >> 4) | ((scales[6] >> 6) << 4);
    m[7] = (scales[11] >> 4) | ((scales[7] >> 6) << 4);
}

static inline uint8_t ck_gemma4_q5_k_value(const ck_gemma4_block_q5_K *block, int subblock, int i)
{
    const uint8_t *ql = block->qs + (subblock / 2) * 32;
    const uint8_t low = (subblock & 1) ? (uint8_t)(ql[i] >> 4) : (uint8_t)(ql[i] & 0x0F);
    const uint8_t high = (block->qh[i] & (uint8_t)(1u << subblock)) ? 16u : 0u;
    return (uint8_t)(low | high);
}

static void ck_gemma4_dequant_q5_k_block(const ck_gemma4_block_q5_K *block, float *out)
{
    uint8_t sc[8];
    uint8_t m[8];
    ck_gemma4_unpack_q5_k_scales(block->scales, sc, m);
    const float d = CK_FP16_TO_FP32(block->d);
    const float dmin = CK_FP16_TO_FP32(block->dmin);
    for (int s = 0; s < 8; ++s) {
        const float scale = d * (float)sc[s];
        const float minv = dmin * (float)m[s];
        for (int i = 0; i < 32; ++i) {
            out[s * 32 + i] = scale * (float)ck_gemma4_q5_k_value(block, s, i) - minv;
        }
    }
}

static void ck_gemma4_rmsnorm_tmp(const float *x, const float *gamma, float *out, int n, float eps)
{
    double ss = 0.0;
    for (int i = 0; i < n; ++i) {
        ss += (double)x[i] * (double)x[i];
    }
    const float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (int i = 0; i < n; ++i) {
        out[i] = x[i] * scale * gamma[i];
    }
}

typedef struct {
    float *per_layer_input;
    const float *hidden;
    const int32_t *token_ids;
    const void *per_layer_token_emb;
    const uint16_t *per_layer_model_proj;
    const float *per_layer_proj_norm;
    int num_layers;
    int embed_dim;
    int per_layer_dim;
    int vocab_size;
    float eps;
} ck_gemma4_prepare_args_t;

static void ck_gemma4_prepare_q5_range(int begin, int end, void *opaque)
{
    const ck_gemma4_prepare_args_t *args =
        (const ck_gemma4_prepare_args_t *)opaque;
    const ck_gemma4_block_q5_K *token_blocks =
        (const ck_gemma4_block_q5_K *)args->per_layer_token_emb;
    const size_t token_blocks_per_row = (size_t)args->num_layers;
    const float token_scale = sqrtf((float)args->per_layer_dim);
    const float model_scale = 1.0f / sqrtf((float)args->embed_dim);
    const float mix_scale = 0.7071067811865475f;
    float token_vec[QK_K];
    float proj_vec[QK_K];
    float proj_normed[QK_K];

    for (int t = begin; t < end; ++t) {
        const int token = args->token_ids[t];
        if (token < 0 || token >= args->vocab_size) continue;
        const float *h = args->hidden + (size_t)t * (size_t)args->embed_dim;
        for (int layer = 0; layer < args->num_layers; ++layer) {
            float *dst = args->per_layer_input +
                ((size_t)t * (size_t)args->num_layers + (size_t)layer) *
                    (size_t)args->per_layer_dim;
            const ck_gemma4_block_q5_K *tok_block = token_blocks +
                (size_t)token * token_blocks_per_row + (size_t)layer;
            ck_gemma4_dequant_q5_k_block(tok_block, token_vec);
            for (int i = 0; i < args->per_layer_dim; ++i) {
                token_vec[i] *= token_scale;
            }

            const uint16_t *model_proj_base = args->per_layer_model_proj +
                (size_t)layer * (size_t)args->per_layer_dim *
                    (size_t)args->embed_dim;
            for (int i = 0; i < args->per_layer_dim; ++i) {
                const uint16_t *row = model_proj_base +
                    (size_t)i * (size_t)args->embed_dim;
                float acc = 0.0f;
                for (int j = 0; j < args->embed_dim; ++j) {
                    acc += ck_bf16_to_f32(row[j]) * h[j];
                }
                proj_vec[i] = acc * model_scale;
            }
            ck_gemma4_rmsnorm_tmp(
                proj_vec, args->per_layer_proj_norm, proj_normed,
                args->per_layer_dim, args->eps);
            for (int i = 0; i < args->per_layer_dim; ++i) {
                dst[i] = (token_vec[i] + proj_normed[i]) * mix_scale;
            }
        }
    }
}

static void ck_gemma4_prepare_bf16_range(int begin, int end, void *opaque)
{
    const ck_gemma4_prepare_args_t *args =
        (const ck_gemma4_prepare_args_t *)opaque;
    const uint16_t *token_embeddings =
        (const uint16_t *)args->per_layer_token_emb;
    const float token_scale = sqrtf((float)args->per_layer_dim);
    const float model_scale = 1.0f / sqrtf((float)args->embed_dim);
    const float mix_scale = 0.7071067811865475f;
    float token_vec[QK_K];
    float proj_vec[QK_K];
    float proj_normed[QK_K];

    for (int t = begin; t < end; ++t) {
        const int token = args->token_ids[t];
        if (token < 0 || token >= args->vocab_size) continue;
        const float *h = args->hidden + (size_t)t * (size_t)args->embed_dim;
        for (int layer = 0; layer < args->num_layers; ++layer) {
            float *dst = args->per_layer_input +
                ((size_t)t * (size_t)args->num_layers + (size_t)layer) *
                    (size_t)args->per_layer_dim;
            const uint16_t *tok_row = token_embeddings +
                ((size_t)token * (size_t)args->num_layers + (size_t)layer) *
                    (size_t)args->per_layer_dim;
            for (int i = 0; i < args->per_layer_dim; ++i) {
                token_vec[i] = ck_bf16_to_f32(tok_row[i]) * token_scale;
            }

            const uint16_t *model_proj_base = args->per_layer_model_proj +
                (size_t)layer * (size_t)args->per_layer_dim *
                    (size_t)args->embed_dim;
            for (int i = 0; i < args->per_layer_dim; ++i) {
                const uint16_t *row = model_proj_base +
                    (size_t)i * (size_t)args->embed_dim;
                float acc = 0.0f;
                for (int j = 0; j < args->embed_dim; ++j) {
                    acc += ck_bf16_to_f32(row[j]) * h[j];
                }
                proj_vec[i] = acc * model_scale;
            }
            ck_gemma4_rmsnorm_tmp(
                proj_vec, args->per_layer_proj_norm, proj_normed,
                args->per_layer_dim, args->eps);
            for (int i = 0; i < args->per_layer_dim; ++i) {
                dst[i] = (token_vec[i] + proj_normed[i]) * mix_scale;
            }
        }
    }
}

static void ck_gemma4_prepare_parallel(
    int tokens, ck_range_fn_t fn, ck_gemma4_prepare_args_t *args)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    int active = pool ? ck_threadpool_n_threads(pool) : 1;
    const char *disabled = getenv("CK_DISABLE_GEMMA4_PREPARE_PARALLEL");
    if (disabled && disabled[0] && strcmp(disabled, "0") != 0) active = 1;
    if (active > tokens) active = tokens;
    ck_threadpool_parallel_for_n(pool, active, 0, tokens, 1, fn, args);
}

void gemma4_per_layer_prepare_forward(float *per_layer_input,
                                      const float *hidden,
                                      const int32_t *token_ids,
                                      const void *per_layer_token_emb,
                                      const uint16_t *per_layer_model_proj,
                                      const float *per_layer_proj_norm,
                                      int tokens,
                                      int num_layers,
                                      int embed_dim,
                                      int per_layer_dim,
                                      int vocab_size,
                                      float eps)
{
    if (!per_layer_input || !hidden || !token_ids || !per_layer_token_emb ||
        !per_layer_model_proj || !per_layer_proj_norm || tokens <= 0 ||
        num_layers <= 0 || embed_dim <= 0 || per_layer_dim != QK_K || vocab_size <= 0) {
        return;
    }

    ck_gemma4_prepare_args_t args = {
        .per_layer_input = per_layer_input,
        .hidden = hidden,
        .token_ids = token_ids,
        .per_layer_token_emb = per_layer_token_emb,
        .per_layer_model_proj = per_layer_model_proj,
        .per_layer_proj_norm = per_layer_proj_norm,
        .num_layers = num_layers,
        .embed_dim = embed_dim,
        .per_layer_dim = per_layer_dim,
        .vocab_size = vocab_size,
        .eps = eps,
    };
    ck_gemma4_prepare_parallel(tokens, ck_gemma4_prepare_q5_range, &args);
}


void gemma4_per_layer_prepare_bf16_forward(float *per_layer_input,
                                           const float *hidden,
                                           const int32_t *token_ids,
                                           const uint16_t *per_layer_token_emb,
                                           const uint16_t *per_layer_model_proj,
                                           const float *per_layer_proj_norm,
                                           int tokens,
                                           int num_layers,
                                           int embed_dim,
                                           int per_layer_dim,
                                           int vocab_size,
                                           float eps)
{
    if (!per_layer_input || !hidden || !token_ids || !per_layer_token_emb ||
        !per_layer_model_proj || !per_layer_proj_norm || tokens <= 0 ||
        num_layers <= 0 || embed_dim <= 0 || per_layer_dim <= 0 || vocab_size <= 0) {
        return;
    }

    if (per_layer_dim > QK_K) {
        return;
    }
    ck_gemma4_prepare_args_t args = {
        .per_layer_input = per_layer_input,
        .hidden = hidden,
        .token_ids = token_ids,
        .per_layer_token_emb = per_layer_token_emb,
        .per_layer_model_proj = per_layer_model_proj,
        .per_layer_proj_norm = per_layer_proj_norm,
        .num_layers = num_layers,
        .embed_dim = embed_dim,
        .per_layer_dim = per_layer_dim,
        .vocab_size = vocab_size,
        .eps = eps,
    };
    ck_gemma4_prepare_parallel(tokens, ck_gemma4_prepare_bf16_range, &args);
}

typedef struct {
    float *hidden;
    const float *per_layer_input;
    const float *inp_gate;
    const float *proj;
    const float *post_norm;
    const float *out_scale;
    int layer;
    int num_layers;
    int embed_dim;
    int per_layer_dim;
    float eps;
} ck_gemma4_embed_args_t;

static void ck_gemma4_embed_range(int begin, int end, void *opaque)
{
    const ck_gemma4_embed_args_t *args =
        (const ck_gemma4_embed_args_t *)opaque;
    float gate_vec[QK_K];
    float branch[4096];
    float branch_normed[4096];
    for (int t = begin; t < end; ++t) {
        float *h = args->hidden + (size_t)t * (size_t)args->embed_dim;
        const float *inp_vec = args->per_layer_input +
            ((size_t)t * (size_t)args->num_layers + (size_t)args->layer) *
                (size_t)args->per_layer_dim;

        for (int i = 0; i < args->per_layer_dim; ++i) {
            const float *row = args->inp_gate +
                (size_t)i * (size_t)args->embed_dim;
            float acc = 0.0f;
            for (int j = 0; j < args->embed_dim; ++j) {
                acc += row[j] * h[j];
            }
            gate_vec[i] = ck_gemma4_gelu(acc) * inp_vec[i];
        }

        for (int j = 0; j < args->embed_dim; ++j) {
            const float *row = args->proj +
                (size_t)j * (size_t)args->per_layer_dim;
            float acc = 0.0f;
            for (int i = 0; i < args->per_layer_dim; ++i) {
                acc += row[i] * gate_vec[i];
            }
            branch[j] = acc;
        }
        ck_gemma4_rmsnorm_tmp(
            branch, args->post_norm, branch_normed,
            args->embed_dim, args->eps);
        const float layer_scale = args->out_scale ? args->out_scale[0] : 1.0f;
        for (int j = 0; j < args->embed_dim; ++j) {
            h[j] = (h[j] + branch_normed[j]) * layer_scale;
        }
    }
}

void gemma4_per_layer_embed_forward(float *hidden,
                                    const float *per_layer_input,
                                    const float *inp_gate,
                                    const float *proj,
                                    const float *post_norm,
                                    const float *out_scale,
                                    int tokens,
                                    int layer,
                                    int num_layers,
                                    int embed_dim,
                                    int per_layer_dim,
                                    float eps)
{
    if (!hidden || !per_layer_input || !inp_gate || !proj || !post_norm ||
        tokens <= 0 || layer < 0 || layer >= num_layers || embed_dim <= 0 ||
        per_layer_dim != QK_K || embed_dim > 4096) {
        return;
    }

    ck_gemma4_embed_args_t args = {
        .hidden = hidden,
        .per_layer_input = per_layer_input,
        .inp_gate = inp_gate,
        .proj = proj,
        .post_norm = post_norm,
        .out_scale = out_scale,
        .layer = layer,
        .num_layers = num_layers,
        .embed_dim = embed_dim,
        .per_layer_dim = per_layer_dim,
        .eps = eps,
    };
    ck_threadpool_t *pool = ck_threadpool_global();
    int active = pool ? ck_threadpool_n_threads(pool) : 1;
    const char *disabled = getenv("CK_DISABLE_GEMMA4_EMBED_PARALLEL");
    if (disabled && disabled[0] && strcmp(disabled, "0") != 0) active = 1;
    if (active > tokens) active = tokens;
    ck_threadpool_parallel_for_n(
        pool, active, 0, tokens, 1, ck_gemma4_embed_range, &args);
}

void assistant_layer_scale_forward(float *hidden,
                                   const float *scale,
                                   int tokens,
                                   int embed_dim)
{
    if (!hidden || !scale || tokens <= 0 || embed_dim <= 0) {
        return;
    }

    const float s = scale[0];
    const size_t n = (size_t)tokens * (size_t)embed_dim;
    for (size_t i = 0; i < n; ++i) {
        hidden[i] *= s;
    }
}

void gemma4_final_logit_softcap_forward(float *logits,
                                        int tokens,
                                        int vocab_size,
                                        float cap)
{
    if (!logits || tokens <= 0 || vocab_size <= 0 || cap <= 0.0f) {
        return;
    }
    const float inv_cap = 1.0f / cap;
    const size_t total = (size_t)tokens * (size_t)vocab_size;
    for (size_t i = 0; i < total; ++i) {
        logits[i] = tanhf(logits[i] * inv_cap) * cap;
    }
}
