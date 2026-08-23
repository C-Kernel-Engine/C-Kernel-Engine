/**
 * @file deepseek_kernels.c
 * @brief Scalar reference kernels for DeepSeek-style research ops.
 *
 * These kernels intentionally prioritize explicit math contracts over speed.
 * They are used to pin PyTorch parity before adding SIMD/threaded variants.
 */

#include <math.h>
#include <float.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "bf16_utils.h"

static inline size_t ds_mhc_idx(int t, int s, int d, int n_streams, int dim)
{
    return ((size_t)t * (size_t)n_streams + (size_t)s) * (size_t)dim + (size_t)d;
}

static inline size_t ds_mix_idx(int t, int out_s, int in_s, int n_streams)
{
    return ((size_t)t * (size_t)n_streams + (size_t)out_s) * (size_t)n_streams + (size_t)in_s;
}

void deepseek_mhc_mix_f32(const float *streams,
                          const float *mix,
                          float *out,
                          int tokens,
                          int n_streams,
                          int dim)
{
    if (!streams || !mix || !out || tokens <= 0 || n_streams <= 0 || dim <= 0) return;

    for (int t = 0; t < tokens; ++t) {
        for (int os = 0; os < n_streams; ++os) {
            for (int d = 0; d < dim; ++d) {
                float acc = 0.0f;
                for (int is = 0; is < n_streams; ++is) {
                    acc += mix[ds_mix_idx(t, os, is, n_streams)] *
                           streams[ds_mhc_idx(t, is, d, n_streams, dim)];
                }
                out[ds_mhc_idx(t, os, d, n_streams, dim)] = acc;
            }
        }
    }
}

void deepseek_mhc_mix_backward_f32(const float *d_out,
                                   const float *streams,
                                   const float *mix,
                                   float *d_streams,
                                   float *d_mix,
                                   int tokens,
                                   int n_streams,
                                   int dim)
{
    if (!d_out || !streams || !mix || !d_streams || !d_mix ||
        tokens <= 0 || n_streams <= 0 || dim <= 0) return;

    const size_t stream_count = (size_t)tokens * (size_t)n_streams * (size_t)dim;
    const size_t mix_count = (size_t)tokens * (size_t)n_streams * (size_t)n_streams;
    for (size_t i = 0; i < stream_count; ++i) d_streams[i] = 0.0f;
    for (size_t i = 0; i < mix_count; ++i) d_mix[i] = 0.0f;

    for (int t = 0; t < tokens; ++t) {
        for (int os = 0; os < n_streams; ++os) {
            for (int is = 0; is < n_streams; ++is) {
                float d_mix_acc = 0.0f;
                const float m = mix[ds_mix_idx(t, os, is, n_streams)];
                for (int d = 0; d < dim; ++d) {
                    const float go = d_out[ds_mhc_idx(t, os, d, n_streams, dim)];
                    d_streams[ds_mhc_idx(t, is, d, n_streams, dim)] += m * go;
                    d_mix_acc += go * streams[ds_mhc_idx(t, is, d, n_streams, dim)];
                }
                d_mix[ds_mix_idx(t, os, is, n_streams)] = d_mix_acc;
            }
        }
    }
}

void deepseek_dsa_topk_softmax_f32(const float *scores,
                                   int *indices,
                                   float *weights,
                                   int tokens,
                                   int heads,
                                   int key_count,
                                   int top_k)
{
    if (!scores || !indices || !weights || tokens <= 0 || heads <= 0 ||
        key_count <= 0 || top_k <= 0) return;

    if (top_k > key_count) top_k = key_count;

    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < heads; ++h) {
            const float *row = scores + ((size_t)t * (size_t)heads + (size_t)h) * (size_t)key_count;
            int *idx = indices + ((size_t)t * (size_t)heads + (size_t)h) * (size_t)top_k;
            float *w = weights + ((size_t)t * (size_t)heads + (size_t)h) * (size_t)top_k;

            for (int k = 0; k < top_k; ++k) {
                idx[k] = -1;
                w[k] = -FLT_MAX;
            }

            for (int j = 0; j < key_count; ++j) {
                const float v = row[j];
                int pos = -1;
                for (int k = 0; k < top_k; ++k) {
                    if (idx[k] < 0 || v > w[k] || (v == w[k] && j < idx[k])) {
                        pos = k;
                        break;
                    }
                }
                if (pos >= 0) {
                    for (int k = top_k - 1; k > pos; --k) {
                        idx[k] = idx[k - 1];
                        w[k] = w[k - 1];
                    }
                    idx[pos] = j;
                    w[pos] = v;
                }
            }

            float max_v = w[0];
            for (int k = 1; k < top_k; ++k) if (w[k] > max_v) max_v = w[k];
            float sum = 0.0f;
            for (int k = 0; k < top_k; ++k) {
                w[k] = expf(w[k] - max_v);
                sum += w[k];
            }
            if (sum > 0.0f) {
                const float inv = 1.0f / sum;
                for (int k = 0; k < top_k; ++k) w[k] *= inv;
            }
        }
    }
}


extern void topk_softmax_backward_f32(const int *indices,
                                      const float *weights,
                                      const float *d_weights,
                                      float *d_scores,
                                      int num_tokens,
                                      int n_experts_or_keys,
                                      int k);

void deepseek_dsa_topk_softmax_backward_f32(const int *indices,
                                            const float *weights,
                                            const float *d_weights,
                                            float *d_scores,
                                            int tokens,
                                            int heads,
                                            int key_count,
                                            int top_k)
{
    if (!indices || !weights || !d_weights || !d_scores ||
        tokens <= 0 || heads <= 0 || key_count <= 0 || top_k <= 0) return;

    topk_softmax_backward_f32(indices,
                              weights,
                              d_weights,
                              d_scores,
                              tokens * heads,
                              key_count,
                              top_k);
}


static inline size_t ds_mla_tok_idx(int t, int d, int dim)
{
    return (size_t)t * (size_t)dim + (size_t)d;
}

static inline size_t ds_mla_thd_idx(int t, int h, int d, int heads, int dim)
{
    return ((size_t)t * (size_t)heads + (size_t)h) * (size_t)dim + (size_t)d;
}

void deepseek_mla_kv_decompress_f32(const float *compressed_kv,
                                    const float *kv_b_proj,
                                    float *k_nope,
                                    float *value,
                                    int tokens,
                                    int heads,
                                    int kv_lora_rank,
                                    int qk_nope_dim,
                                    int v_dim)
{
    if (!compressed_kv || !kv_b_proj || !k_nope || !value ||
        tokens <= 0 || heads <= 0 || kv_lora_rank <= 0 || qk_nope_dim <= 0 || v_dim <= 0) {
        return;
    }

    const int out_per_head = qk_nope_dim + v_dim;
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < heads; ++h) {
            for (int d = 0; d < qk_nope_dim; ++d) {
                const int out_col = h * out_per_head + d;
                float acc = 0.0f;
                for (int r = 0; r < kv_lora_rank; ++r) {
                    acc += kv_b_proj[(size_t)out_col * (size_t)kv_lora_rank + (size_t)r] *
                           compressed_kv[ds_mla_tok_idx(t, r, kv_lora_rank)];
                }
                k_nope[ds_mla_thd_idx(t, h, d, heads, qk_nope_dim)] = acc;
            }
            for (int d = 0; d < v_dim; ++d) {
                const int out_col = h * out_per_head + qk_nope_dim + d;
                float acc = 0.0f;
                for (int r = 0; r < kv_lora_rank; ++r) {
                    acc += kv_b_proj[(size_t)out_col * (size_t)kv_lora_rank + (size_t)r] *
                           compressed_kv[ds_mla_tok_idx(t, r, kv_lora_rank)];
                }
                value[ds_mla_thd_idx(t, h, d, heads, v_dim)] = acc;
            }
        }
    }
}

void deepseek_mla_kv_decompress_bf16(const float *compressed_kv,
                                     const uint16_t *kv_b_proj,
                                     float *k_nope,
                                     float *value,
                                     int tokens,
                                     int heads,
                                     int kv_lora_rank,
                                     int qk_nope_dim,
                                     int v_dim)
{
    if (!compressed_kv || !kv_b_proj || !k_nope || !value ||
        tokens <= 0 || heads <= 0 || kv_lora_rank <= 0 || qk_nope_dim <= 0 || v_dim <= 0) {
        return;
    }

    const int out_per_head = qk_nope_dim + v_dim;
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < heads; ++h) {
            for (int d = 0; d < qk_nope_dim; ++d) {
                const int out_col = h * out_per_head + d;
                float acc = 0.0f;
                for (int r = 0; r < kv_lora_rank; ++r) {
                    acc += bf16_to_float(kv_b_proj[(size_t)out_col * (size_t)kv_lora_rank + (size_t)r]) *
                           compressed_kv[ds_mla_tok_idx(t, r, kv_lora_rank)];
                }
                k_nope[ds_mla_thd_idx(t, h, d, heads, qk_nope_dim)] =
                    bf16_to_float(float_to_bf16(acc));
            }
            for (int d = 0; d < v_dim; ++d) {
                const int out_col = h * out_per_head + qk_nope_dim + d;
                float acc = 0.0f;
                for (int r = 0; r < kv_lora_rank; ++r) {
                    acc += bf16_to_float(kv_b_proj[(size_t)out_col * (size_t)kv_lora_rank + (size_t)r]) *
                           compressed_kv[ds_mla_tok_idx(t, r, kv_lora_rank)];
                }
                value[ds_mla_thd_idx(t, h, d, heads, v_dim)] =
                    bf16_to_float(float_to_bf16(acc));
            }
        }
    }
}

/* Model-agnostic implementation of the DeepSeek-V3 interleaved MLA rotary
 * layout.  Kimi and Instella reuse this exact tensor transform; the ds_ prefix
 * records its origin, not a model-selection restriction. */
static void ds_mla_apply_kimi_rope(const float *src,
                                   float *dst,
                                   const float *cos_row,
                                   const float *sin_row,
                                   int dim)
{
    const int half = dim / 2;
    for (int i = 0; i < half; ++i) {
        const float x_first = src[2 * i];
        const float x_second = src[2 * i + 1];
        const float c = cos_row[i];
        const float s = sin_row[i];
        dst[i] = x_first * c - x_second * s;
        dst[half + i] = x_second * c + x_first * s;
    }
}

void deepseek_mla_partial_rope_concat_f32(const float *q_nope,
                                          const float *q_pe,
                                          const float *k_nope,
                                          const float *k_pe,
                                          const float *cos,
                                          const float *sin,
                                          float *query,
                                          float *key,
                                          int tokens,
                                          int heads,
                                          int qk_nope_dim,
                                          int qk_rope_dim)
{
    if (!q_nope || !q_pe || !k_nope || !k_pe || !cos || !sin || !query || !key ||
        tokens <= 0 || heads <= 0 || qk_nope_dim <= 0 || qk_rope_dim <= 0 || (qk_rope_dim % 2) != 0) {
        return;
    }

    const int q_head_dim = qk_nope_dim + qk_rope_dim;
    for (int t = 0; t < tokens; ++t) {
        const float *cos_row = cos + (size_t)t * (size_t)(qk_rope_dim / 2);
        const float *sin_row = sin + (size_t)t * (size_t)(qk_rope_dim / 2);
        for (int h = 0; h < heads; ++h) {
            float *q_out = query + ds_mla_thd_idx(t, h, 0, heads, q_head_dim);
            float *k_out = key + ds_mla_thd_idx(t, h, 0, heads, q_head_dim);
            const float *qn = q_nope + ds_mla_thd_idx(t, h, 0, heads, qk_nope_dim);
            const float *qp = q_pe + ds_mla_thd_idx(t, h, 0, heads, qk_rope_dim);
            const float *kn = k_nope + ds_mla_thd_idx(t, h, 0, heads, qk_nope_dim);
            const float *kp = k_pe + ds_mla_tok_idx(t, 0, qk_rope_dim);
            for (int d = 0; d < qk_nope_dim; ++d) {
                q_out[d] = qn[d];
                k_out[d] = kn[d];
            }
            ds_mla_apply_kimi_rope(qp, q_out + qk_nope_dim, cos_row, sin_row, qk_rope_dim);
            ds_mla_apply_kimi_rope(kp, k_out + qk_nope_dim, cos_row, sin_row, qk_rope_dim);
        }
    }
}

void deepseek_mla_partial_rope_concat_packed_f32(const float *q_packed,
                                             const float *k_nope,
                                             const float *kv_a_packed,
                                             const float *cos,
                                             const float *sin,
                                             float *query,
                                             float *key,
                                             int tokens,
                                             int heads,
                                             int kv_lora_rank,
                                             int qk_nope_dim,
                                             int qk_rope_dim)
{
    if (!q_packed || !k_nope || !kv_a_packed || !cos || !sin || !query || !key ||
        tokens <= 0 || heads <= 0 || kv_lora_rank <= 0 || qk_nope_dim <= 0 ||
        qk_rope_dim <= 0 || qk_rope_dim > 256 || (qk_rope_dim % 2) != 0) {
        return;
    }

    const int q_head_dim = qk_nope_dim + qk_rope_dim;
    const int kv_a_dim = kv_lora_rank + qk_rope_dim;
    const int half = qk_rope_dim / 2;
    if (qk_rope_dim > 256) {
        return;
    }
    for (int t = 0; t < tokens; ++t) {
        const float *cos_row = cos + (size_t)t * (size_t)half;
        const float *sin_row = sin + (size_t)t * (size_t)half;
        const float *kp = kv_a_packed + (size_t)t * (size_t)kv_a_dim + (size_t)kv_lora_rank;
        for (int h = 0; h < heads; ++h) {
            const float *q_in = q_packed + ds_mla_thd_idx(t, h, 0, heads, q_head_dim);
            const float *kn = k_nope + ds_mla_thd_idx(t, h, 0, heads, qk_nope_dim);
            float *q_out = query + ds_mla_thd_idx(t, h, 0, heads, q_head_dim);
            float *k_out = key + ds_mla_thd_idx(t, h, 0, heads, q_head_dim);

            for (int d = 0; d < qk_nope_dim; ++d) {
                q_out[d] = q_in[d];
                k_out[d] = kn[d];
            }

            const float *qp = q_in + qk_nope_dim;
            float q_pe_tmp[256];
            if (qk_rope_dim > (int)(sizeof(q_pe_tmp) / sizeof(q_pe_tmp[0]))) {
                return;
            }
            for (int i = 0; i < qk_rope_dim; ++i) q_pe_tmp[i] = qp[i];
            for (int i = 0; i < half; ++i) {
                const float q_first = q_pe_tmp[2 * i];
                const float q_second = q_pe_tmp[2 * i + 1];
                const float k_first = kp[2 * i];
                const float k_second = kp[2 * i + 1];
                const float c = cos_row[i];
                const float ss = sin_row[i];
                q_out[qk_nope_dim + i] = q_first * c - q_second * ss;
                q_out[qk_nope_dim + half + i] = q_second * c + q_first * ss;
                k_out[qk_nope_dim + i] = k_first * c - k_second * ss;
                k_out[qk_nope_dim + half + i] = k_second * c + k_first * ss;
            }
        }
    }
}

static inline float ds_mla_bf16_round(float value)
{
    return bf16_to_float(float_to_bf16(value));
}

void deepseek_mla_partial_rope_concat_packed_bf16_storage(
    const float *q_packed,
    const float *k_nope,
    const float *kv_a_packed,
    const float *cos,
    const float *sin,
    float *query,
    float *key,
    int tokens,
    int heads,
    int kv_lora_rank,
    int qk_nope_dim,
    int qk_rope_dim)
{
    if (!q_packed || !k_nope || !kv_a_packed || !cos || !sin || !query || !key ||
        tokens <= 0 || heads <= 0 || kv_lora_rank <= 0 || qk_nope_dim <= 0 ||
        qk_rope_dim <= 0 || (qk_rope_dim % 2) != 0) {
        return;
    }

    const int q_head_dim = qk_nope_dim + qk_rope_dim;
    const int kv_a_dim = kv_lora_rank + qk_rope_dim;
    const int half = qk_rope_dim / 2;
    for (int t = 0; t < tokens; ++t) {
        const float *cos_row = cos + (size_t)t * (size_t)half;
        const float *sin_row = sin + (size_t)t * (size_t)half;
        const float *kp = kv_a_packed + (size_t)t * (size_t)kv_a_dim + (size_t)kv_lora_rank;
        for (int h = 0; h < heads; ++h) {
            const float *q_in = q_packed + ds_mla_thd_idx(t, h, 0, heads, q_head_dim);
            const float *kn = k_nope + ds_mla_thd_idx(t, h, 0, heads, qk_nope_dim);
            float *q_out = query + ds_mla_thd_idx(t, h, 0, heads, q_head_dim);
            float *k_out = key + ds_mla_thd_idx(t, h, 0, heads, q_head_dim);

            for (int d = 0; d < qk_nope_dim; ++d) {
                q_out[d] = ds_mla_bf16_round(q_in[d]);
                k_out[d] = ds_mla_bf16_round(kn[d]);
            }

            float q_pe_tmp[256];
            for (int i = 0; i < qk_rope_dim; ++i) {
                q_pe_tmp[i] = ds_mla_bf16_round(q_in[qk_nope_dim + i]);
            }
            for (int i = 0; i < half; ++i) {
                const float q_first = q_pe_tmp[2 * i];
                const float q_second = q_pe_tmp[2 * i + 1];
                const float k_first = ds_mla_bf16_round(kp[2 * i]);
                const float k_second = ds_mla_bf16_round(kp[2 * i + 1]);
                const float c = ds_mla_bf16_round(cos_row[i]);
                const float s = ds_mla_bf16_round(sin_row[i]);

                const float q_first_cos = ds_mla_bf16_round(q_first * c);
                const float q_second_sin = ds_mla_bf16_round(q_second * s);
                const float q_second_cos = ds_mla_bf16_round(q_second * c);
                const float q_first_sin = ds_mla_bf16_round(q_first * s);
                const float k_first_cos = ds_mla_bf16_round(k_first * c);
                const float k_second_sin = ds_mla_bf16_round(k_second * s);
                const float k_second_cos = ds_mla_bf16_round(k_second * c);
                const float k_first_sin = ds_mla_bf16_round(k_first * s);

                q_out[qk_nope_dim + i] = ds_mla_bf16_round(q_first_cos - q_second_sin);
                q_out[qk_nope_dim + half + i] = ds_mla_bf16_round(q_second_cos + q_first_sin);
                k_out[qk_nope_dim + i] = ds_mla_bf16_round(k_first_cos - k_second_sin);
                k_out[qk_nope_dim + half + i] = ds_mla_bf16_round(k_second_cos + k_first_sin);
            }
        }
    }
}

static inline size_t ds_qkv_idx(int token, int head, int d, int heads, int dim)
{
    return ((size_t)token * (size_t)heads + (size_t)head) * (size_t)dim + (size_t)d;
}

static void ds_softmax(float *x, int n)
{
    if (n <= 0) return;
    float max_v = x[0];
    for (int i = 1; i < n; ++i) if (x[i] > max_v) max_v = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = expf(x[i] - max_v);
        sum += x[i];
    }
    if (sum > 0.0f) {
        const float inv = 1.0f / sum;
        for (int i = 0; i < n; ++i) x[i] *= inv;
    }
}

void deepseek_csa_attention_f32(const float *q,
                                const float *k,
                                const float *v,
                                const int *indices,
                                float *out,
                                float *attn,
                                int query_tokens,
                                int key_tokens,
                                int heads,
                                int dim,
                                int top_k,
                                float scale)
{
    if (!q || !k || !v || !indices || !out || query_tokens <= 0 || key_tokens <= 0 ||
        heads <= 0 || dim <= 0 || top_k <= 0) return;

    for (int tq = 0; tq < query_tokens; ++tq) {
        for (int h = 0; h < heads; ++h) {
            float local_scores[top_k];
            int valid = 0;
            for (int j = 0; j < top_k; ++j) {
                const int tk = indices[((size_t)tq * (size_t)heads + (size_t)h) * (size_t)top_k + (size_t)j];
                if (tk < 0 || tk >= key_tokens) {
                    local_scores[j] = -FLT_MAX;
                    continue;
                }
                float dot = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    dot += q[ds_qkv_idx(tq, h, d, heads, dim)] * k[ds_qkv_idx(tk, h, d, heads, dim)];
                }
                local_scores[j] = dot * scale;
                valid++;
            }

            float *out_row = out + ds_qkv_idx(tq, h, 0, heads, dim);
            for (int d = 0; d < dim; ++d) out_row[d] = 0.0f;
            if (valid == 0) {
                if (attn) {
                    for (int j = 0; j < top_k; ++j) {
                        attn[((size_t)tq * (size_t)heads + (size_t)h) * (size_t)top_k + (size_t)j] = 0.0f;
                    }
                }
                continue;
            }

            ds_softmax(local_scores, top_k);
            for (int j = 0; j < top_k; ++j) {
                const int tk = indices[((size_t)tq * (size_t)heads + (size_t)h) * (size_t)top_k + (size_t)j];
                const float a = (tk >= 0 && tk < key_tokens) ? local_scores[j] : 0.0f;
                if (attn) attn[((size_t)tq * (size_t)heads + (size_t)h) * (size_t)top_k + (size_t)j] = a;
                if (a == 0.0f) continue;
                for (int d = 0; d < dim; ++d) {
                    out_row[d] += a * v[ds_qkv_idx(tk, h, d, heads, dim)];
                }
            }
        }
    }
}

void deepseek_csa_attention_backward_f32(const float *d_out,
                                         const float *q,
                                         const float *k,
                                         const float *v,
                                         const int *indices,
                                         const float *attn,
                                         float *d_q,
                                         float *d_k,
                                         float *d_v,
                                         int query_tokens,
                                         int key_tokens,
                                         int heads,
                                         int dim,
                                         int top_k,
                                         float scale)
{
    if (!d_out || !q || !k || !v || !indices || !attn || !d_q || !d_k || !d_v ||
        query_tokens <= 0 || key_tokens <= 0 || heads <= 0 || dim <= 0 || top_k <= 0) return;

    const size_t q_count = (size_t)query_tokens * (size_t)heads * (size_t)dim;
    const size_t kv_count = (size_t)key_tokens * (size_t)heads * (size_t)dim;
    for (size_t i = 0; i < q_count; ++i) d_q[i] = 0.0f;
    for (size_t i = 0; i < kv_count; ++i) {
        d_k[i] = 0.0f;
        d_v[i] = 0.0f;
    }

    for (int tq = 0; tq < query_tokens; ++tq) {
        for (int h = 0; h < heads; ++h) {
            float d_attn[top_k];
            float attn_dot = 0.0f;
            for (int j = 0; j < top_k; ++j) {
                const int tk = indices[((size_t)tq * (size_t)heads + (size_t)h) * (size_t)top_k + (size_t)j];
                float da = 0.0f;
                if (tk >= 0 && tk < key_tokens) {
                    const float a = attn[((size_t)tq * (size_t)heads + (size_t)h) * (size_t)top_k + (size_t)j];
                    for (int d = 0; d < dim; ++d) {
                        const float go = d_out[ds_qkv_idx(tq, h, d, heads, dim)];
                        da += go * v[ds_qkv_idx(tk, h, d, heads, dim)];
                        d_v[ds_qkv_idx(tk, h, d, heads, dim)] += a * go;
                    }
                    attn_dot += a * da;
                }
                d_attn[j] = da;
            }

            for (int j = 0; j < top_k; ++j) {
                const int tk = indices[((size_t)tq * (size_t)heads + (size_t)h) * (size_t)top_k + (size_t)j];
                if (tk < 0 || tk >= key_tokens) continue;
                const float a = attn[((size_t)tq * (size_t)heads + (size_t)h) * (size_t)top_k + (size_t)j];
                const float d_score = a * (d_attn[j] - attn_dot);
                for (int d = 0; d < dim; ++d) {
                    const float qv = q[ds_qkv_idx(tq, h, d, heads, dim)];
                    const float kv = k[ds_qkv_idx(tk, h, d, heads, dim)];
                    d_q[ds_qkv_idx(tq, h, d, heads, dim)] += scale * d_score * kv;
                    d_k[ds_qkv_idx(tk, h, d, heads, dim)] += scale * d_score * qv;
                }
            }
        }
    }
}

void deepseek_hybrid_attention_f32(const float *q,
                                   const float *k,
                                   const float *v,
                                   const int *indices,
                                   float *out,
                                   float *attn,
                                   int query_tokens,
                                   int key_tokens,
                                   int heads,
                                   int dim,
                                   int top_k,
                                   float scale,
                                   int mode)
{
    if (mode != 0) {
        deepseek_csa_attention_f32(q, k, v, indices, out, attn,
                                   query_tokens, key_tokens, heads, dim, top_k, scale);
        return;
    }

    int dense_indices[query_tokens * heads * key_tokens];
    for (int tq = 0; tq < query_tokens; ++tq) {
        for (int h = 0; h < heads; ++h) {
            for (int tk = 0; tk < key_tokens; ++tk) {
                dense_indices[((size_t)tq * (size_t)heads + (size_t)h) * (size_t)key_tokens + (size_t)tk] = tk;
            }
        }
    }
    deepseek_csa_attention_f32(q, k, v, dense_indices, out, attn,
                               query_tokens, key_tokens, heads, dim, key_tokens, scale);
}


void deepseek_mla_attention_f32_workspace(const float *q,
                                          const float *k,
                                          const float *v,
                                          float *output,
                                          int num_heads,
                                          int num_kv_heads,
                                          int num_tokens,
                                          int qk_head_dim,
                                          int v_head_dim,
                                          float scale,
                                          float *scores,
                                          size_t scores_bytes)
{
    if (!q || !k || !v || !output || num_heads <= 0 || num_kv_heads <= 0 ||
        num_tokens <= 0 || qk_head_dim <= 0 || v_head_dim <= 0 ||
        !isfinite(scale) || scale <= 0.0f) {
        return;
    }
    if ((size_t)num_tokens > SIZE_MAX / sizeof(float) || !scores ||
        scores_bytes < (size_t)num_tokens * sizeof(float)) {
        return;
    }

    for (int t = 0; t < num_tokens; ++t) {
        for (int h = 0; h < num_heads; ++h) {
            const int kv_h = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
            const float *q_vec = q + ds_mla_thd_idx(t, h, 0, num_heads, qk_head_dim);

            float max_score = -FLT_MAX;
            for (int j = 0; j <= t; ++j) {
                const float *k_vec = k + ds_mla_thd_idx(j, kv_h, 0, num_kv_heads, qk_head_dim);
                float dot = 0.0f;
                for (int d = 0; d < qk_head_dim; ++d) {
                    dot += q_vec[d] * k_vec[d];
                }
                const float score = dot * scale;
                scores[j] = score;
                if (score > max_score) max_score = score;
            }

            float sum = 0.0f;
            for (int j = 0; j <= t; ++j) {
                const float e = expf(scores[j] - max_score);
                scores[j] = e;
                sum += e;
            }
            const float inv_sum = sum > 0.0f ? (1.0f / sum) : 0.0f;
            float *out = output + ((size_t)t * (size_t)num_heads + (size_t)h) * (size_t)v_head_dim;
            for (int d = 0; d < v_head_dim; ++d) out[d] = 0.0f;
            for (int j = 0; j <= t; ++j) {
                const float w = scores[j] * inv_sum;
                const float *v_vec = v + ds_mla_thd_idx(j, kv_h, 0, num_kv_heads, v_head_dim);
                for (int d = 0; d < v_head_dim; ++d) {
                    out[d] += w * v_vec[d];
                }
            }
        }
    }

}

void deepseek_mla_attention_f32(const float *q,
                                const float *k,
                                const float *v,
                                float *output,
                                int num_heads,
                                int num_kv_heads,
                                int num_tokens,
                                int qk_head_dim,
                                int v_head_dim)
{
    if (num_tokens <= 0 || (size_t)num_tokens > SIZE_MAX / sizeof(float)) return;
    const size_t scores_bytes = (size_t)num_tokens * sizeof(float);
    float *scores = (float *)malloc(scores_bytes);
    if (!scores) return;
    deepseek_mla_attention_f32_workspace(
        q, k, v, output, num_heads, num_kv_heads, num_tokens,
        qk_head_dim, v_head_dim, 1.0f / sqrtf((float)qk_head_dim),
        scores, scores_bytes);
    free(scores);
}

void deepseek_mla_kv_cache_batch_store_f32(float *k_cache,
                                           float *v_cache,
                                           const float *k,
                                           const float *v,
                                           int num_tokens,
                                           int num_kv_heads,
                                           int qk_head_dim,
                                           int v_head_dim,
                                           int max_seq_len,
                                           int cache_stride)
{
    if (!k_cache || !v_cache || !k || !v || num_tokens <= 0 ||
        num_kv_heads <= 0 || qk_head_dim <= 0 || v_head_dim <= 0 ||
        max_seq_len <= 0 || cache_stride <= 0) {
        return;
    }
    if (qk_head_dim > cache_stride || v_head_dim > cache_stride) {
        return;
    }
    if (num_tokens > max_seq_len) {
        num_tokens = max_seq_len;
    }

    for (int t = 0; t < num_tokens; ++t) {
        for (int h = 0; h < num_kv_heads; ++h) {
            const float *k_src = k + ((size_t)t * (size_t)num_kv_heads + (size_t)h) * (size_t)qk_head_dim;
            const float *v_src = v + ((size_t)t * (size_t)num_kv_heads + (size_t)h) * (size_t)v_head_dim;
            float *k_dst = k_cache + ((size_t)h * (size_t)max_seq_len + (size_t)t) * (size_t)cache_stride;
            float *v_dst = v_cache + ((size_t)h * (size_t)max_seq_len + (size_t)t) * (size_t)cache_stride;
            for (int d = 0; d < qk_head_dim; ++d) k_dst[d] = k_src[d];
            for (int d = qk_head_dim; d < cache_stride; ++d) k_dst[d] = 0.0f;
            for (int d = 0; d < v_head_dim; ++d) v_dst[d] = v_src[d];
            for (int d = v_head_dim; d < cache_stride; ++d) v_dst[d] = 0.0f;
        }
    }
}

void deepseek_mla_kv_cache_store_f32(float *k_cache,
                                     float *v_cache,
                                     const float *k,
                                     const float *v,
                                     int pos,
                                     int num_kv_heads,
                                     int qk_head_dim,
                                     int v_head_dim,
                                     int max_seq_len,
                                     int cache_stride)
{
    if (!k_cache || !v_cache || !k || !v || pos < 0 ||
        num_kv_heads <= 0 || qk_head_dim <= 0 || v_head_dim <= 0 ||
        max_seq_len <= 0 || cache_stride <= 0) {
        return;
    }
    if (pos >= max_seq_len || qk_head_dim > cache_stride || v_head_dim > cache_stride) {
        return;
    }

    for (int h = 0; h < num_kv_heads; ++h) {
        const float *k_src = k + ((size_t)h * (size_t)qk_head_dim);
        const float *v_src = v + ((size_t)h * (size_t)v_head_dim);
        float *k_dst = k_cache + ((size_t)h * (size_t)max_seq_len + (size_t)pos) * (size_t)cache_stride;
        float *v_dst = v_cache + ((size_t)h * (size_t)max_seq_len + (size_t)pos) * (size_t)cache_stride;

        for (int d = 0; d < qk_head_dim; ++d) {
            k_dst[d] = k_src[d];
        }
        for (int d = qk_head_dim; d < cache_stride; ++d) {
            k_dst[d] = 0.0f;
        }
        for (int d = 0; d < v_head_dim; ++d) {
            v_dst[d] = v_src[d];
        }
        for (int d = v_head_dim; d < cache_stride; ++d) {
            v_dst[d] = 0.0f;
        }
    }
}

void deepseek_mla_attention_decode_f32_workspace(const float *q,
                                                 const float *k_cache,
                                                 const float *v_cache,
                                                 float *output,
                                                 int num_heads,
                                                 int num_kv_heads,
                                                 int cache_len,
                                                 int qk_head_dim,
                                                 int v_head_dim,
                                                 int max_seq_len,
                                                 int cache_stride,
                                                 float scale,
                                                 float *scores,
                                                 size_t scores_bytes)
{
    if (!q || !k_cache || !v_cache || !output || num_heads <= 0 ||
        num_kv_heads <= 0 || cache_len <= 0 || qk_head_dim <= 0 ||
        v_head_dim <= 0 || max_seq_len <= 0 || cache_stride <= 0 ||
        !isfinite(scale) || scale <= 0.0f) {
        return;
    }
    if (qk_head_dim > cache_stride || v_head_dim > cache_stride) {
        return;
    }
    if ((size_t)cache_len > SIZE_MAX / sizeof(float) || !scores ||
        scores_bytes < (size_t)cache_len * sizeof(float)) {
        return;
    }

    for (int h = 0; h < num_heads; ++h) {
        const int kv_h = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
        const float *q_vec = q + (size_t)h * (size_t)qk_head_dim;

        float max_score = -FLT_MAX;
        for (int j = 0; j < cache_len; ++j) {
            const float *k_vec = k_cache + ((size_t)kv_h * (size_t)max_seq_len + (size_t)j) * (size_t)cache_stride;
            float dot = 0.0f;
            for (int d = 0; d < qk_head_dim; ++d) {
                dot += q_vec[d] * k_vec[d];
            }
            const float score = dot * scale;
            scores[j] = score;
            if (score > max_score) max_score = score;
        }

        float sum = 0.0f;
        for (int j = 0; j < cache_len; ++j) {
            const float e = expf(scores[j] - max_score);
            scores[j] = e;
            sum += e;
        }

        const float inv_sum = sum > 0.0f ? (1.0f / sum) : 0.0f;
        float *out = output + (size_t)h * (size_t)v_head_dim;
        for (int d = 0; d < v_head_dim; ++d) out[d] = 0.0f;
        for (int j = 0; j < cache_len; ++j) {
            const float w = scores[j] * inv_sum;
            const float *v_vec = v_cache + ((size_t)kv_h * (size_t)max_seq_len + (size_t)j) * (size_t)cache_stride;
            for (int d = 0; d < v_head_dim; ++d) {
                out[d] += w * v_vec[d];
            }
        }
    }

}

void deepseek_mla_attention_decode_f32(const float *q,
                                       const float *k_cache,
                                       const float *v_cache,
                                       float *output,
                                       int num_heads,
                                       int num_kv_heads,
                                       int cache_len,
                                       int qk_head_dim,
                                       int v_head_dim,
                                       int max_seq_len,
                                       int cache_stride)
{
    if (cache_len <= 0 || (size_t)cache_len > SIZE_MAX / sizeof(float)) return;
    const size_t scores_bytes = (size_t)cache_len * sizeof(float);
    float *scores = (float *)malloc(scores_bytes);
    if (!scores) return;
    deepseek_mla_attention_decode_f32_workspace(
        q, k_cache, v_cache, output, num_heads, num_kv_heads, cache_len,
        qk_head_dim, v_head_dim, max_seq_len, cache_stride,
        1.0f / sqrtf((float)qk_head_dim), scores, scores_bytes);
    free(scores);
}
