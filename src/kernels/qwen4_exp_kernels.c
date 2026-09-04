#include "ckernel_engine.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

static inline float qwen4_bf16_round(float value) {
    union { float f; uint32_t u; } bits = {value};
    const uint32_t bias = UINT32_C(0x7fff) + ((bits.u >> 16) & 1u);
    bits.u = (bits.u + bias) & UINT32_C(0xffff0000);
    return bits.f;
}

static inline float qwen4_bf16_load(const uint16_t *value) {
    union { uint32_t u; float f; } bits = {(uint32_t)(*value) << 16};
    return bits.f;
}

static int32_t qwen4_history_token(const int32_t *tokens,
                                   const float *state,
                                   int row,
                                   int shift,
                                   int state_len,
                                   int eos_token_id,
                                   int position) {
    if (shift == 0) return tokens[row];
    for (int step = 1; step <= shift; ++step) {
        const int index = row - step;
        const int32_t prior = index >= 0
            ? tokens[index]
            : (position == 0 ? eos_token_id : (int32_t)state[state_len + index]);
        if (prior == eos_token_id) return eos_token_id;
    }
    const int source = row - shift;
    return source >= 0
        ? tokens[source]
        : (position == 0 ? eos_token_id : (int32_t)state[state_len + source]);
}

static void qwen4_ple_ngram_embed_impl(const int32_t *token_ids,
                                const void *embedding,
                                const int64_t *layer_multipliers,
                                const int64_t *head_offsets,
                                const int64_t *head_vocab_sizes,
                                float *output,
                                const float *token_state_in,
                                float *token_state_out,
                                int rows,
                                int ngram_size,
                                int heads_per_ngram,
                                int head_dim,
                                int eos_token_id,
                                int position_offset,
                                int embedding_is_q5_0) {
    if (!token_ids || !embedding || !layer_multipliers || !head_offsets ||
        !head_vocab_sizes || !output || !token_state_in || !token_state_out || rows <= 0 ||
        ngram_size < 2 || heads_per_ngram <= 0 || head_dim <= 0) return;

    const int state_len = ngram_size - 1;
    const int embed_dim = state_len * heads_per_ngram * head_dim;
    for (int row = 0; row < rows; ++row) {
        float *out_row = output + (size_t)row * embed_dim;
        for (int ngram = 2; ngram <= ngram_size; ++ngram) {
            uint64_t mixed = (uint64_t)(int64_t)token_ids[row] *
                             (uint64_t)layer_multipliers[0];
            for (int gram_position = 1; gram_position < ngram; ++gram_position) {
                const int32_t token = qwen4_history_token(
                    token_ids, token_state_in, row, gram_position, state_len,
                    eos_token_id, position_offset);
                mixed ^= (uint64_t)(int64_t)token *
                         (uint64_t)layer_multipliers[gram_position];
            }
            const int head_begin = (ngram - 2) * heads_per_ngram;
            for (int local_head = 0; local_head < heads_per_ngram; ++local_head) {
                const int head = head_begin + local_head;
                const int64_t vocab = head_vocab_sizes[head];
                int64_t remainder = (int64_t)mixed % vocab;
                if (remainder < 0) remainder += vocab;
                const int64_t embedding_row = head_offsets[head] + remainder;
                float *target = out_row + (size_t)head * head_dim;
                if (embedding_is_q5_0) {
                    const size_t row_bytes = ck_dtype_row_bytes(CK_DT_Q5_0, (size_t)head_dim);
                    const uint8_t *source = (const uint8_t *)embedding +
                                            (size_t)embedding_row * row_bytes;
                    dequant_q5_0_row(source, target, (size_t)head_dim);
                } else {
                    const uint16_t *source = (const uint16_t *)embedding +
                                             (size_t)embedding_row * head_dim;
                    for (int col = 0; col < head_dim; ++col) {
                        target[col] = qwen4_bf16_load(source + col);
                    }
                }
            }
        }
    }

    if (rows >= state_len) {
        for (int index = 0; index < state_len; ++index) {
            token_state_out[index] = (float)token_ids[rows - state_len + index];
        }
    } else {
        if (position_offset == 0) {
            for (int index = 0; index < state_len - rows; ++index) {
                token_state_out[index] = (float)eos_token_id;
            }
        } else {
            memmove(token_state_out, token_state_in + rows,
                    (size_t)(state_len - rows) * sizeof(*token_state_out));
        }
        for (int index = 0; index < rows; ++index) {
            token_state_out[state_len - rows + index] = (float)token_ids[index];
        }
    }
}

void qwen4_ple_ngram_embed_bf16(const int32_t *token_ids,
                                const uint16_t *embedding,
                                const int64_t *layer_multipliers,
                                const int64_t *head_offsets,
                                const int64_t *head_vocab_sizes,
                                float *output,
                                const float *token_state_in,
                                float *token_state_out,
                                int rows,
                                int ngram_size,
                                int heads_per_ngram,
                                int head_dim,
                                int eos_token_id,
                                int position_offset) {
    qwen4_ple_ngram_embed_impl(
        token_ids, embedding, layer_multipliers, head_offsets,
        head_vocab_sizes, output, token_state_in, token_state_out, rows,
        ngram_size, heads_per_ngram, head_dim, eos_token_id, position_offset, 0);
}

void qwen4_ple_ngram_embed_q5_0(const int32_t *token_ids,
                                const void *embedding,
                                const int64_t *layer_multipliers,
                                const int64_t *head_offsets,
                                const int64_t *head_vocab_sizes,
                                float *output,
                                const float *token_state_in,
                                float *token_state_out,
                                int rows,
                                int ngram_size,
                                int heads_per_ngram,
                                int head_dim,
                                int eos_token_id,
                                int position_offset) {
    qwen4_ple_ngram_embed_impl(
        token_ids, embedding, layer_multipliers, head_offsets,
        head_vocab_sizes, output, token_state_in, token_state_out, rows,
        ngram_size, heads_per_ngram, head_dim, eos_token_id, position_offset, 1);
}

static void qwen4_group_rmsnorm_pytorch_bf16(const float *input,
                                              const float *weight,
                                              float *output,
                                              int groups,
                                              int hidden_dim,
                                              float eps) {
    for (int group = 0; group < groups; ++group) {
        const float *in_group = input + (size_t)group * hidden_dim;
        const float *w_group = weight + (size_t)group * hidden_dim;
        float *out_group = output + (size_t)group * hidden_dim;
        rmsnorm_forward_qwen3next_pytorch_bf16_storage(
            in_group, w_group, out_group, NULL, 1, hidden_dim, hidden_dim, eps);
    }
}

static void qwen4_group_rmsnorm_llama(const float *input,
                                      const float *weight,
                                      float *output,
                                      int groups,
                                      int hidden_dim,
                                      float eps) {
    for (int group = 0; group < groups; ++group) {
        const float *in_group = input + (size_t)group * hidden_dim;
        const float *w_group = weight + (size_t)group * hidden_dim;
        float *out_group = output + (size_t)group * hidden_dim;
        rmsnorm_forward_llama_production(
            in_group, w_group, out_group, NULL, 1, hidden_dim, hidden_dim, eps);
    }
}

static float qwen4_pytorch_bf16_dot(const float *left,
                                    const float *right,
                                    int dim) {
#if defined(__AVX2__)
    if (dim >= 32) {
        __m256 streams[4] = {
            _mm256_setzero_ps(), _mm256_setzero_ps(),
            _mm256_setzero_ps(), _mm256_setzero_ps()
        };
        int col = 0;
        for (; col + 32 <= dim; col += 32) {
            for (int stream = 0; stream < 4; ++stream) {
                _Alignas(32) float products[8];
                const int offset = col + stream * 8;
                for (int lane = 0; lane < 8; ++lane) {
                    products[lane] = qwen4_bf16_round(
                        left[offset + lane] * right[offset + lane]);
                }
                streams[stream] = _mm256_add_ps(
                    streams[stream], _mm256_load_ps(products));
            }
        }
        __m256 reduced = _mm256_add_ps(streams[0], streams[1]);
        reduced = _mm256_add_ps(reduced, streams[2]);
        reduced = _mm256_add_ps(reduced, streams[3]);
        _Alignas(32) float lanes[8];
        _mm256_store_ps(lanes, reduced);
        volatile float sum = 0.0f;
        for (int lane = 0; lane < 8; ++lane) sum = sum + lanes[lane];
        for (; col < dim; ++col) {
            sum = sum + qwen4_bf16_round(left[col] * right[col]);
        }
        return qwen4_bf16_round(sum);
    }
#endif
    volatile float sum = 0.0f;
    for (int col = 0; col < dim; ++col) {
        sum = sum + qwen4_bf16_round(left[col] * right[col]);
    }
    return qwen4_bf16_round(sum);
}

static float qwen4_llama_mul_sum_rows(const float *left,
                                      const float *right,
                                      int dim) {
    /* ggml materializes the FP32 multiply, then folds SUM_ROWS in ggml_float. */
    volatile double sum = 0.0;
    for (int col = 0; col < dim; ++col) {
        const float product = left[col] * right[col];
        sum = sum + (double)product;
    }
    return (float)sum;
}

static void qwen4_ple_gate_conv_inject_impl(
    const float *hyper_input,
    const float *key_projected,
    const float *value_projected,
    const float *norm_key_weight,
    const float *norm_query_weight,
    const float *norm_conv_weight,
    const void *conv_weight,
    float *hyper_output,
    float *key_norm_scratch,
    float *query_norm_scratch,
    float *gated_scratch,
    float *conv_norm_scratch,
    const float *conv_state_in,
    float *conv_state_out,
    int rows,
    int streams,
    int hidden_dim,
    int kernel_size,
    int dilation,
    float eps,
    int conv_weight_is_fp16,
    int llama_fp32_arithmetic) {
    if (!hyper_input || !key_projected || !value_projected || !norm_key_weight ||
        !norm_query_weight || !norm_conv_weight || !conv_weight || !hyper_output ||
        !key_norm_scratch || !query_norm_scratch || !gated_scratch ||
        !conv_norm_scratch || !conv_state_in || !conv_state_out || rows <= 0 || streams <= 0 ||
        hidden_dim <= 0 || kernel_size <= 0 || dilation <= 0) return;

    const int channels = streams * hidden_dim;
    const int history = (kernel_size - 1) * dilation;
    const float inv_sqrt_hidden = 1.0f / sqrtf((float)hidden_dim);
    for (int row = 0; row < rows; ++row) {
        const float *hyper_row = hyper_input + (size_t)row * channels;
        const float *key_row = key_projected + (size_t)row * channels;
        const float *value_row = value_projected + (size_t)row * hidden_dim;
        float *key_norm = key_norm_scratch + (size_t)row * channels;
        float *query_norm = query_norm_scratch + (size_t)row * channels;
        float *gated = gated_scratch + (size_t)row * channels;
        float *conv_norm = conv_norm_scratch + (size_t)row * channels;
        if (llama_fp32_arithmetic) {
            qwen4_group_rmsnorm_llama(
                key_row, norm_key_weight, key_norm, streams, hidden_dim, eps);
            qwen4_group_rmsnorm_llama(
                hyper_row, norm_query_weight, query_norm, streams, hidden_dim, eps);
        } else {
            qwen4_group_rmsnorm_pytorch_bf16(
                key_row, norm_key_weight, key_norm, streams, hidden_dim, eps);
            qwen4_group_rmsnorm_pytorch_bf16(
                hyper_row, norm_query_weight, query_norm, streams, hidden_dim, eps);
        }
        for (int stream = 0; stream < streams; ++stream) {
            const size_t base = (size_t)stream * hidden_dim;
            const float dot = llama_fp32_arithmetic
                ? qwen4_llama_mul_sum_rows(
                    key_norm + base, query_norm + base, hidden_dim)
                : qwen4_pytorch_bf16_dot(
                    key_norm + base, query_norm + base, hidden_dim);
            const float scaled_gate = llama_fp32_arithmetic
                ? dot * inv_sqrt_hidden
                : qwen4_bf16_round(dot * inv_sqrt_hidden);
            const float signed_root = copysignf(
                sqrtf(fmaxf(fabsf(scaled_gate), 1.0e-6f)), scaled_gate);
            const float gate = llama_fp32_arithmetic
                ? signed_root
                : qwen4_bf16_round(signed_root);
            const float sigmoid = 1.0f / (1.0f + expf(-gate));
            const float sigmoid_gate = llama_fp32_arithmetic
                ? sigmoid
                : qwen4_bf16_round(sigmoid);
            for (int col = 0; col < hidden_dim; ++col) {
                const float product = sigmoid_gate * value_row[col];
                gated[base + col] = llama_fp32_arithmetic
                    ? product
                    : qwen4_bf16_round(product);
            }
        }
        if (llama_fp32_arithmetic) {
            qwen4_group_rmsnorm_llama(
                gated, norm_conv_weight, conv_norm, streams, hidden_dim, eps);
        } else {
            qwen4_group_rmsnorm_pytorch_bf16(
                gated, norm_conv_weight, conv_norm, streams, hidden_dim, eps);
        }
    }

    for (int row = 0; row < rows; ++row) {
        const float *hyper_row = hyper_input + (size_t)row * channels;
        const float *gated = gated_scratch + (size_t)row * channels;
        float *out_row = hyper_output + (size_t)row * channels;
        for (int channel0 = 0; channel0 < channels; channel0 += 16) {
            const int active = channel0 + 16 <= channels
                ? 16
                : channels - channel0;
            float conv_tile[16];
            for (int lane = 0; lane < active; ++lane) {
                const int channel = channel0 + lane;
                volatile float sum = 0.0f;
                for (int tap = 0; tap < kernel_size; ++tap) {
                    const int source_row =
                        row - (kernel_size - 1 - tap) * dilation;
                    const float source = source_row >= 0
                        ? conv_norm_scratch[
                            (size_t)source_row * channels + channel]
                        : conv_state_in[
                            (size_t)(history + source_row) * channels + channel];
                    const size_t weight_index =
                        (size_t)channel * kernel_size + tap;
                    const float weight = conv_weight_is_fp16
                        ? ck_fp16_to_fp32(
                            ((const uint16_t *)conv_weight)[weight_index])
                        : qwen4_bf16_load(
                            ((const uint16_t *)conv_weight) + weight_index);
                    volatile float product = source * weight;
                    sum = sum + product;
                }
                conv_tile[lane] = llama_fp32_arithmetic
                    ? sum
                    : qwen4_bf16_round(sum);
            }
            if (llama_fp32_arithmetic) {
                recurrent_silu_forward_ggml(
                    conv_tile, conv_tile, 1, active);
            }
            for (int lane = 0; lane < active; ++lane) {
                const int channel = channel0 + lane;
                if (llama_fp32_arithmetic) {
                    const float ple = gated[channel] + conv_tile[lane];
                    out_row[channel] = hyper_row[channel] + ple;
                } else {
                    const float conv = conv_tile[lane];
                    const float silu_raw = conv / (1.0f + expf(-conv));
                    const float silu = qwen4_bf16_round(silu_raw);
                    const float ple = qwen4_bf16_round(gated[channel] + silu);
                    out_row[channel] =
                        qwen4_bf16_round(hyper_row[channel] + ple);
                }
            }
        }
    }

    if (history > 0) {
        if (rows >= history) {
            memcpy(conv_state_out,
                   conv_norm_scratch + (size_t)(rows - history) * channels,
                   (size_t)history * channels * sizeof(*conv_state_out));
        } else {
            memmove(conv_state_out,
                    conv_state_in + (size_t)rows * channels,
                    (size_t)(history - rows) * channels * sizeof(*conv_state_out));
            memcpy(conv_state_out + (size_t)(history - rows) * channels,
                   conv_norm_scratch,
                   (size_t)rows * channels * sizeof(*conv_state_out));
        }
    }
}

void qwen4_ple_gate_conv_inject_bf16(
    const float *hyper_input, const float *key_projected,
    const float *value_projected, const float *norm_key_weight,
    const float *norm_query_weight, const float *norm_conv_weight,
    const uint16_t *conv_weight, float *hyper_output,
    float *key_norm_scratch, float *query_norm_scratch, float *gated_scratch,
    float *conv_norm_scratch, const float *conv_state_in,
    float *conv_state_out, int rows, int streams, int hidden_dim,
    int kernel_size, int dilation, float eps) {
    qwen4_ple_gate_conv_inject_impl(
        hyper_input, key_projected, value_projected, norm_key_weight,
        norm_query_weight, norm_conv_weight, conv_weight, hyper_output,
        key_norm_scratch, query_norm_scratch, gated_scratch, conv_norm_scratch,
        conv_state_in, conv_state_out, rows, streams, hidden_dim, kernel_size,
        dilation, eps, 0, 0);
}

void qwen4_ple_gate_conv_inject_fp16(
    const float *hyper_input, const float *key_projected,
    const float *value_projected, const float *norm_key_weight,
    const float *norm_query_weight, const float *norm_conv_weight,
    const uint16_t *conv_weight, float *hyper_output,
    float *key_norm_scratch, float *query_norm_scratch, float *gated_scratch,
    float *conv_norm_scratch, const float *conv_state_in,
    float *conv_state_out, int rows, int streams, int hidden_dim,
    int kernel_size, int dilation, float eps) {
    qwen4_ple_gate_conv_inject_impl(
        hyper_input, key_projected, value_projected, norm_key_weight,
        norm_query_weight, norm_conv_weight, conv_weight, hyper_output,
        key_norm_scratch, query_norm_scratch, gated_scratch, conv_norm_scratch,
        conv_state_in, conv_state_out, rows, streams, hidden_dim, kernel_size,
        dilation, eps, 1, 0);
}

void qwen4_ple_gate_conv_inject_llama_fp16(
    const float *hyper_input, const float *key_projected,
    const float *value_projected, const float *norm_key_weight,
    const float *norm_query_weight, const float *norm_conv_weight,
    const uint16_t *conv_weight, float *hyper_output,
    float *key_norm_scratch, float *query_norm_scratch, float *gated_scratch,
    float *conv_norm_scratch, const float *conv_state_in,
    float *conv_state_out, int rows, int streams, int hidden_dim,
    int kernel_size, int dilation, float eps) {
    qwen4_ple_gate_conv_inject_impl(
        hyper_input, key_projected, value_projected, norm_key_weight,
        norm_query_weight, norm_conv_weight, conv_weight, hyper_output,
        key_norm_scratch, query_norm_scratch, gated_scratch, conv_norm_scratch,
        conv_state_in, conv_state_out, rows, streams, hidden_dim, kernel_size,
        dilation, eps, 1, 1);
}

static void qwen4_shared_head_rmsnorm(const float *input,
                                      const float *weight,
                                      float *output,
                                      int heads,
                                      int head_dim,
                                      float eps) {
    for (int head = 0; head < heads; ++head) {
        const float *in_head = input + (size_t)head * head_dim;
        float *out_head = output + (size_t)head * head_dim;
        float sum_sq = 0.0f;
        for (int col = 0; col < head_dim; ++col) sum_sq += in_head[col] * in_head[col];
        const float scale = 1.0f / sqrtf(sum_sq / (float)head_dim + eps);
        for (int col = 0; col < head_dim; ++col) {
            out_head[col] = qwen4_bf16_round(in_head[col] * scale * weight[col]);
        }
    }
}

static void qwen4_rope_split_inplace(float *vector,
                                     int rotary_dim,
                                     int position,
                                     float theta) {
    const int half = rotary_dim / 2;
    for (int index = 0; index < half; ++index) {
        const float inverse_frequency = powf(theta, -2.0f * (float)index / (float)rotary_dim);
        const float angle = (float)position * inverse_frequency;
        const float cosine = cosf(angle);
        const float sine = sinf(angle);
        const float first = vector[index];
        const float second = vector[index + half];
        vector[index] = qwen4_bf16_round(first * cosine - second * sine);
        vector[index + half] = qwen4_bf16_round(second * cosine + first * sine);
    }
}

void qwen4_qsa_index_select_bf16(
    const float *projected_qk,
    const float *index_key_cache_in,
    const float *q_norm_weight,
    const float *k_norm_weight,
    float *selected_indices,
    float *index_key_cache_out,
    float *q_norm_scratch,
    float *pooled_key_scratch,
    float *block_score_scratch,
    int32_t *block_index_scratch,
    int rows,
    int query_heads,
    int index_head_dim,
    int token_budget,
    int compress_ratio,
    int rotary_dim,
    int context_length,
    int position,
    float rope_theta,
    float eps) {
    if (!projected_qk || !index_key_cache_in || !q_norm_weight || !k_norm_weight ||
        !selected_indices || !index_key_cache_out || !q_norm_scratch ||
        !pooled_key_scratch || !block_score_scratch || !block_index_scratch ||
        rows <= 0 || query_heads <= 0 || index_head_dim <= 0 || token_budget <= 0 ||
        compress_ratio <= 0 || rotary_dim <= 0 || rotary_dim > index_head_dim ||
        context_length <= 0 || position < 0 || rows > context_length - position) return;

    const int projected_dim = (query_heads + 1) * index_head_dim;
    const int selection_width = token_budget + compress_ratio - 1;
    const int block_topk = token_budget / compress_ratio;
    if (index_key_cache_out != index_key_cache_in && position > 0) {
        memcpy(index_key_cache_out, index_key_cache_in,
               (size_t)position * index_head_dim * sizeof(*index_key_cache_out));
    }

    for (int row = 0; row < rows; ++row) {
        const int absolute_position = position + row;
        const float *projected = projected_qk + (size_t)row * projected_dim;
        float *query_normed = q_norm_scratch;
        qwen4_shared_head_rmsnorm(projected, q_norm_weight, query_normed,
                                  query_heads, index_head_dim, eps);
        for (int head = 0; head < query_heads; ++head) {
            qwen4_rope_split_inplace(query_normed + (size_t)head * index_head_dim,
                                     rotary_dim, absolute_position, rope_theta);
        }
        float *raw_key = index_key_cache_out + (size_t)absolute_position * index_head_dim;
        const float *projected_key = projected + (size_t)query_heads * index_head_dim;
        for (int col = 0; col < index_head_dim; ++col) {
            raw_key[col] = qwen4_bf16_round(projected_key[col]);
        }

        float *selected_row = selected_indices + (size_t)row * selection_width;
        for (int slot = 0; slot < selection_width; ++slot) selected_row[slot] = -1.0f;
        const int visible = absolute_position + 1;
        const int complete_blocks = visible / compress_ratio;
        const int selected_blocks = complete_blocks < block_topk ? complete_blocks : block_topk;
        for (int slot = 0; slot < selected_blocks; ++slot) {
            block_score_scratch[slot] = -INFINITY;
            block_index_scratch[slot] = -1;
        }

        for (int block = 0; block < complete_blocks; ++block) {
            const int block_start = block * compress_ratio;
            for (int col = 0; col < index_head_dim; ++col) {
                float sum = 0.0f;
                for (int token = 0; token < compress_ratio; ++token) {
                    sum += index_key_cache_out[
                        (size_t)(block_start + token) * index_head_dim + col];
                }
                pooled_key_scratch[col] = qwen4_bf16_round(sum / (float)compress_ratio);
            }
            qwen4_shared_head_rmsnorm(pooled_key_scratch, k_norm_weight,
                                      pooled_key_scratch, 1, index_head_dim, eps);
            qwen4_rope_split_inplace(pooled_key_scratch, rotary_dim, block_start, rope_theta);
            float score = 0.0f;
            for (int head = 0; head < query_heads; ++head) {
                const float *query_head = query_normed + (size_t)head * index_head_dim;
                float dot = 0.0f;
                for (int col = 0; col < index_head_dim; ++col) {
                    dot += query_head[col] * pooled_key_scratch[col];
                }
                if (dot > 0.0f) score += dot;
            }
            score /= sqrtf((float)index_head_dim);
            int target = -1;
            for (int slot = 0; slot < selected_blocks; ++slot) {
                if (target < 0 || block_score_scratch[slot] < block_score_scratch[target]) target = slot;
            }
            if (target >= 0 && score > block_score_scratch[target]) {
                block_score_scratch[target] = score;
                block_index_scratch[target] = block;
            }
        }

        for (int outer = 1; outer < selected_blocks; ++outer) {
            const int32_t value = block_index_scratch[outer];
            int inner = outer - 1;
            while (inner >= 0 && block_index_scratch[inner] > value) {
                block_index_scratch[inner + 1] = block_index_scratch[inner];
                --inner;
            }
            block_index_scratch[inner + 1] = value;
        }
        int output_count = 0;
        for (int slot = 0; slot < selected_blocks; ++slot) {
            const int block = block_index_scratch[slot];
            if (block < 0) continue;
            for (int token = 0; token < compress_ratio; ++token) {
                selected_row[output_count++] = (float)(block * compress_ratio + token);
            }
        }
        for (int token = complete_blocks * compress_ratio;
             token < visible && output_count < selection_width; ++token) {
            selected_row[output_count++] = (float)token;
        }
    }
}
