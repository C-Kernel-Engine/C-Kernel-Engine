/* Optional, model-neutral observation of the final query's cross-attention.
 * The provider writes one layer slice of a caller-owned [layer,head,frame]
 * trace. Other slices and output padding remain untouched. */
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static int ck_mul_size(size_t a, size_t b, size_t *out) {
    if (a && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int ck_overlap(const void *a, size_t na, const void *b, size_t nb) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    if (na > UINTPTR_MAX - x || nb > UINTPTR_MAX - y) return 1;
    return x < y + nb && y < x + na;
}

int attention_last_query_prob_trace_f32(
    const float *query, size_t query_elements,
    const float *key, size_t key_elements,
    float *trace, size_t trace_elements,
    float *score_scratch, size_t score_elements,
    size_t layers, size_t layer, size_t heads,
    size_t query_tokens, size_t key_tokens,
    size_t frame_capacity, size_t head_dim, float scale)
{
    size_t query_head, key_head, trace_layer, score_needed, query_needed;
    size_t key_needed, trace_needed, input_bytes, key_bytes, trace_bytes;
    size_t score_bytes;
    if (!query || !key || !trace || !score_scratch || !layers || !heads ||
        !query_tokens || !key_tokens || !frame_capacity || !head_dim ||
        layer >= layers || key_tokens > frame_capacity || !isfinite(scale) ||
        !ck_mul_size(query_tokens, head_dim, &query_head) ||
        !ck_mul_size(frame_capacity, head_dim, &key_head) ||
        !ck_mul_size(heads, frame_capacity, &trace_layer) ||
        !ck_mul_size(heads, key_tokens, &score_needed) ||
        !ck_mul_size(heads, query_head, &query_needed) ||
        !ck_mul_size(heads, key_head, &key_needed) ||
        !ck_mul_size(layers, trace_layer, &trace_needed)) return -1;
    if (query_elements < query_needed || key_elements < key_needed ||
        trace_elements < trace_needed || score_elements < score_needed)
        return -2;
    if (!ck_mul_size(query_elements, sizeof(float), &input_bytes) ||
        !ck_mul_size(key_elements, sizeof(float), &key_bytes) ||
        !ck_mul_size(trace_elements, sizeof(float), &trace_bytes) ||
        !ck_mul_size(score_elements, sizeof(float), &score_bytes)) return -1;
    const void *ptrs[] = {query, key, trace, score_scratch};
    const size_t bytes[] = {input_bytes, key_bytes, trace_bytes, score_bytes};
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = i + 1; j < 4; ++j)
            if (ck_overlap(ptrs[i], bytes[i], ptrs[j], bytes[j])) return -3;

    for (size_t h = 0; h < heads; ++h) {
        const float *q = query + h * query_head + (query_tokens - 1) * head_dim;
        const float *k = key + h * key_head;
        float *scores = score_scratch + h * key_tokens;
        float maximum = -FLT_MAX;
        for (size_t f = 0; f < key_tokens; ++f) {
            float dot = 0.0f;
            for (size_t d = 0; d < head_dim; ++d) {
                if (!isfinite(q[d]) || !isfinite(k[f * head_dim + d]))
                    return -4;
                dot = fmaf(q[d], k[f * head_dim + d], dot);
            }
            const float score = dot * scale;
            if (!isfinite(score)) return -4;
            scores[f] = score;
            maximum = fmaxf(maximum, score);
        }
        double denominator = 0.0;
        for (size_t f = 0; f < key_tokens; ++f) {
            scores[f] = expf(scores[f] - maximum);
            denominator += scores[f];
        }
        if (!(denominator > 0.0) || !isfinite(denominator)) return -4;
        const float inverse = (float)(1.0 / denominator);
        for (size_t f = 0; f < key_tokens; ++f) {
            scores[f] *= inverse;
            if (!isfinite(scores[f])) return -4;
        }
    }
    float *dst = trace + layer * trace_layer;
    for (size_t h = 0; h < heads; ++h)
        for (size_t f = 0; f < key_tokens; ++f)
            dst[h * frame_capacity + f] = score_scratch[h * key_tokens + f];
    return 0;
}
