#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "bf16_utils.h"

static int region_valid(const void *ptr, size_t count, size_t width)
{
    return ptr && (uintptr_t)ptr % width == 0 && count <= SIZE_MAX / width &&
           (uintptr_t)ptr <= UINTPTR_MAX - count * width;
}

static int regions_overlap(const void *a, size_t a_bytes, const void *b, size_t b_bytes)
{
    return (uintptr_t)a < (uintptr_t)b + b_bytes &&
           (uintptr_t)b < (uintptr_t)a + a_bytes;
}

static float read_element(const void *buffer, size_t index, int bf16)
{
    return bf16 ? bf16_to_float(((const uint16_t *)buffer)[index]) : ((const float *)buffer)[index];
}

static double attention_dot(const void *q, const void *k, size_t q_offset,
                            size_t k_offset, size_t width, int bf16)
{
    double sum = 0.0;
    for (size_t i = 0; i < width; ++i) {
        volatile double product = (double)read_element(q, q_offset + i, bf16) *
                                  read_element(k, k_offset + i, bf16);
        sum += product;
    }
    return sum;
}

/* Per-query visibility is two ordered, disjoint half-open key ranges.
 * An empty second range means one interval. Output commits atomically. */
static int attention_ranges_checked(
    const void *query, size_t query_elements,
    const void *key, size_t key_elements,
    const void *value, size_t value_elements,
    const int32_t *ranges, size_t range_elements,
    void *output, size_t output_elements,
    float *scratch, size_t scratch_elements,
    size_t query_tokens, size_t memory_tokens,
    size_t heads, size_t kv_heads, size_t head_dim, float scale, int bf16)
{
    if (!query_tokens || !memory_tokens || memory_tokens > INT32_MAX ||
        !heads || !kv_heads || heads % kv_heads || !head_dim ||
        heads > SIZE_MAX / head_dim || query_tokens > SIZE_MAX / 4 ||
        !isfinite(scale) || scale <= 0.0f) return -2;
    const size_t qw = heads * head_dim;
    const size_t kw = kv_heads * head_dim;
    if (query_tokens > SIZE_MAX / qw || memory_tokens > SIZE_MAX / kw) return -2;
    const size_t q = query_tokens * qw;
    const size_t k = memory_tokens * kw;
    if (q > SIZE_MAX - memory_tokens || range_elements < 4 * query_tokens ||
        query_elements < q || key_elements < k || value_elements < k ||
        output_elements < q || scratch_elements < q + memory_tokens) return -2;

    const void *inputs[] = {query, key, value, ranges};
    const size_t counts[] = {query_elements, key_elements, value_elements, range_elements};
    const size_t element_width = bf16 ? sizeof(uint16_t) : sizeof(float);
    const size_t widths[] = {element_width, element_width, element_width, sizeof(int32_t)};
    if (!region_valid(output, output_elements, element_width) ||
        !region_valid(scratch, scratch_elements, sizeof(float))) return -1;
    const size_t output_bytes = output_elements * element_width;
    const size_t scratch_bytes = scratch_elements * sizeof(float);
    if (regions_overlap(output, output_bytes, scratch, scratch_bytes)) return -1;
    for (size_t i = 0; i < 4; ++i) {
        if (!region_valid(inputs[i], counts[i], widths[i])) return -1;
        const size_t bytes = counts[i] * widths[i];
        if (regions_overlap(output, output_bytes, inputs[i], bytes) ||
            regions_overlap(scratch, scratch_bytes, inputs[i], bytes)) return -1;
    }
    for (size_t t = 0; t < query_tokens; ++t) {
        const int32_t *r = ranges + 4 * t;
        if (r[0] < 0 || r[0] > r[1] || r[1] > r[2] ||
            r[2] > r[3] || (size_t)r[3] > memory_tokens ||
            (r[0] == r[1] && r[2] == r[3])) return -2;
    }
    for (size_t i = 0; i < q; ++i)
        if (!isfinite(read_element(query, i, bf16))) return -1;
    for (size_t i = 0; i < k; ++i)
        if (!isfinite(read_element(key, i, bf16)) || !isfinite(read_element(value, i, bf16))) return -1;

    float *scores = scratch;
    float *staged = scratch + memory_tokens;
    for (size_t t = 0; t < query_tokens; ++t) {
        const int32_t *r = ranges + 4 * t;
        for (size_t h = 0; h < heads; ++h) {
            const size_t query_head = t * qw + h * head_dim;
            const size_t head_offset = (h / (heads / kv_heads)) * head_dim;
            float largest = -INFINITY;
            for (size_t span = 0; span < 4; span += 2) {
                for (size_t j = (size_t)r[span]; j < (size_t)r[span + 1]; ++j) {
                    double dot = attention_dot(query, key, query_head, j * kw + head_offset, head_dim, bf16);
                    if (bf16) dot = bf16_to_float(float_to_bf16((float)dot));
                    const double s = dot * scale;
                    if (!isfinite(s) || fabs(s) > FLT_MAX) return -1;
                    scores[j] = bf16 ? bf16_to_float(float_to_bf16((float)s)) : (float)s;
                    if (!isfinite(scores[j])) return -1;
                    if (scores[j] > largest) largest = scores[j];
                }
            }
            double denom = 0.0;
            for (size_t span = 0; span < 4; span += 2) {
                for (size_t j = (size_t)r[span]; j < (size_t)r[span + 1]; ++j) {
                    scores[j] = (float)exp((double)scores[j] - largest);
                    denom += scores[j];
                }
            }
            if (bf16) {
                for (size_t span = 0; span < 4; span += 2) {
                    for (size_t j = (size_t)r[span]; j < (size_t)r[span + 1]; ++j) {
                        scores[j] = bf16_to_float(float_to_bf16((float)(scores[j] / denom)));
                    }
                }
            }
            for (size_t d = 0; d < head_dim; ++d) {
                double sum = 0.0;
                for (size_t span = 0; span < 4; span += 2) {
                    for (size_t j = (size_t)r[span]; j < (size_t)r[span + 1]; ++j) {
                        volatile double product = (double)scores[j] *
                            read_element(value, j * kw + head_offset + d, bf16);
                        sum += product;
                    }
                }
                float v = (float)(bf16 ? sum : sum / denom);
                if (bf16) v = bf16_to_float(float_to_bf16(v));
                if (!isfinite(v)) return -1;
                staged[t * qw + h * head_dim + d] = v;
            }
        }
    }
    if (bf16) {
        for (size_t i = 0; i < q; ++i) ((uint16_t *)output)[i] = float_to_bf16(staged[i]);
    } else {
        for (size_t i = 0; i < q; ++i) ((float *)output)[i] = staged[i];
    }
    return 0;
}

int ck_attention_ranges_checked_f32(
    const float *query, size_t query_elements, const float *key, size_t key_elements,
    const float *value, size_t value_elements, const int32_t *ranges, size_t range_elements,
    float *output, size_t output_elements, float *scratch, size_t scratch_elements,
    size_t query_tokens, size_t memory_tokens, size_t heads, size_t kv_heads, size_t head_dim, float scale)
{
    return attention_ranges_checked(query, query_elements, key, key_elements, value, value_elements,
        ranges, range_elements, output, output_elements, scratch, scratch_elements,
        query_tokens, memory_tokens, heads, kv_heads, head_dim, scale, 0);
}

int ck_attention_ranges_checked_bf16(
    const uint16_t *query, size_t query_elements, const uint16_t *key, size_t key_elements,
    const uint16_t *value, size_t value_elements, const int32_t *ranges, size_t range_elements,
    uint16_t *output, size_t output_elements, float *scratch, size_t scratch_elements,
    size_t query_tokens, size_t memory_tokens, size_t heads, size_t kv_heads, size_t head_dim, float scale)
{
    return attention_ranges_checked(query, query_elements, key, key_elements, value, value_elements,
        ranges, range_elements, output, output_elements, scratch, scratch_elements,
        query_tokens, memory_tokens, heads, kv_heads, head_dim, scale, 1);
}
