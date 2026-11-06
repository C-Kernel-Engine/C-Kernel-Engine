#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>

static int region_ok(const void *ptr, size_t count, size_t width) {
    return ptr && (uintptr_t)ptr % width == 0 &&
           count <= SIZE_MAX / width &&
           (uintptr_t)ptr <= UINTPTR_MAX - count * width;
}

static int overlaps(const void *a, size_t a_bytes,
                    const void *b, size_t b_bytes) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x < y + b_bytes && y < x + a_bytes;
}

static double dot(const float *a, const float *b, size_t n) {
    double sum = 0.0;
    for (size_t d = 0; d < n; ++d) {
        volatile double product = (double)a[d] * (double)b[d];
        sum += product;
    }
    return sum;
}

/* Full unmasked cross attention, batch one, token-major packed heads. */
int ck_attention_cross_token_major_checked_f32(
    const float *query, size_t query_elements,
    const float *key, size_t key_elements,
    const float *value, size_t value_elements,
    float *output, size_t output_elements,
    float *scratch, size_t scratch_elements,
    size_t query_tokens, size_t memory_tokens,
    size_t heads, size_t head_dim) {
    if (!query_tokens || !memory_tokens || !heads || !head_dim ||
        heads > SIZE_MAX / head_dim) return -2;
    size_t width = heads * head_dim;
    if (query_tokens > SIZE_MAX / width ||
        memory_tokens > SIZE_MAX / width) return -2;
    size_t q_count = query_tokens * width;
    size_t m_count = memory_tokens * width;
    if (q_count > SIZE_MAX - memory_tokens ||
        query_elements < q_count || key_elements < m_count ||
        value_elements < m_count || output_elements < q_count ||
        scratch_elements < memory_tokens + q_count) return -2;
    if (!region_ok(query, query_elements, sizeof(float)) ||
        !region_ok(key, key_elements, sizeof(float)) ||
        !region_ok(value, value_elements, sizeof(float)) ||
        !region_ok(output, output_elements, sizeof(float)) ||
        !region_ok(scratch, scratch_elements, sizeof(float))) return -1;
    size_t output_bytes = output_elements * sizeof(float);
    size_t scratch_bytes = scratch_elements * sizeof(float);
    if (overlaps(output, output_bytes, query, query_elements * sizeof(float)) ||
        overlaps(output, output_bytes, key, key_elements * sizeof(float)) ||
        overlaps(output, output_bytes, value, value_elements * sizeof(float)) ||
        overlaps(output, output_bytes, scratch, scratch_bytes) ||
        overlaps(scratch, scratch_bytes, query, query_elements * sizeof(float)) ||
        overlaps(scratch, scratch_bytes, key, key_elements * sizeof(float)) ||
        overlaps(scratch, scratch_bytes, value, value_elements * sizeof(float))) return -1;
    for (size_t i = 0; i < q_count; ++i)
        if (!isfinite(query[i])) return -1;
    for (size_t i = 0; i < m_count; ++i)
        if (!isfinite(key[i]) || !isfinite(value[i])) return -1;

    float *scores = scratch;
    float *staged = scratch + memory_tokens;
    double scale = 1.0 / sqrt((double)head_dim);
    for (size_t t = 0; t < query_tokens; ++t) {
        for (size_t h = 0; h < heads; ++h) {
            const float *q = query + t * width + h * head_dim;
            double maximum = -INFINITY;
            for (size_t k = 0; k < memory_tokens; ++k) {
                const float *kv = key + k * width + h * head_dim;
                double s = dot(q, kv, head_dim) * scale;
                if (!isfinite(s) || fabs(s) > FLT_MAX) return -1;
                scores[k] = (float)s;
                if (scores[k] > maximum) maximum = scores[k];
            }
            double denominator = 0.0;
            for (size_t k = 0; k < memory_tokens; ++k) {
                scores[k] = (float)exp((double)scores[k] - maximum);
                denominator += (double)scores[k];
            }
            if (!isfinite(denominator) || denominator <= 0.0) return -1;
            for (size_t d = 0; d < head_dim; ++d) {
                double weighted = 0.0;
                for (size_t k = 0; k < memory_tokens; ++k) {
                    volatile double product = (double)scores[k] *
                        (double)value[k * width + h * head_dim + d];
                    weighted += product;
                }
                float result = (float)(weighted / denominator);
                if (!isfinite(result)) return -1;
                staged[t * width + h * head_dim + d] = result;
            }
        }
    }
    for (size_t i = 0; i < q_count; ++i) output[i] = staged[i];
    return 0;
}
