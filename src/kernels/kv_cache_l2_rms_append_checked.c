#include "checked_float_buffer.h"

#define KV_APPEND_ARGS \
    const float *key, size_t key_elements, const float *value, size_t value_elements, \
    float *k_cache, size_t k_cache_elements, float *v_cache, size_t v_cache_elements, \
    size_t tokens, size_t heads, size_t dim, size_t capacity, size_t position, \
    float key_clip, float epsilon

/* Normalize token-major input and store either linear or circular head-major KV. */
static int kv_cache_l2_rms_append(KV_APPEND_ARGS, int circular)
{
    if (!heads || !dim || dim > SIZE_MAX / heads || !tokens || tokens > capacity ||
        position > SIZE_MAX - tokens || (!circular && position > capacity - tokens) ||
        !isfinite(key_clip) || key_clip <= 0.f || !isfinite(epsilon) || epsilon <= 0.f)
        return -2;
    size_t width = heads * dim, active, cache;
    if (!ck_checked_span(tokens, width, width, &active) ||
        !ck_checked_span(capacity, width, width, &cache) ||
        active > key_elements || active > value_elements ||
        cache > k_cache_elements || cache > v_cache_elements) return -2;

    if (!ck_checked_region(key, active) || !ck_checked_region(value, active) ||
        !ck_checked_region(k_cache, cache) || !ck_checked_region(v_cache, cache) ||
        ck_checked_overlap(k_cache, cache, v_cache, cache) ||
        ck_checked_overlap(k_cache, cache, key, active) ||
        ck_checked_overlap(k_cache, cache, value, active) ||
        ck_checked_overlap(v_cache, cache, key, active) ||
        ck_checked_overlap(v_cache, cache, value, active)) return -1;
    if (!ck_checked_finite(key, active) || !ck_checked_finite(value, active)) return -3;

    for (size_t t = 0; t < tokens; ++t) {
        const size_t row = circular ? (position + t) % capacity : position + t;
        for (size_t h = 0; h < heads; ++h) {
            const size_t src = (t * heads + h) * dim;
            const size_t dst = (h * capacity + row) * dim;
            double key_sq = 0.0, value_sq = 0.0;
            for (size_t d = 0; d < dim; ++d) {
                const volatile double ks = (double)key[src + d] * key[src + d];
                const volatile double vs = (double)value[src + d] * value[src + d];
                key_sq += ks;
                value_sq += vs;
            }
            const double key_scale = fmin(1.0, key_clip / (sqrt(key_sq) + epsilon));
            const double value_scale = 1.0 / sqrt(value_sq / (double)dim + epsilon);
            for (size_t d = 0; d < dim; ++d) {
                k_cache[dst + d] = (float)(key[src + d] * key_scale);
                v_cache[dst + d] = (float)(value[src + d] * value_scale);
            }
        }
    }
    return 0;
}

#define KV_APPEND_FORWARD(circular) \
    kv_cache_l2_rms_append(key, key_elements, value, value_elements, \
                           k_cache, k_cache_elements, v_cache, v_cache_elements, \
                           tokens, heads, dim, capacity, position, key_clip, epsilon, circular)

int ck_kv_cache_l2_rms_append_checked_f32(KV_APPEND_ARGS)
{
    return KV_APPEND_FORWARD(0);
}

int ck_kv_cache_l2_rms_append_ring_checked_f32(KV_APPEND_ARGS)
{
    return KV_APPEND_FORWARD(1);
}
