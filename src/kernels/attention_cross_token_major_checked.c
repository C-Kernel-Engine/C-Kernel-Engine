#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "checked_float_buffer.h"
#include "ckernel_head_backward.h"

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

/* Preserve the forward score and exponential storage boundaries for replay. */
static int attention_scores(const float *q, const float *key, size_t memory_tokens,
    size_t width, size_t head, size_t dim, float *scores, double *denominator) {
    double scale = 1.0 / sqrt((double)dim), maximum = -INFINITY;
    for (size_t k = 0; k < memory_tokens; ++k) {
        double value = dot(q, key + k * width + head * dim, dim) * scale;
        if (!isfinite(value) || fabs(value) > FLT_MAX) return -1;
        scores[k] = (float)value;
        if (scores[k] > maximum) maximum = scores[k];
    }
    *denominator = 0.0;
    for (size_t k = 0; k < memory_tokens; ++k) {
        scores[k] = (float)exp((double)scores[k] - maximum);
        *denominator += (double)scores[k];
    }
    return !isfinite(*denominator) || *denominator <= 0.0 ? -1 : 0;
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
    for (size_t t = 0; t < query_tokens; ++t) {
        for (size_t h = 0; h < heads; ++h) {
            const float *q = query + t * width + h * head_dim;
            double denominator;
            if (attention_scores(q, key, memory_tokens, width, h, head_dim, scores, &denominator)) return -1;
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

/* VJP for full, unmasked token-major attention. Parameter-free and batch one.
 * Stage all three outputs so even late overflow leaves caller gradients intact.
 * FP64 dot/reduction arithmetic with explicit FP32 score/exponential/dS stores;
 * shared dK/dV accumulate in ascending query order with an FP32 store per query.
 */
int ck_attention_cross_backward_f32(
    const float *query, size_t query_elements,
    const float *key, size_t key_elements,
    const float *value, size_t value_elements,
    const float *d_output, size_t d_output_elements,
    float *d_query, size_t d_query_elements,
    float *d_key, size_t d_key_elements,
    float *d_value, size_t d_value_elements,
    float *scratch, size_t scratch_elements,
    size_t query_tokens, size_t memory_tokens, size_t heads, size_t head_dim) {
    if (!query_tokens || !memory_tokens || !heads || !head_dim || heads > SIZE_MAX / head_dim) return -2;
    size_t width = heads * head_dim;
    if (query_tokens > SIZE_MAX / width || memory_tokens > SIZE_MAX / width) return -2;
    size_t nq = query_tokens * width, nm = memory_tokens * width;
    if (nm > (SIZE_MAX - nq) / 2) return -2;
    size_t staged = nq + 2 * nm;
    if (memory_tokens > (SIZE_MAX - staged) / 2) return -2;
    size_t needed = staged + 2 * memory_tokens;
    if (query_elements < nq || key_elements < nm || value_elements < nm ||
        d_output_elements < nq || d_query_elements < nq || d_key_elements < nm ||
        d_value_elements < nm || scratch_elements < needed) return -2;
    const float *readers[] = {query, key, value, d_output};
    size_t nr[] = {query_elements, key_elements, value_elements, d_output_elements};
    float *writers[] = {d_query, d_key, d_value, scratch};
    size_t nw[] = {d_query_elements, d_key_elements, d_value_elements, scratch_elements};
    for (size_t i = 0; i < 4; ++i) {
        if (!ck_checked_region(readers[i], nr[i]) || !ck_checked_region(writers[i], nw[i])) return -1;
    }
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j)
            if (ck_checked_overlap(writers[i], nw[i], readers[j], nr[j])) return -1;
        for (size_t j = 0; j < i; ++j)
            if (ck_checked_overlap(writers[i], nw[i], writers[j], nw[j])) return -1;
    }
    if (!ck_checked_finite(query, nq) || !ck_checked_finite(key, nm) ||
        !ck_checked_finite(value, nm) || !ck_checked_finite(d_output, nq)) return -3;
    float *dq = scratch, *dk = dq + nq, *dv = dk + nm;
    float *scores = dv + nm, *ds = scores + memory_tokens;
    memset(scratch, 0, staged * sizeof(float));
    double scale = 1.0 / sqrt((double)head_dim);
    for (size_t t = 0; t < query_tokens; ++t) {
        for (size_t h = 0; h < heads; ++h) {
            const float *q = query + t * width + h * head_dim;
            const float *dy = d_output + t * width + h * head_dim;
            double denominator;
            if (attention_scores(q, key, memory_tokens, width, h, head_dim, scores, &denominator)) return -4;
            double weighted = 0.0;
            for (size_t k = 0; k < memory_tokens; ++k) {
                double dp = dot(dy, value + k * width + h * head_dim, head_dim);
                volatile double term = ((double)scores[k] / denominator) * dp;
                weighted += term;
            }
            for (size_t k = 0; k < memory_tokens; ++k) {
                double dp = dot(dy, value + k * width + h * head_dim, head_dim);
                double result = ((double)scores[k] / denominator) * (dp - weighted) * scale;
                if (!isfinite(result) || fabs(result) > FLT_MAX) return -4;
                ds[k] = (float)result;
            }
            for (size_t d = 0; d < head_dim; ++d) {
                double sum = 0.0;
                for (size_t k = 0; k < memory_tokens; ++k) {
                    volatile double term = (double)ds[k] * (double)key[k * width + h * head_dim + d];
                    sum += term;
                }
                if (!isfinite(sum) || fabs(sum) > FLT_MAX) return -4;
                dq[t * width + h * head_dim + d] = (float)sum;
            }
            for (size_t k = 0; k < memory_tokens; ++k) {
                for (size_t d = 0; d < head_dim; ++d) {
                    size_t index = k * width + h * head_dim + d;
                    volatile double kp = (double)ds[k] * (double)q[d];
                    volatile double vp = ((double)scores[k] / denominator) * (double)dy[d];
                    double kg = (double)dk[index] + kp, vg = (double)dv[index] + vp;
                    if (!isfinite(kg) || !isfinite(vg) || fabs(kg) > FLT_MAX || fabs(vg) > FLT_MAX) return -4;
                    dk[index] = (float)kg;
                    dv[index] = (float)vg;
                }
            }
        }
    }
    memcpy(d_query, dq, nq * sizeof(float));
    memcpy(d_key, dk, nm * sizeof(float));
    memcpy(d_value, dv, nm * sizeof(float));
    return 0;
}
