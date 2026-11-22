#include "checked_float_buffer.h"

/* Read caller-owned K/V at absolute positions [0, position + tokens).
 * Relative bias is indexed by backward distance inside its supplied extent.
 * An extra sink logit contributes to the denominator with a zero value. */
int ck_gqa_relative_sink_checked_f32(
    const float *query, size_t query_elements, const float *key, size_t key_elements,
    const float *value, size_t value_elements, const float *relative, size_t relative_elements,
    const float *sink, size_t sink_elements, float *output, size_t output_elements,
    double *work, size_t work_elements, size_t tokens, size_t heads, size_t kv_heads,
    size_t dim, size_t capacity, size_t position, size_t relative_extent, size_t window)
{
    if (!heads || heads > 256 || !kv_heads || heads % kv_heads ||
        !dim || dim > 4096 || !capacity || capacity > 65536 ||
        !tokens || position >= capacity || tokens > capacity - position ||
        !relative_extent || relative_extent > 16384 || window > 16384)
        return -2;

    size_t rows, q_size, kv_size, rel_size;
    if (!ck_checked_span(tokens, heads, heads, &rows) ||
        !ck_checked_span(rows, dim, dim, &q_size) ||
        !ck_checked_span(kv_heads * capacity, dim, dim, &kv_size) ||
        !ck_checked_span(rows, relative_extent, relative_extent, &rel_size) ||
        q_size > query_elements || q_size > output_elements ||
        kv_size > key_elements || kv_size > value_elements ||
        rel_size > relative_elements || rows > sink_elements || dim > work_elements)
        return -2;

    size_t workspace = dim;
#ifdef _OPENMP
    if (q_size <= SIZE_MAX / sizeof(double) && work_elements >= q_size)
        workspace = q_size;
#endif
    const size_t scratch_bytes = workspace * sizeof(double);
    if (!work || (uintptr_t)work % _Alignof(double) ||
        (uintptr_t)work > UINTPTR_MAX - scratch_bytes ||
        !ck_checked_region(output, q_size)) return -1;
    const float *inputs[] = {query, key, value, relative, sink, output};
    const size_t counts[] = {q_size, kv_size, kv_size, rel_size, rows, q_size};
    for (size_t i = 0; i < 6; ++i) {
        if (!ck_checked_region(inputs[i], counts[i])) return -1;
        const uintptr_t a = (uintptr_t)inputs[i], b = (uintptr_t)work;
        if ((a < b + scratch_bytes && b < a + counts[i] * sizeof(float)) ||
            (i != 5 && ck_checked_overlap(output, q_size, inputs[i], counts[i])))
            return -1;
    }

    if (!ck_checked_finite(query, q_size) || !ck_checked_finite(relative, rel_size) ||
        !ck_checked_finite(sink, rows)) return -3;
    const size_t first = window && position >= window ? position - window + 1 : 0;
    for (size_t h = 0; h < kv_heads; ++h) {
        const size_t offset = (h * capacity + first) * dim;
        const size_t used = (position + tokens - first) * dim;
        if (!ck_checked_finite(key + offset, used) || !ck_checked_finite(value + offset, used))
            return -3;
    }

    const double scale = 1.0 / sqrt((double)dim);
    const size_t group = heads / kv_heads;
#ifdef _OPENMP
    const size_t visible = position + tokens - first;
    const int parallel = workspace > dim && visible >= 128;
#pragma omp parallel for if(parallel)
#endif
    for (size_t row = 0; row < rows; ++row) {
        const size_t qpos = position + row / heads;
        const size_t begin = window && qpos >= window ? qpos - window + 1 : 0;
        const float *q = query + row * dim;
        const float *rel = relative + row * relative_extent;
        const size_t cache = ((row % heads) / group) * capacity;
        double *accumulator = work + (workspace > dim ? row * dim : 0);
        double max_logit = sink[row], denom = 1.0;
        for (size_t d = 0; d < dim; ++d) accumulator[d] = 0.0;
        for (size_t k = begin; k <= qpos; ++k) {
            const float *keys = key + (cache + k) * dim;
            const float *values = value + (cache + k) * dim;
            double dot = 0.0;
            /* Each product of finite FP32 operands is exact in FP64. */
            for (size_t d = 0; d < dim; ++d) dot += (double)q[d] * (double)keys[d];
            const size_t distance = qpos - k;
            const double score = dot * scale + (distance < relative_extent ? rel[distance] : 0.0);
            const double next_max = fmax(score, max_logit);
            const double old_factor = max_logit == next_max ? 1.0 : exp(max_logit - next_max);
            const double new_factor = score == next_max ? 1.0 : exp(score - next_max);
            denom = denom * old_factor + new_factor;
            for (size_t d = 0; d < dim; ++d)
                accumulator[d] = accumulator[d] * old_factor + (double)values[d] * new_factor;
            max_logit = next_max;
        }
        for (size_t d = 0; d < dim; ++d)
            output[row * dim + d] = (float)(accumulator[d] / denom);
    }
    return 0;
}
