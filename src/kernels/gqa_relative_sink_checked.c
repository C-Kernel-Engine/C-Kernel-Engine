#include "checked_float_buffer.h"

#if defined(__ELF__) && (defined(__GNUC__) || defined(__clang__))
#include "checked_parallel.h"
#define CK_GQA_OPTIONAL_POOL 1
#pragma weak ck_get_num_threads
#pragma weak ck_threadpool_global
#pragma weak ck_threadpool_in_dispatch
#pragma weak ck_threadpool_n_threads
#pragma weak ck_threadpool_parallel_for_n
#endif

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize ("fp-contract=off")
#endif

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define CK_GQA_X86_AVX2 1

static __attribute__((target("avx2"))) int finite_cache_avx2(const float *data, size_t length)
{
    const __m256i magnitude_mask = _mm256_set1_epi32(0x7fffffff);
    const __m256i largest_finite = _mm256_set1_epi32(0x7f7fffff);
    size_t i = 0;
    for (; length - i >= 8; i += 8) {
        const __m256i values = _mm256_castps_si256(_mm256_loadu_ps(data + i));
        const __m256i magnitude = _mm256_and_si256(values, magnitude_mask);
        if (_mm256_movemask_epi8(_mm256_cmpgt_epi32(magnitude, largest_finite)))
            return 0;
    }
    return ck_checked_finite(data + i, length - i);
}

/* Each SIMD lane owns a key. Advance every lane in channel order. */
static inline __attribute__((target("avx2"))) void scores_for_four_keys(
    const float *query, const float *keys, size_t dim, double scores[4])
{
    __m256d sums = _mm256_setzero_pd();
    size_t channel = 0;
    for (; channel + 4 <= dim; channel += 4) {
        __m128 k0 = _mm_loadu_ps(keys + channel);
        __m128 k1 = _mm_loadu_ps(keys + dim + channel);
        __m128 k2 = _mm_loadu_ps(keys + 2 * dim + channel);
        __m128 k3 = _mm_loadu_ps(keys + 3 * dim + channel);
        _MM_TRANSPOSE4_PS(k0, k1, k2, k3);
        sums = _mm256_add_pd(sums, _mm256_mul_pd(_mm256_set1_pd(query[channel]), _mm256_cvtps_pd(k0)));
        sums = _mm256_add_pd(sums, _mm256_mul_pd(_mm256_set1_pd(query[channel + 1]), _mm256_cvtps_pd(k1)));
        sums = _mm256_add_pd(sums, _mm256_mul_pd(_mm256_set1_pd(query[channel + 2]), _mm256_cvtps_pd(k2)));
        sums = _mm256_add_pd(sums, _mm256_mul_pd(_mm256_set1_pd(query[channel + 3]), _mm256_cvtps_pd(k3)));
    }
    for (; channel < dim; ++channel) {
        const __m256d k = _mm256_setr_pd(keys[channel], keys[dim + channel],
                                     keys[2 * dim + channel], keys[3 * dim + channel]);
        sums = _mm256_add_pd(sums, _mm256_mul_pd(_mm256_set1_pd(query[channel]), k));
    }
    _mm256_storeu_pd(scores, sums);
}

static __attribute__((target("avx2"))) void attention_row_avx2(
    const float *query, const float *key, const float *value, const float *relative, float sink,
    float *output, double *accumulator, size_t dim, size_t begin, size_t position,
    size_t relative_extent, double scale, size_t ring_capacity)
{
    double max_logit = sink, denom = 1.0;
    for (size_t d = 0; d < dim; ++d) accumulator[d] = 0.0;
    for (size_t first = begin; first <= position;) {
        const size_t physical = ring_capacity ? first % ring_capacity : first;
        size_t available = position - first + 1;
        if (ring_capacity && available > ring_capacity - physical)
            available = ring_capacity - physical;
        double dots[4];
        if (available >= 4) {
            scores_for_four_keys(query, key + physical * dim, dim, dots);
        } else {
            for (size_t lane = 0; lane < available; ++lane) {
                const float *keys = key + (physical + lane) * dim;
                double dot = 0.0;
                for (size_t d = 0; d < dim; ++d) dot += (double)query[d] * keys[d];
                dots[lane] = dot;
            }
        }
        const size_t count = available < 4 ? available : 4;
        for (size_t lane = 0; lane < count; ++lane) {
            const size_t distance = position - first - lane;
            const double score = dots[lane] * scale +
                (distance < relative_extent ? relative[distance] : 0.0);
            const double next_max = fmax(score, max_logit);
            const double old_factor = max_logit == next_max ? 1.0 : exp(max_logit - next_max);
            const double new_factor = score == next_max ? 1.0 : exp(score - next_max);
            denom = denom * old_factor + new_factor;
            const float *values = value + (physical + lane) * dim;
            const __m256d old_vector = _mm256_set1_pd(old_factor), new_vector = _mm256_set1_pd(new_factor);
            size_t d = 0;
            for (; d + 4 <= dim; d += 4) {
                const __m256d previous = _mm256_loadu_pd(accumulator + d);
                const __m256d current = _mm256_cvtps_pd(_mm_loadu_ps(values + d));
                _mm256_storeu_pd(accumulator + d, _mm256_add_pd(
                    _mm256_mul_pd(previous, old_vector), _mm256_mul_pd(current, new_vector)));
            }
            for (; d < dim; ++d)
                accumulator[d] = accumulator[d] * old_factor + (double)values[d] * new_factor;
            max_logit = next_max;
        }
        first += count;
    }
    for (size_t d = 0; d < dim; ++d) output[d] = (float)(accumulator[d] / denom);
}
#endif

#define GQA_RELATIVE_ARGS \
    const float *query, size_t query_elements, const float *key, size_t key_elements, \
    const float *value, size_t value_elements, const float *relative, size_t relative_elements, \
    const float *sink, size_t sink_elements, float *output, size_t output_elements, \
    double *work, size_t work_elements, size_t tokens, size_t heads, size_t kv_heads, \
    size_t dim, size_t capacity, size_t position, size_t relative_extent, size_t window

static inline int finite_cache(const float *data, size_t length)
{
#ifdef CK_GQA_X86_AVX2
    if (length >= 64 && __builtin_cpu_supports("avx2"))
        return finite_cache_avx2(data, length);
#endif
    return ck_checked_finite(data, length);
}

struct gqa_rows {
    const float *query, *key, *value, *relative, *sink;
    float *output;
    double *work;
    size_t heads, group, dim, capacity, position, relative_extent, window;
    double scale;
    int circular, use_row_work;
#ifdef CK_GQA_X86_AVX2
    int use_avx2;
#endif
};

static void evaluate_attention_rows(size_t start, size_t stop, void *opaque)
{
    const struct gqa_rows *task = opaque;
    const float *query = task->query, *key = task->key, *value = task->value;
    const float *relative = task->relative, *sink = task->sink;
    float *output = task->output;
    const size_t heads = task->heads, group = task->group, dim = task->dim;
    const size_t capacity = task->capacity, position = task->position;
    const size_t relative_extent = task->relative_extent, window = task->window;
    const double scale = task->scale;
    const int circular = task->circular;
#ifdef CK_GQA_X86_AVX2
    const int use_avx2 = task->use_avx2;
#endif
    for (size_t row = start; row < stop; ++row) {
        const size_t qpos = position + row / heads;
        const size_t begin = window && qpos >= window ? qpos - window + 1 : 0;
        const float *q = query + row * dim;
        const float *rel = relative + row * relative_extent;
        const size_t cache = ((row % heads) / group) * capacity;
        double *accumulator = task->work + (task->use_row_work ? row * dim : 0);
#ifdef CK_GQA_X86_AVX2
        if (use_avx2 && qpos - begin >= 7) {
            attention_row_avx2(q, key + cache * dim, value + cache * dim, rel, sink[row],
                               output + row * dim, accumulator, dim, begin, qpos, relative_extent, scale, circular ? capacity : 0);
            continue;
        }
#endif
        double max_logit = sink[row], denom = 1.0;
        for (size_t d = 0; d < dim; ++d) accumulator[d] = 0.0;
        for (size_t k = begin; k <= qpos; ++k) {
            const size_t row_index = circular ? k % capacity : k;
            const float *keys = key + (cache + row_index) * dim;
            const float *values = value + (cache + row_index) * dim;
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
}

/* Read the causal span in absolute order, including across a ring wrap. */
static int gqa_relative_sink_checked(GQA_RELATIVE_ARGS, int circular)
{
    if (!heads || heads > 256 || !kv_heads || heads % kv_heads ||
        !dim || dim > 4096 || !capacity || capacity > SIZE_MAX / kv_heads ||
        !tokens || tokens > capacity || position > SIZE_MAX - tokens ||
        (!circular && position > capacity - tokens) || (circular && !window) ||
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

    const size_t first = window && position >= window ? position - window + 1 : 0;
    const size_t visible = position + tokens - first;
    /* The writer commits the entire chunk first, so every query's visible key
     * must fit at once. Otherwise the newest rows can evict an earlier query. */
    if (circular && visible > capacity) return -2;

    size_t workspace = dim;
#ifdef CK_GQA_OPTIONAL_POOL
    const int parallel_work = tokens == 1 ? rows >= 8 && visible >= 128
                                         : rows >= 64 && position + tokens >= 64;
    const int parallel = parallel_work &&
        q_size <= SIZE_MAX / sizeof(double) && work_elements >= q_size &&
        ck_get_num_threads && ck_threadpool_global && ck_threadpool_in_dispatch &&
        ck_threadpool_n_threads && ck_threadpool_parallel_for_n && ck_get_num_threads() > 1;
    if (parallel) workspace = q_size;
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
    const size_t physical = circular ? first % capacity : first;
    const size_t contiguous = visible < capacity - physical ? visible : capacity - physical;
    for (size_t h = 0; h < kv_heads; ++h) {
        const size_t offset = (h * capacity + physical) * dim;
        const size_t prefix = contiguous * dim;
        const size_t suffix = (visible - contiguous) * dim;
        const size_t head_start = h * capacity * dim;
        if (!finite_cache(key + offset, prefix) || !finite_cache(value + offset, prefix) ||
            (suffix && (!finite_cache(key + head_start, suffix) ||
                        !finite_cache(value + head_start, suffix))))
            return -3;
    }

    struct gqa_rows work_rows = {
        .query = query, .key = key, .value = value, .relative = relative, .sink = sink,
        .output = output, .work = work, .heads = heads, .group = heads / kv_heads,
        .dim = dim, .capacity = capacity, .position = position, .relative_extent = relative_extent,
        .window = window, .scale = 1.0 / sqrt((double)dim), .circular = circular,
        .use_row_work = workspace > dim,
#ifdef CK_GQA_X86_AVX2
        .use_avx2 = dim >= 16 && __builtin_cpu_supports("avx2"),
#endif
    };
#ifdef CK_GQA_OPTIONAL_POOL
    if (parallel)
        ck_checked_parallel_for(rows, tokens == 1 ? 1 : 4, evaluate_attention_rows, &work_rows);
    else
#endif
        evaluate_attention_rows(0, rows, &work_rows);
    return 0;
}

#define GQA_RELATIVE_FORWARD(circular) \
    gqa_relative_sink_checked(query, query_elements, key, key_elements, value, value_elements, \
                              relative, relative_elements, sink, sink_elements, output, output_elements, \
                              work, work_elements, tokens, heads, kv_heads, dim, capacity, position, \
                              relative_extent, window, circular)

int ck_gqa_relative_sink_checked_f32(GQA_RELATIVE_ARGS)
{
    return GQA_RELATIVE_FORWARD(0);
}

int ck_gqa_relative_sink_ring_checked_f32(GQA_RELATIVE_ARGS)
{
    return GQA_RELATIVE_FORWARD(1);
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
