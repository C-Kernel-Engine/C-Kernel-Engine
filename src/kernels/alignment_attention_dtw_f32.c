/* Generic attention-to-token-frame alignment. Caller owns every buffer.
 * Selected heads are already packed into [head, token, frame] storage.
 * The normalization/median/DTW order matches the pinned Whisper reference;
 * no model identity or tokenizer semantics enter this operation. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static int mul_size(size_t a, size_t b, size_t *out) {
    if (a && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int overlaps(const void *a, size_t na, const void *b, size_t nb) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    if (na > UINTPTR_MAX - x || nb > UINTPTR_MAX - y) return 1;
    return x < y + nb && y < x + na;
}

static size_t reflect_index(int64_t index, size_t width) {
    const int64_t last = (int64_t)width - 1;
    while (index < 0 || index > last) {
        if (index < 0) index = -index;
        if (index > last) index = 2 * last - index;
    }
    return (size_t)index;
}

/* Return 0 on success. A rejected call never changes token_frames.
 * score/mean/cost/trace capacities are element counts, not bytes. */
int alignment_attention_dtw_f32(
    const float *attention, size_t attention_elements,
    size_t heads, size_t tokens, size_t frames,
    size_t token_stride, size_t head_stride, size_t median_width,
    float *score_scratch, size_t score_elements,
    float *mean_scratch, size_t mean_elements,
    float *cost_scratch, size_t cost_elements,
    uint8_t *trace_scratch, size_t trace_elements,
    int32_t *token_frames, size_t token_capacity)
{
    size_t matrix, cells, input_required, row_required, bytes_matrix;
    if (!attention || !score_scratch || !mean_scratch || !cost_scratch ||
        !trace_scratch || !token_frames || !heads || !tokens || !frames ||
        tokens > INT32_MAX || frames > INT32_MAX ||
        frames > INT64_MAX / 2 || !median_width || !(median_width & 1) ||
        median_width > 31 || token_stride < frames ||
        !mul_size(tokens, frames, &matrix) ||
        !mul_size(tokens + 1, frames + 1, &cells) ||
        !mul_size(heads - 1, head_stride, &input_required) ||
        !mul_size(tokens - 1, token_stride, &row_required) ||
        row_required > SIZE_MAX - frames ||
        input_required > SIZE_MAX - row_required - frames ||
        !mul_size(matrix, sizeof(float), &bytes_matrix)) return -1;
    input_required += row_required + frames;
    if (head_stride < row_required + frames || attention_elements < input_required ||
        score_elements < matrix || mean_elements < matrix ||
        cost_elements < cells || trace_elements < cells ||
        token_capacity < tokens) return -2;
    size_t input_bytes, cost_bytes, output_bytes;
    if (!mul_size(attention_elements, sizeof(float), &input_bytes) ||
        !mul_size(cost_elements, sizeof(float), &cost_bytes) ||
        !mul_size(token_capacity, sizeof(int32_t), &output_bytes)) return -1;
    const void *ptrs[] = {attention, score_scratch, mean_scratch,
                          cost_scratch, trace_scratch, token_frames};
    const size_t lengths[] = {input_bytes, bytes_matrix, bytes_matrix,
                              cost_bytes, trace_elements, output_bytes};
    for (size_t i = 0; i < 6; ++i)
        for (size_t j = i + 1; j < 6; ++j)
            if (overlaps(ptrs[i], lengths[i], ptrs[j], lengths[j])) return -3;
    for (size_t h = 0; h < heads; ++h)
        for (size_t t = 0; t < tokens; ++t)
            for (size_t f = 0; f < frames; ++f)
                if (!isfinite(attention[h * head_stride + t * token_stride + f]))
                    return -4;

    for (size_t n = 0; n < matrix; ++n) mean_scratch[n] = 0.0f;
    const int filter = frames > median_width / 2;
    for (size_t h = 0; h < heads; ++h) {
        const float *head = attention + h * head_stride;
        for (size_t f = 0; f < frames; ++f) {
            double mean = 0.0, variance = 0.0;
            for (size_t t = 0; t < tokens; ++t)
                mean += head[t * token_stride + f];
            mean /= (double)tokens;
            for (size_t t = 0; t < tokens; ++t) {
                double delta = (double)head[t * token_stride + f] - mean;
                variance += delta * delta;
            }
            variance /= (double)tokens;
            if (!(variance > 0.0) || !isfinite(variance)) return -5;
            const double inverse = 1.0 / sqrt(variance);
            for (size_t t = 0; t < tokens; ++t)
                score_scratch[t * frames + f] =
                    (float)(((double)head[t * token_stride + f] - mean) * inverse);
        }
        for (size_t t = 0; t < tokens; ++t) {
            for (size_t f = 0; f < frames; ++f) {
                float value;
                if (!filter) value = score_scratch[t * frames + f];
                else {
                    float window[31];
                    const int64_t half = (int64_t)(median_width / 2);
                    for (size_t k = 0; k < median_width; ++k) {
                        size_t source = reflect_index((int64_t)f + (int64_t)k - half, frames);
                        float item = score_scratch[t * frames + source];
                        size_t m = k;
                        while (m && window[m - 1] > item) {
                            window[m] = window[m - 1];
                            --m;
                        }
                        window[m] = item;
                    }
                    value = window[median_width / 2];
                }
                mean_scratch[t * frames + f] += value / (float)heads;
            }
        }
    }
    for (size_t n = 0; n < matrix; ++n)
        if (!isfinite(mean_scratch[n])) return -5;

    const size_t stride = frames + 1;
    for (size_t n = 0; n < cells; ++n) {
        cost_scratch[n] = INFINITY;
        trace_scratch[n] = 255;
    }
    cost_scratch[0] = 0.0f;
    for (size_t f = 1; f <= frames; ++f) {
        for (size_t t = 1; t <= tokens; ++t) {
            const float diagonal = cost_scratch[(t - 1) * stride + f - 1];
            const float vertical = cost_scratch[(t - 1) * stride + f];
            const float horizontal = cost_scratch[t * stride + f - 1];
            uint8_t direction;
            float prior;
            if (diagonal < vertical && diagonal < horizontal) {
                direction = 0; prior = diagonal;
            } else if (vertical < diagonal && vertical < horizontal) {
                direction = 1; prior = vertical;
            } else {
                direction = 2; prior = horizontal;
            }
            const size_t cell = t * stride + f;
            cost_scratch[cell] = -mean_scratch[(t - 1) * frames + f - 1] + prior;
            trace_scratch[cell] = direction;
        }
    }
    if (!isfinite(cost_scratch[tokens * stride + frames])) return -5;
    for (size_t t = 0; t < tokens; ++t) token_frames[t] = -1;
    size_t t = tokens, f = frames;
    while (t || f) {
        if (t) token_frames[t - 1] = (int32_t)(f ? f - 1 : 0);
        uint8_t direction = t && f ? trace_scratch[t * stride + f] :
                            t ? 1 : 2;
        if (direction == 0) { --t; --f; }
        else if (direction == 1) --t;
        else --f;
    }
    return 0;
}
