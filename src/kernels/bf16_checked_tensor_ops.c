#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "bf16_utils.h"
#include "ckernel_engine.h"

static int valid_span(const void *ptr, size_t count, size_t element)
{
    return ptr && (uintptr_t)ptr % element == 0 &&
           count <= (UINTPTR_MAX - (uintptr_t)ptr) / element;
}

static int overlaps(const void *a, size_t abytes, const void *b, size_t bbytes)
{
    return (uintptr_t)a < (uintptr_t)b + bbytes &&
           (uintptr_t)b < (uintptr_t)a + abytes;
}

static int binary_bf16(const uint16_t *left, const uint16_t *right, size_t rhs_count,
                       uint16_t *out, uint16_t *stage, size_t n, int multiply)
{
    if (!n || !rhs_count || n % rhs_count ||
        !valid_span(left, n, 2) || !valid_span(right, rhs_count, 2) ||
        !valid_span(out, n, 2) || !valid_span(stage, n, 2))
        return -2;
    const size_t bytes = n * 2;
    if (overlaps(stage, bytes, left, bytes) || overlaps(stage, bytes, right, rhs_count * 2) ||
        overlaps(stage, bytes, out, bytes) || overlaps(out, bytes, left, bytes) ||
        overlaps(out, bytes, right, rhs_count * 2))
        return -2;

    size_t column = 0;
    for (size_t i = 0; i < n; ++i) {
        const float a = bf16_to_float(left[i]), b = bf16_to_float(right[column]);
        if (++column == rhs_count)
            column = 0;
        const uint16_t value = float_to_bf16(multiply ? a * b : a + b);
        if (!isfinite(a) || !isfinite(b) || !isfinite(bf16_to_float(value)))
            return -3;
        stage[i] = value;
    }
    memcpy(out, stage, bytes);
    return 0;
}

int ck_add_bf16_checked(const uint16_t *left, const uint16_t *right,
                       uint16_t *out, uint16_t *stage, size_t elements)
{
    return binary_bf16(left, right, elements, out, stage, elements, 0);
}

int ck_scale_bf16_checked(const uint16_t *input, const uint16_t *weight,
                         uint16_t *out, uint16_t *stage, size_t elements, size_t gain_elements)
{
    if (gain_elements != 1)
        return -2;
    return binary_bf16(input, weight, 1, out, stage, elements, 1);
}

int ck_scale_rows_bf16_checked(const uint16_t *input, const uint16_t *weight,
                              uint16_t *out, uint16_t *stage, size_t elements, size_t gain_elements)
{
    return binary_bf16(input, weight, gain_elements, out, stage, elements, 1);
}

int ck_widen_bf16_checked(const uint16_t *input, float *out, size_t elements)
{
    if (!elements || !valid_span(input, elements, sizeof(*input)) ||
        !valid_span(out, elements, sizeof(*out)) ||
        overlaps(input, elements * sizeof(*input), out, elements * sizeof(*out)))
        return -2;
    for (size_t i = 0; i < elements; ++i)
        if (!isfinite(bf16_to_float(input[i])))
            return -3;
    for (size_t i = 0; i < elements; ++i)
        out[i] = bf16_to_float(input[i]);
    return 0;
}

int ck_relu2_bf16_checked(const uint16_t *input, uint16_t *out,
                          float *stage, size_t elements)
{
    if (!elements || !valid_span(input, elements, 2) ||
        !valid_span(out, elements, 2) || !valid_span(stage, elements, 4))
        return -2;
    const size_t bytes = elements * sizeof(*input);
    const size_t stage_bytes = elements * sizeof(*stage);
    if (overlaps(input, bytes, out, bytes) || overlaps(input, bytes, stage, stage_bytes) ||
        overlaps(out, bytes, stage, stage_bytes))
        return -2;
    for (size_t i = 0; i < elements; ++i) {
        const float x = bf16_to_float(input[i]);
        const float positive = x > 0.0f ? x : 0.0f;
        stage[i] = positive * positive;
        if (!isfinite(x) || !isfinite(bf16_to_float(float_to_bf16(stage[i]))))
            return -3;
    }
    for (size_t i = 0; i < elements; ++i)
        out[i] = float_to_bf16(stage[i]);
    return 0;
}

static int gated_bf16(const uint16_t *gate, const uint16_t *up, uint16_t *out,
                      float *stage, size_t n, void (*activate)(float *, size_t))
{
    if (!n || !valid_span(gate, n, 2) || !valid_span(up, n, 2) ||
        !valid_span(out, n, 2) || !valid_span(stage, n, 4))
        return -2;
    if (overlaps(stage, n * 4, gate, n * 2) || overlaps(stage, n * 4, up, n * 2) ||
        overlaps(stage, n * 4, out, n * 2) || overlaps(out, n * 2, gate, n * 2) ||
        overlaps(out, n * 2, up, n * 2))
        return -2;
    for (size_t i = 0; i < n; ++i) {
        stage[i] = bf16_to_float(gate[i]);
        if (!isfinite(stage[i]) || !isfinite(bf16_to_float(up[i])))
            return -3;
    }
    activate(stage, n);
    for (size_t i = 0; i < n; ++i) {
        stage[i] *= bf16_to_float(up[i]);
        if (!isfinite(bf16_to_float(float_to_bf16(stage[i]))))
            return -3;
    }
    for (size_t i = 0; i < n; ++i)
        out[i] = float_to_bf16(stage[i]);
    return 0;
}

int ck_geglu_tanh_bf16_checked(const uint16_t *gate, const uint16_t *up,
                             uint16_t *out, float *stage, size_t elements)
{
    return gated_bf16(gate, up, out, stage, elements, gelu_pytorch_tanh_bf16_storage);
}

static void silu_bf16_storage(float *values, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        const float x = values[i];
        values[i] = bf16_to_float(float_to_bf16(x / (1.0f + expf(-x))));
    }
}

int ck_swiglu_bf16_checked(const uint16_t *gate, const uint16_t *up,
                          uint16_t *out, float *stage, size_t elements)
{
    return gated_bf16(gate, up, out, stage, elements, silu_bf16_storage);
}

static void sigmoid_bf16_storage(float *values, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        values[i] = bf16_to_float(float_to_bf16(1.0f / (1.0f + expf(-values[i]))));
}

int ck_sigmoid_mul_bf16_checked(const uint16_t *gate, const uint16_t *up,
                               uint16_t *out, float *stage, size_t elements)
{
    return gated_bf16(gate, up, out, stage, elements, sigmoid_bf16_storage);
}

int ck_rope_split_positions_bf16_checked(
    const uint16_t *input, const int32_t *positions, const float *inv_freq,
    uint16_t *out, uint16_t *stage, size_t tokens, size_t heads, size_t dim,
    size_t elements, size_t frequencies)
{
    if (!tokens || !heads || !dim || dim % 2 || tokens > SIZE_MAX / heads ||
        tokens * heads > SIZE_MAX / dim)
        return -2;
    const size_t count = tokens * heads * dim;
    const size_t half = dim / 2;
    if (elements != count || frequencies != half)
        return -2;
    if (!valid_span(input, count, 2) || !valid_span(out, count, 2) ||
        !valid_span(stage, count, 2) || !valid_span(positions, tokens, 4) ||
        !valid_span(inv_freq, half, 4))
        return -2;
    const size_t bytes = count * 2;
    if (overlaps(stage, bytes, out, bytes) || overlaps(stage, bytes, input, bytes) ||
        overlaps(stage, bytes, positions, tokens * 4) || overlaps(stage, bytes, inv_freq, half * 4) ||
        overlaps(out, bytes, input, bytes) || overlaps(out, bytes, positions, tokens * 4) ||
        overlaps(out, bytes, inv_freq, half * 4))
        return -2;
    for (size_t d = 0; d < half; ++d)
        if (!isfinite(inv_freq[d]) || inv_freq[d] < 0.0f)
            return -3;
    for (size_t t = 0; t < tokens; ++t) {
        if (positions[t] < 0)
            return -2;
        for (size_t d = 0; d < half; ++d) {
            const float angle = (float)positions[t] * inv_freq[d];
            if (!isfinite(angle))
                return -3;
            const float cosine = bf16_to_float(float_to_bf16(cosf(angle)));
            const float sine = bf16_to_float(float_to_bf16(sinf(angle)));
            for (size_t h = 0; h < heads; ++h) {
                const size_t i = (t * heads + h) * dim + d;
                const float x = bf16_to_float(input[i]), y = bf16_to_float(input[i + half]);
                if (!isfinite(x) || !isfinite(y))
                    return -3;
                const float xc = bf16_to_float(float_to_bf16(x * cosine));
                const float ys = bf16_to_float(float_to_bf16(y * sine));
                const float yc = bf16_to_float(float_to_bf16(y * cosine));
                const float xs = bf16_to_float(float_to_bf16(x * sine));
                stage[i] = float_to_bf16(xc - ys);
                stage[i + half] = float_to_bf16(yc + xs);
                if (!isfinite(bf16_to_float(stage[i])) || !isfinite(bf16_to_float(stage[i + half])))
                    return -3;
            }
        }
    }
    memcpy(out, stage, bytes);
    return 0;
}

int ck_rope_interleaved_partial_bf16_checked(const uint16_t *input, const int32_t *positions,
                                            const float *inv_freq, uint16_t *out, uint16_t *stage,
                                            size_t tokens, size_t heads, size_t dim,
                                            size_t elements, size_t frequencies)
{
    if (!tokens || !heads || !dim || !frequencies || frequencies > dim / 2 ||
        tokens > SIZE_MAX / heads || tokens * heads > SIZE_MAX / dim ||
        elements != tokens * heads * dim)
        return -2;
    if (!valid_span(input, elements, 2) || !valid_span(out, elements, 2) ||
        !valid_span(stage, elements, 2) || !valid_span(positions, tokens, 4) ||
        !valid_span(inv_freq, frequencies, 4))
        return -2;
    const size_t bytes = elements * 2;
    if (overlaps(out, bytes, stage, bytes))
        return -2;
    const void *operands[] = {input, positions, inv_freq};
    const size_t sizes[] = {bytes, tokens * 4, frequencies * 4};
    for (size_t i = 0; i < 3; ++i)
        if (overlaps(out, bytes, operands[i], sizes[i]) ||
            overlaps(stage, bytes, operands[i], sizes[i]))
            return -2;

    for (size_t i = 0; i < elements; ++i)
        if (!isfinite(bf16_to_float(input[i])))
            return -3;
    memcpy(stage, input, bytes);
    for (size_t token = 0; token < tokens; ++token) {
        if (positions[token] < 0)
            return -2;
        for (size_t pair = 0; pair < frequencies; ++pair) {
            const float angle = (float)positions[token] * inv_freq[pair];
            if (!isfinite(angle) || inv_freq[pair] < 0.0f)
                return -3;
            const float cosine = cosf(angle), sine = sinf(angle);
            for (size_t head = 0; head < heads; ++head) {
                const size_t i = (token * heads + head) * dim + pair * 2;
                const float x = bf16_to_float(input[i]), y = bf16_to_float(input[i + 1]);
                const volatile float xc = x * cosine, ys = y * sine;
                const volatile float yc = y * cosine, xs = x * sine;
                const uint16_t even = float_to_bf16(xc - ys), odd = float_to_bf16(yc + xs);
                if (!isfinite(bf16_to_float(even)) || !isfinite(bf16_to_float(odd)))
                    return -3;
                stage[i] = even;
                stage[i + 1] = odd;
            }
        }
    }
    memcpy(out, stage, bytes);
    return 0;
}

int ck_scale_scalar_f32_bf16_checked(const uint16_t *input, uint16_t *out,
                                    uint16_t *stage, size_t elements, float scale)
{
    if (!elements || !isfinite(scale) || !valid_span(input, elements, 2) ||
        !valid_span(out, elements, 2) || !valid_span(stage, elements, 2))
        return -2;
    const size_t bytes = elements * 2;
    if (overlaps(input, bytes, out, bytes) || overlaps(input, bytes, stage, bytes) ||
        overlaps(stage, bytes, out, bytes))
        return -2;
    for (size_t i = 0; i < elements; ++i) {
        const float value = bf16_to_float(input[i]);
        stage[i] = float_to_bf16(value * scale);
        if (!isfinite(value) || !isfinite(bf16_to_float(stage[i])))
            return -3;
    }
    memcpy(out, stage, bytes);
    return 0;
}

int ck_argmax_values_bf16_checked(const uint16_t *values, int32_t *index,
                                  size_t elements, size_t index_elements)
{
    if (!elements || elements > INT32_MAX || index_elements != 1 || !valid_span(values, elements, 2) ||
        !valid_span(index, 1, 4) || overlaps(values, elements * 2, index, 4))
        return -2;
    size_t best = 0;
    float maximum = -INFINITY;
    for (size_t i = 0; i < elements; ++i) {
        const float value = bf16_to_float(values[i]);
        if (!isfinite(value))
            return -3;
        if (value > maximum) {
            maximum = value;
            best = i;
        }
    }
    *index = (int32_t)best;
    return 0;
}
