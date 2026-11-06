#include "ckernel_token_span_mean.h"

#include <math.h>
#include <stdint.h>

static int valid_region(const void *ptr, size_t elements, size_t bytes,
                        size_t alignment) {
    if (!ptr || (uintptr_t)ptr % alignment || elements > SIZE_MAX / bytes)
        return 0;
    return (uintptr_t)ptr <= UINTPTR_MAX - elements * bytes;
}

static int overlaps(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x < y + b_bytes && y < x + a_bytes;
}

int ck_token_span_mean_checked_f32(
    const float *input, size_t input_elements, size_t input_stride,
    size_t token_capacity, size_t valid_tokens, size_t channels,
    const int32_t *spans, size_t span_elements, size_t span_count,
    float *output, size_t output_elements, size_t output_stride) {
    if (!token_capacity || !valid_tokens || valid_tokens > token_capacity ||
        !channels || !span_count || input_stride < channels ||
        output_stride < channels || span_count > SIZE_MAX / 2)
        return -2;
    if (token_capacity - 1 > (SIZE_MAX - channels) / input_stride ||
        input_elements < (token_capacity - 1) * input_stride + channels ||
        span_elements < span_count * 2 ||
        span_count - 1 > (SIZE_MAX - channels) / output_stride ||
        output_elements < (span_count - 1) * output_stride + channels)
        return -2;
    if (!valid_region(input, input_elements, sizeof(float), _Alignof(float)) ||
        !valid_region(spans, span_elements, sizeof(int32_t), _Alignof(int32_t)) ||
        !valid_region(output, output_elements, sizeof(float), _Alignof(float)))
        return -1;
    if (overlaps(input, input_elements * sizeof(float), output,
                 output_elements * sizeof(float)) ||
        overlaps(spans, span_elements * sizeof(int32_t), output,
                 output_elements * sizeof(float)))
        return -1;

    /* Validate every result before publishing any output. The second pass uses
       the same order and leaves padding untouched. */
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t s = 0; s < span_count; ++s) {
            int32_t start = spans[2 * s], end = spans[2 * s + 1];
            if (start < 0 || end <= start || (size_t)end > valid_tokens)
                return -2;
            for (size_t c = 0; c < channels; ++c) {
                double sum = 0.0;
                for (size_t t = (size_t)start; t < (size_t)end; ++t) {
                    float value = input[t * input_stride + c];
                    if (!isfinite(value)) return -1;
                    sum += (double)value;
                }
                float mean = (float)(sum / (double)(end - start));
                if (!isfinite(mean)) return -1;
                if (pass) output[s * output_stride + c] = mean;
            }
        }
    }
    return 0;
}
