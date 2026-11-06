#include <math.h>
#include <stdint.h>
#include <stddef.h>

static int region_ok(const void *ptr, size_t count, size_t width) {
    return ptr && (uintptr_t)ptr % width == 0 && count <= SIZE_MAX / width &&
           (uintptr_t)ptr <= UINTPTR_MAX - count * width;
}

static int regions_overlap(const void *left, size_t left_bytes,
                           const void *right, size_t right_bytes) {
    uintptr_t a = (uintptr_t)left, b = (uintptr_t)right;
    return a < b + right_bytes && b < a + left_bytes;
}

/* Select the final valid token. The span is checked as a semantic assertion,
 * so an externally supplied, in-bounds but incorrect span cannot redirect it. */
int ck_last_valid_row_checked_f32(
    const float *input, size_t input_elements, size_t input_stride,
    size_t token_capacity, size_t valid_tokens, size_t channels,
    const int32_t *span, size_t span_elements,
    float *output, size_t output_elements) {
    if (!token_capacity || !valid_tokens || valid_tokens > token_capacity ||
        valid_tokens > INT32_MAX || !channels || input_stride < channels ||
        (token_capacity - 1) > (SIZE_MAX - channels) / input_stride ||
        input_elements < (token_capacity - 1) * input_stride + channels ||
        span_elements < 2 || output_elements < channels)
        return -2;
    if (!region_ok(input, input_elements, sizeof(float)) ||
        !region_ok(span, span_elements, sizeof(int32_t)) ||
        !region_ok(output, output_elements, sizeof(float)))
        return -1;
    if (regions_overlap(input, input_elements * sizeof(float),
                        output, output_elements * sizeof(float)) ||
        regions_overlap(span, span_elements * sizeof(int32_t),
                        output, output_elements * sizeof(float)))
        return -1;
    if (span[0] != (int32_t)(valid_tokens - 1) ||
        span[1] != (int32_t)valid_tokens)
        return -2;
    const float *last = input + (valid_tokens - 1) * input_stride;
    for (size_t c = 0; c < channels; ++c)
        if (!isfinite(last[c])) return -1;
    for (size_t c = 0; c < channels; ++c)
        output[c] = last[c];
    return 0;
}
