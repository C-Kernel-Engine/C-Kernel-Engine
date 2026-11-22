#include "checked_float_buffer.h"
#include "packed_q8_dot_ordered.h"

int linear_rows_q8_0_checked_f32(const float *input, size_t input_elements,
                                 const void *weight, size_t weight_bytes,
                                 float *output, size_t output_elements,
                                 size_t rows, size_t width, size_t channels)
{
    size_t in, out;
    if (!width || width % QK8_0 || width / QK8_0 > SIZE_MAX / sizeof(block_q8_0) ||
        !ck_checked_span(rows, width, width, &in) ||
        !ck_checked_span(rows, channels, channels, &out) ||
        input_elements < in || output_elements < out) return -2;
    const size_t row_bytes = width / QK8_0 * sizeof(block_q8_0);
    if (channels > SIZE_MAX / row_bytes || weight_bytes != channels * row_bytes) return -2;
    if (!weight || (uintptr_t)weight > UINTPTR_MAX - weight_bytes ||
        !ck_checked_region(input, in) || !ck_checked_region(output, out) ||
        ck_checked_overlap(input, in, output, out)) return -1;
    const uintptr_t a = (uintptr_t)output, b = (uintptr_t)weight;
    if (a < b + weight_bytes && b < a + out * sizeof(float)) return -1;
    if (!ck_checked_finite(input, in)) return -3;

    const unsigned char *packed = weight;
    /* Validate every dot first, so invalid scale bytes or overflow leave output untouched. */
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < channels; ++c)
            if (!isfinite(ck_packed_q8_dot_ordered(input + r * width, packed + c * row_bytes, width)))
                return -3;
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < channels; ++c)
            output[r * channels + c] = ck_packed_q8_dot_ordered(input + r * width, packed + c * row_bytes, width);
    return 0;
}
