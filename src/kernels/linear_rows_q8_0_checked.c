#include "checked_float_buffer.h"
#include "packed_q8_dot_ordered.h"
#include "packed_q8_activation_checked.h"

static int linear_rows_q8_0_checked(const float *input, size_t input_elements,
                                    const void *weight, size_t weight_bytes,
                                    float *output, size_t output_elements,
                                    size_t rows, size_t width, size_t channels,
                                    unsigned char *activation, size_t activation_bytes)
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
    if (activation) {
        const uintptr_t scratch = (uintptr_t)activation, src = (uintptr_t)input;
        if (activation_bytes < row_bytes || scratch > UINTPTR_MAX - row_bytes) return -2;
        if ((scratch < src + in * sizeof(float) && src < scratch + row_bytes) ||
            (scratch < a + out * sizeof(float) && a < scratch + row_bytes) ||
            (scratch < b + weight_bytes && b < scratch + row_bytes)) return -1;
    }
    if (!ck_checked_finite(input, in)) return -3;

    const unsigned char *packed = weight;
    /* Preflight all rows before publishing output. Scratch may change on error. */
    for (int publish = 0; publish < 2; ++publish) {
        for (size_t r = 0; r < rows; ++r) {
            const float *x = input + r * width;
            if (activation && ck_pack_q8_activation_rne(x, activation, width)) return -3;
            for (size_t c = 0; c < channels; ++c) {
                const unsigned char *w = packed + c * row_bytes;
                const float dot = activation ? ck_q8_activation_dot(w, activation, width)
                                             : ck_packed_q8_dot_ordered(x, w, width);
                if (!isfinite(dot)) return -3;
                if (publish) output[r * channels + c] = dot;
            }
        }
    }
    return 0;
}

int linear_rows_q8_0_checked_f32(const float *input, size_t input_elements,
                                 const void *weight, size_t weight_bytes,
                                 float *output, size_t output_elements,
                                 size_t rows, size_t width, size_t channels)
{
    return linear_rows_q8_0_checked(input, input_elements, weight, weight_bytes, output,
                                    output_elements, rows, width, channels, NULL, 0);
}

int linear_rows_q8_0_checked_q8_activation(const float *input, size_t input_elements,
                                          const void *weight, size_t weight_bytes,
                                          float *output, size_t output_elements,
                                          size_t rows, size_t width, size_t channels,
                                          unsigned char *activation, size_t activation_bytes)
{
    if (!activation) return -1;
    return linear_rows_q8_0_checked(input, input_elements, weight, weight_bytes, output,
                                    output_elements, rows, width, channels, activation, activation_bytes);
}
