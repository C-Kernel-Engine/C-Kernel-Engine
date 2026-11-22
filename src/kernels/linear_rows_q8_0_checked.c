#include "checked_float_buffer.h"
#include "packed_q8_dot_ordered.h"
#include "packed_q8_activation_checked.h"

#include <string.h>

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
    size_t scratch_bytes = row_bytes;
    int stage = 0;
    if (activation) {
        if (rows <= (SIZE_MAX - out * sizeof(float)) / row_bytes) {
            const size_t staged_bytes = rows * row_bytes + out * sizeof(float);
            if (activation_bytes >= staged_bytes) {
                scratch_bytes = staged_bytes;
                stage = 1;
            }
        }
        const uintptr_t scratch = (uintptr_t)activation, src = (uintptr_t)input;
        if (activation_bytes < scratch_bytes || scratch > UINTPTR_MAX - scratch_bytes) return -2;
        if ((scratch < src + in * sizeof(float) && src < scratch + scratch_bytes) ||
            (scratch < a + out * sizeof(float) && a < scratch + scratch_bytes) ||
            (scratch < b + weight_bytes && b < scratch + scratch_bytes)) return -1;
    }
    if (!ck_checked_finite(input, in)) return -3;

    const unsigned char *packed = weight;
    if (stage) {
        unsigned char *staged = activation + rows * row_bytes;
        for (size_t r = 0; r < rows; ++r) {
            if (ck_pack_q8_activation_rne(input + r * width, activation + r * row_bytes, width))
                return -3;
        }
        int failed = 0;
#ifdef _OPENMP
        /* A single vocabulary row has enough independent outputs to split. */
#pragma omp parallel for collapse(2) if(out >= 8192) reduction(|:failed)
#endif
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < channels; ++c) {
                const float dot = ck_q8_activation_dot(packed + c * row_bytes,
                                                       activation + r * row_bytes, width);
                failed |= !isfinite(dot);
                memcpy(staged + (r * channels + c) * sizeof(float), &dot, sizeof(dot));
            }
        }
        if (failed) return -3;
        memcpy(output, staged, out * sizeof(float));
        return 0;
    }
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
