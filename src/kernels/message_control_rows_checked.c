#include "checked_float_buffer.h"
#include <float.h>
#include <string.h>

/* No learned scaling here. Give each operand its own RMS before concatenation. */
int rms_pair_concat_rows_checked_f32(const float *left, size_t left_elements,
                                     const float *right, size_t right_elements,
                                     float *output, size_t output_elements, size_t rows,
                                     size_t width, size_t joined_width, float epsilon)
{
    size_t elements, joined;
    if (!width || width > SIZE_MAX / 2 || joined_width != width * 2 ||
        !isfinite(epsilon) || epsilon <= 0.0f ||
        !ck_checked_span(rows, width, width, &elements) ||
        !ck_checked_span(rows, joined_width, joined_width, &joined) ||
        left_elements < elements || right_elements < elements || output_elements < joined)
        return -2;
    if (!ck_checked_region(left, elements) || !ck_checked_region(right, elements) ||
        !ck_checked_region(output, joined) ||
        ck_checked_overlap(left, elements, output, joined) ||
        ck_checked_overlap(right, elements, output, joined)) return -1;
    if (!ck_checked_finite(left, elements) || !ck_checked_finite(right, elements)) return -3;

    for (size_t row = 0; row < rows; ++row) {
        const float *inputs[] = {left + row * width, right + row * width};
        for (size_t side = 0; side < 2; ++side) {
            const float *x = inputs[side];
            double squared = 0.0;
            for (size_t i = 0; i < width; ++i) squared += (double)x[i] * x[i];
            const double scale = 1.0 / sqrt(squared / (double)width + epsilon);
            for (size_t i = 0; i < width; ++i)
                output[row * joined_width + side * width + i] = (float)(x[i] * scale);
        }
    }
    return 0;
}

static double mix(float input, float message, float forget, float gain)
{
    return (1.0 - (double)forget) * input + (double)gain * message;
}

int message_control_rows_checked_f32(const float *input, size_t input_elements,
                                      const float *message, size_t message_elements,
                                      const float *forget, size_t forget_elements,
                                      const float *gain, size_t gain_elements,
                                      float *output, size_t output_elements, size_t elements)
{
    if (!elements || input_elements < elements || message_elements < elements ||
        forget_elements < elements || gain_elements < elements || output_elements < elements)
        return -2;
    if (!ck_checked_region(output, elements)) return -1;
    const float *inputs[] = {input, message, forget, gain};
    for (size_t i = 0; i < sizeof(inputs) / sizeof(inputs[0]); ++i) {
        if (!ck_checked_region(inputs[i], elements) ||
            ck_checked_overlap(inputs[i], elements, output, elements)) return -1;
        if (!ck_checked_finite(inputs[i], elements)) return -3;
    }
    for (size_t i = 0; i < elements; ++i) {
        if (fabsf(forget[i]) > 1.0f || fabsf(gain[i]) > 1.0f ||
            fabs(mix(input[i], message[i], forget[i], gain[i])) > FLT_MAX) return -3;
    }
    for (size_t i = 0; i < elements; ++i)
        output[i] = (float)mix(input[i], message[i], forget[i], gain[i]);
    return 0;
}

int concat_feature_rows_checked_f32(const float *left, size_t left_elements,
                                     const float *right, size_t right_elements,
                                     float *output, size_t output_elements,
                                     size_t rows, size_t left_width, size_t right_width, size_t width)
{
    size_t l, r, o;
    if (!left_width || !right_width || left_width > SIZE_MAX - right_width ||
        width != left_width + right_width ||
        !ck_checked_span(rows, left_width, left_width, &l) ||
        !ck_checked_span(rows, right_width, right_width, &r) ||
        !ck_checked_span(rows, width, width, &o) ||
        left_elements < l || right_elements < r || output_elements < o) return -2;
    if (!ck_checked_region(left, l) || !ck_checked_region(right, r) ||
        !ck_checked_region(output, o) || ck_checked_overlap(left, l, output, o) ||
        ck_checked_overlap(right, r, output, o)) return -1;
    if (!ck_checked_finite(left, l) || !ck_checked_finite(right, r)) return -3;
    for (size_t row = 0; row < rows; ++row) {
        memcpy(output + row * width, left + row * left_width, left_width * sizeof(float));
        memcpy(output + row * width + left_width, right + row * right_width, right_width * sizeof(float));
    }
    return 0;
}
