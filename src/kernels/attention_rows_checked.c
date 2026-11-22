#include "checked_float_buffer.h"

int scaled_head_l2_clip_checked_f32(const float *input, size_t input_elements,
                                      const float *gamma, size_t gamma_elements,
                                      float *output, size_t output_elements,
                                      size_t channels, size_t rows, size_t heads, size_t dim,
                                      float cap, float epsilon)
{
    if (!heads || !dim || heads > SIZE_MAX / dim ||
        !isfinite(cap) || cap <= 0.0f || !isfinite(epsilon) || epsilon <= 0.0f) return -2;
    const size_t width = heads * dim;
    if (channels != width) return -2;
    size_t active;
    if (!ck_checked_span(rows, width, width, &active) ||
        input_elements < active || output_elements < active || gamma_elements < width) return -2;
    if (!ck_checked_region(input, active) || !ck_checked_region(output, active) ||
        !ck_checked_region(gamma, width) ||
        ck_checked_overlap(input, active, output, active) ||
        ck_checked_overlap(gamma, width, output, active)) return -1;
    if (!ck_checked_finite(gamma, width)) return -3;
    for (size_t i = 0; i < active; ++i)
        if (!isfinite(input[i] * gamma[i % width])) return -3;

    for (size_t row = 0; row < rows; ++row) {
        for (size_t h = 0; h < heads; ++h) {
            const size_t col = h * dim;
            const size_t start = row * width + col;
            double squared_norm = 0.0;
            for (size_t d = 0; d < dim; ++d) {
                const float scaled = input[start + d] * gamma[col + d];
                const volatile double squared = (double)scaled * scaled;
                squared_norm += squared;
            }
            const double scale = fmin(1.0, cap / (sqrt(squared_norm) + epsilon));
            for (size_t d = 0; d < dim; ++d) {
                const float scaled = input[start + d] * gamma[col + d];
                output[start + d] = (float)(scaled * scale);
            }
        }
    }
    return 0;
}

int sigmoid_gate_rows_checked_f32(const float *input, size_t input_elements,
                                  const float *gate, size_t gate_elements,
                                  float *output, size_t output_elements, size_t elements)
{
    if (!elements || input_elements < elements || gate_elements < elements ||
        output_elements < elements) return -2;
    if (!ck_checked_region(input, elements) || !ck_checked_region(gate, elements) ||
        !ck_checked_region(output, elements) ||
        ck_checked_overlap(input, elements, output, elements) ||
        ck_checked_overlap(gate, elements, output, elements)) return -1;
    if (!ck_checked_finite(input, elements) || !ck_checked_finite(gate, elements)) return -3;
    for (size_t i = 0; i < elements; ++i) {
        const float ex = expf(-fabsf(gate[i]));
        const float sigmoid = (gate[i] < 0.0f ? ex : 1.0f) / (1.0f + ex);
        output[i] = input[i] * sigmoid;
    }
    return 0;
}
