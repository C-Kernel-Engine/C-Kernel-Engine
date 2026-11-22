#include "checked_float_buffer.h"
#include <float.h>

static double position_scale(size_t position, size_t floor, float alpha)
{
    return position < floor ? 1.0 : 1.0 + (double)alpha * log(((double)position + 1.0) / (double)floor);
}

int log_position_scale_rows_checked_f32(const float *input, size_t input_elements,
                                        float *output, size_t output_elements, size_t rows,
                                        size_t width, size_t position, size_t floor, float alpha)
{
    size_t extent;
    if (!floor || !isfinite(alpha) || alpha < 0.0f || !rows || position > SIZE_MAX - (rows - 1) ||
        !ck_checked_span(rows, width, width, &extent) ||
        input_elements < extent || output_elements < extent) return -2;
    if (!ck_checked_region(input, extent) || !ck_checked_region(output, extent) ||
        ck_checked_overlap(input, extent, output, extent)) return -1;
    if (!ck_checked_finite(input, extent)) return -3;
    for (size_t row = 0; row < rows; ++row) {
        const double scale = position_scale(position + row, floor, alpha);
        for (size_t i = 0; i < width; ++i)
            if (fabs((double)input[row * width + i] * scale) > FLT_MAX) return -3;
    }
    for (size_t row = 0; row < rows; ++row) {
        const double scale = position_scale(position + row, floor, alpha);
        for (size_t i = 0; i < width; ++i)
            output[row * width + i] = (float)((double)input[row * width + i] * scale);
    }
    return 0;
}
