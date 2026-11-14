#include "checked_float_buffer.h"

/* Caller-supplied inverse frequencies make their construction/storage an
 * explicit model artifact. Phase is two FP32 multiplies, without FMA. */
static int tables(const float *positions, size_t position_elements,
    const float *frequencies, size_t frequency_elements, float scale,
    float *sine, size_t sine_elements, float *cosine, size_t cosine_elements,
    size_t rows, size_t columns, size_t stride, int combined)
{
    size_t n;
    if (!rows || !columns || position_elements < rows || frequency_elements < columns ||
        !ck_checked_span(rows, columns, stride, &n) || sine_elements < n || cosine_elements < n)
        return -2;
    if (!isfinite(scale) || !ck_checked_region(positions, rows) ||
        !ck_checked_region(frequencies, columns) || !ck_checked_region(sine, n) ||
        !ck_checked_region(cosine, n) ||
        ck_checked_overlap(positions, rows, sine, n) || ck_checked_overlap(positions, rows, cosine, n) ||
        ck_checked_overlap(frequencies, columns, sine, n) || ck_checked_overlap(frequencies, columns, cosine, n) ||
        (!combined && ck_checked_overlap(sine, n, cosine, n))) return -1;
    if (!ck_checked_finite(positions, rows) || !ck_checked_finite(frequencies, columns)) return -3;
    for (size_t row = 0; row < rows; ++row) {
        float position = positions[row] * scale;
        if (!isfinite(position)) return -3;
        for (size_t col = 0; col < columns; ++col)
            if (!isfinite(position * frequencies[col])) return -3;
    }
    for (size_t row = 0; row < rows; ++row) {
        float position = positions[row] * scale;
        for (size_t col = 0; col < columns; ++col) {
            float phase = position * frequencies[col];
            sine[row * stride + col] = sinf(phase);
            cosine[row * stride + col] = cosf(phase);
        }
    }
    return 0;
}

int sinusoidal_tables_f32_checked(const float *positions, size_t position_elements,
    const float *frequencies, size_t frequency_elements, float scale,
    float *sine, size_t sine_elements, float *cosine, size_t cosine_elements,
    size_t rows, size_t columns)
{
    return tables(positions, position_elements, frequencies, frequency_elements, scale,
        sine, sine_elements, cosine, cosine_elements, rows, columns, columns, 0);
}

int sinusoidal_features_f32_checked(const float *positions, size_t position_elements,
    const float *frequencies, size_t frequency_elements, float scale,
    float *output, size_t output_elements, size_t rows, size_t columns)
{
    size_t n;
    if (columns > SIZE_MAX / 2 || !ck_checked_span(rows, 2 * columns, 2 * columns, &n) ||
        output_elements < n) return -2;
    if (!ck_checked_region(output, n)) return -1;
    return tables(positions, position_elements, frequencies, frequency_elements, scale,
        output, n - columns, output + columns, n - columns, rows, columns, 2 * columns, 1);
}
