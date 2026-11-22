#include "ckernel_normalization_checked.h"
#include "ckernel_engine.h"
#include "checked_float_buffer.h"
#include <limits.h>
#include <string.h>

int layernorm_rows_checked_workspace(size_t rows, size_t channels, size_t *elements) {
    /* Keep the legacy provider's signed loop indices below addition overflow. */
    if (!elements || !rows || !channels || channels > (size_t)INT_MAX - 8 ||
        channels > SIZE_MAX - 2 || rows > SIZE_MAX / (channels + 2) ||
        rows * (channels + 2) > SIZE_MAX / sizeof(float)) return -2;
    *elements = rows * (channels + 2);
    return 0;
}

int layernorm_rows_checked_f32(const float *input, size_t input_elements, size_t input_stride,
    const float *gamma, size_t gamma_elements, const float *beta, size_t beta_elements,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements, size_t rows, size_t channels, float epsilon) {
    size_t ni, no, ns;
    if (layernorm_rows_checked_workspace(rows, channels, &ns) ||
        !ck_checked_span(rows, channels, input_stride, &ni) ||
        !ck_checked_span(rows, channels, output_stride, &no) ||
        input_elements < ni || output_elements < no || gamma_elements < channels ||
        beta_elements < channels || scratch_elements < ns) return -2;
    if (!isfinite(epsilon) || epsilon <= 0 || !ck_checked_region(input, ni) ||
        !ck_checked_region(output, no) || !ck_checked_region(gamma, channels) ||
        !ck_checked_region(beta, channels) || !ck_checked_region(scratch, ns)) return -1;
    if (ck_checked_overlap(output, no, input, ni) ||
        ck_checked_overlap(output, no, gamma, channels) ||
        ck_checked_overlap(output, no, beta, channels) ||
        ck_checked_overlap(output, no, scratch, ns) ||
        ck_checked_overlap(scratch, ns, input, ni) ||
        ck_checked_overlap(scratch, ns, gamma, channels) ||
        ck_checked_overlap(scratch, ns, beta, channels)) return -1;
    if (!ck_checked_finite(gamma, channels) || !ck_checked_finite(beta, channels)) return -3;
    for (size_t r = 0; r < rows; ++r)
        if (!ck_checked_finite(input + r * input_stride, channels)) return -3;
    float *means = scratch + rows * channels;
    float *rstd = means + rows;
    for (size_t r = 0; r < rows; ++r) {
        ck_layernorm_row_fixed_chunk4_f32(input + r * input_stride, gamma, beta,
            scratch + r * channels, means + r, rstd + r, (int)channels, epsilon);
        if (!isfinite(means[r]) || !isfinite(rstd[r]) ||
            !ck_checked_finite(scratch + r * channels, channels)) return -4;
    }
    for (size_t r = 0; r < rows; ++r)
        memcpy(output + r * output_stride, scratch + r * channels, channels * sizeof(float));
    return 0;
}
