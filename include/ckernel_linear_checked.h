#ifndef CKE_LINEAR_CHECKED_H
#define CKE_LINEAR_CHECKED_H

#include <stddef.h>

/* Row-major Y = X W^T + bias. Pass NULL bias with zero bias_elements for a
 * bias-free projection. All buffers are caller owned and nonaliasing.
 * Physical row strides may exceed logical widths. Rejection returns -1 before
 * any output write. This inference provider accumulates in ordered FP64 without FMA.
 * Available SIMD lanes cover independent rows or output channels, preserving
 * reduction order. Weight validation retains per-output ordered FP64 L1 sums. */
int linear_rows_checked_f32(
    const float *input, size_t input_elements, size_t input_stride,
    const float *weight, size_t weight_elements, size_t weight_stride,
    const float *bias, size_t bias_elements,
    float *output, size_t output_elements, size_t output_stride,
    size_t rows, size_t input_channels, size_t output_channels);

#endif
