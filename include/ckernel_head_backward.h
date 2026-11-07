#ifndef CK_HEAD_BACKWARD_H
#define CK_HEAD_BACKWARD_H
#include <stddef.h>
#include <stdint.h>

/* FP32 VJPs: overwrite valid output rows; accumulate overlapping contributions
 * inside each VJP. Inter-operation branch accumulation is a separate graph op.
 * Outputs and scratch must be disjoint from all inputs and each other.
 * Rejected calls leave every output untouched; scratch is unspecified. */
int ck_token_span_mean_backward_f32(
    const float *d_output, size_t d_output_elements, size_t d_output_stride,
    const int32_t *spans, size_t span_elements, size_t span_count,
    float *d_input, size_t d_input_elements, size_t d_input_stride,
    float *scratch, size_t scratch_elements,
    size_t token_capacity, size_t valid_tokens, size_t channels);
int ck_gelu_erf_backward_f32(
    const float *input, size_t input_elements, size_t input_stride,
    const float *d_output, size_t d_output_elements, size_t d_output_stride,
    float *d_input, size_t d_input_elements, size_t d_input_stride,
    float *scratch, size_t scratch_elements, size_t rows, size_t channels);
int ck_grouped_weighted_rows_backward_f32(
    const float *options, size_t option_elements, size_t option_stride,
    const float *fields, size_t field_elements, size_t field_stride,
    const int32_t *group_counts, size_t count_elements,
    const float *d_output, size_t d_output_elements, size_t d_output_stride,
    float *d_options, size_t d_options_elements, size_t d_options_stride,
    float *d_fields, size_t d_fields_elements, size_t d_fields_stride,
    float *scratch, size_t scratch_elements,
    size_t groups, size_t option_rows, size_t channels);
#endif
