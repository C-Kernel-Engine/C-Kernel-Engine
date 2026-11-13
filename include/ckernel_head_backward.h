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
/* Full unmasked token-major attention VJP. Packed heads; independent query and
 * memory lengths. Scratch: Q*H*D + 2*S*H*D + 2*S floats. Three gradients are
 * overwritten atomically; caller owns branch accumulation. */
int ck_attention_cross_backward_f32(
    const float *query, size_t query_elements,
    const float *key, size_t key_elements,
    const float *value, size_t value_elements,
    const float *d_output, size_t d_output_elements,
    float *d_query, size_t d_query_elements,
    float *d_key, size_t d_key_elements,
    float *d_value, size_t d_value_elements,
    float *scratch, size_t scratch_elements,
    size_t query_tokens, size_t memory_tokens, size_t heads, size_t head_dim);

/* Indexed scoring VJPs. Contiguous operands, owner index per option row.
 * Scratch: groups*channels + 2*rows*channels floats. Overwrite both gradients
 * atomically; repeated owners accumulate in ascending option-row order.
 * normalized_dot: clamp derivative zero below epsilon, one at equality.
 * cosine_rows: PyTorch 2.12.1 cosine_similarity NoGrad norm-clamp semantics.
 * pair_features: [a,b,a*b,abs(a-b)], abs derivative zero at equality. */
int ck_indexed_pair_features_backward_f32(
    const float *left, size_t left_elements, const float *right, size_t right_elements,
    const int32_t *indices, size_t indices_elements, const float *d_output, size_t d_output_elements,
    float *d_left, size_t d_left_elements, float *d_right, size_t d_right_elements,
    float *scratch, size_t scratch_elements, size_t groups, size_t rows, size_t channels);
int ck_indexed_normalized_dot_backward_f32(
    const float *left, size_t left_elements, const float *right, size_t right_elements,
    const int32_t *indices, size_t indices_elements, const float *d_output, size_t d_output_elements,
    float *d_left, size_t d_left_elements, float *d_right, size_t d_right_elements,
    float *scratch, size_t scratch_elements, size_t groups, size_t rows, size_t channels, float epsilon);
int ck_indexed_cosine_rows_backward_f32(
    const float *left, size_t left_elements, const float *right, size_t right_elements,
    const int32_t *indices, size_t indices_elements, const float *d_output, size_t d_output_elements,
    float *d_left, size_t d_left_elements, float *d_right, size_t d_right_elements,
    float *scratch, size_t scratch_elements, size_t groups, size_t rows, size_t channels, float epsilon);

/* BF16 storage VJPs. Saved operands, incoming and outgoing gradients use
 * BF16; scratch uses FP32. Exact expansion, existing FP32 VJP arithmetic,
 * then round-to-nearest-even. Reject BF16 rounding overflow before publishing
 * either output. This is not a BF16 optimizer/master-weight contract. */
int ck_token_span_mean_backward_bf16(
    const uint16_t *d_output, size_t d_output_elements, size_t d_output_stride,
    const int32_t *spans, size_t span_elements, size_t span_count,
    uint16_t *d_input, size_t d_input_elements, size_t d_input_stride,
    float *scratch, size_t scratch_elements,
    size_t token_capacity, size_t valid_tokens, size_t channels);
int ck_gelu_erf_backward_bf16(
    const uint16_t *input, size_t input_elements, size_t input_stride,
    const uint16_t *d_output, size_t d_output_elements, size_t d_output_stride,
    uint16_t *d_input, size_t d_input_elements, size_t d_input_stride,
    float *scratch, size_t scratch_elements, size_t rows, size_t channels);
int ck_grouped_weighted_rows_backward_bf16(
    const uint16_t *options, size_t option_elements, size_t option_stride,
    const uint16_t *fields, size_t field_elements, size_t field_stride,
    const int32_t *group_counts, size_t count_elements,
    const uint16_t *d_output, size_t d_output_elements, size_t d_output_stride,
    uint16_t *d_options, size_t d_options_elements, size_t d_options_stride,
    uint16_t *d_fields, size_t d_fields_elements, size_t d_fields_stride,
    float *scratch, size_t scratch_elements,
    size_t groups, size_t option_rows, size_t channels);

/* Learned scalar/gain VJPs. Exp-clamp differentiates at equality; sigmoid
 * differentiates its rounded forward value. Scale gain reduces ascending FP64.
 * Rejection preserves all outputs; scale requires rows+1 scratch floats. */
int ck_scalar_exp_clamped_backward_f32(const float *input, size_t input_elements,
    const float *d_output, size_t d_output_elements, float *d_input, size_t d_input_elements, float maximum);
int ck_scalar_sigmoid_backward_f32(const float *input, size_t input_elements,
    const float *d_output, size_t d_output_elements, float *d_input, size_t d_input_elements);
int ck_scale_vector_backward_f32(const float *input, size_t input_elements,
    const float *gain, size_t gain_elements, const float *d_output, size_t d_output_elements,
    float *d_input, size_t d_input_elements, float *d_gain, size_t d_gain_elements,
    float *scratch, size_t scratch_elements, size_t rows);

#endif
