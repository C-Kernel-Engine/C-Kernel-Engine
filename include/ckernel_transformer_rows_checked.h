#ifndef CKERNEL_TRANSFORMER_ROWS_CHECKED_H
#define CKERNEL_TRANSFORMER_ROWS_CHECKED_H

#include <stddef.h>

/* Shared row bound for the checked routed expert providers and their circuit authoring. */
#define CHECKED_ROUTED_EXPERT_MAX_ROWS 128

#ifdef __cplusplus
extern "C" {
#endif

/* All providers reject invalid spans/nonfinite inputs before output writes. */
int rmsnorm_rows_checked_f32(const float *input, size_t input_elements,
                            size_t input_stride, const float *gamma,
                            size_t gamma_elements, float *output,
                            size_t output_elements, size_t output_stride,
                            size_t rows, size_t width, float eps);
int token_head_layout_checked_f32(const float *input, size_t input_elements,
                                  float *output, size_t output_elements,
                                  size_t tokens, size_t heads, size_t head_dim,
                                  int head_to_token);
int add_rows_checked_f32(const float *left, size_t left_elements,
                         const float *right, size_t right_elements,
                         float *output, size_t output_elements,
                         size_t elements);
int silu_mul_rows_checked_f32(const float *gate, size_t gate_elements,
                              const float *up, size_t up_elements,
                              float *output, size_t output_elements,
                              size_t elements);
int channel_scale_rows_checked_f32(const float *input, size_t input_elements,
                                   size_t input_stride, const float *scale,
                                   size_t scale_elements, float *output,
                                   size_t output_elements, size_t output_stride,
                                   size_t rows, size_t channels);
int softcapped_swiglu_rows_checked_f32(const float *gate, size_t gate_elements,
                                       const float *up, size_t up_elements,
                                       float *output, size_t output_elements,
                                       size_t elements, float gate_cap, float up_cap);
int rope_qk_bounded_checked_f32(float *q, size_t q_elements,
                                float *k, size_t k_elements,
                                size_t heads, size_t kv_heads,
                                size_t tokens, size_t head_dim,
                                size_t pos_offset, float freq_base);
int attention_causal_bounded_checked_f32(
    const float *q, size_t q_elements,
    const float *k, size_t k_elements,
    const float *v, size_t v_elements,
    float *work, size_t work_elements,
    float *output, size_t output_elements,
    size_t heads, size_t kv_heads, size_t tokens, size_t head_dim);
int attention_sliding_bounded_checked_f32(
    const float *q, size_t q_elements,
    const float *k, size_t k_elements,
    const float *v, size_t v_elements,
    float *work, size_t work_elements,
    float *output, size_t output_elements,
    size_t heads, size_t kv_heads, size_t tokens, size_t head_dim,
    size_t sliding_window);

#ifdef __cplusplus
}
#endif
#endif
