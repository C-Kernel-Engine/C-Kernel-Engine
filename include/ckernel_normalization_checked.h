#ifndef CK_NORMALIZATION_CHECKED_H
#define CK_NORMALIZATION_CHECKED_H
#include <stddef.h>
/* Internal arithmetic entry after the checked adapter validates its buffers. */
void ck_layernorm_row_fixed_chunk4_f32(const float *input, const float *gamma,
    const float *beta, float *output, float *mean, float *rstd, int channels,
    float epsilon);

/* Checked inference adapters. Scratch may change on rejection; output must not.
 * Buffers are caller-owned and nonoverlapping. No allocation or hidden state.
 * Arithmetic follows the selected provider contract. */
int layernorm_rows_checked_workspace(size_t rows, size_t channels, size_t *elements);
int layernorm_rows_checked_f32(const float *input, size_t input_elements, size_t input_stride,
    const float *gamma, size_t gamma_elements, const float *beta, size_t beta_elements,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements, size_t rows, size_t channels, float epsilon);
int gelu_rows_tanh_checked_f32(const float *input, size_t input_elements, size_t input_stride,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements, size_t rows, size_t channels);
#endif
