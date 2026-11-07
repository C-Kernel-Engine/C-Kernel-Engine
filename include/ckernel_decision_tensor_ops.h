#ifndef CK_DECISION_TENSOR_OPS_H
#define CK_DECISION_TENSOR_OPS_H
#include <stddef.h>
#include <stdint.h>

int ck_broadcast_add_rows_f32(const float *left, size_t left_elements, const float *right, size_t right_elements, float *output, size_t output_elements, float *scratch, size_t scratch_elements, size_t rows, size_t channels);
int ck_indexed_pair_features_f32(const float *left, size_t left_elements, const float *right, size_t right_elements, const int32_t *indices, size_t indices_elements, float *output, size_t output_elements, float *scratch, size_t scratch_elements, size_t groups, size_t rows, size_t channels);
int ck_indexed_cosine_rows_f32(const float *left, size_t left_elements, const float *right, size_t right_elements, const int32_t *indices, size_t indices_elements, float *output, size_t output_elements, float *scratch, size_t scratch_elements, size_t groups, size_t rows, size_t channels, float epsilon);
int ck_scalar_exp_clamped_f32(const float *input, size_t input_elements, float *output, size_t output_elements, float maximum);
int ck_scalar_sigmoid_f32(const float *input, size_t input_elements, float *output, size_t output_elements);
int ck_scale_vector_f32(const float *input, size_t input_elements, const float *gain, size_t gain_elements, float *output, size_t output_elements, float *scratch, size_t scratch_elements, size_t rows);
#endif
