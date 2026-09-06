/**
 * @file geglu_kernels.c
 * @brief GeGLU kernels split from gelu_kernels.c
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "ckernel_engine.h"
#include "bf16_utils.h"

/* Reuse existing optimized GELU implementation from gelu_kernels.c. */
extern void gelu_fast_inplace(float *data, size_t n);
extern void gelu_ggml_native_inplace(float *data, size_t n);

void geglu_forward_ggml_native(const float *x, float *out, int tokens, int dim)
{
    if (!x || !out || tokens <= 0 || dim <= 0) return;
    /* Preserve ascending compaction when output aliases the gate/up input. */
    for (int t = 0; t < tokens; ++t) {
        const float *row = x + (size_t)t * (size_t)dim * 2;
        float *dst = out + (size_t)t * dim;
        for (int d = 0; d < dim; ++d) {
            const float gate = row[d], up = row[dim + d];
            float value = gate;
            gelu_ggml_native_inplace(&value, 1);
            dst[d] = gate <= -10.0f ? 0.0f : value * up;
        }
    }
}

static inline float ck_gelu_tanh_parity_f32(float x)
{
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    const float x3 = x * x * x;
    const float inner = sqrt_2_over_pi * (x + coeff * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

void geglu_forward_exact(const float *x, float *out, int tokens, int dim)
{
    const int inner_dim = dim * 2;
    for (int t = 0; t < tokens; ++t) {
        const float *x_ptr = x + (size_t)t * (size_t)inner_dim;
        float *out_ptr = out + (size_t)t * (size_t)dim;

        for (int d = 0; d < dim; ++d) {
            out_ptr[d] = ck_gelu_tanh_parity_f32(x_ptr[d]) * x_ptr[dim + d];
        }
    }
}

void geglu_forward_fp32(const float *x, float *out, int tokens, int dim)
{
    if (!x || !out || tokens <= 0 || dim <= 0) {
        return;
    }

    if (ck_strict_parity_enabled()) {
        geglu_forward_exact(x, out, tokens, dim);
        return;
    }

    const int inner_dim = dim * 2;
    for (int t = 0; t < tokens; ++t) {
        const float *x_ptr = x + (size_t)t * (size_t)inner_dim;
        float *out_ptr = out + (size_t)t * (size_t)dim;

        for (int d = 0; d < dim; ++d) {
            out_ptr[d] = x_ptr[d];
        }

        gelu_fast_inplace(out_ptr, (size_t)dim);

        for (int d = 0; d < dim; ++d) {
            out_ptr[d] *= x_ptr[dim + d];
        }
    }
}

void geglu_forward_bf16(const uint16_t *x, uint16_t *out, int tokens, int dim, float *scratch)
{
    if (!x || !out || !scratch || tokens <= 0 || dim <= 0) {
        return;
    }

    const size_t fp32_size = (size_t)tokens * (size_t)dim;
    const size_t input_size = fp32_size * 2;
    float *fp32_input = scratch;
    float *fp32_output = scratch + input_size;

    bf16_tensor_to_float(x, fp32_input, input_size);
    geglu_forward_fp32(fp32_input, fp32_output, tokens, dim);
    float_tensor_to_bf16(fp32_output, out, fp32_size);
}

void geglu_backward_fp32(const float *x,
                         const float *d_out,
                         float *d_x,
                         int tokens,
                         int dim)
{
    if (!x || !d_out || !d_x || tokens <= 0 || dim <= 0) {
        return;
    }

    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    const int inner_dim = dim * 2;

    for (int t = 0; t < tokens; ++t) {
        const float *x_ptr = x + (size_t)t * (size_t)inner_dim;
        const float *d_out_ptr = d_out + (size_t)t * (size_t)dim;
        float *d_x_ptr = d_x + (size_t)t * (size_t)inner_dim;

        for (int d = 0; d < dim; ++d) {
            float a = x_ptr[d];
            float b = x_ptr[dim + d];
            float dout = d_out_ptr[d];

            float a2 = a * a;
            float a3 = a2 * a;
            float g = sqrt_2_over_pi * (a + coeff * a3);
            float tanh_g = tanhf(g);
            float sech2_g = 1.0f - tanh_g * tanh_g;
            float g_prime = sqrt_2_over_pi * (1.0f + 3.0f * coeff * a2);

            float d_gelu = 0.5f * (1.0f + tanh_g) + 0.5f * a * sech2_g * g_prime;
            d_x_ptr[d] = dout * d_gelu * b;

            float gelu_a = 0.5f * a * (1.0f + tanh_g);
            d_x_ptr[dim + d] = dout * gelu_a;
        }
    }
}

void geglu_backward_bf16_mixed(const uint16_t *x,
                               const uint16_t *d_out,
                               float *d_x,
                               int tokens,
                               int dim)
{
    if (!x || !d_out || !d_x || tokens <= 0 || dim <= 0) {
        return;
    }

    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    const int inner_dim = dim * 2;

    for (int t = 0; t < tokens; ++t) {
        const uint16_t *x_ptr = x + (size_t)t * (size_t)inner_dim;
        const uint16_t *d_out_ptr = d_out + (size_t)t * (size_t)dim;
        float *d_x_ptr = d_x + (size_t)t * (size_t)inner_dim;

        for (int d = 0; d < dim; ++d) {
            const float a = bf16_to_float(x_ptr[d]);
            const float b = bf16_to_float(x_ptr[dim + d]);
            const float dout = bf16_to_float(d_out_ptr[d]);

            const float a2 = a * a;
            const float a3 = a2 * a;
            const float g = sqrt_2_over_pi * (a + coeff * a3);
            const float tanh_g = tanhf(g);
            const float sech2_g = 1.0f - tanh_g * tanh_g;
            const float g_prime = sqrt_2_over_pi * (1.0f + 3.0f * coeff * a2);

            const float d_gelu = 0.5f * (1.0f + tanh_g) + 0.5f * a * sech2_g * g_prime;
            d_x_ptr[d] = dout * d_gelu * b;

            const float gelu_a = 0.5f * a * (1.0f + tanh_g);
            d_x_ptr[dim + d] = dout * gelu_a;
        }
    }
}
