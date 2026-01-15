#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "ckernel_engine.h"

// Suppress false positive warnings about uninitialized variables
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

static void convert_int8_to_float(const int8_t *src,
                                  float *dst,
                                  size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        dst[i] = (float)src[i];
    }
}

static int8_t clamp_int8(float value)
{
    int32_t q = (int32_t)lrintf(value);
    if (q > INT8_MAX) {
        q = INT8_MAX;
    } else if (q < INT8_MIN) {
        q = INT8_MIN;
    }
    return (int8_t)q;
}

static void convert_float_to_int8(const float *src,
                                  int8_t *dst,
                                  size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        dst[i] = clamp_int8(src[i]);
    }
}

/*
 * INT8 RMSNorm forward with caller-provided scratch buffers.
 * scratch_input, scratch_output: each [tokens * aligned_embed_dim] floats
 */
void rmsnorm_forward_int8(const int8_t *input,
                          const float *gamma,
                          int8_t *output,
                          float *rstd_cache,
                          int tokens,
                          int d_model,
                          int aligned_embed_dim,
                          float eps,
                          float *scratch_input,
                          float *scratch_output)
{
    if (!input || !gamma || !output) return;
    if (!scratch_input || !scratch_output) return;

    size_t total = (size_t)tokens * (size_t)aligned_embed_dim;

    convert_int8_to_float(input, scratch_input, total);
    rmsnorm_forward(scratch_input, gamma, scratch_output, rstd_cache,
                    tokens, d_model, aligned_embed_dim, eps);
    convert_float_to_int8(scratch_output, output, total);
}

/*
 * INT8 RMSNorm backward with caller-provided scratch buffers.
 * scratch_d_output, scratch_input, scratch_d_input: each [tokens * aligned_embed_dim] floats
 */
void rmsnorm_backward_int8(const int8_t *d_output,
                           const int8_t *input,
                           const float *gamma,
                           const float *rstd_cache,
                           int8_t *d_input,
                           float *d_gamma,
                           int tokens,
                           int d_model,
                           int aligned_embed_dim,
                           float *scratch_d_output,
                           float *scratch_input,
                           float *scratch_d_input)
{
    if (!d_output || !input || !gamma || !rstd_cache || !d_input || !d_gamma) return;
    if (!scratch_d_output || !scratch_input || !scratch_d_input) return;

    size_t total = (size_t)tokens * (size_t)aligned_embed_dim;

    convert_int8_to_float(d_output, scratch_d_output, total);
    convert_int8_to_float(input, scratch_input, total);

    // Zero gamma gradient before accumulation.
    for (int d = 0; d < d_model; ++d) {
        d_gamma[d] = 0.0f;
    }

    rmsnorm_backward(scratch_d_output,
                     scratch_input,
                     gamma,
                     rstd_cache,
                     scratch_d_input,
                     d_gamma,
                     tokens,
                     d_model,
                     aligned_embed_dim);

    convert_float_to_int8(scratch_d_input, d_input, total);
}

#pragma GCC diagnostic pop
