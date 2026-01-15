#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "ckernel_engine.h"

static inline int8_t decode_int4(uint8_t packed, int index)
{
    int8_t nibble;
    if ((index & 1) == 0) {
        nibble = packed & 0x0F;
    } else {
        nibble = (packed >> 4) & 0x0F;
    }
    if (nibble >= 8) {
        nibble -= 16;
    }
    return nibble;
}

static inline uint8_t encode_int4_nibble(int8_t value)
{
    if (value > 7) {
        value = 7;
    } else if (value < -8) {
        value = -8;
    }
    return (uint8_t)(value & 0x0F);
}

static void convert_int4_to_float(const uint8_t *src,
                                  float *dst,
                                  size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        uint8_t packed = src[i >> 1];
        dst[i] = (float)decode_int4(packed, (int)(i & 1));
    }
}

static void convert_float_to_int4(const float *src,
                                  uint8_t *dst,
                                  size_t count)
{
    size_t bytes = (count + 1) / 2;
    for (size_t i = 0; i < bytes; ++i) {
        dst[i] = 0;
    }
    for (size_t i = 0; i < count; ++i) {
        uint8_t quant = encode_int4_nibble((int8_t)lrintf(src[i]));
        size_t byte_idx = i >> 1;
        if ((i & 1) == 0) {
            dst[byte_idx] = (dst[byte_idx] & 0xF0) | quant;
        } else {
            dst[byte_idx] = (dst[byte_idx] & 0x0F) | (quant << 4);
        }
    }
}

/*
 * INT4 RMSNorm forward with caller-provided scratch buffers.
 * scratch_input, scratch_output: each [tokens * aligned_embed_dim] floats
 */
void rmsnorm_forward_int4(const uint8_t *input,
                          const float *gamma,
                          uint8_t *output,
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

    convert_int4_to_float(input, scratch_input, total);
    rmsnorm_forward(scratch_input, gamma, scratch_output, rstd_cache,
                    tokens, d_model, aligned_embed_dim, eps);
    convert_float_to_int4(scratch_output, output, total);
}

/*
 * INT4 RMSNorm backward with caller-provided scratch buffers.
 * scratch_d_output, scratch_input, scratch_d_input: each [tokens * aligned_embed_dim] floats
 */
void rmsnorm_backward_int4(const uint8_t *d_output,
                           const uint8_t *input,
                           const float *gamma,
                           const float *rstd_cache,
                           uint8_t *d_input,
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

    convert_int4_to_float(d_output, scratch_d_output, total);
    convert_int4_to_float(input, scratch_input, total);

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

    convert_float_to_int4(scratch_d_input, d_input, total);
}
