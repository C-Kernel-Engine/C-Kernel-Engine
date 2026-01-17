/**
 * @file layernorm_kernels_bf16.c
 * @brief LayerNorm kernels for BF16 tensors
 *
 * CK-ENGINE KERNEL RULES:
 * =======================
 * 1. NO malloc/free - memory via bump allocator, pointers passed in
 * 2. NO OpenMP - parallelization at orchestrator/codegen layer
 * 3. API must define: inputs, outputs, workspace, and memory layouts
 * 4. Pure computation - deterministic, no side effects
 *
 * After changes: make test && make llamacpp-parity-full
 *
 * LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
 */

#include <stdint.h>

#include "bf16_utils.h"
#include "ckernel_engine.h"

/* Suppress false positive warnings about uninitialized variables */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

/*
 * BF16 LayerNorm forward (rolled) with caller-provided scratch buffers.
 * scratch_input, scratch_output: each [num_tokens_in_slice * aligned_embed_dim] floats
 */
void layernorm_forward_rolled_slice_bf16(const uint16_t *__restrict input_slice_base,
                                         const float *__restrict gamma,
                                         const float *__restrict beta,
                                         uint16_t *__restrict output_slice_base,
                                         float *__restrict mean_cache_slice,
                                         float *__restrict rstd_cache_slice,
                                         int num_tokens_in_slice,
                                         int d_model,
                                         int aligned_embed_dim,
                                         float eps,
                                         float *scratch_input,
                                         float *scratch_output)
{
    if (!scratch_input || !scratch_output) return;

    size_t total = (size_t)num_tokens_in_slice * (size_t)aligned_embed_dim;

    bf16_tensor_to_float(input_slice_base, scratch_input, total);
    layernorm_forward_rolled_slice(scratch_input, gamma, beta,
                                   scratch_output, mean_cache_slice, rstd_cache_slice,
                                   num_tokens_in_slice, d_model, aligned_embed_dim, eps);
    float_tensor_to_bf16(scratch_output, output_slice_base, total);
}

/*
 * BF16 LayerNorm forward (unrolled) with caller-provided scratch buffers.
 */
void layernorm_forward_unrolled_slice_bf16(const uint16_t *__restrict input_slice_base,
                                           const float *__restrict gamma,
                                           const float *__restrict beta,
                                           uint16_t *__restrict output_slice_base,
                                           float *__restrict mean_cache_slice,
                                           float *__restrict rstd_cache_slice,
                                           int num_tokens_in_slice,
                                           int d_model,
                                           float eps,
                                           float *scratch_input,
                                           float *scratch_output)
{
    if (!scratch_input || !scratch_output) return;

    size_t total = (size_t)num_tokens_in_slice * (size_t)d_model;

    bf16_tensor_to_float(input_slice_base, scratch_input, total);
    layernorm_forward_unrolled_slice(scratch_input, gamma, beta,
                                     scratch_output, mean_cache_slice, rstd_cache_slice,
                                     num_tokens_in_slice, d_model, eps);
    float_tensor_to_bf16(scratch_output, output_slice_base, total);
}

/*
 * BF16 LayerNorm backward with caller-provided scratch buffers.
 * scratch_d_output, scratch_input, scratch_d_input: each [tokens * aligned_embed_dim] floats
 */
void layernorm_backward_kernel_bf16(const uint16_t *d_output,
                                    const uint16_t *input,
                                    const float *gamma,
                                    const float *mean,
                                    const float *rstd,
                                    uint16_t *d_input,
                                    float *d_gamma,
                                    float *d_beta,
                                    int tokens, int d_model, int aligned_embed_dim,
                                    float *scratch_d_output,
                                    float *scratch_input,
                                    float *scratch_d_input)
{
    if (!scratch_d_output || !scratch_input || !scratch_d_input) return;

    size_t total = (size_t)tokens * (size_t)aligned_embed_dim;

    bf16_tensor_to_float(d_output, scratch_d_output, total);
    bf16_tensor_to_float(input, scratch_input, total);

    layernorm_backward_kernel(scratch_d_output, scratch_input, gamma, mean, rstd,
                              scratch_d_input, d_gamma, d_beta,
                              tokens, d_model, aligned_embed_dim);

    float_tensor_to_bf16(scratch_d_input, d_input, total);
}

#pragma GCC diagnostic pop
