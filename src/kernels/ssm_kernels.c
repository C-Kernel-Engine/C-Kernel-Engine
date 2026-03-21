/**
 * @file ssm_kernels.c
 * @brief FP32 SSM causal depthwise convolution kernels for qwen3next/Qwen3.5.
 *
 * CK-ENGINE KERNEL RULES:
 * =======================
 * 1. NO malloc/free - memory via bump allocator, pointers passed in
 * 2. NO OpenMP - parallelization at orchestrator/codegen layer
 * 3. API must define: inputs, outputs, workspace, and memory layouts
 * 4. Pure computation - deterministic, no side effects
 *
 * After changes: make test-ssm-conv && make test-kernels
 *
 * This file implements the GGML_OP_SSM_CONV semantics used by qwen3next before
 * the recurrent DeltaNet update:
 *   out[seq, token, ch] = dot(conv_x[seq, ch, token:token+kernel], kernel[ch, :])
 *
 * Memory layouts:
 *   conv_x   : [num_seqs, num_channels, kernel_size - 1 + num_tokens]
 *   kernel   : [num_channels, kernel_size]
 *   out      : [num_seqs, num_tokens, num_channels]
 *   d_out    : same as out
 *   d_conv_x : same as conv_x
 *   d_kernel : same as kernel
 */

#include "ckernel_engine.h"

#include <stddef.h>
#include <string.h>

void ssm_conv1d_forward_ref(const float *conv_x,
                            const float *kernel,
                            float *out,
                            int kernel_size,
                            int num_channels,
                            int num_tokens,
                            int num_seqs)
{
    if (!conv_x || !kernel || !out) {
        return;
    }
    if (kernel_size <= 0 || num_channels <= 0 || num_tokens < 0 || num_seqs <= 0) {
        return;
    }

    const size_t seq_width = (size_t)kernel_size - 1u + (size_t)num_tokens;
    const size_t conv_seq_stride = (size_t)num_channels * seq_width;
    const size_t out_seq_stride = (size_t)num_tokens * (size_t)num_channels;

    for (int seq = 0; seq < num_seqs; ++seq) {
        const float *conv_seq = conv_x + (size_t)seq * conv_seq_stride;
        float *out_seq = out + (size_t)seq * out_seq_stride;

        for (int tok = 0; tok < num_tokens; ++tok) {
            float *out_tok = out_seq + (size_t)tok * (size_t)num_channels;

            for (int ch = 0; ch < num_channels; ++ch) {
                const float *conv_row = conv_seq + (size_t)ch * seq_width + (size_t)tok;
                const float *kernel_row = kernel + (size_t)ch * (size_t)kernel_size;
                float sumf = 0.0f;
                for (int k = 0; k < kernel_size; ++k) {
                    sumf += conv_row[k] * kernel_row[k];
                }
                out_tok[ch] = sumf;
            }
        }
    }
}

void ssm_conv1d_backward_ref(const float *d_out,
                             const float *conv_x,
                             const float *kernel,
                             float *d_conv_x,
                             float *d_kernel,
                             int kernel_size,
                             int num_channels,
                             int num_tokens,
                             int num_seqs)
{
    if (!d_out || !conv_x || !kernel || !d_conv_x || !d_kernel) {
        return;
    }
    if (kernel_size <= 0 || num_channels <= 0 || num_tokens < 0 || num_seqs <= 0) {
        return;
    }

    const size_t seq_width = (size_t)kernel_size - 1u + (size_t)num_tokens;
    const size_t conv_total = (size_t)num_seqs * (size_t)num_channels * seq_width;
    const size_t kernel_total = (size_t)num_channels * (size_t)kernel_size;
    const size_t conv_seq_stride = (size_t)num_channels * seq_width;
    const size_t out_seq_stride = (size_t)num_tokens * (size_t)num_channels;

    memset(d_conv_x, 0, conv_total * sizeof(float));
    memset(d_kernel, 0, kernel_total * sizeof(float));

    for (int seq = 0; seq < num_seqs; ++seq) {
        const float *d_out_seq = d_out + (size_t)seq * out_seq_stride;
        const float *conv_seq = conv_x + (size_t)seq * conv_seq_stride;
        float *d_conv_seq = d_conv_x + (size_t)seq * conv_seq_stride;

        for (int tok = 0; tok < num_tokens; ++tok) {
            const float *d_out_tok = d_out_seq + (size_t)tok * (size_t)num_channels;

            for (int ch = 0; ch < num_channels; ++ch) {
                const float grad = d_out_tok[ch];
                const float *conv_row = conv_seq + (size_t)ch * seq_width + (size_t)tok;
                float *d_conv_row = d_conv_seq + (size_t)ch * seq_width + (size_t)tok;
                const float *kernel_row = kernel + (size_t)ch * (size_t)kernel_size;
                float *d_kernel_row = d_kernel + (size_t)ch * (size_t)kernel_size;

                for (int k = 0; k < kernel_size; ++k) {
                    d_kernel_row[k] += grad * conv_row[k];
                    d_conv_row[k] += grad * kernel_row[k];
                }
            }
        }
    }
}

void ssm_conv1d_forward(const float *conv_x,
                        const float *kernel,
                        float *out,
                        int kernel_size,
                        int num_channels,
                        int num_tokens,
                        int num_seqs)
{
    ssm_conv1d_forward_ref(conv_x, kernel, out, kernel_size, num_channels, num_tokens, num_seqs);
}

void ssm_conv1d_backward(const float *d_out,
                         const float *conv_x,
                         const float *kernel,
                         float *d_conv_x,
                         float *d_kernel,
                         int kernel_size,
                         int num_channels,
                         int num_tokens,
                         int num_seqs)
{
    ssm_conv1d_backward_ref(d_out, conv_x, kernel, d_conv_x, d_kernel, kernel_size, num_channels, num_tokens, num_seqs);
}
