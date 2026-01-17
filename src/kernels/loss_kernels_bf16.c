/**
 * @file loss_kernels_bf16.c
 * @brief Loss function kernels for BF16 tensors
 *
 * CK-ENGINE KERNEL RULES:
 * =======================
 * 1. NO malloc/free - memory via bump allocator, pointers passed in
 * 2. NO OpenMP - parallelization at orchestrator/codegen layer
 * 3. API must define: inputs, outputs, workspace, and memory layouts
 * 4. Pure computation - deterministic, no side effects
 *
 * After changes: make test && make llamacpp-parity-full
 */

#include <stddef.h>
#include <stdint.h>

#include "bf16_utils.h"
#include "ckernel_engine.h"

/*
 * BF16 softmax cross-entropy loss with caller-provided scratch buffers.
 * scratch_logits, scratch_d_logits: each [tokens * vocab_size] floats
 */
void softmax_cross_entropy_loss_bf16(const uint16_t *logits,
                                     const int32_t *targets,
                                     int tokens,
                                     int vocab_size,
                                     uint16_t *d_logits,
                                     float *loss_out,
                                     float *scratch_logits,
                                     float *scratch_d_logits)
{
    if (!logits || !targets || !d_logits || tokens <= 0 || vocab_size <= 0) {
        if (loss_out) *loss_out = 0.0f;
        return;
    }
    if (!scratch_logits || !scratch_d_logits) {
        if (loss_out) *loss_out = 0.0f;
        return;
    }

    const size_t count = (size_t)tokens * (size_t)vocab_size;

    bf16_tensor_to_float(logits, scratch_logits, count);
    softmax_cross_entropy_loss(scratch_logits, targets, tokens, vocab_size, scratch_d_logits, loss_out);
    float_tensor_to_bf16(scratch_d_logits, d_logits, count);
}
