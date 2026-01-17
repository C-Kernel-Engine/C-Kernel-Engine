/**
 * @file loss_kernels.c
 * @brief Loss function kernels (cross-entropy, etc.)
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
 * Cross-entropy: L = -log(softmax(logits)[target])
 */

#include "ckernel_engine.h"

#include <math.h>

void softmax_cross_entropy_loss(const float *logits,
                                const int32_t *targets,
                                int tokens,
                                int vocab_size,
                                float *d_logits,
                                float *loss_out)
{
    if (!logits || !targets || !d_logits || tokens <= 0 || vocab_size <= 0) {
        if (loss_out) {
            *loss_out = 0.0f;
        }
        return;
    }

    double total_loss = 0.0;

    for (int t = 0; t < tokens; ++t) {
        const float *row = logits + (size_t)t * (size_t)vocab_size;
        float *drow = d_logits + (size_t)t * (size_t)vocab_size;
        int target = targets[t];

        float max_logit = row[0];
        for (int v = 1; v < vocab_size; ++v) {
            if (row[v] > max_logit) {
                max_logit = row[v];
            }
        }

        double sum_exp = 0.0;
        for (int v = 0; v < vocab_size; ++v) {
            float e = expf(row[v] - max_logit);
            drow[v] = e;
            sum_exp += e;
        }

        float inv_sum = 1.0f / (float)sum_exp;
        for (int v = 0; v < vocab_size; ++v) {
            drow[v] *= inv_sum;
        }

        if (target >= 0 && target < vocab_size) {
            total_loss += -logf(drow[target] + 1e-10f);
            drow[target] -= 1.0f;
        }

        float scale = 1.0f / (float)tokens;
        for (int v = 0; v < vocab_size; ++v) {
            drow[v] *= scale;
        }
    }

    if (loss_out) {
        *loss_out = (float)(total_loss / (double)tokens);
    }
}
