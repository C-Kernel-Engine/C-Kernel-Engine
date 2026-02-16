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

    const int strict = ck_strict_parity_enabled();
    const double scale = 1.0 / (double)tokens;
    double total_loss = 0.0;

    for (int t = 0; t < tokens; ++t) {
        const float *row = logits + (size_t)t * (size_t)vocab_size;
        float *drow = d_logits + (size_t)t * (size_t)vocab_size;
        int target = targets[t];

        if (strict) {
            /*
             * Strict parity mode:
             * - use double accumulation for sum_exp/loss accumulation,
             * - keep CE in log-sum-exp form to mirror PyTorch numerics,
             * - avoid probability clipping because clipping introduces long-run drift.
             */
            double max_logit = (double)row[0];
            for (int v = 1; v < vocab_size; ++v) {
                const double rv = (double)row[v];
                if (rv > max_logit) {
                    max_logit = rv;
                }
            }

            double sum_exp = 0.0;
            for (int v = 0; v < vocab_size; ++v) {
                sum_exp += exp((double)row[v] - max_logit);
            }
            const double inv_sum = 1.0 / sum_exp;
            for (int v = 0; v < vocab_size; ++v) {
                const double p = exp((double)row[v] - max_logit) * inv_sum;
                drow[v] = (float)(p * scale);
            }

            if (target >= 0 && target < vocab_size) {
                const double log_sum_exp = log(sum_exp);
                const double target_logit = (double)row[target];
                total_loss += -(target_logit - max_logit - log_sum_exp);
                drow[target] -= (float)scale;
            }
        } else {
            /* Fast path still preserves CE semantics via log-sum-exp, with fp32 expf math. */
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
                // Match PyTorch CE semantics: loss via log-sum-exp, not probability clamp.
                // This avoids artificial saturation at -log(1e-10) ~= 23.02585.
                const double log_sum_exp = log(sum_exp);
                const double target_logit = (double)row[target];
                total_loss += -(target_logit - (double)max_logit - log_sum_exp);
                drow[target] -= 1.0f;
            }

            for (int v = 0; v < vocab_size; ++v) {
                drow[v] *= (float)scale;
            }
        }
    }

    if (loss_out) {
        *loss_out = (float)(total_loss / (double)tokens);
    }
}
