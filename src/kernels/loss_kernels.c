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

static void zero_row_f32(float *row, int cols)
{
    for (int i = 0; i < cols; ++i) {
        row[i] = 0.0f;
    }
}

/*
 * Index-target cross entropy (mean reduction) with PyTorch-aligned semantics:
 * - ignore_index fixed to -100,
 * - denominator = number of valid (non-ignored) targets,
 * - all-ignored rows => NaN loss for mean reduction.
 *
 * Invalid targets (outside [0, vocab_size)) are treated as hard failures for
 * loss reporting (loss_out=NaN) while keeping gradients deterministic.
 */
static void softmax_cross_entropy_loss_index_mean_impl(const float *logits,
                                                       const int32_t *targets,
                                                       int tokens,
                                                       int vocab_size,
                                                       float *d_logits,
                                                       float *loss_out,
                                                       int force_strict_math)
{
    if (!logits || !targets || !d_logits || tokens <= 0 || vocab_size <= 0) {
        if (loss_out) {
            *loss_out = 0.0f;
        }
        return;
    }

    const int ignore_index = -100;
    const int strict = (force_strict_math >= 0) ? (force_strict_math ? 1 : 0)
                                                : ck_strict_parity_enabled();
    double total_loss = 0.0;
    int valid_tokens = 0;
    int invalid_target_seen = 0;

    for (int t = 0; t < tokens; ++t) {
        const float *row = logits + (size_t)t * (size_t)vocab_size;
        float *drow = d_logits + (size_t)t * (size_t)vocab_size;
        const int target = targets[t];

        if (target == ignore_index) {
            zero_row_f32(drow, vocab_size);
            continue;
        }
        if (target < 0 || target >= vocab_size) {
            zero_row_f32(drow, vocab_size);
            invalid_target_seen = 1;
            continue;
        }

        if (strict) {
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
                drow[v] = (float)(exp((double)row[v] - max_logit) * inv_sum);
            }

            total_loss += -(double)row[target] + max_logit + log(sum_exp);
        } else {
            float max_logit = row[0];
            for (int v = 1; v < vocab_size; ++v) {
                if (row[v] > max_logit) {
                    max_logit = row[v];
                }
            }

            double sum_exp = 0.0;
            for (int v = 0; v < vocab_size; ++v) {
                const float e = expf(row[v] - max_logit);
                drow[v] = e;
                sum_exp += (double)e;
            }
            const float inv_sum = 1.0f / (float)sum_exp;
            for (int v = 0; v < vocab_size; ++v) {
                drow[v] *= inv_sum;
            }

            total_loss += -(double)row[target] + (double)max_logit + log(sum_exp);
        }

        drow[target] -= 1.0f;
        ++valid_tokens;
    }

    if (valid_tokens > 0) {
        const float scale = 1.0f / (float)valid_tokens;
        for (int t = 0; t < tokens; ++t) {
            float *drow = d_logits + (size_t)t * (size_t)vocab_size;
            for (int v = 0; v < vocab_size; ++v) {
                drow[v] *= scale;
            }
        }
    }

    if (loss_out) {
        if (invalid_target_seen || valid_tokens == 0) {
            *loss_out = NAN;
        } else {
            *loss_out = (float)(total_loss / (double)valid_tokens);
        }
    }
}

/*
 * Legacy CE numerics preserved for all-valid index targets.
 * This is the historical v7 behavior used by long-horizon drift baselines.
 */
static void softmax_cross_entropy_loss_legacy_mean_impl(const float *logits,
                                                        const int32_t *targets,
                                                        int tokens,
                                                        int vocab_size,
                                                        float *d_logits,
                                                        float *loss_out)
{
    const int strict = ck_strict_parity_enabled();
    const double scale = 1.0 / (double)tokens;
    double total_loss = 0.0;

    for (int t = 0; t < tokens; ++t) {
        const float *row = logits + (size_t)t * (size_t)vocab_size;
        float *drow = d_logits + (size_t)t * (size_t)vocab_size;
        const int target = targets[t];

        if (strict) {
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
            float max_logit = row[0];
            for (int v = 1; v < vocab_size; ++v) {
                if (row[v] > max_logit) {
                    max_logit = row[v];
                }
            }

            double sum_exp = 0.0;
            for (int v = 0; v < vocab_size; ++v) {
                const float e = expf(row[v] - max_logit);
                drow[v] = e;
                sum_exp += (double)e;
            }

            const float inv_sum = 1.0f / (float)sum_exp;
            for (int v = 0; v < vocab_size; ++v) {
                drow[v] *= inv_sum;
            }

            if (target >= 0 && target < vocab_size) {
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

static int ce_targets_all_valid_no_ignore(const int32_t *targets, int tokens, int vocab_size)
{
    for (int t = 0; t < tokens; ++t) {
        const int target = targets[t];
        if (target < 0 || target >= vocab_size) {
            return 0;
        }
    }
    return 1;
}

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

    if (ce_targets_all_valid_no_ignore(targets, tokens, vocab_size)) {
        softmax_cross_entropy_loss_legacy_mean_impl(
            logits, targets, tokens, vocab_size, d_logits, loss_out);
        return;
    }

    softmax_cross_entropy_loss_index_mean_impl(
        logits, targets, tokens, vocab_size, d_logits, loss_out, -1);
}

/*
 * PyTorch-index-target aligned CE variant:
 * - stable log-sum-exp per row,
 * - mean reduction over valid (non-ignore) targets,
 * - ignore_index fixed to -100 (PyTorch default).
 *
 * This keeps the same ABI as softmax_cross_entropy_loss so it can be selected
 * by name from parity harnesses without graph/codegen changes.
 */
void softmax_cross_entropy_loss_ptref(const float *logits,
                                      const int32_t *targets,
                                      int tokens,
                                      int vocab_size,
                                      float *d_logits,
                                      float *loss_out)
{
    /*
     * Keep a strict reference variant for parity experiments:
     * - always uses strict math path,
     * - same reduction / ignore semantics as default kernel.
     */
    softmax_cross_entropy_loss_index_mean_impl(
        logits, targets, tokens, vocab_size, d_logits, loss_out, 1);
}
