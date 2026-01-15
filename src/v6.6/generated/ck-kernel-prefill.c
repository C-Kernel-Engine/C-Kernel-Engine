/*
 * ck-kernel-prefill.c - Prefill kernel stub for v6.6
 *
 * This is a placeholder stub. Actual prefill implementation will be
 * generated from IR when we enable prefill mode.
 */

#include <stdint.h>
#include <stddef.h>

// Prefill kernel stub - returns 0 (success)
int ck_prefill_forward(
    const void *weights,
    const int32_t *tokens,
    int n_tokens,
    float *hidden_out,
    void *kv_cache,
    int kv_pos
) {
    // Prefill not yet implemented - use decode path with loop
    (void)weights;
    (void)tokens;
    (void)n_tokens;
    (void)hidden_out;
    (void)kv_cache;
    (void)kv_pos;
    return 0;
}
