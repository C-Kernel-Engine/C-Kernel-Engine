// SPDX-License-Identifier: Apache-2.0

#include "ckernel_engine.h"

#include <stddef.h>

void final_logit_scale_f32(float *logits,
                           int tokens,
                           int vocab_size,
                           float scale)
{
    if (!logits || tokens <= 0 || vocab_size <= 0) {
        return;
    }

    const size_t total = (size_t) tokens * (size_t) vocab_size;
    for (size_t i = 0; i < total; ++i) {
        logits[i] *= scale;
    }
}
