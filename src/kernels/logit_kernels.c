// SPDX-License-Identifier: Apache-2.0

#include "ckernel_engine.h"
#include "bf16_utils.h"

#include <math.h>
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

void final_logit_scale_muse_pytorch_bf16_storage(float *logits,
                                                  int tokens,
                                                  int vocab_size,
                                                  float scale)
{
    if (!logits || tokens <= 0 || vocab_size <= 0) {
        return;
    }

    const size_t total = (size_t)tokens * (size_t)vocab_size;
    for (size_t i = 0; i < total; ++i) {
        const float value = bf16_to_float(float_to_bf16(logits[i]));
        logits[i] = bf16_to_float(float_to_bf16(value * scale));
    }
}

void final_logit_softcap_muse_pytorch_bf16_storage(float *logits,
                                                    int tokens,
                                                    int vocab_size,
                                                    float cap)
{
    if (!logits || tokens <= 0 || vocab_size <= 0 || cap <= 0.0f) {
        return;
    }

    const size_t total = (size_t)tokens * (size_t)vocab_size;
    for (size_t i = 0; i < total; ++i) {
        const float value = bf16_to_float(float_to_bf16(logits[i]));
        const float divided = bf16_to_float(float_to_bf16(value / cap));
        const float activated = bf16_to_float(float_to_bf16(tanhf(divided)));
        logits[i] = bf16_to_float(float_to_bf16(activated * cap));
    }
}
