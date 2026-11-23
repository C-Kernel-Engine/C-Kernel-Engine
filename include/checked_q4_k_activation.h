#ifndef CK_CHECKED_Q4_K_ACTIVATION_H
#define CK_CHECKED_Q4_K_ACTIVATION_H

#include "ckernel_quant.h"

#include <math.h>

void gemv_q4_k_q8_k_avx2(float *, const void *, const void *, int, int);
void gemv_q4_k_q8_k_pair_checked_cpu(float *, float *, const void *,
                                    const void *, const void *, int, int);
void quantize_row_q8_k_ref(const float *, void *, int);

static inline int ck_pack_q4_input_q8_k(const float *input, unsigned char *packed, size_t width)
{
    for (size_t first = 0; first < width; first += QK_K) {
        float maximum = 0.0f;
        for (size_t i = 0; i < QK_K; ++i) maximum = fmaxf(maximum, fabsf(input[first + i]));
        if (maximum != 0.0f && !isfinite(127.0f / maximum)) return -3;
    }
    quantize_row_q8_k_ref(input, packed, (int)width);
    return 0;
}

#endif
