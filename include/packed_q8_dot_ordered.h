#ifndef CK_PACKED_Q8_DOT_ORDERED_H
#define CK_PACKED_Q8_DOT_ORDERED_H

#include "ckernel_quant.h"

static inline float ck_packed_q8_dot_ordered(const float *input, const unsigned char *weight,
                                             size_t width)
{
    double sum = 0.0;
    for (size_t i = 0; i < width; i += QK8_0, weight += sizeof(block_q8_0)) {
        const float scale = ck_fp16_to_fp32_soft((uint16_t)(weight[0] | ((uint16_t)weight[1] << 8)));
        for (size_t j = 0; j < QK8_0; ++j) {
            const int code = weight[j + 2];
            const float value = scale * (float)(code < 128 ? code : code - 256);
            sum += (double)input[i + j] * (double)value;
        }
    }
    return (float)sum;
}

#endif
