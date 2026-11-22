#include "checked_float_buffer.h"
#include "ckernel_quant.h"

int embedding_lookup_q8_0_checked_f32(const int32_t *ids, size_t ids_elements,
                                      const void *table, size_t table_bytes, float *output,
                                      size_t output_elements, size_t tokens, size_t vocabulary,
                                      size_t width)
{
    size_t extent;
    if (!width || width % QK8_0 || width / QK8_0 > SIZE_MAX / sizeof(block_q8_0) ||
        !ck_checked_span(tokens, width, width, &extent) ||
        tokens > ids_elements || extent > output_elements || tokens > SIZE_MAX / sizeof(int32_t)) return -2;
    const size_t row_bytes = width / QK8_0 * sizeof(block_q8_0);
    if (!vocabulary || vocabulary > SIZE_MAX / row_bytes || table_bytes != vocabulary * row_bytes)
        return -2;
    if (!ids || (uintptr_t)ids % _Alignof(int32_t) ||
        (uintptr_t)ids > UINTPTR_MAX - tokens * sizeof(int32_t) ||
        !table || (uintptr_t)table > UINTPTR_MAX - table_bytes || !ck_checked_region(output, extent)) return -1;
    const uintptr_t dst = (uintptr_t)output;
    const uintptr_t stop = dst + extent * sizeof(float);
    if ((dst < (uintptr_t)ids + tokens * sizeof(int32_t) && (uintptr_t)ids < stop) ||
        (dst < (uintptr_t)table + table_bytes && (uintptr_t)table < stop)) return -1;
    const unsigned char *packed = table;
    for (size_t t = 0; t < tokens; ++t) {
        if (ids[t] < 0 || (size_t)ids[t] >= vocabulary) return -3;
        const unsigned char *row = packed + (size_t)ids[t] * row_bytes;
        for (size_t c = 0; c < width / QK8_0; ++c) {
            const unsigned char *p = row + c * sizeof(block_q8_0);
            const uint16_t bits = (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
            if ((bits & 0x7c00u) == 0x7c00u) return -3;
        }
    }
    for (size_t t = 0; t < tokens; ++t) {
        const unsigned char *p = packed + (size_t)ids[t] * row_bytes;
        for (size_t c = 0; c < width; c += QK8_0, p += sizeof(block_q8_0)) {
            const float scale = ck_fp16_to_fp32_soft((uint16_t)(p[0] | ((uint16_t)p[1] << 8)));
            for (size_t j = 0; j < QK8_0; ++j) {
                const int code = p[2 + j];
                output[t * width + c + j] = (float)(code < 128 ? code : code - 256) * scale;
            }
        }
    }
    return 0;
}

int softcap_logits_checked_f32(const float *input, size_t input_elements, float *output,
                               size_t output_elements, size_t elements, float cap)
{
    if (!elements || input_elements < elements || output_elements < elements ||
        !isfinite(cap) || cap <= 0.0f) return -2;
    if (!ck_checked_region(input, elements) || !ck_checked_region(output, elements) ||
        ck_checked_overlap(input, elements, output, elements)) return -1;
    if (!ck_checked_finite(input, elements)) return -3;
    for (size_t i = 0; i < elements; ++i)
        output[i] = (float)((double)cap * tanh((double)input[i] / (double)cap));
    return 0;
}
