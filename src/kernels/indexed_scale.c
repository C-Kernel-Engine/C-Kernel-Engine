#include "checked_float_buffer.h"

int ck_indexed_scale_f32(const float *values, const int32_t *ids,
                         const float *scales, float *output,
                         size_t count, size_t scale_count) {
    if (!count || !scale_count ||
        !ck_checked_region(values, count) || !ck_checked_region(scales, scale_count) ||
        !ck_checked_region(output, count) || !ids ||
        (uintptr_t)ids % _Alignof(int32_t) != 0 ||
        count > (UINTPTR_MAX - (uintptr_t)ids) / sizeof(*ids)) {
        return -1;
    }

    uintptr_t output_begin = (uintptr_t)output;
    uintptr_t ids_begin = (uintptr_t)ids;
    if (ck_checked_overlap(values, count, output, count) ||
        ck_checked_overlap(scales, scale_count, output, count) ||
        (output_begin < ids_begin + count * sizeof(*ids) &&
         ids_begin < output_begin + count * sizeof(*output))) {
        return -1;
    }

    for (size_t i = 0; i < count; ++i) {
        int32_t id = ids[i];
        if (id < 0 || (size_t)id >= scale_count) {
            return -2;
        }
        float product = values[i] * scales[id];
        if (!isfinite(product)) {
            return -3;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        output[i] = values[i] * scales[ids[i]];
    }
    return 0;
}
