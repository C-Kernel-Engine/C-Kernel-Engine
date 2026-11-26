#include <stddef.h>
#include <stdint.h>
#include <string.h>

static int valid_span(const uint16_t *data, size_t elements)
{
    return data && (uintptr_t)data % _Alignof(uint16_t) == 0 &&
           elements <= (UINTPTR_MAX - (uintptr_t)data) / sizeof(*data);
}

static int overlaps(const uint16_t *out, size_t out_elements,
                    const uint16_t *in, size_t in_elements)
{
    return (uintptr_t)out < (uintptr_t)in + in_elements * sizeof(*in) &&
           (uintptr_t)in < (uintptr_t)out + out_elements * sizeof(*out);
}

int ck_concat_features_bf16_checked(const uint16_t *left, const uint16_t *right,
                                     uint16_t *out, size_t rows, size_t left_width,
                                     size_t right_width, size_t left_elements,
                                     size_t right_elements, size_t out_elements,
                                     size_t out_width)
{
    if (!rows || !left_width || !right_width || left_width > SIZE_MAX - right_width)
        return -2;
    const size_t width = left_width + right_width;
    if (out_width != width || rows > SIZE_MAX / width || left_elements != rows * left_width ||
        right_elements != rows * right_width || out_elements != rows * width)
        return -2;
    if (!valid_span(left, left_elements) || !valid_span(right, right_elements) ||
        !valid_span(out, out_elements) || overlaps(out, out_elements, left, left_elements) ||
        overlaps(out, out_elements, right, right_elements))
        return -1;
    for (size_t row = 0; row < rows; ++row) {
        memcpy(out + row * width, left + row * left_width, left_width * sizeof(*out));
        memcpy(out + row * width + left_width, right + row * right_width, right_width * sizeof(*out));
    }
    return 0;
}
