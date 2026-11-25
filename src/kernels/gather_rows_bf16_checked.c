#include <stddef.h>
#include <stdint.h>
#include <string.h>

static int valid_region(const void *pointer, size_t count, size_t element_bytes)
{
    return pointer && (uintptr_t)pointer % element_bytes == 0 &&
           count <= SIZE_MAX / element_bytes &&
           (uintptr_t)pointer <= UINTPTR_MAX - count * element_bytes;
}

static int overlaps(const void *a, size_t a_bytes, const void *b, size_t b_bytes)
{
    return (uintptr_t)a < (uintptr_t)b + b_bytes &&
           (uintptr_t)b < (uintptr_t)a + a_bytes;
}

int ck_gather_rows_bf16_checked(
    const uint16_t *input, size_t input_elements,
    const int32_t *indices, size_t index_elements,
    uint16_t *output, size_t output_elements,
    size_t input_rows, size_t output_rows, size_t width)
{
    if (!input_rows || !output_rows || !width ||
        input_rows > SIZE_MAX / width || output_rows > SIZE_MAX / width ||
        input_elements < input_rows * width || index_elements < output_rows ||
        output_elements < output_rows * width)
        return -2;

    if (!valid_region(input, input_elements, sizeof(*input)) ||
        !valid_region(indices, index_elements, sizeof(*indices)) ||
        !valid_region(output, output_elements, sizeof(*output)) ||
        overlaps(output, output_elements * sizeof(*output),
                 input, input_elements * sizeof(*input)) ||
        overlaps(output, output_elements * sizeof(*output),
                 indices, index_elements * sizeof(*indices)))
        return -1;

    for (size_t i = 0; i < output_rows; ++i)
        if (indices[i] < 0 || (size_t)indices[i] >= input_rows)
            return -2;

    for (size_t i = 0; i < output_rows; ++i)
        memcpy(output + i * width, input + (size_t)indices[i] * width,
               width * sizeof(*output));
    return 0;
}
