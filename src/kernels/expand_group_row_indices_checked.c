#include <stdint.h>
#include <stddef.h>

static int region_ok(const void *ptr, size_t count, size_t element_bytes) {
    return ptr && (uintptr_t)ptr % element_bytes == 0 &&
           count <= SIZE_MAX / element_bytes &&
           (uintptr_t)ptr <= UINTPTR_MAX - count * element_bytes;
}

static int overlaps(const void *a, size_t a_bytes,
                    const void *b, size_t b_bytes) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x < y + b_bytes && y < x + a_bytes;
}

/* Turn ordered group sizes into an identity row index and its owning group.
 * All counts are validated before either output is written. */
int ck_expand_group_row_indices_checked_i32(
    const int32_t *group_counts, size_t group_count_elements,
    size_t groups, size_t rows,
    int32_t *row_indices, size_t row_index_elements,
    int32_t *owner_indices, size_t owner_index_elements) {
    if (!groups || !rows || groups > INT32_MAX || rows > INT32_MAX ||
        group_count_elements < groups || row_index_elements < rows ||
        owner_index_elements < rows)
        return -2;
    if (!region_ok(group_counts, group_count_elements, sizeof(int32_t)) ||
        !region_ok(row_indices, row_index_elements, sizeof(int32_t)) ||
        !region_ok(owner_indices, owner_index_elements, sizeof(int32_t)))
        return -1;
    if (overlaps(group_counts, group_count_elements * sizeof(int32_t),
                 row_indices, row_index_elements * sizeof(int32_t)) ||
        overlaps(group_counts, group_count_elements * sizeof(int32_t),
                 owner_indices, owner_index_elements * sizeof(int32_t)) ||
        overlaps(row_indices, row_index_elements * sizeof(int32_t),
                 owner_indices, owner_index_elements * sizeof(int32_t)))
        return -1;
    size_t total = 0;
    for (size_t group = 0; group < groups; ++group) {
        int32_t count = group_counts[group];
        if (count <= 0 || (size_t)count > rows - total) return -2;
        total += (size_t)count;
    }
    if (total != rows) return -2;
    size_t row = 0;
    for (size_t group = 0; group < groups; ++group) {
        for (int32_t item = 0; item < group_counts[group]; ++item, ++row) {
            row_indices[row] = (int32_t)row;
            owner_indices[row] = (int32_t)group;
        }
    }
    return 0;
}
