#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "bf16_utils.h"

int ck_linear_rows_bf16_checked(const uint16_t *, size_t, const uint16_t *, size_t,
                                uint16_t *, size_t, uint16_t *, size_t, size_t, size_t, size_t);
int ck_linear_rows_bf16_blas_checked(const uint16_t *, size_t, const uint16_t *, size_t,
                                     uint16_t *, size_t, uint16_t *, size_t, double *, size_t,
                                     size_t, size_t, size_t);
int ck_geglu_tanh_bf16_checked(const uint16_t *, const uint16_t *, uint16_t *, float *, size_t);

typedef struct {
    double *blas;
    size_t blas_elements;
    uint16_t *stage;
    size_t stage_elements;
    int use_blas;
} projection_workspace;

static int project(const uint16_t *input, const uint16_t *weight, uint16_t *out,
                   size_t rows, size_t in_width, size_t out_width, projection_workspace work)
{
    if (work.use_blas)
        return ck_linear_rows_bf16_blas_checked(
            input, rows * in_width, weight, in_width * out_width, out,
            rows * out_width, work.stage, work.stage_elements, work.blas,
            work.blas_elements, rows, in_width, out_width);
    return ck_linear_rows_bf16_checked(
        input, rows * in_width, weight, in_width * out_width, out,
        rows * out_width, work.stage, work.stage_elements, rows, in_width, out_width);
}

static int span(const void *ptr, size_t bytes, size_t alignment)
{
    return ptr && (uintptr_t)ptr % alignment == 0 && bytes <= UINTPTR_MAX - (uintptr_t)ptr;
}

static int overlaps(const void *a, size_t a_bytes, const void *b, size_t b_bytes)
{
    return (uintptr_t)a < (uintptr_t)b + b_bytes && (uintptr_t)b < (uintptr_t)a + a_bytes;
}

int ck_routed_geglu_bf16_checked(
    const uint16_t *input, const int32_t *ids, const float *scores,
    const uint16_t *gate_up, const uint16_t *down, uint16_t *out, uint16_t *routes,
    void *workspace, size_t workspace_bytes, size_t rows, size_t width,
    size_t inner, size_t experts, size_t top_k, int use_blas,
    size_t hidden_elements, size_t selected_elements, size_t route_elements,
    size_t gate_elements, size_t down_elements)
{
    // Each sized domain fits a signed 32-bit BLAS extent. Cap products before
    // multiplying so all subsequent byte calculations also fit size_t.
    if (!rows || !width || !inner || !experts || !top_k || top_k > experts ||
        rows > INT_MAX || width > INT_MAX || inner > INT_MAX / 2 || experts > INT32_MAX ||
        width > SIZE_MAX / 64 || inner > SIZE_MAX / 64 || rows > SIZE_MAX / (64 * (width + inner)) ||
        experts > SIZE_MAX / (64 * width) / inner || top_k > SIZE_MAX / (64 * rows) / width)
        return -2;
    const size_t rh = rows * width, ri = rows * inner;
    const size_t rk = rows * top_k, routed = rk * width;
    const size_t hi = width * inner, blas_elements = rh + 2 * hi + 2 * ri;
    const size_t work_bytes = 8 * blas_elements + 4 * inner + 2 * (4 * rh + 5 * ri + routed);
    if (workspace_bytes < work_bytes || (use_blas != 0 && use_blas != 1) ||
        hidden_elements != rh || selected_elements != rk || route_elements != routed ||
        gate_elements != experts * 2 * hi || down_elements != experts * hi)
        return -2;
    const void *inputs[] = {input, ids, scores, gate_up, down};
    const size_t sizes[] = {rh * 2, rk * 4, rk * 4, experts * hi * 4, experts * hi * 2};
    const size_t aligns[] = {2, 4, 4, 2, 2};
    if (!span(out, rh * 2, 2) || !span(routes, routed * 2, 2) ||
        !span(workspace, work_bytes, _Alignof(double)) ||
        overlaps(out, rh * 2, routes, routed * 2) ||
        overlaps(out, rh * 2, workspace, work_bytes) ||
        overlaps(routes, routed * 2, workspace, work_bytes))
        return -2;
    for (size_t i = 0; i < sizeof(inputs) / sizeof(inputs[0]); ++i)
        if (!span(inputs[i], sizes[i], aligns[i]) ||
            overlaps(inputs[i], sizes[i], out, rh * 2) ||
            overlaps(inputs[i], sizes[i], routes, routed * 2) ||
            overlaps(inputs[i], sizes[i], workspace, work_bytes))
            return -2;

    for (size_t row = 0; row < rows; ++row) {
        for (size_t k = 0; k < top_k; ++k) {
            const size_t index = row * top_k + k;
            if (ids[index] < 0 || (size_t)ids[index] >= experts || !isfinite(scores[index]))
                return -3;
            for (size_t previous = 0; previous < k; ++previous)
                if (ids[index] == ids[row * top_k + previous])
                    return -3;
        }
    }

    double *blas = workspace;
    float *gelu = (float *)(blas + blas_elements);
    uint16_t *gathered = (uint16_t *)(gelu + inner);
    uint16_t *gate = gathered + rh;
    uint16_t *active = gate + 2 * ri;
    uint16_t *projected = active + ri;
    uint16_t *stage = projected + rh;
    uint16_t *accum = stage + rh + 2 * ri;
    uint16_t *route_stage = accum + rh;
    memset(accum, 0, rh * sizeof(*accum));
    const projection_workspace projection = {blas, blas_elements, stage, rh + 2 * ri, use_blas};

    // Match expert-major index_add order. The route slot is used only to
    // associate the input row and scaled probability with this expert.
    for (size_t expert = 0; expert < experts; ++expert) {
        size_t count = 0;
        for (size_t index = 0; index < rk; ++index)
            if ((size_t)ids[index] == expert) {
                memcpy(gathered + count * width, input + (index / top_k) * width, width * 2);
                ++count;
            }
        if (!count)
            continue;
        int status = project(gathered, gate_up + expert * 2 * hi, gate, count, width, 2 * inner, projection);
        if (status)
            return status;
        for (size_t row = 0; row < count; ++row) {
            status = ck_geglu_tanh_bf16_checked(gate + row * 2 * inner,
                gate + row * 2 * inner + inner, active + row * inner, gelu, inner);
            if (status)
                return status;
        }
        status = project(active, down + expert * hi, projected, count, inner, width, projection);
        if (status)
            return status;
        size_t row = 0;
        for (size_t index = 0; index < rk; ++index) {
            if ((size_t)ids[index] != expert)
                continue;
            uint16_t *destination = accum + (index / top_k) * width;
            const uint16_t *unweighted = projected + row * width;
            memcpy(route_stage + index * width, unweighted, width * 2);
            for (size_t col = 0; col < width; ++col) {
                const uint16_t scaled = float_to_bf16(bf16_to_float(unweighted[col]) * scores[index]);
                const uint16_t sum = float_to_bf16(bf16_to_float(destination[col]) + bf16_to_float(scaled));
                if (!isfinite(bf16_to_float(sum)))
                    return -3;
                destination[col] = sum;
            }
            ++row;
        }
    }
    memcpy(out, accum, rh * 2);
    memcpy(routes, route_stage, routed * 2);
    return 0;
}
