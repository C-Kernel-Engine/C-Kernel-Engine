#include "checked_float_buffer.h"
#include "ckernel_quant.h"
#include "ckernel_transformer_rows_checked.h"

#include <string.h>

enum expert_storage {
    EXPERT_F32,
    EXPERT_Q8_0,
    EXPERT_IQ4_NL,
};

#define IQ4_NL_BLOCK_BYTES 18

/* GGML IQ4_NL codebook and split-half nibble layout. */
static const int8_t iq4_nl_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

static size_t packed_block_bytes(enum expert_storage storage)
{
    return storage == EXPERT_Q8_0 ? sizeof(block_q8_0) : IQ4_NL_BLOCK_BYTES;
}

/* Packed blocks are read in GGUF byte order and expanded inside the dot.
 * The FP32 activation and ordered FP64 accumulation policy is shared. */
static float expert_dot(const float *input, const unsigned char *weight,
                        size_t width, enum expert_storage storage)
{
    double sum = 0.0;
    if (storage == EXPERT_F32) {
        for (size_t i = 0; i < width; ++i) {
            float value;
            memcpy(&value, weight + i * sizeof(float), sizeof(float));
            sum += (double)input[i] * (double)value;
        }
    } else {
        const size_t block_bytes = packed_block_bytes(storage);
        for (size_t i = 0; i < width; i += QK8_0, weight += block_bytes) {
            const float scale = ck_fp16_to_fp32_soft(
                (uint16_t)(weight[0] | ((uint16_t)weight[1] << 8)));
            for (size_t j = 0; j < QK8_0; ++j) {
                int value;
                if (storage == EXPERT_Q8_0) {
                    value = weight[j + 2];
                    if (value >= 128) value -= 256;
                } else {
                    const unsigned pair = weight[2 + j % 16];
                    value = iq4_nl_values[j < 16 ? pair & 15 : pair >> 4];
                }
                const float expanded = scale * (float)value;
                sum += (double)input[i + j] * (double)expanded;
            }
        }
    }
    return (float)sum;
}

static int contains_overlap(const void *p, size_t pn, const void *q, size_t qn)
{
    return (uintptr_t)p < (uintptr_t)q + qn && (uintptr_t)q < (uintptr_t)p + pn;
}

static int region(const void *p, size_t bytes)
{
    return p && bytes && (uintptr_t)p <= UINTPTR_MAX - bytes;
}

#define ROUTED_EXPERT_ARGS \
    const float *hidden, const int32_t *ids, const float *route_weights, \
    const int32_t *expert_slots, const void *gate, const void *up, const void *down, \
    float *expert_output, float *output, float *work, \
    size_t rows, size_t hidden_dim, size_t inner, size_t experts, size_t slots, \
    size_t top_k, size_t matrix_bytes, size_t hidden_elements, size_t id_elements, size_t route_elements, \
    size_t expert_slot_elements, size_t expert_output_elements, size_t output_capacity, \
    size_t work_elements, float gate_cap, float up_cap

static int routed_softcapped_experts(ROUTED_EXPERT_ARGS, enum expert_storage storage)
{
    /* Bounds keep workspace arithmetic valid on supported size_t widths. */
    if (!rows || rows > 16 || !hidden_dim || hidden_dim > 8192 ||
        !inner || inner > 8192 || !experts || experts > 1024 || !slots ||
        slots > experts || !top_k || top_k > experts ||
        !isfinite(gate_cap) || gate_cap <= 0.0f || !isfinite(up_cap) || up_cap <= 0.0f)
        return -1;
    const int packed = storage != EXPERT_F32;
    if (packed && (hidden_dim % QK8_0 || inner % QK8_0)) return -1;
    const size_t block_bytes = packed ? packed_block_bytes(storage) : sizeof(float);
    const size_t block_elements = packed ? QK8_0 : 1;
    const size_t gate_row = hidden_dim / block_elements * block_bytes;
    const size_t down_row = inner / block_elements * block_bytes;
    const size_t expert_bytes = inner * gate_row;
    const size_t route_count = rows * top_k;
    const size_t output_elements = rows * hidden_dim;
    const size_t row_outputs = route_count * hidden_dim;
    const size_t scratch_elements = 3 * inner + row_outputs + output_elements;
    if (hidden_elements < output_elements || id_elements < route_count || route_elements < route_count ||
        expert_slot_elements < experts || expert_output_elements < row_outputs ||
        output_capacity < output_elements || work_elements < scratch_elements) return -1;
    if (expert_bytes != hidden_dim * down_row || expert_bytes > SIZE_MAX / slots ||
        matrix_bytes != expert_bytes * slots) return -1;
    const void *buffers[] = {hidden, ids, route_weights, expert_slots, gate, up, down,
                             expert_output, output, work};
    const size_t sizes[] = {output_elements * sizeof(float), route_count * sizeof(int32_t),
                            route_count * sizeof(float), experts * sizeof(int32_t),
                            matrix_bytes, matrix_bytes, matrix_bytes, row_outputs * sizeof(float),
                            output_elements * sizeof(float),
                            scratch_elements * sizeof(float)};
    for (size_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); ++i) {
        if (!region(buffers[i], sizes[i])) return -1;
        if (i < 4 || i >= 7) {
            if ((uintptr_t)buffers[i] % _Alignof(float)) return -1;
        }
        if (i < 7) continue;
        for (size_t j = 0; j < i; ++j)
            if (contains_overlap(buffers[i], sizes[i], buffers[j], sizes[j])) return -1;
    }
    if (!ck_checked_finite(hidden, output_elements) ||
        !ck_checked_finite(route_weights, route_count)) return -2;

    unsigned char assigned[1024] = {0};
    for (size_t id = 0; id < experts; ++id) {
        const int32_t slot = expert_slots[id];
        if (slot == -1) continue;
        if (slot < 0 || (size_t)slot >= slots || assigned[slot]) return -2;
        assigned[slot] = 1;
    }
    for (size_t row = 0; row < rows; ++row) {
        for (size_t k = 0; k < top_k; ++k) {
            const size_t route = row * top_k + k;
            const int32_t id = ids[route];
            if (id < 0 || (size_t)id >= experts || expert_slots[id] < 0 ||
                (size_t)expert_slots[id] >= slots || route_weights[route] < 0.0f) return -2;
            for (size_t prev = 0; prev < k; ++prev)
                if (id == ids[row * top_k + prev]) return -2;
        }
    }

    float *gate_values = work;
    float *up_values = work + inner;
    float *activation = work + 2 * inner;
    float *staged_experts = work + 3 * inner;
    float *staged_output = staged_experts + row_outputs;
    memset(staged_output, 0, output_elements * sizeof(float));
    for (size_t row = 0; row < rows; ++row) {
        const float *x = hidden + row * hidden_dim;
        float *y = staged_output + row * hidden_dim;
        /* The reference sums contributions in ascending expert-ID order. */
        for (size_t id = 0; id < experts; ++id) {
            for (size_t index = 0; index < top_k; ++index) {
                const size_t route = row * top_k + index;
                if ((size_t)ids[route] != id) continue;
                const size_t offset = (size_t)expert_slots[id] * expert_bytes;
                const unsigned char *g = (const unsigned char *)gate + offset;
                const unsigned char *u = (const unsigned char *)up + offset;
                const unsigned char *d = (const unsigned char *)down + offset;
                for (size_t i = 0; i < inner; ++i) {
                    gate_values[i] = expert_dot(x, g + i * gate_row, hidden_dim, storage);
                    up_values[i] = expert_dot(x, u + i * gate_row, hidden_dim, storage);
                }
                if (softcapped_swiglu_rows_checked_f32(gate_values, inner, up_values, inner,
                        activation, inner, inner, gate_cap, up_cap)) return -3;
                for (size_t i = 0; i < hidden_dim; ++i) {
                    const float v = expert_dot(activation, d + i * down_row, inner, storage);
                    staged_experts[route * hidden_dim + i] = v;
                    y[i] += v * route_weights[route];
                    if (!isfinite(v) || !isfinite(y[i])) return -3;
                }
            }
        }
    }
    memcpy(expert_output, staged_experts, row_outputs * sizeof(float));
    memcpy(output, staged_output, output_elements * sizeof(float));
    return 0;
}

#define ROUTED_EXPERT_FORWARD(storage) \
    routed_softcapped_experts(hidden, ids, route_weights, expert_slots, gate, up, down, \
                             expert_output, output, work, rows, hidden_dim, inner, experts, \
                             slots, top_k, matrix_bytes, hidden_elements, id_elements, route_elements, \
                             expert_slot_elements, expert_output_elements, output_capacity, \
                             work_elements, gate_cap, up_cap, storage)

int routed_softcapped_experts_checked_f32(ROUTED_EXPERT_ARGS)
{
    return ROUTED_EXPERT_FORWARD(EXPERT_F32);
}

int routed_softcapped_experts_checked_q8_0(ROUTED_EXPERT_ARGS)
{
    return ROUTED_EXPERT_FORWARD(EXPERT_Q8_0);
}

int routed_softcapped_experts_checked_iq4_nl(ROUTED_EXPERT_ARGS)
{
    return ROUTED_EXPERT_FORWARD(EXPERT_IQ4_NL);
}
