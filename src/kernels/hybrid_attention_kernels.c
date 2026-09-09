#include "ckernel_engine.h"
#include "bf16_utils.h"
#include "ck_threadpool.h"

#include <dlfcn.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef float (*ck_hybrid_libm_f32_fn)(float);
static ck_hybrid_libm_f32_fn ck_hybrid_llama_expf = NULL;
static void *ck_hybrid_libm_handle = NULL;
static pthread_once_t ck_hybrid_libm_once = PTHREAD_ONCE_INIT;

static void ck_bind_hybrid_llama_libm(void) {
    ck_hybrid_libm_handle = dlopen("libm.so.6", RTLD_NOW | RTLD_LOCAL);
    if (ck_hybrid_libm_handle) {
        ck_hybrid_llama_expf =
            (ck_hybrid_libm_f32_fn)dlsym(ck_hybrid_libm_handle, "expf");
    }
    if (!ck_hybrid_llama_expf) {
        fprintf(stderr,
                "HARD KERNEL CONTRACT FAULT: llama.cpp attention gate "
                "requires expf from libm.so.6\n");
        abort();
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
static float hybrid_sigmoid(float x) {
    pthread_once(&ck_hybrid_libm_once, ck_bind_hybrid_llama_libm);
    return 1.0f / (1.0f + ck_hybrid_llama_expf(-x));
}

void split_q_gate_forward(const float *packed_qg,
                          float *q,
                          float *gate,
                          int rows,
                          int q_dim,
                          int gate_dim,
                          int group_dim) {
    const int packed_dim = q_dim + gate_dim;
    if (!packed_qg || !q || !gate || rows <= 0 || q_dim <= 0 || gate_dim <= 0) {
        return;
    }
    if (group_dim <= 0) {
        group_dim = q_dim;
    }
    const int q_groups = q_dim / group_dim;
    const int gate_group_dim = (q_groups > 0 && gate_dim % q_groups == 0) ? (gate_dim / q_groups) : gate_dim;
    for (int row = 0; row < rows; ++row) {
        const float *src = packed_qg + (size_t) row * (size_t) packed_dim;
        float *q_dst = q + (size_t) row * (size_t) q_dim;
        float *gate_dst = gate + (size_t) row * (size_t) gate_dim;
        if (q_groups > 0 && q_groups * group_dim == q_dim && q_groups * gate_group_dim == gate_dim) {
            for (int group = 0; group < q_groups; ++group) {
                const size_t src_group_off = (size_t) group * (size_t) (group_dim + gate_group_dim);
                memcpy(
                    q_dst + (size_t) group * (size_t) group_dim,
                    src + src_group_off,
                    (size_t) group_dim * sizeof(float));
                memcpy(
                    gate_dst + (size_t) group * (size_t) gate_group_dim,
                    src + src_group_off + (size_t) group_dim,
                    (size_t) gate_group_dim * sizeof(float));
            }
        } else {
            memcpy(q_dst, src, (size_t) q_dim * sizeof(float));
            memcpy(gate_dst, src + q_dim, (size_t) gate_dim * sizeof(float));
        }
    }
}

void split_q_gate_backward(const float *d_q,
                           const float *d_gate,
                           float *d_packed_qg,
                           int rows,
                           int q_dim,
                           int gate_dim,
                           int group_dim) {
    const int packed_dim = q_dim + gate_dim;
    if (!d_q || !d_gate || !d_packed_qg || rows <= 0 || q_dim <= 0 || gate_dim <= 0) {
        return;
    }
    if (group_dim <= 0) {
        group_dim = q_dim;
    }
    const int q_groups = q_dim / group_dim;
    const int gate_group_dim = (q_groups > 0 && gate_dim % q_groups == 0) ? (gate_dim / q_groups) : gate_dim;
    for (int row = 0; row < rows; ++row) {
        const float *dq_src = d_q + (size_t) row * (size_t) q_dim;
        const float *dg_src = d_gate + (size_t) row * (size_t) gate_dim;
        float *dst = d_packed_qg + (size_t) row * (size_t) packed_dim;
        if (q_groups > 0 && q_groups * group_dim == q_dim && q_groups * gate_group_dim == gate_dim) {
            for (int group = 0; group < q_groups; ++group) {
                const size_t dst_group_off = (size_t) group * (size_t) (group_dim + gate_group_dim);
                memcpy(
                    dst + dst_group_off,
                    dq_src + (size_t) group * (size_t) group_dim,
                    (size_t) group_dim * sizeof(float));
                memcpy(
                    dst + dst_group_off + (size_t) group_dim,
                    dg_src + (size_t) group * (size_t) gate_group_dim,
                    (size_t) gate_group_dim * sizeof(float));
            }
        } else {
            memcpy(dst, dq_src, (size_t) q_dim * sizeof(float));
            memcpy(dst + q_dim, dg_src, (size_t) gate_dim * sizeof(float));
        }
    }
}

typedef struct {
    const float *x;
    const float *gate;
    float *out;
    int dim;
} ck_attn_gate_sigmoid_mul_args_t;

static void attn_gate_sigmoid_mul_row_range(int begin,
                                            int end,
                                            void *opaque) {
    const ck_attn_gate_sigmoid_mul_args_t *args =
        (const ck_attn_gate_sigmoid_mul_args_t *)opaque;
    for (int row = begin; row < end; ++row) {
        const float *x_row =
            args->x + (size_t)row * (size_t)args->dim;
        const float *gate_row =
            args->gate + (size_t)row * (size_t)args->dim;
        float *out_row =
            args->out + (size_t)row * (size_t)args->dim;
        const int dim = args->dim;
        for (int col = 0; col < dim; ++col) {
            out_row[col] = x_row[col] * hybrid_sigmoid(gate_row[col]);
        }
    }
}

static int hybrid_ranges_overlap(const float *lhs,
                                 const float *rhs,
                                 size_t elements) {
    const uintptr_t lhs_begin = (uintptr_t)lhs;
    const uintptr_t rhs_begin = (uintptr_t)rhs;
    const size_t bytes = elements * sizeof(float);
    return lhs_begin < rhs_begin + bytes && rhs_begin < lhs_begin + bytes;
}

void attn_gate_sigmoid_mul_forward(const float *x,
                                   const float *gate,
                                   float *out,
                                   int rows,
                                   int num_heads,
                                   int state_dim) {
    if (!x || !gate || !out || rows <= 0 || num_heads <= 0 ||
        state_dim <= 0 || num_heads > INT_MAX / state_dim) {
        return;
    }
    const int dim = num_heads * state_dim;
    const size_t elements = (size_t)rows * (size_t)dim;
    ck_attn_gate_sigmoid_mul_args_t args = {x, gate, out, dim};
    const size_t minimum_elements_per_thread = 32768u;

    const int partial_x_overlap = out != x &&
        hybrid_ranges_overlap(out, x, elements);
    const int gate_overlap = hybrid_ranges_overlap(out, gate, elements);
    if (partial_x_overlap || gate_overlap ||
        elements < 2u * minimum_elements_per_thread) {
        attn_gate_sigmoid_mul_row_range(0, rows, &args);
        return;
    }

    ck_threadpool_t *pool = ck_threadpool_global();
    int active = pool ? ck_threadpool_n_threads(pool) : 1;
    size_t useful_threads =
        (elements + minimum_elements_per_thread - 1u) /
        minimum_elements_per_thread;
    if ((size_t)active > useful_threads) {
        active = (int)useful_threads;
    }
    if (active > rows) {
        active = rows;
    }

    if (active > 1 && ck_threadpool_thread_id(pool) <= 0) {
        ck_threadpool_parallel_for_n(
            pool, active, 0, rows, 4,
            attn_gate_sigmoid_mul_row_range, &args);
    } else {
        attn_gate_sigmoid_mul_row_range(0, rows, &args);
    }
}

void attn_gate_sigmoid_mul_pytorch_bf16_storage(const float *x,
                                                const float *gate,
                                                float *out,
                                                int rows,
                                                int num_heads,
                                                int state_dim) {
    if (!x || !gate || !out || rows <= 0 || num_heads <= 0 || state_dim <= 0) {
        return;
    }
    const int dim = num_heads * state_dim;
    for (int row = 0; row < rows; ++row) {
        const float *x_row = x + (size_t)row * (size_t)dim;
        const float *gate_row = gate + (size_t)row * (size_t)dim;
        float *out_row = out + (size_t)row * (size_t)dim;
        for (int col = 0; col < dim; col += 16) {
            const int width = dim - col < 16 ? dim - col : 16;
            float sigmoid[16];
            recurrent_sigmoid_forward_pytorch_bf16_input_fp32_output(
                gate_row + col, sigmoid, 1, width);
            for (int lane = 0; lane < width; ++lane) {
                const float x_bf16 = bf16_to_float(float_to_bf16(x_row[col + lane]));
                const float sigmoid_bf16 = bf16_to_float(float_to_bf16(sigmoid[lane]));
                out_row[col + lane] = bf16_to_float(float_to_bf16(
                    x_bf16 * sigmoid_bf16));
            }
        }
    }
}

void attn_gate_softplus_mul_forward(const float *x,
                                    const float *gate,
                                    float *out,
                                    int rows,
                                    int num_heads,
                                    int state_dim) {
    const int dim = num_heads * state_dim;
    for (int row = 0; row < rows; ++row) {
        const float *x_row = x + (size_t) row * (size_t) dim;
        const float *gate_row = gate + (size_t) row * (size_t) num_heads;
        float *out_row = out + (size_t) row * (size_t) dim;
        for (int head = 0; head < num_heads; ++head) {
            const float value = gate_row[head];
            const float scale = value > 20.0f
                ? value
                : log1pf(expf(value));
            const size_t base = (size_t) head * (size_t) state_dim;
            for (int col = 0; col < state_dim; ++col) {
                out_row[base + (size_t) col] =
                    x_row[base + (size_t) col] * scale;
            }
        }
    }
}

void attn_gate_sigmoid_mul_backward(const float *d_out,
                                    const float *x,
                                    const float *gate,
                                    float *d_x,
                                    float *d_gate,
                                    int rows,
                                    int num_heads,
                                    int state_dim) {
    const int dim = num_heads * state_dim;
    for (int row = 0; row < rows; ++row) {
        const float *d_out_row = d_out + (size_t) row * (size_t) dim;
        const float *x_row = x + (size_t) row * (size_t) dim;
        const float *gate_row = gate + (size_t) row * (size_t) dim;
        float *d_x_row = d_x + (size_t) row * (size_t) dim;
        float *d_gate_row = d_gate + (size_t) row * (size_t) dim;
        for (int col = 0; col < dim; ++col) {
            const float sig = hybrid_sigmoid(gate_row[col]);
            d_x_row[col] = d_out_row[col] * sig;
            d_gate_row[col] = d_out_row[col] * x_row[col] * sig * (1.0f - sig);
        }
    }
}
