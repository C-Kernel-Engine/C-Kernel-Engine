#include "ckernel_engine.h"

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>


static int ck_mamba_debug_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("CK_DEBUG_MAMBA");
        cached = (v && v[0] && v[0] != '0') ? 1 : 0;
    }
    return cached;
}

static void ck_mamba_debug_finite(const char *name, const float *x, size_t n) {
    if (!ck_mamba_debug_enabled() || !x || n == 0) {
        return;
    }
    size_t finite = 0, nan = 0, inf = 0;
    float min_v = 0.0f, max_v = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float v = x[i];
        if (isnan(v)) {
            ++nan;
        } else if (!isfinite(v)) {
            ++inf;
        } else {
            if (finite == 0 || v < min_v) min_v = v;
            if (finite == 0 || v > max_v) max_v = v;
            ++finite;
        }
    }
    fprintf(stderr, "[CK_DEBUG_MAMBA] %s finite=%zu/%zu nan=%zu inf=%zu min=%g max=%g\n",
            name, finite, n, nan, inf, finite ? (double)min_v : 0.0, finite ? (double)max_v : 0.0);
}

static inline float mamba2_sigmoid_f32(float x) {
    if (x >= 0.0f) {
        const float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    {
        const float z = expf(x);
        return z / (1.0f + z);
    }
}

static inline float mamba2_silu_f32(float x) {
    return x * mamba2_sigmoid_f32(x);
}

static inline float mamba2_softplus_f32(float x) {
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return expf(x);
    }
    return log1pf(expf(x));
}

void mamba2_in_proj_split_f32(const float *projected,
                              float *gate,
                              float *hidden_bc,
                              float *dt,
                              int rows,
                              int d_mlp,
                              int intermediate_dim,
                              int conv_dim,
                              int num_heads) {
    if (!projected || !gate || !hidden_bc || !dt ||
        rows <= 0 || d_mlp < 0 || intermediate_dim <= 0 || conv_dim <= 0 || num_heads <= 0) {
        return;
    }

    const int projection_dim = 2 * d_mlp + intermediate_dim + conv_dim + num_heads;
    const int gate_offset = 2 * d_mlp;
    const int hidden_bc_offset = gate_offset + intermediate_dim;
    const int dt_offset = hidden_bc_offset + conv_dim;

    ck_mamba_debug_finite("split.projected[0]", projected, (size_t)projection_dim);
    for (int row = 0; row < rows; ++row) {
        const float *src = projected + (size_t)row * (size_t)projection_dim;
        memcpy(gate + (size_t)row * (size_t)intermediate_dim,
               src + gate_offset,
               (size_t)intermediate_dim * sizeof(float));
        memcpy(hidden_bc + (size_t)row * (size_t)conv_dim,
               src + hidden_bc_offset,
               (size_t)conv_dim * sizeof(float));
        memcpy(dt + (size_t)row * (size_t)num_heads,
               src + dt_offset,
               (size_t)num_heads * sizeof(float));
    }
    ck_mamba_debug_finite("split.gate[0]", gate, (size_t)intermediate_dim);
    ck_mamba_debug_finite("split.hidden_bc[0]", hidden_bc, (size_t)conv_dim);
    ck_mamba_debug_finite("split.dt[0]", dt, (size_t)num_heads);
}

void mamba2_conv1d_f32_channel_range(const float *state_in,
                                     const float *x,
                                     const float *weight,
                                     const float *bias,
                                     float *conv_out,
                                     float *state_out,
                                     int rows,
                                     int conv_dim,
                                     int kernel_size,
                                     int channel_begin,
                                     int channel_end) {
    if (!state_in || !x || !weight || !conv_out || !state_out ||
        rows <= 0 || conv_dim <= 0 || kernel_size <= 0 ||
        channel_begin < 0 || channel_begin >= channel_end ||
        channel_end > conv_dim) {
        return;
    }

    /* Time rows are dependent, but channels are independent. Keeping one
     * channel's rolling state local preserves its exact update order while
     * allowing disjoint channel ranges to execute concurrently.
     */
    for (int ch = channel_begin; ch < channel_end; ++ch) {
        float state_work[(size_t)kernel_size];
        const size_t base = (size_t)ch * (size_t)kernel_size;
        memcpy(state_work, state_in + base,
               (size_t)kernel_size * sizeof(float));
        for (int row = 0; row < rows; ++row) {
            for (int k = 0; k < kernel_size - 1; ++k) {
                state_work[(size_t)k] = state_work[(size_t)k + 1u];
            }
            state_work[(size_t)kernel_size - 1u] =
                x[(size_t)row * (size_t)conv_dim + (size_t)ch];

            float acc = bias ? bias[ch] : 0.0f;
            for (int k = 0; k < kernel_size; ++k) {
                acc += state_work[(size_t)k] *
                       weight[(size_t)ch * (size_t)kernel_size + (size_t)k];
            }
            conv_out[(size_t)row * (size_t)conv_dim + (size_t)ch] = mamba2_silu_f32(acc);
        }
        memcpy(state_out + base, state_work,
               (size_t)kernel_size * sizeof(float));
    }
}

void mamba2_conv1d_decode_f32(const float *state_in,
                              const float *x,
                              const float *weight,
                              const float *bias,
                              float *conv_out,
                              float *state_out,
                              int rows,
                              int conv_dim,
                              int kernel_size) {
    if (!state_in || !x || !weight || !conv_out || !state_out ||
        rows <= 0 || conv_dim <= 0 || kernel_size <= 0) {
        return;
    }

    ck_mamba_debug_finite("conv.state_in", state_in, (size_t)conv_dim * (size_t)kernel_size);
    ck_mamba_debug_finite("conv.x[0]", x, (size_t)conv_dim);
    mamba2_conv1d_f32_channel_range(
        state_in, x, weight, bias, conv_out, state_out,
        rows, conv_dim, kernel_size, 0, conv_dim);
    ck_mamba_debug_finite("conv.out[0]", conv_out, (size_t)conv_dim);
    ck_mamba_debug_finite("conv.state_out", state_out, (size_t)conv_dim * (size_t)kernel_size);
}

void mamba2_dt_softplus_f32(const float *dt,
                            const float *dt_bias,
                            float *dt_out,
                            int rows,
                            int num_heads,
                            float dt_min,
                            float dt_max) {
    if (!dt || !dt_out || rows <= 0 || num_heads <= 0) {
        return;
    }

    ck_mamba_debug_finite("dt.in[0]", dt, (size_t)num_heads);
    ck_mamba_debug_finite("dt.bias", dt_bias, (size_t)num_heads);
    for (int row = 0; row < rows; ++row) {
        for (int h = 0; h < num_heads; ++h) {
            float v = dt[(size_t)row * (size_t)num_heads + (size_t)h];
            if (dt_bias) {
                v += dt_bias[h];
            }
            v = mamba2_softplus_f32(v);
            if (dt_min < dt_max) {
                if (v < dt_min) {
                    v = dt_min;
                } else if (v > dt_max) {
                    v = dt_max;
                }
            }
            dt_out[(size_t)row * (size_t)num_heads + (size_t)h] = v;
        }
    }
    ck_mamba_debug_finite("dt.out[0]", dt_out, (size_t)num_heads);
}

void mamba2_selective_state_update_decode_f32(const float *state_in,
                                              const float *x,
                                              const float *dt,
                                              const float *a,
                                              const float *b,
                                              const float *c,
                                              const float *d,
                                              float *state_out,
                                              float *y,
                                              int rows,
                                              int num_heads,
                                              int head_dim,
                                              int state_dim,
                                              int num_groups) {
    if (!state_in || !x || !dt || !a || !b || !c || !d || !state_out || !y ||
        rows <= 0 || num_heads <= 0 || head_dim <= 0 || state_dim <= 0 || num_groups <= 0) {
        return;
    }

    const int packed_xbc = (x == b && b == c);
    const size_t inner_dim = (size_t)num_heads * (size_t)head_dim;
    const size_t bc_dim = (size_t)num_groups * (size_t)state_dim;
    const size_t packed_stride = inner_dim + 2u * bc_dim;

    for (int row = 0; row < rows; ++row) {
        const float *packed_row = packed_xbc ? x + (size_t)row * packed_stride : NULL;
        for (int h = 0; h < num_heads; ++h) {
            const int heads_per_group = (num_heads + num_groups - 1) / num_groups;
            int group = h / heads_per_group;
            if (group >= num_groups) group = num_groups - 1;
            const float dt_h = dt[(size_t)row * (size_t)num_heads + (size_t)h];
            const float d_a = expf(dt_h * a[h]);
            const float d_h = d[h];
            const float *b_row = packed_xbc
                ? (packed_row + inner_dim + (size_t)group * (size_t)state_dim)
                : (b + ((size_t)row * (size_t)num_groups + (size_t)group) * (size_t)state_dim);
            const float *c_row = packed_xbc
                ? (packed_row + inner_dim + bc_dim + (size_t)group * (size_t)state_dim)
                : (c + ((size_t)row * (size_t)num_groups + (size_t)group) * (size_t)state_dim);

            for (int hd = 0; hd < head_dim; ++hd) {
                const size_t x_idx = ((size_t)row * (size_t)num_heads + (size_t)h) * (size_t)head_dim + (size_t)hd;
                const float x_val = packed_xbc ? packed_row[(size_t)h * (size_t)head_dim + (size_t)hd] : x[x_idx];
                const size_t state_base =
                    (((size_t)row * (size_t)num_heads + (size_t)h) * (size_t)head_dim + (size_t)hd) *
                    (size_t)state_dim;
                float acc = 0.0f;
                for (int s = 0; s < state_dim; ++s) {
                    const size_t si = state_base + (size_t)s;
                    const float new_state = state_in[si] * d_a + dt_h * b_row[s] * x_val;
                    state_out[si] = new_state;
                    acc += new_state * c_row[s];
                }
                y[x_idx] = acc + d_h * x_val;
            }
        }
    }
}


void mamba2_selective_scan_f32_head_range(const float *state_init,
                                          const float *x,
                                          const float *dt,
                                          const float *a,
                                          const float *b,
                                          const float *c,
                                          const float *d,
                                          float *state_out,
                                          float *y,
                                          int batch,
                                          int seq_len,
                                          int num_heads,
                                          int head_dim,
                                          int state_dim,
                                          int num_groups,
                                          int head_begin,
                                          int head_end) {
    if (!state_init || !x || !dt || !a || !b || !c || !d || !state_out || !y ||
        batch <= 0 || seq_len <= 0 || num_heads <= 0 || head_dim <= 0 ||
        state_dim <= 0 || num_groups <= 0 || head_begin < 0 ||
        head_begin >= head_end || head_end > num_heads) {
        return;
    }

    const size_t state_per_batch = (size_t)num_heads * (size_t)head_dim * (size_t)state_dim;
    const size_t head_state = (size_t)head_dim * (size_t)state_dim;
    const int packed_xbc = (x == b && b == c);
    const size_t inner_dim = (size_t)num_heads * (size_t)head_dim;
    const size_t bc_dim = (size_t)num_groups * (size_t)state_dim;
    const size_t packed_stride = inner_dim + 2u * bc_dim;
    const int heads_per_group = (num_heads + num_groups - 1) / num_groups;

    for (int bs = 0; bs < batch; ++bs) {
        float *state_batch = state_out + (size_t)bs * state_per_batch;
        memcpy(state_batch + (size_t)head_begin * head_state,
               state_init + (size_t)bs * state_per_batch + (size_t)head_begin * head_state,
               (size_t)(head_end - head_begin) * head_state * sizeof(float));
        for (int t = 0; t < seq_len; ++t) {
            for (int h = head_begin; h < head_end; ++h) {
                /* Nemotron-H prefill and decode both use the repeating
                 * B/C group map from repeat_interleave semantics.  Keeping
                 * this in sync is required for full-prefix and incremental
                 * decode equivalence.
                 */
                int group = h / heads_per_group;
                if (group >= num_groups) group = num_groups - 1;
                const float dt_h = dt[((size_t)bs * (size_t)seq_len + (size_t)t) * (size_t)num_heads + (size_t)h];
                const float d_a = expf(dt_h * a[h]);
                const float d_h = d[h];
                const float *packed_row = x + ((size_t)bs * (size_t)seq_len + (size_t)t) * packed_stride;
                const float *b_row = packed_xbc
                    ? (packed_row + inner_dim + (size_t)group * (size_t)state_dim)
                    : (b + (((size_t)bs * (size_t)seq_len + (size_t)t) * (size_t)num_groups + (size_t)group) * (size_t)state_dim);
                const float *c_row = packed_xbc
                    ? (packed_row + inner_dim + bc_dim + (size_t)group * (size_t)state_dim)
                    : (c + (((size_t)bs * (size_t)seq_len + (size_t)t) * (size_t)num_groups + (size_t)group) * (size_t)state_dim);

                for (int hd = 0; hd < head_dim; ++hd) {
                    const size_t x_idx = (((size_t)bs * (size_t)seq_len + (size_t)t) * (size_t)num_heads + (size_t)h) * (size_t)head_dim + (size_t)hd;
                    const float x_val = packed_xbc ? packed_row[(size_t)h * (size_t)head_dim + (size_t)hd] : x[x_idx];
                    const size_t state_base = ((size_t)h * (size_t)head_dim + (size_t)hd) * (size_t)state_dim;
                    float acc = 0.0f;
                    for (int st = 0; st < state_dim; ++st) {
                        const size_t si = state_base + (size_t)st;
                        const float new_state = state_batch[si] * d_a + dt_h * b_row[st] * x_val;
                        state_batch[si] = new_state;
                        acc += new_state * c_row[st];
                    }
                    y[x_idx] = acc + d_h * x_val;
                }
            }
        }
    }
}

void mamba2_selective_scan_f32(const float *state_init,
                               const float *x,
                               const float *dt,
                               const float *a,
                               const float *b,
                               const float *c,
                               const float *d,
                               float *state_out,
                               float *y,
                               int batch,
                               int seq_len,
                               int num_heads,
                               int head_dim,
                               int state_dim,
                               int num_groups) {
    if (!state_init || !x || !dt || !a || !b || !c || !d || !state_out || !y ||
        batch <= 0 || seq_len <= 0 || num_heads <= 0 || head_dim <= 0 || state_dim <= 0 || num_groups <= 0) {
        return;
    }

    const size_t state_per_batch = (size_t)num_heads * (size_t)head_dim * (size_t)state_dim;
    ck_mamba_debug_finite("scan.state_init", state_init, state_per_batch);
    ck_mamba_debug_finite("scan.x[0]", x, (size_t)num_heads * (size_t)head_dim + 2u * (size_t)num_groups * (size_t)state_dim);
    ck_mamba_debug_finite("scan.dt[0]", dt, (size_t)num_heads);
    ck_mamba_debug_finite("scan.a", a, (size_t)num_heads);
    ck_mamba_debug_finite("scan.d", d, (size_t)num_heads);
    mamba2_selective_scan_f32_head_range(
        state_init, x, dt, a, b, c, d, state_out, y,
        batch, seq_len, num_heads, head_dim, state_dim, num_groups,
        0, num_heads);
    ck_mamba_debug_finite("scan.state_out", state_out, (size_t)batch * state_per_batch);
    ck_mamba_debug_finite("scan.y[0]", y, (size_t)num_heads * (size_t)head_dim);
}

void mamba2_rmsnorm_gate_f32(const float *x,
                             const float *gate,
                             const float *weight,
                             float *out,
                             int rows,
                             int inner_dim,
                             int group_size,
                             float eps) {
    if (!x || !gate || !weight || !out || rows <= 0 || inner_dim <= 0 || group_size <= 0) {
        return;
    }

    ck_mamba_debug_finite("rmsgate.x[0]", x, (size_t)inner_dim);
    ck_mamba_debug_finite("rmsgate.gate[0]", gate, (size_t)inner_dim);
    ck_mamba_debug_finite("rmsgate.weight", weight, (size_t)inner_dim);
    for (int row = 0; row < rows; ++row) {
        const float *x_row = x + (size_t)row * (size_t)inner_dim;
        const float *gate_row = gate + (size_t)row * (size_t)inner_dim;
        float *out_row = out + (size_t)row * (size_t)inner_dim;

        for (int start = 0; start < inner_dim; start += group_size) {
            int end = start + group_size;
            if (end > inner_dim) {
                end = inner_dim;
            }
            const int count = end - start;
            float ms = 0.0f;
            for (int col = start; col < end; ++col) {
                const float gated = x_row[col] * mamba2_silu_f32(gate_row[col]);
                ms += gated * gated;
            }
            const float inv_rms = 1.0f / sqrtf(ms / (float)count + eps);
            for (int col = start; col < end; ++col) {
                const float gated = x_row[col] * mamba2_silu_f32(gate_row[col]);
                out_row[col] = gated * inv_rms * weight[col];
            }
        }
    }
    ck_mamba_debug_finite("rmsgate.out[0]", out, (size_t)inner_dim);
}
