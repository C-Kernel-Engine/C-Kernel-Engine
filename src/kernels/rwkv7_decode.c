/* RWKV-7 T=1 FP32 decode reference. See include/ckernel_rwkv7_decode.h. */
#include "ckernel_rwkv7_decode.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

/* Compare byte ranges without undefined relational comparisons of pointers. */
static int overlaps(const void *a, size_t an, const void *b, size_t bn) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    if (an > UINTPTR_MAX - x || bn > UINTPTR_MAX - y) return 1;
    return x < y + bn && y < x + an;
}

int ck_rwkv7_token_shift_lerp(const float *x,
                                  const float *shift_in,
                                  const float *mu,
                                  float *xm,
                                  float *shift_out,
                                  int dim) {
    if (!x || !shift_in || !mu || !xm || !shift_out || dim <= 0) {
        return -1;
    }
    size_t bytes = (size_t)dim * sizeof(float);
    if (overlaps(xm, bytes, shift_out, bytes)) return -1;
    const float *inputs[] = {x, shift_in, mu};
    for (int j = 0; j < 3; ++j) {
        if ((xm != inputs[j] && overlaps(xm, bytes, inputs[j], bytes)) ||
            (shift_out != inputs[j] && overlaps(shift_out, bytes, inputs[j], bytes))) return -1;
    }
    for (int i = 0; i < dim; ++i) {
        const float previous = shift_in[i];
        const float current = x[i];
        const float m = mu[i];
        xm[i] = current + m * (previous - current);
        shift_out[i] = current;
    }
    return 0;
}

int ck_rwkv7_time_mix_shifts(const float *x,
                             const float *shift_in,
                             const float *mu_r,
                             const float *mu_w,
                             const float *mu_k,
                             const float *mu_v,
                             const float *mu_a,
                             const float *mu_g,
                             float *xr,
                             float *xw,
                             float *xk,
                             float *xv,
                             float *xa,
                             float *xg,
                             float *shift_out,
                             int dim) {
    if (!x || !shift_in || !mu_r || !mu_w || !mu_k || !mu_v || !mu_a ||
        !mu_g || !xr || !xw || !xk || !xv || !xa || !xg || !shift_out ||
        dim <= 0) {
        return -1;
    }
    const size_t bytes = (size_t)dim * sizeof(float);
    float *outs[] = {xr, xw, xk, xv, xa, xg, shift_out};
    for (int a = 0; a < 7; ++a) {
        for (int b = a + 1; b < 7; ++b) {
            if (overlaps(outs[a], bytes, outs[b], bytes)) return -1;
        }
    }
    const float *inputs[] = {x, shift_in, mu_r, mu_w, mu_k, mu_v, mu_a, mu_g};
    for (int j = 0; j < 8; ++j) {
        for (int o = 0; o < 7; ++o) {
            if (outs[o] == inputs[j] &&
                !(outs[o] == shift_out && inputs[j] == shift_in)) {
                return -1;
            }
            if (outs[o] != inputs[j] &&
                overlaps(outs[o], bytes, inputs[j], bytes)) {
                return -1;
            }
        }
    }
    /* Fused per-index pass: each lane reads x[i]/shift_in[i] once, so the
     * shared shift is written exactly once instead of six times. */
    for (int i = 0; i < dim; ++i) {
        const float current = x[i];
        const float previous = shift_in[i];
        const float d = previous - current;
        xr[i] = current + mu_r[i] * d;
        xw[i] = current + mu_w[i] * d;
        xk[i] = current + mu_k[i] * d;
        xv[i] = current + mu_v[i] * d;
        xa[i] = current + mu_a[i] * d;
        xg[i] = current + mu_g[i] * d;
        shift_out[i] = current;
    }
    return 0;
}

int ck_rwkv7_norm_kk(const float *kk_in,
                         float *kk_out,
                         int num_heads,
                         int head_dim) {
    if (!kk_in || !kk_out || num_heads <= 0 || head_dim <= 0 ||
        num_heads > 256 || head_dim > 512) {
        return -1;
    }
    for (int h = 0; h < num_heads; ++h) {
        const float *src = kk_in + (size_t)h * (size_t)head_dim;
        float *dst = kk_out + (size_t)h * (size_t)head_dim;
        float norm = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            norm += src[i] * src[i];
        }
        norm = sqrtf(norm);
        /* Divisor max(norm,1e-12); an all-zero row stays zero. */
        if (norm < 1e-12f) {
            norm = 1e-12f;
        }
        const float inv = 1.0f / norm;
        for (int i = 0; i < head_dim; ++i) {
            dst[i] = src[i] * inv;
        }
    }
    return 0;
}

int ck_rwkv7_scale_k(const float *k_in,
                         const float *a,
                         const float *k_a,
                         float *k_out,
                         int dim) {
    if (!k_in || !a || !k_a || !k_out || dim <= 0) {
        return -1;
    }
    for (int i = 0; i < dim; ++i) {
        k_out[i] = k_in[i] * (1.0f + (k_a[i] - 1.0f) * a[i]);
    }
    return 0;
}

int ck_rwkv7_apply_kk_factor(const float *k,
                             const float *k_k,
                             float *kk_raw,
                             int dim) {
    if (!k || !k_k || !kk_raw || dim <= 0) {
        return -1;
    }
    for (int i = 0; i < dim; ++i) {
        kk_raw[i] = k[i] * k_k[i];
    }
    return 0;
}

static inline float ck_rwkv7_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

int ck_rwkv7_lora_gates(const float *xw,
                            const float *xa,
                            const float *xg,
                            const float *w1,
                            const float *w2,
                            const float *w0,
                            const float *a1,
                            const float *a2,
                            const float *a0,
                            const float *g1,
                            const float *g2,
                            int dim,
                            int rank_w,
                            int rank_a,
                            int rank_g,
                            float *w_log,
                            float *a_out,
                            float *g_out) {
    if (!xw || !xa || !xg || !w1 || !w2 || !w0 || !a1 || !a2 || !a0 || !g1 ||
        !g2 || !w_log || !a_out || !g_out || dim <= 0 || rank_w <= 0 ||
        rank_a <= 0 || rank_g <= 0 || rank_w > 512 || rank_a > 512 ||
        rank_g > 512) {
        return -1;
    }
    /* w: tanh between factors, sigmoid after (+w0), scaled by -e^-0.5. */
    {
        float tmp[512];
        for (int r = 0; r < rank_w; ++r) {
            float acc = 0.0f;
            for (int i = 0; i < dim; ++i) {
                acc += xw[i] * w1[(size_t)i * (size_t)rank_w + (size_t)r];
            }
            tmp[r] = tanhf(acc);
        }
        for (int i = 0; i < dim; ++i) {
            float acc = w0[i];
            for (int r = 0; r < rank_w; ++r) {
                acc += tmp[r] * w2[(size_t)r * (size_t)dim + (size_t)i];
            }
            w_log[i] = -0.6065306597126334f * ck_rwkv7_sigmoid(acc);
        }
    }
    /* a: no activation between, sigmoid after (+a0). */
    {
        float tmp[512];
        for (int r = 0; r < rank_a; ++r) {
            float acc = 0.0f;
            for (int i = 0; i < dim; ++i) {
                acc += xa[i] * a1[(size_t)i * (size_t)rank_a + (size_t)r];
            }
            tmp[r] = acc;
        }
        for (int i = 0; i < dim; ++i) {
            float acc = a0[i];
            for (int r = 0; r < rank_a; ++r) {
                acc += tmp[r] * a2[(size_t)r * (size_t)dim + (size_t)i];
            }
            a_out[i] = ck_rwkv7_sigmoid(acc);
        }
    }
    /* g: sigmoid between factors, nothing after, no bias. */
    {
        float tmp[512];
        for (int r = 0; r < rank_g; ++r) {
            float acc = 0.0f;
            for (int i = 0; i < dim; ++i) {
                acc += xg[i] * g1[(size_t)i * (size_t)rank_g + (size_t)r];
            }
            tmp[r] = ck_rwkv7_sigmoid(acc);
        }
        for (int i = 0; i < dim; ++i) {
            float acc = 0.0f;
            for (int r = 0; r < rank_g; ++r) {
                acc += tmp[r] * g2[(size_t)r * (size_t)dim + (size_t)i];
            }
            g_out[i] = acc;
        }
    }
    return 0;
}

int ck_rwkv7_vgate_vmix(const float *v_in,
                            const float *v_first,
                            const float *xv,
                            const float *v1,
                            const float *v2,
                            const float *v0,
                            int dim,
                            int rank_v,
                            float *v_out,
                            float *v_first_out,
                            float *v_gate) {
    if (!v_in || !v_out || !v_first_out || dim <= 0) {
        return -1;
    }
    if (xv == NULL) {
        /* Layer 0: produce v_first, no gate read. */
        for (int i = 0; i < dim; ++i) {
            v_out[i] = v_in[i];
            v_first_out[i] = v_in[i];
        }
        return 0;
    }
    if (!v_first || !v1 || !v2 || !v0 || rank_v <= 0 || rank_v > 512) {
        return -1;
    }
    float tmp[512];
    for (int r = 0; r < rank_v; ++r) {
        float acc = 0.0f;
        for (int i = 0; i < dim; ++i) {
            acc += xv[i] * v1[(size_t)i * (size_t)rank_v + (size_t)r];
        }
        tmp[r] = acc;
    }
    for (int i = 0; i < dim; ++i) {
        float acc = v0[i];
        for (int r = 0; r < rank_v; ++r) {
            acc += tmp[r] * v2[(size_t)r * (size_t)dim + (size_t)i];
        }
        const float gate = ck_rwkv7_sigmoid(acc);
        if (v_gate) {
            v_gate[i] = gate;
        }
        v_out[i] = v_in[i] + (v_first[i] - v_in[i]) * gate;
        v_first_out[i] = v_first[i];
    }
    return 0;
}

int ck_rwkv7_state_step(const float *w_log,
                        const float *k,
                        const float *v,
                        const float *kk,
                        const float *a,
                        const float *state_in,
                        float *state_out,
                        int head_dim) {
    if (!w_log || !k || !v || !kk || !a || !state_in || !state_out ||
        head_dim <= 0 || head_dim > 512) {
        return -1;
    }
    float decay[512];
    float bvec[512];
    for (int j = 0; j < head_dim; ++j) {
        decay[j] = expf(w_log[j]);
        bvec[j] = kk[j] * a[j];
    }
    for (int i = 0; i < head_dim; ++i) {
        float t = 0.0f;
        for (int l = 0; l < head_dim; ++l) {
            t += state_in[(size_t)i * (size_t)head_dim + (size_t)l] * kk[l];
        }
        for (int j = 0; j < head_dim; ++j) {
            state_out[(size_t)i * (size_t)head_dim + (size_t)j] =
                state_in[(size_t)i * (size_t)head_dim + (size_t)j] * decay[j] -
                t * bvec[j] + v[i] * k[j];
        }
    }
    return 0;
}

int ck_rwkv7_state_readout(const float *state,
                           const float *r,
                           float *y,
                           int head_dim) {
    if (!state || !r || !y || head_dim <= 0 || head_dim > 512) {
        return -1;
    }
    for (int i = 0; i < head_dim; ++i) {
        float acc = 0.0f;
        for (int l = 0; l < head_dim; ++l) {
            acc += state[(size_t)i * (size_t)head_dim + (size_t)l] * r[l];
        }
        y[i] = acc;
    }
    return 0;
}

int ck_rwkv7_wkv_decode(const float *r,
                            const float *w_log,
                            const float *k,
                            const float *v,
                            const float *kk,
                            const float *a,
                            const float *g,
                            const float *rk,
                            const float *ln_w,
                            const float *ln_b,
                            float norm_eps,
                            const float *state_in,
                            float *state_out,
                            float *y_out,
                            int num_heads,
                            int head_dim) {
    if (!r || !w_log || !k || !v || !kk || !a || !g || !rk || !ln_w || !ln_b ||
        !state_in || !state_out || !y_out || num_heads <= 0 || head_dim <= 0 ||
        !(norm_eps > 0.0f) || !isfinite(norm_eps) ||
        !isfinite(norm_eps * (float)head_dim)) {
        return -1;
    }
    if (num_heads > 256 || head_dim > 512) {
        return -1;
    }

    /* Per-head generalised delta-rule step + raw readout (pre-norm). */
    for (int h = 0; h < num_heads; ++h) {
        const float *rh = r + (size_t)h * (size_t)head_dim;
        const float *kh = k + (size_t)h * (size_t)head_dim;
        const float *vh = v + (size_t)h * (size_t)head_dim;
        const float *wh = w_log + (size_t)h * (size_t)head_dim;
        const float *khh = kk + (size_t)h * (size_t)head_dim;
        const float *ah = a + (size_t)h * (size_t)head_dim;
        const float *sin = state_in + (size_t)h * (size_t)head_dim * (size_t)head_dim;
        float *sout = state_out + (size_t)h * (size_t)head_dim * (size_t)head_dim;
        float *yh = y_out + (size_t)h * (size_t)head_dim;

        if (ck_rwkv7_state_step(wh, kh, vh, khh, ah, sin, sout, head_dim) != 0 ||
            ck_rwkv7_state_readout(sout, rh, yh, head_dim) != 0) {
            return -1;
        }
    }

    /* GroupNorm over C with H groups, eps = norm_eps*head_dim. */
    const float eps = norm_eps * (float)head_dim;
    for (int h = 0; h < num_heads; ++h) {
        float *yh = y_out + (size_t)h * (size_t)head_dim;
        const float *lwh = ln_w + (size_t)h * (size_t)head_dim;
        const float *lbh = ln_b + (size_t)h * (size_t)head_dim;
        float mean = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            mean += yh[i];
        }
        mean /= (float)head_dim;
        float var = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            const float d = yh[i] - mean;
            var += d * d;
        }
        var /= (float)head_dim;
        const float inv = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < head_dim; ++i) {
            yh[i] = (yh[i] - mean) * inv * lwh[i] + lbh[i];
        }
    }

    /* Per-head bonus s_h = sum(r_h*k_h*rk_h); y += s_h*v; then y *= g. */
    const int dim = num_heads * head_dim;
    for (int h = 0; h < num_heads; ++h) {
        const float *rh = r + (size_t)h * (size_t)head_dim;
        const float *kh = k + (size_t)h * (size_t)head_dim;
        const float *vh = v + (size_t)h * (size_t)head_dim;
        const float *rkh = rk + (size_t)h * (size_t)head_dim;
        float *yh = y_out + (size_t)h * (size_t)head_dim;
        float s = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            s += rh[i] * kh[i] * rkh[i];
        }
        for (int i = 0; i < head_dim; ++i) {
            yh[i] += s * vh[i];
        }
    }
    for (int i = 0; i < dim; ++i) {
        y_out[i] *= g[i];
    }
    return 0;
}

size_t ck_rwkv7_channelmix_scratch_elems(int dim, int hidden_dim) {
    if (dim <= 0 || hidden_dim <= 0 || hidden_dim > 8192) {
        return 0;
    }
    return (size_t)dim + (size_t)hidden_dim;
}

int ck_rwkv7_channelmix_decode(const float *x,
                               const float *shift_in,
                               const float *mu,
                               const float *Wk,
                               const float *Wv,
                               float *shift_out,
                               float *out,
                               int dim,
                               int hidden_dim,
                               float *scratch,
                               size_t scratch_elems) {
    if (!x || !shift_in || !mu || !Wk || !Wv || !shift_out || !out || dim <= 0 ||
        hidden_dim <= 0 || hidden_dim > 8192) {
        return -1;
    }
    if (!scratch ||
        scratch_elems < (size_t)dim + (size_t)hidden_dim ||
        scratch == (const float *)x || scratch == (const float *)shift_in ||
        scratch == (const float *)mu || scratch == (const float *)Wk ||
        scratch == (const float *)Wv || scratch == (const float *)shift_out ||
        scratch == (const float *)out) {
        return -1;
    }
    const size_t bytes = (size_t)dim * sizeof(float);
    if ((size_t)hidden_dim > SIZE_MAX / bytes) return -1;
    const size_t weight_bytes = bytes * (size_t)hidden_dim;
    const size_t scratch_bytes = ((size_t)dim + (size_t)hidden_dim) * sizeof(float);
    if (overlaps(shift_out, bytes, out, bytes)) return -1;
    const float *inputs[] = {x, shift_in, mu, Wk, Wv};
    const size_t sizes[] = {bytes, bytes, bytes, weight_bytes, weight_bytes};
    for (int j = 0; j < 5; ++j) {
        if (overlaps(scratch, scratch_bytes, inputs[j], sizes[j])) return -1;
        if (!(j == 0 && shift_out == x) &&
            !(j == 1 && shift_out == shift_in) &&
            overlaps(shift_out, bytes, inputs[j], sizes[j])) return -1;
        if (!(j == 0 && out == x) &&
            overlaps(out, bytes, inputs[j], sizes[j])) return -1;
    }
    if (overlaps(scratch, scratch_bytes, out, bytes) ||
        overlaps(scratch, scratch_bytes, shift_out, bytes)) return -1;
    /* Hoisted token-shift mix; per-index read-before-write keeps
     * shift_out == shift_in (the circuit's in-place carry) and
     * shift_out == x aliasing safe. */
    float *xk = scratch;
    float *hidden = scratch + dim;
    for (int i = 0; i < dim; ++i) {
        xk[i] = x[i] + mu[i] * (shift_in[i] - x[i]);
        shift_out[i] = x[i];
    }
    /* hidden[j] = relu(Wk[j,:] @ xk)^2, Wk [K,C] row-major. */
    for (int j = 0; j < hidden_dim; ++j) {
        const float *row = Wk + (size_t)j * (size_t)dim;
        float acc = 0.0f;
        for (int i = 0; i < dim; ++i) {
            acc += row[i] * xk[i];
        }
        const float rel = acc > 0.0f ? acc : 0.0f;
        hidden[j] = rel * rel;
    }
    /* out = Wv @ hidden, Wv [C,K] row-major; runs after the last xk read,
     * so out == x aliasing is safe. */
    for (int i = 0; i < dim; ++i) {
        const float *row = Wv + (size_t)i * (size_t)hidden_dim;
        float acc = 0.0f;
        for (int j = 0; j < hidden_dim; ++j) {
            acc += row[j] * hidden[j];
        }
        out[i] = acc;
    }
    return 0;
}

size_t ck_rwkv7_time_mix_scratch_elems(int dim) {
    if (dim <= 0 || (size_t)dim > SIZE_MAX / (7u * sizeof(float))) {
        return 0;
    }
    return (size_t)dim * 7u;
}int ck_rwkv7_time_mix_tail(const float *r,
                           const float *k_raw,
                           const float *v_in,
                           const float *v_first_in,
                           const float *xw,
                           const float *xa,
                           const float *xg,
                           const float *xv,
                           const float *w1,
                           const float *w2,
                           const float *w0,
                           const float *a1,
                           const float *a2,
                           const float *a0,
                           const float *g1,
                           const float *g2,
                           const float *v1,
                           const float *v2,
                           const float *v0,
                           const float *k_k,
                           const float *k_a,
                           const float *rk,
                           const float *ln_w,
                           const float *ln_b,
                           float norm_eps,
                           const float *state_in,
                           float *state_out,
                           float *y_preproj,
                           float *v_first_out,
                           int dim,
                           int num_heads,
                           int head_dim,
                           int rank_w,
                           int rank_a,
                           int rank_g,
                           int rank_v,
                           float *scratch,
                           size_t scratch_elems) {
    const int layer0 = (xv == NULL);
    if (!r || !k_raw || !v_in || !xw || !xa || !xg || !w1 || !w2 || !w0 ||
        !a1 || !a2 || !a0 || !g1 || !g2 || !k_k || !k_a || !rk || !ln_w ||
        !ln_b || !state_in || !state_out || !y_preproj || !v_first_out ||
        !scratch || dim <= 0 || num_heads <= 0 || head_dim <= 0 ||
        num_heads > 256 || head_dim > 512 || rank_w <= 0 || rank_w > 512 ||
        rank_a <= 0 || rank_a > 512 || rank_g <= 0 || rank_g > 512 ||
        !(norm_eps > 0.0f) || !isfinite(norm_eps) ||
        !isfinite(norm_eps * (float)head_dim)) {
        return -1;
    }
    if ((size_t)dim != (size_t)num_heads * (size_t)head_dim) {
        return -1;
    }
    if (layer0) {
        /* Layer 0 produces v_first: no residual v state exists yet, so the
         * residual inputs must be NULL (produce-only path never reads them). */
        if (v_first_in || v1 || v2 || v0) {
            return -1;
        }
    } else {
        if (!v_first_in || !v1 || !v2 || !v0 || rank_v <= 0 || rank_v > 512) {
            return -1;
        }
    }
    if (scratch_elems < (size_t)dim * 7u) {
        return -1;
    }
    const size_t vec_bytes = (size_t)dim * sizeof(float);
    if (vec_bytes / sizeof(float) != (size_t)dim) return -1;
    if ((size_t)head_dim > SIZE_MAX / vec_bytes) return -1;
    const size_t state_bytes = vec_bytes * (size_t)head_dim;
    const size_t scratch_bytes = vec_bytes * 7u;
    /* Scratch must be disjoint from every other argument. */
    const void *ptrs[] = {r, k_raw, v_in, v_first_in, xw, xa, xg, xv,
                          w1, w2, w0, a1, a2, a0, g1, g2, v1, v2, v0,
                          k_k, k_a, rk, ln_w, ln_b, state_in, state_out,
                          y_preproj, v_first_out};
    /* LoRA factor sizes: down [C,R], up [R,C], bias [C]. State is [H,N,N]. */
    size_t lens[] = {(size_t)dim, (size_t)dim, (size_t)dim, (size_t)dim,
                     (size_t)dim, (size_t)dim, (size_t)dim, (size_t)dim,
                     (size_t)dim * (size_t)rank_w, (size_t)rank_w * (size_t)dim, (size_t)dim,
                     (size_t)dim * (size_t)rank_a, (size_t)rank_a * (size_t)dim, (size_t)dim,
                     (size_t)dim * (size_t)rank_g, (size_t)rank_g * (size_t)dim,
                     0, 0, 0,
                     (size_t)dim, (size_t)dim,
                     (size_t)num_heads * (size_t)head_dim, (size_t)dim, (size_t)dim,
                     state_bytes / sizeof(float), state_bytes / sizeof(float),
                     (size_t)dim, (size_t)dim};
    if (!layer0) {
        lens[16] = (size_t)dim * (size_t)rank_v;
        lens[17] = (size_t)rank_v * (size_t)dim;
        lens[18] = (size_t)dim;
    }
    for (int j = 0; j < 28; ++j) {
        if (!ptrs[j] || lens[j] == 0) continue;
        if (lens[j] > SIZE_MAX / sizeof(float)) return -1;
        if (overlaps(scratch, scratch_bytes, ptrs[j], lens[j] * sizeof(float))) return -1;
    }
    /* Outputs disjoint from each other; state_out may == state_in,
     * v_first_out may == v_first_in. All other output-input overlaps fail. */
    if (overlaps(state_out, state_bytes, y_preproj, vec_bytes) ||
        overlaps(state_out, state_bytes, v_first_out, vec_bytes) ||
        overlaps(y_preproj, vec_bytes, v_first_out, vec_bytes)) {
        return -1;
    }
    const void *ro_inputs[] = {r, k_raw, v_in, xw, xa, xg, xv, w1, w2, w0,
                               a1, a2, a0, g1, g2, v1, v2, v0, k_k, k_a,
                               rk, ln_w, ln_b};
    size_t ro_lens[] = {(size_t)dim, (size_t)dim, (size_t)dim, (size_t)dim,
                        (size_t)dim, (size_t)dim, (size_t)dim,
                        (size_t)dim * (size_t)rank_w, (size_t)rank_w * (size_t)dim, (size_t)dim,
                        (size_t)dim * (size_t)rank_a, (size_t)rank_a * (size_t)dim, (size_t)dim,
                        (size_t)dim * (size_t)rank_g, (size_t)rank_g * (size_t)dim,
                        0, 0, 0,
                        (size_t)dim, (size_t)dim, (size_t)dim, (size_t)dim, (size_t)dim};
    if (!layer0) {
        ro_lens[15] = (size_t)dim * (size_t)rank_v;
        ro_lens[16] = (size_t)rank_v * (size_t)dim;
        ro_lens[17] = (size_t)dim;
    }
    for (int j = 0; j < 23; ++j) {
        if (!ro_inputs[j] || ro_lens[j] == 0) continue;
        size_t n = ro_lens[j];
        if (overlaps(y_preproj, vec_bytes, ro_inputs[j], n * sizeof(float))) return -1;
        if (overlaps(v_first_out, vec_bytes, ro_inputs[j], n * sizeof(float))) return -1;
        /* state_out has no self-match in ro_inputs (state_in is not listed),
         * so the check always applies; exact state_out==state_in aliasing is
         * handled by the dedicated state_in comparison below. */
        if (overlaps(state_out, state_bytes, ro_inputs[j], n * sizeof(float))) return -1;
    }
    /* y/v_first outputs must also be disjoint from the state and v_first
     * input streams (ro_inputs above covers r..ln_b only). y has no allowed
     * alias; v_first_out may exactly alias v_first_in (checked below). */
    if (overlaps(y_preproj, vec_bytes, state_in, state_bytes)) return -1;
    if (v_first_in && overlaps(y_preproj, vec_bytes, v_first_in, vec_bytes)) {
        return -1;
    }
    if (overlaps(v_first_out, vec_bytes, state_in, state_bytes)) return -1;
    if (v_first_in && overlaps(state_out, state_bytes, v_first_in, vec_bytes)) {
        return -1;
    }
    if (v_first_in && v_first_out != v_first_in &&
        overlaps(v_first_out, vec_bytes, v_first_in, vec_bytes)) {
        return -1;
    }
    if (state_out != state_in &&
        overlaps(state_out, state_bytes, state_in, state_bytes)) {
        return -1;
    }

    float *w_log = scratch;
    float *a_buf = scratch + (size_t)dim;
    float *g_buf = scratch + (size_t)dim * 2u;
    float *v_buf = scratch + (size_t)dim * 3u;
    float *kk_raw = scratch + (size_t)dim * 4u;
    float *kk = scratch + (size_t)dim * 5u;
    float *k_sc = scratch + (size_t)dim * 6u;

    if (ck_rwkv7_lora_gates(xw, xa, xg, w1, w2, w0, a1, a2, a0, g1, g2,
                            dim, rank_w, rank_a, rank_g,
                            w_log, a_buf, g_buf) != 0) {
        return -1;
    }
    if (layer0) {
        if (ck_rwkv7_vgate_vmix(v_in, NULL, NULL, NULL, NULL, NULL,
                                dim, 1, v_buf, v_first_out, NULL) != 0) {
            return -1;
        }
    } else {
        if (ck_rwkv7_vgate_vmix(v_in, v_first_in, xv, v1, v2, v0,
                                dim, rank_v, v_buf, v_first_out, NULL) != 0) {
            return -1;
        }
    }
    if (ck_rwkv7_apply_kk_factor(k_raw, k_k, kk_raw, dim) != 0) {
        return -1;
    }
    if (ck_rwkv7_norm_kk(kk_raw, kk, num_heads, head_dim) != 0) {
        return -1;
    }
    if (ck_rwkv7_scale_k(k_raw, a_buf, k_a, k_sc, dim) != 0) {
        return -1;
    }
    if (ck_rwkv7_wkv_decode(r, w_log, k_sc, v_buf, kk, a_buf, g_buf, rk,
                            ln_w, ln_b, norm_eps, state_in, state_out,
                            y_preproj, num_heads, head_dim) != 0) {
        return -1;
    }
    return 0;
}

size_t ck_rwkv7_time_mix_block_scratch_elems(int dim) {
    if (dim <= 0 || (size_t)dim > SIZE_MAX / (19u * sizeof(float))) {
        return 0;
    }
    return (size_t)dim * 19u;
}

int ck_rwkv7_time_mix_block(const float *h,
                            const float *att_shift_in,
                            const float *v_first_in,
                            const float *state_in,
                            const float *ln_g,
                            const float *ln_b,
                            const float *mu_r,
                            const float *mu_w,
                            const float *mu_k,
                            const float *mu_v,
                            const float *mu_a,
                            const float *mu_g,
                            const float *Wr,
                            const float *Wk,
                            const float *Wv,
                            const float *Wo,
                            const float *w1,
                            const float *w2,
                            const float *w0,
                            const float *a1,
                            const float *a2,
                            const float *a0,
                            const float *g1,
                            const float *g2,
                            const float *v1,
                            const float *v2,
                            const float *v0,
                            const float *k_k,
                            const float *k_a,
                            const float *rk,
                            const float *gn_w,
                            const float *gn_b,
                            float norm_eps,
                            int dim,
                            int num_heads,
                            int head_dim,
                            int rank_w,
                            int rank_a,
                            int rank_g,
                            int rank_v,
                            int is_layer_zero,
                            ck_rwkv7_gemv_fn gemv,
                            ck_rwkv7_layer_norm_fn layer_norm,
                            void *ctx,
                            float *h_out,
                            float *att_shift_out,
                            float *state_out,
                            float *v_first_out,
                            float *scratch,
                            size_t scratch_elems) {
    const int layer0 = is_layer_zero ? 1 : 0;
    if (!h || !att_shift_in || !ln_g || !ln_b || !mu_r || !mu_w || !mu_k ||
        !mu_v || !mu_a || !mu_g || !Wr || !Wk || !Wv || !Wo || !w1 || !w2 ||
        !w0 || !a1 || !a2 || !a0 || !g1 || !g2 || !k_k || !k_a || !rk ||
        !gn_w || !gn_b || !state_in || !state_out || !h_out ||
        !att_shift_out || !v_first_out || !gemv || !layer_norm || !scratch ||
        dim <= 0 || num_heads <= 0 || head_dim <= 0 || num_heads > 256 ||
        head_dim > 512 || rank_w <= 0 || rank_w > 512 || rank_a <= 0 ||
        rank_a > 512 || rank_g <= 0 || rank_g > 512 ||
        !(norm_eps > 0.0f) || !isfinite(norm_eps) ||
        !isfinite(norm_eps * (float)head_dim) ||
        (is_layer_zero != 0 && is_layer_zero != 1)) {
        return -1;
    }
    if ((size_t)dim != (size_t)num_heads * (size_t)head_dim) {
        return -1;
    }
    if (layer0) {
        if (v_first_in || v1 || v2 || v0) return -1;
    } else if (!v_first_in || !v1 || !v2 || !v0 || rank_v <= 0 ||
               rank_v > 512) {
        return -1;
    }
    if (scratch_elems < (size_t)dim * 19u) {
        return -1;
    }
    const size_t vec_bytes = (size_t)dim * sizeof(float);
    if ((size_t)head_dim > SIZE_MAX / vec_bytes) return -1;
    const size_t state_bytes = vec_bytes * (size_t)head_dim;
    const size_t scratch_bytes = vec_bytes * 19u;

    /* Output aliasing: each output may exactly alias its own input stream;
     * the four outputs must otherwise be pairwise disjoint and disjoint
     * from every input. */
    float *outs[] = {h_out, att_shift_out, state_out, v_first_out};
    const void *allow[] = {h, att_shift_in, state_in, v_first_in};
    const size_t out_bytes[] = {vec_bytes, vec_bytes, state_bytes, vec_bytes};
    for (int a = 0; a < 4; ++a) {
        for (int b = a + 1; b < 4; ++b) {
            if (overlaps(outs[a], out_bytes[a], outs[b], out_bytes[b])) return -1;
        }
    }
    const void *ro[] = {h, att_shift_in, v_first_in, state_in, ln_g, ln_b,
                        mu_r, mu_w, mu_k, mu_v, mu_a, mu_g,
                        Wr, Wk, Wv, Wo,
                        w1, w2, w0, a1, a2, a0, g1, g2, v1, v2, v0,
                        k_k, k_a, rk, gn_w, gn_b};
    size_t ro_e[] = {(size_t)dim, (size_t)dim, (size_t)dim, 0,
                     (size_t)dim, (size_t)dim,
                     (size_t)dim, (size_t)dim, (size_t)dim, (size_t)dim,
                     (size_t)dim, (size_t)dim,
                     (size_t)dim * (size_t)dim, (size_t)dim * (size_t)dim,
                     (size_t)dim * (size_t)dim, (size_t)dim * (size_t)dim,
                     (size_t)dim * (size_t)rank_w, (size_t)rank_w * (size_t)dim, (size_t)dim,
                     (size_t)dim * (size_t)rank_a, (size_t)rank_a * (size_t)dim, (size_t)dim,
                     (size_t)dim * (size_t)rank_g, (size_t)rank_g * (size_t)dim,
                     0, 0, 0,
                     (size_t)dim, (size_t)dim, (size_t)dim, (size_t)dim, (size_t)dim};
    ro_e[3] = state_bytes / sizeof(float);
    if (!layer0) {
        ro_e[24] = (size_t)dim * (size_t)rank_v;
        ro_e[25] = (size_t)rank_v * (size_t)dim;
        ro_e[26] = (size_t)dim;
    }
    for (int j = 0; j < 32; ++j) {
        if (!ro[j] || ro_e[j] == 0) continue;
        if (ro_e[j] > SIZE_MAX / sizeof(float)) return -1;
        const size_t nbytes = ro_e[j] * sizeof(float);
        if (overlaps(scratch, scratch_bytes, ro[j], nbytes)) return -1;
        for (int o = 0; o < 4; ++o) {
            if (outs[o] == ro[j] && ro[j] == allow[o]) continue;
            if (overlaps(outs[o], out_bytes[o], ro[j], nbytes)) return -1;
        }
    }
    for (int o = 0; o < 4; ++o) {
        if (overlaps(scratch, scratch_bytes, outs[o], out_bytes[o])) return -1;
    }

    float *attn_in = scratch;
    float *xr = scratch + (size_t)dim;
    float *xw = scratch + (size_t)dim * 2u;
    float *xk = scratch + (size_t)dim * 3u;
    float *xvm = scratch + (size_t)dim * 4u;
    float *xa = scratch + (size_t)dim * 5u;
    float *xg = scratch + (size_t)dim * 6u;
    float *r = scratch + (size_t)dim * 7u;
    float *k_raw = scratch + (size_t)dim * 8u;
    float *v_in = scratch + (size_t)dim * 9u;
    float *y_pre = scratch + (size_t)dim * 10u;
    float *attn_out = scratch + (size_t)dim * 11u;
    float *tail_scratch = scratch + (size_t)dim * 12u;

    if (layer_norm(h, ln_g, ln_b, attn_in, dim, ctx) != 0) {
        return -1;
    }
    if (ck_rwkv7_time_mix_shifts(attn_in, att_shift_in, mu_r, mu_w, mu_k,
                                 mu_v, mu_a, mu_g, xr, xw, xk, xvm, xa,
                                 xg, att_shift_out, dim) != 0) {
        return -1;
    }
    if (gemv(Wr, xr, r, ctx) != 0 || gemv(Wk, xk, k_raw, ctx) != 0 ||
        gemv(Wv, xvm, v_in, ctx) != 0) {
        return -1;
    }
    if (ck_rwkv7_time_mix_tail(r, k_raw, v_in, v_first_in, xw, xa, xg,
                               layer0 ? NULL : xvm, w1, w2, w0, a1, a2, a0,
                               g1, g2, layer0 ? NULL : v1,
                               layer0 ? NULL : v2, layer0 ? NULL : v0, k_k,
                               k_a, rk, gn_w, gn_b, norm_eps, state_in,
                               state_out, y_pre, v_first_out, dim,
                               num_heads, head_dim, rank_w, rank_a, rank_g,
                               layer0 ? 1 : rank_v, tail_scratch,
                               (size_t)dim * 7u) != 0) {
        return -1;
    }
    if (gemv(Wo, y_pre, attn_out, ctx) != 0) {
        return -1;
    }
    for (int i = 0; i < dim; ++i) {
        h_out[i] = h[i] + attn_out[i];
    }
    return 0;
}
