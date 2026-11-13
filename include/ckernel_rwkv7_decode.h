#ifndef CKERNEL_RWKV7_DECODE_H
#define CKERNEL_RWKV7_DECODE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * RWKV-7 single-token (T=1, B=1) FP32 decode reference.
 *
 * Token shift: xm = x + mu*(shift-x) with shift_out = x.
 *
 * WKV generalised delta rule, multiplicative decay, per head with state
 * [N,N] in (value,key) axes:
 *   decay[j]     = exp(w_log[j])
 *   b[j]         = kk[j]*a[j]
 *   S_new[i,j]   = S[i,j]*decay[j] - (S[i,:].kk)*b[j] + v[i]*k[j]
 *   y[i]         = sum_l S_new[i,l]*r[l]
 * GroupNorm eps is norm_eps*head_dim. Bonus is per head:
 * s_h = sum(r_h*k_h*rk_h), y_h += s_h*v_h, with rk [H,N].
 * Output is y_out = (groupnorm(y)+bonus)*g elementwise; the final CxC
 * output projection stays with the caller's GEMV.
 * v-mixing towards v_first and the LoRA chains are separate units below;
 * the WKV takes final w_log/k/v/kk/a/g (kk L2-normalised, k scaled by
 * 1+(a-1)*k_a per modeling_rwkv7.py's Lerp(k_a,1) box; this matches the
 * publisher reference used by the shipped checkpoints.
 *
 * Layouts (all FP32, row-major, flat C = H*N):
 *  - vectors: [C]; rk: [H,N] (row per head)
 *  - state:   [H,N,N], head h at offset h*N*N, row i at +i*N+j
 *  - LoRA down factors [C,R], up factors [R,C]; biases [C] (g has none)
 *  - channelmix Wk [K,C] (expand), Wv [C,K] (project)
 *
 * Time-mix tail order (single entry point below, mirrors
 * modeling_rwkv7.py Rwkv7Attention.forward without its GEMVs/LayerNorms):
 *   lora_gates -> vgate_vmix -> kk_raw = k_raw*k_k -> norm_kk ->
 *   scale_k -> wkv_decode. The kk factor multiply is also exposed alone;
 *   the final CxC output projection stays with the caller's GEMV.
 *
 * Limits (violations are hard errors): head_dim <= 512, num_heads <= 256,
 * LoRA ranks <= 512, hidden_dim <= 8192, norm_eps > 0.
 *
 * All functions return 0 on success, -1 on bad args, except
 * ck_rwkv7_channelmix_scratch_elems, which returns the required scratch
 * element count (0 on bad args). No malloc/free.
 * Unless an alias is explicitly documented, outputs must be disjoint from
 * inputs and each other; callers own storage sizes. State-step/state-decode
 * permit state_out == state_in, and norm_kk permits kk_out == kk_in.
 */

int ck_rwkv7_token_shift_lerp(const float *x,
                                  const float *shift_in,
                                  const float *mu,
                                  float *xm,
                                  float *shift_out,
                                  int dim);
/* Each output may exactly alias an input. The two outputs must be disjoint;
 * shifted/partial output-input overlaps are rejected before any write. */

/* Six-way time-mix shift bundle (one entry point for the diagram's bottom
 * Lerp row): xr/xw/xk/xv/xa/xg[i] = x[i] + mu_*[i]*(shift_in[i]-x[i]) with
 * shift_out = x. The fused per-index loop reads x[i]/shift_in[i] once, so a
 * single pass replaces six ck_rwkv7_token_shift_lerp calls and the shared
 * shift is written exactly once. The seven outputs must be pairwise
 * disjoint; shift_out may exactly alias shift_in (in-place carry). All other
 * output-input overlaps (including exact output == x aliasing, which would
 * let sibling mixes collide) are rejected before any write. Read-only inputs
 * may overlap each other. */
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
                             int dim);

/* kk_raw -> per-head L2-normalised kk (divisor max(norm,1e-12)). */
int ck_rwkv7_norm_kk(const float *kk_in,
                         float *kk_out,
                         int num_heads,
                         int head_dim);

/* k_out = k_in * (1 + (a-1) * k_a), elementwise over [C] (diagram Lerp(k_a,1)
 * box with mix weight a. */
int ck_rwkv7_scale_k(const float *k_in,
                         const float *a,
                         const float *k_a,
                         float *k_out,
                         int dim);

/* kk_raw = k .* k_k, elementwise over [C] (reference: kk = k * k_k before
 * per-head L2 normalisation). kk_raw may exactly alias k or k_k. */
int ck_rwkv7_apply_kk_factor(const float *k,
                             const float *k_k,
                             float *kk_raw,
                             int dim);

/* Unfused FP32 LoRA chains (decode, B=1,T=1).
 * Computes w_log = -e^-0.5*sigmoid(tanh(xw@w1)@w2+w0),
 * a = sigmoid((xa@a1)@a2+a0), g = sigmoid(xg@g1)@g2.
 * Fusion is a later perf change behind its own parity gate. */
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
                            float *g_out);

/* Value-residual gate and mix. Layer 0: pass xv/v1/v2/v0 as NULL to produce
 * v_first (v_out = v_first_out = v_in copy). Layer >0: v_gate =
 * sigmoid((xv@v1)@v2+v0), v_out = v_in+(v_first-v_in)*v_gate,
 * v_first_out = v_first copy. v_gate may be NULL if unneeded. */
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
                            float *v_gate);

/* Single-head state transition (the unit a SIMD/chunked variant replaces):
 * S_new[i,j] = S[i,j]*exp(w_log[j]) - (S[i,:].kk)*(kk[j]*a[j]) + v[i]*k[j],
 * state axes (value,key), all vectors [N], states [N,N] row-major. */
int ck_rwkv7_state_step(const float *w_log,
                        const float *k,
                        const float *v,
                        const float *kk,
                        const float *a,
                        const float *state_in,
                        float *state_out,
                        int head_dim);

/* Single-head readout y = S_new.r, state [N,N] row-major, r/y [N]. */
int ck_rwkv7_state_readout(const float *state,
                           const float *r,
                           float *y,
                           int head_dim);

/* Full single-token time-mix step (composition): per head, ck_rwkv7_state_step
 * then ck_rwkv7_state_readout; then GroupNorm over C (eps = norm_eps*head_dim),
 * per-head bonus s_h = sum(r_h*k_h*rk_h) added as s_h*v_h, and y *= g.
 * The final CxC output projection stays with the caller's GEMV. */
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
                            int head_dim);

/* Scratch elements needed by ck_rwkv7_channelmix_decode (xk + hidden);
 * 0 on bad args. */
size_t ck_rwkv7_channelmix_scratch_elems(int dim, int hidden_dim);

/* Scratch elements needed by ck_rwkv7_time_mix_tail (w_log + a + g + v +
 * kk_raw + kk + k_scaled = 7*dim floats); 0 on bad args. */
size_t ck_rwkv7_time_mix_scratch_elems(int dim);

/* Single-token time-mix tail (composition of the units above, no GEMV/LayerNorm).
 * Inputs are post-shift/post-projection vectors: r/k_raw/v_in [C], LoRA inputs
 * xw/xa/xg/xv [C] (xv == NULL selects the layer-0 produce-only v path, in
 * which case v_first_in/v1/v2/v0 must also be NULL and rank_v is unchecked).
 * Computes w_log/a/g via lora_gates, v via vgate_vmix, kk_raw = k_raw*k_k,
 * kk via norm_kk, k via scale_k, then wkv_decode into state_out/y_preproj.
 * y_preproj is (groupnorm(y)+bonus)*g; the final CxC output projection stays
 * with the caller's GEMV. Requires dim == num_heads*head_dim.
 * `scratch` must hold ck_rwkv7_time_mix_scratch_elems(dim) floats and be
 * disjoint from every other argument. state_out may exactly alias state_in;
 * v_first_out may exactly alias v_first_in. All other output-input and
 * output-output overlaps are rejected before any write. */
int ck_rwkv7_time_mix_tail(const float *r,
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
                           size_t scratch_elems);

/* Squared-ReLU feedforward decode step with caller-provided scratch.
 * xk = x + mu*(shift_in-x); hidden = relu(Wk@xk)^2, Wk [K,C] row-major;
 * out = Wv@hidden, Wv [C,K] row-major; shift_out = x.
 * `scratch` must hold ck_rwkv7_channelmix_scratch_elems(dim,hidden_dim)
 * floats. Either shift_out or out may exactly alias x, but the two outputs
 * must be disjoint. Other output-input overlap and all scratch overlap are
 * rejected before any write. Read-only inputs may overlap each other. */
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
                               size_t scratch_elems);

/* GEMV/LayerNorm callbacks for ck_rwkv7_time_mix_block. GEMV computes
 * y = W x with W [dim,dim] row-major (one row per output channel), matching
 * the engine's NT decode path; LayerNorm computes y = norm(x;gamma,beta).
 * Both return 0 on success, nonzero on failure (propagated as -1). The
 * block never calls them with overlapping arguments. */
typedef int (*ck_rwkv7_gemv_fn)(const float *W,
                                const float *x,
                                float *y,
                                void *ctx);
typedef int (*ck_rwkv7_layer_norm_fn)(const float *x,
                                      const float *gamma,
                                      const float *beta,
                                      float *y,
                                      int dim,
                                      void *ctx);

/* Scratch elements needed by ck_rwkv7_time_mix_block (attn_in + 6 mixes +
 * r/k/v + y_pre + attn_out + 7*dim tail scratch = 19*dim floats);
 * 0 on bad args. */
size_t ck_rwkv7_time_mix_block_scratch_elems(int dim);

/* Full single-token time-mix block (composition, no weight ownership).
 * Computes attn_in = LN(h), six shifts, r/k/v GEMVs, the time-mix tail,
 * the W_o GEMV, and h_out = h + attn_out, updating att_shift/state/v_first.
 * is_layer_zero selects the produce-only v path (v_first_in/v1/v2/v0 must be
 * NULL, rank_v unchecked); otherwise the residual v path. Requires
 * dim == num_heads*head_dim. `scratch` must hold
 * ck_rwkv7_time_mix_block_scratch_elems(dim) floats and be disjoint from
 * every other argument. h_out may exactly alias h, att_shift_out may exactly
 * alias att_shift_in, state_out may exactly alias state_in, and v_first_out
 * may exactly alias v_first_in; the four outputs must otherwise be pairwise
 * disjoint and disjoint from all inputs. Read-only inputs may overlap each
 * other. */
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
                            size_t scratch_elems);

#ifdef __cplusplus
}
#endif

#endif
