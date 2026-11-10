"""RWKV-7 Stage-1 caller wiring: full time-mix + channelmix block via ctypes.

Drives one T=1 decode step through 2 layers (layer 0 produce-only v path +
layer 1 consume path) using only shipped kernels:

  ln1 -> time_mix_shifts -> 3x linear -> time_mix_tail -> out-proj GEMV ->
  residual -> ln2 -> channelmix_decode -> residual -> ln_out -> head GEMV

and checks hidden/logits/state against a torch oracle that uses the
publisher k-scale (k*(1+(a-1)*k_a)), plus a second step proving the
recurrent carry (att/ffn shifts, wkv, v_first).
"""
import argparse
import ctypes

import numpy as np
import torch

from lib_loader import load_lib
from test_utils import numpy_to_ptr

lib = load_lib("libckernel_engine.so", "libckernel_native.so")

lib.ck_rwkv7_time_mix_shifts.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
]
lib.ck_rwkv7_time_mix_shifts.restype = ctypes.c_int

lib.ck_rwkv7_time_mix_scratch_elems.argtypes = [ctypes.c_int]
lib.ck_rwkv7_time_mix_scratch_elems.restype = ctypes.c_size_t

lib.ck_rwkv7_time_mix_tail.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
]
lib.ck_rwkv7_time_mix_tail.restype = ctypes.c_int

lib.ck_rwkv7_channelmix_scratch_elems.argtypes = [ctypes.c_int, ctypes.c_int]
lib.ck_rwkv7_channelmix_scratch_elems.restype = ctypes.c_size_t

lib.ck_rwkv7_channelmix_decode.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
]
lib.ck_rwkv7_channelmix_decode.restype = ctypes.c_int

lib.gemm_blocked_serial.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
lib.gemm_blocked_serial.restype = None

lib.layernorm_rows_checked_workspace.argtypes = [
    ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]
lib.layernorm_rows_checked_workspace.restype = ctypes.c_int

lib.layernorm_rows_checked_f32.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
]
lib.layernorm_rows_checked_f32.restype = ctypes.c_int

lib.ck_residual_add_token_major.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
]
lib.ck_residual_add_token_major.restype = None

lib.ck_rwkv7_time_mix_block_scratch_elems.argtypes = [ctypes.c_int]
lib.ck_rwkv7_time_mix_block_scratch_elems.restype = ctypes.c_size_t

GEMV_FN = ctypes.CFUNCTYPE(ctypes.c_int,
                           ctypes.POINTER(ctypes.c_float),
                           ctypes.POINTER(ctypes.c_float),
                           ctypes.POINTER(ctypes.c_float),
                           ctypes.c_void_p)
LN_FN = ctypes.CFUNCTYPE(ctypes.c_int,
                         ctypes.POINTER(ctypes.c_float),
                         ctypes.POINTER(ctypes.c_float),
                         ctypes.POINTER(ctypes.c_float),
                         ctypes.POINTER(ctypes.c_float),
                         ctypes.c_int, ctypes.c_void_p)

lib.ck_rwkv7_time_mix_block.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_float,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    GEMV_FN, LN_FN, ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
]
lib.ck_rwkv7_time_mix_block.restype = ctypes.c_int

INV_SQRT_E = 0.6065306597126334
NORM_EPS = 1e-5


def layernorm_torch(x, gamma, beta, eps=NORM_EPS):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps) * gamma + beta


def torch_step(r, w_log, k, v, kk, a, g, rk, ln_w, ln_b, state):
    H, N, _ = state.shape
    out_state = torch.empty_like(state)
    y_raw = torch.empty(H * N)
    for h in range(H):
        S = state[h]
        decay = torch.exp(w_log[h])
        b = kk[h] * a[h]
        ab = (-kk[h]).unsqueeze(1) @ b.unsqueeze(0)
        S_new = S * decay.unsqueeze(0) + S @ ab + v[h].unsqueeze(1) @ k[h].unsqueeze(0)
        out_state[h] = S_new
        y_raw[h * N:(h + 1) * N] = S_new @ r[h]
    eps = NORM_EPS * N
    y = y_raw.view(H, N)
    mean = y.mean(dim=1, keepdim=True)
    var = ((y - mean) ** 2).mean(dim=1, keepdim=True)
    y = (y - mean) / torch.sqrt(var + eps) * ln_w.view(H, N) + ln_b.view(H, N)
    s = (r * k * rk).sum(dim=1, keepdim=True)
    y = y + s * v
    return out_state, (y.reshape(-1) * g.reshape(-1))


def run_block_steps(seed, H=2, N=8, K=32, Rd=4, Ra=3, Rg=3, Rv=3,
                    V=24, steps=2, use_driver=False, check_determinism=False,
                    tol_h=5e-5, tol_l=5e-5, tol_s=2e-5):
    """Full steps through 2 layers with explicit per-layer state carry.

    use_driver routes the time-mix half through ck_rwkv7_time_mix_block
    (GEMV/LN callbacks); otherwise the manual unit-by-unit path runs.
    check_determinism repeats each C step from saved inputs and requires
    bit-exact outputs. Returns (ok, last_hidden, last_logits).
    """
    tag = (f"H={H} N={N} K={K} R=({Rd},{Ra},{Rg},{Rv}) "
           f"driver={int(use_driver)}")
    rng = np.random.default_rng(seed)
    C = H * N
    layers = 2
    p = numpy_to_ptr

    def randn(*shape, scale=0.15):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def rand_mu():
        return (rng.random(C).astype(np.float32) * 0.5 + 0.25).astype(np.float32)

    zero_bias = np.zeros(C, dtype=np.float32)
    L = []
    for _ in range(layers):
        L.append({
            "mu": [rand_mu() for _ in range(6)],
            "Wr": randn(C, C), "Wk": randn(C, C), "Wv": randn(C, C),
            "Wo": randn(C, C),
            "w1": randn(C, Rd, scale=0.4), "w2": randn(Rd, C, scale=0.2),
            "w0": randn(C, scale=0.3),
            "a1": randn(C, Ra, scale=0.4), "a2": randn(Ra, C, scale=0.2),
            "a0": randn(C, scale=0.3),
            "g1": randn(C, Rg, scale=0.4), "g2": randn(Rg, C, scale=0.2),
            "v1": randn(C, Rv, scale=0.4), "v2": randn(Rv, C, scale=0.2),
            "v0": randn(C, scale=0.3),
            "k_k": (1.0 + randn(C, scale=0.05)).astype(np.float32),
            "k_a": (np.full(C, 1.02, dtype=np.float32) + randn(C, scale=0.02)).astype(np.float32),
            "rk": np.full((H, N), -0.04, dtype=np.float32),
            "lnw": np.ones(C, dtype=np.float32),
            "lnb": np.zeros(C, dtype=np.float32),
            "ln1g": (np.ones(C) + randn(C, scale=0.1)).astype(np.float32),
            "ln1b": randn(C, scale=0.1).astype(np.float32),
            "ln2g": (np.ones(C) + randn(C, scale=0.1)).astype(np.float32),
            "ln2b": randn(C, scale=0.1).astype(np.float32),
            "fmu": rand_mu(),
            "Wfk": randn(K, C, scale=0.2), "Wfv": randn(C, K, scale=0.2),
        })
    ln0g = (np.ones(C) + randn(C, scale=0.1)).astype(np.float32)
    ln0b = randn(C, scale=0.1).astype(np.float32)
    lnog = (np.ones(C) + randn(C, scale=0.1)).astype(np.float32)
    lnob = randn(C, scale=0.1).astype(np.float32)
    Head = randn(V, C, scale=0.2)

    ln_ws = ctypes.c_size_t()
    assert lib.layernorm_rows_checked_workspace(1, C, ctypes.byref(ln_ws)) == 0
    ln_scratch = np.zeros(ln_ws.value, dtype=np.float32)

    def c_linear(x, W, out_dim=None):
        od = out_dim or C
        y = np.zeros(od, dtype=np.float32)
        bias = np.zeros(od, dtype=np.float32)
        # Engine NT layout: A[M,K]=x, B[N,K]=W rows are output channels,
        # y[o] = sum_c x[c]*W[o,c] (+0 bias; RWKV linears are bias-free).
        lib.gemm_blocked_serial(p(x), p(W), p(bias), p(y), 1, od, C)
        return y

    def c_layernorm(x, gamma, beta):
        y = np.zeros(C, dtype=np.float32)
        rc = lib.layernorm_rows_checked_f32(
            p(x), C, C, p(gamma), C, p(beta), C,
            p(y), C, C, p(ln_scratch), ln_ws.value, 1, C,
            ctypes.c_float(NORM_EPS))
        assert rc == 0, f"layernorm rc={rc}"
        return y

    def c_resid(a, b):
        y = np.zeros(C, dtype=np.float32)
        lib.ck_residual_add_token_major(p(a), p(b), p(y), 1, C)
        return y

    def _gemv_cb(W, x, y, ctx):
        bias = np.zeros(C, dtype=np.float32)
        lib.gemm_blocked_serial(x, W, p(bias), y, 1, C, C)
        return 0
    gemv_cb = GEMV_FN(_gemv_cb)

    def _ln_cb(x, gamma, beta, y, dim, ctx):
        return lib.layernorm_rows_checked_f32(
            x, C, C, gamma, C, beta, C, y, C, C,
            p(ln_scratch), ln_ws.value, 1, C,
            ctypes.c_float(NORM_EPS))
    ln_cb = LN_FN(_ln_cb)

    def c_block_tm(h, att_shift, v_first, wkv_state, par, layer_id):
        """Time-mix half via the C block driver (LN->shifts->GEMVs->tail)."""
        h_new = np.zeros(C, dtype=np.float32)
        ns = np.zeros(C, dtype=np.float32)
        nw = np.zeros((H, N, N), dtype=np.float32)
        nvf = np.zeros(C, dtype=np.float32)
        nsc = lib.ck_rwkv7_time_mix_block_scratch_elems(C)
        assert nsc == 19 * C
        sc = np.zeros(nsc, dtype=np.float32)
        is0 = 1 if layer_id == 0 else 0
        v_args = (None, None, None) if layer_id == 0 else (
            p(par["v1"]), p(par["v2"]), p(par["v0"]))
        rc = lib.ck_rwkv7_time_mix_block(
            p(h), p(att_shift), None if layer_id == 0 else p(v_first),
            p(wkv_state),
            p(par["ln1g"]), p(par["ln1b"]),
            *[p(m) for m in par["mu"]],
            p(par["Wr"]), p(par["Wk"]), p(par["Wv"]), p(par["Wo"]),
            p(par["w1"]), p(par["w2"]), p(par["w0"]),
            p(par["a1"]), p(par["a2"]), p(par["a0"]),
            p(par["g1"]), p(par["g2"]),
            *v_args,
            p(par["k_k"]), p(par["k_a"]), p(par["rk"]),
            p(par["lnw"]), p(par["lnb"]), ctypes.c_float(NORM_EPS),
            C, H, N, Rd, Ra, Rg, Rv, is0,
            gemv_cb, ln_cb, None,
            p(h_new), p(ns), p(nw), p(nvf), p(sc), nsc)
        assert rc == 0, f"block rc={rc}"
        return h_new, ns, nw, nvf

    def c_full_step(h, att, ffn, wkv, vf):
        v_first = vf
        for l in range(layers):
            par = L[l]
            if use_driver:
                h, ns, nw, nvf = c_block_tm(h, att[l], v_first, wkv[l], par, l)
                v_first = nvf
            else:
                attn_in = c_layernorm(h, par["ln1g"], par["ln1b"])
                mixes = [np.zeros(C, dtype=np.float32) for _ in range(6)]
                ns = np.zeros(C, dtype=np.float32)
                rc = lib.ck_rwkv7_time_mix_shifts(
                    p(attn_in), p(att[l]), *[p(m) for m in par["mu"]],
                    *[p(o) for o in mixes], p(ns), C)
                assert rc == 0
                xr, xw, xk, xvv, xa, xg = mixes
                r = c_linear(xr, par["Wr"])
                k_raw = c_linear(xk, par["Wk"])
                v_in = c_linear(xvv, par["Wv"])
                y_pre = np.zeros(C, dtype=np.float32)
                nw = np.zeros((H, N, N), dtype=np.float32)
                nvf = np.zeros(C, dtype=np.float32)
                sc = np.zeros(7 * C, dtype=np.float32)
                if l == 0:
                    rc = lib.ck_rwkv7_time_mix_tail(
                        p(r), p(k_raw), p(v_in), None,
                        p(xw), p(xa), p(xg), None,
                        p(par["w1"]), p(par["w2"]), p(par["w0"]),
                        p(par["a1"]), p(par["a2"]), p(par["a0"]),
                        p(par["g1"]), p(par["g2"]), None, None, None,
                        p(par["k_k"]), p(par["k_a"]), p(par["rk"]),
                        p(par["lnw"]), p(par["lnb"]), ctypes.c_float(NORM_EPS),
                        p(wkv[l]), p(nw), p(y_pre), p(nvf),
                        C, H, N, Rd, Ra, Rg, 1, p(sc), 7 * C)
                    v_first = nvf.copy()
                else:
                    rc = lib.ck_rwkv7_time_mix_tail(
                        p(r), p(k_raw), p(v_in), p(v_first),
                        p(xw), p(xa), p(xg), p(xvv),
                        p(par["w1"]), p(par["w2"]), p(par["w0"]),
                        p(par["a1"]), p(par["a2"]), p(par["a0"]),
                        p(par["g1"]), p(par["g2"]),
                        p(par["v1"]), p(par["v2"]), p(par["v0"]),
                        p(par["k_k"]), p(par["k_a"]), p(par["rk"]),
                        p(par["lnw"]), p(par["lnb"]), ctypes.c_float(NORM_EPS),
                        p(wkv[l]), p(nw), p(y_pre), p(nvf),
                        C, H, N, Rd, Ra, Rg, Rv, p(sc), 7 * C)
                    v_first = nvf.copy()
                assert rc == 0, f"tail rc={rc}"
                h = c_resid(h, c_linear(y_pre, par["Wo"]))
            ffn_in = c_layernorm(h, par["ln2g"], par["ln2b"])
            nfs = np.zeros(C, dtype=np.float32)
            fout = np.zeros(C, dtype=np.float32)
            fs = np.zeros(C + K, dtype=np.float32)
            rc = lib.ck_rwkv7_channelmix_decode(
                p(ffn_in), p(ffn[l]), p(par["fmu"]),
                p(par["Wfk"]), p(par["Wfv"]),
                p(nfs), p(fout), C, K, p(fs), C + K)
            assert rc == 0, f"ffn rc={rc}"
            h = c_resid(h, fout)
            att[l], ffn[l], wkv[l] = ns, nfs, nw
        hout = c_layernorm(h, lnog, lnob)
        logits = c_linear(hout, Head, out_dim=V)
        return h, hout, logits, v_first

    def t(x):
        if isinstance(x, torch.Tensor):
            return x.clone().detach()
        return torch.from_numpy(x.copy())

    def t_full_step(h, att, ffn, wkv, vf):
        v_first = vf
        for l in range(layers):
            par = L[l]
            attn_in = layernorm_torch(h, t(par["ln1g"]), t(par["ln1b"]))
            d = t(att[l]) - attn_in
            xr, xw, xk, xvv, xa, xg = [attn_in + t(m) * d for m in par["mu"]]
            att[l] = attn_in.clone()
            r = xr @ t(par["Wr"]).T
            k_raw = xk @ t(par["Wk"]).T
            v_in = xvv @ t(par["Wv"]).T
            w = -INV_SQRT_E * torch.sigmoid(torch.tanh(xw @ t(par["w1"])) @ t(par["w2"]) + t(par["w0"]))
            a = torch.sigmoid((xa @ t(par["a1"])) @ t(par["a2"]) + t(par["a0"]))
            g = torch.sigmoid(xg @ t(par["g1"])) @ t(par["g2"])
            if l == 0:
                v, v_first = v_in, v_in.clone()
            else:
                gate = torch.sigmoid((xvv @ t(par["v1"])) @ t(par["v2"]) + t(par["v0"]))
                v = v_in + (v_first - v_in) * gate
                v_first = v_first.clone()
            kk = (k_raw * t(par["k_k"])).view(H, N)
            kk = kk / torch.clamp(torch.sqrt((kk * kk).sum(dim=1, keepdim=True)), min=1e-12)
            ks = (k_raw * (1 + (a - 1) * t(par["k_a"]))).view(H, N)
            wkv[l], y_pre = torch_step(
                r.view(H, N), w.view(H, N), ks, v.view(H, N), kk,
                a.view(H, N), g, t(par["rk"]), t(par["lnw"]), t(par["lnb"]), wkv[l])
            h = h + y_pre @ t(par["Wo"]).T
            ffn_in = layernorm_torch(h, t(par["ln2g"]), t(par["ln2b"]))
            xkf = ffn_in + t(par["fmu"]) * (t(ffn[l]) - ffn_in)
            ffn[l] = ffn_in.clone()
            h = h + (torch.relu(xkf @ t(par["Wfk"]).T) ** 2) @ t(par["Wfv"]).T
        hout = layernorm_torch(h, t(lnog), t(lnob))
        return h, hout, hout @ t(Head).T, v_first

    ok = True
    att_c = [np.zeros(C, dtype=np.float32) for _ in range(layers)]
    ffn_c = [np.zeros(C, dtype=np.float32) for _ in range(layers)]
    wkv_c = [np.zeros((H, N, N), dtype=np.float32) for _ in range(layers)]
    att_t = [torch.zeros(C) for _ in range(layers)]
    ffn_t = [torch.zeros(C) for _ in range(layers)]
    wkv_t = [torch.zeros(H, N, N) for _ in range(layers)]
    vf_c, vf_t = None, None
    print(f"block {tag}:")
    if use_driver:
        # Driver guards: bad layer flag and NULL callbacks must fail.
        par0 = L[0]
        h0g = np.zeros(C, dtype=np.float32)
        ns0 = np.zeros(C, dtype=np.float32)
        nw0 = np.zeros((H, N, N), dtype=np.float32)
        nv0 = np.zeros(C, dtype=np.float32)
        nsc0 = lib.ck_rwkv7_time_mix_block_scratch_elems(C)
        assert nsc0 == 19 * C
        sc0 = np.zeros(nsc0, dtype=np.float32)

        def _guard(flag, gemv, ln):
            return lib.ck_rwkv7_time_mix_block(
                p(h0g), p(ns0), None, p(nw0),
                p(par0["ln1g"]), p(par0["ln1b"]),
                *[p(m) for m in par0["mu"]],
                p(par0["Wr"]), p(par0["Wk"]), p(par0["Wv"]), p(par0["Wo"]),
                p(par0["w1"]), p(par0["w2"]), p(par0["w0"]),
                p(par0["a1"]), p(par0["a2"]), p(par0["a0"]),
                p(par0["g1"]), p(par0["g2"]), None, None, None,
                p(par0["k_k"]), p(par0["k_a"]), p(par0["rk"]),
                p(par0["lnw"]), p(par0["lnb"]), ctypes.c_float(NORM_EPS),
                C, H, N, Rd, Ra, Rg, Rv, flag,
                gemv, ln, None,
                p(h0g), p(ns0), p(nw0), p(nv0), p(sc0), nsc0)

        gb = (_guard(2, gemv_cb, ln_cb), _guard(1, GEMV_FN(), ln_cb),
              _guard(1, gemv_cb, LN_FN()))
        print(f"  driver guards={gb[0]},{gb[1]},{gb[2]}")
        ok &= gb == (-1, -1, -1)
    for step in range(steps):
        h_init = randn(C)
        hc = c_layernorm(h_init.copy(), ln0g, ln0b)
        ht = layernorm_torch(t(h_init), t(ln0g), t(ln0b))
        if check_determinism:
            snap = (hc.copy(), [a.copy() for a in att_c],
                    [f.copy() for f in ffn_c], [w.copy() for w in wkv_c],
                    None if vf_c is None else vf_c.copy())

            def _once(s):
                h0, aa, ff, ww, vv = s
                aa = [a.copy() for a in aa]
                ff = [f.copy() for f in ff]
                ww = [w.copy() for w in ww]
                vv = None if vv is None else vv.copy()
                o = c_full_step(h0.copy(), aa, ff, ww, vv)
                return o + (aa, ff, ww)

            r1 = _once(snap)
            r2 = _once(snap)
            det = (np.array_equal(r1[0], r2[0]) and np.array_equal(r1[2], r2[2])
                   and all(np.array_equal(a, b) for a, b in zip(r1[4], r2[4]))
                   and all(np.array_equal(a, b) for a, b in zip(r1[5], r2[5]))
                   and all(np.array_equal(a, b) for a, b in zip(r1[6], r2[6]))
                   and np.array_equal(r1[3], r2[3]))
            print(f"  determinism step {step}: {'PASS' if det else 'FAIL'}")
            ok &= det
            hc, hout_c, logits_c, vf_c = r1[0], r1[1], r1[2], r1[3]
            att_c, ffn_c, wkv_c = (list(x) for x in (r1[4], r1[5], r1[6]))
        else:
            hc, hout_c, logits_c, vf_c = c_full_step(hc, att_c, ffn_c, wkv_c, vf_c)
        ht, hout_t, logits_t, vf_t = t_full_step(ht, att_t, ffn_t, wkv_t, vf_t)
        dh = float(np.max(np.abs(hc - ht.numpy())))
        dl = float(np.max(np.abs(logits_c - logits_t.numpy())))
        ds = max(float(np.max(np.abs(a - b.numpy())))
                 for a, b in zip(wkv_c, wkv_t))
        ah = float(np.max(np.abs(hc)))
        mh = float(np.mean(hc - ht.numpy()))
        print(f"  step {step}: hidden={dh:.2e} (absmax={ah:.2f} mean={mh:.1e}) "
              f"logits={dl:.2e} wkv={ds:.2e}")
        ok &= dh <= tol_h and dl <= tol_l and ds <= tol_s
    print("  ->", "PASS" if ok else "FAIL")
    return ok, hc, logits_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ok = True
    ok_m, h_m, l_m = run_block_steps(701)
    ok_d, h_d, l_d = run_block_steps(701, use_driver=True)
    ok &= ok_m and ok_d
    exact = np.array_equal(h_m, h_d) and np.array_equal(l_m, l_d)
    print(f"driver-vs-manual bit-exact: {exact}")
    ok &= exact
    # Stage 3a: config-accurate shapes (N=64 head dim, real LoRA ranks).
    # Wider dots carry bigger magnitudes (GEMV outs O(3), 64-term readouts),
    # so FP32 accumulation-order noise scales up ~100x vs the tiny config;
    # tolerances scale accordingly (relative error stays ~1e-4).
    big = dict(H=8, N=64, K=1024, Rd=32, Ra=16, Rg=16, Rv=16, V=64,
               steps=1, tol_h=1e-3, tol_l=5e-5, tol_s=1e-4)
    ok_a, _, _ = run_block_steps(702, **big)
    ok &= ok_a
    ok_ad, _, _ = run_block_steps(702, use_driver=True,
                                  check_determinism=True, **big)
    ok &= ok_ad
    if not ok:
        raise SystemExit(1)
    print("All RWKV-7 block wiring checks passed.")


if __name__ == "__main__":
    main()
