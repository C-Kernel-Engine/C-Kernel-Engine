"""RWKV-7 T=1 FP32 decode parity (torch oracle, single-token eager order)."""
import argparse
import ctypes

import numpy as np
import torch

from lib_loader import load_lib
from test_utils import numpy_to_ptr

lib = load_lib("libckernel_engine.so", "libckernel_native.so")

lib.ck_rwkv7_token_shift_lerp.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
]
lib.ck_rwkv7_token_shift_lerp.restype = ctypes.c_int

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

lib.ck_rwkv7_norm_kk.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int,
]
lib.ck_rwkv7_norm_kk.restype = ctypes.c_int

lib.ck_rwkv7_scale_k.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
]
lib.ck_rwkv7_scale_k.restype = ctypes.c_int

lib.ck_rwkv7_wkv_decode.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
]
lib.ck_rwkv7_wkv_decode.restype = ctypes.c_int

lib.ck_rwkv7_lora_gates.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
lib.ck_rwkv7_lora_gates.restype = ctypes.c_int

lib.ck_rwkv7_vgate_vmix.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
lib.ck_rwkv7_vgate_vmix.restype = ctypes.c_int

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

lib.ck_rwkv7_state_step.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
]
lib.ck_rwkv7_state_step.restype = ctypes.c_int

lib.ck_rwkv7_state_readout.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
]
lib.ck_rwkv7_state_readout.restype = ctypes.c_int

lib.ck_rwkv7_apply_kk_factor.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
]
lib.ck_rwkv7_apply_kk_factor.restype = ctypes.c_int

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

INV_SQRT_E = 0.6065306597126334

NORM_EPS = 1e-5


def torch_step(r, w_log, k, v, kk, a, g, rk, ln_w, ln_b, state):
    """Single-token generalised delta-rule reference (FP32, B=1):
    S_new = S*exp(w_log) + S@((-kk)(kk*a)^T) + v@k^T, y = S_new@r,
    then per-head GroupNorm, bonus, gate."""
    H, N, _ = state.shape
    out_state = torch.empty_like(state)
    y_raw = torch.empty(H * N)
    for h in range(H):
        S = state[h]
        decay = torch.exp(w_log[h])  # [N]
        b = kk[h] * a[h]  # [N]
        ab = (-kk[h]).unsqueeze(1) @ b.unsqueeze(0)  # [N,N]
        S_new = S * decay.unsqueeze(0) + S @ ab + v[h].unsqueeze(1) @ k[h].unsqueeze(0)
        out_state[h] = S_new
        y_raw[h * N:(h + 1) * N] = S_new @ r[h]
    eps = NORM_EPS * N
    y = y_raw.view(H, N)
    mean = y.mean(dim=1, keepdim=True)
    var = ((y - mean) ** 2).mean(dim=1, keepdim=True)
    y = (y - mean) / torch.sqrt(var + eps) * ln_w.view(H, N) + ln_b.view(H, N)
    s = (r * k * rk).sum(dim=1, keepdim=True)  # per-head bonus scalar
    y = y + s * v
    y = y.reshape(-1) * g.reshape(-1)
    return out_state, y


def run_case(H, N, seed):
    rng = np.random.default_rng(seed)
    C = H * N

    def randn(*shape, scale=0.5):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    r = randn(C)
    w_log = (-np.abs(rng.standard_normal(C)) * 0.6).astype(np.float32)
    k_raw = randn(C)
    v = randn(C)
    kk_raw = randn(C)
    a = (1 / (1 + np.exp(-rng.standard_normal(C).astype(np.float32)))).astype(np.float32)
    k_k = (1.0 + randn(C, scale=0.05)).astype(np.float32)
    k_a = (np.full(C, 1.02, dtype=np.float32) + randn(C, scale=0.02)).astype(np.float32)
    g = randn(C, scale=0.3)
    rk = (np.full((H, N), -0.04, dtype=np.float32) + randn(H, N, scale=0.01)).astype(np.float32)
    ln_w = (np.ones(C) + randn(C, scale=0.1)).astype(np.float32)
    ln_b = (randn(C, scale=0.1)).astype(np.float32)
    state = (randn(H, N, N, scale=0.25)).astype(np.float32)

    # kk normalize helper
    kk = np.zeros_like(kk_raw)
    rc = lib.ck_rwkv7_norm_kk(numpy_to_ptr(kk_raw), numpy_to_ptr(kk), H, N)
    assert rc == 0
    t_kk = torch.from_numpy(kk_raw.copy()).view(H, N)
    t_kk = t_kk / torch.clamp(torch.sqrt((t_kk * t_kk).sum(dim=1, keepdim=True)), min=1e-12)
    dkk = float(np.max(np.abs(kk - t_kk.numpy().reshape(-1))))

    # k scale helper
    k = np.zeros_like(k_raw)
    rc = lib.ck_rwkv7_scale_k(
        numpy_to_ptr(k_raw), numpy_to_ptr(a), numpy_to_ptr(k_a), numpy_to_ptr(k), C)
    assert rc == 0
    ref_k = k_raw * (1 + (k_a - 1) * a)
    dk = float(np.max(np.abs(k - ref_k)))

    # full wkv step
    state_out = np.zeros_like(state)
    y_out = np.zeros(C, dtype=np.float32)
    rc = lib.ck_rwkv7_wkv_decode(
        numpy_to_ptr(r), numpy_to_ptr(w_log), numpy_to_ptr(k), numpy_to_ptr(v),
        numpy_to_ptr(kk), numpy_to_ptr(a), numpy_to_ptr(g), numpy_to_ptr(rk),
        numpy_to_ptr(ln_w), numpy_to_ptr(ln_b), ctypes.c_float(NORM_EPS),
        numpy_to_ptr(state), numpy_to_ptr(state_out), numpy_to_ptr(y_out), H, N)
    assert rc == 0
    t_state, t_y = torch_step(
        torch.from_numpy(r.copy()).view(H, N), torch.from_numpy(w_log.copy()).view(H, N),
        torch.from_numpy(k.copy()).view(H, N), torch.from_numpy(v.copy()).view(H, N),
        torch.from_numpy(kk.copy()).view(H, N), torch.from_numpy(a.copy()).view(H, N),
        torch.from_numpy(g.copy()), torch.from_numpy(rk.copy()),
        torch.from_numpy(ln_w.copy()), torch.from_numpy(ln_b.copy()),
        torch.from_numpy(state.copy()))
    dy = float(np.max(np.abs(y_out - t_y.numpy())))
    ds = float(np.max(np.abs(state_out - t_state.numpy())))

    # 8-step rollout through C vs torch loop (state carry only; same inputs)
    steps = 8
    s_c, s_t = state.copy(), torch.from_numpy(state.copy())
    max_dy, max_ds = 0.0, 0.0
    for _ in range(steps):
        so = np.zeros_like(s_c)
        yo = np.zeros(C, dtype=np.float32)
        rc = lib.ck_rwkv7_wkv_decode(
            numpy_to_ptr(r), numpy_to_ptr(w_log), numpy_to_ptr(k), numpy_to_ptr(v),
            numpy_to_ptr(kk), numpy_to_ptr(a), numpy_to_ptr(g), numpy_to_ptr(rk),
            numpy_to_ptr(ln_w), numpy_to_ptr(ln_b), ctypes.c_float(NORM_EPS),
            numpy_to_ptr(s_c), numpy_to_ptr(so), numpy_to_ptr(yo), H, N)
        assert rc == 0
        ts, ty = torch_step(
            torch.from_numpy(r.copy()).view(H, N), torch.from_numpy(w_log.copy()).view(H, N),
            torch.from_numpy(k.copy()).view(H, N), torch.from_numpy(v.copy()).view(H, N),
            torch.from_numpy(kk.copy()).view(H, N), torch.from_numpy(a.copy()).view(H, N),
            torch.from_numpy(g.copy()), torch.from_numpy(rk.copy()),
            torch.from_numpy(ln_w.copy()), torch.from_numpy(ln_b.copy()), s_t)
        max_dy = max(max_dy, float(np.max(np.abs(yo - ty.numpy()))))
        max_ds = max(max_ds, float(np.max(np.abs(so - ts.numpy()))))
        s_c, s_t = so, ts

    print(f"H={H} N={N}: kk={dkk:.2e} kscale={dk:.2e} wkv_y={dy:.2e} "
          f"wkv_s={ds:.2e} roll_y={max_dy:.2e} roll_s={max_ds:.2e}")
    ok = dkk <= 1e-6 and dk <= 1e-6 and dy <= 5e-5 and ds <= 2e-5 \
        and max_dy <= 5e-5 and max_ds <= 2e-5
    print("  ->", "PASS" if ok else "FAIL")
    return ok


def run_lora_case(H, N, Rd, Ra, Rg, Rv, seed):
    """LoRA gate chains + vgate/vmix parity, plus end-to-end."""
    rng = np.random.default_rng(seed)
    C = H * N

    def randn(*shape, scale=0.5):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    xw, xa, xg = randn(C, scale=0.5), randn(C, scale=0.5), randn(C, scale=0.5)
    w1, w2 = randn(C, Rd, scale=0.4), randn(Rd, C, scale=0.2)
    w0 = randn(C, scale=0.3)
    a1, a2 = randn(C, Ra, scale=0.4), randn(Ra, C, scale=0.2)
    a0 = randn(C, scale=0.3)
    g1, g2 = randn(C, Rg, scale=0.4), randn(Rg, C, scale=0.2)
    v1, v2 = randn(C, Rv, scale=0.4), randn(Rv, C, scale=0.2)
    v0 = randn(C, scale=0.3)
    v_in, v_first = randn(C), randn(C)
    xv = randn(C, scale=0.5)

    w_log = np.zeros(C, dtype=np.float32)
    a_out = np.zeros(C, dtype=np.float32)
    g_out = np.zeros(C, dtype=np.float32)
    rc = lib.ck_rwkv7_lora_gates(
        numpy_to_ptr(xw), numpy_to_ptr(xa), numpy_to_ptr(xg),
        numpy_to_ptr(w1), numpy_to_ptr(w2), numpy_to_ptr(w0),
        numpy_to_ptr(a1), numpy_to_ptr(a2), numpy_to_ptr(a0),
        numpy_to_ptr(g1), numpy_to_ptr(g2),
        C, Rd, Ra, Rg,
        numpy_to_ptr(w_log), numpy_to_ptr(a_out), numpy_to_ptr(g_out))
    assert rc == 0

    t = lambda x: torch.from_numpy(x.copy())
    t_w = -INV_SQRT_E * torch.sigmoid(torch.tanh(t(xw) @ t(w1)) @ t(w2) + t(w0))
    t_a = torch.sigmoid((t(xa) @ t(a1)) @ t(a2) + t(a0))
    t_g = torch.sigmoid(t(xg) @ t(g1)) @ t(g2)
    dw = float(torch.max(torch.abs(torch.from_numpy(w_log.copy()) - t_w)).item())
    da = float(torch.max(torch.abs(torch.from_numpy(a_out.copy()) - t_a)).item())
    dg = float(torch.max(torch.abs(torch.from_numpy(g_out.copy()) - t_g)).item())

    # layer >0 vmix
    v_out = np.zeros(C, dtype=np.float32)
    vf_out = np.zeros(C, dtype=np.float32)
    gate = np.zeros(C, dtype=np.float32)
    rc = lib.ck_rwkv7_vgate_vmix(
        numpy_to_ptr(v_in), numpy_to_ptr(v_first), numpy_to_ptr(xv),
        numpy_to_ptr(v1), numpy_to_ptr(v2), numpy_to_ptr(v0),
        C, Rv, numpy_to_ptr(v_out), numpy_to_ptr(vf_out), numpy_to_ptr(gate))
    assert rc == 0
    t_gate = torch.sigmoid((t(xv) @ t(v1)) @ t(v2) + t(v0))
    t_v = t(v_in) + (t(v_first) - t(v_in)) * t_gate
    dv = float(torch.max(torch.abs(torch.from_numpy(v_out.copy()) - t_v)).item())
    dgate = float(torch.max(torch.abs(torch.from_numpy(gate.copy()) - t_gate)).item())
    dvf = float(np.max(np.abs(vf_out - v_first)))

    # layer 0 produce-only path (xv None -> v_first = v_in)
    v0_out = np.zeros(C, dtype=np.float32)
    vf0_out = np.zeros(C, dtype=np.float32)
    rc = lib.ck_rwkv7_vgate_vmix(
        numpy_to_ptr(v_in), None, None, None, None, None,
        C, Rv, numpy_to_ptr(v0_out), numpy_to_ptr(vf0_out), None)
    assert rc == 0
    dl0 = float(max(np.max(np.abs(v0_out - v_in)), np.max(np.abs(vf0_out - v_in))))

    # end-to-end: gates -> kk norm -> k scale -> vmix -> wkv step
    r = randn(C)
    k_raw = randn(C)
    kk_raw = randn(C)
    k_k = (1.0 + randn(C, scale=0.05)).astype(np.float32)
    k_a = (np.full(C, 1.02, dtype=np.float32) + randn(C, scale=0.02)).astype(np.float32)
    rk = np.full((H, N), -0.04, dtype=np.float32)
    ln_w = np.ones(C, dtype=np.float32)
    ln_b = np.zeros(C, dtype=np.float32)
    state = (randn(H, N, N, scale=0.25)).astype(np.float32)
    kk = np.zeros(C, dtype=np.float32)
    ks = np.zeros(C, dtype=np.float32)
    assert lib.ck_rwkv7_norm_kk(numpy_to_ptr(kk_raw * k_k), numpy_to_ptr(kk), H, N) == 0
    assert lib.ck_rwkv7_scale_k(numpy_to_ptr(k_raw), numpy_to_ptr(a_out),
                                    numpy_to_ptr(k_a), numpy_to_ptr(ks), C) == 0
    so = np.zeros_like(state)
    yo = np.zeros(C, dtype=np.float32)
    rc = lib.ck_rwkv7_wkv_decode(
        numpy_to_ptr(r), numpy_to_ptr(w_log), numpy_to_ptr(ks), numpy_to_ptr(v_out),
        numpy_to_ptr(kk), numpy_to_ptr(a_out), numpy_to_ptr(g_out), numpy_to_ptr(rk),
        numpy_to_ptr(ln_w), numpy_to_ptr(ln_b), ctypes.c_float(NORM_EPS),
        numpy_to_ptr(state), numpy_to_ptr(so), numpy_to_ptr(yo), H, N)
    assert rc == 0
    t_kk = (t(kk_raw) * t(k_k)).view(H, N)
    t_kk = t_kk / torch.clamp(torch.sqrt((t_kk * t_kk).sum(dim=1, keepdim=True)), min=1e-12)
    t_ks = (t(k_raw) * (1 + (t(k_a) - 1) * t_a)).view(H, N)
    t_s, t_y = torch_step(t(r).view(H, N), t_w.view(H, N), t_ks,
                          t_v.view(H, N), t_kk, t_a.view(H, N),
                          t_g, t(rk), t(ln_w), t(ln_b), t(state))
    de2e_y = float(torch.max(torch.abs(torch.from_numpy(yo.copy()) - t_y)).item())
    de2e_s = float(torch.max(torch.abs(torch.from_numpy(so.copy()) - t_s)).item())

    print(f"lora H={H} N={N} R=({Rd},{Ra},{Rg},{Rv}): w={dw:.2e} a={da:.2e} "
          f"g={dg:.2e} v={dv:.2e} gate={dgate:.2e} vf={dvf:.2e} l0={dl0:.2e} "
          f"e2e_y={de2e_y:.2e} e2e_s={de2e_s:.2e}")
    ok = dw <= 2e-6 and da <= 2e-6 and dg <= 2e-6 and dv <= 2e-6 \
        and dgate <= 2e-6 and dvf == 0.0 and dl0 == 0.0 \
        and de2e_y <= 5e-5 and de2e_s <= 2e-5
    print("  ->", "PASS" if ok else "FAIL")
    return ok


def run_channelmix_case(C, K, seed):
    """Channelmix parity (xk hoist, shift_out) plus fail-closed guards."""
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(C) * 0.5).astype(np.float32)
    shift = (rng.standard_normal(C) * 0.5).astype(np.float32)
    mu = (rng.random(C).astype(np.float32) * 0.5 + 0.25).astype(np.float32)
    Wk = (rng.standard_normal((K, C)) * 0.3).astype(np.float32)
    Wv = (rng.standard_normal((C, K)) * 0.3).astype(np.float32)

    assert lib.ck_rwkv7_channelmix_scratch_elems(C, K) == C + K
    assert lib.ck_rwkv7_channelmix_scratch_elems(0, K) == 0
    scratch = np.zeros(C + K, dtype=np.float32)
    shift_out = np.zeros(C, dtype=np.float32)
    out = np.zeros(C, dtype=np.float32)
    rc = lib.ck_rwkv7_channelmix_decode(
        numpy_to_ptr(x), numpy_to_ptr(shift), numpy_to_ptr(mu),
        numpy_to_ptr(Wk), numpy_to_ptr(Wv),
        numpy_to_ptr(shift_out), numpy_to_ptr(out), C, K,
        numpy_to_ptr(scratch), C + K)
    assert rc == 0
    xk = x + mu * (shift - x)
    ref = Wv @ (np.maximum(Wk @ xk, 0) ** 2)
    do = float(np.max(np.abs(out - ref)))
    ds = float(np.max(np.abs(shift_out - x)))

    # In-place carry (shift_out == shift_in) is supported by the circuit's
    # ffn_shift buffer; per-index read-before-write keeps it exact.
    inplace_shift = shift.copy()
    inplace_out = np.zeros(C, dtype=np.float32)
    inplace = lib.ck_rwkv7_channelmix_decode(
        numpy_to_ptr(x), numpy_to_ptr(inplace_shift), numpy_to_ptr(mu),
        numpy_to_ptr(Wk), numpy_to_ptr(Wv),
        numpy_to_ptr(inplace_shift), numpy_to_ptr(inplace_out), C, K,
        numpy_to_ptr(scratch), C + K)
    inplace_ok = (
        inplace == 0
        and float(np.max(np.abs(inplace_shift - x))) == 0.0
        and float(np.max(np.abs(inplace_out - ref))) <= 1e-5
    )
    # fail-closed guards: null/short/aliased scratch
    no_scratch = lib.ck_rwkv7_channelmix_decode(
        numpy_to_ptr(x), numpy_to_ptr(shift), numpy_to_ptr(mu),
        numpy_to_ptr(Wk), numpy_to_ptr(Wv),
        numpy_to_ptr(shift_out), numpy_to_ptr(out), C, K, None, 0)
    short = lib.ck_rwkv7_channelmix_decode(
        numpy_to_ptr(x), numpy_to_ptr(shift), numpy_to_ptr(mu),
        numpy_to_ptr(Wk), numpy_to_ptr(Wv),
        numpy_to_ptr(shift_out), numpy_to_ptr(out), C, K,
        numpy_to_ptr(scratch), C + K - 1)
    aliased = lib.ck_rwkv7_channelmix_decode(
        numpy_to_ptr(x), numpy_to_ptr(shift), numpy_to_ptr(mu),
        numpy_to_ptr(Wk), numpy_to_ptr(Wv),
        numpy_to_ptr(shift_out), numpy_to_ptr(out), C, K,
        numpy_to_ptr(x), C + K)
    print(f"channelmix C={C} K={K}: out={do:.2e} shift={ds:.2e} "
          f"inplace={inplace_ok} guards={no_scratch},{short},{aliased}")
    ok = do <= 1e-5 and ds == 0.0 and inplace_ok \
        and (no_scratch, short, aliased) == (-1, -1, -1)
    print("  ->", "PASS" if ok else "FAIL")
    return ok


def run_shift_case(dim, seed):
    """Token-shift parity at mu=0/1/0.25, in-place support, carry chain."""
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(dim) * 0.5 + 0.25).astype(np.float32)
    shift = (rng.standard_normal(dim) * 0.5 - 0.25).astype(np.float32)
    assert np.max(np.abs(x - shift)) > 1e-3  # distinct inputs: mixing is observable
    ok = True
    for mu_val in (0.0, 1.0, 0.25):
        mu = np.full(dim, np.float32(mu_val))
        xm = np.zeros(dim, dtype=np.float32)
        shift_out = np.zeros(dim, dtype=np.float32)
        rc = lib.ck_rwkv7_token_shift_lerp(
            numpy_to_ptr(x), numpy_to_ptr(shift), numpy_to_ptr(mu),
            numpy_to_ptr(xm), numpy_to_ptr(shift_out), dim)
        ref = x + mu_val * (shift - x)
        dxm = float(np.max(np.abs(xm - ref)))
        ds = float(np.max(np.abs(shift_out - x)))
        case_ok = rc == 0 and dxm <= 1e-6 and ds == 0.0
        print(f"shift mu={mu_val}: xm={dxm:.2e} shift_out={ds:.2e} "
              f"-> {'PASS' if case_ok else 'FAIL'}")
        ok &= case_ok

    # in-place supported: shift_out == shift_in buffer
    mu = np.full(dim, np.float32(0.25))
    buf = shift.copy()
    xm = np.zeros(dim, dtype=np.float32)
    rc = lib.ck_rwkv7_token_shift_lerp(
        numpy_to_ptr(x), numpy_to_ptr(buf), numpy_to_ptr(mu),
        numpy_to_ptr(xm), numpy_to_ptr(buf), dim)
    ref = x + 0.25 * (shift - x)
    in_place_ok = rc == 0 and float(np.max(np.abs(xm - ref))) <= 1e-6 \
        and float(np.max(np.abs(buf - x))) == 0.0
    print(f"shift in-place: -> {'PASS' if in_place_ok else 'FAIL'}")
    ok &= in_place_ok

    # carry chain: next shift equals the original current input
    x0 = (rng.standard_normal(dim) * 0.5).astype(np.float32)
    x1 = (rng.standard_normal(dim) * 0.5).astype(np.float32)
    s0 = (rng.standard_normal(dim) * 0.5).astype(np.float32)
    carry = s0.copy()
    tmp = np.zeros(dim, dtype=np.float32)
    assert lib.ck_rwkv7_token_shift_lerp(
        numpy_to_ptr(x0), numpy_to_ptr(carry), numpy_to_ptr(mu),
        numpy_to_ptr(tmp), numpy_to_ptr(carry), dim) == 0
    chain_ok = float(np.max(np.abs(carry - x0))) == 0.0
    xm1 = np.zeros(dim, dtype=np.float32)
    assert lib.ck_rwkv7_token_shift_lerp(
        numpy_to_ptr(x1), numpy_to_ptr(carry), numpy_to_ptr(mu),
        numpy_to_ptr(xm1), numpy_to_ptr(carry), dim) == 0
    ref1 = x1 + 0.25 * (x0 - x1)
    chain_ok &= float(np.max(np.abs(xm1 - ref1))) <= 1e-6 \
        and float(np.max(np.abs(carry - x1))) == 0.0
    print(f"shift carry-chain: -> {'PASS' if chain_ok else 'FAIL'}")
    ok &= chain_ok
    print(f"shift dim={dim}: -> {'PASS' if ok else 'FAIL'}")
    return ok


def run_state_case(N, seed):
    """Direct single-head state_step + readout parity vs torch reference."""
    rng = np.random.default_rng(seed)

    def randn(*shape, scale=0.5):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    w_log = (-np.abs(rng.standard_normal(N)) * 0.6).astype(np.float32)
    k, v = randn(N), randn(N)
    kk = randn(N)
    kk = (kk / max(float(np.sqrt((kk * kk).sum())), 1e-12)).astype(np.float32)
    a = (1 / (1 + np.exp(-rng.standard_normal(N).astype(np.float32)))).astype(np.float32)
    r = randn(N)
    state = (randn(N, N, scale=0.25)).astype(np.float32)

    so = np.zeros((N, N), dtype=np.float32)
    yo = np.zeros(N, dtype=np.float32)
    assert lib.ck_rwkv7_state_step(
        numpy_to_ptr(w_log), numpy_to_ptr(k), numpy_to_ptr(v),
        numpy_to_ptr(kk), numpy_to_ptr(a),
        numpy_to_ptr(state), numpy_to_ptr(so), N) == 0
    assert lib.ck_rwkv7_state_readout(
        numpy_to_ptr(so), numpy_to_ptr(r), numpy_to_ptr(yo), N) == 0

    t = lambda x: torch.from_numpy(x.copy())
    S, decay = t(state), torch.exp(t(w_log))
    b = t(kk) * t(a)
    t_s = S * decay.unsqueeze(0) + S @ ((-t(kk)).unsqueeze(1) @ b.unsqueeze(0)) \
        + t(v).unsqueeze(1) @ t(k).unsqueeze(0)
    t_y = t_s @ t(r)
    ds = float(torch.max(torch.abs(torch.from_numpy(so.copy()) - t_s)).item())
    dy = float(torch.max(torch.abs(torch.from_numpy(yo.copy()) - t_y)).item())
    print(f"state N={N}: step={ds:.2e} readout={dy:.2e}")
    ok = ds <= 2e-5 and dy <= 2e-5
    print("  ->", "PASS" if ok else "FAIL")
    return ok


def run_shifts_case(dim, seed):
    """Six-way shift bundle parity vs scalar lerp, carry, and guards."""
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(dim) * 0.5).astype(np.float32)
    shift = (rng.standard_normal(dim) * 0.5).astype(np.float32)
    mus = [(rng.random(dim).astype(np.float32) * 0.5 + 0.25).astype(np.float32)
           for _ in range(6)]
    outs = [np.zeros(dim, dtype=np.float32) for _ in range(6)]
    carry = np.zeros(dim, dtype=np.float32)
    p = numpy_to_ptr
    rc = lib.ck_rwkv7_time_mix_shifts(
        p(x), p(shift), *[p(m) for m in mus],
        *[p(o) for o in outs], p(carry), dim)
    assert rc == 0
    ok = True
    for o, m in zip(outs, mus):
        ref = x + m * (shift - x)
        ok &= float(np.max(np.abs(o - ref))) <= 1e-6
    ok &= float(np.max(np.abs(carry - x))) == 0.0
    # Matches six scalar lerps lane-for-lane.
    for o, m in zip(outs, mus):
        xm = np.zeros(dim, dtype=np.float32)
        so = np.zeros(dim, dtype=np.float32)
        assert lib.ck_rwkv7_token_shift_lerp(
            p(x), p(shift), p(m), p(xm), p(so), dim) == 0
        ok &= bool(np.array_equal(o, xm)) and bool(np.array_equal(carry, so))
    # In-place carry (shift_out == shift_in) is supported.
    buf = shift.copy()
    outs2 = [np.zeros(dim, dtype=np.float32) for _ in range(6)]
    rc = lib.ck_rwkv7_time_mix_shifts(
        p(x), p(buf), *[p(m) for m in mus],
        *[p(o) for o in outs2], p(buf), dim)
    ok &= rc == 0 and float(np.max(np.abs(buf - x))) == 0.0
    for o, o2 in zip(outs, outs2):
        ok &= bool(np.array_equal(o, o2))
    # Guards: output-output alias, output-input alias, null, bad dim.
    g1 = lib.ck_rwkv7_time_mix_shifts(
        p(x), p(shift), *[p(m) for m in mus],
        p(outs[0]), p(outs[1]), p(outs[2]), p(outs[3]), p(outs[4]),
        p(outs[0]), p(carry), dim)
    g2 = lib.ck_rwkv7_time_mix_shifts(
        p(x), p(shift), *[p(m) for m in mus],
        p(x), p(outs[1]), p(outs[2]), p(outs[3]), p(outs[4]), p(outs[5]),
        p(carry), dim)
    g3 = lib.ck_rwkv7_time_mix_shifts(
        None, p(shift), *[p(m) for m in mus],
        *[p(o) for o in outs], p(carry), dim)
    g4 = lib.ck_rwkv7_time_mix_shifts(
        p(x), p(shift), *[p(m) for m in mus],
        *[p(o) for o in outs], p(carry), 0)
    print(f"shifts dim={dim}: guards={g1},{g2},{g3},{g4}")
    ok &= (g1, g2, g3, g4) == (-1, -1, -1, -1)
    print(f"shifts dim={dim}: -> {'PASS' if ok else 'FAIL'}")
    return ok


def run_kk_factor_case(dim, seed):
    """kk_raw = k * k_k parity, including exact-alias (in-place) use."""
    rng = np.random.default_rng(seed)
    k = (rng.standard_normal(dim) * 0.5).astype(np.float32)
    k_k = (1.0 + rng.standard_normal(dim) * 0.05).astype(np.float32)
    out = np.zeros(dim, dtype=np.float32)
    assert lib.ck_rwkv7_apply_kk_factor(
        numpy_to_ptr(k), numpy_to_ptr(k_k), numpy_to_ptr(out), dim) == 0
    ref = k * k_k
    d = float(np.max(np.abs(out - ref)))
    alias = k.copy()
    assert lib.ck_rwkv7_apply_kk_factor(
        numpy_to_ptr(alias), numpy_to_ptr(k_k), numpy_to_ptr(alias), dim) == 0
    da = float(np.max(np.abs(alias - ref)))
    print(f"kk_factor dim={dim}: out={d:.2e} alias={da:.2e}")
    ok = d == 0.0 and da == 0.0
    print("  ->", "PASS" if ok else "FAIL")
    return ok


def run_timemix_tail_case(H, N, Rd, Ra, Rg, Rv, seed, layer0):
    """Full tail parity: lora -> vmix -> kk_factor -> norm -> scale -> wkv."""
    rng = np.random.default_rng(seed)
    C = H * N

    def randn(*shape, scale=0.5):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    r = randn(C)
    k_raw = randn(C)
    v_in = randn(C)
    v_first = randn(C)
    xw, xa, xg = randn(C), randn(C), randn(C)
    xv = None if layer0 else randn(C)
    w1, w2 = randn(C, Rd, scale=0.4), randn(Rd, C, scale=0.2)
    w0 = randn(C, scale=0.3)
    a1, a2 = randn(C, Ra, scale=0.4), randn(Ra, C, scale=0.2)
    a0 = randn(C, scale=0.3)
    g1, g2 = randn(C, Rg, scale=0.4), randn(Rg, C, scale=0.2)
    v1, v2 = (None, None) if layer0 else (randn(C, Rv, scale=0.4), randn(Rv, C, scale=0.2))
    v0 = None if layer0 else randn(C, scale=0.3)
    k_k = (1.0 + randn(C, scale=0.05)).astype(np.float32)
    k_a = (np.full(C, 1.02, dtype=np.float32) + randn(C, scale=0.02)).astype(np.float32)
    rk = np.full((H, N), -0.04, dtype=np.float32)
    ln_w = np.ones(C, dtype=np.float32)
    ln_b = np.zeros(C, dtype=np.float32)
    state = randn(H, N, N, scale=0.25)

    assert lib.ck_rwkv7_time_mix_scratch_elems(C) == 7 * C
    assert lib.ck_rwkv7_time_mix_scratch_elems(0) == 0
    scratch = np.zeros(7 * C, dtype=np.float32)
    state_out = np.zeros_like(state)
    y_out = np.zeros(C, dtype=np.float32)
    vf_out = np.zeros(C, dtype=np.float32)
    p = numpy_to_ptr
    rc = lib.ck_rwkv7_time_mix_tail(
        p(r), p(k_raw), p(v_in), None if layer0 else p(v_first),
        p(xw), p(xa), p(xg), None if layer0 else p(xv),
        p(w1), p(w2), p(w0), p(a1), p(a2), p(a0), p(g1), p(g2),
        None if layer0 else p(v1), None if layer0 else p(v2), None if layer0 else p(v0),
        p(k_k), p(k_a), p(rk), p(ln_w), p(ln_b), ctypes.c_float(NORM_EPS),
        p(state), p(state_out), p(y_out), p(vf_out),
        C, H, N, Rd, Ra, Rg, 1 if layer0 else Rv,
        p(scratch), 7 * C)
    assert rc == 0

    t = lambda x: torch.from_numpy(x.copy())
    t_w = -INV_SQRT_E * torch.sigmoid(torch.tanh(t(xw) @ t(w1)) @ t(w2) + t(w0))
    t_a = torch.sigmoid((t(xa) @ t(a1)) @ t(a2) + t(a0))
    t_g = torch.sigmoid(t(xg) @ t(g1)) @ t(g2)
    if layer0:
        t_v = t(v_in)
        t_vf = t(v_in)
    else:
        t_gate = torch.sigmoid((t(xv) @ t(v1)) @ t(v2) + t(v0))
        t_v = t(v_in) + (t(v_first) - t(v_in)) * t_gate
        t_vf = t(v_first)
    t_kk = (t(k_raw) * t(k_k)).view(H, N)
    t_kk = t_kk / torch.clamp(torch.sqrt((t_kk * t_kk).sum(dim=1, keepdim=True)), min=1e-12)
    t_ks = (t(k_raw) * (1 + (t(k_a) - 1) * t_a)).view(H, N)
    t_s, t_y = torch_step(t(r).view(H, N), t_w.view(H, N), t_ks,
                          t_v.view(H, N), t_kk, t_a.view(H, N),
                          t_g, t(rk), t(ln_w), t(ln_b), t(state))
    dy = float(torch.max(torch.abs(torch.from_numpy(y_out.copy()) - t_y)).item())
    ds = float(torch.max(torch.abs(torch.from_numpy(state_out.copy()) - t_s)).item())
    dvf = float(torch.max(torch.abs(torch.from_numpy(vf_out.copy()) - t_vf)).item())

    # Guards: short scratch, null scratch, dim mismatch must fail.
    short = lib.ck_rwkv7_time_mix_tail(
        p(r), p(k_raw), p(v_in), None if layer0 else p(v_first),
        p(xw), p(xa), p(xg), None if layer0 else p(xv),
        p(w1), p(w2), p(w0), p(a1), p(a2), p(a0), p(g1), p(g2),
        None if layer0 else p(v1), None if layer0 else p(v2), None if layer0 else p(v0),
        p(k_k), p(k_a), p(rk), p(ln_w), p(ln_b), ctypes.c_float(NORM_EPS),
        p(state), p(state_out), p(y_out), p(vf_out),
        C, H, N, Rd, Ra, Rg, 1 if layer0 else Rv,
        p(scratch), 7 * C - 1)
    noscratch = lib.ck_rwkv7_time_mix_tail(
        p(r), p(k_raw), p(v_in), None if layer0 else p(v_first),
        p(xw), p(xa), p(xg), None if layer0 else p(xv),
        p(w1), p(w2), p(w0), p(a1), p(a2), p(a0), p(g1), p(g2),
        None if layer0 else p(v1), None if layer0 else p(v2), None if layer0 else p(v0),
        p(k_k), p(k_a), p(rk), p(ln_w), p(ln_b), ctypes.c_float(NORM_EPS),
        p(state), p(state_out), p(y_out), p(vf_out),
        C, H, N, Rd, Ra, Rg, 1 if layer0 else Rv, None, 0)
    bad_dim = lib.ck_rwkv7_time_mix_tail(
        p(r), p(k_raw), p(v_in), None if layer0 else p(v_first),
        p(xw), p(xa), p(xg), None if layer0 else p(xv),
        p(w1), p(w2), p(w0), p(a1), p(a2), p(a0), p(g1), p(g2),
        None if layer0 else p(v1), None if layer0 else p(v2), None if layer0 else p(v0),
        p(k_k), p(k_a), p(rk), p(ln_w), p(ln_b), ctypes.c_float(NORM_EPS),
        p(state), p(state_out), p(y_out), p(vf_out),
        C + 1, H, N, Rd, Ra, Rg, 1 if layer0 else Rv,
        p(scratch), 7 * C)
    # Layer-0 strictness: stale residual v inputs with xv == NULL must fail.
    if layer0:
        stale = np.zeros(C, dtype=np.float32)
        stale_l0 = lib.ck_rwkv7_time_mix_tail(
            p(r), p(k_raw), p(v_in), p(stale),
            p(xw), p(xa), p(xg), None,
            p(w1), p(w2), p(w0), p(a1), p(a2), p(a0), p(g1), p(g2),
            p(stale), p(stale), p(stale),
            p(k_k), p(k_a), p(rk), p(ln_w), p(ln_b), ctypes.c_float(NORM_EPS),
            p(state), p(state_out), p(y_out), p(vf_out),
            C, H, N, Rd, Ra, Rg, 1,
            p(scratch), 7 * C)
    else:
        stale_l0 = -1
    print(f"tail H={H} N={N} layer0={int(layer0)}: y={dy:.2e} s={ds:.2e} vf={dvf:.2e} "
          f"guards={short},{noscratch},{bad_dim},{stale_l0}")
    ok = dy <= 5e-5 and ds <= 2e-5 and dvf <= 2e-6 \
        and (short, noscratch, bad_dim, stale_l0) == (-1, -1, -1, -1)
    print("  ->", "PASS" if ok else "FAIL")
    return ok


def run_alias_guards():
    # Invalid aliases must fail without changing any caller-owned storage.
    C, K = 4, 6
    x = np.arange(C, dtype=np.float32)
    shift = x + 10
    mu = np.full(C, 0.25, dtype=np.float32)
    arena = np.full(64, 91, dtype=np.float32)
    before = arena.copy()
    p = numpy_to_ptr
    for second in (arena[:C], arena[1:C+1]):
        assert lib.ck_rwkv7_token_shift_lerp(p(x), p(shift), p(mu),
                                            p(arena[:C]), p(second), C) == -1
        np.testing.assert_array_equal(arena, before)
    assert lib.ck_rwkv7_token_shift_lerp(p(arena[:C]), p(shift), p(mu),
                                        p(arena[1:C+1]), p(arena[12:12+C]), C) == -1
    np.testing.assert_array_equal(arena, before)
    # Supported exact input alias retains the original current token in shift.
    current = x.copy()
    carried = np.zeros_like(x)
    assert lib.ck_rwkv7_token_shift_lerp(p(current), p(shift), p(mu),
                                        p(current), p(carried), C) == 0
    np.testing.assert_allclose(current, x + mu * (shift-x))
    np.testing.assert_array_equal(carried, x)
    wk = np.ones((K, C), dtype=np.float32)
    wv = np.ones((C, K), dtype=np.float32)
    for offset in (0, 1):
        assert lib.ck_rwkv7_channelmix_decode(p(x), p(shift), p(mu), p(wk), p(wv),
            p(arena[:C]), p(arena[offset:offset+C]), C, K, p(arena[20:]), C+K) == -1
        np.testing.assert_array_equal(arena, before)
    assert lib.ck_rwkv7_channelmix_decode(p(x), p(shift), p(mu), p(wk), p(wv),
        p(arena[:C]), p(arena[12:12+C]), C, K, p(arena[1:]), C+K) == -1
    np.testing.assert_array_equal(arena, before)
    print("Alias rejection and unchanged-output guards: PASS")
    for heads, width in ((257, 1), (1, 513), (0, 4)):
        assert lib.ck_rwkv7_norm_kk(p(x), p(arena), heads, width) == -1
        np.testing.assert_array_equal(arena, before)
    for eps in (float('nan'), float('inf'), 0.0, -1.0):
        assert lib.ck_rwkv7_wkv_decode(
            p(x), p(x), p(x), p(x), p(x), p(x), p(x), p(x), p(x), p(x),
            ctypes.c_float(eps), p(arena), p(arena), p(arena[32:]), 1, 4) == -1
        np.testing.assert_array_equal(arena, before)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run_alias_guards()
    cfgs = [(2, 8, 7), (4, 8, 13)]
    if not args.quick:
        cfgs.append((4, 16, 29))
    ok = True
    for cfg in cfgs:
        ok &= run_case(*cfg)
    ok &= run_state_case(8, 201)
    ok &= run_state_case(1, 203)
    ok &= run_state_case(7, 204)
    if not args.quick:
        ok &= run_state_case(16, 202)
    ok &= run_shift_case(32, 401)
    ok &= run_shifts_case(32, 402)
    ok &= run_channelmix_case(32, 48, 301)
    if not args.quick:
        ok &= run_channelmix_case(64, 96, 302)
    ok &= run_lora_case(4, 8, 6, 5, 4, 3, 101)
    if not args.quick:
        ok &= run_lora_case(4, 16, 8, 6, 5, 4, 102)
    ok &= run_kk_factor_case(32, 501)
    ok &= run_timemix_tail_case(4, 8, 6, 5, 4, 3, 601, False)
    ok &= run_timemix_tail_case(4, 8, 6, 5, 4, 3, 602, True)
    if not args.quick:
        ok &= run_timemix_tail_case(4, 16, 8, 6, 5, 4, 603, False)
    z = np.zeros(4, dtype=np.float32)
    rc = lib.ck_rwkv7_wkv_decode(
        None, numpy_to_ptr(z), numpy_to_ptr(z), numpy_to_ptr(z),
        numpy_to_ptr(z), numpy_to_ptr(z), numpy_to_ptr(z), numpy_to_ptr(z),
        numpy_to_ptr(z), numpy_to_ptr(z), ctypes.c_float(NORM_EPS),
        numpy_to_ptr(z), numpy_to_ptr(z), numpy_to_ptr(z), 1, 4)
    print(f"null-guard rc={rc} (want -1)")
    ok &= (rc == -1)
    rc = lib.ck_rwkv7_norm_kk(None, numpy_to_ptr(z), 1, 4)
    ok &= (rc == -1)
    if not ok:
        raise SystemExit(1)
    print("All RWKV-7 decode parity checks passed.")


if __name__ == "__main__":
    main()
