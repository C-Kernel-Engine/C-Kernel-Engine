"""RWKV-7 T=1 FP32 decode parity vs modeling_rwkv7.py eager math (torch oracle)."""
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

INV_SQRT_E = 0.6065306597126334

NORM_EPS = 1e-5


def torch_step(r, w_log, k, v, kk, a, g, rk, ln_w, ln_b, state):
    """Mirror modeling_rwkv7.rwkv7_recurrent for one token (FP32, B=1)."""
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
    ref_k = k_raw * (1 + (a - 1) * k_a)
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
    """LoRA gates + vgate/vmix parity vs modeling_rwkv7.lora_gates, plus e2e."""
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
    t_ks = (t(k_raw) * (1 + (t_a - 1) * t(k_a))).view(H, N)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfgs = [(2, 8, 7), (4, 8, 13)]
    if not args.quick:
        cfgs.append((4, 16, 29))
    ok = True
    for cfg in cfgs:
        ok &= run_case(*cfg)
    ok &= run_lora_case(4, 8, 6, 5, 4, 3, 101)
    if not args.quick:
        ok &= run_lora_case(4, 16, 8, 6, 5, 4, 102)
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
