import argparse
import ctypes
import math
import os

import numpy as np


def _cpu_flags():
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("flags"):
                    return set(line.split(":")[1].strip().split())
    except OSError:
        return set()
    return set()


def _maybe_force_safe_torch():
    # Force conservative ISA for PyTorch reference on older CPUs to avoid SIGILL.
    if os.environ.get("CK_TORCH_SAFE") == "0":
        return

    flags = _cpu_flags()
    if os.environ.get("CK_TORCH_SAFE") == "1" or "avx2" not in flags:
        isa = "AVX" if "avx" in flags else "SSE4_2"
        os.environ.setdefault("ATEN_CPU_CAPABILITY", "default")
        os.environ.setdefault("DNNL_MAX_CPU_ISA", isa)
        os.environ.setdefault("ONEDNN_MAX_CPU_ISA", isa)
        os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", isa)
        os.environ.setdefault("MKL_DEBUG_CPU_TYPE", "5")
        os.environ.setdefault("MKL_DISABLE_FAST_MM", "1")


_maybe_force_safe_torch()

import torch

from lib_loader import load_lib


lib = load_lib("libckernel_engine.so")


class CKLayerForwardParams(ctypes.Structure):
    _fields_ = [
        ("tokens", ctypes.c_int),
        ("embed_dim", ctypes.c_int),
        ("aligned_embed_dim", ctypes.c_int),
        ("num_heads", ctypes.c_int),
        ("num_kv_heads", ctypes.c_int),
        ("head_dim", ctypes.c_int),
        ("aligned_head_dim", ctypes.c_int),
        ("aligned_context_window", ctypes.c_int),
        ("intermediate_dim", ctypes.c_int),
        ("aligned_intermediate_dim", ctypes.c_int),
        ("eps", ctypes.c_float),
        ("rope_pos_offset", ctypes.c_int),
        ("input", ctypes.POINTER(ctypes.c_float)),
        ("ln1_gamma", ctypes.POINTER(ctypes.c_float)),
        ("ln2_gamma", ctypes.POINTER(ctypes.c_float)),
        ("rope_cos", ctypes.POINTER(ctypes.c_float)),
        ("rope_sin", ctypes.POINTER(ctypes.c_float)),
        ("wq", ctypes.POINTER(ctypes.c_float)),
        ("bq", ctypes.POINTER(ctypes.c_float)),
        ("wk", ctypes.POINTER(ctypes.c_float)),
        ("bk", ctypes.POINTER(ctypes.c_float)),
        ("wv", ctypes.POINTER(ctypes.c_float)),
        ("bv", ctypes.POINTER(ctypes.c_float)),
        ("wo", ctypes.POINTER(ctypes.c_float)),
        ("bo", ctypes.POINTER(ctypes.c_float)),
        ("w1", ctypes.POINTER(ctypes.c_float)),
        ("b1", ctypes.POINTER(ctypes.c_float)),
        ("w2", ctypes.POINTER(ctypes.c_float)),
        ("b2", ctypes.POINTER(ctypes.c_float)),
        ("ln1_out", ctypes.POINTER(ctypes.c_float)),
        ("ln1_rstd", ctypes.POINTER(ctypes.c_float)),
        ("q", ctypes.POINTER(ctypes.c_float)),
        ("k", ctypes.POINTER(ctypes.c_float)),
        ("v", ctypes.POINTER(ctypes.c_float)),
        ("scores", ctypes.POINTER(ctypes.c_float)),
        ("attn_out", ctypes.POINTER(ctypes.c_float)),
        ("proj_tmp", ctypes.POINTER(ctypes.c_float)),
        ("proj_scratch", ctypes.POINTER(ctypes.c_float)),
        ("residual1", ctypes.POINTER(ctypes.c_float)),
        ("ln2_out", ctypes.POINTER(ctypes.c_float)),
        ("ln2_rstd", ctypes.POINTER(ctypes.c_float)),
        ("fc1_out", ctypes.POINTER(ctypes.c_float)),
        ("swiglu_out", ctypes.POINTER(ctypes.c_float)),
        ("mlp_out", ctypes.POINTER(ctypes.c_float)),
        ("output", ctypes.POINTER(ctypes.c_float)),
    ]


lib.ck_layer_forward_rmsnorm_swiglu.argtypes = [
    ctypes.POINTER(CKLayerForwardParams)
]
lib.ck_layer_forward_rmsnorm_swiglu.restype = None
lib.ck_layer_forward_rmsnorm_swiglu_ref.argtypes = [
    ctypes.POINTER(CKLayerForwardParams)
]
lib.ck_layer_forward_rmsnorm_swiglu_ref.restype = None

try:
    lib.ck_set_strict_parity.argtypes = [ctypes.c_int]
    lib.ck_set_strict_parity.restype = None
except AttributeError:
    pass


def aligned_empty(shape, dtype=np.float32, align=64):
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    buf = np.empty(nbytes + align, dtype=np.uint8)
    offset = (-buf.ctypes.data) % align
    arr = buf[offset:offset + nbytes].view(dtype).reshape(shape)
    return arr


def ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def align_up(n, a):
    return (n + a - 1) // a * a


def rmsnorm_ref(x, gamma, eps):
    var = x.pow(2).mean(dim=-1, keepdim=True)
    rstd = (var + eps).rsqrt()
    return x * rstd * gamma


def rmsnorm_rstd_ref(x, eps):
    var = x.pow(2).mean(dim=-1)
    return (var + eps).rsqrt()


def rope_cache(head_dim, max_seq_len, base=10000.0):
    half_dim = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32) * 2.0 / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    cos_cache = torch.cos(angles)
    sin_cache = torch.sin(angles)
    return cos_cache, sin_cache


def rope_apply(x, cos_cache, sin_cache, pos_offset=0):
    T, D = x.shape
    half_dim = D // 2
    out = x.clone()
    for t in range(T):
        cos_row = cos_cache[pos_offset + t]
        sin_row = sin_cache[pos_offset + t]
        for i in range(half_dim):
            x0 = x[t, i]
            x1 = x[t, i + half_dim]
            c = cos_row[i]
            s = sin_row[i]
            out[t, i] = x0 * c - x1 * s
            out[t, i + half_dim] = x0 * s + x1 * c
    return out


def stage_diff(name, c_val, ref_val):
    diff = (c_val - ref_val).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    mean_abs = diff.mean().item() if diff.numel() else 0.0
    max_rel = (diff / (ref_val.abs() + 1e-9)).max().item() if diff.numel() else 0.0
    print(f"  {name:12s} max_abs={max_abs:.2e} mean_abs={mean_abs:.2e} max_rel={max_rel:.2e}")
    return max_abs


def maybe_dump_stages(dump_stages, stages):
    if not dump_stages:
        return
    print("Stage diffs:")
    for name, c_val, ref_val in stages:
        stage_diff(name, c_val, ref_val)


def run_layer_test(num_kv_heads,
                   use_rope=False,
                   rope_theta=10000.0,
                   dump_stages=False,
                   tokens=8,
                   embed_dim=32,
                   num_heads=4,
                   intermediate_dim=64,
                   eps=1e-5,
                   weight_std=0.02,
                   seed=0,
                   skip_c=False,
                   skip_ref=False,
                   strict_ref=False,
                   strict_parity_kernels=True,
                   tol=1e-3):
    torch.manual_seed(seed)
    T = tokens
    D = embed_dim
    head_dim = D // num_heads

    def randn_scaled(*shape):
        return torch.randn(*shape, dtype=torch.float32) * weight_std

    aligned_embed_dim = align_up(D, 16)
    aligned_head_dim = align_up(head_dim, 16)
    aligned_intermediate_dim = align_up(intermediate_dim, 16)
    aligned_context_window = align_up(T, 16)

    x = aligned_empty((T, aligned_embed_dim))
    x[:, :D] = torch.randn(T, D, dtype=torch.float32).numpy()
    x[:, D:] = 0.0

    ln1_gamma = aligned_empty((aligned_embed_dim,))
    ln2_gamma = aligned_empty((aligned_embed_dim,))
    ln1_gamma[:D] = (1.0 + randn_scaled(D)).numpy()
    ln2_gamma[:D] = (1.0 + randn_scaled(D)).numpy()
    ln1_gamma[D:] = 0.0
    ln2_gamma[D:] = 0.0

    wq = aligned_empty((num_heads, aligned_head_dim, aligned_embed_dim))
    wk = aligned_empty((num_kv_heads, aligned_head_dim, aligned_embed_dim))
    wv = aligned_empty((num_kv_heads, aligned_head_dim, aligned_embed_dim))
    bq = aligned_empty((num_heads, aligned_head_dim))
    bk = aligned_empty((num_kv_heads, aligned_head_dim))
    bv = aligned_empty((num_kv_heads, aligned_head_dim))
    wq.fill(0.0)
    wk.fill(0.0)
    wv.fill(0.0)
    bq.fill(0.0)
    bk.fill(0.0)
    bv.fill(0.0)

    wq_t = randn_scaled(num_heads, head_dim, D)
    wk_t = randn_scaled(num_kv_heads, head_dim, D)
    wv_t = randn_scaled(num_kv_heads, head_dim, D)
    bq_t = randn_scaled(num_heads, head_dim)
    bk_t = randn_scaled(num_kv_heads, head_dim)
    bv_t = randn_scaled(num_kv_heads, head_dim)

    wq[:, :head_dim, :D] = wq_t.numpy()
    wk[:, :head_dim, :D] = wk_t.numpy()
    wv[:, :head_dim, :D] = wv_t.numpy()
    bq[:, :head_dim] = bq_t.numpy()
    bk[:, :head_dim] = bk_t.numpy()
    bv[:, :head_dim] = bv_t.numpy()

    wo = aligned_empty((num_heads, aligned_embed_dim, aligned_head_dim))
    wo.fill(0.0)
    wo_t = randn_scaled(num_heads, D, head_dim)
    wo[:, :D, :head_dim] = wo_t.numpy()

    bo = aligned_empty((aligned_embed_dim,))
    bo[:D] = randn_scaled(D).numpy()
    bo[D:] = 0.0

    w1 = aligned_empty((2 * aligned_intermediate_dim, aligned_embed_dim))
    b1 = aligned_empty((2 * aligned_intermediate_dim,))
    w2 = aligned_empty((aligned_embed_dim, aligned_intermediate_dim))
    b2 = aligned_empty((aligned_embed_dim,))
    w1.fill(0.0)
    w2.fill(0.0)
    b1.fill(0.0)
    b2.fill(0.0)

    w1_t = randn_scaled(2 * intermediate_dim, D)
    b1_t = randn_scaled(2 * intermediate_dim)
    w2_t = randn_scaled(D, intermediate_dim)
    b2_t = randn_scaled(D)

    w1[: 2 * intermediate_dim, :D] = w1_t.numpy()
    b1[: 2 * intermediate_dim] = b1_t.numpy()
    w2[:D, :intermediate_dim] = w2_t.numpy()
    b2[:D] = b2_t.numpy()

    ln1_out = aligned_empty((T, aligned_embed_dim))
    ln1_rstd = aligned_empty((T,))
    q = aligned_empty((num_heads, T, aligned_head_dim))
    k = aligned_empty((num_kv_heads, T, aligned_head_dim))
    v = aligned_empty((num_kv_heads, T, aligned_head_dim))
    scores = aligned_empty((num_heads, aligned_context_window, aligned_context_window))
    attn_out = aligned_empty((num_heads, T, aligned_head_dim))
    proj_tmp = aligned_empty((T, aligned_embed_dim))
    proj_scratch = aligned_empty((T, aligned_embed_dim))
    residual1 = aligned_empty((T, aligned_embed_dim))
    ln2_out = aligned_empty((T, aligned_embed_dim))
    ln2_rstd = aligned_empty((T,))
    fc1_out = aligned_empty((T, 2 * aligned_intermediate_dim))
    swiglu_out = aligned_empty((T, aligned_intermediate_dim))
    mlp_out = aligned_empty((T, aligned_embed_dim))
    output = aligned_empty((T, aligned_embed_dim))

    if strict_ref:
        ln1_out_ref = aligned_empty((T, aligned_embed_dim))
        ln1_rstd_ref = aligned_empty((T,))
        q_ref = aligned_empty((num_heads, T, aligned_head_dim))
        k_ref = aligned_empty((num_kv_heads, T, aligned_head_dim))
        v_ref = aligned_empty((num_kv_heads, T, aligned_head_dim))
        scores_ref = aligned_empty((num_heads, aligned_context_window, aligned_context_window))
        attn_out_ref = aligned_empty((num_heads, T, aligned_head_dim))
        proj_tmp_ref = aligned_empty((T, aligned_embed_dim))
        proj_scratch_ref = aligned_empty((T, aligned_embed_dim))
        residual1_ref = aligned_empty((T, aligned_embed_dim))
        ln2_out_ref = aligned_empty((T, aligned_embed_dim))
        ln2_rstd_ref = aligned_empty((T,))
        fc1_out_ref = aligned_empty((T, 2 * aligned_intermediate_dim))
        swiglu_out_ref = aligned_empty((T, aligned_intermediate_dim))
        mlp_out_ref = aligned_empty((T, aligned_embed_dim))
        output_ref = aligned_empty((T, aligned_embed_dim))

    rope_cos_ptr = ctypes.POINTER(ctypes.c_float)()
    rope_sin_ptr = ctypes.POINTER(ctypes.c_float)()
    rope_offset = 0
    cos_cache = None
    sin_cache = None
    if use_rope:
        cos_cache, sin_cache = rope_cache(head_dim, T, base=rope_theta)
        cos_np = cos_cache.numpy().astype(np.float32, copy=False)
        sin_np = sin_cache.numpy().astype(np.float32, copy=False)
        rope_cos_ptr = cos_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rope_sin_ptr = sin_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    if not skip_c:
        if hasattr(lib, "ck_set_strict_parity"):
            lib.ck_set_strict_parity(1 if strict_parity_kernels else 0)
        params = CKLayerForwardParams(
            tokens=T,
            embed_dim=D,
            aligned_embed_dim=aligned_embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            aligned_head_dim=aligned_head_dim,
            aligned_context_window=aligned_context_window,
            intermediate_dim=intermediate_dim,
            aligned_intermediate_dim=aligned_intermediate_dim,
            eps=eps,
            rope_pos_offset=rope_offset,
            input=ptr(x),
            ln1_gamma=ptr(ln1_gamma),
            ln2_gamma=ptr(ln2_gamma),
            rope_cos=rope_cos_ptr,
            rope_sin=rope_sin_ptr,
            wq=ptr(wq),
            bq=ptr(bq),
            wk=ptr(wk),
            bk=ptr(bk),
            wv=ptr(wv),
            bv=ptr(bv),
            wo=ptr(wo),
            bo=ptr(bo),
            w1=ptr(w1),
            b1=ptr(b1),
            w2=ptr(w2),
            b2=ptr(b2),
            ln1_out=ptr(ln1_out),
            ln1_rstd=ptr(ln1_rstd),
            q=ptr(q),
            k=ptr(k),
            v=ptr(v),
            scores=ptr(scores),
            attn_out=ptr(attn_out),
            proj_tmp=ptr(proj_tmp),
            proj_scratch=ptr(proj_scratch),
            residual1=ptr(residual1),
            ln2_out=ptr(ln2_out),
            ln2_rstd=ptr(ln2_rstd),
            fc1_out=ptr(fc1_out),
            swiglu_out=ptr(swiglu_out),
            mlp_out=ptr(mlp_out),
            output=ptr(output),
        )

        lib.ck_layer_forward_rmsnorm_swiglu(ctypes.byref(params))

    if skip_ref:
        out_c = torch.from_numpy(output[:, :D]).float()
        print(f"Layer forward (kv_heads={num_kv_heads}, rope={use_rope}) C-only stats: max={out_c.abs().max().item():.2e}")
        return

    if strict_ref:
        params_ref = CKLayerForwardParams(
            tokens=T,
            embed_dim=D,
            aligned_embed_dim=aligned_embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            aligned_head_dim=aligned_head_dim,
            aligned_context_window=aligned_context_window,
            intermediate_dim=intermediate_dim,
            aligned_intermediate_dim=aligned_intermediate_dim,
            eps=eps,
            rope_pos_offset=rope_offset,
            input=ptr(x),
            ln1_gamma=ptr(ln1_gamma),
            ln2_gamma=ptr(ln2_gamma),
            rope_cos=rope_cos_ptr,
            rope_sin=rope_sin_ptr,
            wq=ptr(wq),
            bq=ptr(bq),
            wk=ptr(wk),
            bk=ptr(bk),
            wv=ptr(wv),
            bv=ptr(bv),
            wo=ptr(wo),
            bo=ptr(bo),
            w1=ptr(w1),
            b1=ptr(b1),
            w2=ptr(w2),
            b2=ptr(b2),
            ln1_out=ptr(ln1_out_ref),
            ln1_rstd=ptr(ln1_rstd_ref),
            q=ptr(q_ref),
            k=ptr(k_ref),
            v=ptr(v_ref),
            scores=ptr(scores_ref),
            attn_out=ptr(attn_out_ref),
            proj_tmp=ptr(proj_tmp_ref),
            proj_scratch=ptr(proj_scratch_ref),
            residual1=ptr(residual1_ref),
            ln2_out=ptr(ln2_out_ref),
            ln2_rstd=ptr(ln2_rstd_ref),
            fc1_out=ptr(fc1_out_ref),
            swiglu_out=ptr(swiglu_out_ref),
            mlp_out=ptr(mlp_out_ref),
            output=ptr(output_ref),
        )

        lib.ck_layer_forward_rmsnorm_swiglu_ref(ctypes.byref(params_ref))

        if skip_c:
            out_ref = torch.from_numpy(output_ref[:, :D]).float()
            print(f"Layer forward (kv_heads={num_kv_heads}, rope={use_rope}) ref-only stats: max={out_ref.abs().max().item():.2e}")
            return

        ref_out = torch.from_numpy(output_ref[:, :D]).float()
        out_c = torch.from_numpy(output[:, :D]).float()
        diff = (out_c - ref_out).abs().max().item()

        stages = [
            ("ln1_out", torch.from_numpy(ln1_out[:, :D]).float(), torch.from_numpy(ln1_out_ref[:, :D]).float()),
            ("ln1_rstd", torch.from_numpy(ln1_rstd[:T]).float(), torch.from_numpy(ln1_rstd_ref[:T]).float()),
            ("q", torch.from_numpy(q[:, :, :head_dim]).float(), torch.from_numpy(q_ref[:, :, :head_dim]).float()),
            ("k", torch.from_numpy(k[:, :, :head_dim]).float(), torch.from_numpy(k_ref[:, :, :head_dim]).float()),
            ("v", torch.from_numpy(v[:, :, :head_dim]).float(), torch.from_numpy(v_ref[:, :, :head_dim]).float()),
            ("scores", torch.from_numpy(scores[:, :T, :T]).float(), torch.from_numpy(scores_ref[:, :T, :T]).float()),
            ("attn_out", torch.from_numpy(attn_out[:, :, :head_dim]).float(), torch.from_numpy(attn_out_ref[:, :, :head_dim]).float()),
            ("proj", torch.from_numpy(proj_tmp[:, :D]).float(), torch.from_numpy(proj_tmp_ref[:, :D]).float()),
            ("residual1", torch.from_numpy(residual1[:, :D]).float(), torch.from_numpy(residual1_ref[:, :D]).float()),
            ("ln2_out", torch.from_numpy(ln2_out[:, :D]).float(), torch.from_numpy(ln2_out_ref[:, :D]).float()),
            ("ln2_rstd", torch.from_numpy(ln2_rstd[:T]).float(), torch.from_numpy(ln2_rstd_ref[:T]).float()),
            ("fc1_out", torch.from_numpy(fc1_out[:, : 2 * intermediate_dim]).float(),
             torch.from_numpy(fc1_out_ref[:, : 2 * intermediate_dim]).float()),
            ("swiglu_out", torch.from_numpy(swiglu_out[:, :intermediate_dim]).float(),
             torch.from_numpy(swiglu_out_ref[:, :intermediate_dim]).float()),
            ("mlp_out", torch.from_numpy(mlp_out[:, :D]).float(), torch.from_numpy(mlp_out_ref[:, :D]).float()),
            ("output", out_c, ref_out),
        ]

        print(f"Layer forward (kv_heads={num_kv_heads}, rope={use_rope}, ref=c) max diff: {diff:.2e}")
        maybe_dump_stages(dump_stages, stages)
        if diff > tol:
            raise AssertionError(f"Layer forward mismatch: max diff {diff}")
        return

    x_ref = torch.from_numpy(x[:, :D]).float()
    ln1_gamma_ref = torch.from_numpy(ln1_gamma[:D]).float()
    ln2_gamma_ref = torch.from_numpy(ln2_gamma[:D]).float()
    bo_ref = torch.from_numpy(bo[:D]).float()
    b1_ref = torch.from_numpy(b1[: 2 * intermediate_dim]).float()
    b2_ref = torch.from_numpy(b2[:D]).float()

    h1 = rmsnorm_ref(x_ref, ln1_gamma_ref, eps)

    q_ref = torch.zeros(num_heads, T, head_dim, dtype=torch.float32)
    k_ref = torch.zeros(num_kv_heads, T, head_dim, dtype=torch.float32)
    v_ref = torch.zeros(num_kv_heads, T, head_dim, dtype=torch.float32)
    weights_ref = torch.zeros(num_heads, T, T, dtype=torch.float32)
    attn_ref = torch.zeros(num_heads, T, head_dim, dtype=torch.float32)

    for h in range(num_heads):
        kv_head = (h * num_kv_heads) // num_heads
        wq_h = torch.from_numpy(wq[h, :head_dim, :D]).float()
        wk_h = torch.from_numpy(wk[kv_head, :head_dim, :D]).float()
        wv_h = torch.from_numpy(wv[kv_head, :head_dim, :D]).float()
        bq_h = torch.from_numpy(bq[h, :head_dim]).float()
        bk_h = torch.from_numpy(bk[kv_head, :head_dim]).float()
        bv_h = torch.from_numpy(bv[kv_head, :head_dim]).float()

        q_h = h1 @ wq_h.t() + bq_h
        k_h = h1 @ wk_h.t() + bk_h
        v_h = h1 @ wv_h.t() + bv_h

        if use_rope:
            q_h = rope_apply(q_h, cos_cache, sin_cache, pos_offset=rope_offset)
            k_h = rope_apply(k_h, cos_cache, sin_cache, pos_offset=rope_offset)

        q_ref[h] = q_h
        k_ref[kv_head] = k_h
        v_ref[kv_head] = v_h

        scores_h = (q_h @ k_h.t()) / math.sqrt(head_dim)
        mask = torch.triu(torch.ones_like(scores_h), diagonal=1).bool()
        scores_h = scores_h.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores_h, dim=-1)
        weights_ref[h] = weights
        attn_ref[h] = weights @ v_h

    proj = torch.zeros(T, D, dtype=torch.float32)
    for h in range(num_heads):
        wo_h = torch.from_numpy(wo[h, :D, :head_dim]).float()
        proj += attn_ref[h] @ wo_h.t()
    proj += bo_ref

    res1 = x_ref + proj
    h2 = rmsnorm_ref(res1, ln2_gamma_ref, eps)

    w1_ref = torch.from_numpy(w1[: 2 * intermediate_dim, :D]).float()
    w2_ref = torch.from_numpy(w2[:D, :intermediate_dim]).float()
    up = h2 @ w1_ref.t() + b1_ref
    gate, value = up[:, :intermediate_dim], up[:, intermediate_dim:]
    swiglu = torch.nn.functional.silu(gate) * value
    mlp = swiglu @ w2_ref.t() + b2_ref
    ref_out = res1 + mlp

    if skip_c:
        print(f"Layer forward (kv_heads={num_kv_heads}, rope={use_rope}) ref-only stats: max={ref_out.abs().max().item():.2e}")
        return

    out_c = torch.from_numpy(output[:, :D]).float()
    diff = (out_c - ref_out).abs().max().item()

    stages = [
        ("ln1_out", torch.from_numpy(ln1_out[:, :D]).float(), h1),
        ("ln1_rstd", torch.from_numpy(ln1_rstd[:T]).float(), rmsnorm_rstd_ref(x_ref, eps)),
        ("q", torch.from_numpy(q[:, :, :head_dim]).float(), q_ref),
        ("k", torch.from_numpy(k[:, :, :head_dim]).float(), k_ref),
        ("v", torch.from_numpy(v[:, :, :head_dim]).float(), v_ref),
        ("scores", torch.from_numpy(scores[:, :T, :T]).float(), weights_ref),
        ("attn_out", torch.from_numpy(attn_out[:, :, :head_dim]).float(), attn_ref),
        ("proj", torch.from_numpy(proj_tmp[:, :D]).float(), proj),
        ("residual1", torch.from_numpy(residual1[:, :D]).float(), res1),
        ("ln2_out", torch.from_numpy(ln2_out[:, :D]).float(), h2),
        ("ln2_rstd", torch.from_numpy(ln2_rstd[:T]).float(), rmsnorm_rstd_ref(res1, eps)),
        ("fc1_out", torch.from_numpy(fc1_out[:, : 2 * intermediate_dim]).float(), up),
        ("swiglu_out", torch.from_numpy(swiglu_out[:, :intermediate_dim]).float(), swiglu),
        ("mlp_out", torch.from_numpy(mlp_out[:, :D]).float(), mlp),
        ("output", out_c, ref_out),
    ]

    print(f"Layer forward (kv_heads={num_kv_heads}, rope={use_rope}) max diff: {diff:.2e}")
    maybe_dump_stages(dump_stages, stages)
    # Blocked GEMM + GQA accumulation order introduces small float32 drift.
    if diff > tol:
        raise AssertionError(f"Layer forward mismatch: max diff {diff}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-stages", action="store_true", help="Print per-stage diffs.")
    parser.add_argument("--tokens", type=int, default=8, help="Context length (T).")
    parser.add_argument("--embed", type=int, default=32, help="Embedding dimension (D).")
    parser.add_argument("--heads", type=int, default=4, help="Attention heads (H).")
    parser.add_argument("--kv-heads", type=int, default=None, help="KV heads (defaults to --heads).")
    parser.add_argument("--intermediate", type=int, default=64, help="MLP intermediate dimension.")
    parser.add_argument("--eps", type=float, default=1e-5, help="RMSNorm epsilon.")
    parser.add_argument("--weight-std", type=float, default=0.02,
                        help="Stddev for synthetic weights/biases (default: 0.02, model-like).")
    parser.add_argument("--rope", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable or disable RoPE (default: off).")
    parser.add_argument("--rope-theta", type=float, default=10000.0, help="RoPE theta/base.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="Absolute tolerance for max diff.")
    parser.add_argument("--skip-c", action="store_true", help="Skip C kernel call (ref-only).")
    parser.add_argument("--skip-ref", action="store_true", help="Skip reference path (C-only).")
    parser.add_argument("--strict-ref", action="store_true",
                        help="Use C reference path (naive GEMM) instead of PyTorch.")
    parser.add_argument("--strict-parity-kernels", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable exact/strict kernel paths for PyTorch parity checks.")
    parser.add_argument("--all", action="store_true",
                        help="Run both RoPE on/off and KV head variants for the given dims.")
    args = parser.parse_args()
    dump = args.dump_stages or os.environ.get("CK_DUMP_STAGES") == "1"

    if args.skip_c and args.skip_ref:
        raise SystemExit("Choose only one of --skip-c or --skip-ref.")
    if args.strict_ref and args.skip_ref:
        raise SystemExit("Cannot combine --strict-ref with --skip-ref.")

    if args.embed % args.heads != 0:
        raise SystemExit("embed must be divisible by heads")

    if args.all:
        kv_variants = [args.heads, max(1, args.heads // 2)]
        for kv in kv_variants:
            if args.heads % kv != 0:
                continue
            run_layer_test(num_kv_heads=kv,
                           use_rope=False,
                           rope_theta=args.rope_theta,
                           dump_stages=dump,
                           tokens=args.tokens,
                           embed_dim=args.embed,
                           num_heads=args.heads,
                           intermediate_dim=args.intermediate,
                           eps=args.eps,
                           weight_std=args.weight_std,
                           seed=args.seed,
                           skip_c=args.skip_c,
                           skip_ref=args.skip_ref,
                           strict_ref=args.strict_ref,
                           strict_parity_kernels=args.strict_parity_kernels,
                           tol=args.tol)
            run_layer_test(num_kv_heads=kv,
                           use_rope=True,
                           rope_theta=args.rope_theta,
                           dump_stages=dump,
                           tokens=args.tokens,
                           embed_dim=args.embed,
                           num_heads=args.heads,
                           intermediate_dim=args.intermediate,
                           eps=args.eps,
                           weight_std=args.weight_std,
                           seed=args.seed,
                           skip_c=args.skip_c,
                           skip_ref=args.skip_ref,
                           strict_ref=args.strict_ref,
                           strict_parity_kernels=args.strict_parity_kernels,
                           tol=args.tol)
    else:
        kv_heads = args.kv_heads if args.kv_heads is not None else args.heads
        if args.heads % kv_heads != 0:
            raise SystemExit("heads must be divisible by kv-heads")
        use_rope = args.rope if args.rope is not None else False
        run_layer_test(num_kv_heads=kv_heads,
                       use_rope=use_rope,
                       rope_theta=args.rope_theta,
                       dump_stages=dump,
                       tokens=args.tokens,
                       embed_dim=args.embed,
                       num_heads=args.heads,
                       intermediate_dim=args.intermediate,
                       eps=args.eps,
                       weight_std=args.weight_std,
                       seed=args.seed,
                       skip_c=args.skip_c,
                       skip_ref=args.skip_ref,
                       strict_ref=args.strict_ref,
                       strict_parity_kernels=args.strict_parity_kernels,
                       tol=args.tol)
