import ctypes
import os

import numpy as np
import torch

from lib_loader import load_lib
from test_utils import get_cpu_info, max_diff, print_system_info

# Enable verbose debug output with: DEBUG=1 python test_kv_cache_layer_decode.py
DEBUG = os.environ.get("DEBUG", "0") == "1"

# Keep decode parity strict by default; allow override via env if needed.
os.environ.setdefault("CK_FLASH_ATTN_STRICT", "1")


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
    ctypes.POINTER(CKLayerForwardParams),
]
lib.ck_layer_forward_rmsnorm_swiglu.restype = None

lib.ck_layer_forward_rmsnorm_swiglu_decode.argtypes = [
    ctypes.POINTER(CKLayerForwardParams),
    ctypes.c_int,
    ctypes.c_int,
]
lib.ck_layer_forward_rmsnorm_swiglu_decode.restype = None

lib.rope_precompute_cache.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
]
lib.rope_precompute_cache.restype = None

lib.kv_cache_repack_head_major_inplace.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.kv_cache_repack_head_major_inplace.restype = None


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


def run_decode_parity_test(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    cache_capacity = 16
    prompt_len = 8
    total_len = 12

    num_heads = 4
    num_kv_heads = 2
    embed_dim = 48
    head_dim = embed_dim // num_heads
    intermediate_dim = 50
    eps = 1e-5
    rope_theta = 10000.0

    aligned_embed_dim = align_up(embed_dim, 16)
    aligned_head_dim = align_up(head_dim, 16)
    aligned_intermediate_dim = align_up(intermediate_dim, 16)
    aligned_context_window_full = align_up(total_len, 16)
    aligned_context_window_prefill = align_up(prompt_len, 16)
    aligned_context_window_decode = align_up(cache_capacity, 16)

    x = aligned_empty((cache_capacity, aligned_embed_dim))
    x.fill(0.0)
    x[:total_len, :embed_dim] = torch.randn(total_len, embed_dim, dtype=torch.float32).numpy()

    ln1_gamma = aligned_empty((aligned_embed_dim,))
    ln2_gamma = aligned_empty((aligned_embed_dim,))
    ln1_gamma.fill(0.0)
    ln2_gamma.fill(0.0)
    ln1_gamma[:embed_dim] = torch.randn(embed_dim, dtype=torch.float32).numpy()
    ln2_gamma[:embed_dim] = torch.randn(embed_dim, dtype=torch.float32).numpy()

    wq = aligned_empty((num_heads, aligned_head_dim, aligned_embed_dim)); wq.fill(0.0)
    wk = aligned_empty((num_kv_heads, aligned_head_dim, aligned_embed_dim)); wk.fill(0.0)
    wv = aligned_empty((num_kv_heads, aligned_head_dim, aligned_embed_dim)); wv.fill(0.0)
    bq = aligned_empty((num_heads, aligned_head_dim)); bq.fill(0.0)
    bk = aligned_empty((num_kv_heads, aligned_head_dim)); bk.fill(0.0)
    bv = aligned_empty((num_kv_heads, aligned_head_dim)); bv.fill(0.0)

    wq[:, :head_dim, :embed_dim] = torch.randn(num_heads, head_dim, embed_dim, dtype=torch.float32).numpy()
    wk[:, :head_dim, :embed_dim] = torch.randn(num_kv_heads, head_dim, embed_dim, dtype=torch.float32).numpy()
    wv[:, :head_dim, :embed_dim] = torch.randn(num_kv_heads, head_dim, embed_dim, dtype=torch.float32).numpy()
    bq[:, :head_dim] = torch.randn(num_heads, head_dim, dtype=torch.float32).numpy()
    bk[:, :head_dim] = torch.randn(num_kv_heads, head_dim, dtype=torch.float32).numpy()
    bv[:, :head_dim] = torch.randn(num_kv_heads, head_dim, dtype=torch.float32).numpy()

    wo = aligned_empty((num_heads, aligned_embed_dim, aligned_head_dim)); wo.fill(0.0)
    wo[:, :embed_dim, :head_dim] = torch.randn(num_heads, embed_dim, head_dim, dtype=torch.float32).numpy()
    bo = aligned_empty((aligned_embed_dim,)); bo.fill(0.0)
    bo[:embed_dim] = torch.randn(embed_dim, dtype=torch.float32).numpy()

    w1 = aligned_empty((2 * aligned_intermediate_dim, aligned_embed_dim)); w1.fill(0.0)
    b1 = aligned_empty((2 * aligned_intermediate_dim,)); b1.fill(0.0)
    w2 = aligned_empty((aligned_embed_dim, aligned_intermediate_dim)); w2.fill(0.0)
    b2 = aligned_empty((aligned_embed_dim,)); b2.fill(0.0)

    w1[: 2 * intermediate_dim, :embed_dim] = torch.randn(2 * intermediate_dim, embed_dim, dtype=torch.float32).numpy()
    b1[: 2 * intermediate_dim] = torch.randn(2 * intermediate_dim, dtype=torch.float32).numpy()
    w2[:embed_dim, :intermediate_dim] = torch.randn(embed_dim, intermediate_dim, dtype=torch.float32).numpy()
    b2[:embed_dim] = torch.randn(embed_dim, dtype=torch.float32).numpy()

    rope_cos = aligned_empty((cache_capacity, head_dim // 2))
    rope_sin = aligned_empty((cache_capacity, head_dim // 2))
    lib.rope_precompute_cache(ptr(rope_cos), ptr(rope_sin),
                              ctypes.c_int(cache_capacity),
                              ctypes.c_int(head_dim),
                              ctypes.c_float(rope_theta))

    def alloc_buffers(token_slots, kv_tokens):
        return {
            "ln1_out": aligned_empty((token_slots, aligned_embed_dim)),
            "ln1_rstd": aligned_empty((token_slots,)),
            "q": aligned_empty((num_heads, token_slots, aligned_head_dim)),
            "k": aligned_empty((num_kv_heads, kv_tokens, aligned_head_dim)),
            "v": aligned_empty((num_kv_heads, kv_tokens, aligned_head_dim)),
            "attn_out": aligned_empty((num_heads, token_slots, aligned_head_dim)),
            "proj_tmp": aligned_empty((token_slots, aligned_embed_dim)),
            "proj_scratch": aligned_empty((token_slots, aligned_embed_dim)),
            "residual1": aligned_empty((token_slots, aligned_embed_dim)),
            "ln2_out": aligned_empty((token_slots, aligned_embed_dim)),
            "ln2_rstd": aligned_empty((token_slots,)),
            "fc1_out": aligned_empty((token_slots, 2 * aligned_intermediate_dim)),
            "swiglu_out": aligned_empty((token_slots, aligned_intermediate_dim)),
            "mlp_out": aligned_empty((token_slots, aligned_embed_dim)),
            "output": aligned_empty((token_slots, aligned_embed_dim)),
        }

    buf_full = alloc_buffers(total_len, total_len)
    buf_prefill = alloc_buffers(prompt_len, prompt_len)
    buf_decode = alloc_buffers(1, 1)

    def alloc_kv_cache():
        # Allocate cache and guard as separate contiguous arrays.
        # The cache MUST be contiguous because C code assumes:
        #   head_stride = cache_capacity * aligned_head_dim
        guard_value = 123.0
        cache = aligned_empty((num_kv_heads, cache_capacity, aligned_head_dim))
        cache.fill(0.0)
        guard = aligned_empty((num_kv_heads, 1, aligned_head_dim))
        guard.fill(guard_value)
        assert cache.flags['C_CONTIGUOUS'], "KV cache must be C-contiguous for C code"
        return cache, guard, guard_value

    k_cache, k_guard, k_guard_value = alloc_kv_cache()
    v_cache, v_guard, v_guard_value = alloc_kv_cache()

    for buf in (buf_full, buf_prefill, buf_decode):
        for arr in buf.values():
            arr.fill(0.0)

    p_full = CKLayerForwardParams(
        tokens=total_len,
        embed_dim=embed_dim,
        aligned_embed_dim=aligned_embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        aligned_head_dim=aligned_head_dim,
        aligned_context_window=aligned_context_window_full,
        intermediate_dim=intermediate_dim,
        aligned_intermediate_dim=aligned_intermediate_dim,
        eps=eps,
        rope_pos_offset=0,
        input=ptr(x),
        ln1_gamma=ptr(ln1_gamma),
        ln2_gamma=ptr(ln2_gamma),
        rope_cos=ptr(rope_cos),
        rope_sin=ptr(rope_sin),
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
        ln1_out=ptr(buf_full["ln1_out"]),
        ln1_rstd=ptr(buf_full["ln1_rstd"]),
        q=ptr(buf_full["q"]),
        k=ptr(buf_full["k"]),
        v=ptr(buf_full["v"]),
        scores=None,
        attn_out=ptr(buf_full["attn_out"]),
        proj_tmp=ptr(buf_full["proj_tmp"]),
        proj_scratch=ptr(buf_full["proj_scratch"]),
        residual1=ptr(buf_full["residual1"]),
        ln2_out=ptr(buf_full["ln2_out"]),
        ln2_rstd=ptr(buf_full["ln2_rstd"]),
        fc1_out=ptr(buf_full["fc1_out"]),
        swiglu_out=ptr(buf_full["swiglu_out"]),
        mlp_out=ptr(buf_full["mlp_out"]),
        output=ptr(buf_full["output"]),
    )
    lib.ck_layer_forward_rmsnorm_swiglu(ctypes.byref(p_full))

    p_prefill = CKLayerForwardParams(
        tokens=prompt_len,
        embed_dim=embed_dim,
        aligned_embed_dim=aligned_embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        aligned_head_dim=aligned_head_dim,
        aligned_context_window=aligned_context_window_prefill,
        intermediate_dim=intermediate_dim,
        aligned_intermediate_dim=aligned_intermediate_dim,
        eps=eps,
        rope_pos_offset=0,
        input=ptr(x),
        ln1_gamma=ptr(ln1_gamma),
        ln2_gamma=ptr(ln2_gamma),
        rope_cos=ptr(rope_cos),
        rope_sin=ptr(rope_sin),
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
        ln1_out=ptr(buf_prefill["ln1_out"]),
        ln1_rstd=ptr(buf_prefill["ln1_rstd"]),
        q=ptr(buf_prefill["q"]),
        k=ptr(buf_prefill["k"]),
        v=ptr(buf_prefill["v"]),
        scores=None,
        attn_out=ptr(buf_prefill["attn_out"]),
        proj_tmp=ptr(buf_prefill["proj_tmp"]),
        proj_scratch=ptr(buf_prefill["proj_scratch"]),
        residual1=ptr(buf_prefill["residual1"]),
        ln2_out=ptr(buf_prefill["ln2_out"]),
        ln2_rstd=ptr(buf_prefill["ln2_rstd"]),
        fc1_out=ptr(buf_prefill["fc1_out"]),
        swiglu_out=ptr(buf_prefill["swiglu_out"]),
        mlp_out=ptr(buf_prefill["mlp_out"]),
        output=ptr(buf_prefill["output"]),
    )

    # Prefill for prompt tokens (fills KV cache for positions < prompt_len).
    lib.ck_layer_forward_rmsnorm_swiglu(ctypes.byref(p_prefill))
    # Copy K/V from prefill buffer to cache, head by head (preserves layout).
    for h in range(num_kv_heads):
        k_cache[h, :prompt_len, :] = buf_prefill["k"][h, :, :]
        v_cache[h, :prompt_len, :] = buf_prefill["v"][h, :, :]

    token_input = aligned_empty((1, aligned_embed_dim))
    token_input.fill(0.0)
    decode_outputs = aligned_empty((total_len, aligned_embed_dim))
    decode_outputs.fill(0.0)
    decode_outputs[:prompt_len] = buf_prefill["output"][:prompt_len]

    p_decode = CKLayerForwardParams(
        tokens=1,
        embed_dim=embed_dim,
        aligned_embed_dim=aligned_embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        aligned_head_dim=aligned_head_dim,
        aligned_context_window=aligned_context_window_decode,
        intermediate_dim=intermediate_dim,
        aligned_intermediate_dim=aligned_intermediate_dim,
        eps=eps,
        rope_pos_offset=0,
        input=ptr(token_input),
        ln1_gamma=ptr(ln1_gamma),
        ln2_gamma=ptr(ln2_gamma),
        rope_cos=ptr(rope_cos),
        rope_sin=ptr(rope_sin),
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
        ln1_out=ptr(buf_decode["ln1_out"]),
        ln1_rstd=ptr(buf_decode["ln1_rstd"]),
        q=ptr(buf_decode["q"]),
        k=ptr(k_cache),
        v=ptr(v_cache),
        scores=None,
        attn_out=ptr(buf_decode["attn_out"]),
        proj_tmp=ptr(buf_decode["proj_tmp"]),
        proj_scratch=ptr(buf_decode["proj_scratch"]),
        residual1=ptr(buf_decode["residual1"]),
        ln2_out=ptr(buf_decode["ln2_out"]),
        ln2_rstd=ptr(buf_decode["ln2_rstd"]),
        fc1_out=ptr(buf_decode["fc1_out"]),
        swiglu_out=ptr(buf_decode["swiglu_out"]),
        mlp_out=ptr(buf_decode["mlp_out"]),
        output=ptr(buf_decode["output"]),
    )

    # Debug: Verify cache layout before decode loop
    if DEBUG:
        print("\n=== DEBUG: Cache layout verification ===")
        print(f"k_cache shape: {k_cache.shape}, strides: {k_cache.strides}")
        print(f"C_CONTIGUOUS: {k_cache.flags['C_CONTIGUOUS']}")
        expected_strides = (cache_capacity * aligned_head_dim * 4, aligned_head_dim * 4, 4)
        print(f"Expected strides: {expected_strides}")
        if k_cache.strides != expected_strides:
            print("WARNING: Cache strides mismatch - C code will use wrong offsets!")

        print("\n=== DEBUG: K values for decode positions (from full pass) ===")
        for t in range(prompt_len, total_len):
            print(f"Full K[1,{t},5] = {buf_full['k'][1, t, 5]:.4f}")

        print("\n=== DEBUG: K cache before decode (positions 0 to prompt_len-1) ===")
        for t_check in range(prompt_len):
            full_k_t = buf_full["k"][:, t_check, :head_dim]
            cache_k_t = k_cache[:, t_check, :head_dim]
            diff_t = np.max(np.abs(full_k_t - cache_k_t))
            status = "MATCH" if diff_t < 1e-5 else "MISMATCH"
            print(f"K position {t_check}: max_diff = {diff_t:.2e} [{status}]")

    # Decode remaining tokens one-by-one, updating KV cache.
    for t in range(prompt_len, total_len):
        # Debug: Track cache corruption during decode
        if DEBUG:
            k_pos_last = k_cache[1, prompt_len - 1, 5].copy()

        token_input.fill(0.0)
        token_input[0, :embed_dim] = x[t, :embed_dim]
        p_decode.rope_pos_offset = t
        lib.ck_layer_forward_rmsnorm_swiglu_decode(ctypes.byref(p_decode),
                                                   ctypes.c_int(t),
                                                   ctypes.c_int(cache_capacity))
        decode_outputs[t] = buf_decode["output"][0]

        # Debug: Check if previous positions got corrupted
        if DEBUG:
            k_pos_last_after = k_cache[1, prompt_len - 1, 5]
            if abs(k_pos_last - k_pos_last_after) > 1e-6:
                print(f"!!! K[1,{prompt_len-1},5] CORRUPTED during decode of token {t}:")
                print(f"    Before: {k_pos_last}, After: {k_pos_last_after}")
                print(f"    (Expected K for pos {t}: {buf_full['k'][1, t, 5]:.4f})")

        if head_dim < aligned_head_dim:
            k_slice = k_cache[:, t, head_dim:aligned_head_dim]
            v_slice = v_cache[:, t, head_dim:aligned_head_dim]
            if np.any(k_slice != 0.0) or np.any(v_slice != 0.0):
                raise AssertionError("KV cache padded lanes are not zeroed")

    # Debug: Per-token output comparison
    if DEBUG:
        print("\n=== DEBUG: Per-token output comparison ===")
        for t in range(total_len):
            full_t = buf_full["output"][t, :4]
            dec_t = decode_outputs[t, :4]
            diff_t = np.max(np.abs(buf_full["output"][t, :embed_dim] - decode_outputs[t, :embed_dim]))
            status = "MATCH" if diff_t < 1e-5 else "MISMATCH"
            print(f"Token {t:2d}: diff={diff_t:.2e} [{status}]")
            if status == "MISMATCH":
                print(f"    full={full_t}")
                print(f"    decode={dec_t}")

    out_full = torch.from_numpy(buf_full["output"][:total_len])
    out_decode = torch.from_numpy(decode_outputs[:total_len])

    diff = max_diff(out_full, out_decode)
    return diff


if __name__ == "__main__":
    print_system_info()
    diff = run_decode_parity_test(seed=0)
    tol = float(os.environ.get("CK_KV_DECODE_TOL", "1e-5"))
    print("\n================================================================================")
    print("  TEST: KV-cache Layer Decode Parity")
    print("================================================================================\n")
    print("  ACCURACY")
    print("  ----------------------------------------")
    print(f"  output  max_diff={diff:.2e}  tol={tol:.0e}  [{'PASS' if diff <= tol else 'FAIL'}]\n")

    if diff > tol:
        raise SystemExit(1)
