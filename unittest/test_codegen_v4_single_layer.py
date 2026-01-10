"""
IR v4 codegen sanity: generate a 1-layer model, run C forward, compare to PyTorch.

This is a small, deterministic test that exercises the v4 codegen path end-to-end:
  config -> IR v4 -> generated C -> shared lib -> forward() vs PyTorch reference.
"""
import ctypes
import json
import math
import os
import subprocess
import sys

import numpy as np
import torch

from lib_loader import find_lib
from test_utils import TestReport, TestResult, get_cpu_info, max_diff, print_system_info


ROOT = os.path.dirname(os.path.dirname(__file__))
SCRIPTS_V4_DIR = os.path.join(ROOT, "scripts", "v4")
if SCRIPTS_V4_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_V4_DIR)
BUILD_DIR = os.path.join(ROOT, "build")
CONFIG_PATH = os.path.join(ROOT, "unittest", "fixtures", "mini_v4.config.json")


def align_elems(elems, elem_bytes=4, align_bytes=64):
    total_bytes = elems * elem_bytes
    aligned_bytes = ((total_bytes + align_bytes - 1) // align_bytes) * align_bytes
    return aligned_bytes // elem_bytes


def precompute_rope(theta, max_seq_len, head_dim):
    half = head_dim // 2
    cos = np.zeros((max_seq_len, half), dtype=np.float32)
    sin = np.zeros_like(cos)
    for pos in range(max_seq_len):
        for i in range(half):
            freq = 1.0 / (theta ** ((2.0 * i) / (half * 2.0)))
            angle = pos * freq
            cos[pos, i] = math.cos(angle)
            sin[pos, i] = math.sin(angle)
    return cos, sin


def apply_rope(x, cos, sin):
    # x: [H, T, D]
    h, t, d = x.shape
    half = d // 2
    cos_t = torch.from_numpy(cos[:t]).to(x)
    sin_t = torch.from_numpy(sin[:t]).to(x)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos_b = cos_t.unsqueeze(0)
    sin_b = sin_t.unsqueeze(0)
    x[..., 0::2] = x1 * cos_b - x2 * sin_b
    x[..., 1::2] = x1 * sin_b + x2 * cos_b
    return x


def attention_gqa(q, k, v, num_heads, num_kv_heads):
    # q: [H, T, D], k/v: [KV, T, D]
    h, t, d = q.shape
    out = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(d)
    for head in range(num_heads):
        kv_head = (head * num_kv_heads) // num_heads
        qh = q[head]
        kh = k[kv_head]
        vh = v[kv_head]
        scores = (qh @ kh.T) * scale
        mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[head] = probs @ vh
    return out


def rmsnorm(x, gamma, eps):
    # x: [T, E]
    mean_sq = (x * x).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(mean_sq + eps)
    return x * rstd * gamma


def parse_offset(value):
    if isinstance(value, str):
        return int(value, 16)
    return int(value)


def collect_buffers(section):
    buf_map = {}
    for buf in section["header"]["buffers"]:
        buf_map[buf["name"]] = buf
    for buf in section.get("globals", []):
        buf_map[buf["name"]] = buf
    for layer in section["layers"]:
        for buf in layer["buffers"]:
            buf_map[buf["name"]] = buf
    for buf in section["footer"]["buffers"]:
        buf_map[buf["name"]] = buf
    return buf_map


def write_buffer(mem, offset, arr):
    flat = arr.astype(np.float32, copy=False).ravel()
    size = flat.nbytes
    view = mem[offset:offset + size]
    view_f = view.view(np.float32)
    view_f[:] = flat


def read_buffer(mem, offset, shape):
    size = int(np.prod(shape)) * 4
    view = mem[offset:offset + size]
    view_f = view.view(np.float32)
    return view_f.reshape(shape).copy()


def build_v4_runtime():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import build_ir_v4 as v4

    output_dir = os.path.join(BUILD_DIR, "mini_v4_test")
    args = [
        f"--config={CONFIG_PATH}",
        "--name=mini_v4",
        f"--prefix={output_dir}",
        "--tokens=4",
        "--modes=prefill",
        "--fusion=off",
        "--parallel=off",
        "--dtype=fp32",
    ]
    rc = v4.main(args)
    if rc != 0:
        raise RuntimeError("build_ir_v4 failed")

    generated_c = os.path.join(output_dir, "generated_mini_v4_prefill.c")
    layout_json = os.path.join(output_dir, "layout_prefill.json")
    if not os.path.exists(generated_c):
        raise FileNotFoundError(generated_c)
    if not os.path.exists(layout_json):
        raise FileNotFoundError(layout_json)

    libck = find_lib("libckernel_engine.so")
    if not os.path.exists(libck):
        raise FileNotFoundError(libck)

    out_lib = os.path.join(output_dir, "libgenerated_mini_v4_prefill.so")
    cmd = [
        "gcc", "-shared", "-fPIC", "-O2",
        "-I", os.path.join(ROOT, "include"),
        generated_c,
        "-L", BUILD_DIR,
        "-lckernel_engine",
        "-lm",
        "-o", out_lib,
    ]
    subprocess.run(cmd, check=True)

    return out_lib, layout_json


def main():
    print_system_info()

    out_lib, layout_json = build_v4_runtime()

    os.environ["LD_LIBRARY_PATH"] = f"{BUILD_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    lib = ctypes.cdll.LoadLibrary(out_lib)

    class Model(ctypes.Structure):
        _fields_ = [
            ("base", ctypes.c_void_p),
            ("total_bytes", ctypes.c_size_t),
        ]

    model = Model()
    lib.mini_v4_prefill_model_allocate.argtypes = [ctypes.POINTER(Model)]
    lib.mini_v4_prefill_model_allocate.restype = ctypes.c_int
    lib.mini_v4_prefill_model_free.argtypes = [ctypes.POINTER(Model)]
    lib.mini_v4_prefill_model_free.restype = None
    lib.mini_v4_prefill_forward.argtypes = [
        ctypes.POINTER(Model),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    lib.mini_v4_prefill_forward.restype = None

    if lib.mini_v4_prefill_model_allocate(ctypes.byref(model)) != 0:
        raise RuntimeError("model_allocate failed")

    try:
        base_ptr = ctypes.cast(model.base, ctypes.POINTER(ctypes.c_uint8))
        mem = np.ctypeslib.as_array(base_ptr, shape=(model.total_bytes,))
        mem[:] = 0

        with open(layout_json, "r") as f:
            layout = json.load(f)
        section = layout["sections"][0]
        buffers = collect_buffers(section)
        cfg = layout["config"]

        E = cfg["embed_dim"]
        H = cfg["num_heads"]
        KV = cfg["num_kv_heads"]
        D = cfg["head_dim"]
        I = cfg["intermediate_dim"]
        V = cfg["vocab_size"]
        T = cfg["max_seq_len"]
        eps = float(cfg.get("rms_norm_eps", 1e-6))
        theta = float(cfg.get("rope_theta", 10000.0))

        rng = np.random.default_rng(0)

        token_emb = (rng.standard_normal((V, E)) * 0.02).astype(np.float32)
        wq = (rng.standard_normal((H, D, E)) * 0.02).astype(np.float32)
        wk = (rng.standard_normal((KV, D, E)) * 0.02).astype(np.float32)
        wv = (rng.standard_normal((KV, D, E)) * 0.02).astype(np.float32)
        wo = (rng.standard_normal((H, E, D)) * 0.02).astype(np.float32)
        w1 = (rng.standard_normal((2 * I, E)) * 0.02).astype(np.float32)
        w2 = (rng.standard_normal((E, I)) * 0.02).astype(np.float32)

        ln1_gamma = np.ones((E,), dtype=np.float32)
        ln2_gamma = np.ones((E,), dtype=np.float32)
        final_ln_weight = np.ones((E,), dtype=np.float32)

        cos_cache, sin_cache = precompute_rope(theta, T, D)

        write_buffer(mem, parse_offset(buffers["token_emb"]["offset"]), token_emb)
        write_buffer(mem, parse_offset(buffers["layer.0.ln1_gamma"]["offset"]), ln1_gamma)
        write_buffer(mem, parse_offset(buffers["layer.0.ln2_gamma"]["offset"]), ln2_gamma)
        write_buffer(mem, parse_offset(buffers["layer.0.wq"]["offset"]), wq)
        write_buffer(mem, parse_offset(buffers["layer.0.wk"]["offset"]), wk)
        write_buffer(mem, parse_offset(buffers["layer.0.wv"]["offset"]), wv)
        write_buffer(mem, parse_offset(buffers["layer.0.wo"]["offset"]), wo)
        write_buffer(mem, parse_offset(buffers["layer.0.w1"]["offset"]), w1)
        write_buffer(mem, parse_offset(buffers["layer.0.w2"]["offset"]), w2)
        write_buffer(mem, parse_offset(buffers["final_ln_weight"]["offset"]), final_ln_weight)

        if "rope_cos_cache" in buffers:
            write_buffer(mem, parse_offset(buffers["rope_cos_cache"]["offset"]), cos_cache)
        if "rope_sin_cache" in buffers:
            write_buffer(mem, parse_offset(buffers["rope_sin_cache"]["offset"]), sin_cache)

        tokens = np.array([1, 7, 3, 11], dtype=np.int32)
        tokens_ptr = tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        lib.mini_v4_prefill_forward(ctypes.byref(model), tokens_ptr, tokens.size)

        logits_shape = buffers["logits"]["shape"]
        logits_shape = [int(x) for x in logits_shape]
        logits_c = read_buffer(mem, parse_offset(buffers["logits"]["offset"]), logits_shape)

        # PyTorch reference
        tokens_t = torch.from_numpy(tokens.astype(np.int64))
        x = torch.from_numpy(token_emb)[tokens_t]
        x = rmsnorm(x, torch.from_numpy(ln1_gamma), eps)

        q = torch.einsum("ta,hea->hte", x, torch.from_numpy(wq))
        k = torch.einsum("ta,hea->hte", x, torch.from_numpy(wk))
        v = torch.einsum("ta,hea->hte", x, torch.from_numpy(wv))

        q = apply_rope(q, cos_cache, sin_cache)
        k = apply_rope(k, cos_cache, sin_cache)

        attn = attention_gqa(q, k, v, H, KV)

        proj = torch.zeros((tokens_t.numel(), E), dtype=torch.float32)
        for h in range(H):
            proj += attn[h] @ torch.from_numpy(wo[h]).T

        residual1 = x + proj
        ln2 = rmsnorm(residual1, torch.from_numpy(ln2_gamma), eps)

        fc1 = ln2 @ torch.from_numpy(w1).T
        gate = fc1[:, :I]
        up = fc1[:, I:]
        swiglu = torch.nn.functional.silu(gate) * up
        mlp_out = swiglu @ torch.from_numpy(w2).T
        out = residual1 + mlp_out

        final = rmsnorm(out, torch.from_numpy(final_ln_weight), eps)
        logits_ref = final @ torch.from_numpy(token_emb).T

        diff = max_diff(torch.from_numpy(logits_c), logits_ref)

        report = TestReport(
            test_name="IR v4 single-layer forward (prefill)",
            dtype="fp32",
            shape=f"T={tokens.size}, E={E}, H={H}, KV={KV}, I={I}",
            cpu_info=get_cpu_info(),
        )
        report.add_result(TestResult(
            name="generated_c vs pytorch",
            passed=diff < 5e-3,
            max_diff=diff,
            tolerance=5e-3,
        ))
        report.print_report()

        assert diff < 5e-3, f"max diff too large: {diff}"
    finally:
        lib.mini_v4_prefill_model_free(ctypes.byref(model))


if __name__ == "__main__":
    main()
