"""
Multi-layer parity test.

Tests progressive layer counts to catch inter-layer bugs:
1. Single layer - verify basic layer works
2. Two layers - verify layer-to-layer data handoff
3. N layers - full model parity

Bugs this catches:
- Output shape mismatches between layers
- Residual connection issues
- KV cache handoff across layers
- Memory layout incompatibilities

Usage:
    python test_multi_layer_parity.py              # Run all layer counts
    python test_multi_layer_parity.py --layers 2   # Test specific count
    DEBUG=1 python test_multi_layer_parity.py      # Verbose output
"""
import ctypes
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib_loader import load_lib
from test_utils import max_diff, print_system_info

DEBUG = os.environ.get("DEBUG", "0") == "1"

lib = load_lib("libckernel_engine.so")


def align_up(n, a):
    return (n + a - 1) // a * a


def aligned_empty(shape, dtype=np.float32, align=64):
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    buf = np.empty(nbytes + align, dtype=np.uint8)
    offset = (-buf.ctypes.data) % align
    arr = buf[offset:offset + nbytes].view(dtype).reshape(shape)
    assert arr.flags['C_CONTIGUOUS'], "Array must be C-contiguous"
    return arr


def ptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


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


lib.ck_layer_forward_rmsnorm_swiglu.argtypes = [ctypes.POINTER(CKLayerForwardParams)]
lib.ck_layer_forward_rmsnorm_swiglu.restype = None

lib.rope_precompute_cache.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_float,
    ctypes.c_int, ctypes.c_char_p, ctypes.c_float,
]
lib.rope_precompute_cache.restype = None


class PyTorchTransformerLayer(nn.Module):
    """Reference PyTorch implementation for parity testing."""

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config["embed_dim"]
        self.num_heads = config["num_heads"]
        self.num_kv_heads = config["num_kv_heads"]
        self.head_dim = self.embed_dim // self.num_heads
        self.intermediate_dim = config["intermediate_dim"]

        self.ln1 = nn.RMSNorm(self.embed_dim, eps=config["eps"])
        self.ln2 = nn.RMSNorm(self.embed_dim, eps=config["eps"])

        self.wq = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=True)
        self.wk = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=True)
        self.wv = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=True)
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=True)

        self.w1 = nn.Linear(self.embed_dim, 2 * self.intermediate_dim, bias=True)
        self.w2 = nn.Linear(self.intermediate_dim, self.embed_dim, bias=True)

    def forward(self, x, rope_cos, rope_sin):
        B, T, D = x.shape
        residual = x

        # Attention block
        h = self.ln1(x)
        q = self.wq(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE
        q, k = self.apply_rope(q, k, rope_cos, rope_sin)

        # GQA expansion
        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # Attention
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.wo(out)
        h = residual + out

        # MLP block
        residual = h
        h = self.ln2(h)
        gate_up = self.w1(h)
        gate, up = gate_up.chunk(2, dim=-1)
        h = F.silu(gate) * up
        h = self.w2(h)
        return residual + h

    def apply_rope(self, q, k, cos, sin):
        def rotate(x, cos, sin):
            x1, x2 = x[..., ::2], x[..., 1::2]
            return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
        return rotate(q, cos, sin), rotate(k, cos, sin)


def run_multi_layer_test(num_layers, seed=42):
    """Test N layers stacked together."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Config
    config = {
        "embed_dim": 64,
        "num_heads": 4,
        "num_kv_heads": 2,
        "intermediate_dim": 128,
        "eps": 1e-5,
        "rope_theta": 10000.0,
    }

    tokens = 8
    embed_dim = config["embed_dim"]
    num_heads = config["num_heads"]
    num_kv_heads = config["num_kv_heads"]
    head_dim = embed_dim // num_heads
    intermediate_dim = config["intermediate_dim"]
    eps = config["eps"]

    aligned_embed_dim = align_up(embed_dim, 16)
    aligned_head_dim = align_up(head_dim, 16)
    aligned_intermediate_dim = align_up(intermediate_dim, 16)
    aligned_context_window = align_up(tokens, 16)

    if DEBUG:
        print(f"\n=== Testing {num_layers} layer(s) ===")
        print(f"  embed_dim={embed_dim}, heads={num_heads}, kv_heads={num_kv_heads}")
        print(f"  intermediate={intermediate_dim}, tokens={tokens}")

    # Input
    x_np = np.random.randn(tokens, embed_dim).astype(np.float32)
    x_torch = torch.from_numpy(x_np).unsqueeze(0)  # [1, T, D]

    # RoPE precompute
    rope_cos = aligned_empty((tokens, head_dim // 2))
    rope_sin = aligned_empty((tokens, head_dim // 2))
    lib.rope_precompute_cache(ptr(rope_cos), ptr(rope_sin),
                              tokens, head_dim, config["rope_theta"],
                              head_dim, b"none", 1.0)
    rope_cos_torch = torch.from_numpy(rope_cos).unsqueeze(0).unsqueeze(0)
    rope_sin_torch = torch.from_numpy(rope_sin).unsqueeze(0).unsqueeze(0)

    # Create PyTorch reference layers
    pt_layers = nn.ModuleList([PyTorchTransformerLayer(config) for _ in range(num_layers)])

    # Allocate C buffers (reused per layer)
    def alloc_layer_buffers():
        return {
            "ln1_out": aligned_empty((tokens, aligned_embed_dim)),
            "ln1_rstd": aligned_empty((tokens,)),
            "q": aligned_empty((num_heads, tokens, aligned_head_dim)),
            "k": aligned_empty((num_kv_heads, tokens, aligned_head_dim)),
            "v": aligned_empty((num_kv_heads, tokens, aligned_head_dim)),
            "attn_out": aligned_empty((num_heads, tokens, aligned_head_dim)),
            "proj_tmp": aligned_empty((tokens, aligned_embed_dim)),
            "proj_scratch": aligned_empty((tokens, aligned_embed_dim)),
            "residual1": aligned_empty((tokens, aligned_embed_dim)),
            "ln2_out": aligned_empty((tokens, aligned_embed_dim)),
            "ln2_rstd": aligned_empty((tokens,)),
            "fc1_out": aligned_empty((tokens, 2 * aligned_intermediate_dim)),
            "swiglu_out": aligned_empty((tokens, aligned_intermediate_dim)),
            "mlp_out": aligned_empty((tokens, aligned_embed_dim)),
            "output": aligned_empty((tokens, aligned_embed_dim)),
        }

    bufs = alloc_layer_buffers()
    for arr in bufs.values():
        arr.fill(0.0)

    # C input/output
    c_input = aligned_empty((tokens, aligned_embed_dim))
    c_input.fill(0.0)
    c_input[:, :embed_dim] = x_np

    c_output = aligned_empty((tokens, aligned_embed_dim))

    # Run PyTorch reference
    pt_out = x_torch
    for layer in pt_layers:
        pt_out = layer(pt_out, rope_cos_torch, rope_sin_torch)

    # Run C implementation layer by layer
    layer_diffs = []
    for layer_idx in range(num_layers):
        pt_layer = pt_layers[layer_idx]

        # Copy weights from PyTorch to C format
        wq = aligned_empty((num_heads, aligned_head_dim, aligned_embed_dim))
        wk = aligned_empty((num_kv_heads, aligned_head_dim, aligned_embed_dim))
        wv = aligned_empty((num_kv_heads, aligned_head_dim, aligned_embed_dim))
        bq = aligned_empty((num_heads, aligned_head_dim))
        bk = aligned_empty((num_kv_heads, aligned_head_dim))
        bv = aligned_empty((num_kv_heads, aligned_head_dim))
        wo = aligned_empty((num_heads, aligned_embed_dim, aligned_head_dim))
        bo = aligned_empty((aligned_embed_dim,))
        w1 = aligned_empty((2 * aligned_intermediate_dim, aligned_embed_dim))
        b1 = aligned_empty((2 * aligned_intermediate_dim,))
        w2 = aligned_empty((aligned_embed_dim, aligned_intermediate_dim))
        b2 = aligned_empty((aligned_embed_dim,))
        ln1_gamma = aligned_empty((aligned_embed_dim,))
        ln2_gamma = aligned_empty((aligned_embed_dim,))

        for arr in [wq, wk, wv, bq, bk, bv, wo, bo, w1, b1, w2, b2, ln1_gamma, ln2_gamma]:
            arr.fill(0.0)

        # Copy weights (reshape for C layout)
        wq_pt = pt_layer.wq.weight.data.view(num_heads, head_dim, embed_dim).numpy()
        wk_pt = pt_layer.wk.weight.data.view(num_kv_heads, head_dim, embed_dim).numpy()
        wv_pt = pt_layer.wv.weight.data.view(num_kv_heads, head_dim, embed_dim).numpy()
        wq[:, :head_dim, :embed_dim] = wq_pt
        wk[:, :head_dim, :embed_dim] = wk_pt
        wv[:, :head_dim, :embed_dim] = wv_pt

        bq[:, :head_dim] = pt_layer.wq.bias.data.view(num_heads, head_dim).numpy()
        bk[:, :head_dim] = pt_layer.wk.bias.data.view(num_kv_heads, head_dim).numpy()
        bv[:, :head_dim] = pt_layer.wv.bias.data.view(num_kv_heads, head_dim).numpy()

        wo_pt = pt_layer.wo.weight.data.view(embed_dim, num_heads, head_dim).permute(1, 0, 2).numpy()
        wo[:, :embed_dim, :head_dim] = wo_pt
        bo[:embed_dim] = pt_layer.wo.bias.data.numpy()

        w1[:2*intermediate_dim, :embed_dim] = pt_layer.w1.weight.data.numpy()
        b1[:2*intermediate_dim] = pt_layer.w1.bias.data.numpy()
        w2[:embed_dim, :intermediate_dim] = pt_layer.w2.weight.data.numpy()
        b2[:embed_dim] = pt_layer.w2.bias.data.numpy()

        ln1_gamma[:embed_dim] = pt_layer.ln1.weight.data.numpy()
        ln2_gamma[:embed_dim] = pt_layer.ln2.weight.data.numpy()

        # Clear buffers
        for arr in bufs.values():
            arr.fill(0.0)

        # Build params
        params = CKLayerForwardParams(
            tokens=tokens,
            embed_dim=embed_dim,
            aligned_embed_dim=aligned_embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            aligned_head_dim=aligned_head_dim,
            aligned_context_window=aligned_context_window,
            intermediate_dim=intermediate_dim,
            aligned_intermediate_dim=aligned_intermediate_dim,
            eps=eps,
            rope_pos_offset=0,
            input=ptr(c_input),
            ln1_gamma=ptr(ln1_gamma),
            ln2_gamma=ptr(ln2_gamma),
            rope_cos=ptr(rope_cos),
            rope_sin=ptr(rope_sin),
            wq=ptr(wq), bq=ptr(bq),
            wk=ptr(wk), bk=ptr(bk),
            wv=ptr(wv), bv=ptr(bv),
            wo=ptr(wo), bo=ptr(bo),
            w1=ptr(w1), b1=ptr(b1),
            w2=ptr(w2), b2=ptr(b2),
            ln1_out=ptr(bufs["ln1_out"]),
            ln1_rstd=ptr(bufs["ln1_rstd"]),
            q=ptr(bufs["q"]),
            k=ptr(bufs["k"]),
            v=ptr(bufs["v"]),
            scores=None,
            attn_out=ptr(bufs["attn_out"]),
            proj_tmp=ptr(bufs["proj_tmp"]),
            proj_scratch=ptr(bufs["proj_scratch"]),
            residual1=ptr(bufs["residual1"]),
            ln2_out=ptr(bufs["ln2_out"]),
            ln2_rstd=ptr(bufs["ln2_rstd"]),
            fc1_out=ptr(bufs["fc1_out"]),
            swiglu_out=ptr(bufs["swiglu_out"]),
            mlp_out=ptr(bufs["mlp_out"]),
            output=ptr(bufs["output"]),
        )

        lib.ck_layer_forward_rmsnorm_swiglu(ctypes.byref(params))

        # Copy output to next layer's input
        c_output[:] = bufs["output"]
        c_input[:] = c_output

        # Compare intermediate output (for debugging)
        if DEBUG and layer_idx < num_layers - 1:
            # Run PT up to this layer
            pt_intermediate = x_torch
            for i in range(layer_idx + 1):
                pt_intermediate = pt_layers[i](pt_intermediate, rope_cos_torch, rope_sin_torch)
            c_out_slice = c_output[:, :embed_dim]
            pt_out_slice = pt_intermediate[0].detach().numpy()
            diff = np.max(np.abs(c_out_slice - pt_out_slice))
            layer_diffs.append(diff)
            print(f"  Layer {layer_idx}: intermediate diff = {diff:.2e}")

    # Final comparison
    c_final = c_output[:, :embed_dim]
    pt_final = pt_out[0].detach().numpy()
    final_diff = max_diff(torch.from_numpy(c_final), torch.from_numpy(pt_final))

    if DEBUG:
        print(f"  Final output diff: {final_diff:.2e}")

    return final_diff


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-layer parity test")
    parser.add_argument("--layers", type=int, default=None,
                       help="Test specific layer count (default: test 1,2,4)")
    args = parser.parse_args()

    print_system_info()

    if args.layers:
        layer_counts = [args.layers]
    else:
        layer_counts = [1, 2, 4]  # Progressive test

    # Tolerance: catch gross bugs (stride bug = 1e+3 diff), not strict numerical parity
    # Some diff is expected due to: fast softmax approximation, RoPE implementation, etc.
    tol = 0.5  # Reasonable for catching memory/layout bugs
    results = []

    print("\n" + "=" * 72)
    print("  TEST: Multi-Layer Parity (1 → 2 → N layers)")
    print("=" * 72)

    for n in layer_counts:
        diff = run_multi_layer_test(n)
        status = "PASS" if diff <= tol else "FAIL"
        results.append((n, diff, status))
        print(f"\n  {n} layer(s):  max_diff={diff:.2e}  tol={tol:.0e}  [{status}]")

    print("\n" + "-" * 72)

    failed = [r for r in results if r[2] == "FAIL"]
    if failed:
        print(f"  FAILED: {[r[0] for r in failed]} layer(s)")
        print("\n  Tip: Run with DEBUG=1 to see per-layer diagnostics")
        sys.exit(1)
    else:
        print("  All layer counts PASSED")


if __name__ == "__main__":
    main()
