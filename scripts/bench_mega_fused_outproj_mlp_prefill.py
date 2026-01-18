#!/usr/bin/env python3
"""
Benchmark mega_fused_outproj_mlp_prefill vs a baseline pipeline.

Baseline:
  - flatten head-major attn_out -> token-major
  - ck_gemm_nt_quant (WO)
  - residual add
  - RMSNorm2
  - fused_mlp_swiglu_prefill_w1w2_quant
  - residual add
"""
import argparse
import ctypes
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "unittest"))

from lib_loader import load_lib  # type: ignore
from test_utils import time_function, TimingResult  # type: ignore

CK_DT_Q4_K = 7
CK_DT_Q6_K = 8
CK_DT_Q8_0 = 9
CK_DT_Q8_K = 10
CK_DT_Q5_0 = 11

DTYPE_STR_TO_CK = {
    "q4_k": CK_DT_Q4_K,
    "q6_k": CK_DT_Q6_K,
    "q8_0": CK_DT_Q8_0,
    "q5_0": CK_DT_Q5_0,
}


def row_bytes(dt: int, n_elements: int) -> int:
    if dt == CK_DT_Q5_0:
        block_size, block_bytes = 32, 22
    elif dt == CK_DT_Q8_0:
        block_size, block_bytes = 32, 34
    elif dt == CK_DT_Q4_K:
        block_size, block_bytes = 256, 144
    elif dt == CK_DT_Q6_K:
        block_size, block_bytes = 256, 210
    elif dt == CK_DT_Q8_K:
        block_size, block_bytes = 256, 292
    else:
        raise ValueError(f"unsupported dt={dt}")
    n_blocks = (n_elements + block_size - 1) // block_size
    return n_blocks * block_bytes


def ptr_f32(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def ptr_void(arr: np.ndarray):
    return ctypes.c_void_p(arr.ctypes.data)


def make_q5_0_weights(rows: int, cols: int, scale: float = 0.02) -> np.ndarray:
    rb = row_bytes(CK_DT_Q5_0, cols)
    data = np.random.randint(0, 256, size=rows * rb, dtype=np.uint8)
    scale_bytes = np.frombuffer(np.float16(scale).tobytes(), dtype=np.uint8)
    blocks_per_row = (cols + 31) // 32
    for r in range(rows):
        base = r * rb
        for b in range(blocks_per_row):
            off = base + b * 22
            data[off:off + 2] = scale_bytes
    return data


def make_q4_k_weights(rows: int, cols: int, scale: float = 0.02) -> np.ndarray:
    rb = row_bytes(CK_DT_Q4_K, cols)
    data = np.random.randint(0, 256, size=rows * rb, dtype=np.uint8)
    scale_bytes = np.frombuffer(np.float16(scale).tobytes(), dtype=np.uint8)
    dmin_bytes = np.frombuffer(np.float16(0.0).tobytes(), dtype=np.uint8)
    blocks_per_row = (cols + 255) // 256
    for r in range(rows):
        base = r * rb
        for b in range(blocks_per_row):
            off = base + b * 144
            data[off:off + 2] = scale_bytes
            data[off + 2:off + 4] = dmin_bytes
    return data


def make_q6_k_weights(rows: int, cols: int, scale: float = 0.02) -> np.ndarray:
    rb = row_bytes(CK_DT_Q6_K, cols)
    data = np.random.randint(0, 256, size=rows * rb, dtype=np.uint8)
    scale_bytes = np.frombuffer(np.float16(scale).tobytes(), dtype=np.uint8)
    blocks_per_row = (cols + 255) // 256
    for r in range(rows):
        base = r * rb
        for b in range(blocks_per_row):
            off = base + b * 210
            data[off + 208:off + 210] = scale_bytes
    return data


@dataclass
class BenchDims:
    tokens: int
    embed_dim: int
    aligned_embed_dim: int
    num_heads: int
    head_dim: int
    aligned_head_dim: int
    intermediate: int
    aligned_intermediate: int
    eps: float


@dataclass
class BenchWeights:
    wo: np.ndarray
    w1: np.ndarray
    w2: np.ndarray
    ln2_gamma: np.ndarray
    wo_dt: int
    w1_dt: int
    w2_dt: int


@dataclass
class BenchBuffers:
    attn_out: np.ndarray
    residual: np.ndarray
    proj_in: np.ndarray
    proj_out: np.ndarray
    h1: np.ndarray
    ln2_out: np.ndarray
    rstd: np.ndarray
    mlp_out: np.ndarray
    out_ref: np.ndarray
    out_fused: np.ndarray
    mlp_scratch: np.ndarray
    fused_scratch: np.ndarray


@dataclass
class BenchResult:
    baseline: TimingResult
    fused: TimingResult

    @property
    def speedup(self) -> float:
        return self.baseline.mean_us / self.fused.mean_us


class OutprojMLPBench:
    def __init__(self):
        self.lib = load_lib("libckernel_engine.so")
        self._bind()

    def _bind(self):
        lib = self.lib
        lib.ck_gemm_nt_quant.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.ck_gemm_nt_quant.restype = None

        lib.rmsnorm_forward.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        lib.rmsnorm_forward.restype = None

        lib.fused_mlp_swiglu_prefill_w1w2_quant.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.fused_mlp_swiglu_prefill_w1w2_quant.restype = None

        lib.fused_mlp_swiglu_prefill_w1w2_quant_scratch_size.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.fused_mlp_swiglu_prefill_w1w2_quant_scratch_size.restype = ctypes.c_size_t

        lib.mega_fused_outproj_mlp_prefill.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p,
        ]
        lib.mega_fused_outproj_mlp_prefill.restype = None

        lib.mega_fused_outproj_mlp_prefill_scratch_size.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.mega_fused_outproj_mlp_prefill_scratch_size.restype = ctypes.c_size_t

    def run_baseline(self, dims: BenchDims, weights: BenchWeights, buf: BenchBuffers):
        proj_view = buf.attn_out.transpose(1, 0, 2).reshape(dims.tokens, dims.aligned_embed_dim)
        np.copyto(buf.proj_in, proj_view)

        self.lib.ck_gemm_nt_quant(ptr_f32(buf.proj_in),
                                  ptr_void(weights.wo),
                                  None,
                                  ptr_f32(buf.proj_out),
                                  ctypes.c_int(dims.tokens),
                                  ctypes.c_int(dims.aligned_embed_dim),
                                  ctypes.c_int(dims.aligned_embed_dim),
                                  ctypes.c_int(weights.wo_dt))

        np.add(buf.proj_out, buf.residual, out=buf.h1)

        self.lib.rmsnorm_forward(ptr_f32(buf.h1),
                                 ptr_f32(weights.ln2_gamma),
                                 ptr_f32(buf.ln2_out),
                                 ptr_f32(buf.rstd),
                                 ctypes.c_int(dims.tokens),
                                 ctypes.c_int(dims.embed_dim),
                                 ctypes.c_int(dims.aligned_embed_dim),
                                 ctypes.c_float(dims.eps))

        self.lib.fused_mlp_swiglu_prefill_w1w2_quant(ptr_f32(buf.ln2_out),
                                                    ptr_void(weights.w1),
                                                    None,
                                                    ctypes.c_int(weights.w1_dt),
                                                    ptr_void(weights.w2),
                                                    None,
                                                    ctypes.c_int(weights.w2_dt),
                                                    ptr_f32(buf.mlp_out),
                                                    ctypes.c_int(dims.tokens),
                                                    ctypes.c_int(dims.embed_dim),
                                                    ctypes.c_int(dims.aligned_embed_dim),
                                                    ctypes.c_int(dims.intermediate),
                                                    ctypes.c_int(dims.aligned_intermediate),
                                                    ptr_void(buf.mlp_scratch))

        np.add(buf.mlp_out, buf.h1, out=buf.out_ref)

    def run_fused(self, dims: BenchDims, weights: BenchWeights, buf: BenchBuffers):
        self.lib.mega_fused_outproj_mlp_prefill(ptr_f32(buf.out_fused),
                                                ptr_f32(buf.attn_out),
                                                ptr_f32(buf.residual),
                                                ptr_f32(weights.ln2_gamma),
                                                ptr_void(weights.wo),
                                                None,
                                                ctypes.c_int(weights.wo_dt),
                                                ptr_void(weights.w1),
                                                None,
                                                ctypes.c_int(weights.w1_dt),
                                                ptr_void(weights.w2),
                                                None,
                                                ctypes.c_int(weights.w2_dt),
                                                ctypes.c_int(dims.tokens),
                                                ctypes.c_int(dims.embed_dim),
                                                ctypes.c_int(dims.aligned_embed_dim),
                                                ctypes.c_int(dims.num_heads),
                                                ctypes.c_int(dims.aligned_head_dim),
                                                ctypes.c_int(dims.intermediate),
                                                ctypes.c_int(dims.aligned_intermediate),
                                                ctypes.c_float(dims.eps),
                                                ptr_void(buf.fused_scratch))


def load_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest(model_dir: str) -> dict:
    manifest_path = os.path.join(model_dir, "weights_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return {e["name"]: e for e in manifest["entries"]}


def read_weight_bytes(bump_path: str, entry: dict) -> np.ndarray:
    with open(bump_path, "rb") as f:
        f.seek(entry["file_offset"])
        data = f.read(entry["size"])
    return np.frombuffer(data, dtype=np.uint8).copy()


def read_weight_f32(bump_path: str, entry: dict) -> np.ndarray:
    with open(bump_path, "rb") as f:
        f.seek(entry["file_offset"])
        data = f.read(entry["size"])
    return np.frombuffer(data, dtype=np.float32).copy()


def make_buffers(dims: BenchDims, bench: OutprojMLPBench) -> BenchBuffers:
    attn_out = np.random.randn(dims.num_heads, dims.tokens, dims.aligned_head_dim).astype(np.float32)
    residual = np.random.randn(dims.tokens, dims.aligned_embed_dim).astype(np.float32)
    proj_in = np.zeros((dims.tokens, dims.aligned_embed_dim), dtype=np.float32)
    proj_out = np.zeros_like(proj_in)
    h1 = np.zeros_like(proj_in)
    ln2_out = np.zeros_like(proj_in)
    rstd = np.zeros(dims.tokens, dtype=np.float32)
    mlp_out = np.zeros_like(proj_in)
    out_ref = np.zeros_like(proj_in)
    out_fused = np.zeros_like(proj_in)

    mlp_scratch_bytes = bench.lib.fused_mlp_swiglu_prefill_w1w2_quant_scratch_size(
        ctypes.c_int(dims.aligned_embed_dim),
        ctypes.c_int(dims.aligned_intermediate),
    )
    fused_scratch_bytes = bench.lib.mega_fused_outproj_mlp_prefill_scratch_size(
        ctypes.c_int(dims.tokens),
        ctypes.c_int(dims.aligned_embed_dim),
        ctypes.c_int(dims.num_heads),
        ctypes.c_int(dims.aligned_head_dim),
        ctypes.c_int(dims.aligned_intermediate),
    )

    mlp_scratch = np.zeros(mlp_scratch_bytes, dtype=np.uint8)
    fused_scratch = np.zeros(fused_scratch_bytes, dtype=np.uint8)

    return BenchBuffers(attn_out=attn_out,
                        residual=residual,
                        proj_in=proj_in,
                        proj_out=proj_out,
                        h1=h1,
                        ln2_out=ln2_out,
                        rstd=rstd,
                        mlp_out=mlp_out,
                        out_ref=out_ref,
                        out_fused=out_fused,
                        mlp_scratch=mlp_scratch,
                        fused_scratch=fused_scratch)


def run_bench_for_dims(bench: OutprojMLPBench, dims: BenchDims, weights: BenchWeights,
                       warmup: int, iters: int) -> BenchResult:
    buf = make_buffers(dims, bench)
    baseline_fn = lambda: bench.run_baseline(dims, weights, buf)
    fused_fn = lambda: bench.run_fused(dims, weights, buf)
    baseline = time_function(baseline_fn, warmup=warmup, iterations=iters, name="baseline")
    fused = time_function(fused_fn, warmup=warmup, iterations=iters, name="fused")
    return BenchResult(baseline=baseline, fused=fused)


def print_result(label: str, dims: BenchDims, res: BenchResult):
    print(
        f"{label:<24} tokens={dims.tokens:<4d} "
        f"baseline={res.baseline.mean_us:9.1f} us "
        f"fused={res.fused.mean_us:9.1f} us "
        f"speedup={res.speedup:5.2f}x"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark mega_fused_outproj_mlp_prefill.")
    ap.add_argument("--model-dir", default="/home/antshiv/.cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF")
    ap.add_argument("--seq-lens", default="32,64,128")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    seq_lens = [int(x) for x in args.seq_lens.split(",") if x]

    bench = OutprojMLPBench()

    cfg = load_config(args.model_dir)
    embed_dim = int(cfg["hidden_size"])
    intermediate = int(cfg["intermediate_size"])
    num_heads = int(cfg["num_attention_heads"])
    head_dim = embed_dim // num_heads
    eps = float(cfg.get("rms_norm_eps", 1e-6))

    entries = load_manifest(args.model_dir)
    bump_path = os.path.join(args.model_dir, "weights.bump")

    print("[bench] dims:", {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
    })

    # Synthetic (q5_0 / q4_k)
    print("\n[SYNTHETIC] wo/w1=q5_0, w2=q4_k")
    for tokens in seq_lens:
        dims = BenchDims(tokens=tokens,
                         embed_dim=embed_dim,
                         aligned_embed_dim=embed_dim,
                         num_heads=num_heads,
                         head_dim=head_dim,
                         aligned_head_dim=head_dim,
                         intermediate=intermediate,
                         aligned_intermediate=intermediate,
                         eps=eps)
        weights = BenchWeights(
            wo=make_q5_0_weights(dims.aligned_embed_dim, dims.aligned_embed_dim),
            w1=make_q5_0_weights(2 * dims.aligned_intermediate, dims.aligned_embed_dim),
            w2=make_q4_k_weights(dims.aligned_embed_dim, dims.aligned_intermediate),
            ln2_gamma=np.random.randn(dims.aligned_embed_dim).astype(np.float32),
            wo_dt=CK_DT_Q5_0,
            w1_dt=CK_DT_Q5_0,
            w2_dt=CK_DT_Q4_K,
        )
        res = run_bench_for_dims(bench, dims, weights, args.warmup, args.iters)
        print_result("synthetic-q4k", dims, res)

    # Real weights (first layer for each W2 dtype)
    layers = []
    for name, entry in entries.items():
        if name.startswith("layer.") and name.endswith(".w2"):
            layer = int(name.split(".")[1])
            if entry["dtype"] in ("q4_k", "q6_k"):
                layers.append((layer, entry["dtype"]))
    layers = sorted(set(layers), key=lambda x: x[0])
    if layers:
        print("\n[REAL] weights.bump per-layer")

    for layer, w2_dtype in layers[:2]:
        w1_e = entries[f"layer.{layer}.w1"]
        w2_e = entries[f"layer.{layer}.w2"]
        wo_e = entries[f"layer.{layer}.wo"]
        ln2_e = entries[f"layer.{layer}.ln2_gamma"]

        if w1_e["dtype"] not in ("q5_0", "q8_0") or wo_e["dtype"] not in ("q5_0", "q8_0"):
            print(f"real-L{layer}-{w2_dtype:<5} tokens=----  SKIP (w1/wo dtype)")
            continue

        weights = BenchWeights(
            wo=read_weight_bytes(bump_path, wo_e),
            w1=read_weight_bytes(bump_path, w1_e),
            w2=read_weight_bytes(bump_path, w2_e),
            ln2_gamma=read_weight_f32(bump_path, ln2_e),
            wo_dt=DTYPE_STR_TO_CK[wo_e["dtype"]],
            w1_dt=DTYPE_STR_TO_CK[w1_e["dtype"]],
            w2_dt=DTYPE_STR_TO_CK[w2_e["dtype"]],
        )

        for tokens in seq_lens:
            dims = BenchDims(tokens=tokens,
                             embed_dim=embed_dim,
                             aligned_embed_dim=embed_dim,
                             num_heads=num_heads,
                             head_dim=head_dim,
                             aligned_head_dim=head_dim,
                             intermediate=intermediate,
                             aligned_intermediate=intermediate,
                             eps=eps)
            res = run_bench_for_dims(bench, dims, weights, args.warmup, args.iters)
            print_result(f"real-L{layer}-{w2_dtype}", dims, res)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
