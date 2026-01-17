#!/usr/bin/env python3
"""
Benchmark mega_fused_attention_prefill vs baseline (quantized weights).

Baseline:
- fused_rmsnorm_qkv_prefill_head_major_quant
- flash attention (head-major, strided)
- out-proj via ck_gemm_nt_quant
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

CK_DT_Q5_0 = 11
CK_DT_Q8_0 = 9

DTYPE_STR_TO_CK = {
    "q5_0": CK_DT_Q5_0,
    "q8_0": CK_DT_Q8_0,
}


def row_bytes_q5_0(n_elements: int) -> int:
    block_size, block_bytes = 32, 22
    n_blocks = (n_elements + block_size - 1) // block_size
    return n_blocks * block_bytes


def row_bytes_q8_0(n_elements: int) -> int:
    block_size, block_bytes = 32, 34
    n_blocks = (n_elements + block_size - 1) // block_size
    return n_blocks * block_bytes


def ptr_f32(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def ptr_void(arr: np.ndarray):
    return ctypes.c_void_p(arr.ctypes.data)

def make_q5_0_weights(rows: int, cols: int, scale: float = 0.02) -> np.ndarray:
    rb = row_bytes_q5_0(cols)
    data = np.random.randint(0, 256, size=rows * rb, dtype=np.uint8)
    scale_bytes = np.frombuffer(np.float16(scale).tobytes(), dtype=np.uint8)
    blocks_per_row = (cols + 31) // 32
    for r in range(rows):
        base = r * rb
        for b in range(blocks_per_row):
            off = base + b * 22
            data[off:off + 2] = scale_bytes
    return data


def make_q8_0_weights(rows: int, cols: int, scale: float = 0.02) -> np.ndarray:
    rb = row_bytes_q8_0(cols)
    data = np.random.randint(0, 256, size=rows * rb, dtype=np.uint8)
    scale_bytes = np.frombuffer(np.float16(scale).tobytes(), dtype=np.uint8)
    blocks_per_row = (cols + 31) // 32
    for r in range(rows):
        base = r * rb
        for b in range(blocks_per_row):
            off = base + b * 34
            data[off:off + 2] = scale_bytes
    return data

@dataclass
class AttnDims:
    tokens: int
    embed_dim: int
    aligned_embed_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    aligned_head_dim: int
    cache_capacity: int
    eps: float


@dataclass
class AttnWeights:
    wq: np.ndarray
    wk: np.ndarray
    wv: np.ndarray
    wo: np.ndarray
    bq: np.ndarray
    bk: np.ndarray
    bv: np.ndarray
    wq_dt: int
    wk_dt: int
    wv_dt: int
    wo_dt: int


@dataclass
class AttnBuffers:
    x: np.ndarray
    gamma: np.ndarray
    q: np.ndarray
    kv_cache_k: np.ndarray
    kv_cache_v: np.ndarray
    attn_out: np.ndarray
    proj_in: np.ndarray
    out_ref: np.ndarray
    out_fused: np.ndarray
    qkv_scratch: np.ndarray
    scratch: np.ndarray


@dataclass
class BenchResult:
    baseline: TimingResult
    fused: TimingResult

    @property
    def speedup(self) -> float:
        return self.baseline.mean_us / self.fused.mean_us


class AttnBench:
    def __init__(self):
        self.lib = load_lib("libckernel_engine.so")
        self._bind()

    def _bind(self):
        lib = self.lib
        lib.fused_rmsnorm_qkv_prefill_head_major_quant.argtypes = [
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
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
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
        lib.fused_rmsnorm_qkv_prefill_head_major_quant.restype = None

        lib.fused_rmsnorm_qkv_prefill_head_major_quant_scratch_size.argtypes = [ctypes.c_int]
        lib.fused_rmsnorm_qkv_prefill_head_major_quant_scratch_size.restype = ctypes.c_size_t

        lib.attention_forward_causal_head_major_gqa_flash_strided.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.attention_forward_causal_head_major_gqa_flash_strided.restype = None

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

        lib.mega_fused_attention_prefill.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # output
            ctypes.POINTER(ctypes.c_float),  # input
            ctypes.POINTER(ctypes.c_float),  # residual
            ctypes.POINTER(ctypes.c_float),  # ln1_gamma
            ctypes.c_void_p,                 # wq
            ctypes.POINTER(ctypes.c_float),  # bq
            ctypes.c_int,                    # wq_dt
            ctypes.c_void_p,                 # wk
            ctypes.POINTER(ctypes.c_float),  # bk
            ctypes.c_int,                    # wk_dt
            ctypes.c_void_p,                 # wv
            ctypes.POINTER(ctypes.c_float),  # bv
            ctypes.c_int,                    # wv_dt
            ctypes.c_void_p,                 # wo
            ctypes.POINTER(ctypes.c_float),  # bo
            ctypes.c_int,                    # wo_dt
            ctypes.POINTER(ctypes.c_float),  # kv_cache_k
            ctypes.POINTER(ctypes.c_float),  # kv_cache_v
            ctypes.c_void_p,                 # rope_cos
            ctypes.c_void_p,                 # rope_sin
            ctypes.c_int,                    # start_pos
            ctypes.c_int,                    # tokens
            ctypes.c_int,                    # cache_capacity
            ctypes.c_int,                    # embed_dim
            ctypes.c_int,                    # aligned_embed_dim
            ctypes.c_int,                    # num_heads
            ctypes.c_int,                    # num_kv_heads
            ctypes.c_int,                    # head_dim
            ctypes.c_int,                    # aligned_head_dim
            ctypes.c_float,                  # eps
            ctypes.c_void_p,                 # scratch
        ]
        lib.mega_fused_attention_prefill.restype = None

        lib.mega_fused_attention_prefill_scratch_size.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.mega_fused_attention_prefill_scratch_size.restype = ctypes.c_size_t

    def run_baseline(self, dims: AttnDims, weights: AttnWeights, buf: AttnBuffers):
        lib = self.lib
        lib.fused_rmsnorm_qkv_prefill_head_major_quant(
            ptr_f32(buf.x),
            ptr_f32(buf.gamma),
            ptr_void(weights.wq), ptr_f32(weights.bq), ctypes.c_int(weights.wq_dt),
            ptr_void(weights.wk), ptr_f32(weights.bk), ctypes.c_int(weights.wk_dt),
            ptr_void(weights.wv), ptr_f32(weights.bv), ctypes.c_int(weights.wv_dt),
            ptr_f32(buf.q),
            ptr_f32(buf.kv_cache_k),
            ptr_f32(buf.kv_cache_v),
            ctypes.c_int(dims.tokens),
            ctypes.c_int(dims.embed_dim),
            ctypes.c_int(dims.aligned_embed_dim),
            ctypes.c_int(dims.num_heads),
            ctypes.c_int(dims.num_kv_heads),
            ctypes.c_int(dims.head_dim),
            ctypes.c_int(dims.aligned_head_dim),
            ctypes.c_int(dims.cache_capacity),
            ctypes.c_float(dims.eps),
            ptr_void(buf.qkv_scratch),
        )

        lib.attention_forward_causal_head_major_gqa_flash_strided(
            ptr_f32(buf.q),
            ptr_f32(buf.kv_cache_k),
            ptr_f32(buf.kv_cache_v),
            ptr_f32(buf.attn_out),
            ctypes.c_int(dims.num_heads),
            ctypes.c_int(dims.num_kv_heads),
            ctypes.c_int(dims.tokens),
            ctypes.c_int(dims.head_dim),
            ctypes.c_int(dims.aligned_head_dim),
            ctypes.c_int(dims.cache_capacity),
        )

        # Flatten head-major -> token-major
        for t in range(dims.tokens):
            for h in range(dims.num_heads):
                buf.proj_in[t, h * dims.aligned_head_dim:(h + 1) * dims.aligned_head_dim] = buf.attn_out[h, t]

        lib.ck_gemm_nt_quant(ptr_f32(buf.proj_in), ptr_void(weights.wo), None, ptr_f32(buf.out_ref),
                             ctypes.c_int(dims.tokens), ctypes.c_int(dims.aligned_embed_dim),
                             ctypes.c_int(dims.aligned_embed_dim), ctypes.c_int(weights.wo_dt))

    def run_fused(self, dims: AttnDims, weights: AttnWeights, buf: AttnBuffers):
        rope_null = ctypes.c_void_p(0)
        self.lib.mega_fused_attention_prefill(
            ptr_f32(buf.out_fused),
            ptr_f32(buf.x),
            None,
            ptr_f32(buf.gamma),
            ptr_void(weights.wq), ptr_f32(weights.bq), ctypes.c_int(weights.wq_dt),
            ptr_void(weights.wk), ptr_f32(weights.bk), ctypes.c_int(weights.wk_dt),
            ptr_void(weights.wv), ptr_f32(weights.bv), ctypes.c_int(weights.wv_dt),
            ptr_void(weights.wo), None, ctypes.c_int(weights.wo_dt),
            ptr_f32(buf.kv_cache_k),
            ptr_f32(buf.kv_cache_v),
            rope_null,
            rope_null,
            ctypes.c_int(0),
            ctypes.c_int(dims.tokens),
            ctypes.c_int(dims.cache_capacity),
            ctypes.c_int(dims.embed_dim),
            ctypes.c_int(dims.aligned_embed_dim),
            ctypes.c_int(dims.num_heads),
            ctypes.c_int(dims.num_kv_heads),
            ctypes.c_int(dims.head_dim),
            ctypes.c_int(dims.aligned_head_dim),
            ctypes.c_float(dims.eps),
            ptr_void(buf.scratch),
        )


def make_buffers(dims: AttnDims, qkv_scratch_bytes: int, scratch_bytes: int) -> AttnBuffers:
    x = np.random.randn(dims.tokens, dims.aligned_embed_dim).astype(np.float32)
    gamma = np.random.randn(dims.aligned_embed_dim).astype(np.float32)

    q = np.zeros((dims.num_heads, dims.tokens, dims.aligned_head_dim), dtype=np.float32)
    kv_cache_k = np.zeros((dims.num_kv_heads, dims.cache_capacity, dims.aligned_head_dim), dtype=np.float32)
    kv_cache_v = np.zeros_like(kv_cache_k)
    attn_out = np.zeros_like(q)
    proj_in = np.zeros((dims.tokens, dims.aligned_embed_dim), dtype=np.float32)
    out_ref = np.zeros_like(proj_in)
    out_fused = np.zeros_like(proj_in)

    qkv_scratch = np.zeros(qkv_scratch_bytes, dtype=np.uint8)
    scratch = np.zeros(scratch_bytes, dtype=np.uint8)

    return AttnBuffers(
        x=x,
        gamma=gamma,
        q=q,
        kv_cache_k=kv_cache_k,
        kv_cache_v=kv_cache_v,
        attn_out=attn_out,
        proj_in=proj_in,
        out_ref=out_ref,
        out_fused=out_fused,
        qkv_scratch=qkv_scratch,
        scratch=scratch,
    )


def make_synthetic_weights(dims: AttnDims, wv_dt: int) -> AttnWeights:
    wq = make_q5_0_weights(dims.num_heads * dims.aligned_head_dim, dims.aligned_embed_dim)
    wk = make_q5_0_weights(dims.num_kv_heads * dims.aligned_head_dim, dims.aligned_embed_dim)

    if wv_dt == CK_DT_Q8_0:
        wv = make_q8_0_weights(dims.num_kv_heads * dims.aligned_head_dim, dims.aligned_embed_dim)
    else:
        wv = make_q5_0_weights(dims.num_kv_heads * dims.aligned_head_dim, dims.aligned_embed_dim)

    wo = make_q5_0_weights(dims.aligned_embed_dim, dims.aligned_embed_dim)

    bq = np.random.randn(dims.num_heads * dims.aligned_head_dim).astype(np.float32)
    bk = np.random.randn(dims.num_kv_heads * dims.aligned_head_dim).astype(np.float32)
    bv = np.random.randn(dims.num_kv_heads * dims.aligned_head_dim).astype(np.float32)

    return AttnWeights(
        wq=wq, wk=wk, wv=wv, wo=wo,
        bq=bq, bk=bk, bv=bv,
        wq_dt=CK_DT_Q5_0, wk_dt=CK_DT_Q5_0, wv_dt=wv_dt, wo_dt=CK_DT_Q5_0
    )


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


def find_layers_by_wv_dtype(entries: dict) -> dict:
    layers = {"q8_0": [], "q5_0": []}
    for name, entry in entries.items():
        if name.startswith("layer.") and name.endswith(".wv"):
            layer = int(name.split(".")[1])
            dt = entry["dtype"]
            if dt in layers:
                layers[dt].append(layer)
    return layers


def run_bench_for_dims(bench: AttnBench, dims: AttnDims, weights: AttnWeights,
                       warmup: int, iters: int) -> BenchResult:
    qkv_scratch_bytes = bench.lib.fused_rmsnorm_qkv_prefill_head_major_quant_scratch_size(
        ctypes.c_int(dims.aligned_embed_dim)
    )
    scratch_bytes = bench.lib.mega_fused_attention_prefill_scratch_size(
        ctypes.c_int(dims.tokens), ctypes.c_int(dims.aligned_embed_dim),
        ctypes.c_int(dims.num_heads), ctypes.c_int(dims.aligned_head_dim)
    )
    buf = make_buffers(dims, qkv_scratch_bytes, scratch_bytes)

    baseline_fn = lambda: bench.run_baseline(dims, weights, buf)
    fused_fn = lambda: bench.run_fused(dims, weights, buf)

    baseline = time_function(baseline_fn, warmup=warmup, iterations=iters, name="baseline")
    fused = time_function(fused_fn, warmup=warmup, iterations=iters, name="fused")
    return BenchResult(baseline=baseline, fused=fused)


def print_result(label: str, dims: AttnDims, res: BenchResult):
    print(
        f"{label:<24} tokens={dims.tokens:<4d} "
        f"baseline={res.baseline.mean_us:9.1f} us "
        f"fused={res.fused.mean_us:9.1f} us "
        f"speedup={res.speedup:5.2f}x"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark mega_fused_attention_prefill.")
    ap.add_argument("--model-dir", default="/home/antshiv/.cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF")
    ap.add_argument("--seq-lens", default="32,64,128,256")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--q8-outproj", action="store_true",
                    help="Enable head-major Q8_0 out-proj fast path (sets CK_Q8_0_OUTPROJ=1).")
    args = ap.parse_args()

    if args.q8_outproj:
        os.environ["CK_Q8_0_OUTPROJ"] = "1"

    seq_lens = [int(x) for x in args.seq_lens.split(",") if x]

    bench = AttnBench()

    cfg = load_config(args.model_dir)
    embed_dim = int(cfg["hidden_size"])
    num_heads = int(cfg["num_attention_heads"])
    num_kv_heads = int(cfg["num_key_value_heads"])
    head_dim = embed_dim // num_heads
    aligned_embed_dim = embed_dim
    aligned_head_dim = head_dim
    eps = float(cfg.get("rms_norm_eps", 1e-6))

    print("[bench] dims:", {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
    })

    print("\n[SYNTHETIC] wq/wk/wo=q5_0, wv=q5_0")
    for tokens in seq_lens:
        dims = AttnDims(tokens, embed_dim, aligned_embed_dim, num_heads, num_kv_heads,
                        head_dim, aligned_head_dim, tokens, eps)
        weights = make_synthetic_weights(dims, wv_dt=CK_DT_Q5_0)
        res = run_bench_for_dims(bench, dims, weights, args.warmup, args.iters)
        print_result("synthetic-q5wv", dims, res)

    print("\n[SYNTHETIC] wq/wk/wo=q5_0, wv=q8_0")
    for tokens in seq_lens:
        dims = AttnDims(tokens, embed_dim, aligned_embed_dim, num_heads, num_kv_heads,
                        head_dim, aligned_head_dim, tokens, eps)
        weights = make_synthetic_weights(dims, wv_dt=CK_DT_Q8_0)
        res = run_bench_for_dims(bench, dims, weights, args.warmup, args.iters)
        print_result("synthetic-q8wv", dims, res)

    entries = load_manifest(args.model_dir)
    layers = find_layers_by_wv_dtype(entries)
    bump_path = os.path.join(args.model_dir, "weights.bump")

    real_layers = []
    if layers["q8_0"]:
        real_layers.append(layers["q8_0"][0])
    if layers["q5_0"]:
        real_layers.append(layers["q5_0"][0])

    if real_layers:
        print("\n[REAL] weights.bump per-layer")

    for layer in real_layers:
        wq_e = entries[f"layer.{layer}.wq"]
        wk_e = entries[f"layer.{layer}.wk"]
        wv_e = entries[f"layer.{layer}.wv"]
        wo_e = entries[f"layer.{layer}.wo"]
        bq_e = entries[f"layer.{layer}.bq"]
        bk_e = entries[f"layer.{layer}.bk"]
        bv_e = entries[f"layer.{layer}.bv"]
        ln1_e = entries[f"layer.{layer}.ln1_gamma"]

        wq = read_weight_bytes(bump_path, wq_e)
        wk = read_weight_bytes(bump_path, wk_e)
        wv = read_weight_bytes(bump_path, wv_e)
        wo = read_weight_bytes(bump_path, wo_e)
        bq = read_weight_f32(bump_path, bq_e)
        bk = read_weight_f32(bump_path, bk_e)
        bv = read_weight_f32(bump_path, bv_e)
        gamma = read_weight_f32(bump_path, ln1_e)

        weights = AttnWeights(
            wq=wq, wk=wk, wv=wv, wo=wo,
            bq=bq, bk=bk, bv=bv,
            wq_dt=DTYPE_STR_TO_CK[wq_e["dtype"]],
            wk_dt=DTYPE_STR_TO_CK[wk_e["dtype"]],
            wv_dt=DTYPE_STR_TO_CK[wv_e["dtype"]],
            wo_dt=DTYPE_STR_TO_CK[wo_e["dtype"]],
        )

        dtype_str = wv_e["dtype"]
        for tokens in seq_lens:
            dims = AttnDims(tokens, embed_dim, aligned_embed_dim, num_heads, num_kv_heads,
                            head_dim, aligned_head_dim, tokens, eps)
            qkv_scratch_bytes = bench.lib.fused_rmsnorm_qkv_prefill_head_major_quant_scratch_size(
                ctypes.c_int(dims.aligned_embed_dim)
            )
            scratch_bytes = bench.lib.mega_fused_attention_prefill_scratch_size(
                ctypes.c_int(dims.tokens), ctypes.c_int(dims.aligned_embed_dim),
                ctypes.c_int(dims.num_heads), ctypes.c_int(dims.aligned_head_dim)
            )
            buf = make_buffers(dims, qkv_scratch_bytes, scratch_bytes)
            buf.gamma[:] = gamma

            baseline_fn = lambda: bench.run_baseline(dims, weights, buf)
            fused_fn = lambda: bench.run_fused(dims, weights, buf)

            baseline = time_function(baseline_fn, warmup=args.warmup, iterations=args.iters, name="baseline")
            fused = time_function(fused_fn, warmup=args.warmup, iterations=args.iters, name="fused")
            res = BenchResult(baseline=baseline, fused=fused)

            print_result(f"real-L{layer}-{dtype_str}", dims, res)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
