#!/usr/bin/env python3
"""
Benchmark fused_rmsnorm_qkv_prefill_head_major_quant vs baseline (rmsnorm + q8 + gemm).

Runs two modes:
- synthetic: random inputs/weights (random bytes for quant weights)
- real: weights from weights.bump + weights_manifest.json
"""
import argparse
import ctypes
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# Add unittest dir to path for lib_loader/test_utils
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "unittest"))

from lib_loader import load_lib  # type: ignore
from test_utils import time_function, TimingResult  # type: ignore

# CKDataType enum values (must match include/ckernel_dtype.h)
CK_DT_Q8_0 = 9
CK_DT_Q5_0 = 11

DTYPE_STR_TO_CK = {
    "q8_0": CK_DT_Q8_0,
    "q5_0": CK_DT_Q5_0,
}


def ck_dtype_row_bytes(dt: int, n_elements: int) -> int:
    # Quantized block sizes (per include/ckernel_dtype.h)
    if dt == CK_DT_Q8_0:
        block_size, block_bytes = 32, 34
    elif dt == CK_DT_Q5_0:
        block_size, block_bytes = 32, 22
    else:
        raise ValueError(f"Unsupported dt={dt}")
    n_blocks = (n_elements + block_size - 1) // block_size
    return n_blocks * block_bytes


def ptr_f32(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def ptr_void(arr: np.ndarray):
    return ctypes.c_void_p(arr.ctypes.data)


@dataclass
class QKVDims:
    tokens: int
    embed_dim: int
    aligned_embed_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    aligned_head_dim: int
    kv_stride_tokens: int
    eps: float


@dataclass
class QKVWeights:
    wq: np.ndarray
    wk: np.ndarray
    wv: np.ndarray
    bq: np.ndarray
    bk: np.ndarray
    bv: np.ndarray
    wq_dt: int
    wk_dt: int
    wv_dt: int


@dataclass
class QKVBuffers:
    x: np.ndarray
    gamma: np.ndarray
    normed: np.ndarray
    q8_rows: np.ndarray
    q_out: np.ndarray
    k_out: np.ndarray
    v_out: np.ndarray
    q_fused: np.ndarray
    k_fused: np.ndarray
    v_fused: np.ndarray
    scratch: np.ndarray


@dataclass
class BenchResult:
    baseline: TimingResult
    fused: TimingResult

    @property
    def speedup(self) -> float:
        return self.baseline.mean_us / self.fused.mean_us


class QKVBench:
    def __init__(self):
        self.lib = load_lib("libckernel_engine.so")
        self._bind()

    def _bind(self):
        lib = self.lib
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

        lib.quantize_row_q8_0.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.quantize_row_q8_0.restype = None

        lib.gemm_nt_q8_0_q8_0.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.gemm_nt_q8_0_q8_0.restype = None

        lib.gemm_nt_q5_0_q8_0.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.gemm_nt_q5_0_q8_0.restype = None

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

        lib.fused_rmsnorm_qkv_prefill_head_major_quant_scratch_size.argtypes = [
            ctypes.c_int
        ]
        lib.fused_rmsnorm_qkv_prefill_head_major_quant_scratch_size.restype = ctypes.c_size_t

    def _gemm_dispatch(self, A_q8_ptr, B_ptr, bias_ptr, C_ptr, M, N, K, dt):
        if dt == CK_DT_Q5_0:
            self.lib.gemm_nt_q5_0_q8_0(A_q8_ptr, B_ptr, bias_ptr, C_ptr, M, N, K)
        elif dt == CK_DT_Q8_0:
            self.lib.gemm_nt_q8_0_q8_0(A_q8_ptr, B_ptr, bias_ptr, C_ptr, M, N, K)
        else:
            raise ValueError(f"Unsupported dt={dt}")

    def run_baseline(self, dims: QKVDims, weights: QKVWeights, buf: QKVBuffers):
        lib = self.lib
        # RMSNorm
        lib.rmsnorm_forward(
            ptr_f32(buf.x),
            ptr_f32(buf.gamma),
            ptr_f32(buf.normed),
            None,
            ctypes.c_int(dims.tokens),
            ctypes.c_int(dims.embed_dim),
            ctypes.c_int(dims.aligned_embed_dim),
            ctypes.c_float(dims.eps),
        )

        # Quantize to Q8_0 per token
        q8_row_bytes = ck_dtype_row_bytes(CK_DT_Q8_0, dims.aligned_embed_dim)
        for t in range(dims.tokens):
            row_ptr = ptr_f32(buf.normed[t])
            dst_ptr = ctypes.c_void_p(buf.q8_rows.ctypes.data + t * q8_row_bytes)
            lib.quantize_row_q8_0(row_ptr, dst_ptr, ctypes.c_int(dims.aligned_embed_dim))

        A_q8_ptr = ptr_void(buf.q8_rows)

        # Q heads
        wq_row_bytes = ck_dtype_row_bytes(weights.wq_dt, dims.aligned_embed_dim)
        for h in range(dims.num_heads):
            wq_head_ptr = ctypes.c_void_p(weights.wq.ctypes.data + h * dims.aligned_head_dim * wq_row_bytes)
            bq_head_ptr = ptr_f32(weights.bq[h * dims.aligned_head_dim:(h + 1) * dims.aligned_head_dim])
            out_ptr = ptr_f32(buf.q_out[h])
            self._gemm_dispatch(
                A_q8_ptr, wq_head_ptr, bq_head_ptr, out_ptr,
                dims.tokens, dims.aligned_head_dim, dims.aligned_embed_dim, weights.wq_dt
            )

        # K/V heads
        wk_row_bytes = ck_dtype_row_bytes(weights.wk_dt, dims.aligned_embed_dim)
        wv_row_bytes = ck_dtype_row_bytes(weights.wv_dt, dims.aligned_embed_dim)
        for h in range(dims.num_kv_heads):
            wk_head_ptr = ctypes.c_void_p(weights.wk.ctypes.data + h * dims.aligned_head_dim * wk_row_bytes)
            wv_head_ptr = ctypes.c_void_p(weights.wv.ctypes.data + h * dims.aligned_head_dim * wv_row_bytes)
            bk_head_ptr = ptr_f32(weights.bk[h * dims.aligned_head_dim:(h + 1) * dims.aligned_head_dim])
            bv_head_ptr = ptr_f32(weights.bv[h * dims.aligned_head_dim:(h + 1) * dims.aligned_head_dim])
            self._gemm_dispatch(
                A_q8_ptr, wk_head_ptr, bk_head_ptr, ptr_f32(buf.k_out[h]),
                dims.tokens, dims.aligned_head_dim, dims.aligned_embed_dim, weights.wk_dt
            )
            self._gemm_dispatch(
                A_q8_ptr, wv_head_ptr, bv_head_ptr, ptr_f32(buf.v_out[h]),
                dims.tokens, dims.aligned_head_dim, dims.aligned_embed_dim, weights.wv_dt
            )

    def run_fused(self, dims: QKVDims, weights: QKVWeights, buf: QKVBuffers):
        self.lib.fused_rmsnorm_qkv_prefill_head_major_quant(
            ptr_f32(buf.x),
            ptr_f32(buf.gamma),
            ptr_void(weights.wq), ptr_f32(weights.bq), ctypes.c_int(weights.wq_dt),
            ptr_void(weights.wk), ptr_f32(weights.bk), ctypes.c_int(weights.wk_dt),
            ptr_void(weights.wv), ptr_f32(weights.bv), ctypes.c_int(weights.wv_dt),
            ptr_f32(buf.q_fused),
            ptr_f32(buf.k_fused),
            ptr_f32(buf.v_fused),
            ctypes.c_int(dims.tokens),
            ctypes.c_int(dims.embed_dim),
            ctypes.c_int(dims.aligned_embed_dim),
            ctypes.c_int(dims.num_heads),
            ctypes.c_int(dims.num_kv_heads),
            ctypes.c_int(dims.head_dim),
            ctypes.c_int(dims.aligned_head_dim),
            ctypes.c_int(dims.kv_stride_tokens),
            ctypes.c_float(dims.eps),
            ptr_void(buf.scratch),
        )


def align_up(n: int, align: int) -> int:
    return ((n + align - 1) // align) * align


def make_buffers(dims: QKVDims, scratch_bytes: int) -> QKVBuffers:
    x = np.random.randn(dims.tokens, dims.aligned_embed_dim).astype(np.float32)
    gamma = np.random.randn(dims.aligned_embed_dim).astype(np.float32)
    normed = np.zeros_like(x)
    q8_row_bytes = ck_dtype_row_bytes(CK_DT_Q8_0, dims.aligned_embed_dim)
    q8_rows = np.zeros(dims.tokens * q8_row_bytes, dtype=np.uint8)

    q_out = np.zeros((dims.num_heads, dims.tokens, dims.aligned_head_dim), dtype=np.float32)
    k_out = np.zeros((dims.num_kv_heads, dims.tokens, dims.aligned_head_dim), dtype=np.float32)
    v_out = np.zeros((dims.num_kv_heads, dims.tokens, dims.aligned_head_dim), dtype=np.float32)

    q_fused = np.zeros_like(q_out)
    k_fused = np.zeros_like(k_out)
    v_fused = np.zeros_like(v_out)

    scratch = np.zeros(scratch_bytes, dtype=np.uint8)

    return QKVBuffers(
        x=x,
        gamma=gamma,
        normed=normed,
        q8_rows=q8_rows,
        q_out=q_out,
        k_out=k_out,
        v_out=v_out,
        q_fused=q_fused,
        k_fused=k_fused,
        v_fused=v_fused,
        scratch=scratch,
    )


def make_synthetic_weights(dims: QKVDims, wv_dt: int) -> QKVWeights:
    wq_dt = CK_DT_Q5_0
    wk_dt = CK_DT_Q5_0
    wq_row_bytes = ck_dtype_row_bytes(wq_dt, dims.aligned_embed_dim)
    wk_row_bytes = ck_dtype_row_bytes(wk_dt, dims.aligned_embed_dim)
    wv_row_bytes = ck_dtype_row_bytes(wv_dt, dims.aligned_embed_dim)

    wq = np.random.randint(0, 256, size=dims.num_heads * dims.aligned_head_dim * wq_row_bytes, dtype=np.uint8)
    wk = np.random.randint(0, 256, size=dims.num_kv_heads * dims.aligned_head_dim * wk_row_bytes, dtype=np.uint8)
    wv = np.random.randint(0, 256, size=dims.num_kv_heads * dims.aligned_head_dim * wv_row_bytes, dtype=np.uint8)

    bq = np.random.randn(dims.num_heads * dims.aligned_head_dim).astype(np.float32)
    bk = np.random.randn(dims.num_kv_heads * dims.aligned_head_dim).astype(np.float32)
    bv = np.random.randn(dims.num_kv_heads * dims.aligned_head_dim).astype(np.float32)

    return QKVWeights(
        wq=wq, wk=wk, wv=wv,
        bq=bq, bk=bk, bv=bv,
        wq_dt=wq_dt, wk_dt=wk_dt, wv_dt=wv_dt,
    )


def load_manifest(model_dir: str) -> Dict[str, dict]:
    manifest_path = os.path.join(model_dir, "weights_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    entries = {e["name"]: e for e in manifest["entries"]}
    return entries


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


def load_real_weights(model_dir: str, layer: int, dims: QKVDims) -> QKVWeights:
    entries = load_manifest(model_dir)
    bump_path = os.path.join(model_dir, "weights.bump")

    def entry(name: str) -> dict:
        key = f"layer.{layer}.{name}"
        if key not in entries:
            raise KeyError(f"Missing entry {key}")
        return entries[key]

    wq_e = entry("wq")
    wk_e = entry("wk")
    wv_e = entry("wv")
    bq_e = entry("bq")
    bk_e = entry("bk")
    bv_e = entry("bv")
    gamma_e = entry("ln1_gamma")

    wq_dt = DTYPE_STR_TO_CK[wq_e["dtype"]]
    wk_dt = DTYPE_STR_TO_CK[wk_e["dtype"]]
    wv_dt = DTYPE_STR_TO_CK[wv_e["dtype"]]

    wq = read_weight_bytes(bump_path, wq_e)
    wk = read_weight_bytes(bump_path, wk_e)
    wv = read_weight_bytes(bump_path, wv_e)

    bq = read_weight_f32(bump_path, bq_e)
    bk = read_weight_f32(bump_path, bk_e)
    bv = read_weight_f32(bump_path, bv_e)
    gamma = read_weight_f32(bump_path, gamma_e)

    # Sanity: sizes match expected rows
    wq_row_bytes = ck_dtype_row_bytes(wq_dt, dims.aligned_embed_dim)
    wk_row_bytes = ck_dtype_row_bytes(wk_dt, dims.aligned_embed_dim)
    wv_row_bytes = ck_dtype_row_bytes(wv_dt, dims.aligned_embed_dim)
    exp_wq = dims.num_heads * dims.aligned_head_dim * wq_row_bytes
    exp_wk = dims.num_kv_heads * dims.aligned_head_dim * wk_row_bytes
    exp_wv = dims.num_kv_heads * dims.aligned_head_dim * wv_row_bytes
    if wq.size != exp_wq or wk.size != exp_wk or wv.size != exp_wv:
        raise ValueError("Weight size mismatch vs expected row bytes")

    return QKVWeights(
        wq=wq, wk=wk, wv=wv,
        bq=bq, bk=bk, bv=bv,
        wq_dt=wq_dt, wk_dt=wk_dt, wv_dt=wv_dt,
    ), gamma


def load_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def run_bench_for_dims(
    bench: QKVBench,
    dims: QKVDims,
    weights: QKVWeights,
    gamma: np.ndarray,
    warmup: int,
    iters: int,
) -> BenchResult:
    scratch_bytes = bench.lib.fused_rmsnorm_qkv_prefill_head_major_quant_scratch_size(
        ctypes.c_int(dims.aligned_embed_dim)
    )
    buf = make_buffers(dims, scratch_bytes)
    buf.gamma[:] = gamma

    baseline_fn = lambda: bench.run_baseline(dims, weights, buf)
    fused_fn = lambda: bench.run_fused(dims, weights, buf)

    baseline = time_function(baseline_fn, warmup=warmup, iterations=iters, name="baseline")
    fused = time_function(fused_fn, warmup=warmup, iterations=iters, name="fused")
    return BenchResult(baseline=baseline, fused=fused)


def print_result(label: str, dims: QKVDims, res: BenchResult):
    print(
        f"{label:<24} tokens={dims.tokens:<4d} "
        f"baseline={res.baseline.mean_us:9.1f} us "
        f"fused={res.fused.mean_us:9.1f} us "
        f"speedup={res.speedup:5.2f}x"
    )


def find_layers_by_wv_dtype(model_dir: str) -> Dict[str, List[int]]:
    entries = load_manifest(model_dir)
    layers = {"q8_0": [], "q5_0": []}
    for name, entry in entries.items():
        if name.startswith("layer.") and name.endswith(".wv"):
            layer = int(name.split(".")[1])
            dt = entry["dtype"]
            if dt in layers:
                layers[dt].append(layer)
    return layers


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark fused prefill QKV (quantized).")
    ap.add_argument("--model-dir", default="/home/antshiv/.cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF")
    ap.add_argument("--seq-lens", default="32,64,128,256,512")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    seq_lens = [int(x) for x in args.seq_lens.split(",") if x]

    bench = QKVBench()

    # Base dims from Qwen2 config
    cfg = load_config(args.model_dir)
    embed_dim = int(cfg["hidden_size"])
    num_heads = int(cfg["num_attention_heads"])
    num_kv_heads = int(cfg["num_key_value_heads"])
    head_dim = embed_dim // num_heads
    aligned_embed_dim = align_up(embed_dim, 32)
    aligned_head_dim = align_up(head_dim, 32)
    eps = float(cfg.get("rms_norm_eps", 1e-6))

    print("[bench] dims:", {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "aligned_embed_dim": aligned_embed_dim,
        "aligned_head_dim": aligned_head_dim,
    })

    # Synthetic benchmarks
    print("\n[SYNTHETIC] wq/wk=q5_0, wv=q8_0")
    for tokens in seq_lens:
        dims = QKVDims(
            tokens=tokens,
            embed_dim=embed_dim,
            aligned_embed_dim=aligned_embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            aligned_head_dim=aligned_head_dim,
            kv_stride_tokens=tokens,
            eps=eps,
        )
        weights = make_synthetic_weights(dims, wv_dt=CK_DT_Q8_0)
        gamma = np.random.randn(aligned_embed_dim).astype(np.float32)
        res = run_bench_for_dims(bench, dims, weights, gamma, args.warmup, args.iters)
        print_result("synthetic-q8wv", dims, res)

    print("\n[SYNTHETIC] wq/wk=q5_0, wv=q5_0")
    for tokens in seq_lens:
        dims = QKVDims(
            tokens=tokens,
            embed_dim=embed_dim,
            aligned_embed_dim=aligned_embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            aligned_head_dim=aligned_head_dim,
            kv_stride_tokens=tokens,
            eps=eps,
        )
        weights = make_synthetic_weights(dims, wv_dt=CK_DT_Q5_0)
        gamma = np.random.randn(aligned_embed_dim).astype(np.float32)
        res = run_bench_for_dims(bench, dims, weights, gamma, args.warmup, args.iters)
        print_result("synthetic-q5wv", dims, res)

    # Real-weight benchmarks
    layers = find_layers_by_wv_dtype(args.model_dir)
    real_layers = []
    if layers["q8_0"]:
        real_layers.append(layers["q8_0"][0])
    if layers["q5_0"]:
        real_layers.append(layers["q5_0"][0])

    if real_layers:
        print("\n[REAL] weights.bump per-layer")
    for layer in real_layers:
        weights, gamma = load_real_weights(args.model_dir, layer, QKVDims(
            tokens=1,
            embed_dim=embed_dim,
            aligned_embed_dim=aligned_embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            aligned_head_dim=aligned_head_dim,
            kv_stride_tokens=1,
            eps=eps,
        ))
        dtype_str = "q8_0" if weights.wv_dt == CK_DT_Q8_0 else "q5_0"
        for tokens in seq_lens:
            dims = QKVDims(
                tokens=tokens,
                embed_dim=embed_dim,
                aligned_embed_dim=aligned_embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                aligned_head_dim=aligned_head_dim,
                kv_stride_tokens=tokens,
                eps=eps,
            )
            res = run_bench_for_dims(bench, dims, weights, gamma, args.warmup, args.iters)
            print_result(f"real-L{layer}-{dtype_str}", dims, res)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
