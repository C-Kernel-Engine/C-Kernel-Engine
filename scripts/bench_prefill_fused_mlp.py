#!/usr/bin/env python3
"""
Benchmark fused_mlp_swiglu_prefill_w1w2_quant vs baseline (quantize + GEMMs).

Runs two modes:
- synthetic: random inputs/weights
- real: weights from weights.bump + weights_manifest.json
"""
import argparse
import ctypes
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

# Add unittest dir to path for lib_loader/test_utils
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
class MLPDims:
    tokens: int
    embed_dim: int
    aligned_embed_dim: int
    intermediate: int
    aligned_intermediate: int


@dataclass
class MLPWeights:
    w1: np.ndarray
    w2: np.ndarray
    w1_dt: int
    w2_dt: int


@dataclass
class MLPBuffers:
    x: np.ndarray
    q8_rows: np.ndarray
    gate: np.ndarray
    up: np.ndarray
    hidden: np.ndarray
    q8k_rows: np.ndarray
    out_ref: np.ndarray
    out_fused: np.ndarray
    scratch: np.ndarray


@dataclass
class BenchResult:
    baseline: TimingResult
    fused: TimingResult

    @property
    def speedup(self) -> float:
        return self.baseline.mean_us / self.fused.mean_us


class MLPBench:
    def __init__(self):
        self.lib = load_lib("libckernel_engine.so")
        self._bind()

    def _bind(self):
        lib = self.lib
        lib.quantize_row_q8_0.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.quantize_row_q8_0.restype = None

        lib.quantize_row_q8_k.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.quantize_row_q8_k.restype = None

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

        lib.gemm_nt_q4_k_q8_k.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.gemm_nt_q4_k_q8_k.restype = None

        lib.gemm_nt_q6_k_q8_k.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.gemm_nt_q6_k_q8_k.restype = None

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

    def run_baseline(self, dims: MLPDims, weights: MLPWeights, buf: MLPBuffers):
        lib = self.lib

        q8_row_bytes = row_bytes(CK_DT_Q8_0, dims.aligned_embed_dim)
        for t in range(dims.tokens):
            row_ptr = ptr_f32(buf.x[t])
            dst_ptr = ctypes.c_void_p(buf.q8_rows.ctypes.data + t * q8_row_bytes)
            lib.quantize_row_q8_0(row_ptr, dst_ptr, ctypes.c_int(dims.aligned_embed_dim))

        w1_row_bytes = row_bytes(weights.w1_dt, dims.aligned_embed_dim)
        w1_gate_ptr = ctypes.c_void_p(weights.w1.ctypes.data)
        w1_up_ptr = ctypes.c_void_p(weights.w1.ctypes.data + dims.aligned_intermediate * w1_row_bytes)

        lib.gemm_nt_q5_0_q8_0(ptr_void(buf.q8_rows), w1_gate_ptr, None,
                              ptr_f32(buf.gate),
                              ctypes.c_int(dims.tokens), ctypes.c_int(dims.aligned_intermediate),
                              ctypes.c_int(dims.aligned_embed_dim))
        lib.gemm_nt_q5_0_q8_0(ptr_void(buf.q8_rows), w1_up_ptr, None,
                              ptr_f32(buf.up),
                              ctypes.c_int(dims.tokens), ctypes.c_int(dims.aligned_intermediate),
                              ctypes.c_int(dims.aligned_embed_dim))

        gate_clamped = np.clip(buf.gate, -20.0, 20.0)
        buf.hidden[:] = gate_clamped / (1.0 + np.exp(-gate_clamped)) * buf.up

        q8k_row_bytes = row_bytes(CK_DT_Q8_K, dims.aligned_intermediate)
        for t in range(dims.tokens):
            row_ptr = ptr_f32(buf.hidden[t])
            dst_ptr = ctypes.c_void_p(buf.q8k_rows.ctypes.data + t * q8k_row_bytes)
            lib.quantize_row_q8_k(row_ptr, dst_ptr, ctypes.c_int(dims.aligned_intermediate))

        if weights.w2_dt == CK_DT_Q4_K:
            lib.gemm_nt_q4_k_q8_k(ptr_void(buf.q8k_rows), ptr_void(weights.w2), None,
                                  ptr_f32(buf.out_ref),
                                  ctypes.c_int(dims.tokens), ctypes.c_int(dims.aligned_embed_dim),
                                  ctypes.c_int(dims.aligned_intermediate))
        else:
            lib.gemm_nt_q6_k_q8_k(ptr_void(buf.q8k_rows), ptr_void(weights.w2), None,
                                  ptr_f32(buf.out_ref),
                                  ctypes.c_int(dims.tokens), ctypes.c_int(dims.aligned_embed_dim),
                                  ctypes.c_int(dims.aligned_intermediate))

    def run_fused(self, dims: MLPDims, weights: MLPWeights, buf: MLPBuffers):
        self.lib.fused_mlp_swiglu_prefill_w1w2_quant(
            ptr_f32(buf.x),
            ptr_void(weights.w1),
            None,
            ctypes.c_int(weights.w1_dt),
            ptr_void(weights.w2),
            None,
            ctypes.c_int(weights.w2_dt),
            ptr_f32(buf.out_fused),
            ctypes.c_int(dims.tokens),
            ctypes.c_int(dims.embed_dim),
            ctypes.c_int(dims.aligned_embed_dim),
            ctypes.c_int(dims.intermediate),
            ctypes.c_int(dims.aligned_intermediate),
            ptr_void(buf.scratch),
        )


def make_buffers(dims: MLPDims, scratch_bytes: int) -> MLPBuffers:
    x = np.random.randn(dims.tokens, dims.aligned_embed_dim).astype(np.float32)
    q8_row_bytes = row_bytes(CK_DT_Q8_0, dims.aligned_embed_dim)
    q8_rows = np.zeros(dims.tokens * q8_row_bytes, dtype=np.uint8)
    gate = np.zeros((dims.tokens, dims.aligned_intermediate), dtype=np.float32)
    up = np.zeros_like(gate)
    hidden = np.zeros_like(gate)
    q8k_row_bytes = row_bytes(CK_DT_Q8_K, dims.aligned_intermediate)
    q8k_rows = np.zeros(dims.tokens * q8k_row_bytes, dtype=np.uint8)
    out_ref = np.zeros((dims.tokens, dims.aligned_embed_dim), dtype=np.float32)
    out_fused = np.zeros_like(out_ref)
    scratch = np.zeros(scratch_bytes, dtype=np.uint8)

    return MLPBuffers(
        x=x,
        q8_rows=q8_rows,
        gate=gate,
        up=up,
        hidden=hidden,
        q8k_rows=q8k_rows,
        out_ref=out_ref,
        out_fused=out_fused,
        scratch=scratch,
    )


def make_synthetic_weights(dims: MLPDims, w2_dt: int) -> MLPWeights:
    w1_dt = CK_DT_Q5_0
    w1 = make_q5_0_weights(2 * dims.aligned_intermediate, dims.aligned_embed_dim)
    if w2_dt == CK_DT_Q4_K:
        w2 = make_q4_k_weights(dims.aligned_embed_dim, dims.aligned_intermediate)
    else:
        w2 = make_q6_k_weights(dims.aligned_embed_dim, dims.aligned_intermediate)

    return MLPWeights(w1=w1, w2=w2, w1_dt=w1_dt, w2_dt=w2_dt)


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


def find_layers_by_w2_dtype(entries: dict) -> dict:
    layers = {"q4_k": [], "q6_k": []}
    for name, entry in entries.items():
        if name.startswith("layer.") and name.endswith(".w2"):
            layer = int(name.split(".")[1])
            dt = entry["dtype"]
            if dt in layers:
                layers[dt].append(layer)
    return layers


def run_bench_for_dims(bench: MLPBench, dims: MLPDims, weights: MLPWeights,
                       warmup: int, iters: int) -> BenchResult:
    scratch_bytes = bench.lib.fused_mlp_swiglu_prefill_w1w2_quant_scratch_size(
        ctypes.c_int(dims.aligned_embed_dim), ctypes.c_int(dims.aligned_intermediate)
    )
    buf = make_buffers(dims, scratch_bytes)

    baseline_fn = lambda: bench.run_baseline(dims, weights, buf)
    fused_fn = lambda: bench.run_fused(dims, weights, buf)

    baseline = time_function(baseline_fn, warmup=warmup, iterations=iters, name="baseline")
    fused = time_function(fused_fn, warmup=warmup, iterations=iters, name="fused")
    return BenchResult(baseline=baseline, fused=fused)


def print_result(label: str, dims: MLPDims, res: BenchResult):
    print(
        f"{label:<24} tokens={dims.tokens:<4d} "
        f"baseline={res.baseline.mean_us:9.1f} us "
        f"fused={res.fused.mean_us:9.1f} us "
        f"speedup={res.speedup:5.2f}x"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark fused prefill MLP (quantized).")
    ap.add_argument("--model-dir", default="/home/antshiv/.cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF")
    ap.add_argument("--seq-lens", default="32,64,128,256,512")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    seq_lens = [int(x) for x in args.seq_lens.split(",") if x]

    bench = MLPBench()

    cfg = load_config(args.model_dir)
    embed_dim = int(cfg["hidden_size"])
    intermediate = int(cfg["intermediate_size"])
    aligned_embed_dim = embed_dim
    aligned_intermediate = intermediate

    print("[bench] dims:", {
        "embed_dim": embed_dim,
        "intermediate": intermediate,
    })

    print("\n[SYNTHETIC] w1=q5_0, w2=q6_k")
    for tokens in seq_lens:
        dims = MLPDims(tokens, embed_dim, aligned_embed_dim, intermediate, aligned_intermediate)
        weights = make_synthetic_weights(dims, w2_dt=CK_DT_Q6_K)
        res = run_bench_for_dims(bench, dims, weights, args.warmup, args.iters)
        print_result("synthetic-q6k", dims, res)

    print("\n[SYNTHETIC] w1=q5_0, w2=q4_k")
    for tokens in seq_lens:
        dims = MLPDims(tokens, embed_dim, aligned_embed_dim, intermediate, aligned_intermediate)
        weights = make_synthetic_weights(dims, w2_dt=CK_DT_Q4_K)
        res = run_bench_for_dims(bench, dims, weights, args.warmup, args.iters)
        print_result("synthetic-q4k", dims, res)

    entries = load_manifest(args.model_dir)
    layers = find_layers_by_w2_dtype(entries)
    bump_path = os.path.join(args.model_dir, "weights.bump")

    real_layers = []
    if layers["q6_k"]:
        real_layers.append(layers["q6_k"][0])
    if layers["q4_k"]:
        real_layers.append(layers["q4_k"][0])

    if real_layers:
        print("\n[REAL] weights.bump per-layer")

    for layer in real_layers:
        w1_entry = entries[f"layer.{layer}.w1"]
        w2_entry = entries[f"layer.{layer}.w2"]
        w1 = read_weight_bytes(bump_path, w1_entry)
        w2 = read_weight_bytes(bump_path, w2_entry)
        w1_dt = DTYPE_STR_TO_CK[w1_entry["dtype"]]
        w2_dt = DTYPE_STR_TO_CK[w2_entry["dtype"]]

        weights = MLPWeights(w1=w1, w2=w2, w1_dt=w1_dt, w2_dt=w2_dt)
        dtype_str = w2_entry["dtype"]

        for tokens in seq_lens:
            dims = MLPDims(tokens, embed_dim, aligned_embed_dim, intermediate, aligned_intermediate)
            res = run_bench_for_dims(bench, dims, weights, args.warmup, args.iters)
            print_result(f"real-L{layer}-{dtype_str}", dims, res)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
