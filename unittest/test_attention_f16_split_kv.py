"""Regression tests for llama.cpp-compatible FP16 split-KV decode attention.

The Qwen3-VL mixed-prefix decode path crosses a semantic boundary at KV=512:
llama.cpp rounds Q/K through FP16, computes FP16 online-softmax partials per
worker chunk, then combines those partials in FP32 chunk order.  A conventional
single FP32 reduction is mathematically reasonable, but is not the same model
contract and can change long-decode top-1 tokens.
"""

import ctypes
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import numpy as np

from lib_loader import load_lib


lib = load_lib("libckernel_engine.so")

_FLOAT_P = ctypes.POINTER(ctypes.c_float)
_U16_P = ctypes.POINTER(ctypes.c_uint16)
_SPLIT_ARGS = [
    _FLOAT_P, _U16_P, _U16_P, _FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
lib.attention_forward_decode_head_major_gqa_flash_f16cache_split.argtypes = _SPLIT_ARGS
lib.attention_forward_decode_head_major_gqa_flash_f16cache_split.restype = None
_LEGACY_ARGS = _SPLIT_ARGS[:-1]
lib.attention_forward_decode_head_major_gqa_flash_f16cache.argtypes = _LEGACY_ARGS
lib.attention_forward_decode_head_major_gqa_flash_f16cache.restype = None
lib.attention_forward_decode_head_major_gqa_flash_f16cache_contract.argtypes = (
    _LEGACY_ARGS + [ctypes.c_int]
)
lib.attention_forward_decode_head_major_gqa_flash_f16cache_contract.restype = ctypes.c_int
lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_contract.argtypes = [
    _FLOAT_P, _U16_P, _U16_P, _FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_contract.restype = ctypes.c_int
lib.ck_get_num_threads.argtypes = []
lib.ck_get_num_threads.restype = ctypes.c_int
lib.ck_set_strict_parity.argtypes = [ctypes.c_int]
lib.ck_set_strict_parity.restype = None
lib.attention_forward_causal_head_major_gqa_flash_strided.argtypes = [
    _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
]
lib.attention_forward_causal_head_major_gqa_flash_strided.restype = None


_LLAMA_HELPER_SOURCE = r"""
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static bool read_exact(const char * path, void * dst, size_t bytes) {
    FILE * f = fopen(path, "rb");
    if (!f) return false;
    const size_t got = fread(dst, 1, bytes, f);
    fclose(f);
    return got == bytes;
}

static bool write_exact(const char * path, const void * src, size_t bytes) {
    FILE * f = fopen(path, "wb");
    if (!f) return false;
    const size_t put = fwrite(src, 1, bytes, f);
    fclose(f);
    return put == bytes;
}

static size_t tensor_offset(const ggml_tensor * t, int i0, int i1, int i2, int i3) {
    return (size_t) i0 * t->nb[0] + (size_t) i1 * t->nb[1] +
           (size_t) i2 * t->nb[2] + (size_t) i3 * t->nb[3];
}

int main(int argc, char ** argv) {
    ggml_cpu_init();
    if (argc == 2 && strcmp(argv[1], "--isa") == 0) {
        printf("avx2=%d avx_vnni=%d avx512=%d avx512_vnni=%d\n",
               ggml_cpu_has_avx2(), ggml_cpu_has_avx_vnni(),
               ggml_cpu_has_avx512(), ggml_cpu_has_avx512_vnni());
        return 0;
    }
    if (argc != 10 && argc != 11 && argc != 13) {
        fprintf(stderr, "usage: %s q.f32 k.f16 v.f16 out.f32 H Hkv KV D threads [valid_KV [Q past]]\n", argv[0]);
        return 2;
    }
    const int H = atoi(argv[5]);
    const int Hkv = atoi(argv[6]);
    const int KV = atoi(argv[7]);
    const int D = atoi(argv[8]);
    const int threads = atoi(argv[9]);
    const int valid_KV = argc >= 11 ? atoi(argv[10]) : KV;
    const int Q = argc == 13 ? atoi(argv[11]) : 1;
    const int past = argc == 13 ? atoi(argv[12]) : 0;
    if (H <= 0 || Hkv <= 0 || KV <= 0 || valid_KV <= 0 || valid_KV > KV ||
        Q <= 0 || past < 0 || past + Q > valid_KV ||
        D <= 0 || threads <= 0 || H % Hkv != 0) return 2;

    const size_t q_count = (size_t) H * (size_t) Q * (size_t) D;
    const size_t kv_count = (size_t) Hkv * (size_t) KV * (size_t) D;
    const size_t memory = 64u * 1024u * 1024u +
                          q_count * sizeof(float) +
                          (2u * kv_count + (size_t) KV) * sizeof(ggml_fp16_t);
    ggml_init_params init = { memory, nullptr, false };
    ggml_context * ctx = ggml_init(init);
    if (!ctx) return 3;
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, Q, H, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, KV, Hkv, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, KV, Hkv, 1);
    if (!read_exact(argv[1], q->data, q_count * sizeof(float)) ||
        !read_exact(argv[2], k->data, kv_count * sizeof(ggml_fp16_t)) ||
        !read_exact(argv[3], v->data, kv_count * sizeof(ggml_fp16_t))) {
        ggml_free(ctx);
        return 4;
    }

    ggml_tensor * mask = nullptr;
    if (Q > 1 || valid_KV != KV) {
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, KV, Q);
        ggml_fp16_t * mask_data = (ggml_fp16_t *) mask->data;
        for (int tq = 0; tq < Q; ++tq) {
            const int row_limit = Q > 1 ? past + tq + 1 : valid_KV;
            for (int i = 0; i < KV; ++i) {
                mask_data[(size_t) tq * (size_t) KV + (size_t) i] =
                    ggml_fp32_to_fp16(i < row_limit && i < valid_KV ? 0.0f : -INFINITY);
            }
        }
    }
    ggml_tensor * out = ggml_flash_attn_ext(
        ctx, q, k, v, mask, 1.0f / sqrtf((float) D), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_threadpool_params tp_params = ggml_threadpool_params_default(threads);
    tp_params.paused = false;
    ggml_threadpool * threadpool = ggml_threadpool_new(&tp_params);
    ggml_cplan plan = ggml_graph_plan(graph, threads, threadpool);
    plan.work_data = plan.work_size ? (uint8_t *) malloc(plan.work_size) : nullptr;
    if (plan.work_size && !plan.work_data) return 5;
    const ggml_status status = ggml_graph_compute(graph, &plan);
    if (status != GGML_STATUS_SUCCESS) return 6;

    float * host = (float *) malloc(q_count * sizeof(float));
    if (!host) return 7;
    for (int h = 0; h < H; ++h) {
        for (int tq = 0; tq < Q; ++tq) {
            for (int d = 0; d < D; ++d) {
                memcpy(host + ((size_t) h * (size_t) Q + (size_t) tq) * (size_t) D + (size_t) d,
                       (const char *) out->data + tensor_offset(out, d, h, tq, 0), sizeof(float));
            }
        }
    }
    const bool ok = write_exact(argv[4], host, q_count * sizeof(float));
    free(host);
    free(plan.work_data);
    ggml_threadpool_free(threadpool);
    ggml_free(ctx);
    return ok ? 0 : 8;
}
"""


_LLAMA_HELPER_CACHE = {}


def _llama_root():
    value = os.environ.get("V8_QWEN3VL_ENCODER_PARITY_LLAMA_CPP_ROOT", "").strip()
    return Path(value).resolve() if value else (Path(__file__).resolve().parents[1] / "llama.cpp")


def _ensure_llama_helper():
    root = _llama_root()
    bin_dir = root / "build" / "bin"
    required = [bin_dir / "libggml.so", bin_dir / "libggml-cpu.so", bin_dir / "libggml-base.so"]
    if not (root / "ggml" / "include" / "ggml.h").is_file() or not all(p.is_file() for p in required):
        raise RuntimeError(f"llama.cpp GGML headers/libraries not found under {root}")
    cache_key = (
        str(root),
        tuple((str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in required),
    )
    if cache_key in _LLAMA_HELPER_CACHE:
        return _LLAMA_HELPER_CACHE[cache_key]
    digest = hashlib.sha256()
    digest.update(_LLAMA_HELPER_SOURCE.encode("utf-8"))
    digest.update(str(root).encode("utf-8"))
    for library in required:
        digest.update(str(library.resolve()).encode("utf-8"))
        digest.update(library.read_bytes())
    fingerprint = digest.hexdigest()[:16]
    helper_dir = Path(tempfile.gettempdir()) / f"ck_attention_f16_split_kv_{fingerprint}"
    helper_dir.mkdir(parents=True, exist_ok=True)
    source = helper_dir / "llama_f16_split_kv.cpp"
    binary = helper_dir / "llama_f16_split_kv"
    if not source.is_file() or source.read_text() != _LLAMA_HELPER_SOURCE:
        source.write_text(_LLAMA_HELPER_SOURCE)
    if not binary.is_file() or binary.stat().st_mtime < source.stat().st_mtime:
        command = [
            "g++", "-O2", "-std=c++11",
            "-I", str(root / "ggml" / "include"),
            "-o", str(binary), str(source),
            "-L", str(bin_dir), "-lggml", "-lggml-cpu", "-lggml-base",
            "-lm", "-lpthread", f"-Wl,-rpath,{bin_dir}",
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"llama.cpp split-KV helper compile failed:\n{completed.stderr}")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    identity = subprocess.run(
        [str(binary), "--isa"], capture_output=True, text=True, env=env,
    )
    if identity.returncode != 0:
        raise RuntimeError(
            f"llama.cpp split-KV helper ISA query failed ({identity.returncode}):\n"
            f"{identity.stderr}"
        )
    isa = identity.stdout.strip()
    if os.environ.get("CK_REQUIRE_LLAMA_AVX512", "").strip() not in ("", "0") and "avx512=1" not in isa:
        raise RuntimeError(
            "FP16 attention oracle requires an AVX-512 llama.cpp build, "
            f"but helper reports: {isa}"
        )
    print(f"llama.cpp FP16 attention oracle: root={root} fingerprint={fingerprint} {isa}")
    result = (binary, bin_dir)
    _LLAMA_HELPER_CACHE[cache_key] = result
    return result


def _llama_split_output(q, k_bits, v_bits, head_dim, threads, padded_kv_tokens=None):
    helper, bin_dir = _ensure_llama_helper()
    heads, _ = q.shape
    kv_heads, kv_tokens, _ = k_bits.shape
    padded_kv_tokens = int(padded_kv_tokens or kv_tokens)
    if padded_kv_tokens < kv_tokens:
        raise ValueError("padded_kv_tokens cannot be smaller than the valid KV count")
    q_compact = np.ascontiguousarray(q[:, :head_dim], dtype=np.float32)
    k_compact = np.zeros((kv_heads, padded_kv_tokens, head_dim), dtype=np.uint16)
    v_compact = np.zeros((kv_heads, padded_kv_tokens, head_dim), dtype=np.uint16)
    k_compact[:, :kv_tokens, :] = np.ascontiguousarray(k_bits[:, :, :head_dim], dtype=np.uint16)
    v_compact[:, :kv_tokens, :] = np.ascontiguousarray(v_bits[:, :, :head_dim], dtype=np.uint16)
    with tempfile.TemporaryDirectory(prefix="ck_f16_split_kv_") as tmp:
        tmp = Path(tmp)
        q_path, k_path, v_path, out_path = [tmp / name for name in ("q.f32", "k.f16", "v.f16", "out.f32")]
        q_compact.tofile(q_path)
        k_compact.tofile(k_path)
        v_compact.tofile(v_path)
        command = [
            str(helper), str(q_path), str(k_path), str(v_path), str(out_path),
            str(heads), str(kv_heads), str(padded_kv_tokens), str(head_dim), str(threads),
            str(kv_tokens),
        ]
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('LD_LIBRARY_PATH', '')}"
        completed = subprocess.run(command, capture_output=True, text=True, env=env)
        if completed.returncode != 0:
            raise RuntimeError(
                f"llama.cpp split-KV helper failed ({completed.returncode}):\n{completed.stderr}"
            )
        return np.fromfile(out_path, dtype=np.float32).reshape(heads, head_dim)


def _llama_prefill_output(q, k_bits, v_bits, head_dim, past_tokens, threads):
    helper, bin_dir = _ensure_llama_helper()
    heads, q_tokens, _ = q.shape
    kv_heads, kv_tokens, _ = k_bits.shape
    q_compact = np.ascontiguousarray(q[:, :, :head_dim], dtype=np.float32)
    k_compact = np.ascontiguousarray(k_bits[:, :, :head_dim], dtype=np.uint16)
    v_compact = np.ascontiguousarray(v_bits[:, :, :head_dim], dtype=np.uint16)
    with tempfile.TemporaryDirectory(prefix="ck_f16_prefill_") as tmp:
        tmp = Path(tmp)
        q_path, k_path, v_path, out_path = [tmp / name for name in ("q.f32", "k.f16", "v.f16", "out.f32")]
        q_compact.tofile(q_path)
        k_compact.tofile(k_path)
        v_compact.tofile(v_path)
        command = [
            str(helper), str(q_path), str(k_path), str(v_path), str(out_path),
            str(heads), str(kv_heads), str(kv_tokens), str(head_dim), str(threads),
            str(kv_tokens), str(q_tokens), str(past_tokens),
        ]
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('LD_LIBRARY_PATH', '')}"
        completed = subprocess.run(command, capture_output=True, text=True, env=env)
        if completed.returncode != 0:
            raise RuntimeError(
                f"llama.cpp tiled prefill helper failed ({completed.returncode}):\n{completed.stderr}"
            )
        return np.fromfile(out_path, dtype=np.float32).reshape(heads, q_tokens, head_dim)


def _llama_kv_partition_extent(kv_tokens):
    """Model llama.cpp's padded physical decode-graph K tensor extent."""
    kv_tokens = int(kv_tokens)
    return ((kv_tokens + 255) // 256) * 256 if kv_tokens >= 512 else kv_tokens


def _f32_ptr(a):
    return a.ctypes.data_as(_FLOAT_P)


def _u16_ptr(a):
    return a.ctypes.data_as(_U16_P)


def _half_bits(a):
    return np.ascontiguousarray(a.astype(np.float16)).view(np.uint16)


def _pairwise_sum_f32(values):
    values = np.asarray(values, dtype=np.float32)
    while values.size > 1:
        half = values.size // 2
        values = (values[:half] + values[half:half * 2]).astype(np.float32)
    return np.float32(values[0])


def _dot_f16_avx512_contract(q32, k32):
    accum = [np.zeros(16, dtype=np.float32) for _ in range(4)]
    limit = q32.size - q32.size % 64
    for offset in range(0, limit, 64):
        for lane in range(4):
            begin = offset + lane * 16
            accum[lane] = (
                accum[lane] + q32[begin:begin + 16] * k32[begin:begin + 16]
            ).astype(np.float32)
    merged = ((accum[0] + accum[2]) + (accum[1] + accum[3])).astype(np.float32)
    result = _pairwise_sum_f32(merged)
    for offset in range(limit, q32.size):
        result = np.float32(result + np.float32(q32[offset] * k32[offset]))
    return result


def _split_oracle(q, k_bits, v_bits, head_dim, chunks):
    """Independent scalar-contract oracle with explicit FP16 accumulators."""
    num_heads, aligned = q.shape
    num_kv_heads, kv_tokens, _ = k_bits.shape
    chunks = max(1, min(chunks, kv_tokens))
    chunk_size = (kv_tokens + chunks - 1) // chunks
    scale = np.float32(1.0 / math.sqrt(head_dim))
    q_half = q.astype(np.float16)
    k_half = k_bits.view(np.float16)
    v_half = v_bits.view(np.float16)
    partial_max = np.full((num_heads, chunks), -np.inf, dtype=np.float32)
    partial_sum = np.zeros((num_heads, chunks), dtype=np.float32)
    partial_acc = np.zeros((num_heads, chunks, aligned), dtype=np.float32)

    for h in range(num_heads):
        kv_head = h * num_kv_heads // num_heads
        q32 = q_half[h, :head_dim].astype(np.float32)
        for chunk in range(chunks):
            begin = chunk * chunk_size
            end = min(begin + chunk_size, kv_tokens)
            maximum = np.float32(-np.inf)
            total = np.float32(0.0)
            acc = np.zeros(head_dim, dtype=np.float16)
            for token in range(begin, end):
                dot = _dot_f16_avx512_contract(
                    q32, k_half[kv_head, token, :head_dim].astype(np.float32)
                )
                score = np.float32(dot * scale)
                old_max = maximum
                max_scale = np.float32(1.0)
                value_scale = np.float32(1.0)
                if score > maximum:
                    maximum = score
                    max_scale = (
                        np.float32(math.exp(float(old_max - maximum)))
                        if np.isfinite(old_max)
                        else np.float32(0.0)
                    )
                    acc = (acc.astype(np.float32) * max_scale).astype(np.float16)
                else:
                    value_scale = np.float32(math.exp(float(score - maximum)))
                acc = (
                    acc.astype(np.float32)
                    + value_scale * v_half[kv_head, token, :head_dim].astype(np.float32)
                ).astype(np.float16)
                total = np.float32(total * max_scale + value_scale)

            partial_max[h, chunk] = maximum
            partial_sum[h, chunk] = total
            partial_acc[h, chunk, :head_dim] = acc.astype(np.float32)

    out = np.zeros((num_heads, aligned), dtype=np.float32)
    for h in range(num_heads):
        maximum = np.float32(-np.inf)
        total = np.float32(0.0)
        for chunk in range(chunks):
            chunk_sum = partial_sum[h, chunk]
            if chunk_sum == 0.0:
                continue
            new_max = np.float32(max(maximum, partial_max[h, chunk]))
            old_scale = (
                np.float32(math.exp(float(maximum - new_max)))
                if np.isfinite(maximum)
                else np.float32(0.0)
            )
            chunk_scale = np.float32(math.exp(float(partial_max[h, chunk] - new_max)))
            out[h, :head_dim] = (
                out[h, :head_dim] * old_scale
                + partial_acc[h, chunk, :head_dim] * chunk_scale
            ).astype(np.float32)
            total = np.float32(total * old_scale + chunk_sum * chunk_scale)
            maximum = new_max
        if total > 0.0:
            out[h, :head_dim] = (out[h, :head_dim] / total).astype(np.float32)
    return out


def _fp32_single_reference(q, k_bits, v_bits, head_dim):
    """The previous CK-style FP32 reduction, used only as a rejection oracle."""
    num_heads, aligned = q.shape
    num_kv_heads, _, _ = k_bits.shape
    k = k_bits.view(np.float16).astype(np.float32)
    v = v_bits.view(np.float16).astype(np.float32)
    out = np.zeros((num_heads, aligned), dtype=np.float32)
    scale = np.float32(1.0 / math.sqrt(head_dim))
    for h in range(num_heads):
        kv_head = h * num_kv_heads // num_heads
        scores = np.sum(k[kv_head, :, :head_dim] * q[h, :head_dim], axis=1, dtype=np.float32)
        scores = scores * scale
        weights = np.exp(scores - np.max(scores)).astype(np.float32)
        weights /= np.sum(weights, dtype=np.float32)
        out[h, :head_dim] = weights @ v[kv_head, :, :head_dim]
    return out


def _unfused_f16_causal_reference(q, k, v):
    heads, tokens, head_dim = q.shape
    kv_heads = k.shape[0]
    qh = q.astype(np.float16).astype(np.float32)
    kh = k.astype(np.float16).astype(np.float32)
    vh = v.astype(np.float16).astype(np.float32)
    scale = np.float32(1.0 / math.sqrt(head_dim))
    out = np.zeros_like(q)
    for h in range(heads):
        kv_head = h * kv_heads // heads
        for token in range(tokens):
            count = token + 1
            scores = np.sum(
                kh[kv_head, :count] * qh[h, token][None, :],
                axis=1,
                dtype=np.float32,
            ) * scale
            probs = np.exp(scores - np.max(scores)).astype(np.float32)
            probs /= np.sum(probs, dtype=np.float32)
            probs = probs.astype(np.float16).astype(np.float32)
            out[h, token] = np.sum(
                vh[kv_head, :count] * probs[:, None],
                axis=0,
                dtype=np.float32,
            )
    return out


def _unfused_f16_causal_case():
    rng = np.random.default_rng(20260710)
    heads, kv_heads, tokens, head_dim = 4, 2, 17, 32
    q = rng.normal(0.0, 0.7, (heads, tokens, head_dim)).astype(np.float32)
    k = rng.normal(0.0, 0.7, (kv_heads, tokens, head_dim)).astype(np.float32)
    v = rng.normal(0.0, 0.9, (kv_heads, tokens, head_dim)).astype(np.float32)
    actual = np.zeros_like(q)
    old = os.environ.get("CK_STRICT_ATTN_F16_UNFUSED")
    try:
        os.environ["CK_STRICT_ATTN_F16_UNFUSED"] = "1"
        lib.ck_set_strict_parity(1)
        lib.attention_forward_causal_head_major_gqa_flash_strided(
            _f32_ptr(q), _f32_ptr(k), _f32_ptr(v), _f32_ptr(actual),
            heads, kv_heads, tokens, head_dim, head_dim, tokens,
        )
    finally:
        lib.ck_set_strict_parity(0)
        if old is None:
            os.environ.pop("CK_STRICT_ATTN_F16_UNFUSED", None)
        else:
            os.environ["CK_STRICT_ATTN_F16_UNFUSED"] = old
    expected = _unfused_f16_causal_reference(q, k, v)
    diff = float(np.max(np.abs(actual - expected)))
    return Result(
        "unfused_f16_causal(T=17,H=4,D=32)",
        diff,
        3.0e-5,
        diff <= 3.0e-5,
    )


def _inputs(seed, heads, kv_heads, kv_tokens, head_dim, aligned):
    rng = np.random.default_rng(seed)
    q = rng.normal(0.0, 0.55, (heads, aligned)).astype(np.float32)
    k = rng.normal(0.0, 0.55, (kv_heads, kv_tokens, aligned)).astype(np.float32)
    v = rng.normal(0.0, 0.75, (kv_heads, kv_tokens, aligned)).astype(np.float32)
    if head_dim < aligned:
        q[:, head_dim:] = 0.0
        k[:, :, head_dim:] = 0.0
        v[:, :, head_dim:] = 0.0
    return q, _half_bits(k), _half_bits(v)


def _run_explicit(q, k, v, head_dim, chunks):
    heads, aligned = q.shape
    kv_heads, kv_tokens, _ = k.shape
    out = np.full_like(q, np.nan)
    lib.attention_forward_decode_head_major_gqa_flash_f16cache_split(
        _f32_ptr(q), _u16_ptr(k), _u16_ptr(v), _f32_ptr(out),
        heads, kv_heads, kv_tokens, kv_tokens, head_dim, aligned, chunks,
    )
    return out


def _run_legacy(q, k, v, head_dim):
    heads, aligned = q.shape
    kv_heads, kv_tokens, _ = k.shape
    out = np.full_like(q, np.nan)
    lib.attention_forward_decode_head_major_gqa_flash_f16cache(
        _f32_ptr(q), _u16_ptr(k), _u16_ptr(v), _f32_ptr(out),
        heads, kv_heads, kv_tokens, kv_tokens, head_dim, aligned,
    )
    return out


def _run_contract(q, k, v, head_dim, reduction, fill=np.nan):
    heads, aligned = q.shape
    kv_heads, kv_tokens, _ = k.shape
    out = np.full_like(q, fill)
    status = lib.attention_forward_decode_head_major_gqa_flash_f16cache_contract(
        _f32_ptr(q), _u16_ptr(k), _u16_ptr(v), _f32_ptr(out),
        heads, kv_heads, kv_tokens, kv_tokens, head_dim, aligned, reduction,
    )
    return status, out


def _prefill_append_matches_decode_loop_case():
    heads, kv_heads, q_tokens, past_tokens = 4, 2, 3, 5
    capacity, head_dim, aligned = 16, 8, 8
    rng = np.random.default_rng(7003)
    q = np.ascontiguousarray(rng.standard_normal((heads, q_tokens, aligned), dtype=np.float32))
    k = _half_bits(rng.standard_normal((kv_heads, capacity, aligned), dtype=np.float32))
    v = _half_bits(rng.standard_normal((kv_heads, capacity, aligned), dtype=np.float32))
    actual = np.zeros_like(q)
    status = lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_contract(
        _f32_ptr(q), _u16_ptr(k), _u16_ptr(v), _f32_ptr(actual),
        heads, kv_heads, q_tokens, past_tokens, capacity, head_dim, aligned, 2,
    )
    expected = np.zeros_like(q)
    for token in range(q_tokens):
        q_token = np.ascontiguousarray(q[:, token, :])
        out_token = np.zeros((heads, aligned), dtype=np.float32)
        token_status = lib.attention_forward_decode_head_major_gqa_flash_f16cache_contract(
            _f32_ptr(q_token), _u16_ptr(k), _u16_ptr(v), _f32_ptr(out_token),
            heads, kv_heads, past_tokens + token + 1, capacity,
            head_dim, aligned, 2,
        )
        if token_status != 0:
            status = token_status
            break
        expected[:, token, :] = out_token
    diff = float(np.max(np.abs(actual - expected)))
    return Result(
        "prefill_append_matches_decode_loop",
        diff if status == 0 else float("inf"),
        0.0,
        status == 0 and np.array_equal(actual, expected),
    )


def _prefill_auto_matches_llama_case(
    name, seed, heads, kv_heads, q_tokens, past_tokens, head_dim, threads
):
    capacity, aligned = past_tokens + q_tokens, head_dim
    rng = np.random.default_rng(seed)
    q = np.ascontiguousarray(
        rng.normal(0.0, 0.55, (heads, q_tokens, aligned)).astype(np.float32)
    )
    k = _half_bits(
        rng.normal(0.0, 0.55, (kv_heads, capacity, aligned)).astype(np.float32)
    )
    v = _half_bits(
        rng.normal(0.0, 0.75, (kv_heads, capacity, aligned)).astype(np.float32)
    )
    actual = np.zeros_like(q)
    status = lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_contract(
        _f32_ptr(q), _u16_ptr(k), _u16_ptr(v), _f32_ptr(actual),
        heads, kv_heads, q_tokens, past_tokens, capacity, head_dim, aligned, 3,
    )
    expected = _llama_prefill_output(q, k, v, head_dim, past_tokens, threads=threads)
    diff = float(np.max(np.abs(actual - expected))) if status == 0 else math.inf
    return Result(
        name,
        diff,
        0.0,
        status == 0 and np.array_equal(actual, expected),
    )


@dataclass
class Result:
    name: str
    diff: float
    tolerance: float
    passed: bool


def _case(name, seed, heads, kv_heads, kv_tokens, head_dim, aligned, chunks, tolerance):
    q, k, v = _inputs(seed, heads, kv_heads, kv_tokens, head_dim, aligned)
    actual = _run_explicit(q, k, v, head_dim, chunks)
    # The Python model above is useful for explaining the storage/reduction
    # contract, but its dot traversal models one ISA and is not authoritative
    # across AVX2, AVX-512, and NEON. Gate the production entry point against
    # GGML using the same worker/chunk count instead.
    expected = _llama_split_output(
        q, k, v, head_dim, chunks, _llama_kv_partition_extent(kv_tokens)
    )
    active_diff = float(np.max(np.abs(actual[:, :head_dim] - expected)))
    padding_diff = (
        float(np.max(np.abs(actual[:, head_dim:])))
        if aligned > head_dim
        else 0.0
    )
    diff = max(active_diff, padding_diff)
    return Result(name, diff, tolerance, diff <= tolerance), (q, k, v, actual)


def main():
    results = []
    threads = max(1, int(lib.ck_get_num_threads()))
    results.append(_prefill_append_matches_decode_loop_case())
    # Current llama.cpp changes flash-attention arithmetic at Q=64. Exercise
    # both sides of that production dispatch boundary and the actual Qwen3-VL
    # visual segment shape; leaf decode coverage cannot certify this route.
    for case in (
        ("flash_auto_short_text_before(Q=5)", 2026071505, 4, 2, 5, 0, 32),
        ("flash_auto_short_text_after(Q=14)", 2026071514, 4, 2, 14, 1013, 32),
        ("flash_auto_below_threshold(Q=63)", 2026071563, 4, 2, 63, 5, 32),
        ("flash_auto_at_threshold(Q=64)", 2026071564, 4, 2, 64, 5, 32),
        (
            "flash_auto_qwen3vl_visual(Q=1008,KV=1013,H=32,Hkv=8,D=128)",
            2026071508, 32, 8, 1008, 5, 128,
        ),
    ):
        results.append(_prefill_auto_matches_llama_case(*case, threads=threads))
    results.append(_unfused_f16_causal_case())
    below, below_data = _case(
        "f16_split_below_threshold(KV=511,C=1)", 511, 8, 2, 511, 64, 64, 1, 0.0,
    )
    results.append(below)
    threshold, threshold_data = _case(
        f"f16_split_threshold(KV=512,C={threads})",
        512, 8, 2, 512, 64, 64, threads, 0.0,
    )
    results.append(threshold)
    merge_rounding, _ = _case(
        f"f16_split_merge_rounding(KV=512,H=1,D=32,C={threads})",
        2, 1, 1, 512, 32, 32, threads, 0.0,
    )
    results.append(merge_rounding)
    qwen, qwen_data = _case(
        f"f16_split_qwen3vl(KV=1058,H=32,D=128,C={threads})",
        1058, 32, 8, 1058, 128, 128, threads, 0.0,
    )
    results.append(qwen)
    qwen_step20, qwen_step20_data = _case(
        f"f16_split_qwen3vl_step20(KV=1047,P=1280,H=32,D=128,C={threads})",
        1047, 32, 8, 1047, 128, 128, threads, 0.0,
    )
    results.append(qwen_step20)
    qwen_step60, qwen_step60_data = _case(
        f"f16_split_qwen3vl_step60(KV=1087,P=1280,H=32,D=128,C={threads})",
        1087, 32, 8, 1087, 128, 128, threads, 0.0,
    )
    results.append(qwen_step60)
    qwen_mid, qwen_mid_data = _case(
        f"f16_split_qwen3vl_mid(KV=1307,H=32,D=128,C={threads})",
        1307, 32, 8, 1307, 128, 128, threads, 0.0,
    )
    results.append(qwen_mid)
    qwen_long, qwen_long_data = _case(
        f"f16_split_qwen3vl_long(KV=1609,H=32,D=128,C={threads})",
        1609, 32, 8, 1609, 128, 128, threads, 0.0,
    )
    results.append(qwen_long)
    padded, _ = _case(
        f"f16_split_padded_gqa(KV=513,D=80,A=128,C={threads})",
        513, 8, 2, 513, 80, 128, threads, 0.0,
    )
    results.append(padded)

    for name, data in (
        ("explicit_contract_route_below(KV=511)", below_data),
        ("explicit_contract_route_threshold(KV=512)", threshold_data),
        ("explicit_contract_route_qwen3vl(KV=1058)", qwen_data),
        ("explicit_contract_route_qwen3vl_step20(KV=1047,P=1280)", qwen_step20_data),
        ("explicit_contract_route_qwen3vl_step60(KV=1087,P=1280)", qwen_step60_data),
        ("explicit_contract_route_qwen3vl_mid(KV=1307)", qwen_mid_data),
        ("explicit_contract_route_qwen3vl_long(KV=1609)", qwen_long_data),
    ):
        q, k, v, _ = data
        chunks = threads if k.shape[1] >= 512 else 1
        expected = _run_explicit(q, k, v, q.shape[1], chunks)
        status, actual = _run_contract(q, k, v, q.shape[1], 1)
        diff = float(np.max(np.abs(actual - expected))) if status == 0 else math.inf
        results.append(Result(name, diff, 0.0, status == 0 and diff == 0.0))

    q, k, v, _ = qwen_step20_data
    expected_single = _run_explicit(q, k, v, q.shape[1], 1)
    single_status, actual_single = _run_contract(q, k, v, q.shape[1], 2)
    single_diff = (
        float(np.max(np.abs(actual_single - expected_single)))
        if single_status == 0
        else math.inf
    )
    results.append(Result(
        "explicit_single_range_prefill_route_qwen3vl(KV=1047)",
        single_diff,
        0.0,
        single_status == 0 and single_diff == 0.0,
    ))

    q, k, v, _ = below_data
    legacy = _run_legacy(q, k, v, q.shape[1])
    fp32_status, fp32_contract = _run_contract(q, k, v, q.shape[1], 0)
    fp32_diff = float(np.max(np.abs(fp32_contract - legacy))) if fp32_status == 0 else math.inf
    results.append(Result(
        "explicit_fp32_preserves_legacy_abi(KV=511)",
        fp32_diff,
        0.0,
        fp32_status == 0 and fp32_diff == 0.0,
    ))

    unsupported_status, unsupported_out = _run_contract(
        q, k, v, q.shape[1], 999, fill=np.float32(123.25),
    )
    unsupported_ok = unsupported_status == -2 and bool(np.all(unsupported_out == np.float32(123.25)))
    results.append(Result(
        "unsupported_contract_rejected_without_output_write",
        0.0 if unsupported_ok else 1.0,
        0.0,
        unsupported_ok,
    ))

    try:
        for name, data, chunks in (
            ("llama_oracle_below(KV=511)", below_data, 1),
            ("llama_oracle_threshold(KV=512)", threshold_data, threads),
            ("llama_oracle_qwen3vl(KV=1058)", qwen_data, threads),
            ("llama_oracle_qwen3vl_step20(KV=1047,P=1280)", qwen_step20_data, threads),
            ("llama_oracle_qwen3vl_step60(KV=1087,P=1280)", qwen_step60_data, threads),
            ("llama_oracle_qwen3vl_mid(KV=1307)", qwen_mid_data, threads),
            ("llama_oracle_qwen3vl_long(KV=1609)", qwen_long_data, threads),
        ):
            q, k, v, _ = data
            ck = _run_explicit(q, k, v, q.shape[1], chunks)
            llama = _llama_split_output(
                q, k, v, q.shape[1], threads, _llama_kv_partition_extent(k.shape[1])
            )
            diff = float(np.max(np.abs(ck[:, :q.shape[1]] - llama)))
            results.append(Result(name, diff, 0.0, diff == 0.0))
    except RuntimeError as exc:
        print(f"llama.cpp oracle error: {exc}")
        results.append(Result("llama_oracle_available", 1.0, 0.0, False))

    q, k, v, _ = qwen_data
    v_sensitive = _half_bits(v.view(np.float16).astype(np.float32) * np.float32(8.0))
    split_out = _run_explicit(q, k, v_sensitive, 128, 20)
    fp32_out = _fp32_single_reference(q, k, v_sensitive, 128)
    contract_gap = float(np.max(np.abs(split_out - fp32_out)))
    sensitivity_ok = contract_gap >= 5.0e-4
    results.append(Result(
        "reject_single_fp32_reduction(KV=1058)",
        0.0 if sensitivity_ok else 1.0,
        0.0,
        sensitivity_ok,
    ))

    print(f"FP16 split-KV attention contract: CK threads={threads}, fp32_contract_gap={contract_gap:.6e}")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{result.name}  max_diff={result.diff:.6e}  "
            f"tol={result.tolerance:.1e}  [{status}]"
        )
    passed = sum(result.passed for result in results)
    print(f"FP16 split-KV attention: {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
