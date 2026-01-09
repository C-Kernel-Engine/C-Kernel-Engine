#!/usr/bin/env python3
"""
Flash attention correctness and performance tests.

Validates:
  - Online-softmax kernel matches a PyTorch causal reference
  - Multiple shapes, including tail head dimensions
  - Optional comparison against llama.cpp ggml flash attention (if available)
"""

import ctypes
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

from test_utils import (
    TestReport,
    TestResult,
    get_cpu_info,
    max_diff,
    numpy_to_ptr,
    print_system_info,
    time_function,
)


ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src/kernels/attention_flash_true.c"
WRAPPER_PATH = ROOT / "unittest/flash_attention_omp_wrapper.c"
LLAMA_CPP_BIN = ROOT / "llama.cpp/build/bin"

def get_tile_k():
    env = os.environ.get("CK_FLASH_ATTN_TILE_K")
    if env is None:
        return 32
    try:
        val = int(env)
    except ValueError:
        return 32
    return val if val > 0 else 32


def get_omp_threads():
    env = os.environ.get("CK_FLASH_ATTN_OMP_THREADS")
    if env is not None:
        try:
            val = int(env)
        except ValueError:
            val = 0
        if val > 0:
            return val
    return os.cpu_count() or 1


def configure_threading():
    threads = get_omp_threads()
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    return threads, torch.get_num_threads(), torch.get_num_interop_threads()


def get_extra_long_contexts():
    env = os.environ.get("CK_FLASH_ATTN_LONG_CONTEXTS")
    if not env:
        return []
    contexts = []
    for part in env.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            val = int(part)
        except ValueError:
            continue
        if val > 0:
            contexts.append(val)
    return contexts


def llama_perf_enabled():
    env = os.environ.get("CK_FLASH_ATTN_LLAMA_PERF", "1").lower()
    return env not in ("0", "false", "no")


def llama_perf_max_tk():
    env = os.environ.get("CK_FLASH_ATTN_LLAMA_PERF_MAX_TK")
    if env is None:
        return 8192
    try:
        val = int(env)
    except ValueError:
        return 8192
    return val if val > 0 else 8192


def layernorm_np(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def gelu_np(x):
    t = torch.from_numpy(x)
    return torch.nn.functional.gelu(t).numpy()


def make_activation_np(shape, seed, dist="normal"):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape).astype(np.float32)
    if dist == "ln":
        x = layernorm_np(x)
    elif dist == "ln_gelu":
        x = layernorm_np(x)
        x = gelu_np(x)
    return np.ascontiguousarray(x, dtype=np.float32)


def expand_kv_for_gqa(k_np, v_np, num_heads, num_kv_heads):
    T_k, _, D_h = k_np.shape
    k_exp = np.empty((T_k, num_heads, D_h), dtype=np.float32)
    v_exp = np.empty((T_k, num_heads, D_h), dtype=np.float32)
    for h in range(num_heads):
        kv_head = (h * num_kv_heads) // num_heads
        k_exp[:, h, :] = k_np[:, kv_head, :]
        v_exp[:, h, :] = v_np[:, kv_head, :]
    return k_exp, v_exp


def load_activation_fixture():
    path = os.environ.get("CK_FLASH_ATTN_FIXTURE")
    if not path:
        return None
    fixture_path = Path(path)
    if not fixture_path.exists():
        print(f"fixture: skipped (not found): {fixture_path}")
        return None

    data = np.load(fixture_path)
    if "q" not in data.files or "k" not in data.files or "v" not in data.files:
        print("fixture: skipped (missing q/k/v arrays)")
        return None

    q_np = np.array(data["q"], dtype=np.float32, copy=False)
    k_np = np.array(data["k"], dtype=np.float32, copy=False)
    v_np = np.array(data["v"], dtype=np.float32, copy=False)
    if q_np.ndim != 3 or k_np.ndim != 3 or v_np.ndim != 3:
        print("fixture: skipped (expected q/k/v with shape [T, H, D])")
        return None

    return {
        "name": fixture_path.name,
        "q": np.ascontiguousarray(q_np),
        "k": np.ascontiguousarray(k_np),
        "v": np.ascontiguousarray(v_np),
    }


def build_flash_attn_lib(fast_exp=False):
    build_dir = Path(tempfile.gettempdir()) / "ckernel_flash_attn"
    build_dir.mkdir(parents=True, exist_ok=True)
    tile_k = get_tile_k()
    suffix = "fast" if fast_exp else "accurate"
    lib_path = build_dir / f"libck_flash_attn_{suffix}_k{tile_k}.so"

    if lib_path.exists():
        lib_mtime = lib_path.stat().st_mtime
        src_mtime = SRC_PATH.stat().st_mtime
        wrapper_mtime = WRAPPER_PATH.stat().st_mtime
        if lib_mtime >= max(src_mtime, wrapper_mtime):
            return lib_path

    base_cmd = [
        "gcc",
        "-O3",
        "-fPIC",
        "-shared",
        "-march=native",
        "-fopenmp",
        "-I",
        str(ROOT / "include"),
        "-o",
        str(lib_path),
        str(SRC_PATH),
        str(WRAPPER_PATH),
        f"-DCK_FLASH_ATTN_TILE_K={tile_k}",
        "-lm",
    ]
    if fast_exp:
        base_cmd.insert(-1, "-DCK_FLASH_ATTN_FAST_EXP=1")

    result = subprocess.run(base_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to compile flash attention kernel:\n"
            f"{result.stderr}"
        )

    return lib_path


def load_flash_attn_lib(fast_exp=False):
    lib_path = build_flash_attn_lib(fast_exp=fast_exp)
    lib = ctypes.CDLL(str(lib_path))
    lib.attention_flash_decode.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]
    lib.attention_flash_decode.restype = None
    lib.ck_flash_attention_decode_omp.argtypes = [
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
    lib.ck_flash_attention_decode_omp.restype = None
    lib.ck_flash_attn_choose_tile_k.argtypes = [ctypes.c_int]
    lib.ck_flash_attn_choose_tile_k.restype = ctypes.c_int
    lib.ck_flash_attn_fast_exp_kind.argtypes = []
    lib.ck_flash_attn_fast_exp_kind.restype = ctypes.c_int
    return lib


def expected_tile_k(D_h, tile_k_max):
    tile = tile_k_max
    if D_h > 128:
        tile = tile_k_max // 4
    elif D_h > 64:
        tile = tile_k_max // 2
    if tile_k_max >= 8 and tile < 8:
        tile = 8
    if tile > tile_k_max:
        tile = tile_k_max
    if tile < 1:
        tile = 1
    return int(tile)


def expected_fast_exp_kind(cpu_info):
    if cpu_info.avx512f:
        return 512
    if cpu_info.avx:
        return 256
    return 0


def causal_mask(T_q, T_k):
    q_pos_offset = max(T_k - T_q, 0)
    q_pos = torch.arange(q_pos_offset, q_pos_offset + T_q).unsqueeze(1)
    k_pos = torch.arange(T_k).unsqueeze(0)
    return k_pos > q_pos


def make_torch_reference(q_np, k_np, v_np):
    T_q, H, D_h = q_np.shape
    T_k = k_np.shape[0]
    scale = 1.0 / math.sqrt(D_h)

    q = torch.from_numpy(q_np).float().permute(1, 0, 2)  # H, T_q, D
    k = torch.from_numpy(k_np).float().permute(1, 0, 2)  # H, T_k, D
    v = torch.from_numpy(v_np).float().permute(1, 0, 2)  # H, T_k, D
    mask = causal_mask(T_q, T_k).unsqueeze(0)  # 1, T_q, T_k

    def reference():
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        return out.permute(1, 0, 2).contiguous()

    return reference


def make_c_flash(lib, q_np, k_np, v_np, out_np, T_q, T_k, H, D_h):
    scale = 1.0 / math.sqrt(D_h)
    q_ptr = numpy_to_ptr(q_np)
    k_ptr = numpy_to_ptr(k_np)
    v_ptr = numpy_to_ptr(v_np)
    out_ptr = numpy_to_ptr(out_np)

    def c_flash():
        lib.attention_flash_decode(
            out_ptr,
            q_ptr,
            k_ptr,
            v_ptr,
            ctypes.c_int(T_q),
            ctypes.c_int(T_k),
            ctypes.c_int(H),
            ctypes.c_int(D_h),
            ctypes.c_float(scale),
        )

    return c_flash


def make_c_flash_omp_wrapper(lib, q_np, k_np, v_np, out_np, T_k, H, D_h, H_kv=None):
    if H_kv is None:
        H_kv = H
    q_token = np.ascontiguousarray(q_np[0])
    k_cache = np.ascontiguousarray(k_np.transpose(1, 0, 2))
    v_cache = np.ascontiguousarray(v_np.transpose(1, 0, 2))
    q_ptr = numpy_to_ptr(q_token)
    k_ptr = numpy_to_ptr(k_cache)
    v_ptr = numpy_to_ptr(v_cache)
    out_ptr = numpy_to_ptr(out_np)

    def c_flash():
        lib.ck_flash_attention_decode_omp(
            q_ptr,
            k_ptr,
            v_ptr,
            out_ptr,
            ctypes.c_int(H),
            ctypes.c_int(H_kv),
            ctypes.c_int(T_k),
            ctypes.c_int(T_k),
            ctypes.c_int(D_h),
            ctypes.c_int(D_h),
        )

    return c_flash, q_token, k_cache, v_cache


GGML_HELPER_SRC = r"""
#include "ggml.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static size_t offset_bytes(const struct ggml_tensor * t, int i0, int i1, int i2, int i3) {
    return (size_t)i0 * t->nb[0] + (size_t)i1 * t->nb[1] + (size_t)i2 * t->nb[2] + (size_t)i3 * t->nb[3];
}

static int max_k_for_query(int t_q, int T_q, int T_k) {
    int q_pos_offset = (T_k > T_q) ? (T_k - T_q) : 0;
    int max_k = q_pos_offset + t_q;
    if (max_k >= T_k) {
        max_k = T_k - 1;
    }
    return max_k;
}

static bool read_f32(const char * path, float * dst, size_t count) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        return false;
    }
    size_t n = fread(dst, sizeof(float), count, f);
    fclose(f);
    return n == count;
}

static bool write_f32(const char * path, const float * src, size_t count) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        return false;
    }
    size_t n = fwrite(src, sizeof(float), count, f);
    fclose(f);
    return n == count;
}

int main(int argc, char ** argv) {
    if (argc < 10) {
        fprintf(stderr,
                "usage: %s q.bin k.bin v.bin out.bin T_q T_k H D_h scale [iters] [threads] [mode]\n",
                argv[0]);
        return 1;
    }

    const char * q_path = argv[1];
    const char * k_path = argv[2];
    const char * v_path = argv[3];
    const char * out_path = argv[4];
    const int T_q = atoi(argv[5]);
    const int T_k = atoi(argv[6]);
    const int H = atoi(argv[7]);
    const int D_h = atoi(argv[8]);
    const float scale = (float) atof(argv[9]);

    if (T_q <= 0 || T_k <= 0 || H <= 0 || D_h <= 0) {
        fprintf(stderr, "invalid dimensions\n");
        return 1;
    }

    int iters = (argc >= 11) ? atoi(argv[10]) : 20;
    int threads = (argc >= 12) ? atoi(argv[11]) : 1;
    int mode = (argc >= 13) ? atoi(argv[12]) : 0;
    if (iters <= 0) {
        iters = 20;
    }
    if (threads <= 0) {
        threads = 1;
    }

    const size_t q_elems = (size_t) T_q * (size_t) H * (size_t) D_h;
    const size_t k_elems = (size_t) T_k * (size_t) H * (size_t) D_h;

    float * q = (float *) malloc(q_elems * sizeof(float));
    float * k = (float *) malloc(k_elems * sizeof(float));
    float * v = (float *) malloc(k_elems * sizeof(float));
    float * out_host = (float *) malloc(q_elems * sizeof(float));

    if (!q || !k || !v || !out_host) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    if (!read_f32(q_path, q, q_elems) ||
        !read_f32(k_path, k, k_elems) ||
        !read_f32(v_path, v, k_elems)) {
        fprintf(stderr, "failed to read inputs\n");
        return 1;
    }

    const size_t mem_size = 64u * 1024u * 1024u;
    struct ggml_init_params params = { mem_size, NULL, false };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "ggml_init failed\n");
        return 1;
    }

    struct ggml_tensor * q_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D_h, T_q, H, 1);
    struct ggml_tensor * k_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D_h, T_k, H, 1);
    struct ggml_tensor * v_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D_h, T_k, H, 1);
    struct ggml_tensor * mask_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, T_k, T_q, 1, 1);

    char * q_data = (char *) q_t->data;
    char * k_data = (char *) k_t->data;
    char * v_data = (char *) v_t->data;
    char * m_data = (char *) mask_t->data;

    for (int t = 0; t < T_q; ++t) {
        for (int h = 0; h < H; ++h) {
            for (int d = 0; d < D_h; ++d) {
                const size_t src = ((size_t) t * (size_t) H + (size_t) h) * (size_t) D_h + (size_t) d;
                const size_t off = offset_bytes(q_t, d, t, h, 0);
                memcpy(q_data + off, &q[src], sizeof(float));
            }
        }
    }

    for (int t = 0; t < T_k; ++t) {
        for (int h = 0; h < H; ++h) {
            for (int d = 0; d < D_h; ++d) {
                const size_t src = ((size_t) t * (size_t) H + (size_t) h) * (size_t) D_h + (size_t) d;
                const size_t off_k = offset_bytes(k_t, d, t, h, 0);
                const size_t off_v = offset_bytes(v_t, d, t, h, 0);
                memcpy(k_data + off_k, &k[src], sizeof(float));
                memcpy(v_data + off_v, &v[src], sizeof(float));
            }
        }
    }

    for (int t_q = 0; t_q < T_q; ++t_q) {
        const int max_k = max_k_for_query(t_q, T_q, T_k);
        for (int t_k = 0; t_k < T_k; ++t_k) {
            float mask_val = (t_k > max_k) ? -INFINITY : 0.0f;
            ggml_fp16_t m16 = ggml_fp32_to_fp16(mask_val);
            const size_t off = offset_bytes(mask_t, t_k, t_q, 0, 0);
            memcpy(m_data + off, &m16, sizeof(ggml_fp16_t));
        }
    }

    struct ggml_tensor * out = ggml_flash_attn_ext(ctx, q_t, k_t, v_t, mask_t, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    if (mode == 1) {
        const int warmup = 2;
        for (int i = 0; i < warmup; ++i) {
            ggml_graph_compute_with_ctx(ctx, gf, threads);
        }
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            ggml_graph_compute_with_ctx(ctx, gf, threads);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count();
        double avg_us = us / (double) iters;
        printf("%.3f\n", avg_us);
        ggml_free(ctx);
        free(q);
        free(k);
        free(v);
        free(out_host);
        return 0;
    }

    ggml_graph_compute_with_ctx(ctx, gf, 1);

    char * out_data = (char *) out->data;
    for (int t = 0; t < T_q; ++t) {
        for (int h = 0; h < H; ++h) {
            for (int d = 0; d < D_h; ++d) {
                const size_t dst = ((size_t) t * (size_t) H + (size_t) h) * (size_t) D_h + (size_t) d;
                const size_t off = offset_bytes(out, d, h, t, 0);
                memcpy(&out_host[dst], out_data + off, sizeof(float));
            }
        }
    }

    if (!write_f32(out_path, out_host, q_elems)) {
        fprintf(stderr, "failed to write output\n");
        return 1;
    }

    ggml_free(ctx);
    free(q);
    free(k);
    free(v);
    free(out_host);
    return 0;
}
"""


def ensure_llama_cpp_helper():
    libggml = LLAMA_CPP_BIN / "libggml.so"
    if not libggml.exists():
        return None

    helper_dir = Path(tempfile.gettempdir()) / "ckernel_flash_attn"
    helper_dir.mkdir(parents=True, exist_ok=True)
    helper_src = helper_dir / "ggml_flash_attn_ref.cpp"
    helper_bin = helper_dir / "ggml_flash_attn_ref"

    if not helper_src.exists() or helper_src.read_text() != GGML_HELPER_SRC:
        helper_src.write_text(GGML_HELPER_SRC)

    if not helper_bin.exists() or helper_bin.stat().st_mtime < helper_src.stat().st_mtime:
        cmd = [
            "g++",
            "-O2",
            "-std=c++11",
            "-I",
            str(ROOT / "llama.cpp/ggml/include"),
            "-o",
            str(helper_bin),
            str(helper_src),
            "-L",
            str(LLAMA_CPP_BIN),
            "-lggml",
            "-lggml-cpu",
            "-lggml-base",
            "-lm",
            "-lpthread",
            "-Wl,-rpath," + str(LLAMA_CPP_BIN),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("llama.cpp compare: skipped (compile failed)")
            print(result.stderr.strip())
            return None

    return helper_bin


def try_llama_cpp_compare(case, q_np, k_np, v_np):
    helper_bin = ensure_llama_cpp_helper()
    if helper_bin is None:
        print("llama.cpp compare: skipped (libggml.so not found)")
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        q_path = tmpdir / "q.bin"
        k_path = tmpdir / "k.bin"
        v_path = tmpdir / "v.bin"
        out_path = tmpdir / "out.bin"

        q_np.tofile(q_path)
        k_np.tofile(k_path)
        v_np.tofile(v_path)

        scale = 1.0 / math.sqrt(case["D_h"])
        cmd = [
            str(helper_bin),
            str(q_path),
            str(k_path),
            str(v_path),
            str(out_path),
            str(case["T_q"]),
            str(case["T_k"]),
            str(case["H"]),
            str(case["D_h"]),
            str(scale),
        ]
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{LLAMA_CPP_BIN}:{env.get('LD_LIBRARY_PATH', '')}"
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            print("llama.cpp compare: skipped (run failed)")
            print(result.stderr.strip())
            return None

        return np.fromfile(out_path, dtype=np.float32).reshape(q_np.shape)


def try_llama_cpp_perf(case, threads, iterations):
    helper_bin = ensure_llama_cpp_helper()
    if helper_bin is None:
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        q_path = tmpdir / "q.bin"
        k_path = tmpdir / "k.bin"
        v_path = tmpdir / "v.bin"
        out_path = tmpdir / "out.bin"

        rng = np.random.default_rng(case["seed"])
        q_np = rng.standard_normal((case["T_q"], case["H"], case["D_h"])).astype(np.float32)
        k_np = rng.standard_normal((case["T_k"], case["H"], case["D_h"])).astype(np.float32)
        v_np = rng.standard_normal((case["T_k"], case["H"], case["D_h"])).astype(np.float32)

        q_np.tofile(q_path)
        k_np.tofile(k_path)
        v_np.tofile(v_path)

        scale = 1.0 / math.sqrt(case["D_h"])
        cmd = [
            str(helper_bin),
            str(q_path),
            str(k_path),
            str(v_path),
            str(out_path),
            str(case["T_q"]),
            str(case["T_k"]),
            str(case["H"]),
            str(case["D_h"]),
            str(scale),
            str(iterations),
            str(threads),
            "1",
        ]
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{LLAMA_CPP_BIN}:{env.get('LD_LIBRARY_PATH', '')}"
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            print("llama.cpp perf: skipped (run failed)")
            print(result.stderr.strip())
            return None

        try:
            return float(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            print("llama.cpp perf: skipped (invalid output)")
            return None


def run_flash_case(lib, case, timed=False, warmup=10, iterations=200):
    dist = case.get("dist", "normal")
    q_np = make_activation_np((case["T_q"], case["H"], case["D_h"]), case["seed"], dist=dist)
    k_np = make_activation_np((case["T_k"], case["H"], case["D_h"]), case["seed"] + 1, dist=dist)
    v_np = make_activation_np((case["T_k"], case["H"], case["D_h"]), case["seed"] + 2, dist=dist)
    out_np = np.zeros_like(q_np)

    ref_fn = make_torch_reference(q_np, k_np, v_np)
    c_flash = make_c_flash(
        lib,
        q_np,
        k_np,
        v_np,
        out_np,
        case["T_q"],
        case["T_k"],
        case["H"],
        case["D_h"],
    )

    ref = ref_fn()
    c_flash()
    out = torch.from_numpy(out_np.copy())

    diff = max_diff(out, ref)
    tol = case.get("tol", 1e-4)

    pytorch_time = None
    kernel_time = None
    if timed:
        pytorch_time = time_function(ref_fn, warmup=warmup, iterations=iterations, name="PyTorch")
        kernel_time = time_function(c_flash, warmup=warmup, iterations=iterations, name="C Flash")

    suffix = f",dist={dist}" if dist != "normal" else ""
    result = TestResult(
        name=f"{case['label']} (Tq={case['T_q']},Tk={case['T_k']},H={case['H']},D={case['D_h']}{suffix})",
        passed=diff <= tol,
        max_diff=diff,
        tolerance=tol,
        pytorch_time=pytorch_time,
        kernel_time=kernel_time,
    )

    return result, q_np, k_np, v_np, ref


def run_wrapper_case(lib, case, timed=False, warmup=10, iterations=200):
    if case["T_q"] != 1:
        raise ValueError("Wrapper tests only support T_q=1 decode.")

    dist = case.get("dist", "normal")
    H_kv = case.get("H_kv", case["H"])
    q_np = make_activation_np((case["T_q"], case["H"], case["D_h"]), case["seed"], dist=dist)
    k_np = make_activation_np((case["T_k"], H_kv, case["D_h"]), case["seed"] + 1, dist=dist)
    v_np = make_activation_np((case["T_k"], H_kv, case["D_h"]), case["seed"] + 2, dist=dist)
    out_np = np.zeros((case["H"], case["D_h"]), dtype=np.float32)

    if H_kv != case["H"]:
        k_exp, v_exp = expand_kv_for_gqa(k_np, v_np, case["H"], H_kv)
        ref_fn = make_torch_reference(q_np, k_exp, v_exp)
    else:
        ref_fn = make_torch_reference(q_np, k_np, v_np)
    c_flash, _, _, _ = make_c_flash_omp_wrapper(
        lib,
        q_np,
        k_np,
        v_np,
        out_np,
        case["T_k"],
        case["H"],
        case["D_h"],
        H_kv=H_kv,
    )

    ref = ref_fn()[0]
    c_flash()
    out = torch.from_numpy(out_np.copy())

    diff = max_diff(out, ref)
    tol = case.get("tol", 1e-4)

    pytorch_time = None
    kernel_time = None
    if timed:
        pytorch_time = time_function(ref_fn, warmup=warmup, iterations=iterations, name="PyTorch")
        kernel_time = time_function(c_flash, warmup=warmup, iterations=iterations, name="C Flash (OMP)")

    suffix = f",kv={H_kv}" if H_kv != case["H"] else ""
    dist_suffix = f",dist={dist}" if dist != "normal" else ""
    result = TestResult(
        name=f"{case['label']} (Tq=1,Tk={case['T_k']},H={case['H']},D={case['D_h']}{suffix}{dist_suffix})",
        passed=diff <= tol,
        max_diff=diff,
        tolerance=tol,
        pytorch_time=pytorch_time,
        kernel_time=kernel_time,
    )

    return result


def run_fixture_case(lib, fixture):
    q_np = fixture["q"]
    k_np = fixture["k"]
    v_np = fixture["v"]

    T_q, H, D_h = q_np.shape
    T_k = k_np.shape[0]
    H_kv = k_np.shape[1]

    tol = 1e-4
    env_tol = os.environ.get("CK_FLASH_ATTN_FIXTURE_TOL")
    if env_tol is not None:
        try:
            tol = float(env_tol)
        except ValueError:
            tol = 1e-4

    if H_kv != H:
        if T_q != 1:
            print("fixture: skipped (GQA fixture requires T_q=1)")
            return None
        k_exp, v_exp = expand_kv_for_gqa(k_np, v_np, H, H_kv)
        ref_fn = make_torch_reference(q_np, k_exp, v_exp)
        out_np = np.zeros((H, D_h), dtype=np.float32)
        c_flash, _, _, _ = make_c_flash_omp_wrapper(
            lib,
            q_np,
            k_np,
            v_np,
            out_np,
            T_k,
            H,
            D_h,
            H_kv=H_kv,
        )
        ref = ref_fn()[0]
        c_flash()
        out = torch.from_numpy(out_np.copy())
    else:
        out_np = np.zeros_like(q_np)
        ref_fn = make_torch_reference(q_np, k_np, v_np)
        c_flash = make_c_flash(lib, q_np, k_np, v_np, out_np, T_q, T_k, H, D_h)
        ref = ref_fn()
        c_flash()
        out = torch.from_numpy(out_np.copy())

    diff = max_diff(out, ref)
    return TestResult(
        name=f"fixture_{fixture['name']} (Tq={T_q},Tk={T_k},H={H},D={D_h},kv={H_kv})",
        passed=diff <= tol,
        max_diff=diff,
        tolerance=tol,
        pytorch_time=None,
        kernel_time=None,
    )


def run_config_tests(lib_acc, lib_fast, tile_k_max):
    cpu_info = get_cpu_info()
    report = TestReport(
        test_name="Flash Attention - Config",
        dtype="fp32",
        shape=f"tile_k_max={tile_k_max}, tile_heuristic=on",
        cpu_info=cpu_info,
    )

    dims = [32, 64, 96, 128, 192]
    for d_h in dims:
        actual = lib_acc.ck_flash_attn_choose_tile_k(d_h)
        expected = expected_tile_k(d_h, tile_k_max)
        diff = abs(actual - expected)
        report.add_result(TestResult(
            name=f"tile_k D={d_h} -> {actual}",
            passed=diff == 0,
            max_diff=float(diff),
            tolerance=0.0,
            pytorch_time=None,
            kernel_time=None,
        ))

    if lib_fast is not None:
        def fast_label(kind):
            if kind == 512:
                return "avx512"
            if kind == 256:
                return "avx"
            return "scalar"

        actual_kind = lib_fast.ck_flash_attn_fast_exp_kind()
        expected_kind = expected_fast_exp_kind(cpu_info)
        diff = abs(actual_kind - expected_kind)
        report.add_result(TestResult(
            name=f"fast_exp_vec={fast_label(actual_kind)}",
            passed=diff == 0,
            max_diff=float(diff),
            tolerance=0.0,
            pytorch_time=None,
            kernel_time=None,
        ))

    return report


def run_accuracy_tests(lib, tile_k, fast_lib=None):
    report = TestReport(
        test_name="Flash Attention (Causal, Online Softmax) - Accuracy",
        dtype="fp32",
        shape=f"tile_k_max={tile_k}, tile_heuristic=on, fast_exp=off",
        cpu_info=get_cpu_info(),
    )

    cases = [
        {"label": "decode_small", "T_q": 1, "T_k": 16, "H": 2, "D_h": 32, "seed": 1},
        {"label": "decode_tail", "T_q": 1, "T_k": 63, "H": 4, "D_h": 40, "seed": 2, "tol": 2e-4},
        {"label": "prefill", "T_q": 8, "T_k": 8, "H": 4, "D_h": 64, "seed": 3},
        {"label": "offset", "T_q": 3, "T_k": 9, "H": 3, "D_h": 24, "seed": 4},
        {"label": "prefill_hd128", "T_q": 4, "T_k": 4, "H": 2, "D_h": 128, "seed": 5, "tol": 3e-4},
        {"label": "decode_hd192", "T_q": 1, "T_k": 6, "H": 2, "D_h": 192, "seed": 6, "tol": 4e-4},
        {"label": "decode_ln_gelu", "T_q": 1, "T_k": 32, "H": 4, "D_h": 64, "seed": 7, "dist": "ln_gelu"},
    ]

    fast_tol = 1e-2
    env_tol = os.environ.get("CK_FLASH_ATTN_FAST_TOL")
    if env_tol is not None:
        try:
            fast_tol = float(env_tol)
        except ValueError:
            fast_tol = 1e-2

    if fast_lib is None:
        try:
            fast_lib = load_flash_attn_lib(fast_exp=True)
        except RuntimeError as exc:
            print("fast exp accuracy: skipped (compile failed)")
            print(str(exc).strip())

    first_case_data = None

    for case in cases:
        result, q_np, k_np, v_np, ref = run_flash_case(lib, case, timed=False)
        report.add_result(result)
        if first_case_data is None:
            first_case_data = (case, q_np, k_np, v_np, ref)

        if fast_lib is not None:
            fast_case = dict(case)
            fast_case["label"] = f"fastexp_{case['label']}"
            fast_case["tol"] = fast_tol
            fast_result, _, _, _, _ = run_flash_case(fast_lib, fast_case, timed=False)
            report.add_result(fast_result)

    wrapper_case = {"label": "decode_wrapper_omp", "T_q": 1, "T_k": 64, "H": 4, "D_h": 64, "seed": 8}
    report.add_result(run_wrapper_case(lib, wrapper_case, timed=False))

    gqa_case = {
        "label": "decode_gqa_8h_2kv",
        "T_q": 1,
        "T_k": 64,
        "H": 8,
        "H_kv": 2,
        "D_h": 64,
        "seed": 9,
        "tol": 2e-4,
    }
    report.add_result(run_wrapper_case(lib, gqa_case, timed=False))

    fixture = load_activation_fixture()
    if fixture is not None:
        fixture_result = run_fixture_case(lib, fixture)
        if fixture_result is not None:
            report.add_result(fixture_result)

    if first_case_data is not None:
        case, q_np, k_np, v_np, ref = first_case_data
        out_llama = try_llama_cpp_compare(case, q_np, k_np, v_np)
        if out_llama is not None:
            diff = max_diff(torch.from_numpy(out_llama), ref)
            tol = 1e-3
            report.add_result(TestResult(
                name=f"llama.cpp ({case['label']})",
                passed=diff <= tol,
                max_diff=diff,
                tolerance=tol,
                pytorch_time=None,
                kernel_time=None,
            ))

    return report


def run_performance_tests(lib, tile_k, fast_exp, warmup=10, iterations=200):
    torch_threads = torch.get_num_threads()
    torch_interop = torch.get_num_interop_threads()
    report = TestReport(
        test_name="Flash Attention (Causal, Online Softmax) - Performance (OMP Wrapper)",
        dtype="fp32",
        shape=(
            f"tile_k_max={tile_k}, tile_heuristic=on, "
            f"fast_exp={'on' if fast_exp else 'off'}, omp_threads={get_omp_threads()}, "
            f"torch_threads={torch_threads}, torch_interop={torch_interop}"
        ),
        cpu_info=get_cpu_info(),
    )

    tol = 1e-3 if fast_exp else 1e-4
    cases = [
        {"label": "decode_4h_512", "T_q": 1, "T_k": 512, "H": 4, "D_h": 64, "seed": 10, "tol": tol},
        {"label": "decode_8h_512", "T_q": 1, "T_k": 512, "H": 8, "D_h": 64, "seed": 11, "tol": tol},
        {"label": "decode_32h_512", "T_q": 1, "T_k": 512, "H": 32, "D_h": 64, "seed": 12, "tol": tol},
        {"label": "decode_4h_1k", "T_q": 1, "T_k": 1024, "H": 4, "D_h": 64, "seed": 13, "tol": tol},
        {"label": "decode_4h_8k", "T_q": 1, "T_k": 8192, "H": 4, "D_h": 64, "seed": 14, "tol": tol,
         "warmup": 5, "iterations": 60},
        {"label": "decode_4h_24k", "T_q": 1, "T_k": 24576, "H": 4, "D_h": 64, "seed": 15, "tol": tol,
         "warmup": 3, "iterations": 20},
        {"label": "decode_4h_50k", "T_q": 1, "T_k": 51200, "H": 4, "D_h": 64, "seed": 16, "tol": tol,
         "warmup": 2, "iterations": 10},
        {"label": "decode_4h_hd128", "T_q": 1, "T_k": 256, "H": 4, "D_h": 128, "seed": 17, "tol": tol},
        {"label": "decode_gqa_16h_2kv", "T_q": 1, "T_k": 1024, "H": 16, "H_kv": 2, "D_h": 64, "seed": 18,
         "tol": tol, "warmup": 5, "iterations": 80},
    ]

    for tk in get_extra_long_contexts():
        cases.append({
            "label": f"decode_4h_{tk}",
            "T_q": 1,
            "T_k": tk,
            "H": 4,
            "D_h": 64,
            "seed": 20 + (tk % 1000),
            "tol": tol,
            "warmup": 1,
            "iterations": 5,
        })

    print()
    print("  CASE MATRIX")
    print("  " + "-" * 40)
    max_label = max(len(case["label"]) for case in cases)
    header = (
        f"  {'label':<{max_label}}  "
        f"{'T_q':>3}  {'T_k(ctx)':>9}  {'heads':>5}  {'kv':>3}  {'D_h':>4}  {'warmup':>6}  {'iters':>5}"
    )
    print(header)
    print("  " + "-" * len(header.strip()))
    for case in cases:
        kv = case.get("H_kv", case["H"])
        print(
            f"  {case['label']:<{max_label}}  "
            f"{case['T_q']:>3}  {case['T_k']:>9}  {case['H']:>5}  {kv:>3}  {case['D_h']:>4}  "
            f"{case.get('warmup', warmup):>6}  {case.get('iterations', iterations):>5}"
        )

    llama_rows = []
    llama_enabled = llama_perf_enabled()
    llama_max_tk = llama_perf_max_tk()
    omp_threads = get_omp_threads()

    for case in cases:
        result = run_wrapper_case(
            lib,
            case,
            timed=True,
            warmup=case.get("warmup", warmup),
            iterations=case.get("iterations", iterations),
        )
        report.add_result(result)

        if llama_enabled and case["T_k"] <= llama_max_tk and case.get("H_kv", case["H"]) == case["H"]:
            iters = case.get("iterations", iterations)
            llama_us = try_llama_cpp_perf(case, omp_threads, iters)
            if llama_us is not None:
                llama_rows.append((result.name, llama_us, result.kernel_time.mean_us))

    if llama_rows:
        print()
        print("  LLAMA.CPP PERFORMANCE (ggml_flash_attn_ext)")
        print("  " + "-" * 40)
        max_name_len = max(len(name) for name, _, _ in llama_rows)
        print(f"  {'Kernel':<{max_name_len}}  {'llama.cpp (us)':<15}  {'C Kernel (us)':<15}  {'Speedup':<10}")
        print("  " + "-" * 60)
        for name, llama_us, ck_us in llama_rows:
            sp = llama_us / ck_us if ck_us > 0 else 0.0
            sp_str = f"{sp:.2f}x"
            print(f"  {name:<{max_name_len}}  {llama_us:<15.1f}  {ck_us:<15.1f}  {sp_str}")

    return report


def main():
    configure_threading()
    print_system_info()
    tile_k = get_tile_k()
    lib_acc = load_flash_attn_lib(fast_exp=False)
    lib_fast = None
    try:
        lib_fast = load_flash_attn_lib(fast_exp=True)
    except RuntimeError as exc:
        print("fast exp build: skipped (compile failed)")
        print(str(exc).strip())

    config_report = run_config_tests(lib_acc, lib_fast, tile_k)
    config_report.print_report()

    acc_report = run_accuracy_tests(lib_acc, tile_k, fast_lib=lib_fast)
    acc_report.print_report()

    fast_env = os.environ.get("CK_FLASH_ATTN_FAST_EXP", "1").lower()
    want_fast = fast_env not in ("0", "false", "no")
    fast_exp = want_fast and lib_fast is not None
    lib_perf = lib_fast if fast_exp else lib_acc
    perf_report = run_performance_tests(lib_perf, tile_k, fast_exp, warmup=10, iterations=200)
    perf_report.print_report()

    if not config_report.all_passed() or not acc_report.all_passed() or not perf_report.all_passed():
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
