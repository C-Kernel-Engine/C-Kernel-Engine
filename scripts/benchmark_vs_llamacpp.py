#!/usr/bin/env python3
"""
benchmark_vs_llamacpp.py - Performance Comparison: CK-Engine vs llama.cpp

Benchmarks both engines on the same model and prompt, measuring:
- Prefill throughput (tokens/sec)
- Decode throughput (tokens/sec)
- Total latency

Usage:
    python scripts/benchmark_vs_llamacpp.py [--tokens N] [--runs N] [--model MODEL]

Examples:
    # Quick benchmark (default: 50 tokens, 3 runs)
    python scripts/benchmark_vs_llamacpp.py

    # Extended benchmark
    python scripts/benchmark_vs_llamacpp.py --tokens 200 --runs 5

    # Specific model
    python scripts/benchmark_vs_llamacpp.py --model Qwen/Qwen2-1.5B-Instruct-GGUF
"""

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

# Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'

PROJECT_ROOT = Path(__file__).parent.parent
LLAMACPP_DIR = PROJECT_ROOT / "llama.cpp"
LLAMACPP_CLI = LLAMACPP_DIR / "build" / "bin" / "llama-cli"
LLAMACPP_BENCH = LLAMACPP_DIR / "build" / "bin" / "llama-bench"


@dataclass
class BenchmarkResult:
    name: str
    prefill_tokens: int
    prefill_time_ms: float
    decode_tokens: int
    decode_time_ms: float

    @property
    def prefill_tps(self) -> float:
        return self.prefill_tokens / (self.prefill_time_ms / 1000) if self.prefill_time_ms > 0 else 0

    @property
    def decode_tps(self) -> float:
        return self.decode_tokens / (self.decode_time_ms / 1000) if self.decode_time_ms > 0 else 0

    @property
    def total_time_ms(self) -> float:
        return self.prefill_time_ms + self.decode_time_ms


def find_gguf_model(model_name: str) -> Optional[Path]:
    """Find the GGUF model file for a given model name"""
    cache_dir = Path.home() / ".cache" / "ck-engine-v6.5" / "models"

    # Try to find the model directory
    model_dir_name = model_name.replace("/", "--")
    model_dir = cache_dir / model_dir_name

    if not model_dir.exists():
        # Try common variations
        for d in cache_dir.iterdir():
            if model_dir_name.lower() in d.name.lower():
                model_dir = d
                break

    if not model_dir.exists():
        return None

    # Find the GGUF file
    gguf_files = list(model_dir.glob("*.gguf"))
    if not gguf_files:
        return None

    # Prefer Q4_K_M or similar
    for f in gguf_files:
        if "Q4_K" in f.name or "q4_k" in f.name:
            return f

    return gguf_files[0]


def find_ck_engine_lib(model_name: str) -> Optional[Path]:
    """Find the CK-Engine shared library for a model"""
    cache_dir = Path.home() / ".cache" / "ck-engine-v6.5" / "models"
    model_dir_name = model_name.replace("/", "--")

    for d in cache_dir.iterdir():
        if model_dir_name.lower() in d.name.lower() or d.name == model_dir_name:
            lib_path = d / "ck-kernel-inference.so"
            if lib_path.exists():
                return lib_path

    return None


def find_ck_engine_weights(model_name: str) -> Optional[Path]:
    """Find the CK-Engine weights file for a model"""
    cache_dir = Path.home() / ".cache" / "ck-engine-v6.5" / "models"
    model_dir_name = model_name.replace("/", "--")

    for d in cache_dir.iterdir():
        if model_dir_name.lower() in d.name.lower() or d.name == model_dir_name:
            weights_path = d / "weights.bump"
            if weights_path.exists():
                return weights_path

    return None


def benchmark_ck_engine(lib_path: Path, weights_path: Path,
                        prompt_tokens: List[int], decode_tokens: int,
                        num_runs: int = 3) -> Optional[BenchmarkResult]:
    """Benchmark CK-Engine"""

    try:
        lib = ctypes.CDLL(str(lib_path))
    except Exception as e:
        print(f"{RED}Failed to load CK-Engine: {e}{RESET}")
        return None

    # Set up function signatures
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_free.argtypes = []
    lib.ck_model_free.restype = None
    lib.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_forward.restype = ctypes.c_int
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.ck_model_embed_tokens.restype = ctypes.c_int
    lib.ck_model_kv_cache_enable.argtypes = [ctypes.c_int]
    lib.ck_model_kv_cache_enable.restype = ctypes.c_int
    lib.ck_model_kv_cache_reset.argtypes = []
    lib.ck_model_kv_cache_reset.restype = None
    lib.ck_model_sample_argmax.argtypes = []
    lib.ck_model_sample_argmax.restype = ctypes.c_int

    # Initialize model
    ret = lib.ck_model_init(str(weights_path).encode())
    if ret != 0:
        print(f"{RED}CK-Engine init failed: {ret}{RESET}")
        return None

    prefill_times = []
    decode_times = []

    tokens_array = (ctypes.c_int32 * len(prompt_tokens))(*prompt_tokens)

    try:
        for run in range(num_runs):
            # Reset KV cache
            lib.ck_model_kv_cache_enable(32768)
            lib.ck_model_kv_cache_reset()

            # Embed tokens
            lib.ck_model_embed_tokens(tokens_array, len(prompt_tokens))

            # Prefill
            t0 = time.perf_counter()
            lib.ck_model_forward(None)
            t1 = time.perf_counter()
            prefill_times.append((t1 - t0) * 1000)

            # Decode
            t0 = time.perf_counter()
            for i in range(decode_tokens):
                next_token = lib.ck_model_sample_argmax()
                ret = lib.ck_model_decode(next_token, None)
                if ret != 0:
                    break
            t1 = time.perf_counter()
            decode_times.append((t1 - t0) * 1000)

    finally:
        lib.ck_model_free()

    # Average across runs (skip first run for warmup if multiple runs)
    if num_runs > 1:
        prefill_times = prefill_times[1:]
        decode_times = decode_times[1:]

    return BenchmarkResult(
        name="CK-Engine",
        prefill_tokens=len(prompt_tokens),
        prefill_time_ms=sum(prefill_times) / len(prefill_times),
        decode_tokens=decode_tokens,
        decode_time_ms=sum(decode_times) / len(decode_times)
    )


def benchmark_llamacpp(gguf_path: Path, prompt: str, decode_tokens: int,
                       num_runs: int = 3) -> Optional[BenchmarkResult]:
    """Benchmark llama.cpp using llama-cli"""

    if not LLAMACPP_CLI.exists():
        print(f"{RED}llama-cli not found at {LLAMACPP_CLI}{RESET}")
        return None

    prefill_times = []
    decode_times = []

    for run in range(num_runs):
        # Run llama-cli with timing
        cmd = [
            str(LLAMACPP_CLI),
            "-m", str(gguf_path),
            "-p", prompt,
            "-n", str(decode_tokens),
            "--temp", "0",  # Greedy decoding
            "-t", "1",  # Single thread for fair comparison
            "--no-mmap",
            "-ngl", "0",  # CPU only
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            # Parse timing from output
            # llama.cpp outputs timing info like:
            # llama_perf_context_print: prompt eval time = X.XX ms / N tokens
            # llama_perf_context_print: eval time = Y.YY ms / M tokens

            prefill_match = re.search(
                r'prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens',
                result.stderr + result.stdout
            )
            decode_match = re.search(
                r'eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(tokens|runs)',
                result.stderr + result.stdout
            )

            if prefill_match and decode_match:
                prefill_times.append(float(prefill_match.group(1)))
                decode_times.append(float(decode_match.group(1)))
            else:
                # Try alternative parsing
                # Some versions output differently
                lines = (result.stderr + result.stdout).split('\n')
                for line in lines:
                    if 'prompt eval' in line.lower() and 'ms' in line:
                        match = re.search(r'([\d.]+)\s*ms', line)
                        if match:
                            prefill_times.append(float(match.group(1)))
                    elif 'eval time' in line.lower() and 'ms' in line:
                        match = re.search(r'([\d.]+)\s*ms', line)
                        if match:
                            decode_times.append(float(match.group(1)))

        except subprocess.TimeoutExpired:
            print(f"{RED}llama.cpp timeout on run {run + 1}{RESET}")
        except Exception as e:
            print(f"{RED}llama.cpp error: {e}{RESET}")

    if not prefill_times or not decode_times:
        print(f"{RED}Could not parse llama.cpp timing output{RESET}")
        return None

    # Average across runs (skip first run for warmup if multiple runs)
    if len(prefill_times) > 1:
        prefill_times = prefill_times[1:]
    if len(decode_times) > 1:
        decode_times = decode_times[1:]

    # Get token count from prompt
    # Approximate - llama.cpp uses its own tokenizer
    prompt_tokens = len(prompt.split()) * 1.3  # Rough estimate

    return BenchmarkResult(
        name="llama.cpp",
        prefill_tokens=int(prompt_tokens),
        prefill_time_ms=sum(prefill_times) / len(prefill_times),
        decode_tokens=decode_tokens,
        decode_time_ms=sum(decode_times) / len(decode_times)
    )


def benchmark_llamacpp_bench(gguf_path: Path, prompt_tokens: int, decode_tokens: int) -> Optional[BenchmarkResult]:
    """Use llama-bench for more accurate benchmarking"""

    if not LLAMACPP_BENCH.exists():
        return None

    cmd = [
        str(LLAMACPP_BENCH),
        "-m", str(gguf_path),
        "-p", str(prompt_tokens),
        "-n", str(decode_tokens),
        "-t", "1",  # Single thread
        "-ngl", "0",  # CPU only
        "-r", "3",  # 3 repetitions
        "-o", "json"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        # Parse JSON output
        for line in result.stdout.split('\n'):
            if line.strip().startswith('['):
                data = json.loads(line)
                if data:
                    item = data[0]
                    return BenchmarkResult(
                        name="llama.cpp (bench)",
                        prefill_tokens=item.get('n_prompt', prompt_tokens),
                        prefill_time_ms=1000.0 / item.get('avg_ts', 1),  # Convert t/s to ms
                        decode_tokens=item.get('n_gen', decode_tokens),
                        decode_time_ms=item.get('avg_ts', 0)  # This is already in the right format
                    )

    except Exception as e:
        print(f"{YELLOW}llama-bench failed: {e}{RESET}")

    return None


def print_comparison(ck_result: BenchmarkResult, llama_result: BenchmarkResult):
    """Print side-by-side comparison"""

    print(f"\n{CYAN}{'='*70}{RESET}")
    print(f"{CYAN}{BOLD}Performance Comparison: CK-Engine vs llama.cpp{RESET}")
    print(f"{CYAN}{'='*70}{RESET}")

    print(f"\n{'Metric':<25} {'CK-Engine':>15} {'llama.cpp':>15} {'Speedup':>12}")
    print(f"{'-'*70}")

    # Prefill
    speedup = llama_result.prefill_time_ms / ck_result.prefill_time_ms if ck_result.prefill_time_ms > 0 else 0
    color = GREEN if speedup > 1.0 else RED
    print(f"{'Prefill Time (ms)':<25} {ck_result.prefill_time_ms:>15.1f} {llama_result.prefill_time_ms:>15.1f} {color}{speedup:>11.2f}x{RESET}")

    speedup = ck_result.prefill_tps / llama_result.prefill_tps if llama_result.prefill_tps > 0 else 0
    color = GREEN if speedup > 1.0 else RED
    print(f"{'Prefill (tok/s)':<25} {ck_result.prefill_tps:>15.1f} {llama_result.prefill_tps:>15.1f} {color}{speedup:>11.2f}x{RESET}")

    # Decode
    speedup = llama_result.decode_time_ms / ck_result.decode_time_ms if ck_result.decode_time_ms > 0 else 0
    color = GREEN if speedup > 1.0 else RED
    print(f"{'Decode Time (ms)':<25} {ck_result.decode_time_ms:>15.1f} {llama_result.decode_time_ms:>15.1f} {color}{speedup:>11.2f}x{RESET}")

    speedup = ck_result.decode_tps / llama_result.decode_tps if llama_result.decode_tps > 0 else 0
    color = GREEN if speedup > 1.0 else RED
    print(f"{'Decode (tok/s)':<25} {ck_result.decode_tps:>15.1f} {llama_result.decode_tps:>15.1f} {color}{speedup:>11.2f}x{RESET}")

    # Total
    speedup = llama_result.total_time_ms / ck_result.total_time_ms if ck_result.total_time_ms > 0 else 0
    color = GREEN if speedup > 1.0 else RED
    print(f"{'-'*70}")
    print(f"{'Total Time (ms)':<25} {ck_result.total_time_ms:>15.1f} {llama_result.total_time_ms:>15.1f} {color}{speedup:>11.2f}x{RESET}")

    print(f"\n{CYAN}{'='*70}{RESET}")

    if speedup > 1.0:
        print(f"{GREEN}CK-Engine is {speedup:.2f}x faster overall{RESET}")
    else:
        print(f"{YELLOW}llama.cpp is {1/speedup:.2f}x faster overall{RESET}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark CK-Engine vs llama.cpp")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B-Instruct-GGUF",
                        help="Model to benchmark")
    parser.add_argument("--tokens", type=int, default=50,
                        help="Number of tokens to generate")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of benchmark runs (first is warmup)")
    parser.add_argument("--threads", type=int, default=1,
                        help="Number of threads for llama.cpp")
    args = parser.parse_args()

    print(f"{CYAN}{'='*70}{RESET}")
    print(f"{CYAN}{BOLD}CK-Engine vs llama.cpp Performance Benchmark{RESET}")
    print(f"{CYAN}{'='*70}{RESET}")
    print(f"\nModel: {args.model}")
    print(f"Decode tokens: {args.tokens}")
    print(f"Runs: {args.runs}")

    # Find model files
    gguf_path = find_gguf_model(args.model)
    ck_lib_path = find_ck_engine_lib(args.model)
    ck_weights_path = find_ck_engine_weights(args.model)

    if not gguf_path:
        print(f"\n{RED}GGUF model not found for {args.model}{RESET}")
        print("Run: python scripts/v6.5/ck_run_v6_5.py run <model> to download and convert")
        return 1

    if not ck_lib_path or not ck_weights_path:
        print(f"\n{RED}CK-Engine model not found for {args.model}{RESET}")
        print("Run: python scripts/v6.5/ck_run_v6_5.py run <model> --force-compile")
        return 1

    print(f"\nGGUF: {gguf_path}")
    print(f"CK-Engine lib: {ck_lib_path}")
    print(f"CK-Engine weights: {ck_weights_path}")

    # Standard test prompt (matches Qwen chat format)
    prompt_tokens = [
        151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198,
        151644, 872, 198, 3838, 374, 279, 6864, 315, 9625, 30, 151645, 198,
        151644, 77091, 198
    ]
    prompt_text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"

    # Benchmark CK-Engine
    print(f"\n{CYAN}Benchmarking CK-Engine...{RESET}")
    ck_result = benchmark_ck_engine(
        ck_lib_path, ck_weights_path,
        prompt_tokens, args.tokens, args.runs
    )

    if ck_result:
        print(f"  Prefill: {ck_result.prefill_time_ms:.1f}ms ({ck_result.prefill_tps:.1f} tok/s)")
        print(f"  Decode:  {ck_result.decode_time_ms:.1f}ms ({ck_result.decode_tps:.1f} tok/s)")
    else:
        print(f"  {RED}Failed{RESET}")

    # Benchmark llama.cpp
    print(f"\n{CYAN}Benchmarking llama.cpp...{RESET}")
    llama_result = benchmark_llamacpp(
        gguf_path, prompt_text, args.tokens, args.runs
    )

    if llama_result:
        print(f"  Prefill: {llama_result.prefill_time_ms:.1f}ms ({llama_result.prefill_tps:.1f} tok/s)")
        print(f"  Decode:  {llama_result.decode_time_ms:.1f}ms ({llama_result.decode_tps:.1f} tok/s)")
    else:
        print(f"  {RED}Failed to parse timing{RESET}")

        # Try llama-bench as fallback
        print(f"\n{CYAN}Trying llama-bench...{RESET}")
        llama_result = benchmark_llamacpp_bench(
            gguf_path, len(prompt_tokens), args.tokens
        )
        if llama_result:
            print(f"  Prefill: {llama_result.prefill_time_ms:.1f}ms")
            print(f"  Decode:  {llama_result.decode_time_ms:.1f}ms ({llama_result.decode_tps:.1f} tok/s)")

    # Print comparison
    if ck_result and llama_result:
        print_comparison(ck_result, llama_result)
    else:
        print(f"\n{YELLOW}Could not complete comparison - one or both benchmarks failed{RESET}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
