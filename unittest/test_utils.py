"""
Test utilities for C-Kernel-Engine unit tests.

Provides:
- CPU capability detection (AVX, AVX-512, etc.)
- Performance timing utilities
- Dtype information
- Pretty-printed test reports
"""
import ctypes
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════════════
# CPU Capability Detection
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CPUInfo:
    """CPU capability information."""
    model_name: str = "Unknown"
    vendor: str = "Unknown"
    # SIMD capabilities
    sse: bool = False
    sse2: bool = False
    sse3: bool = False
    ssse3: bool = False
    sse4_1: bool = False
    sse4_2: bool = False
    avx: bool = False
    avx2: bool = False
    fma: bool = False
    avx512f: bool = False
    avx512bw: bool = False
    avx512vl: bool = False
    avx512bf16: bool = False
    # Other features
    num_cores: int = 1
    cache_line_size: int = 64

    @property
    def best_simd(self) -> str:
        """Return the best available SIMD instruction set."""
        if self.avx512f:
            extras = []
            if self.avx512bf16:
                extras.append("BF16")
            if self.avx512bw:
                extras.append("BW")
            if self.avx512vl:
                extras.append("VL")
            suffix = f" ({', '.join(extras)})" if extras else ""
            return f"AVX-512{suffix}"
        elif self.avx2:
            return "AVX2 + FMA" if self.fma else "AVX2"
        elif self.avx:
            return "AVX"
        elif self.sse4_2:
            return "SSE4.2"
        elif self.sse2:
            return "SSE2"
        else:
            return "Scalar"


def _detect_cpu_linux() -> CPUInfo:
    """Detect CPU capabilities on Linux via /proc/cpuinfo and lscpu."""
    info = CPUInfo()

    # Read /proc/cpuinfo
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()

        for line in cpuinfo.split("\n"):
            if line.startswith("model name"):
                info.model_name = line.split(":")[1].strip()
            elif line.startswith("vendor_id"):
                info.vendor = line.split(":")[1].strip()
            elif line.startswith("flags"):
                flags = line.split(":")[1].strip().split()
                info.sse = "sse" in flags
                info.sse2 = "sse2" in flags
                info.sse3 = "sse3" in flags or "pni" in flags
                info.ssse3 = "ssse3" in flags
                info.sse4_1 = "sse4_1" in flags
                info.sse4_2 = "sse4_2" in flags
                info.avx = "avx" in flags
                info.avx2 = "avx2" in flags
                info.fma = "fma" in flags
                info.avx512f = "avx512f" in flags
                info.avx512bw = "avx512bw" in flags
                info.avx512vl = "avx512vl" in flags
                info.avx512bf16 = "avx512_bf16" in flags
                break
    except FileNotFoundError:
        pass

    # Get core count
    try:
        info.num_cores = os.cpu_count() or 1
    except:
        pass

    return info


def _detect_cpu_macos() -> CPUInfo:
    """Detect CPU capabilities on macOS via sysctl."""
    info = CPUInfo()

    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True
        )
        info.model_name = result.stdout.strip()

        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.features"],
            capture_output=True, text=True
        )
        features = result.stdout.strip().upper().split()
        info.sse = "SSE" in features
        info.sse2 = "SSE2" in features
        info.sse3 = "SSE3" in features
        info.ssse3 = "SSSE3" in features
        info.sse4_1 = "SSE4.1" in features
        info.sse4_2 = "SSE4.2" in features
        info.avx = "AVX1.0" in features or "AVX" in features
        info.avx2 = "AVX2" in features
        info.fma = "FMA" in features

        # Check leaf7 features for AVX-512
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.leaf7_features"],
            capture_output=True, text=True
        )
        leaf7 = result.stdout.strip().upper().split()
        info.avx512f = "AVX512F" in leaf7
        info.avx512bw = "AVX512BW" in leaf7
        info.avx512vl = "AVX512VL" in leaf7
        info.avx512bf16 = "AVX512BF16" in leaf7
    except:
        pass

    info.num_cores = os.cpu_count() or 1
    return info


def detect_cpu() -> CPUInfo:
    """Detect CPU capabilities for the current system."""
    system = platform.system()
    if system == "Linux":
        return _detect_cpu_linux()
    elif system == "Darwin":
        return _detect_cpu_macos()
    else:
        return CPUInfo(num_cores=os.cpu_count() or 1)


# Global CPU info (cached)
_cpu_info: Optional[CPUInfo] = None

def get_cpu_info() -> CPUInfo:
    """Get cached CPU info."""
    global _cpu_info
    if _cpu_info is None:
        _cpu_info = detect_cpu()
    return _cpu_info


# ═══════════════════════════════════════════════════════════════════════════════
# Timing Utilities
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TimingResult:
    """Result of a timed operation."""
    name: str
    mean_us: float  # microseconds
    std_us: float
    min_us: float
    max_us: float
    iterations: int

    @property
    def mean_ms(self) -> float:
        return self.mean_us / 1000.0

    def __str__(self) -> str:
        return f"{self.name}: {self.mean_us:.1f} +/- {self.std_us:.1f} us (min={self.min_us:.1f}, max={self.max_us:.1f})"


def time_function(
    fn: Callable,
    warmup: int = 5,
    iterations: int = 100,
    name: str = "function"
) -> TimingResult:
    """
    Time a function with warmup and multiple iterations.

    Args:
        fn: Function to time (should take no arguments)
        warmup: Number of warmup iterations
        iterations: Number of timed iterations
        name: Name for the result

    Returns:
        TimingResult with timing statistics
    """
    # Warmup
    for _ in range(warmup):
        fn()

    # Sync if using CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed iterations
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # Convert to microseconds

    times = np.array(times)
    return TimingResult(
        name=name,
        mean_us=float(np.mean(times)),
        std_us=float(np.std(times)),
        min_us=float(np.min(times)),
        max_us=float(np.max(times)),
        iterations=iterations
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test Report Formatting
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    """Result of a single test comparison."""
    name: str
    passed: bool
    max_diff: float
    tolerance: float
    pytorch_time: Optional[TimingResult] = None
    kernel_time: Optional[TimingResult] = None

    @property
    def speedup(self) -> Optional[float]:
        """Return speedup of kernel vs PyTorch (>1 means kernel is faster)."""
        if self.pytorch_time and self.kernel_time:
            return self.pytorch_time.mean_us / self.kernel_time.mean_us
        return None


@dataclass
class TestReport:
    """Complete test report with all results and system info."""
    test_name: str
    dtype: str = "fp32"
    shape: str = ""
    results: List[TestResult] = field(default_factory=list)
    cpu_info: Optional[CPUInfo] = None

    def add_result(self, result: TestResult):
        self.results.append(result)

    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def print_report(self):
        """Print a comprehensive test report."""
        cpu = self.cpu_info or get_cpu_info()

        # Header
        print()
        print("=" * 80)
        print(f"  TEST: {self.test_name}")
        print("=" * 80)

        # System info
        print()
        print("  SYSTEM INFO")
        print("  " + "-" * 40)
        print(f"  CPU:        {cpu.model_name}")
        print(f"  Cores:      {cpu.num_cores}")
        print(f"  SIMD:       {cpu.best_simd}")
        print(f"  Dtype:      {self.dtype}")
        if self.shape:
            print(f"  Shape:      {self.shape}")

        # Results
        print()
        print("  ACCURACY")
        print("  " + "-" * 40)

        max_name_len = max(len(r.name) for r in self.results) if self.results else 10
        for r in self.results:
            status = "\033[92mPASS\033[0m" if r.passed else "\033[91mFAIL\033[0m"
            print(f"  {r.name:<{max_name_len}}  max_diff={r.max_diff:.2e}  tol={r.tolerance:.0e}  [{status}]")

        # Performance (if available)
        has_timing = any(r.pytorch_time or r.kernel_time for r in self.results)
        if has_timing:
            print()
            print("  PERFORMANCE")
            print("  " + "-" * 40)
            header = (
                f"  {'Kernel':<{max_name_len}}  "
                f"{'PyTorch (us)':<15}  {'C Kernel (us)':<15}  {'Speedup':<9}  "
                f"{'max_diff':<10}  {'tol':<6}"
            )
            print(header)
            print("  " + "-" * len(header.strip()))

            for r in self.results:
                pt_str = f"{r.pytorch_time.mean_us:.1f}" if r.pytorch_time else "N/A"
                ck_str = f"{r.kernel_time.mean_us:.1f}" if r.kernel_time else "N/A"
                speedup = r.speedup
                if speedup is not None:
                    if speedup >= 1.0:
                        sp_str = f"\033[92m{speedup:.2f}x\033[0m"
                    else:
                        sp_str = f"\033[93m{speedup:.2f}x\033[0m"
                else:
                    sp_str = "N/A"
                diff_str = f"{r.max_diff:.2e}" if r.max_diff is not None else "N/A"
                tol_str = f"{r.tolerance:.0e}" if r.tolerance is not None else "N/A"
                print(
                    f"  {r.name:<{max_name_len}}  {pt_str:<15}  {ck_str:<15}  {sp_str:<9}  "
                    f"{diff_str:<10}  {tol_str:<6}"
                )

        # Summary
        print()
        print("  " + "-" * 40)
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        if self.all_passed():
            print(f"  \033[92mALL TESTS PASSED ({passed}/{total})\033[0m")
        else:
            print(f"  \033[91mSOME TESTS FAILED ({passed}/{total} passed)\033[0m")
        print("=" * 80)
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════════

def tensor_to_ptr(t: torch.Tensor) -> ctypes.POINTER(ctypes.c_float):
    """Convert a torch tensor to a ctypes float pointer."""
    return t.contiguous().view(-1).numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def numpy_to_ptr(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    """Convert a numpy array to a ctypes float pointer (faster than tensor_to_ptr)."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute maximum absolute difference between two tensors."""
    return (a - b).abs().max().item()


def print_system_info():
    """Print system information header."""
    cpu = get_cpu_info()
    print()
    print("=" * 60)
    print("  C-Kernel-Engine Unit Test Suite")
    print("=" * 60)
    print(f"  CPU:    {cpu.model_name}")
    print(f"  Cores:  {cpu.num_cores}")
    print(f"  SIMD:   {cpu.best_simd}")
    print(f"  PyTorch: {torch.__version__}")
    print("=" * 60)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Quick Test Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_comparison_test(
    test_name: str,
    pytorch_fn: Callable[[], torch.Tensor],
    kernel_fn: Callable[[], torch.Tensor],
    kernel_name: str = "C Kernel",
    tolerance: float = 1e-5,
    warmup: int = 5,
    iterations: int = 100,
    dtype: str = "fp32",
    shape: str = ""
) -> TestResult:
    """
    Run a comparison test between PyTorch and C kernel.

    Returns a TestResult with accuracy and timing information.
    """
    # Run once to get outputs
    ref = pytorch_fn()
    out = kernel_fn()

    diff = max_diff(out, ref)
    passed = diff <= tolerance

    # Time both
    pt_time = time_function(pytorch_fn, warmup, iterations, "PyTorch")
    ck_time = time_function(kernel_fn, warmup, iterations, kernel_name)

    return TestResult(
        name=kernel_name,
        passed=passed,
        max_diff=diff,
        tolerance=tolerance,
        pytorch_time=pt_time,
        kernel_time=ck_time
    )
