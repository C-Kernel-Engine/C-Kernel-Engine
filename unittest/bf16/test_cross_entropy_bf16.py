"""
Softmax cross-entropy BF16 kernel unit test.

The BF16 wrapper computes softmax + CE loss and returns BF16 d_logits.
We compare against a PyTorch reference that first quantizes logits to BF16
then performs the softmax/CE math in FP32 (matching the C implementation).
"""
import ctypes
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UNITS = ROOT / "unittest"
for path in (ROOT, UNITS):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import numpy as np
import torch

from lib_loader import load_lib
from test_utils import (
    TestReport,
    TestResult,
    get_cpu_info,
    max_diff,
    time_function,
    print_system_info,
)
from bf16_utils import float32_to_bf16, bf16_to_float32, numpy_to_uint16_ptr

cpu = get_cpu_info()
if not cpu.avx512bf16:
    print("BF16 kernels require AVX-512 BF16; skipping this test on the current CPU.")
    sys.exit(0)

lib = load_lib("libckernel_engine.so")

# Function signatures - updated with scratch buffer params (no internal malloc)
lib.softmax_cross_entropy_loss_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # logits
    ctypes.POINTER(ctypes.c_int32),   # targets
    ctypes.c_int,                     # tokens
    ctypes.c_int,                     # vocab_size
    ctypes.POINTER(ctypes.c_uint16),  # d_logits
    ctypes.POINTER(ctypes.c_float),   # loss_out
    ctypes.POINTER(ctypes.c_float),   # scratch_logits [tokens * vocab_size]
    ctypes.POINTER(ctypes.c_float),   # scratch_d_logits [tokens * vocab_size]
]
lib.softmax_cross_entropy_loss_bf16.restype = None


def run_tests(T=64, V=512, warmup=10, iterations=500):
    np.random.seed(0)
    logits_np = (np.random.randn(T, V).astype(np.float32) * 0.5)
    targets = np.random.randint(0, V, size=(T,), dtype=np.int32)

    logits_bf16 = float32_to_bf16(logits_np.reshape(-1))
    d_logits_bf16 = np.zeros_like(logits_bf16)
    loss_out = np.zeros(1, dtype=np.float32)

    # Allocate scratch buffers (caller-provided, no malloc in kernel)
    scratch_logits = np.zeros(T * V, dtype=np.float32)
    scratch_d_logits = np.zeros(T * V, dtype=np.float32)
    scratch_logits_ptr = scratch_logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    scratch_d_logits_ptr = scratch_d_logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    targets_t = torch.from_numpy(targets).long()
    logits_t = torch.from_numpy(logits_np.copy()).to(dtype=torch.bfloat16)
    logits_f = logits_t.to(dtype=torch.float32)

    report = TestReport(
        test_name="Softmax Cross-Entropy (BF16)",
        dtype="bf16",
        shape=f"T={T}, V={V}",
        cpu_info=get_cpu_info(),
    )

    def pytorch_ref():
        log_probs = torch.log_softmax(logits_f, dim=-1)
        loss = -log_probs[torch.arange(T), targets_t].mean()
        probs = torch.softmax(logits_f, dim=-1)
        probs[torch.arange(T), targets_t] -= 1.0
        probs /= float(T)
        return loss, probs

    loss_ref, d_logits_ref = pytorch_ref()

    def c_kernel():
        lib.softmax_cross_entropy_loss_bf16(
            numpy_to_uint16_ptr(logits_bf16),
            targets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int(T),
            ctypes.c_int(V),
            numpy_to_uint16_ptr(d_logits_bf16),
            loss_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            scratch_logits_ptr,
            scratch_d_logits_ptr,
        )

    c_kernel()

    d_logits_c = torch.from_numpy(bf16_to_float32(d_logits_bf16.copy()).reshape(T, V))
    loss_c = float(loss_out[0])

    loss_diff = abs(loss_c - float(loss_ref))
    grad_diff = max_diff(d_logits_c, d_logits_ref)

    report.add_result(
        TestResult(
            name="loss",
            passed=loss_diff <= 1e-2,
            max_diff=loss_diff,
            tolerance=1e-2,
            pytorch_time=time_function(lambda: pytorch_ref()[0], warmup=warmup, iterations=iterations, name="PyTorch"),
            kernel_time=time_function(c_kernel, warmup=warmup, iterations=iterations, name="C CE BF16"),
        )
    )
    report.add_result(
        TestResult(
            name="d_logits",
            passed=grad_diff <= 2e-2,
            max_diff=grad_diff,
            tolerance=2e-2,
        )
    )

    return report


if __name__ == "__main__":
    print_system_info()

    report = run_tests()
    report.print_report()

    if not report.all_passed():
        sys.exit(1)

