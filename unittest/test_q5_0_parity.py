import ctypes
import numpy as np
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
LIB_PATH = PROJECT_ROOT / "build/libckernel_test.so"

def setup_test_lib():
    src_files = [
        "src/kernels/gemm_kernels_q5_0.c",
        "src/kernels/gemm_kernels_q5_0_sse_v2.c",
        "src/kernels/gemm_kernels_q4k_q8k.c",
        "src/kernels/quantize_row_q8_k_sse.c",
        "src/kernels/gemm_kernels_q4k_sse.c",
        "src/cpu_features.c",
    ]
    include_dir = PROJECT_ROOT / "include"
    (PROJECT_ROOT / "build").mkdir(exist_ok=True)
    cmd = f"gcc -O3 -march=native -fPIC -shared -I{include_dir} {' '.join(str(PROJECT_ROOT / f) for f in src_files)} -o {LIB_PATH} -lm"
    os.system(cmd)
    return ctypes.CDLL(str(LIB_PATH))

def test_q5_0_parity():
    lib = setup_test_lib()
    
    lib.gemm_nt_q5_0_ref.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.gemm_nt_q5_0_sse_v2.argtypes = lib.gemm_nt_q5_0_ref.argtypes

    M, N, K = 1, 128, 4096
    
    num_blocks = (N * K) // 32
    weights_buffer = (ctypes.c_ubyte * (num_blocks * 22))()
    np.random.seed(42)
    for i in range(num_blocks):
        off = i * 22
        # d = random scale
        d_val = np.random.rand() * 0.1
        d_fp16 = np.frombuffer(np.array([d_val], dtype=np.float16).tobytes(), dtype=np.uint8)
        weights_buffer[off] = d_fp16[0]
        weights_buffer[off+1] = d_fp16[1]
        for j in range(4): weights_buffer[off+2+j] = np.random.randint(0, 256)
        for j in range(16): weights_buffer[off+6+j] = np.random.randint(0, 256)
    
    A = np.random.randn(M, K).astype(np.float32)
    C_ref = np.zeros((M, N), dtype=np.float32)
    C_sse = np.zeros((M, N), dtype=np.float32)

    print(f"Sample A: {A[0, :4]}")
    print(f"Sample weights[6:10]: {[weights_buffer[i] for i in range(6, 10)]}")

    lib.gemm_nt_q5_0_ref(A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), weights_buffer, None, C_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), M, N, K)
    lib.gemm_nt_q5_0_sse_v2(A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), weights_buffer, None, C_sse.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), M, N, K)

    mse = np.mean((C_ref - C_sse)**2)
    max_diff = np.max(np.abs(C_ref - C_sse))
    print(f"MSE: {mse:.6e}, Max Diff: {max_diff:.6e}")
    
    if max_diff > 1.0:
        print("FAIL: SSE doesn't match Ref")
    else:
        print("PASS: SSE matches Ref (within noise)")

if __name__ == "__main__":
    test_q5_0_parity()