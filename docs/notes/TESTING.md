# Kernel Testing Guide

This document explains how to build and test individual kernels in C-Kernel-Engine
using the provided Python unittest scripts.

All tests compare the C kernels against PyTorch reference implementations
(forward and backward) and fail on any significant mismatch.

---

## 1. Build Shared Libraries

From the project root:

```bash
cd C-Kernel-Engine

# Build the full engine (all kernels)
make

# Build small, kernel-specific libraries
make libckernel_gelu.so
make libckernel_rmsnorm.so
make libckernel_layernorm.so
make libckernel_softmax.so
```

All kernel-specific libs are compiled with the same ISA flags as the main build
(AVX / AVX2 / AVX-512), so they are safe to run on the current CPU.

There is also a helper target that builds all kernel-specific libs:

```bash
make test-libs
```

This is invoked by `make test`.

BF16 kernels require AVX-512 BF16 hardware support. You can run the BF16 test
suite with:

```bash
make test-bf16
```

On unsupported CPUs, BF16 tests print a clear skip message.

---

## 2. Running Kernel Tests (Python)

Each kernel family has a corresponding test script under `unittest/`.
These scripts:

- Load the appropriate shared library via `ctypes`.
- Generate random tensors with PyTorch.
- Compute PyTorch reference outputs/gradients.
- Call the C kernels and compare results.

### 2.1 GELU

Build the GELU-only library and run tests:

```bash
make libckernel_gelu.so
python3 unittest/test_gelu.py
```

This script checks:

- Forward: `gelu_fast_inplace` vs `torch.nn.functional.gelu(..., approximate="tanh")`
- Backward (exact): `gelu_backward_exact` vs PyTorch autograd
- Backward (fast): `gelu_backward_fast` is treated as approximate; its max diff
  is printed but not enforced in `make test`.

### 2.2 RMSNorm

Build the RMSNorm-only library and run tests:

```bash
make libckernel_rmsnorm.so
python3 unittest/test_rmsnorm.py
```

This script checks:

- Forward: `rmsnorm_forward` vs a PyTorch RMSNorm reference:
  ```python
  var = x.pow(2).mean(dim=-1, keepdim=True)
  rstd = (var + eps).rsqrt()
  y = x * rstd * gamma
  ```
- Backward: `rmsnorm_backward` vs PyTorch autograd for `dX` and `dGamma`.

### 2.3 LayerNorm

Build uses the best available implementation for your CPU (AVX / AVX2 / AVX-512).

```bash
make libckernel_layernorm.so
python3 unittest/test_layernorm.py
```

Checks:

- Forward:
  - `layernorm_naive_serial`
  - `layernorm_forward_rolled_slice`
  - `layernorm_forward_unrolled_slice`
  vs `torch.layer_norm`.
- Backward:
  - `layernorm_backward_kernel` vs PyTorch autograd for `dX`, `dGamma`, `dBeta`.

### 2.4 Softmax (Causal)

```bash
make libckernel_softmax.so
python3 unittest/test_softmax.py
python3 unittest/test_softmax_backward.py
```

Checks:

- Forward: `causal_softmax_head_major` vs a PyTorch masked softmax reference.
- Backward: `backward_causal_softmax_head_major` vs a pure PyTorch
  Jacobian–vector-product implementation.

### 2.5 GEMM and MLP

GEMM tests:

```bash
python3 unittest/test_gemm.py
```

  Checks all GEMM variants (`gemm_naive_parallel`, `gemm_avx512_parallel`,
  `gemm_fine_grained_parallel`, `gemm_blocked_serial`) against
  `A @ B.T + bias` for LLM-relevant shapes.

MLP tests:

```bash
python3 unittest/test_mlp.py
```

  Checks:

  - Forward: `mlp_token_parallel` vs `Linear(D→4D) → GELU → Linear(4D→D)` in PyTorch.
  - Backward: FC1/FC2 + GELU derivatives vs full PyTorch autograd
    for `dX`, `dW1`, `db1`, `dW2`, `db2`.

### 2.6 SwiGLU

Build the SwiGLU-only library and run tests:

```bash
make libckernel_swiglu.so
python3 unittest/test_swiglu.py
```

Checks:

  - Forward: `swiglu_forward` vs a PyTorch reference:

  ```python
  gate, value = x[:, :D], x[:, D:]
  y = F.silu(gate) * value
  ```

  - Backward: `swiglu_backward` vs PyTorch autograd for `dX` (both gate and value parts).

### 2.7 Sigmoid

Build the Sigmoid-only library and run tests:

```bash
make libckernel_sigmoid.so
python3 unittest/test_sigmoid.py
```

Checks:

- Forward: `sigmoid_forward` vs `torch.sigmoid`.
- Backward: `sigmoid_backward` vs PyTorch autograd for `dX`.

### 2.8 Flash Attention + KV Cache (Inference Decode Path)

These tests validate the new inference-oriented attention path:

```bash
python3 unittest/test_kv_cache_attention.py
python3 unittest/test_kv_cache_layer_decode.py
```

They check:
- Flash-style prefill attention matches the reference score-matrix attention.
- KV-cache decode attention matches full causal attention token-by-token.
- Full layer output matches prefill+decode stitching (and KV padded lanes are zeroed).

---

## 3. Running the Full Test Suite

```bash
cd C-Kernel-Engine
make
make test
```

This will:

- Build the full engine library (`libckernel_engine.so`).
- Build all kernel-specific libs (`make test-libs`).
- Run all Python unit tests listed in `make tests-list`.

---

## 4. End-to-End / Orchestration Tests

These test the full decoder layer orchestration and the tiny training loop.

### 4.1 Layer Parity (strict C ref)

Runs the decoder layer against the strict C reference path (naive GEMM),
which is tight enough for AVX-only CPUs:

```bash
make layer-parity-scalar TOL=2e-3 ARGS="--tokens 256 --embed 64 --heads 4 --kv-heads 2 --intermediate 128 --rope --strict-ref"
```

### 4.2 Tiny End-to-End Training Parity (C vs PyTorch)

Runs a tiny model for 1 step and compares loss + final weights vs PyTorch:

```bash
make tiny-parity
```

Multi-step:

```bash
python3 scripts/tiny_train_parity.py --config tiny.config.json --steps 5 --lr 1e-3
```

### 4.3 One-Command Test Sweep

Runs kernel tests, layer parity, and tiny training parity:

```bash
make all-tests
```

You can tune the layer parity args with:

```bash
make all-tests ALL_TEST_LAYER_ARGS="--tokens 256 --embed 64 --heads 4 --kv-heads 2 --intermediate 128 --rope --strict-ref" ALL_TEST_LAYER_TOL=1e-3
```

---

## 5. Notes

- All tests print max differences and sample tensor entries when mismatches
  exceed tight tolerances (~1e-6).
- Exact kernels (GELU exact backward, LayerNorm, RMSNorm, softmax) are treated
  as correctness-critical: any significant difference causes the test to fail.
- Approximate kernels (e.g., fast GELU backward) are considered experimental
  and do not affect `make test` unless explicitly enabled in their scripts.
