#!/usr/bin/env python3
"""Executable PyTorch parity gate for Qwen3.6 BF16 Gated DeltaNet decode."""

from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[1]


def _library_path() -> Path:
    explicit = os.environ.get("CK_ENGINE_LIB")
    candidates = [Path(explicit)] if explicit else []
    candidates.extend((ROOT / "build" / "libckernel_engine.so", ROOT / "libckernel_engine.so"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    pytest.skip("libckernel_engine.so is not built")


def _load_provider():
    torch_library = Path(torch.__file__).resolve().parent / "lib" / "libtorch_cpu.so"
    if torch_library.is_file():
        os.environ.setdefault("CK_MKL_LIBRARY", str(torch_library))
        os.environ.setdefault("CK_SLEEF_LIBRARY", str(torch_library))
    lib = ctypes.CDLL(str(_library_path()))
    provider = lib.gated_deltanet_pytorch_grouped_bf16_forward
    pointer = ctypes.POINTER(ctypes.c_float)
    provider.argtypes = [
        pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
    ]
    provider.restype = None
    return provider


def _pointer(array: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _bf16_values(array: np.ndarray) -> np.ndarray:
    return torch.from_numpy(array).to(torch.bfloat16).float().numpy()


def _pytorch_reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror Transformers torch_recurrent_gated_delta_rule for one token.

    The graph normalizes compact Q/K before this provider. Transformers then
    repeat_interleave expands them across adjacent value heads, performs the
    recurrent state update in FP32, and stores the emitted output to BF16.
    CKE receives the pre-sigmoid beta projection, so its BF16 sigmoid boundary
    is reproduced explicitly here.
    """

    num_heads, state_dim = v.shape
    group_count = q.shape[0]
    repeats = num_heads // group_count
    tq = torch.from_numpy(q).repeat_interleave(repeats, dim=0)
    tk = torch.from_numpy(k).repeat_interleave(repeats, dim=0)
    tv = torch.from_numpy(v)
    tg = torch.from_numpy(g)
    tbeta = torch.sigmoid(torch.from_numpy(beta)).to(torch.bfloat16).float()
    current = torch.from_numpy(state) * torch.exp(tg)[:, None, None]
    memory = (current * tk[:, :, None]).sum(dim=1)
    delta = (tv - memory) * tbeta[:, None]
    current = current + tk[:, :, None] * delta[:, None, :]
    output = (current * (tq / math.sqrt(state_dim))[:, :, None]).sum(dim=1)
    output = output.to(torch.bfloat16).float()
    return current.numpy(), output.numpy()


@pytest.mark.parametrize(
    "num_heads,group_count,state_dim,seed",
    ((6, 2, 16, 7), (48, 16, 128, 11)),
)
def test_qwen36_bf16_decode_matches_pytorch_grouped_heads(
    num_heads: int,
    group_count: int,
    state_dim: int,
    seed: int,
) -> None:
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    q = _bf16_values((0.05 * rng.standard_normal((group_count, state_dim))).astype(np.float32))
    k = _bf16_values((0.05 * rng.standard_normal((group_count, state_dim))).astype(np.float32))
    v = _bf16_values((0.05 * rng.standard_normal((num_heads, state_dim))).astype(np.float32))
    g = _bf16_values((-0.1 + 0.02 * rng.standard_normal(num_heads)).astype(np.float32))
    beta = _bf16_values((0.1 * rng.standard_normal(num_heads)).astype(np.float32))
    state = (0.02 * rng.standard_normal((num_heads, state_dim, state_dim))).astype(np.float32)

    ck_state = np.empty_like(state)
    ck_output = np.empty_like(v)
    _load_provider()(
        _pointer(q), _pointer(k), _pointer(v), _pointer(g), _pointer(beta),
        _pointer(state), _pointer(ck_state), _pointer(ck_output),
        num_heads, group_count, state_dim, ctypes.c_float(1e-6),
    )
    reference_state, reference_output = _pytorch_reference(q, k, v, g, beta, state)

    if state_dim == 128:
        # Production Qwen recurrent geometry must preserve the persistent FP32
        # state exactly; a one-ULP mismatch can accumulate across decode tokens.
        np.testing.assert_array_equal(ck_state.view(np.uint32), reference_state.view(np.uint32))
        np.testing.assert_array_equal(ck_output.view(np.uint32), reference_output.view(np.uint32))
    else:
        np.testing.assert_allclose(ck_state, reference_state, atol=1e-7, rtol=0.0)
        np.testing.assert_allclose(ck_output, reference_output, atol=5e-7, rtol=0.0)
        assert np.count_nonzero(ck_output == reference_output) / ck_output.size >= 0.999
