"""Independent CPU PyTorch replay of the chunked gated DeltaNet provider."""
import ctypes as C
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def native_chunk():
    torch = pytest.importorskip("torch")
    if torch.backends.cpu.get_cpu_capability() not in {"AVX2", "AVX512"}:
        pytest.skip("the chunk-64 oracle requires an x86 PyTorch build with MKL")
    root = Path(__file__).resolve().parents[1]
    library_path = Path(os.environ.get("CK_ENGINE_LIB", root / "build/libckernel_engine.so"))
    if not library_path.is_file():
        pytest.skip("the native engine is not built")
    cpu_math = Path(torch.__file__).resolve().parent / "lib/libtorch_cpu.so"
    math = C.CDLL(str(cpu_math))
    if not hasattr(math, "sgemm_") or not hasattr(math, "vsExp"):
        pytest.skip("the installed PyTorch CPU library does not expose the MKL symbols")
    os.environ.setdefault("CK_MKL_LIBRARY", str(cpu_math))
    library = C.CDLL(str(library_path))
    kernel = library.gated_deltanet_chunked_bf16_prefill_forward
    array = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    kernel.argtypes = [array] * 10 + [C.c_int] * 4 + [C.c_float]
    kernel.restype = None
    return kernel


def _bf16_l2_norm(tensor):
    return tensor * (tensor.square().sum(dim=-1, keepdim=True) + 1e-6).rsqrt()


def _chunk64_pytorch_oracle(q, k, value, decay, beta, initial):
    # Math contract from the Transformers 5.13.0 Qwen3.5 CPU fallback.
    # Later releases normalize Q/K in FP32 and use a triangular solver.
    import torch
    from torch.nn.functional import pad

    chunk = 64
    rows, width = q.shape[1], q.shape[-1]
    padding = (-rows) % chunk
    q, k, value = [
        pad(x.transpose(1, 2).contiguous().float(), (0, 0, 0, padding))
        .unflatten(-2, (-1, chunk))
        for x in (q, k, value)
    ]
    q = q * (width ** -0.5)
    decay, beta = [
        pad(x.transpose(1, 2).contiguous().float(), (0, padding))
        .unflatten(-1, (-1, chunk))
        for x in (decay, beta)
    ]
    cumulative = decay.cumsum(dim=-1)
    pairwise_decay = (cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)).tril().exp().tril()
    key_beta = k * beta.unsqueeze(-1)
    transform = -(key_beta @ k.transpose(-1, -2) * pairwise_decay).tril(-1)
    for row in range(1, chunk):
        coeff = transform[..., row, :row].clone()
        transform[..., row, :row] = coeff + (coeff.unsqueeze(-1) * transform[..., :row, :row]).sum(-2)
    transform = transform + torch.eye(chunk, dtype=q.dtype)
    new_values = transform @ (value * beta.unsqueeze(-1))
    cumulative_keys = transform @ (key_beta * cumulative.exp().unsqueeze(-1))

    state = initial
    output = []
    for block in range(q.shape[-3]):
        query, key = q[:, :, block], k[:, :, block]
        step = cumulative[:, :, block]
        residual = new_values[:, :, block] - cumulative_keys[:, :, block] @ state
        inter = (query * step.exp().unsqueeze(-1)) @ state
        intra = (query @ key.transpose(-1, -2)) * pairwise_decay[:, :, block]
        output.append(inter + intra @ residual)
        final_key = key * (step[..., -1:] - step).exp().unsqueeze(-1)
        state = state * step[..., -1].exp()[..., None, None] + final_key.transpose(-1, -2) @ residual
    return torch.cat(output, dim=-2)[:, :, :rows].transpose(1, 2).bfloat16(), state


@pytest.mark.parametrize(("rows", "groups", "heads"), (
    (1, 2, 6),
    (17, 16, 48),
    (63, 2, 6),
    (64, 2, 6),
    (65, 2, 6),
    (129, 2, 6),
))
def test_chunked_prefill_matches_pytorch_state_and_output(native_chunk, rows, groups, heads):
    import torch

    torch.set_num_threads(1)
    width = 128
    repeat = heads // groups
    generator = torch.Generator().manual_seed(667 + rows)
    q, k, v = [
        (0.2 * torch.randn((1, rows, count, width), generator=generator)).bfloat16()
        for count in (groups, groups, heads)
    ]
    g = -torch.rand((1, rows, heads), generator=generator) * 0.2
    beta = (0.45 * torch.randn((1, rows, heads), generator=generator)).bfloat16()
    packed = torch.cat([a.reshape(rows, -1) for a in (q, k, v)], dim=-1)
    expanded_q = q.repeat_interleave(repeat, dim=2)
    expanded_k = k.repeat_interleave(repeat, dim=2)
    normalized_q = _bf16_l2_norm(expanded_q)
    normalized_k = _bf16_l2_norm(expanded_k)
    compact_q = normalized_q[:, :, ::repeat]
    compact_k = normalized_k[:, :, ::repeat]
    initial = torch.randn((1, heads, width, width), generator=generator) * 0.01
    expected, expected_state = _chunk64_pytorch_oracle(
        normalized_q, normalized_k, v, g, beta.sigmoid(), initial,
    )

    def array(tensor):
        return tensor.float().contiguous().numpy()

    operands = tuple(map(array, (compact_q, compact_k, v, packed, g, beta)))
    workspace = np.empty(558080 // 4, dtype=np.float32)
    for in_place in (False, True):
        state_input = array(initial).copy()
        state = state_input if in_place else np.empty_like(state_input)
        output = np.empty_like(array(v))
        native_chunk(
            *operands, state_input, state, output, workspace,
            rows, heads, groups, width, 1e-6,
        )
        if not in_place:
            np.testing.assert_array_equal(state_input, array(initial))
        np.testing.assert_array_equal(output, array(expected))
        np.testing.assert_array_equal(state, array(expected_state))
