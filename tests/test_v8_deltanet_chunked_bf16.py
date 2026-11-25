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
    module = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

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
    compact_q = module.l2norm(expanded_q)[:, :, ::repeat]
    compact_k = module.l2norm(expanded_k)[:, :, ::repeat]
    initial = torch.randn((1, heads, width, width), generator=generator) * 0.01
    expected, expected_state = module.torch_chunk_gated_delta_rule(
        expanded_q, expanded_k, v, g, beta.sigmoid(),
        initial_state=initial, output_final_state=True, use_qk_l2norm_in_kernel=True,
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
