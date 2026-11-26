"""Generated compact BF16 routing, per-expert execution and ordered accumulation."""
import ctypes as C
from pathlib import Path

import numpy as np
import pytest

from version.v8.decision.build_groundhog_moe import Experts, build_moe_replay
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph


@pytest.mark.parametrize("blas", [False, True])
def test_connected_moe_generated_against_torch(tmp_path, monkeypatch, blas):
    torch = pytest.importorskip("torch")
    if blas:
        path = Path(torch.__file__).parent / "lib/libtorch_cpu.so"
        if not path.is_file() or not hasattr(C.CDLL(str(path)), "dgemm_"):
            pytest.skip("LP64 BLAS unavailable")
        monkeypatch.setenv("CK_BLAS_LIBRARY", str(path))

    rows, width = 5, 16
    geometry = Experts(8, 3, 12)
    gen = torch.Generator().manual_seed(917)
    def random(shape):
        return (0.4 * torch.randn(shape, generator=gen)).bfloat16()
    data = random((rows, width))
    p = "model.language_model.layers.0."
    weights = {
        f"unit.{width}": torch.ones(width, dtype=torch.bfloat16),
        p + "router.scale": random((width,)) + 1,
        p + "router.proj.weight": random((geometry.count, width)),
        p + "router.per_expert_scale": (random((geometry.count,)) + 1).float(),
        p + "pre_feedforward_layernorm_2.weight": random((width,)) + 1,
        p + "experts.gate_up_proj": random((geometry.count, 2 * geometry.inner, width)),
        p + "experts.down_proj": random((geometry.count, width, geometry.inner)),
    }

    def numpy(tensor):
        return tensor.contiguous().view(torch.uint16).numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()

    graph = build_moe_replay(rows, width, geometry, 1e-6, blas=blas)
    invoke, arena, view, _ = compile_checked_weight_graph(
        tmp_path, graph.graph, {k: numpy(v) for k, v in weights.items()}, graph.weights,
        source_config=graph.config,
    )
    input_buffer = view("input", np.uint16)
    def output(name):
        return torch.from_numpy(view("l0_" + name, np.uint16).copy()).view(torch.bfloat16)

    for inp in (data, data.flip(0), data):
        input_buffer[:] = numpy(inp)
        assert invoke(arena, len(arena)) == 0
        x = inp.float()
        normed = x * (x.square().mean(dim=-1, keepdim=True) + 1e-6).pow(-0.5)
        router = normed.bfloat16() * weights[p + "router.scale"] * (width ** -0.5)
        logits = (router.double() @ weights[p + "router.proj.weight"].double().T).float().bfloat16()
        torch.testing.assert_close(output("router_logits"), logits, atol=0, rtol=0)
        probs, ids = torch.topk(logits.float().softmax(-1), geometry.top_k, dim=-1)
        scaled = probs / probs.sum(-1, keepdim=True) * weights[p + "router.per_expert_scale"][ids]
        np.testing.assert_array_equal(view("l0_expert_ids", np.int32), ids)
        np.testing.assert_allclose(view("l0_expert_weights"), scaled, atol=3e-7, rtol=0)
        gate_in = (normed * weights[p + "pre_feedforward_layernorm_2.weight"].float()).bfloat16()
        expected_routes = torch.empty((rows, geometry.top_k, width), dtype=torch.bfloat16)
        expected_sum = torch.zeros_like(inp)
        rank_sum = torch.zeros_like(inp)
        for expert in range(geometry.count):
            row, slot = (ids == expert).nonzero(as_tuple=True)
            if not len(row):
                continue
            gate, up = (gate_in[row].double() @ weights[p + "experts.gate_up_proj"][expert].double().T
                        ).float().bfloat16().chunk(2, dim=-1)
            active = torch.nn.functional.gelu(gate, approximate="tanh") * up
            down = (active.double() @ weights[p + "experts.down_proj"][expert].double().T).float().bfloat16()
            expected_routes[row, slot] = down
            expected_sum[row] += (down.float() * scaled[row, slot, None]).bfloat16()
        for slot in range(geometry.top_k):
            rank_sum += (expected_routes[:, slot].float() * scaled[:, slot, None]).bfloat16()
        assert not torch.equal(rank_sum, expected_sum)
        torch.testing.assert_close(output("expert_routes"), expected_routes, atol=0, rtol=0)
        torch.testing.assert_close(output("expert_sum"), expected_sum, atol=0, rtol=0)
    before = output("expert_sum")
    input_buffer[0, 0] = 0x7F80
    assert invoke(arena, len(arena)) == -3
    torch.testing.assert_close(output("expert_sum"), before, atol=0, rtol=0)
