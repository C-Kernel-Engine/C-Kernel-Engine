"""Generated replay for BF16 CBLAS projection and partition top-k."""
import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph
from version.v8.scripts.replay_bf16_projection_bundle_v8 import projection_graph


def test_partition_router_matches_cpu_topk_cutoff(tmp_path):
    torch = pytest.importorskip("torch")
    rows, experts, topk = 20, 128, 8
    graph = projection_graph(rows, experts, topk)
    graph["name"] = "partition_topk_replay"
    graph["native_entry"]["function"] = "ck_partition_topk_replay"
    graph["activation_buffers"] = {"logits": {"shape": [rows, experts]},
                                   "ids": {"shape": [rows, topk]}, "probabilities": {"shape": [rows, topk]}}
    graph["activation_bindings"] = {name: name for name in graph["activation_buffers"]}
    graph["block_types"]["block"]["body"]["ops"] = [{
        "id": "routes", "op": "full_softmax_topk_router", "kernel": "moe_softmax_topk_router_partition_f32",
        "returns_status": True,
        "graph_slots": {"inputs": {"logits": "external:logits"}, "outputs": {"indices": "ids", "weights": "probabilities"}},
        "params": {"R": rows, "E": experts, "K": topk, "routed_scaling_factor": 1.0},
    }]
    config = {"embed_dim": experts, "num_heads": 1, "num_kv_heads": 1, "head_dim": experts,
              "num_experts": experts, "experts_per_tok": topk, "num_layers": 1,
              "intermediate_size": experts, "context_length": rows, "max_seq_len": rows,
              "activation_buffer_dtypes": {"logits": "fp32", "ids": "i32", "probabilities": "fp32"}}
    run, arena, view, _ = compile_checked_weight_graph(tmp_path, graph, {}, {}, source_config=config)
    rng = torch.Generator().manual_seed(1281)
    logits = torch.randint(-20, 7, (rows, experts), generator=rng).float() / 8
    logits[0].fill_(0)
    logits[1] = torch.arange(experts).float() / experts
    probs = torch.softmax(logits, dim=-1)
    expected, ranked = probs.topk(topk)
    expected = expected / expected.sum(-1, keepdim=True)
    input_buffer = view("logits", np.float32)
    ids, selected = view("ids", np.int32), view("probabilities", np.float32)
    input_buffer[:] = logits.numpy()
    assert run(arena, len(arena)) == 0
    np.testing.assert_array_equal(ids, ranked.numpy())
    np.testing.assert_allclose(selected, expected.numpy(), rtol=2e-6, atol=1e-7)
    assert not np.array_equal(ids[0], np.arange(topk)), "fixture must distinguish rank-only tie behavior"
    before_ids, before_probabilities = ids.copy(), selected.copy()
    input_buffer[-1, -1] = np.nan
    assert run(arena, len(arena)) != 0
    np.testing.assert_array_equal(ids, before_ids)
    np.testing.assert_array_equal(selected, before_probabilities)
    input_buffer[:] = logits.numpy()
    assert run(arena, len(arena)) == 0
    np.testing.assert_array_equal(ids, before_ids)
    np.testing.assert_array_equal(selected, before_probabilities)


def test_checked_compact_cblas_projection(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    path = os.environ.get("CK_BLAS_LIBRARY") or Path(torch.__file__).parent / "lib" / "libtorch_cpu.so"
    try:
        getattr(ctypes.CDLL(str(path)), "cblas_gemm_bf16bf16f32")
    except (OSError, AttributeError):
        pytest.skip("LP64 CBLAS BF16 GEMM provider is unavailable")
    monkeypatch.setenv("CK_BLAS_LIBRARY", str(path))
    rows, width, channels = 4, 13, 5
    graph = projection_graph(rows, width, channels)
    op = graph["block_types"]["block"]["body"]["ops"][0]
    op["kernel"] = "linear_rows_bf16_cblas_checked"
    op["params"]["call_constants"].pop("staged_elements")
    op["params"]["call_constants"]["work_elements"] = rows * channels
    rng = torch.Generator().manual_seed(451)
    x = torch.randn((rows, width), generator=rng).bfloat16()
    weight = torch.randn((channels, width), generator=rng).bfloat16()
    weights = {"projection.weight": weight.view(torch.uint16).numpy()}
    config = {"embed_dim": width, "num_heads": 1, "num_kv_heads": 1, "head_dim": width,
              "num_layers": 1, "intermediate_size": channels, "context_length": rows, "max_seq_len": rows,
              "activation_buffer_dtypes": {"input": "bf16", "output": "bf16"}}
    run, arena, view, _ = compile_checked_weight_graph(tmp_path, graph, weights,
        {"projection.weight": ([channels, width], "BF16")}, source_config=config)
    input_buffer, output = view("input", np.uint16), view("output", np.uint16)
    input_buffer[:] = x.view(torch.uint16).numpy()
    assert run(arena, len(arena)) == 0
    expected = torch.nn.functional.linear(x, weight).view(torch.uint16).numpy()
    np.testing.assert_array_equal(output, expected)
    input_buffer[-1, -1] = 0x7FC0
    assert run(arena, len(arena)) != 0
    np.testing.assert_array_equal(output, expected)
    input_buffer[:] = x.view(torch.uint16).numpy()
    assert run(arena, len(arena)) == 0
    np.testing.assert_array_equal(output, expected)
