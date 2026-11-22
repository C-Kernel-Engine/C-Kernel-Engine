"""Exercise ordered per-index scaling in a checked generated graph."""
import ctypes as C
from pathlib import Path
import subprocess

import numpy as np
import pytest

from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph


def router_graph(rows, experts, top_k):
    count = rows * top_k
    buffers = {"logits": [rows, experts], "indices": [rows, top_k],
               "weights": [rows, top_k], "scaled": [rows, top_k]}
    operations = [
        {
            "id": "route", "op": "full_softmax_topk_router",
            "kernel": "moe_softmax_topk_router_llama_f32", "returns_status": True,
            "graph_slots": {"inputs": {"logits": "external:logits"},
                            "outputs": {"indices": "indices", "weights": "weights"}},
            "params": {"R": rows, "E": experts, "K": top_k, "routed_scaling_factor": 1.0},
        },
        {
            "id": "scale", "op": "indexed_scale_checked",
            "kernel": "indexed_scale_checked_f32", "returns_status": True,
            "graph_slots": {"inputs": {"values": "weights", "ids": "indices"},
                            "outputs": {"output": "scaled"}},
            "weight_refs": {"scales": "expert_scales"},
            "params": {"N": count, "V": experts,
                       "call_constants": {"count": count, "scale_count": experts}},
        },
    ]
    graph = {
        "version": 3, "name": "checked_scaled_router", "family": "bounded_graph",
        "checked_native_entry": True,
        "activation_buffers": {name: {"shape": shape} for name, shape in buffers.items()},
        "activation_bindings": {name: name for name in buffers},
        "contract": {"runtime_invariants": {"inference_only": True}},
        "native_entry": {
            "function": "ck_test_scaled_router",
            "params": [{"c_type": "uint8_t *", "name": "arena"},
                       {"c_type": "size_t", "name": "arena_bytes"}],
            "arena": {"pointer": "arena", "bytes": "arena_bytes"},
        },
        "sequence": ["block"],
        "block_types": {"block": {"sequence": ["header", "body", "footer"],
                                  "header": [], "footer": [],
                                  "body": {"type": "dense", "ops": operations}}},
    }
    config = {"num_layers": 1, "embed_dim": 16, "num_heads": 1, "num_kv_heads": 1,
              "head_dim": 16, "intermediate_size": 16, "context_length": rows,
              "max_seq_len": rows, "num_experts": experts, "experts_per_tok": top_k,
              "activation_buffer_dtypes": {"indices": "i32"}}
    return graph, config


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    rows, experts, top_k = 3, 8, 2
    graph, config = router_graph(rows, experts, top_k)
    scales = np.linspace(0.5, 2.5, experts, dtype=np.float32)
    folder = tmp_path_factory.mktemp("checked_scaled_router")
    return scales, compile_checked_weight_graph(
        folder, graph, {"expert_scales": scales}, {"expert_scales": ([experts], "F32")},
        source_config=config,
    )


def test_generated_router_scales_selected_weights(runtime):
    torch = pytest.importorskip("torch")
    scales, (run, arena, view, _) = runtime
    rng = np.random.default_rng(213)
    first, second = rng.normal(size=(2, 3, len(scales))).astype(np.float32)
    for logits in (first, second, first):
        view("logits")[:] = logits
        assert run(arena, len(arena)) == 0
        probabilities = torch.from_numpy(logits).softmax(dim=-1)
        selected, ids = torch.topk(probabilities, k=2)
        expected = selected / selected.sum(dim=-1, keepdim=True) * torch.from_numpy(scales)[ids]
        np.testing.assert_array_equal(view("indices", np.int32), ids.numpy())
        np.testing.assert_allclose(view("scaled"), expected.numpy(), atol=2e-7, rtol=0)
        np.testing.assert_array_equal(view("scaled"), view("weights") * scales[view("indices", np.int32)])
        assert not np.allclose(view("scaled").sum(axis=-1), 1.0)


def test_checked_scale_preserves_output_on_rejection(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src/kernels/indexed_scale.c"
    shared = tmp_path / "indexed_scale.so"
    subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared",
                    "-fPIC", "-O2", str(source), "-o", str(shared)], check=True)
    library = C.CDLL(str(shared))
    fp = C.POINTER(C.c_float)
    ip = C.POINTER(C.c_int32)
    scale = library.ck_indexed_scale_f32
    scale.argtypes = [fp, ip, fp, fp, C.c_size_t, C.c_size_t]
    scale.restype = C.c_int
    values = (C.c_float * 3)(0.25, 0.75, 0.5)
    ids = (C.c_int32 * 3)(1, 0, 1)
    scales = (C.c_float * 2)(2, 4)
    output = (C.c_float * 4)(7, 7, 7, 7)
    for invalid_id in (-1, 2):
        ids[2] = invalid_id
        assert scale(values, ids, scales, output, 3, 2) != 0
        assert list(output) == [7] * 4
    ids[2] = 1
    for invalid_scale in (float("nan"), float("inf"), np.finfo(np.float32).max):
        scales[1] = invalid_scale
        values[2] = 4
        assert scale(values, ids, scales, output, 3, 2) != 0
        assert list(output) == [7] * 4
    scales[1] = 4
    values[2] = 0.5
    assert scale(values, ids, scales, values, 3, 2) != 0
    assert list(values) == [0.25, 0.75, 0.5]
    assert scale(values, ids, scales, output, 3, 2) == 0
    assert list(output) == [1, 1.5, 2, 7]


def test_generated_scale_rejects_undersized_scale_table(tmp_path):
    graph, config = router_graph(3, 8, 2)
    with pytest.raises(RuntimeError, match="capacity exceeds planned storage|weight shape"):
        compile_checked_weight_graph(
            tmp_path, graph, {"expert_scales": np.ones(4, dtype=np.float32)},
            {"expert_scales": ([4], "F32")}, source_config=config,
        )
