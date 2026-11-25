"""Compile a checked GQA entry and compare explicit key visibility."""
import ctypes as C
import json
from pathlib import Path

import numpy as np
import pytest

from version.v8.scripts.compile_checked_bundle_v8 import compile_native_graph
from version.v8.decision.packed_question_layout import Question, pack_questions


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    return build_runtime(tmp_path_factory.mktemp("attention_ranges"), 4, 8)


def build_runtime(folder, queries, memory):
    buffers = {
        "query": {"shape": [queries, 12]},
        "key": {"shape": [memory, 6]},
        "value": {"shape": [memory, 6]},
        "ranges": {"shape": [queries, 4]},
        "output": {"shape": [queries, 12]},
    }
    op = {
        "id": "attend", "op": "attention_ranges_checked",
        "kernel": "attention_ranges_checked_f32", "returns_status": True,
        "graph_slots": {
            "inputs": {name: f"external:{name}" for name in ("query", "key", "value", "ranges")},
            "outputs": {"output": "output"},
        },
        "params": {
            "Q": queries, "S": memory, "H": 4, "G": 2, "D": 3,
            "X": queries * 12 + memory, "scale": 0.5,
            "call_constants": {
                "query_elements": queries * 12, "key_elements": memory * 6, "value_elements": memory * 6,
                "range_elements": queries * 4, "output_elements": queries * 12,
                "scratch_elements": queries * 12 + memory,
                "query_tokens": queries, "memory_tokens": memory, "heads": 4, "kv_heads": 2,
                "head_dim": 3,
            },
        },
    }
    graph = {
        "version": 3, "name": "bounded_attention_ranges", "family": "bounded_graph",
        "contract": {"runtime_invariants": {"inference_only": True}},
        "checked_native_entry": True, "activation_buffers": buffers,
        "activation_bindings": {name: name for name in buffers},
        "native_entry": {
            "function": "ck_test_attention_ranges",
            "params": [{"c_type": "uint8_t *", "name": "arena"},
                       {"c_type": "size_t", "name": "arena_bytes"}],
            "arena": {"pointer": "arena", "bytes": "arena_bytes"},
        },
        "sequence": ["block"],
        "block_types": {"block": {
            "sequence": ["header", "body", "footer"], "header": [],
            "body": {"type": "dense", "ops": [op]}, "footer": [],
        }},
    }
    source = {
        "config": {"num_layers": 1, "embed_dim": 12, "num_heads": 4, "num_kv_heads": 2,
                   "head_dim": 3, "intermediate_size": 24, "context_length": memory,
                   "max_seq_len": memory, "activation_buffer_dtypes": {"ranges": "i32"}},
        "entries": [], "quant_summary": {}, "template": graph,
    }
    path = folder / "circuit.json"
    path.write_text(json.dumps(graph))
    layout, calls, _, lib, invoke = compile_native_graph(folder, source, path)
    assert [row["function"] for row in calls["operations"]] == ["ck_attention_ranges_checked_f32"]
    offsets = {row["name"]: row["abs_offset"] for row in layout["memory"]["activations"]["buffers"]}
    return buffers, offsets, layout["memory"]["arena"]["total_size"], lib, invoke


def put_request(runtime):
    buffers, offsets, size, _lib, invoke = runtime
    storage = (C.c_uint8 * (size + 63))()
    arena = (C.c_uint8 * size).from_buffer(storage, (-C.addressof(storage)) & 63)
    arrays = {
        name: np.ndarray(spec["shape"], dtype="i4" if name == "ranges" else "f4",
                         offset=offsets[name], buffer=arena)
        for name, spec in buffers.items()
    }
    rng = np.random.default_rng(611)
    for name in ("query", "key", "value"):
        arrays[name][:] = rng.normal(size=arrays[name].shape)
    return arena, arrays, invoke


def dense_oracle(arrays):
    q = arrays["query"].reshape(4, 4, 3).astype("f8").transpose(1, 0, 2)
    k = np.repeat(arrays["key"].reshape(8, 2, 3).astype("f8").transpose(1, 0, 2), 2, axis=0)
    v = np.repeat(arrays["value"].reshape(8, 2, 3).astype("f8").transpose(1, 0, 2), 2, axis=0)
    mask = np.zeros((4, 8), dtype=bool)
    for row, (a, b, c, d) in enumerate(arrays["ranges"]):
        mask[row, a:b] = True
        mask[row, c:d] = True
    scores = np.where(mask, (q @ k.transpose(0, 2, 1)) * 0.5, -np.inf)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    output = (probs @ v).transpose(1, 0, 2).reshape(4, 12)
    return output, (q, k, v, mask)


def test_generated_independent_question_ranges_and_invalid_input(runtime):
    arena, a, invoke = put_request(runtime)
    a["ranges"][:] = [[0, 2, 2, 4], [0, 2, 2, 4], [0, 2, 5, 8], [0, 2, 5, 8]]
    assert invoke(arena, len(arena)) == 0
    expected, _ = dense_oracle(a)
    np.testing.assert_allclose(a["output"], expected, rtol=1e-6, atol=1e-6)

    before = a["output"].copy()
    a["value"][2:4] += 23
    assert invoke(arena, len(arena)) == 0
    np.testing.assert_array_equal(a["output"][2:], before[2:])
    assert not np.allclose(a["output"][:2], before[:2])

    a["output"].fill(-91)
    a["ranges"][-1] = [0, 4, 3, 8]
    assert invoke(arena, len(arena)) != 0
    np.testing.assert_array_equal(a["output"], -91)
    a["ranges"][-1] = [0, 2, 5, 8]
    assert invoke(arena, len(arena)) == 0


def test_generated_clean_prefix_and_active_block_against_pytorch(runtime):
    torch = pytest.importorskip("torch")
    arena, a, invoke = put_request(runtime)
    a["ranges"][:] = [0, 2, 4, 8]
    assert invoke(arena, len(arena)) == 0
    _, (q, k, v, mask) = dense_oracle(a)
    t = torch.nn.functional.scaled_dot_product_attention(
        torch.tensor(q), torch.tensor(k), torch.tensor(v),
        attn_mask=torch.tensor(mask), scale=0.5,
    ).transpose(0, 1).reshape(4, 12)
    np.testing.assert_allclose(a["output"], t.numpy(), rtol=1e-6, atol=1e-6)


def test_publisher_token_blocks_against_separate_branch_oracle(tmp_path):
    torch = pytest.importorskip("torch")
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/decision/block_packed_bison_request.json").read_text()
    )
    prefix = tuple(fixture["prefix"])
    questions = tuple(
        Question(row["name"], tuple(row["tokens"]), tuple(map(tuple, row["options"])))
        for row in fixture["questions"]
    )
    prefix_len = len(prefix)
    total = prefix_len + sum(len(q.tokens) for q in questions)
    # Use a smaller window to exercise expiry within this short request.
    window = 24
    packed = pack_questions(prefix, questions, vocab_size=262144, sliding_window=window, capacity=total)
    assert packed.position_ids == tuple(range(prefix_len)) + tuple(
        position for q in questions for position in range(prefix_len, prefix_len + len(q.tokens))
    )
    assert packed.answers == (
        ("refund", prefix_len + len(questions[0].tokens) - 1, (("yes", 236776), ("no", 236799))),
        ("team", total - 1, (("billing", 236776), ("technical", 236799), ("fraud", 236780))),
    )
    with pytest.raises(ValueError, match="exceeds token capacity"):
        pack_questions(prefix, questions, vocab_size=262144, sliding_window=window, capacity=total - 1)

    arena, a, invoke = put_request(build_runtime(tmp_path, total, total))
    q = torch.from_numpy(a["query"]).reshape(total, 4, 3).transpose(0, 1)
    k = torch.from_numpy(a["key"]).reshape(total, 2, 3).transpose(0, 1).repeat_interleave(2, dim=0)
    v = torch.from_numpy(a["value"]).reshape(total, 2, 3).transpose(0, 1).repeat_interleave(2, dim=0)

    def reference(start, end, sliding):
        memory = list(range(prefix_len)) + list(range(start, end)) if start else list(range(end))
        local_q = torch.arange(len(memory) - (end - start), len(memory))[:, None]
        local_k = torch.arange(len(memory))[None, :]
        mask = (local_k <= local_q) & (local_k > local_q - window) if sliding else None
        out = torch.nn.functional.scaled_dot_product_attention(
            q[:, start:end], k[:, memory], v[:, memory], attn_mask=mask, scale=0.5,
        )
        return out.transpose(0, 1).reshape(end - start, 12).numpy()

    for sliding, ranges in [(False, packed.full_ranges), (True, packed.sliding_ranges)]:
        expected = [reference(0, prefix_len, sliding)]
        start = prefix_len
        for question in questions:
            end = start + len(question.tokens)
            expected.append(reference(start, end, sliding))
            start = end
        a["ranges"][:] = ranges
        assert invoke(arena, len(arena)) == 0
        np.testing.assert_allclose(a["output"], np.concatenate(expected), rtol=1e-6, atol=1e-6)

    a["ranges"][:] = packed.full_ranges
    row = packed.answers[1][1]
    a["ranges"][row] = [0, total, total, total]
    assert invoke(arena, len(arena)) == 0
    assert not np.allclose(a["output"][row], reference(row - len(questions[1].tokens) + 1, total, False)[-1])
