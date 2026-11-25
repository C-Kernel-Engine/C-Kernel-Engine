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


def build_runtime(folder, queries, memory, *, heads=4, kv_heads=2, head_dim=3, scale=0.5, bf16=False):
    q_width, kv_width = heads * head_dim, kv_heads * head_dim
    q_elements, kv_elements = queries * q_width, memory * kv_width
    scratch = q_elements + memory
    provider = "attention_ranges_checked_bf16" if bf16 else "attention_ranges_checked_f32"
    buffers = {
        "query": {"shape": [queries, q_width]},
        "key": {"shape": [memory, kv_width]},
        "value": {"shape": [memory, kv_width]},
        "ranges": {"shape": [queries, 4]},
        "output": {"shape": [queries, q_width]},
    }
    op = {
        "id": "attend", "op": "attention_ranges_checked",
        "kernel": provider, "returns_status": True,
        "graph_slots": {
            "inputs": {name: f"external:{name}" for name in ("query", "key", "value", "ranges")},
            "outputs": {"output": "output"},
        },
        "params": {
            "Q": queries, "S": memory, "H": heads, "G": kv_heads, "D": head_dim,
            "X": scratch, "scale": scale,
            "call_constants": {
                "query_elements": q_elements, "key_elements": kv_elements, "value_elements": kv_elements,
                "range_elements": queries * 4, "output_elements": q_elements, "scratch_elements": scratch,
                "query_tokens": queries, "memory_tokens": memory, "heads": heads, "kv_heads": kv_heads,
                "head_dim": head_dim,
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
    dtypes = {"ranges": "i32"}
    if bf16:
        dtypes.update(dict.fromkeys(("query", "key", "value", "output"), "bf16"))
    source = {
        "config": {"num_layers": 1, "embed_dim": q_width, "num_heads": heads, "num_kv_heads": kv_heads,
                   "head_dim": head_dim, "intermediate_size": q_width * 2, "context_length": memory,
                   "max_seq_len": memory, "activation_buffer_dtypes": dtypes},
        "entries": [], "quant_summary": {}, "template": graph,
    }
    path = folder / "circuit.json"
    path.write_text(json.dumps(graph))
    layout, calls, _, lib, invoke = compile_native_graph(folder, source, path)
    assert [row["function"] for row in calls["operations"]] == [f"ck_{provider}"]
    offsets = {row["name"]: row["abs_offset"] for row in layout["memory"]["activations"]["buffers"]}
    return buffers, offsets, layout["memory"]["arena"]["total_size"], lib, invoke


def put_request(runtime, *, bf16=False):
    buffers, offsets, size, _lib, invoke = runtime
    storage = (C.c_uint8 * (size + 63))()
    arena = (C.c_uint8 * size).from_buffer(storage, (-C.addressof(storage)) & 63)
    arrays = {
        name: np.ndarray(spec["shape"], dtype="i4" if name == "ranges" else "u2" if bf16 else "f4",
                         offset=offsets[name], buffer=arena)
        for name, spec in buffers.items()
    }
    if not bf16:
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


def test_generated_bf16_rounds_scores_and_probabilities_against_eager_pytorch(tmp_path):
    torch = pytest.importorskip("torch")
    folder = tmp_path / "bf16"
    folder.mkdir()
    arena, arrays, invoke = put_request(
        build_runtime(folder, 5, 8, head_dim=7, bf16=True), bf16=True,
    )
    generator = torch.Generator().manual_seed(37)
    inputs = {}
    for name in ("query", "key", "value"):
        inputs[name] = torch.randn(arrays[name].shape, generator=generator).bfloat16()
        arrays[name][:] = inputs[name].view(torch.uint16).numpy()
    arrays["ranges"][:] = [[0, 3, 3, 3], [0, 3, 3, 5], [0, 3, 3, 5], [0, 3, 5, 8], [0, 3, 5, 8]]
    q = inputs["query"].reshape(5, 4, 7).transpose(0, 1)
    k = inputs["key"].reshape(8, 2, 7).transpose(0, 1).repeat_interleave(2, dim=0)
    v = inputs["value"].reshape(8, 2, 7).transpose(0, 1).repeat_interleave(2, dim=0)
    mask = torch.zeros((5, 8), dtype=torch.bool)
    mask[0, :3] = True
    mask[1:3, :5] = True
    mask[3:, :3] = True
    mask[3:, 5:] = True
    scores = (q @ k.transpose(-1, -2) * 0.5).masked_fill(~mask, -torch.inf)
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).bfloat16()
    expected = (probs @ v).transpose(0, 1).reshape(5, 28)
    assert invoke(arena, len(arena)) == 0
    np.testing.assert_array_equal(arrays["output"], expected.view(torch.uint16).numpy())
    before = arrays["output"].copy()
    arrays["key"][0, 0] = 0x7F80
    assert invoke(arena, len(arena)) != 0
    np.testing.assert_array_equal(arrays["output"], before)
