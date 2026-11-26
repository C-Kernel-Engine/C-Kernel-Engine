"""Compare connected packed decision execution to independent branch execution."""
import numpy as np
import pytest

from version.v8.decision.build_bison_backbone import Backbone, Layer
from version.v8.decision.build_groundhog_moe import Experts
from version.v8.decision.packed_question_layout import Question, pack_questions
from version.v8.scripts.checked_bundle_evidence_v8 import checked_target_flags
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph


@pytest.mark.parametrize(("blas", "target_isa", "moe"), [
    (False, "portable", False), (True, "portable", False), (False, "avx2", False),
    (False, "portable", True),
])
def test_generated_mixed_backbone_scores_and_recovers(tmp_path, monkeypatch, blas, target_isa, moe):
    try:
        checked_target_flags(target_isa)
    except ValueError as error:
        pytest.skip(str(error))
    torch = pytest.importorskip("torch")
    config_module = pytest.importorskip("transformers.models.gemma4.configuration_gemma4")
    reference = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
    if blas:
        import ctypes
        from pathlib import Path

        library = Path(torch.__file__).parent / "lib/libtorch_cpu.so"
        if not library.is_file() or not hasattr(ctypes.CDLL(str(library)), "dgemm_"):
            pytest.skip("the local CPU oracle has no LP64 dgemm_ symbol")
        monkeypatch.setenv("CK_BLAS_LIBRARY", str(library))

    torch.manual_seed(67)
    prefix = (0, 1, 2)
    questions = (Question("first", (4, 5), (("a", 0), ("b", 1))),
                 Question("second", (6, 7, 4), (("c", 2),)))
    request = pack_questions(prefix, questions, vocab_size=8, sliding_window=3, capacity=8)
    config = config_module.Gemma4TextConfig(
        hidden_size=16, intermediate_size=20, num_attention_heads=4, num_key_value_heads=2,
        head_dim=4, global_head_dim=8, num_global_key_value_heads=1, attention_k_eq_v=True,
        num_hidden_layers=2, layer_types=["sliding_attention", "full_attention"],
        num_kv_shared_layers=0, hidden_size_per_layer_input=0, sliding_window=3,
        rms_norm_eps=1e-6, enable_moe_block=moe, num_experts=8, top_k_experts=2, moe_intermediate_size=12,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0,
                               "partial_rotary_factor": 0.5},
        },
    )
    config._attn_implementation = "eager"
    layers = [reference.Gemma4TextDecoderLayer(config, i).eval().bfloat16() for i in range(2)]
    if moe:
        with torch.no_grad():
            for layer in layers:
                layer.experts.gate_up_proj.normal_(std=0.3)
                layer.experts.down_proj.normal_(std=0.3)
    rope = reference.Gemma4TextRotaryEmbedding(config)
    assert (rope.full_attention_inv_freq == 0).any()
    table = torch.randn((8, 16)).bfloat16()
    options = torch.randn((3, 16)).bfloat16()
    final = reference.Gemma4RMSNorm(16).bfloat16()
    graph = Backbone(8, 8, 16, 4, 20, 1e-6, blas=blas)
    geometry = Experts(8, 2, 12) if moe else None
    hidden = graph.layer(0, Layer(2, 4, "sliding"), "input", moe=geometry)
    hidden = graph.layer(1, Layer(1, 8, "full", True), hidden, moe=geometry)
    graph.readout(hidden, 2, 3)

    def bits(value):
        return value.detach().contiguous().view(torch.uint16).numpy()

    weights = {"unit.4": bits(torch.ones(4, dtype=torch.bfloat16)),
               "unit.8": bits(torch.ones(8, dtype=torch.bfloat16)),
               "rope.sliding.4": rope.sliding_attention_inv_freq.numpy(),
               "rope.full.8": rope.full_attention_inv_freq.numpy(),
               "model.language_model.norm.weight": bits(final.weight),
               "option.weight": bits(options)}
    for i, layer in enumerate(layers):
        for name, tensor in layer.state_dict().items():
            weights[f"model.language_model.layers.{i}.{name}"] = (
                tensor.detach().float().numpy() if name == "router.per_expert_scale" else bits(tensor)
            )
    if moe:
        weights["unit.16"] = bits(torch.ones(16, dtype=torch.bfloat16))
    invoke, arena, view, _ = compile_checked_weight_graph(
        tmp_path, graph.graph, weights, graph.weights, source_config=graph.config, target_isa=target_isa,
    )
    view("embedding_table", np.uint16)[:] = bits(table)
    view("token_rows", np.int32)[:] = request.token_ids
    view("positions", np.int32)[:] = request.position_ids
    for name in ("sliding_ranges", "full_ranges"):
        view(name, np.int32)[:] = getattr(request, name)
    view("answer_rows", np.int32)[:] = [row for _, row, _ in request.answers]
    assert invoke(arena, len(arena)) == 0
    actual = torch.from_numpy(view("scores", np.uint16).copy()).view(torch.bfloat16)
    expected = []
    with torch.inference_mode():
        for question in questions:
            tokens = prefix + question.tokens
            state = table[list(tokens)][None]
            positions = torch.arange(len(tokens))[None]
            row = positions[0, :, None]
            col = positions[0, None, :]
            for layer, kind in zip(layers, config.layer_types, strict=True):
                allowed = (col <= row) & (col > row - 3) if kind == "sliding_attention" else (
                    (row >= len(prefix)) | (col < len(prefix)))
                mask = torch.zeros(allowed.shape, dtype=torch.bfloat16).masked_fill(
                    ~allowed, torch.finfo(torch.bfloat16).min)[None, None]
                state = layer(state, shared_kv_states={}, position_embeddings=rope(state, positions, kind),
                              attention_mask=mask, position_ids=positions)
            expected.append(torch.nn.functional.linear(final(state[0, -1]), options))
    torch.testing.assert_close(actual, torch.stack(expected), atol=0.03125, rtol=0.015625)

    # Changing only the second question leaves the first question's scores exact.
    view("token_rows", np.int32)[-1] = 0
    assert invoke(arena, len(arena)) == 0
    changed = view("scores", np.uint16).copy()
    np.testing.assert_array_equal(changed[0], bits(actual)[0])
    assert not np.array_equal(changed[1], bits(actual)[1])
    view("token_rows", np.int32)[-1] = request.token_ids[-1]
    view("positions", np.int32)[-1] = -1
    assert invoke(arena, len(arena)) == -2
    np.testing.assert_array_equal(view("scores", np.uint16), changed)
    view("positions", np.int32)[-1] = request.position_ids[-1]
    assert invoke(arena, len(arena)) == 0
    np.testing.assert_array_equal(view("scores", np.uint16), bits(actual))

    view("full_ranges", np.int32)[-1] = [0, 8, 8, 8]
    assert invoke(arena, len(arena)) == 0
    np.testing.assert_array_equal(view("scores", np.uint16)[0], bits(actual)[0])
    assert not np.array_equal(view("scores", np.uint16)[1], bits(actual)[1])
    view("full_ranges", np.int32)[-1] = request.full_ranges[-1]
    view("answer_rows", np.int32)[0] = 0
    assert invoke(arena, len(arena)) == 0
    assert not np.array_equal(view("scores", np.uint16)[0], bits(actual)[0])
    np.testing.assert_array_equal(view("scores", np.uint16)[1], bits(actual)[1])
