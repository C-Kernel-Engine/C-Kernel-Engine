"""Native Bison token embedding, single-round RMSNorm and Q/K/V projection."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from version.v8.decision.build_bison_backbone_ingress import build_circuit
from version.v8.decision.capture_bison_block_oracle import CHECKPOINT_FILES, PREFIX, REVISION
from version.v8.decision.packed_question_layout import Question, pack_questions
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph


def _native(folder, torch, token_rows, embedding_table, weights, epsilon, *, target_isa="portable"):
    def bits(tensor):
        return tensor.contiguous().view(torch.uint16).numpy()

    table_rows, width = embedding_table.shape
    graph, config = build_circuit(
        len(token_rows), table_rows=table_rows, width=width,
        q_width=weights["q_proj.weight"].shape[0], kv_width=weights["k_proj.weight"].shape[0],
        epsilon=epsilon,
    )
    declarations = {name: (list(tensor.shape), "BF16") for name, tensor in weights.items()}
    invoke, arena, view, _ = compile_checked_weight_graph(
        folder, graph, {name: bits(tensor) for name, tensor in weights.items()}, declarations,
        source_config=config, target_isa=target_isa,
    )
    view("embedding_table", np.uint16)[:] = bits(embedding_table)
    indices = view("token_rows", np.int32)
    indices[:] = token_rows
    outputs = {key: torch.from_numpy(view(key, np.uint16)).view(torch.bfloat16)
               for key in ("input", "normalized", "q", "k", "v")}

    def run():
        return invoke(arena, len(arena))

    return run, indices, outputs


def test_generated_ingress_one_round_and_invalid_token_recovery(tmp_path):
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(64)
    table = torch.randn(5, 8, generator=generator).bfloat16()
    table[4] = torch.tensor([1.875, -0.625, -2.0, -0.875, 1.25, -1.75, 1.0, 1.625])
    gamma = torch.tensor([4.6875, 1.9140625, 4.34375, 1.4453125, 3.046875,
                          3.59375, 0.6484375, 3.875], dtype=torch.bfloat16)
    weights = {"input_layernorm.weight": gamma}
    for key, width in (("q", 8), ("k", 4), ("v", 4)):
        weights[f"{key}_proj.weight"] = torch.randn(width, 8, generator=generator).bfloat16()
    ids = [4, 2, 4, 0]
    epsilon = 1e-6
    run, indices, outputs = _native(tmp_path, torch, ids, table, weights, epsilon)
    assert run() == 0
    embedded = table[ids]
    unit = embedded.float() * (embedded.float().square().mean(-1, keepdim=True) + epsilon).pow(-0.5)
    normalized = (unit * gamma.float()).bfloat16()
    # This value crosses a BF16 midpoint when the inverse uses scalar powf.
    assert normalized[0, 3].item() == -0.8671875
    assert not torch.equal(normalized, unit.bfloat16() * gamma)
    torch.testing.assert_close(outputs["input"], embedded, atol=0, rtol=0)
    torch.testing.assert_close(outputs["normalized"], normalized, atol=0, rtol=0)
    for key in ("q", "k", "v"):
        reference = (normalized.double() @ weights[f"{key}_proj.weight"].double().T).float().bfloat16()
        torch.testing.assert_close(outputs[key], reference, atol=0, rtol=0)
    before = {name: tensor.clone() for name, tensor in outputs.items()}
    for invalid in (-1, len(table)):
        indices[-1] = invalid
        assert run() == -2
        for name, tensor in outputs.items():
            torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)
    indices[-1] = ids[-1]
    assert run() == 0
    for name, tensor in outputs.items():
        torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)


def test_pinned_bison_first_projection(tmp_path):
    configured = os.getenv("CK_BISON_BF16_CHECKPOINT")
    if not configured:
        pytest.skip("CK_BISON_BF16_CHECKPOINT is not configured")
    torch = pytest.importorskip("torch")
    from safetensors import safe_open
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    checkpoint = Path(configured)
    fixture = Path(__file__).parent / "fixtures/decision/block_packed_bison_request.json"
    request = json.loads(fixture.read_text())
    assert request["checkpoint"] == REVISION
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    text = json.loads((checkpoint / "config.json").read_text())["text_config"]
    assert text["layer_types"][0] == "sliding_attention"
    width, epsilon = text["hidden_size"], text["rms_norm_eps"]
    questions = tuple(Question(q["name"], tuple(q["tokens"]), tuple(map(tuple, q["options"])))
                      for q in request["questions"])
    layout = pack_questions(tuple(request["prefix"]), questions, vocab_size=text["vocab_size"],
                            sliding_window=text["sliding_window"], capacity=168)
    original, inverse = np.unique(layout.token_ids, return_inverse=True)
    names = {"input_layernorm.weight": PREFIX + "layers.0.input_layernorm.weight"}
    names.update({f"{key}_proj.weight": PREFIX + f"layers.0.self_attn.{key}_proj.weight"
                  for key in ("q", "k", "v")})
    embedding_name = PREFIX + "embed_tokens.weight"
    shards = {weight_map[name] for name in (embedding_name, *names.values())}
    for name in sorted(shards | {"config.json", "model.safetensors.index.json"}):
        with (checkpoint / name).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == CHECKPOINT_FILES[name], name

    def load(name, rows=None):
        with safe_open(checkpoint / weight_map[name], framework="pt", device="cpu") as source:
            if rows is None:
                return source.get_tensor(name)
            table = source.get_slice(name)
            return torch.stack([table[int(token)] for token in rows])

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        with torch.inference_mode():
            raw_rows = load(embedding_name, original)
            assert raw_rows.shape == (len(original), width)
            # Fold the published, BF16-rounded constant into the selected weight rows.
            embedding_table = raw_rows * torch.tensor(width**0.5, dtype=torch.bfloat16)
            weights = {key: load(name) for key, name in names.items()}
            run, _, outputs = _native(tmp_path, torch, inverse, embedding_table, weights, epsilon,
                                       target_isa="avx2")
            assert run() == 0
            hidden = raw_rows[inverse] * torch.tensor(width**0.5, dtype=torch.bfloat16)
            norm = Gemma4RMSNorm(width, eps=epsilon)
            norm.load_state_dict({"weight": weights["input_layernorm.weight"]}, assign=True)
            normalized = norm(hidden)
            torch.testing.assert_close(outputs["input"], hidden, atol=0, rtol=0)
            torch.testing.assert_close(outputs["normalized"], normalized, atol=0, rtol=0)
            print(f"embedding and RMSNorm: {normalized.numel()} BF16 values each, exact")
            for name in ("q", "k", "v"):
                weight = weights[f"{name}_proj.weight"]
                actual = outputs[name]
                # The checked provider declares FP64 accumulation and FP32-to-BF16 RNE.
                contract_reference = (normalized.double() @ weight.double().T).float().bfloat16()
                torch.testing.assert_close(actual, contract_reference, atol=0, rtol=0)
                # Record the stock model's FP32 accumulation separately. Cancellation
                # near zero can span multiple BF16 neighbors even for small absolute error.
                eager = torch.nn.functional.linear(normalized, weight)
                error = (actual.float() - eager.float()).abs()
                step = (torch.nextafter(eager, actual).float() - eager.float()).abs()
                outside = error > step
                outside_max = error[outside].max().item() if outside.any() else 0.0
                print(f"{name}: {actual.numel()} exact against FP64 contract; "
                      f"stock BF16 Linear {(actual == eager).sum().item()}/{actual.numel()} exact, "
                      f"max_abs={error.max():.8g}, rmse={error.square().mean().sqrt():.8g}, "
                      f"outside one ULP={outside.sum().item()}, outside max={outside_max:.8g}")
    finally:
        torch.set_num_threads(previous_threads)
