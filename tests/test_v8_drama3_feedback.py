"""Exercise generated speech-frame feedback on nonzero BF16 operands."""
import numpy as np
import pytest

from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph
from version.v8.tts.build_drama3_feedback_circuit import build_circuit


def test_generated_frame_feedback_matches_pytorch(tmp_path):
    torch = pytest.importorskip("torch")
    width, vocabulary = 17, 5
    graph, config = build_circuit(width, vocabulary)
    run, arena, view, _ = compile_checked_weight_graph(
        tmp_path, graph, {}, {}, source_config=config,
    )
    generator = torch.Generator().manual_seed(2048)
    table = torch.randn((10 * vocabulary, width), generator=generator).bfloat16()
    text = torch.randn((1, width), generator=generator).bfloat16()
    indices = torch.arange(1, 10) * vocabulary + torch.arange(9) % vocabulary
    view("codebook_embeddings", np.uint16)[:] = table.view(torch.uint16).numpy()
    view("text_embedding", np.uint16)[:] = text.view(torch.uint16).numpy()
    for port, index in enumerate(indices.tolist(), 1):
        view(f"row{port}", np.int32)[0] = index
    assert run(arena, len(arena)) == 0

    expected_backbone = ((text + table[indices[0]]) / (2 ** 0.5)).float()
    expected_tower = torch.nn.functional.embedding_bag(indices[None, 1:], table, mode="sum")
    np.testing.assert_array_equal(view("backbone_embedding"), expected_backbone.numpy())
    np.testing.assert_array_equal(view("tower_enrichment", np.uint16), expected_tower.view(torch.uint16).numpy())

    saved = view("backbone_embedding").copy()
    view("row1", np.int32)[0] = 10 * vocabulary
    assert run(arena, len(arena)) != 0
    np.testing.assert_array_equal(view("backbone_embedding"), saved)
    view("row1", np.int32)[0] = indices[0].item()
    assert run(arena, len(arena)) == 0
    np.testing.assert_array_equal(view("backbone_embedding"), saved)
