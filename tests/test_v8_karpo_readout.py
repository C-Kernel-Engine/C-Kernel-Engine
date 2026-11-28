"""Run the ordinary generated Karpo marker projection and question softmax."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from version.v8.decision.build_karpo_readout import build_circuit
from version.v8.decision.capture_karpo_readout_reference import (
    ARTIFACT_SHA256, CHECKPOINT_REVISION, PUBLISHER_REVISION, digest,
)
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph


def test_generated_readout_with_unequal_choice_counts(tmp_path):
    torch = pytest.importorskip("torch")
    graph, config, declarations = build_circuit(3, width=9, labels=5)
    rng = np.random.default_rng(8324)
    weights = rng.standard_normal((5, 9), dtype=np.float32)
    hidden = (rng.standard_normal((3, 9), dtype=np.float32) * 3).astype(np.float32)
    run, arena, view, _ = compile_checked_weight_graph(
        tmp_path, graph, {"readout.weight": weights}, declarations, source_config=config,
    )
    view("hidden")[:] = hidden
    counts = view("choice_counts", np.int32)
    counts[:] = [2, 5, 1]
    assert run(arena, len(arena)) == 0
    reference = torch.from_numpy(hidden) @ torch.from_numpy(weights).T
    np.testing.assert_allclose(view("scores"), reference.numpy(), atol=4e-6, rtol=0)
    expected = np.zeros((3, 5), dtype=np.float32)
    for row, count in enumerate(counts):
        expected[row, :count] = reference[row, :count].softmax(0).numpy()
    np.testing.assert_allclose(view("probabilities"), expected, atol=7e-7, rtol=0)
    np.testing.assert_array_equal(view("probabilities")[2], [1, 0, 0, 0, 0])

    saved = view("probabilities").copy()
    counts[2] = 6
    assert run(arena, len(arena)) != 0
    np.testing.assert_array_equal(view("probabilities"), saved)
    view("hidden")[:] = hidden[::-1]
    counts[:] = [1, 5, 2]
    assert run(arena, len(arena)) == 0
    np.testing.assert_array_equal(view("probabilities"), saved[::-1])

def test_released_readout_on_publisher_markers(tmp_path):
    checkpoint = os.environ.get("CK_KARPO_READOUT_CHECKPOINT")
    reference = os.environ.get("CK_KARPO_READOUT_REFERENCE")
    if checkpoint is None and reference is None:
        pytest.skip("set CK_KARPO_READOUT_CHECKPOINT and CK_KARPO_READOUT_REFERENCE for released assets")
    assert checkpoint and reference, "released acceptance needs both asset paths"
    from safetensors.numpy import load_file

    weights_path = Path(checkpoint) / "readout.safetensors"
    packet = Path(reference) / "markers.npz"
    report = json.loads((Path(reference) / "report.json").read_text())
    assert digest(weights_path) == ARTIFACT_SHA256[weights_path.name]
    assert report["checkpoint_revision"] == CHECKPOINT_REVISION
    assert report["publisher_revision"] == PUBLISHER_REVISION
    assert digest(packet) == report["markers_sha256"]
    assert report["question_ids"] == ["eligible", "action"]
    weights = load_file(weights_path)["weight"]
    graph, config, declarations = build_circuit(len(report["question_ids"]))
    run, arena, view, _ = compile_checked_weight_graph(
        tmp_path, graph, {"readout.weight": weights}, declarations, source_config=config,
    )
    with np.load(packet, allow_pickle=False) as captured:
        counts = captured["counts"]
        np.testing.assert_array_equal(counts, [2, 3])
        view("choice_counts", np.int32)[:] = counts
        view("hidden")[:] = captured["hidden"]
        assert run(arena, len(arena)) == 0
        scores = view("scores")
        probabilities = view("probabilities")
        expected_scores = captured["scores"]
        expected_probabilities = captured["probabilities"]
        print(f"max score error: {np.max(np.abs(scores - expected_scores)):.8g}")
        print(f"max probability error: {np.max(np.abs(probabilities - expected_probabilities)):.8g}")
        np.testing.assert_allclose(scores, expected_scores, atol=3e-5, rtol=0)
        np.testing.assert_allclose(probabilities, expected_probabilities, atol=1e-6, rtol=0)
