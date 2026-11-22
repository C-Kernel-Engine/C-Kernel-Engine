"""Exercise the generated ten-codebook reconstruction prefix."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph
from version.v8.tts.build_drama3_codec_rvq_circuit import build_circuit


def compile_rvq(folder, frames, weights, **geometry):
    graph, config = build_circuit(frames, **geometry)
    declarations = {name: (list(value.shape), "F32") for name, value in weights.items()}
    return compile_checked_weight_graph(folder, graph, weights, declarations, source_config=config)


def invoke(runtime, codes):
    run, arena, view, _ = runtime
    for index, ids in enumerate(codes):
        view(f"ids{index}", np.int32)[:] = ids
    return run(arena, len(arena))


def test_generated_rvq_nonzero_order_rejection_and_recovery(tmp_path):
    rng = np.random.default_rng(389)
    vocabularies = [7] + [5] * 9
    frames, book_width, channels = 3, 4, 9
    weights = {}
    codes = np.empty((10, frames), dtype=np.int32)
    for index, vocabulary in enumerate(vocabularies):
        for kind, shape in (("table", (vocabulary, book_width)),
                            ("weight", (channels, book_width)), ("bias", (channels,))):
            weights[f"book{index}.{kind}"] = rng.normal(size=shape).astype(np.float32)
        codes[index] = rng.integers(vocabulary, size=frames)
    runtime = compile_rvq(tmp_path, frames, weights, vocabularies=vocabularies,
                          book_width=book_width, channels=channels)
    view = runtime[2]
    assert invoke(runtime, codes) == 0
    original = view("quantized").copy()

    # Negative control with a bad ID in the last codebook.
    invalid = codes.copy()
    invalid[-1, -1] = vocabularies[-1]
    assert invoke(runtime, invalid) != 0
    np.testing.assert_array_equal(view("quantized"), original)
    assert invoke(runtime, codes) == 0
    np.testing.assert_array_equal(view("quantized"), original)

    torch = pytest.importorskip("torch")
    contributions = []
    for index in range(10):
        prefix = f"book{index}"
        table, weight, bias = (torch.from_numpy(weights[f"{prefix}.{key}"])
                               for key in ("table", "weight", "bias"))
        terms = torch.nn.functional.linear(table[codes[index]], weight, bias)
        contributions.append(terms)
        np.testing.assert_allclose(view(f"projection{index}"), terms.numpy(), rtol=2e-6, atol=2e-6)
    expected = contributions[0] + sum(contributions[1:])
    np.testing.assert_allclose(original, expected.numpy(), rtol=2e-6, atol=5e-6)

    # Verify the order of addition independently of GEMM rounding.
    actual_terms = [view(f"projection{index}") for index in range(10)]
    np.testing.assert_array_equal(original, actual_terms[0] + sum(actual_terms[1:]))


@pytest.mark.parametrize("frames", [1, 3, 8])
def test_pinned_publisher_rvq(tmp_path, frames, record_property):
    location = os.environ.get("CK_DRAMA3_CODEC_REFERENCE_DIR")
    if location is None:
        pytest.skip("CK_DRAMA3_CODEC_REFERENCE_DIR is not configured")
    folder = Path(location)
    report = json.loads((folder / "reference-result.json").read_text())
    assert report["model_revision"] == "78e5ab79e49aa9a25abe8799eff2c4d73a1c3216"
    assert report["implementation"] == "dfb6b43b360b04e53daa7c36aed0bc448789d3a4"
    assert report["codec_sha256"] == "74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f"
    for name in ("rvq_weights.npz", f"reference-codes-{frames}.npz"):
        with (folder / name).open("rb") as file:
            assert hashlib.file_digest(file, "sha256").hexdigest() == report["replay_sha256"][name]
    with np.load(folder / "rvq_weights.npz") as archive:
        weights = dict(archive)
    with np.load(folder / f"reference-codes-{frames}.npz") as fixture:
        expected = fixture["quantizer_pre_transformer"][0].T
        runtime = compile_rvq(tmp_path, frames, weights)
        assert invoke(runtime, fixture["codes"][0]) == 0
        actual = runtime[2]("quantized")
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=4e-6)
        record_property("max_abs", float(np.abs(actual - expected).max()))
