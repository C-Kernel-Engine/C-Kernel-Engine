"""Independent PyTorch replay of the generated Phonon 31M Mimi ingress."""

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph
from version.v8.tts.build_phonon31m_codec_ingress import CODEC_WIDTH, LATENT_WIDTH, build_circuit


CHECKPOINT_SHA256 = "8a04cf833cfe69b67c2ec968b2d31244655b9accc84ade9c0ebcc176f2074607"


def check_ingress(torch, tmp_path, frames, std, mean, projection):
    graph, config, declarations = build_circuit(frames)
    run, arena, view, _ = compile_checked_weight_graph(tmp_path, graph, {
        "latent_std": std.numpy(), "mimi_output_projection.weight": projection[:, :, 0].numpy(),
    }, declarations, source_config=config)
    latents = torch.randn(frames, LATENT_WIDTH, generator=torch.Generator().manual_seed(131 + frames))
    view("latent_mean")[:] = mean.numpy()
    view("latents")[:] = latents.numpy()
    expected_latents = latents * std + mean
    expected = torch.nn.functional.conv1d(expected_latents.T.unsqueeze(0), projection)[0].T
    assert run(arena, len(arena)) == 0
    np.testing.assert_array_equal(view("denormalized"), expected_latents.numpy())
    error = np.max(np.abs(view("codec_features") - expected.numpy()))
    print(f"{frames} frames, maximum Mimi ingress error {error:.8g}")
    np.testing.assert_allclose(view("codec_features"), expected.numpy(), atol=3e-5, rtol=0)
    original = view("codec_features").copy()
    view("latents")[0, 0] = np.nan
    assert run(arena, len(arena)) != 0
    np.testing.assert_array_equal(view("codec_features"), original)
    view("latents")[:] = latents.numpy()
    assert run(arena, len(arena)) == 0
    np.testing.assert_array_equal(view("codec_features"), original)


@pytest.mark.parametrize("frames", [1, 3])
def test_generated_ingress_synthetic(tmp_path, frames):
    torch = pytest.importorskip("torch")
    rng = torch.Generator().manual_seed(972)
    std, mean = (torch.randn(LATENT_WIDTH, generator=rng) for _ in range(2))
    projection = torch.randn(CODEC_WIDTH, LATENT_WIDTH, 1, generator=rng)
    check_ingress(torch, tmp_path, frames, std, mean, projection)


def test_generated_ingress_with_released_weights(tmp_path):
    checkpoint = os.environ.get("CK_PHONON31M_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set CK_PHONON31M_CHECKPOINT to the pinned Phonon 31M release")
    torch = pytest.importorskip("torch")
    from safetensors import safe_open

    model = Path(checkpoint) / "model.safetensors"
    with model.open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == CHECKPOINT_SHA256
    with safe_open(model, framework="pt", device="cpu") as weights:
        std, mean = [weights.get_tensor("flow_lm.emb_" + suffix).float() for suffix in ("std", "mean")]
        projection = weights.get_tensor("mimi.quantizer.output_proj.weight")
        check_ingress(torch, tmp_path, 3, std, mean, projection)
