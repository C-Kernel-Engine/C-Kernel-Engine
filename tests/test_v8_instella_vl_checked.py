"""Run Instella's checked graphs against independent nonzero arithmetic."""
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "version/v8/scripts"))
from instella_vl_checked_v8 import CompiledClipStage, clip_block, input_norm, patch_projection, projector


def normalization(x, weights, name):
    centered = x - x.mean(axis=-1, keepdims=True)
    return (centered / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + 1e-5)
            * weights[name + ".weight"] + weights[name + ".bias"])


def projection(x, weights, name):
    return x @ weights[name + ".weight"].T + weights.get(name + ".bias", 0)


class InstellaGeneratedTests(unittest.TestCase):
    def test_block_reloads_real_weight_slots_without_recompiling(self):
        rng = np.random.default_rng(81)
        graph = clip_block(5, 8, 2, 12)
        x = rng.normal(size=(5, 8)).astype(np.float32)

        def params():
            values = {key: (rng.normal(size=shape) * 0.2).astype(np.float32)
                      for key, (shape, _) in graph.weights.items()}
            values["quick_gelu_scale"].fill(1.702)
            return values

        def reference(weights):
            a = normalization(x.astype(np.float64), weights, "layer_norm1")
            q, k, v = [projection(a, weights, f"self_attn.{name}_proj").reshape(5, 2, 4)
                       .transpose(1, 0, 2) for name in ("q", "k", "v")]
            score = (q @ k.transpose(0, 2, 1)) * 0.5
            probability = np.exp(score - score.max(axis=-1, keepdims=True))
            probability /= probability.sum(axis=-1, keepdims=True)
            context = (probability @ v).transpose(1, 0, 2).reshape(5, 8)
            hidden = x + projection(context, weights, "self_attn.out_proj")
            g = projection(normalization(hidden, weights, "layer_norm2"), weights, "mlp.fc1")
            return hidden + projection(g / (1 + np.exp(-1.702 * g)), weights, "mlp.fc2")

        first, second = params(), params()
        with tempfile.TemporaryDirectory() as tmp:
            stage = CompiledClipStage(Path(tmp) / "block", graph, first)
            np.testing.assert_allclose(stage(x), reference(first), rtol=0, atol=2e-6)
            stage.load(second)
            np.testing.assert_allclose(stage(x), reference(second), rtol=0, atol=2e-6)
            stage.load(first)
            np.testing.assert_allclose(stage(x), reference(first), rtol=0, atol=2e-6)
            bad = x.copy()
            bad[0, -1] = np.nan
            with self.assertRaisesRegex(RuntimeError, "generated CLIP stage failed"):
                stage(bad)

    def test_stem_and_projector_execute_generated_operations(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(3, 5)).astype(np.float32)
        position = rng.normal(size=(3, 7)).astype(np.float32)

        def build(tmp, graph):
            weights = {name: (rng.normal(size=shape) * 0.25).astype(np.float32)
                       for name, (shape, _) in graph.weights.items()}
            return CompiledClipStage(Path(tmp) / graph.circuit["name"], graph, weights), weights

        with tempfile.TemporaryDirectory() as tmp:
            patch, weights = build(tmp, patch_projection(3, 5, 7))
            patches = patch(x)
            np.testing.assert_allclose(patches, projection(x.astype(np.float64), weights, "patch"),
                                       atol=1e-6, rtol=0)
            norm, weights = build(tmp, input_norm(3, 7))
            stem = norm(patches, position=position)
            np.testing.assert_allclose(stem, normalization(
                (patches + position).astype(np.float64), weights, "pre_layrnorm"), atol=1e-6, rtol=0)
            proj, weights = build(tmp, projector(3, 7, 4))
            hidden = projection(stem.astype(np.float64), weights, "0")
            erf = np.vectorize(math.erf)(hidden / math.sqrt(2.0))
            expected = projection(0.5 * hidden * (1 + erf), weights, "2")
            np.testing.assert_allclose(proj(stem), expected, atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
