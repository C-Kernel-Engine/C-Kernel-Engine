"""Portable RADIO topology and image preparation checks."""

import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from version.v8.scripts.instella_vl_checked_v8 import CompiledClipStage
from version.v8.scripts.nemotron_radio_checked_v8 import CompiledRadioEncoder, radio_block, tile_patches


class RadioTests(unittest.TestCase):
    def test_omni_cannot_enter_v2_encoder(self):
        omni = {"model_type": "NemotronH_Nano_Omni_Reasoning_V3",
                "vision_config": {"version": "c-radio_v4-h"},
                "llm_config": {"hidden_size": 2688}}
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "not_created"
            with self.assertRaisesRegex(ValueError, "unsupported Nemotron VL"):
                CompiledRadioEncoder(target, omni, None, None, None)
            self.assertFalse(target.exists())

    def test_spatial_and_channel_order(self):
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        image[1, 2, 1] = 255
        image[0, 16, 2] = 255
        patches = tile_patches(image, np.zeros(3), np.ones(3))
        self.assertEqual(patches[0, 256+16+2], 1)
        self.assertEqual(patches[1, 512], 1)
        self.assertEqual(np.count_nonzero(patches), 2)
        with self.assertRaises(ValueError):
            tile_patches(image[:-1], np.zeros(3), np.ones(3))

    def test_generated_block_matches_independent_reference(self):
        rng = np.random.default_rng(23)
        graph = radio_block(5, 8, 2, 12)
        w = {name: (rng.normal(size=shape) * 0.2).astype(np.float32)
             for name, (shape, _) in graph.weights.items()}
        x = (rng.normal(size=(5, 8)) / 10).astype(np.float32)

        def norm(value, key):
            centered = value - value.mean(axis=-1, keepdims=True)
            variance = np.mean(centered**2, axis=-1, keepdims=True)
            return centered / np.sqrt(variance + 1e-6) * w[key + ".weight"] + w[key + ".bias"]

        def linear(value, key):
            return value @ w[key+".weight"].T + w[key+".bias"]

        normed = norm(x.astype(np.float64), "norm1")
        q, k, v = [linear(normed, name).reshape(5, 2, 4).transpose(1, 0, 2)
                   for name in ("q", "k", "v")]
        scores = (q @ k.transpose(0, 2, 1)) / 2
        p = np.exp(scores - scores.max(axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
        residual = x + linear((p @ v).transpose(1, 0, 2).reshape(5, 8), "attn.proj")
        hidden = linear(norm(residual, "norm2"), "mlp.fc1")
        expected = residual + linear(hidden * 0.5 * (1 + np.vectorize(math.erf)(hidden / math.sqrt(2))),
                                     "mlp.fc2")
        with tempfile.TemporaryDirectory() as tmp:
            stage = CompiledClipStage(Path(tmp) / "radio", graph, w)
            np.testing.assert_allclose(stage(x), expected, atol=2e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
