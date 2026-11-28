"""Portable generated weighted-projector regression with a nonzero reference."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from version.v8.scripts.nemotron_vl_checked_v8 import compile_weighted_projector


def bf16(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def fp32(x):
    return (x.astype(np.uint32) << 16).view(np.float32)


class NemotronWeightedProjectorTest(unittest.TestCase):
    def test_generated_weighted_projector_with_intermediate_stores(self):
        rng = np.random.default_rng(73)
        rows, width, inner, output = 3, 13, 17, 5
        x = bf16(rng.normal(size=(rows, width)))
        weights = {
            "mlp1.0.weight": bf16(rng.uniform(0.2, 1.6, width)),
            "mlp1.1.weight": bf16(rng.normal(size=(inner, width)) * 0.4),
            "mlp1.3.weight": bf16(rng.normal(size=(output, inner)) * 0.3),
        }
        gamma, a, b = [fp32(w) for w in weights.values()]

        def reference(features):
            f = fp32(features)
            norm = bf16(f / np.sqrt(np.mean(f * f, axis=-1, keepdims=True) + 1e-5) * gamma)
            expanded = bf16(fp32(norm).astype(np.float64) @ a.T)
            activated = bf16(np.maximum(fp32(expanded), 0) ** 2)
            embeddings = bf16(fp32(activated).astype(np.float64) @ b.T)
            return norm, expanded, activated, embeddings

        with tempfile.TemporaryDirectory() as tmp:
            run, arena, view, _ = compile_weighted_projector(
                Path(tmp), weights, rows=rows, width=width, inner=inner, output=output,
            )
            features = view("features", np.uint16)
            for inputs in (x, bf16(-fp32(x)), x):
                features[:] = inputs
                self.assertEqual(run(arena, len(arena)), 0)
                for name, expected in zip(("norm", "expanded", "activated", "embeddings"),
                                          reference(inputs)):
                    np.testing.assert_array_equal(view(name, np.uint16), expected)
            self.assertTrue(np.any(fp32(view("expanded", np.uint16)) < 0))
            self.assertTrue(np.any(fp32(view("expanded", np.uint16)) > 0))
            prior = view("embeddings", np.uint16).copy()
            features[-1, -1] = 0x7FC0
            self.assertNotEqual(run(arena, len(arena)), 0)
            np.testing.assert_array_equal(view("embeddings", np.uint16), prior)

    def test_squared_relu_overflow_does_not_commit_output(self):
        weights = {
            "mlp1.0.weight": bf16([1, 1]),
            "mlp1.1.weight": bf16([[1, 0], [0, 1e25]]),
            "mlp1.3.weight": bf16([[1, 1]]),
        }
        with tempfile.TemporaryDirectory() as tmp:
            run, arena, view, _ = compile_weighted_projector(
                Path(tmp), weights, rows=1, width=2, inner=2, output=1,
            )
            features, activated = [view(n, np.uint16) for n in ("features", "activated")]
            features[:] = bf16([[1, 0]])
            self.assertEqual(run(arena, len(arena)), 0)
            prior = activated.copy()
            features[:] = bf16([[1, 1]])
            self.assertNotEqual(run(arena, len(arena)), 0)
            np.testing.assert_array_equal(activated, prior)
            features[:] = bf16([[1, 0]])
            self.assertEqual(run(arena, len(arena)), 0)


if __name__ == "__main__":
    unittest.main()
