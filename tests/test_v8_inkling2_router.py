from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from version.v8.scripts.build_inkling2_mini_projection_circuit import expert_router
from version.v8.scripts.compile_checked_bundle_v8 import compile_declared_f32_graph


class InklingRouterTests(unittest.TestCase):
    def test_generated_router_ranks_with_bias_and_weights_without_it(self):
        graph = expert_router(tokens=3, width=2, experts=4, top_k=2)
        hidden = np.array([[2, -1], [-1, 2], [0, 0]], dtype=np.float32)
        weight = np.array([[2, .5], [-1, 1], [.5, 2], [-1, -2]], dtype=np.float32)
        bias = np.array([-.3, 0, 0, .5], dtype=np.float32)
        logits = hidden @ weight.T
        scores = 1 / (1 + np.exp(-logits))
        expected_ids = np.argsort(-(scores + bias), axis=1, kind="stable")[:, :2]
        selected = np.take_along_axis(scores, expected_ids, axis=1)
        expected_weights = (selected / selected.sum(axis=1, keepdims=True)) * 8
        self.assertFalse(np.array_equal(expected_ids, np.argsort(-scores, axis=1, kind="stable")[:, :2]))
        self.assertEqual(expected_ids[-1, 1], 1)

        with TemporaryDirectory() as temp:
            fn, arena, view, _ = compile_declared_f32_graph(
                Path(temp), graph, {"model.llm.layers.6.mlp.gate.weight": weight},
            )
            view("hidden")[:] = hidden
            view("correction_bias")[:] = bias
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(view("logits"), logits)
            np.testing.assert_array_equal(view("indices", np.int32), expected_ids)
            np.testing.assert_allclose(view("route_weights"), expected_weights, rtol=1e-6)
            before = view("route_weights").copy()
            view("correction_bias")[-1] = np.nan
            self.assertNotEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(view("indices", np.int32), expected_ids)
            np.testing.assert_array_equal(view("route_weights"), before)
            view("correction_bias")[:] = bias
            self.assertEqual(fn(arena, len(arena)), 0)


if __name__ == "__main__":
    unittest.main()
