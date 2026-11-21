"""Nonzero generated-C regression for isolated Inkling layer-6 projections."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from version.v8.scripts.build_inkling2_mini_projection_circuit import attention_projections, controller_gates
from version.v8.scripts.checked_circuit_authoring_v8 import declared_f32_weight_shapes
from version.v8.scripts.compile_checked_bundle_v8 import compile_declared_f32_graph


class InklingGeneratedProjectionTests(unittest.TestCase):
    def test_distinct_query_key_value_inputs_and_gate_branches(self):
        graphs = (attention_projections(tokens=3, width=7, q=9, kv=4, relative=6, heads=3),
                  controller_gates(tokens=3, width=7, rank=4))
        rng = np.random.default_rng(414)
        for graph in graphs:
            with self.subTest(graph=graph["name"]), tempfile.TemporaryDirectory() as temp:
                weights = {name: rng.normal(0, 0.2, size=shape).astype("f4")
                           for name, shape in declared_f32_weight_shapes(graph).items()}
                fn, arena, view, _ = compile_declared_f32_graph(Path(temp), graph, weights)
                ops = graph["block_types"]["block"]["body"]["ops"]
                inputs = {op["graph_slots"]["inputs"]["input"][9:] for op in ops
                          if op["graph_slots"]["inputs"]["input"].startswith("external:")}
                expected = {}
                for name in sorted(inputs):
                    values = rng.normal(size=view(name).shape).astype("f4")
                    view(name)[:] = values
                    expected[name] = values
                for op in ops:
                    name = op["id"]
                    source = op["graph_slots"]["inputs"]["input"].removeprefix("external:")
                    if "weight_refs" in op:
                        weight = weights[op["weight_refs"]["weight"]]
                        expected[name] = (expected[source].astype("f8") @ weight.astype("f8").T).astype("f4")
                    else:
                        expected[name] = np.tanh(expected[source])
                self.assertEqual(fn(arena, len(arena)), 0)
                for op in ops:
                    np.testing.assert_allclose(view(op["id"]), expected[op["id"]], rtol=1e-6, atol=1e-7)
                view(sorted(inputs)[0])[0, 0] = np.nan
                self.assertNotEqual(fn(arena, len(arena)), 0)


if __name__ == "__main__":
    unittest.main()
