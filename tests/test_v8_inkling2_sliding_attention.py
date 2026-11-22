from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from tests.test_v8_inkling2_mini_causal_kv import entry
from tests.test_v8_inkling2_mini_relative_attention import dense_reference
from version.v8.scripts.build_inkling2_mini_projection_circuit import LAYER, sliding_attention_replay
from version.v8.scripts.checked_circuit_authoring_v8 import declared_f32_weight_shapes
from version.v8.scripts.compile_checked_bundle_v8 import compile_declared_f32_graph


class SlidingAttentionTests(unittest.TestCase):
    def test_generated_sliding_attention_projection_gate_and_continuation(self):
        rng = np.random.default_rng(491)
        kw = dict(width=7, kv_width=6, kv_heads=2, query_heads=4, slots=(6,), taps=4,
                  relative_dim=2, band=3, window=4, q_clip=.4, rel_clip=.25)
        inputs = {name: rng.normal(size=(5, 7)).astype(np.float32)
                  for name in ("query_input", "k_input", "v_input")}
        with TemporaryDirectory() as tmp:
            slots = {}
            for phase, tokens in (("split", 4), ("full", 5)):
                graph = sliding_attention_replay(prefill_tokens=tokens, **kw)
                if phase == "split":
                    weights = {name: rng.normal(0, .4, size=shape).astype(np.float32)
                               for name, shape in declared_f32_weight_shapes(graph).items()}
                folder = Path(tmp) / phase
                folder.mkdir()
                _, arena, view, lib = compile_declared_f32_graph(folder, graph, weights)
                for key, values in inputs.items():
                    view("inkling2_prefill_slot6_" + key)[:] = values[:tokens]
                    if phase == "split":
                        view("inkling2_append_slot6_" + key)[:] = values[tokens:]
                self.assertEqual(entry(lib, "prefill", 6)(arena, len(arena)), 0)
                if phase == "split":
                    self.assertEqual(entry(lib, "append", 6)(arena, len(arena)), 0)
                slots[phase] = view

            view = slots["split"]
            full = slots["full"]
            actual = np.concatenate([view(f"inkling2_{name}_slot6_projected_attention") for name in ("prefill", "append")])
            np.testing.assert_array_equal(actual, full("inkling2_prefill_slot6_projected_attention"))
            def weight(name):
                return weights[LAYER + "attn." + name]

            def project(name):
                return inputs["query_input"] @ weight(name + ".weight").T

            def scale_clip(source, name, dim, cap):
                scaled = (source * weight(name + ".scale")).reshape(5, 4, dim)
                magnitude = np.linalg.norm(scaled, axis=-1, keepdims=True)
                self.assertTrue(np.any(magnitude > cap))
                return scaled * np.minimum(1, cap / (magnitude + 1e-6))

            q = scale_clip(project("wq_du"), "q_l2_clip", 3, .4)
            rel = scale_clip(project("wr_du"), "rel_l2_clip", 2, .25) @ weight("rel_logits_proj.proj").T
            sink = project("attn_sink_proj")
            attention = dense_reference(q, view("cache_k_6"), view("cache_v_6"), rel, sink, 4).reshape(5, 12)
            gate = 1 / (1 + np.exp(-project("gate_attn")))
            expected = (attention * gate) @ weight("wo_ud.weight").T
            np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-6)
            ungated = attention @ weight("wo_ud.weight").T
            self.assertGreater(float(np.abs(actual - ungated).max()), .01)


if __name__ == "__main__":
    unittest.main()
