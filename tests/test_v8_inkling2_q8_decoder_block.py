from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from tests.test_v8_inkling2_mini_causal_kv import entry
from tests.test_v8_inkling2_routed_moe import pack_quant
from version.v8.scripts.build_inkling2_decoder_block_v8 import first_block_weights, first_decoder_block
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph


class InklingQ8DecoderBlockTests(unittest.TestCase):
    def test_first_q8_decoder_block_split_and_uninterrupted(self):
        rng = np.random.default_rng(141)
        spec = dict(width=32, kv_heads=1, query_heads=2, dim=16, relative_dim=2,
                    band=3, window=4, inner=32, experts=4, top_k=2, slots=4, taps=4)
        hidden = rng.normal(0, .2, (5, 32)).astype(np.float32)
        bias = np.array([.2, -.1, .25, .05], dtype=np.float32)
        with TemporaryDirectory() as tmp:
            views = {}
            for label, count in (("split", 4), ("full", 5)):
                graph = first_decoder_block(prefill_tokens=count, **spec)
                declarations = first_block_weights(graph)
                if label == "split":
                    weights = {}
                    for name, (shape, mode) in declarations.items():
                        if mode == "Q8_0":
                            codes = rng.integers(-4, 5, shape, dtype=np.int8)
                            weights[name] = pack_quant(codes, "q8_0")
                        else:
                            weights[name] = rng.normal(0, .4, shape).astype(np.float32)
                folder = Path(tmp) / label
                folder.mkdir()
                _, arena, view, lib = compile_checked_weight_graph(folder, graph, weights, declarations)
                view("expert_slots", np.int32)[:] = np.arange(4, dtype=np.int32)
                view("router_bias")[:] = bias
                view("inkling2_prefill_slot0_hidden")[:] = hidden[:count]
                self.assertEqual(entry(lib, "prefill", 0)(arena, len(arena)), 0)
                if label == "split":
                    view("inkling2_append_slot0_hidden")[:] = hidden[4:]
                    self.assertEqual(entry(lib, "append", 0)(arena, len(arena)), 0)
                views[label] = view

            split, full = views["split"], views["full"]
            output = np.concatenate([split(f"inkling2_{phase}_slot0_layer_output")
                                     for phase in ("prefill", "append")])
            np.testing.assert_array_equal(output, full("inkling2_prefill_slot0_layer_output"))
            for kind in ("k_0", "v_0", "attn_post_conv", "ffn_post_conv"):
                np.testing.assert_array_equal(split("history_" + kind), full("history_" + kind))
            for kind in ("k", "v"):
                np.testing.assert_array_equal(split(f"cache_{kind}_0"), full(f"cache_{kind}_0")[:, :5])
            partial = full("inkling2_prefill_slot0_attn_post_conv")
            contribution = full("inkling2_prefill_slot0_ffn_post_conv")
            bank = np.stack((hidden, partial), axis=1)
            normalized = bank / np.sqrt((bank * bank).mean(axis=-1, keepdims=True) + 1e-6)
            query = weights["blk.0.ffn_res_query.weight"]
            scores = normalized @ query / np.sqrt(query.size)
            probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            expected = (normalized * probabilities[..., None]).sum(axis=1)
            np.testing.assert_allclose(full("inkling2_prefill_slot0_depth_mix"), expected,
                                       rtol=2e-6, atol=2e-7)
            self.assertGreater(float(abs(contribution).max()), 1e-3)
            np.testing.assert_array_equal(output, partial + contribution)
            self.assertGreater(float(abs(output - (hidden + partial + contribution)).max()), 1e-2)


if __name__ == "__main__":
    unittest.main()
