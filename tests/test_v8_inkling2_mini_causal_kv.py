"""Check actual generated slot ownership and ordered K/V history."""
import ctypes as C
from pathlib import Path
import tempfile
import unittest

import numpy as np

from version.v8.scripts.build_inkling2_mini_projection_circuit import causal_kv_replay
from version.v8.scripts.checked_circuit_authoring_v8 import declared_f32_weight_shapes
from version.v8.scripts.compile_checked_bundle_v8 import compile_declared_f32_graph


def entry(lib, phase, slot):
    fn = getattr(lib, f"ck_inkling2_{phase}_slot{slot}")
    fn.argtypes = [C.POINTER(C.c_uint8), C.c_size_t]
    fn.restype = C.c_int
    return fn


def reference(x, projection, filters, history):
    raw = (x.astype("f8") @ projection.astype("f8").T).astype("f4")
    span = np.concatenate((history, raw))
    taps = filters.shape[1]
    out = np.empty_like(raw)
    for token in range(len(raw)):
        window = span[token:token + taps].T.astype("f8")
        out[token] = (raw[token] + np.sum(window * filters, axis=1)) / np.sqrt(taps + 1)
    return out, span[-(taps - 1):]


class CausalKVTests(unittest.TestCase):
    def test_ordered_fp32_provider_matches_per_tap_rounding(self):
        rng = np.random.default_rng(362)
        graph = causal_kv_replay(width=7, kv_width=5, kv_heads=1, slots=(6,))
        for node in graph["block_types"]["block"]["body"]["ops"]:
            if node["kernel"] == "causal_depthwise_residual_checked_f32":
                node["kernel"] = "causal_depthwise_residual_checked_ordered_f32"
        weights = {name: rng.normal(size=shape).astype("f4")
                   for name, shape in declared_f32_weight_shapes(graph).items()}
        with tempfile.TemporaryDirectory() as temp:
            _, arena, view, lib = compile_declared_f32_graph(Path(temp), graph, weights)
            for kind in ("k", "v"):
                view(f"inkling2_prefill_slot6_{kind}_input")[:] = rng.normal(size=(8, 7))
                view(f"history_{kind}_6")[:] = rng.normal(size=(3, 5))
            history = view("history_k_6").copy()
            self.assertEqual(entry(lib, "prefill", 6)(arena, len(arena)), 0)
            raw = view("inkling2_prefill_slot6_k_projection")
            filters = weights["model.llm.layers.6.attn.k_sconv.weight"]
            span = np.concatenate((history, raw))
            total = np.zeros_like(raw)
            for tap in range(4):
                total += span[tap:tap + len(raw)] * filters[:, tap]
            expected = (total + raw) * np.float32(1 / np.sqrt(np.float32(5)))
            np.testing.assert_array_equal(view("inkling2_prefill_slot6_k_output"), expected)
            np.testing.assert_array_equal(view("history_k_6"), raw[-3:])

    def test_prefill_append_and_slot_alias_negative(self):
        rng = np.random.default_rng(414)
        graph = causal_kv_replay(width=7, kv_width=5, kv_heads=1, key_clip=.08)
        weights = {name: rng.normal(0, .2, size=shape).astype("f4")
                   for name, shape in declared_f32_weight_shapes(graph).items()}
        with tempfile.TemporaryDirectory() as temp:
            _, arena, view, lib = compile_declared_f32_graph(Path(temp), graph, weights)
            expected, initial = {}, {}
            for slot in (6, 24, 36):
                for kind in ("k", "v"):
                    history = rng.normal(size=(3, 5)).astype("f4")
                    rows = rng.normal(size=(9, 7)).astype("f4")
                    prefix = f"model.llm.layers.6.attn.{kind}"
                    result, final = reference(rows, weights[f"model.llm.layers.6.attn.w{kind}_dv.weight"],
                                              weights[f"{prefix}_sconv.weight"], history)
                    expected[slot, kind] = result, final
                    initial[slot, kind] = history
                    view(f"history_{kind}_{slot}")[:] = history
                    view(f"cache_{kind}_{slot}").fill(-777)
                    view(f"inkling2_prefill_slot{slot}_{kind}_input")[:] = rows[:8]
                    view(f"inkling2_append_slot{slot}_{kind}_input")[:] = rows[8:]
                self.assertEqual(entry(lib, "prefill", slot)(arena, len(arena)), 0)
                for kind in ("k", "v"):
                    np.testing.assert_allclose(view(f"inkling2_prefill_slot{slot}_{kind}_output"),
                                               expected[slot, kind][0][:8], rtol=1e-6, atol=1e-7)
                    self.assertTrue(np.all(view(f"cache_{kind}_{slot}")[:, 8] == -777))
            before = {slot: view(f"history_k_{slot}").copy() for slot in (6, 24, 36)}
            self.assertFalse(np.shares_memory(view("history_k_6"), view("history_k_24")))
            with self.assertRaisesRegex(ValueError, "independent state"):
                causal_kv_replay(slots=(6, 6))
            view("history_k_24")[:] = before[6]
            self.assertEqual(entry(lib, "append", 24)(arena, len(arena)), 0)
            self.assertGreater(float(abs(view("inkling2_append_slot24_k_output")
                                         - expected[24, "k"][0][-1:]).max()), 1e-4)
            # Restore both histories of the overwritten slot via a fresh prefill.
            for kind in ("k", "v"):
                view(f"history_{kind}_24")[:] = initial[24, kind]
            self.assertEqual(entry(lib, "prefill", 24)(arena, len(arena)), 0)
            self.assertEqual(view("history_k_24").tobytes(), before[24].tobytes())
            for slot in (36, 6, 24):
                self.assertEqual(entry(lib, "append", slot)(arena, len(arena)), 0)
                for kind in ("k", "v"):
                    result, final = expected[slot, kind]
                    np.testing.assert_allclose(view(f"inkling2_append_slot{slot}_{kind}_output"),
                                               result[8:], rtol=1e-6, atol=1e-7)
                    np.testing.assert_array_equal(view(f"history_{kind}_{slot}"), final)
                    raw = result.astype("f8")
                    if kind == "k":
                        gain = np.minimum(1, .08 / (np.linalg.norm(raw, axis=-1, keepdims=True) + 1e-6))
                    else:
                        gain = 1 / np.sqrt(np.mean(raw ** 2, axis=-1, keepdims=True) + 1e-6)
                    np.testing.assert_allclose(view(f"cache_{kind}_{slot}")[0],
                                               raw * gain, rtol=1e-6, atol=1e-7)

            bad_history = view("history_k_6")
            bad_history[-1, -1] = np.nan
            previous_history = bad_history.tobytes()
            output = view("inkling2_append_slot6_k_output")
            cache_before = view("cache_k_6").tobytes()
            output.fill(-777)
            self.assertNotEqual(entry(lib, "append", 6)(arena, len(arena)), 0)
            self.assertEqual(bad_history.tobytes(), previous_history)
            self.assertTrue(np.all(output == -777))
            self.assertEqual(view("cache_k_6").tobytes(), cache_before)


if __name__ == "__main__":
    unittest.main()
