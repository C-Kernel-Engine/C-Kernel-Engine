"""Connected cached attention with visible relative, window, and sink effects."""

import ctypes
import math
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from tests.test_v8_inkling2_mini_causal_kv import entry
from version.v8.scripts.build_inkling2_mini_projection_circuit import relative_attention_replay
from version.v8.scripts.checked_circuit_authoring_v8 import declared_f32_weight_shapes
from version.v8.scripts.compile_checked_bundle_v8 import compile_declared_f32_graph


def dense_reference(query, key, value, relative, sink, window, position=0):
    """Independent dense softmax including an explicit zero-value sink."""
    tokens, heads, dim = query.shape
    kv_heads = key.shape[0]
    output = np.empty_like(query)
    for t in range(tokens):
        pos = position + t
        start = max(0, pos + 1 - window) if window else 0
        distances = pos - np.arange(start, pos + 1)
        for h in range(heads):
            kv = h // (heads // kv_heads)
            scores = key[kv, start:pos + 1].astype("f8") @ query[t, h].astype("f8") / np.sqrt(dim)
            in_band = distances < relative.shape[-1]
            scores[in_band] += relative[t, h, distances[in_band]]
            logits = np.append(scores, sink[t, h])
            prob = np.exp(logits - logits.max())
            prob /= prob.sum()
            output[t, h] = prob[:-1] @ value[kv, start:pos + 1].astype("f8")
    return output


class RelativeAttentionTests(unittest.TestCase):
    def test_cache_capacity_above_32k(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            library = Path(temp) / "attention.so"
            subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                            "-I", str(root / "include"), str(root / "src/kernels/gqa_relative_sink_checked.c"),
                            "-lm", "-o", str(library)], check=True)
            fn = ctypes.CDLL(str(library)).ck_gqa_relative_sink_checked_f32
            pointer, size = ctypes.POINTER(ctypes.c_float), ctypes.c_size_t
            fn.argtypes = [pointer, size] * 6 + [ctypes.POINTER(ctypes.c_double), size] + [size] * 8
            fn.restype = ctypes.c_int
            capacity = 34816
            position = capacity - 1
            zero = (ctypes.c_float * 1)(0)
            keys = (ctypes.c_float * capacity)()
            values = (ctypes.c_float * capacity)()
            values[position] = 42
            result = (ctypes.c_float * 1)(-1)
            scratch = (ctypes.c_double * 1)()
            self.assertEqual(fn(zero, 1, keys, capacity, values, capacity, zero, 1, zero, 1,
                                result, 1, scratch, 1, 1, 1, 1, 1, capacity, position, 1, 1), 0)
            self.assertEqual(result[0], 21)
            self.assertEqual(fn(zero, 1, keys, capacity, values, capacity, zero, 1, zero, 1,
                                result, 1, scratch, 1, 1, 1, 1, 1, capacity, capacity, 1, 1), -2)
            self.assertEqual(result[0], 21)

    def test_ordered_tiles_match_scalar_recurrence_with_tails(self):
        root = Path(__file__).resolve().parents[1]
        rng = np.random.default_rng(291)
        with tempfile.TemporaryDirectory() as temp:
            library = Path(temp) / "attention.so"
            subprocess.run(["cc", "-std=c11", "-O2", "-march=native", "-ffp-contract=fast",
                            "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                            str(root / "src/kernels/gqa_relative_sink_checked.c"), "-lm", "-o", str(library)],
                           check=True)
            fn = ctypes.CDLL(str(library)).ck_gqa_relative_sink_checked_f32
            fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t] * 7 + [ctypes.c_size_t] * 8
            fn.restype = ctypes.c_int
            tokens, heads, kv_heads, position, width = 4, 4, 2, 23, 5
            capacity = position + tokens
            for dim, window in ((19, 0), (16, 13)):
                q = rng.standard_normal((tokens, heads, dim), dtype=np.float32)
                k = rng.standard_normal((kv_heads, capacity, dim), dtype=np.float32)
                v = rng.standard_normal(k.shape, dtype=np.float32)
                bias = rng.standard_normal((tokens, heads, width), dtype=np.float32)
                sink = rng.standard_normal((tokens, heads), dtype=np.float32)
                output = np.empty_like(q)
                work = np.empty(dim, dtype=np.float64)
                spans = [arg for array in (q, k, v, bias, sink, output, work)
                         for arg in (array.ctypes.data, array.size)]
                self.assertEqual(fn(*spans, tokens, heads, kv_heads, dim, capacity, position, width, window), 0)
                expected = np.empty_like(output)
                for t, h in np.ndindex(tokens, heads):
                    current = position + t
                    first = max(0, current + 1 - window) if window else 0
                    peak, denom = float(sink[t, h]), 1.0
                    acc = [0.0] * dim
                    for j in range(first, current + 1):
                        dot = 0.0
                        for channel in range(dim):
                            dot += float(q[t, h, channel]) * float(k[h // 2, j, channel])
                        score = dot * (1.0 / math.sqrt(dim))
                        if current - j < width:
                            score += float(bias[t, h, current - j])
                        new_peak = max(score, peak)
                        old = math.exp(peak - new_peak)
                        new = math.exp(score - new_peak)
                        denom = denom * old + new
                        acc = [prior * old + float(value) * new for prior, value in zip(acc, v[h // 2, j])]
                        peak = new_peak
                    expected[t, h] = [element / denom for element in acc]
                np.testing.assert_array_equal(output.view("u4"), expected.view("u4"))

    def test_connected_prefix_and_decode_match_with_bias_window_and_sink(self):
        rng = np.random.default_rng(276)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundles = {}
            for label, count in (("split", 8), ("whole", 9)):
                graph = relative_attention_replay(prefill_tokens=count, width=7, kv_width=6,
                                                  kv_heads=2, query_heads=4, band=3, window=4,
                                                  slots=(6, 24), taps=4)
                if label == "split":
                    weights = {name: rng.normal(0, .2, size=shape).astype("f4")
                               for name, shape in declared_f32_weight_shapes(graph).items()}
                path = root / label
                path.mkdir()
                _, arena, view, lib = compile_declared_f32_graph(path, graph, weights)
                bundles[label] = arena, view, lib

            for slot in (6, 24):
                values = {name: rng.normal(size=shape).astype("f4") for name, shape in
                          (("k_input", (9, 7)), ("v_input", (9, 7)), ("query", (9, 4, 3)),
                           ("relative_band", (9, 4, 3)), ("sink", (9, 4)))}
                for label, (arena, view, lib) in bundles.items():
                    count = 8 if label == "split" else 9
                    for kind in ("k", "v"):
                        view(f"history_{kind}_{slot}").fill(0)
                        # Unwritten cache padding must never enter attention.
                        view(f"cache_{kind}_{slot}").fill(np.nan)
                    for name, data in values.items():
                        view(f"inkling2_prefill_slot{slot}_{name}")[:] = data[:count]
                    self.assertEqual(entry(lib, "prefill", slot)(arena, len(arena)), 0)
                    if label == "split":
                        for name, data in values.items():
                            view(f"inkling2_append_slot{slot}_{name}")[:] = data[8:]
                        self.assertEqual(entry(lib, "append", slot)(arena, len(arena)), 0)

                _, split, _ = bundles["split"]
                _, whole, _ = bundles["whole"]
                actual = np.concatenate((split(f"inkling2_prefill_slot{slot}_attention"),
                                         split(f"inkling2_append_slot{slot}_attention")))
                self.assertEqual(actual.tobytes(), whole(f"inkling2_prefill_slot{slot}_attention").tobytes())
                query, bias, sink = (values[name] for name in ("query", "relative_band", "sink"))
                key, value = split(f"cache_k_{slot}"), split(f"cache_v_{slot}")
                np.testing.assert_allclose(actual, dense_reference(query, key, value, bias, sink, 4),
                                           rtol=1e-6, atol=1e-7)
                for reference in (dense_reference(query, key, value, bias, sink, 0),
                                  dense_reference(query, key, value, np.zeros_like(bias), sink, 4),
                                  dense_reference(query, key, value, bias, np.full_like(sink, -100), 4)):
                    self.assertGreater(float(abs(actual - reference).max()), 1e-2)


if __name__ == "__main__":
    unittest.main()
