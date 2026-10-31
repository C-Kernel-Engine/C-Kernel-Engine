"""Two-layer attention observation through normal v8 lowering and codegen."""

import ctypes
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph


class AttentionLastQueryProbTraceGeneratedTest(unittest.TestCase):
    def test_two_layer_trace_and_rejection(self):
        layers, heads, queries, frames, dim = 2, 2, 1, 5, 3
        q_size, k_size = heads * queries * dim, heads * frames * dim
        trace_size, score_size = layers * heads * frames, heads * frames
        op = {
            "id": "observe", "op": "attention_last_query_prob_trace",
            "kernel": "attention_last_query_prob_trace_f32", "returns_status": True,
            "params": {"H": heads, "Tq": queries, "D": dim,
                       "L": layers, "F_capacity": frames,
                       "attention_scale": dim ** -0.5,
                       "call_constants": {
                           "query_elements": q_size, "key_elements": k_size,
                           "trace_elements": trace_size,
                           "score_elements": score_size,
                           "layers": layers, "heads": heads,
                           "query_tokens": queries, "key_tokens": frames,
                           "frame_capacity": frames, "head_dim": dim,
                           "layer": 0}},
            "graph_slots": {
                "inputs": {"query": "external:query", "key": "external:key"},
                "outputs": {"trace": "attention_trace"}},
        }
        second = copy.deepcopy(op)
        second["id"] = "observe_second_layer"
        second["params"]["call_constants"]["layer"] = 1
        buffers = {"query": {"shape": [heads, queries, dim]},
                   "key": {"shape": [heads, frames, dim]},
                   "attention_trace": {"shape": [layers, heads, frames],
                                       "lifetime": "sequence"}}
        circuit = {
            "version": 3, "name": "synthetic_attention_observation",
            "family": "bounded_graph", "checked_native_entry": True,
            "contract": {"runtime_invariants": {"inference_only": True}},
            "activation_buffers": buffers,
            "activation_bindings": {name: name for name in buffers},
            "native_entry": {
                "function": "ck_synthetic_attention_observation",
                "params": [{"c_type": "uint8_t *", "name": "arena"},
                           {"c_type": "size_t", "name": "arena_bytes"}],
                "arena": {"pointer": "arena", "bytes": "arena_bytes"}},
            "sequence": ["component"],
            "block_types": {"component": {"sequence": ["header", "body", "footer"],
                                          "header": [],
                                          "body": {"type": "dense", "ops": [op, second]},
                                          "footer": []}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "circuit.json"
            path.write_text(json.dumps(circuit))
            source = {"config": {"model": circuit["name"],
                                  "arch": circuit["name"], "num_layers": 1,
                                  "embed_dim": heads * dim, "num_heads": heads,
                                  "num_kv_heads": heads, "head_dim": dim,
                                  "intermediate_size": 4,
                                  "context_length": queries,
                                  "max_seq_len": queries,
                                  "vocab_size": 1},
                      "entries": [], "quant_summary": {}, "template": circuit}
            try:
                layout, calls, _library, _loaded, execute = compile_native_graph(
                    root, source, path)
            except subprocess.CalledProcessError as exc:
                self.fail(f"generated compile failed: {exc.stderr}")
            self.assertEqual(calls["errors"], [])
            self.assertEqual([item["function"] for item in calls["operations"]],
                             ["attention_last_query_prob_trace_f32"] * layers)
            arena_size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (arena_size + 63))()
            arena = (ctypes.c_uint8 * arena_size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            locations = {item["name"]: item for item in
                         layout["memory"]["activations"]["buffers"]}
            def view(name, shape):
                return np.ndarray(shape, np.float32, buffer=arena,
                                  offset=locations[name]["abs_offset"])
            query = view("query", (heads, queries, dim))
            key = view("key", (heads, frames, dim))
            trace = view("attention_trace", (layers, heads, frames))
            rng = np.random.default_rng(778)
            query[:] = rng.normal(size=query.shape)
            key[:] = rng.normal(size=key.shape)
            trace.fill(-77)
            self.assertEqual(execute(arena, arena_size), 0)
            expected = np.einsum("hqd,hfd->hqf", query, key)[:, -1] * dim ** -0.5
            expected = np.exp(expected - expected.max(axis=1, keepdims=True))
            expected /= expected.sum(axis=1, keepdims=True)
            for layer in range(layers):
                np.testing.assert_allclose(trace[layer], expected, rtol=2e-6, atol=2e-6)
            key[0, 0, 0] = np.nan
            trace.fill(-77)
            self.assertNotEqual(execute(arena, arena_size), 0)
            np.testing.assert_array_equal(trace, -77)


if __name__ == "__main__":
    unittest.main()
