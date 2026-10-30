"""Model-neutral selected-attention alignment through the normal v8 pipeline."""

import ctypes
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph
from tests.test_v8_alignment_attention_dtw import reference


class AlignmentAttentionDtwGeneratedTest(unittest.TestCase):
    def test_generated_entry_and_invalid_input(self):
        heads, tokens, frames = 2, 5, 9
        op = {
            "id": "align", "op": "alignment_attention_dtw",
            "kernel": "alignment_attention_dtw_f32", "returns_status": True,
            "params": {"H": heads, "T": tokens, "F": frames,
                       "F_capacity": frames,
                       "heads": heads, "median_width": 3,
                       "call_constants": {
                           "heads": heads,
                           "attention_elements": heads * tokens * frames,
                           "tokens": tokens, "frames": frames,
                           "token_stride": frames,
                           "head_stride": tokens * frames,
                           "score_elements": tokens * frames,
                           "mean_elements": tokens * frames,
                           "cost_elements": (tokens + 1) * (frames + 1),
                           "trace_elements": (tokens + 1) * (frames + 1),
                           "token_capacity": tokens}},
            "graph_slots": {"inputs": {"attention": "external:attention"},
                            "outputs": {"token_frames": "token_frames"}},
        }
        buffers = {"attention": {"shape": [heads, tokens, frames]},
                   "token_frames": {"shape": [tokens]}}
        circuit = {"version": 3, "name": "synthetic_attention_alignment",
                   "family": "bounded_graph", "checked_native_entry": True,
                   "contract": {"runtime_invariants": {"inference_only": True}},
                   "activation_buffers": buffers,
                   "activation_bindings": {name: name for name in buffers},
                   "native_entry": {"function": "ck_synthetic_attention_alignment",
                                    "params": [{"c_type": "uint8_t *", "name": "arena"},
                                               {"c_type": "size_t", "name": "arena_bytes"}],
                                    "arena": {"pointer": "arena", "bytes": "arena_bytes"}},
                   "sequence": ["component"], "block_types": {"component": {
                       "sequence": ["header", "body", "footer"], "header": [],
                       "body": {"type": "dense", "ops": [op]}, "footer": []}}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "circuit.json"
            path.write_text(json.dumps(circuit))
            source = {"config": {"model": circuit["name"],
                                  "arch": circuit["name"], "num_layers": 1,
                                  "embed_dim": 2, "num_heads": 1,
                                  "num_kv_heads": 1, "head_dim": 2,
                                  "intermediate_size": 2,
                                  "context_length": tokens, "max_seq_len": tokens,
                                  "vocab_size": 1},
                      "entries": [], "quant_summary": {}, "template": circuit}
            layout, calls, _library, _loaded, execute = compile_native_graph(
                root, source, path)
            self.assertEqual(calls["errors"], [])
            self.assertEqual(calls["operations"][-1]["function"],
                             "alignment_attention_dtw_f32")
            arena_size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (arena_size + 63))()
            arena = (ctypes.c_uint8 * arena_size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            locations = {item["name"]: item for item in
                         layout["memory"]["activations"]["buffers"]}
            attention = np.ndarray((heads, tokens, frames), np.float32,
                                   buffer=arena,
                                   offset=locations["attention"]["abs_offset"])
            output = np.ndarray((tokens,), np.int32, buffer=arena,
                                offset=locations["token_frames"]["abs_offset"])
            attention[:] = np.random.default_rng(551).uniform(
                .01, 1, attention.shape)
            expected = reference(attention.copy(), 3)
            output.fill(-77)
            self.assertEqual(execute(arena, arena_size), 0)
            np.testing.assert_array_equal(output, expected)
            attention[0, 0, 0] = np.nan
            output.fill(-77)
            self.assertNotEqual(execute(arena, arena_size), 0)
            np.testing.assert_array_equal(output, -77)


if __name__ == "__main__":
    unittest.main()
