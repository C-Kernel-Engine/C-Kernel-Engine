"""Connected generated-C ingress for Clef's decision head."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from tests.test_v8_checked_op_constants import write_synthetic_weights
from tests.test_v8_token_span_mean_checked import sequential_fp64_span_mean
from tests.v8_checked_graph_test_support import compile_native_graph

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v8/decision"))
from build_clef_flash_head_ingress_circuit import OUTPUT, build_circuit  # noqa: E402


def weights_for(channels, width, vocabulary, *, seed=271):
    rng = np.random.default_rng(seed)
    result = {
        "hidden_norm.weight": rng.uniform(0.7, 1.3, channels).astype("f4"),
        "hidden_norm.bias": rng.uniform(-0.1, 0.1, channels).astype("f4"),
        "output_embedding.weight": rng.normal(0, 0.2, (vocabulary, channels)).astype("f4"),
    }
    for name in ("memory_projection", "question_projection",
                 "option_context_projection", "option_lexical_projection",
                 "option_question_projection", "global_projection"):
        result[f"{name}.weight"] = rng.normal(
            0, 0.15, (width, channels)).astype("f4")
    return result


def compile_case(folder, graph, weights, *, enclosing_epsilon=1e-5,
                 extra_exports=()):
    folder.mkdir(parents=True, exist_ok=True)
    bundle = write_synthetic_weights(folder, weights)
    circuit_path = folder / "circuit.json"
    circuit_path.write_text(json.dumps(graph))
    tokens, channels = graph["activation_buffers"]["hidden"]["shape"]
    source = {"config": {
        "model": graph["name"], "arch": graph["name"],
        "num_layers": 1, "embed_dim": channels,
        "num_heads": 1, "num_kv_heads": 1, "head_dim": channels,
        "intermediate_size": channels, "context_length": tokens,
        "max_seq_len": tokens, "vocab_size": len(weights["output_embedding.weight"]),
        "epsilon": enclosing_epsilon,
        "activation_buffer_dtypes": {name: "i32" for name in
            ("input_ids", "question_spans", "option_spans", "global_span",
             "option_counts", "question_type_ids",
             "option_row_indices", "option_question_indices")}},
        "entries": bundle["entries"], "quant_summary": {}, "template": graph}
    layout, call_ir, _library, loaded, execute = compile_native_graph(
        folder, source, circuit_path, extra_exports=extra_exports)
    size = layout["memory"]["arena"]["total_size"]
    raw = (ctypes.c_uint8 * (size + 63))()
    arena = (ctypes.c_uint8 * size).from_buffer(raw,
                                                (-ctypes.addressof(raw)) & 63)
    payload = (folder / "weights.fixture").read_bytes()
    planned = {entry["name"]: entry for entry in
               layout["memory"]["weights"]["entries"]}
    for entry in bundle["entries"]:
        offset = planned[entry["name"]]["abs_offset"]
        arena[offset:offset + entry["size"]] = payload[
            entry["file_offset"]:entry["file_offset"] + entry["size"]]
    activations = {entry["name"]: entry for entry in
                   layout["memory"]["activations"]["buffers"]}

    def view(name):
        shape = graph["activation_buffers"][name]["shape"]
        dtype = np.int32 if name in source["config"]["activation_buffer_dtypes"] else np.float32
        return np.ndarray(shape, dtype=dtype, buffer=arena,
                          offset=activations[name]["abs_offset"])

    return arena, execute, view, call_ir, loaded


class ClefFlashHeadIngressTest(unittest.TestCase):
    def test_committed_circuit_regenerates(self):
        self.assertEqual(OUTPUT.read_text(),
                         json.dumps(build_circuit(), indent=2) + "\n")

    def test_final_valid_row_kernel_boundaries(self):
        graph = build_circuit(channels=8, width=4, vocabulary=16)
        with tempfile.TemporaryDirectory() as temp:
            _arena, _execute, _view, _call_ir, loaded = compile_case(
                Path(temp), graph, weights_for(8, 4, 16),
                extra_exports=("ck_last_valid_row_checked_f32",))
            kernel = loaded.ck_last_valid_row_checked_f32
            ptr_f = ctypes.POINTER(ctypes.c_float)
            ptr_i = ctypes.POINTER(ctypes.c_int32)
            kernel.argtypes = [ptr_f, ctypes.c_size_t, ctypes.c_size_t,
                               ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                               ptr_i, ctypes.c_size_t, ptr_f, ctypes.c_size_t]
            kernel.restype = ctypes.c_int
            source = np.arange(15, dtype="f4").reshape(3, 5)
            span = np.array([1, 2], dtype="i4")
            output = np.full(7, -19, dtype="f4")

            def call(*, stride=5, capacity=3, valid=2, output_array=output):
                return kernel(source.ctypes.data_as(ptr_f), source.size,
                              stride, capacity, valid, 3,
                              span.ctypes.data_as(ptr_i), span.size,
                              output_array.ctypes.data_as(ptr_f),
                              output_array.size)

            self.assertEqual(call(), 0)
            np.testing.assert_array_equal(output[:3], source[1, :3])
            np.testing.assert_array_equal(output[3:], -19)
            for wrong in ([0, 1], [0, 2]):
                span[:] = wrong
                output[:] = -19
                self.assertNotEqual(call(), 0)
                np.testing.assert_array_equal(output, -19)
            span[:] = [1, 2]
            self.assertNotEqual(call(stride=2), 0)
            self.assertNotEqual(call(valid=4), 0)
            self.assertNotEqual(call(output_array=output[:2]), 0)
            np.testing.assert_array_equal(output, -19)
            source[1, 0] = np.inf
            self.assertNotEqual(call(), 0)
            np.testing.assert_array_equal(output, -19)
            source[1, 0] = 5
            self.assertEqual(call(), 0)
            np.testing.assert_array_equal(output[:3], source[1, :3])

    def test_option_owner_kernel_boundaries(self):
        graph = build_circuit(channels=8, width=4, vocabulary=16)
        with tempfile.TemporaryDirectory() as temp:
            _arena, _execute, _view, _call_ir, loaded = compile_case(
                Path(temp), graph, weights_for(8, 4, 16),
                extra_exports=("ck_expand_group_row_indices_checked_i32",))
            kernel = loaded.ck_expand_group_row_indices_checked_i32
            ptr = ctypes.POINTER(ctypes.c_int32)
            kernel.argtypes = [ptr, ctypes.c_size_t, ctypes.c_size_t,
                               ctypes.c_size_t, ptr, ctypes.c_size_t,
                               ptr, ctypes.c_size_t]
            kernel.restype = ctypes.c_int
            counts = np.array([2, 1], dtype="i4")
            rows = np.full(4, -51, dtype="i4")
            owners = np.full(4, -52, dtype="i4")

            def call(*, row_output=rows, owner_output=owners, row_count=3):
                return kernel(counts.ctypes.data_as(ptr), counts.size,
                              counts.size, row_count,
                              row_output.ctypes.data_as(ptr), row_output.size,
                              owner_output.ctypes.data_as(ptr), owner_output.size)

            self.assertEqual(call(), 0)
            np.testing.assert_array_equal(rows, [0, 1, 2, -51])
            np.testing.assert_array_equal(owners, [0, 0, 1, -52])
            for invalid in ([2, 2], [0, 3], [-1, 4]):
                counts[:] = invalid
                rows[:] = -51
                owners[:] = -52
                self.assertNotEqual(call(), 0)
                np.testing.assert_array_equal(rows, -51)
                np.testing.assert_array_equal(owners, -52)
            counts[:] = [2, 1]
            self.assertNotEqual(call(row_output=rows[:2]), 0)
            self.assertNotEqual(call(owner_output=rows), 0)
            np.testing.assert_array_equal(rows, -51)
            np.testing.assert_array_equal(owners, -52)
            self.assertEqual(call(), 0)
            np.testing.assert_array_equal(rows[:3], [0, 1, 2])

    def test_connected_ingress_and_failure_propagation(self):
        channels, width, vocabulary = 8, 4, 16
        graph = build_circuit(channels=channels, width=width,
                              vocabulary=vocabulary)
        weights = weights_for(channels, width, vocabulary)
        with tempfile.TemporaryDirectory() as temp:
            arena, execute, view, call_ir, _loaded = compile_case(
                Path(temp), graph, weights)
            self.assertEqual(len(call_ir["operations"]), 15)
            self.assertEqual(call_ir["errors"], [])
            rng = np.random.default_rng(18)
            hidden = rng.normal(0, 1, (6, channels)).astype("f4")
            ids = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
            questions = np.array([[0, 2], [3, 6]], dtype=np.int32)
            options = np.array([[1, 3], [2, 5], [5, 6]], dtype=np.int32)
            view("hidden")[:] = hidden
            view("input_ids")[:] = ids
            view("question_spans")[:] = questions
            view("option_spans")[:] = options
            view("global_span")[:] = [[5, 6]]
            view("option_counts")[:] = [2, 1]
            view("option_lexical_projected")[:] = -901
            self.assertEqual(execute(arena, len(arena)), 0)
            np.testing.assert_array_equal(view("option_row_indices"), [0, 1, 2])
            np.testing.assert_array_equal(view("option_question_indices"), [0, 0, 1])
            np.testing.assert_array_equal(
                view("lexical_token_rows"), weights["output_embedding.weight"][ids])
            normalized = view("normalized_hidden").copy()
            # The reference takes the final valid hidden row, not an average
            # over the question or option spans.
            np.testing.assert_array_equal(view("global_vector")[0], normalized[-1])
            for name, spans, expected in (
                ("question_vectors", questions, normalized),
                ("option_context_vectors", options, normalized),
                ("lexical_option_vectors", options, weights["output_embedding.weight"][ids]),
            ):
                np.testing.assert_array_equal(view(name),
                                              sequential_fp64_span_mean(expected, spans))
            for source, output, weight in (
                ("normalized_hidden", "memory", "memory_projection.weight"),
                ("question_vectors", "question_projected", "question_projection.weight"),
                ("option_context_vectors", "option_context_projected", "option_context_projection.weight"),
                ("lexical_option_vectors", "option_lexical_projected", "option_lexical_projection.weight"),
                ("question_vectors", "option_question_projected", "option_question_projection.weight"),
                ("global_vector", "global_projected", "global_projection.weight"),
            ):
                expected = (view(source).astype("f8") @ weights[weight].astype("f8").T).astype("f4")
                np.testing.assert_allclose(view(output), expected, atol=1e-6, rtol=0)
            baseline = view("option_lexical_projected").copy()
            query_baseline = view("option_queries").copy()
            np.testing.assert_allclose(
                query_baseline,
                view("option_context_projected") +
                view("option_lexical_projected") +
                view("option_question_projected")[[0, 0, 1]],
                atol=1e-6, rtol=0)
            view("input_ids")[0] = vocabulary
            view("option_lexical_projected")[:] = -901
            self.assertNotEqual(execute(arena, len(arena)), 0)
            np.testing.assert_array_equal(view("option_lexical_projected"), -901)
            view("input_ids")[0] = ids[0]
            view("option_spans")[2] = [6, 7]
            self.assertNotEqual(execute(arena, len(arena)), 0)
            np.testing.assert_array_equal(view("option_lexical_projected"), -901)
            view("option_spans")[:] = options
            self.assertEqual(execute(arena, len(arena)), 0)
            np.testing.assert_array_equal(view("option_lexical_projected"), baseline)

            global_baseline = view("global_projected").copy()
            for wrong_span in ([4, 5], [0, 2]):
                view("global_span")[:] = [wrong_span]
                view("global_vector")[:] = -903
                view("global_projected")[:] = -904
                self.assertNotEqual(execute(arena, len(arena)), 0)
                np.testing.assert_array_equal(view("global_vector"), -903)
                np.testing.assert_array_equal(view("global_projected"), -904)
            view("global_span")[:] = [[5, 6]]
            self.assertEqual(execute(arena, len(arena)), 0)
            np.testing.assert_array_equal(view("global_projected"), global_baseline)
            view("option_counts")[:] = [2, 2]
            view("option_queries")[:] = -905
            view("option_row_indices")[:] = -906
            view("option_question_indices")[:] = -907
            self.assertNotEqual(execute(arena, len(arena)), 0)
            np.testing.assert_array_equal(view("option_queries"), -905)
            np.testing.assert_array_equal(view("option_row_indices"), -906)
            np.testing.assert_array_equal(view("option_question_indices"), -907)
            view("option_counts")[:] = [2, 1]
            self.assertEqual(execute(arena, len(arena)), 0)
            np.testing.assert_array_equal(view("option_queries"), query_baseline)
            view("option_counts")[:] = [1, 2]
            self.assertEqual(execute(arena, len(arena)), 0)
            self.assertFalse(np.array_equal(view("option_queries"), query_baseline))

    def test_layernorm_epsilon_is_circuit_owned(self):
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError:
            self.skipTest("PyTorch comparison unavailable")
        channels, width, vocabulary = 8, 4, 16
        graph = build_circuit(channels=channels, width=width,
                              vocabulary=vocabulary)
        weights = weights_for(channels, width, vocabulary, seed=59)
        hidden = np.full((6, channels), 0.25, dtype="f4")
        hidden[:, 0] += 0.0001
        results = []
        with tempfile.TemporaryDirectory() as temp:
            for directory, enclosing_epsilon in (("a", 1e-5), ("b", 1e-6)):
                arena, execute, view, _call_ir, _loaded = compile_case(
                    Path(temp) / directory, graph, weights,
                    enclosing_epsilon=enclosing_epsilon)
                view("hidden")[:] = hidden
                view("input_ids")[:] = [1, 2, 3, 4, 5, 6]
                view("question_spans")[:] = [[0, 2], [3, 6]]
                view("option_spans")[:] = [[1, 3], [2, 5], [5, 6]]
                view("global_span")[:] = [[5, 6]]
                view("option_counts")[:] = [2, 1]
                self.assertEqual(execute(arena, len(arena)), 0)
                results.append(view("normalized_hidden").copy())
        np.testing.assert_array_equal(results[0], results[1])
        reference = functional.layer_norm(
            torch.from_numpy(hidden), (channels,),
            torch.from_numpy(weights["hidden_norm.weight"]),
            torch.from_numpy(weights["hidden_norm.bias"]), eps=1e-5)
        np.testing.assert_allclose(results[0], reference.numpy(), atol=2e-6, rtol=0)

    def test_ingress_against_independent_pytorch(self):
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError:
            self.skipTest("PyTorch comparison unavailable")
        channels, width, vocabulary = 8, 4, 16
        graph = build_circuit(channels=channels, width=width,
                              vocabulary=vocabulary)
        weights = weights_for(channels, width, vocabulary, seed=42)
        with tempfile.TemporaryDirectory() as temp:
            arena, execute, view, _call_ir, _loaded = compile_case(
                Path(temp), graph, weights)
            hidden = np.random.default_rng(31).normal(
                0, 1, (6, channels)).astype("f4")
            ids = np.array([7, 3, 8, 2, 5, 1], dtype=np.int32)
            questions = [(0, 3), (3, 6)]
            options = [(1, 2), (2, 4), (4, 6)]
            view("hidden")[:] = hidden
            view("input_ids")[:] = ids
            view("question_spans")[:] = questions
            view("option_spans")[:] = options
            view("global_span")[:] = [[5, 6]]
            view("option_counts")[:] = [2, 1]
            self.assertEqual(execute(arena, len(arena)), 0)
            h = torch.from_numpy(hidden)
            normalized = functional.layer_norm(
                h, (channels,), torch.from_numpy(weights["hidden_norm.weight"]),
                torch.from_numpy(weights["hidden_norm.bias"]), eps=1e-5)
            lexical = torch.from_numpy(weights["output_embedding.weight"])[ids]
            q = torch.stack([normalized[a:b].mean(0) for a, b in questions])
            context = torch.stack([normalized[a:b].mean(0) for a, b in options])
            lex_options = torch.stack([lexical[a:b].mean(0) for a, b in options])
            expected = {
                "normalized_hidden": normalized,
                "lexical_token_rows": lexical,
                "question_vectors": q,
                "option_context_vectors": context,
                "lexical_option_vectors": lex_options,
                "global_vector": normalized[-1:],
            }
            for name, source in (
                ("memory", normalized), ("question_projected", q),
                ("option_context_projected", context),
                ("option_lexical_projected", lex_options),
                ("option_question_projected", q),
                ("global_projected", normalized[-1:]),
            ):
                projection = ("memory_projection" if name == "memory" else
                              name.removesuffix("_projected") + "_projection")
                expected[name] = functional.linear(
                    source, torch.from_numpy(weights[f"{projection}.weight"]))
            for name, reference in expected.items():
                np.testing.assert_allclose(view(name), reference.numpy(),
                                           atol=2e-6, rtol=0, err_msg=name)
            expected_queries = (expected["option_context_projected"] +
                                expected["option_lexical_projected"] +
                                expected["option_question_projected"][[0, 0, 1]])
            np.testing.assert_allclose(view("option_queries"),
                                       expected_queries.numpy(),
                                       atol=2e-6, rtol=0)

    def test_option_queries_with_unequal_counts(self):
        graph = build_circuit(tokens=7, questions=3, options=5,
                              channels=8, width=4, vocabulary=16)
        weights = weights_for(8, 4, 16, seed=89)
        with tempfile.TemporaryDirectory() as temp:
            arena, execute, view, _call_ir, _loaded = compile_case(
                Path(temp), graph, weights)
            rng = np.random.default_rng(117)
            view("hidden")[:] = rng.normal(0, 1, (7, 8)).astype("f4")
            view("input_ids")[:] = [1, 2, 3, 4, 5, 6, 7]
            view("question_spans")[:] = [[0, 2], [2, 4], [4, 7]]
            view("option_spans")[:] = [[0, 1], [1, 2], [2, 4],
                                       [4, 5], [5, 7]]
            view("global_span")[:] = [[6, 7]]
            view("option_counts")[:] = [2, 1, 2]
            owners = np.array([0, 0, 1, 2, 2], dtype=np.int32)
            self.assertEqual(execute(arena, len(arena)), 0)
            np.testing.assert_array_equal(view("option_row_indices"), np.arange(5))
            np.testing.assert_array_equal(view("option_question_indices"), owners)
            expected = (view("option_context_projected") +
                        view("option_lexical_projected") +
                        view("option_question_projected")[owners])
            np.testing.assert_allclose(view("option_queries"), expected,
                                       atol=1e-6, rtol=0)

    def test_pinned_head_weights_against_pytorch(self):
        asset = os.environ.get("CKE_CLEF_FLASH_HEAD_PATH")
        if not asset:
            self.skipTest("pinned Clef head weights unavailable")
        try:
            import torch
            import torch.nn.functional as functional
            from safetensors import safe_open
        except ImportError:
            self.skipTest("PyTorch/safetensors comparison unavailable")
        expected_hash = "19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0"
        actual_hash = hashlib.sha256(Path(asset).read_bytes()).hexdigest()
        self.assertEqual(actual_hash, expected_hash, "pinned head archive identity")
        channels, width, vocabulary = 4096, 1024, 16
        graph = build_circuit(channels=channels, width=width,
                              vocabulary=vocabulary)
        names = ("hidden_norm.weight", "hidden_norm.bias",
                 "memory_projection.weight", "question_projection.weight",
                 "option_context_projection.weight",
                 "option_lexical_projection.weight",
                 "option_question_projection.weight", "global_projection.weight")
        with safe_open(asset, framework="pt") as archive:
            weights = {name: archive.get_tensor(name).float().numpy()
                       for name in names}
        rng = np.random.default_rng(713)
        # The released head archive does not contain the tied backbone output
        # embeddings; this deterministic table tests that input contract only.
        weights["output_embedding.weight"] = rng.normal(
            0, 0.02, (vocabulary, channels)).astype("f4")
        with tempfile.TemporaryDirectory() as temp:
            arena, execute, view, _call_ir, _loaded = compile_case(
                Path(temp), graph, weights)
            hidden = rng.normal(0, 0.2, (6, channels)).astype("f4")
            ids = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
            question_spans = [(0, 2), (3, 6)]
            option_spans = [(1, 3), (2, 5), (5, 6)]
            view("hidden")[:] = hidden
            view("input_ids")[:] = ids
            view("question_spans")[:] = question_spans
            view("option_spans")[:] = option_spans
            view("global_span")[:] = [[5, 6]]
            view("option_counts")[:] = [2, 1]
            self.assertEqual(execute(arena, len(arena)), 0)
            normalized = functional.layer_norm(
                torch.from_numpy(hidden), (channels,),
                torch.from_numpy(weights["hidden_norm.weight"]),
                torch.from_numpy(weights["hidden_norm.bias"]), eps=1e-5)
            lexical = torch.from_numpy(weights["output_embedding.weight"])[ids]
            question = torch.stack([normalized[a:b].mean(0) for a, b in question_spans])
            context = torch.stack([normalized[a:b].mean(0) for a, b in option_spans])
            lexical_options = torch.stack([lexical[a:b].mean(0) for a, b in option_spans])
            sources = {
                "memory_projection": (normalized, "memory"),
                "question_projection": (question, "question_projected"),
                "option_context_projection": (context, "option_context_projected"),
                "option_lexical_projection": (lexical_options, "option_lexical_projected"),
                "option_question_projection": (question, "option_question_projected"),
                "global_projection": (normalized[-1:], "global_projected"),
            }
            for name, (source, output) in sources.items():
                reference = functional.linear(
                    source, torch.from_numpy(weights[f"{name}.weight"])).numpy()
                np.testing.assert_allclose(view(output), reference,
                                           atol=1e-4, rtol=0, err_msg=output)
            expected_queries = (
                functional.linear(context, torch.from_numpy(
                    weights["option_context_projection.weight"])) +
                functional.linear(lexical_options, torch.from_numpy(
                    weights["option_lexical_projection.weight"])) +
                functional.linear(question, torch.from_numpy(
                    weights["option_question_projection.weight"]))[[0, 0, 1]])
            np.testing.assert_allclose(view("option_queries"),
                                       expected_queries.numpy(),
                                       atol=1e-4, rtol=0)


if __name__ == "__main__":
    unittest.main()
