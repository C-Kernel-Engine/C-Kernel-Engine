from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from version.v8.scripts.build_inkling2_mini_projection_circuit import routed_expert_mixture
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph
from version.v8.scripts.replay_inkling2_routed_moe import GGML_TYPE_IQ4_NL, IQ4_LEVELS, unpack_gguf


def pack_quant(integers, storage):
    rows = integers.reshape(-1, 32)
    scale = 0.0078125 if storage == "iq4_nl" else 0.125
    scales = np.broadcast_to(np.array([scale], dtype="<f2").view(np.uint8), (len(rows), 2))
    if storage == "iq4_nl":
        codes = rows[:, :16] | (rows[:, 16:] << 4)
    else:
        codes = rows.view(np.uint8)
    return np.concatenate([scales, codes], axis=1).ravel()


def reference_mixture(hidden, router, bias, gate, up, down, bank, top_k):
    logits = hidden @ router.T
    scores = 1 / (1 + np.exp(-logits))
    ids = np.argsort(-(scores + bias), axis=-1, kind="stable")[:, :top_k]
    weights = np.take_along_axis(scores, ids, axis=-1)
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-20) * 8
    routed = np.empty((hidden.shape[0], top_k, hidden.shape[1]), dtype=np.float32)
    mixed = np.zeros_like(hidden)
    for row, routes in enumerate(ids):
        for slot in np.argsort(routes):
            expert = bank[routes[slot]]
            g = gate[expert] @ hidden[row]
            u = up[expert] @ hidden[row]
            act = (4 * np.tanh(g / 4)) / (1 + np.exp(-g)) * (25 * np.tanh(u / 25))
            routed[row, slot] = down[expert] @ act
            mixed[row] += weights[row, slot] * routed[row, slot]
    return ids, weights, routed.reshape(-1, hidden.shape[1]), mixed


class RoutedMoeTests(unittest.TestCase):
    def test_routing_drives_dispatch_for_each_weight_storage(self):
        hidden_dim, inner, experts, top_k, tokens = 32, 32, 4, 3, 2
        rng = np.random.default_rng(174)
        hidden = rng.normal(0, .2, (tokens, hidden_dim)).astype(np.float32)
        route = rng.normal(0, .2, (experts, hidden_dim)).astype(np.float32)
        bias = np.array([.2, -.1, .25, .05], dtype=np.float32)
        bank = np.array([3, 0, 2, 1], dtype=np.int32)
        matrices = [rng.integers(-16, 16, (experts, inner, hidden_dim), dtype=np.int8) for _ in range(3)]
        for storage in ("f32", "q8_0", "iq4_nl"):
            if storage == "iq4_nl":
                inputs = [(m & 15).astype(np.uint8) for m in matrices]
                expanded = [IQ4_LEVELS[m] * np.float32(.0078125) for m in inputs]
            else:
                inputs = matrices
                expanded = [m.astype(np.float32) * np.float32(.125) for m in matrices]
            ids, route_weights, routed, mixed = reference_mixture(
                hidden, route, bias, *expanded, bank, top_k,
            )
            wrong_bank = np.arange(experts, dtype=np.int32)
            wrong = reference_mixture(hidden, route, bias, *expanded, wrong_bank, top_k)[-1]
            self.assertGreater(np.max(np.abs(mixed - wrong)), 0.1)

            with self.subTest(storage=storage), TemporaryDirectory() as temp:
                graph = routed_expert_mixture(tokens, hidden_dim, inner, experts, top_k, experts, storage=storage)
                ops = graph["block_types"]["block"]["body"]["ops"]
                router = ops[0]["weight_refs"]["weight"]
                weights = {router: route}
                declarations = {router: (list(route.shape), "F32")}
                for name, integers, decoded in zip(ops[-1]["weight_refs"].values(), inputs, expanded):
                    payload = decoded if storage == "f32" else pack_quant(integers, storage)
                    weights[name] = payload
                    declarations[name] = (list(decoded.shape), "F32" if storage == "f32" else storage.upper())
                run, arena, view, _ = compile_checked_weight_graph(Path(temp), graph, weights, declarations)
                view("hidden")[:] = hidden
                view("correction_bias")[:] = bias
                view("expert_slots", np.int32)[:] = bank
                self.assertEqual(run(arena, len(arena)), 0)
                np.testing.assert_array_equal(view("indices", np.int32), ids)
                np.testing.assert_allclose(view("route_weights"), route_weights, atol=1e-6)
                np.testing.assert_allclose(view("expert_output"), routed, atol=1e-5, rtol=1e-5)
                np.testing.assert_allclose(view("mixture"), mixed, atol=2e-5, rtol=1e-5)
                before = view("mixture").copy()
                view("expert_slots", np.int32)[ids[0, 0]] = -1
                self.assertNotEqual(run(arena, len(arena)), 0)
                np.testing.assert_array_equal(view("mixture"), before)
                view("expert_slots", np.int32)[:] = bank
                self.assertEqual(run(arena, len(arena)), 0)
                np.testing.assert_array_equal(view("mixture"), before)
                if storage == "iq4_nl":
                    np.testing.assert_array_equal(
                        unpack_gguf(weights[name], expanded[-1].shape, GGML_TYPE_IQ4_NL), expanded[-1],
                    )


    def test_q8_router_order_is_bitwise_and_sensitive_to_slot_order(self):
        tokens, width, experts, top_k = 4, 32, 5, 4
        rng = np.random.default_rng(513)
        graph = routed_expert_mixture(tokens, width, width, experts, top_k, experts,
                                      storage="q8_0", q8_activations=True)
        ops = graph["block_types"]["block"]["body"]["ops"]
        ops[-1]["kernel"] += "_scaled_situ"
        router = ops[0]["weight_refs"]["weight"]
        weights = {router: rng.normal(0, .3, (experts, width)).astype(np.float32)}
        specs = {router: ([experts, width], "F32")}
        for name in ops[-1]["weight_refs"].values():
            shape = [experts, width, width]
            weights[name] = pack_quant(rng.integers(-12, 13, shape, dtype=np.int8), "q8_0")
            specs[name] = (shape, "Q8_0")
        with TemporaryDirectory() as temp:
            run, arena, view, _ = compile_checked_weight_graph(Path(temp), graph, weights, specs)
            view("hidden")[:] = rng.normal(0, .3, (tokens, width)).astype(np.float32)
            view("correction_bias")[:] = [0.1, -.06, .2, .33, -.09]
            view("expert_slots", np.int32)[:] = [3, 0, 1, 4, 2]
            self.assertEqual(run(arena, len(arena)), 0)
            contribution = view("expert_output").reshape(tokens, top_k, width)
            weighted = contribution * view("route_weights")[:, :, None]
            expected = weighted[:, 0].copy()
            for slot in range(1, top_k):
                expected += weighted[:, slot]
            np.testing.assert_array_equal(view("mixture").view("u4"), expected.view("u4"))
            ids = view("indices", np.int32)
            ascending = np.take_along_axis(weighted, np.argsort(ids, axis=1)[:, :, None], axis=1)
            incorrect = ascending[:, 0].copy()
            for slot in range(1, top_k):
                incorrect += ascending[:, slot]
            self.assertFalse(np.array_equal(incorrect.view("u4"), expected.view("u4")))

    def test_routed_provider_rejects_wrong_storage_even_with_enough_bytes(self):
        graph = routed_expert_mixture(1, 32, 32, 1, 1, 1, storage="iq4_nl")
        ops = graph["block_types"]["block"]["body"]["ops"]
        router = ops[0]["weight_refs"]["weight"]
        weights = {router: np.zeros((1, 32), dtype=np.float32)}
        declarations = {router: ([1, 32], "F32")}
        for name in ops[-1]["weight_refs"].values():
            weights[name] = np.zeros(32 * 34, dtype=np.uint8)
            declarations[name] = ([1, 32, 32], "Q8_0")
        with TemporaryDirectory() as temp, self.assertRaisesRegex(RuntimeError, "dtype|storage"):
            compile_checked_weight_graph(Path(temp), graph, weights, declarations)


if __name__ == "__main__":
    unittest.main()
