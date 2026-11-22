from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from version.v8.scripts.build_inkling2_cross_pass_v8 import CONTROL_PORTS, control_weight_shapes, cross_pass_controls
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph


class InklingCrossPassTests(unittest.TestCase):
    def test_prev_pass_messages_run_all_four_controllers(self):
        rows, states, width, rank = 3, 2, 7, 4
        rng = np.random.default_rng(437)
        graph = cross_pass_controls(rows, width, rank, states)
        decl = {name: (shape, "F32") for name, shape in control_weight_shapes(graph).items()}
        weights = {name: rng.normal(0, .2, shape).astype(np.float32) for name, (shape, _) in decl.items()}
        with TemporaryDirectory() as tmp:
            fn, arena, view, _ = compile_checked_weight_graph(Path(tmp), graph, weights, decl)
            initial = {}
            for name in ("attention_input", "ffn_input", "prior_bank", "prior_output"):
                initial[name] = rng.standard_normal(view(name).shape).astype(np.float32)
                view(name)[:] = initial[name]
            self.assertEqual(fn(arena, len(arena)), 0)

            def normalized(x):
                return x / np.sqrt(1e-6 + np.mean(x * x, axis=-1, keepdims=True))

            bank = normalized(np.concatenate(
                (initial["prior_bank"], initial["prior_output"][:, None]), axis=1))
            outputs = {}
            for port in CONTROL_PORTS:
                query = weights[f"blk.6.loop_msg_query.{port}"]
                scores = bank @ query / np.sqrt(width)
                probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                message = (bank * probabilities[..., None]).sum(axis=1)
                source = initial["ffn_input" if port == "ffn" else "attention_input"]
                joined = np.concatenate((normalized(source), normalized(message)), axis=1)
                stem = f"blk.6.loop_{port}_ctrl_"
                down = joined @ weights[stem + "down.weight"].T
                gain = np.tanh(down @ weights[stem + "gain.weight"].T)
                forget = np.tanh(down @ weights[stem + "forget.weight"].T)
                expected = (1 - forget) * source + gain * message
                np.testing.assert_allclose(view(f"{port}_message"), message, atol=1e-6)
                np.testing.assert_allclose(view(f"{port}_controlled"), expected, atol=2e-6)
                self.assertGreater(float(abs(expected - source).max()), 0.01)
                outputs[port] = view(f"{port}_controlled").copy()

            view("prior_bank")[:] = initial["prior_bank"] * -1
            view("prior_output")[:] = initial["prior_output"] * -1
            self.assertEqual(fn(arena, len(arena)), 0)
            self.assertGreater(float(abs(outputs["q"] - view("q_controlled")).max()), .01)
            view("prior_bank")[:] = initial["prior_bank"]
            view("prior_output")[:] = initial["prior_output"]
            self.assertEqual(fn(arena, len(arena)), 0)
            for port in CONTROL_PORTS:
                np.testing.assert_array_equal(view(f"{port}_controlled"), outputs[port])


if __name__ == "__main__":
    unittest.main()
