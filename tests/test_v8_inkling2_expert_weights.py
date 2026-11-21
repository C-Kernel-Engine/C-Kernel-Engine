from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from version.v8.scripts.build_inkling2_mini_projection_circuit import selected_expert
from version.v8.scripts.compile_checked_bundle_v8 import compile_declared_f32_graph
from version.v8.scripts.inkling2_expert_weights_v8 import (
    decode_e4m3fn, decode_expert_rows, split_gate_up_rows,
)


class InklingExpertWeightsTests(unittest.TestCase):
    def test_nvfp4_nibble_order_outer_scale_and_interleaving(self):
        weights = np.array([[0x71] * 8, [0xA4] * 8], dtype=np.uint8)
        scales = np.array([[0x38], [0x40]], dtype=np.uint8)
        decoded = decode_expert_rows(weights, scales, format_name="nvfp4", amax=3, denominator=2)
        gate, up = split_gate_up_rows(decoded)
        np.testing.assert_array_equal(gate, [[0.75, 9] * 8])
        np.testing.assert_array_equal(up, [[6, -3] * 8])
        with self.assertRaisesRegex(ValueError, "complete interleaved"):
            split_gate_up_rows(decoded[:1])
        with self.assertRaisesRegex(ValueError, "outer scale"):
            decode_expert_rows(weights, scales, format_name="nvfp4", amax=3, denominator=0)

    def test_mxfp8_scales_are_per_row_per_32_columns(self):
        values = np.broadcast_to(np.array([0x38] * 32 + [0xB8] * 32, dtype=np.uint8), (2, 64))
        scales = np.array([[127, 128], [129, 126]], dtype=np.uint8)
        expected = [[1] * 32 + [-2] * 32, [4] * 32 + [-0.5] * 32]
        np.testing.assert_array_equal(decode_expert_rows(values, scales, format_name="mxfp8"), expected)
        with self.assertRaisesRegex(ValueError, "scale grid"):
            decode_expert_rows(values, scales[:1], format_name="mxfp8")
        scales[0, 0] = 255
        with self.assertRaisesRegex(ValueError, "MXFP8 scale"):
            decode_expert_rows(values, scales, format_name="mxfp8")

    def test_e4m3_subnormal_max_and_nan(self):
        np.testing.assert_array_equal(
            decode_e4m3fn(np.array([0x01, 0x81, 0x7E, 0xFE], dtype=np.uint8)),
            [2**-9, -(2**-9), 448, -448],
        )
        for code in (0x7F, 0xFF):
            with self.subTest(code=code), self.assertRaisesRegex(ValueError, "NaN code"):
                decode_e4m3fn(np.array([code], dtype=np.uint8))

    def test_generated_complete_expert_keeps_sigmoid_gate_uncapped(self):
        graph = selected_expert(tokens=2, width=3, inner=2)
        hidden = np.array([[2, -3, 2], [-1, 4, 5]], dtype=np.float32)
        gate_weight = np.array([[4, -2, 1], [3, 3, -4]], dtype=np.float32)
        up_weight = np.array([[7, -8, 3], [11, 7, 2]], dtype=np.float32)
        down_weight = np.array([[.5, -.75], [1, 1], [2, .25]], dtype=np.float32)
        ops = graph["block_types"]["block"]["body"]["ops"]
        weights = {op["weight_refs"]["weight"]: weight
                   for op, weight in zip((ops[0], ops[1], ops[3]), (gate_weight, up_weight, down_weight))}
        gate = hidden @ gate_weight.T
        up = hidden @ up_weight.T
        capped_gate = 4 * np.tanh(gate / 4)
        capped_up = 25 * np.tanh(up / 25)
        activation = capped_gate / (1 + np.exp(-gate)) * capped_up
        wrong = capped_gate / (1 + np.exp(-capped_gate)) * capped_up
        self.assertGreater(np.abs(activation - wrong).max(), 0.1)

        with TemporaryDirectory() as temp:
            fn, arena, view, _ = compile_declared_f32_graph(Path(temp), graph, weights)
            view("hidden")[:] = hidden
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(view("gate"), gate)
            np.testing.assert_array_equal(view("up"), up)
            np.testing.assert_allclose(view("activated"), activation, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(view("down"), activation @ down_weight.T, rtol=1e-6, atol=1e-6)
            before = view("down").copy()
            self.assertNotEqual(fn(arena, 1), 0)
            np.testing.assert_array_equal(view("down"), before)
            self.assertEqual(fn(arena, len(arena)), 0)


if __name__ == "__main__":
    unittest.main()
