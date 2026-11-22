import ctypes
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from tests.test_v8_inkling2_mini_causal_kv import entry
from tests.test_v8_inkling2_routed_moe import pack_quant
from version.v8.scripts.build_inkling2_decoder_block_v8 import (
    decoder_block, first_block_weights, first_decoder_block, sliding_decoder_block,
)
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph


class InklingQ8DecoderBlockTests(unittest.TestCase):
    def test_first_q8_decoder_block_split_and_uninterrupted(self):
        for quantized in (False, True):
            with self.subTest(q8_activations=quantized):
                self.check_split(q8_activations=quantized)

    def test_controlled_sliding_block_split_and_uninterrupted(self):
        self.check_split(q8_activations=True, controlled=True)

    def test_controlled_full_attention_split_and_reference(self):
        self.check_split(q8_activations=True, controlled=True, full_attention=True)

    def check_split(self, *, q8_activations, controlled=False, full_attention=False):
        rng = np.random.default_rng(141)
        spec = dict(width=32, kv_heads=1, query_heads=2, dim=16, relative_dim=2,
                    band=3, window=4, inner=32, experts=4, top_k=2, slots=4, taps=4,
                    q8_activations=q8_activations)
        hidden = rng.normal(0, .2, (5, 32)).astype(np.float32)
        banks = {}
        build_block = first_decoder_block
        slot = 0
        if controlled:
            build_block = sliding_decoder_block
            slot = 24
            spec.update(layer=6, state_slot=slot, depth_states=1, previous_pass_states=1, control_rank=4)
            banks = {
                "depth_bank": rng.normal(0, .4, (5, 1, 32)).astype(np.float32),
                "previous_bank": rng.normal(0, .4, (5, 1, 32)).astype(np.float32),
                "previous_output": rng.normal(0, .4, (5, 32)).astype(np.float32),
            }
        if full_attention:
            build_block = decoder_block
            slot = 29
            spec.update(layer=11, state_slot=slot, full_attention=True, block_boundary=False,
                        window=0, epsilon=1e-6, log_floor=2, log_alpha=.1)
        prefill = f"inkling2_prefill_slot{slot}"
        append = f"inkling2_append_slot{slot}"
        bias = np.array([.2, -.1, .25, .05], dtype=np.float32)
        with TemporaryDirectory() as tmp:
            views = {}
            for label, count in (("split", 4), ("full", 5)):
                graph = build_block(prefill_tokens=count, **spec)
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
                view(prefill + "_hidden")[:] = hidden[:count]
                for name, value in banks.items():
                    view(prefill + "_" + name)[:] = value[:count]
                self.assertEqual(entry(lib, "prefill", slot)(arena, len(arena)), 0)
                if label == "split":
                    view(append + "_hidden")[:] = hidden[4:]
                    for name, value in banks.items():
                        view(append + "_" + name)[:] = value[4:]
                    self.assertEqual(entry(lib, "append", slot)(arena, len(arena)), 0)
                if controlled and label == "full":
                    def reset_state():
                        for name in (f"history_k_{slot}", f"history_v_{slot}", f"cache_k_{slot}",
                                     f"cache_v_{slot}", "history_attn_post_conv", "history_ffn_post_conv"):
                            view(name).fill(0)

                    expected = view(prefill + "_layer_output").copy()
                    if not full_attention:
                        np.testing.assert_array_equal(
                            view(prefill + "_next_depth_bank"),
                            np.concatenate((banks["depth_bank"], hidden[:, None, :]), axis=1),
                        )
                    reset_state()
                    view(prefill + "_previous_output")[:] = -banks["previous_output"]
                    self.assertEqual(entry(lib, "prefill", slot)(arena, len(arena)), 0)
                    self.assertGreater(float(abs(view(prefill + "_layer_output") - expected).max()), 1e-3)
                    reset_state()
                    view(prefill + "_previous_output")[:] = banks["previous_output"]
                    self.assertEqual(entry(lib, "prefill", slot)(arena, len(arena)), 0)
                    np.testing.assert_array_equal(view(prefill + "_layer_output"), expected)
                views[label] = view

            split, full = views["split"], views["full"]
            output = np.concatenate([split(f"inkling2_{phase}_slot{slot}_layer_output")
                                     for phase in ("prefill", "append")])
            np.testing.assert_array_equal(output, full(prefill + "_layer_output"))
            for kind in (f"k_{slot}", f"v_{slot}", "attn_post_conv", "ffn_post_conv"):
                np.testing.assert_array_equal(split("history_" + kind), full("history_" + kind))
            for kind in ("k", "v"):
                np.testing.assert_array_equal(split(f"cache_{kind}_{slot}"), full(f"cache_{kind}_{slot}")[:, :5])
            if full_attention:
                self.check_full_attention_boundaries(full, prefill, weights)
            if controlled:
                return
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


    def check_full_attention_boundaries(self, view, prefill, weights):
        for kind in ("k", "v"):
            projected = view(f"{prefill}_{kind}_projection").astype(np.float64)
            dynamic = view(f"{prefill}_{kind}_dynamic").astype(np.float64)
            filters = weights[f"blk.11.attn_{kind}_conv.weight"].astype(np.float64)
            padded = np.pad(projected, ((3, 0), (0, 0)))
            expected = np.stack([
                (projected[token] + (padded[token:token+4].T *
                                     (filters + dynamic[token, ::-1])).sum(axis=1)) / 3
                for token in range(5)
            ])
            np.testing.assert_allclose(view(f"{prefill}_{kind}_output"), expected, rtol=2e-6, atol=2e-7)
        scale = 1 + .1 * np.log(np.maximum(1, np.arange(1, 6) / 2))
        for name in ("query", "relative_scaled"):
            raw = view(f"{prefill}_{name}").reshape(5, -1)
            scaled = view(f"{prefill}_{name}_log_scale").reshape(5, -1)
            np.testing.assert_allclose(scaled, raw * scale[:, None], rtol=2e-6, atol=2e-7)
        self.assertGreater(float(np.abs(view(prefill + "_attention_residual") -
                                        view(prefill + "_attn_post_conv")).max()), .01)

    def test_full_attention_kernels_reject_without_writes(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as tmp:
            library = Path(tmp) / "full-attention.so"
            subprocess.run(
                ["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                 "-I", str(root / "include"), str(root / "src/kernels/causal_depthwise_residual_checked.c"),
                 str(root / "src/kernels/log_position_scale_rows_checked.c"), "-lm", "-o", str(library)],
                check=True,
            )
            loaded = ctypes.CDLL(str(library))
            conv = loaded.causal_dynamic_depthwise_residual_checked_f32
            scale = loaded.log_position_scale_rows_checked_f32
            pointer, size = ctypes.POINTER(ctypes.c_float), ctypes.c_size_t
            conv.argtypes = [pointer, size] * 5 + [size] * 6
            scale.argtypes = [pointer, size] * 2 + [size] * 4 + [ctypes.c_float]
            src = (ctypes.c_float * 4)(1, 2, 3, 4)
            history = (ctypes.c_float * 2)(5, 6)
            static = (ctypes.c_float * 4)(.5, -.25, 1, -.5)
            taps = (ctypes.c_float * 4)(.5, .1, .9, -.2)
            out = (ctypes.c_float * 4)(42, 42, 42, 42)
            self.assertNotEqual(conv(src, 4, static, 4, taps, 3, history, 2, out, 4,
                                     2, 2, 1, 2, 1, 2), 0)
            self.assertEqual(list(history), [5, 6])
            self.assertEqual(list(out), [42] * 4)
            self.assertNotEqual(scale(src, 4, out, 4, 2, 2, size(-1).value, 2, .1), 0)
            self.assertEqual(list(out), [42] * 4)
            self.assertEqual(conv(src, 4, static, 4, taps, 4, history, 2, out, 4,
                                  2, 2, 1, 2, 1, 2), 0)
            self.assertEqual(list(history), [3, 4])
            self.assertEqual(scale(src, 4, out, 4, 2, 2, 0, 2, .1), 0)
            self.assertEqual(list(out), [1, 2, 3, 4])

    def test_q8_activation_dispatch_preserves_scalar_reduction(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "probe.c"
            library = Path(tmp) / "probe.so"
            source.write_text(
                '#include "packed_q8_activation_checked.h"\n'
                'float scalar_dot(const unsigned char *w, const unsigned char *x, size_t n) {\n'
                '    return ck_q8_activation_dot_scalar(w, x, n);\n}\n'
                'float dispatch_dot(const unsigned char *w, const unsigned char *x, size_t n) {\n'
                '    return ck_q8_activation_dot(w, x, n);\n}\n'
            )
            subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                            "-ffp-contract=off", "-I", str(root / "include"), str(source),
                            "-lm", "-o", str(library)], check=True)
            loaded = ctypes.CDLL(str(library))
            for fn in (loaded.scalar_dot, loaded.dispatch_dot):
                fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
                fn.restype = ctypes.c_float
            rng = np.random.default_rng(2319)
            for width in (32, 96, 1152, 2304, 3072):
                blocks = width // 32
                weight, activation = (np.empty((blocks, 34), dtype=np.uint8) for _ in range(2))
                for _ in range(100):
                    scales = rng.uniform(.001, .6, (2, blocks)).astype("<f2").view("u1").reshape(2, blocks, 2)
                    weight[:, :2], activation[:, :2] = scales
                    weight[:, 2:] = rng.integers(-128, 128, (blocks, 32), dtype=np.int16).astype("i1").view("u1")
                    activation[:, 2:] = rng.integers(-127, 128, (blocks, 32), dtype=np.int16).astype("i1").view("u1")
                    args = weight.ctypes.data, activation.ctypes.data, width
                    scalar = np.float32(loaded.scalar_dot(*args))
                    actual = np.float32(loaded.dispatch_dot(*args))
                    self.assertEqual(scalar.tobytes(), actual.tobytes())

    def test_checked_activation_rounds_ties_and_rejects_before_writes(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as tmp:
            library = Path(tmp) / "linear.so"
            subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                            "-I", str(root / "include"),
                            str(root / "src/kernels/linear_rows_q8_0_checked.c"),
                            "-lm", "-o", str(library)], check=True)
            probe = ctypes.CDLL(str(library)).linear_rows_q8_0_checked_q8_activation
            float_p = ctypes.POINTER(ctypes.c_float)
            byte_p = ctypes.POINTER(ctypes.c_ubyte)
            size = ctypes.c_size_t
            probe.argtypes = [float_p, size, byte_p, size, float_p, size, size, size, size, byte_p, size]
            probe.restype = ctypes.c_int
            activation = (ctypes.c_float * 32)(.5, 1.5, -1.5, -2.5, 127)
            weights = (ctypes.c_ubyte * (34 * 4))()
            for row in range(4):
                weights[row * 34 + 1] = 0x3C  # FP16 block scale 1.0.
                weights[row * 34 + 2 + row] = 1
            output = (ctypes.c_float * 4)(42, 42, 42, 42)
            scratch = (ctypes.c_ubyte * 34)()

            def run(capacity):
                return probe(activation, len(activation), weights, len(weights), output, len(output),
                             1, 32, 4, scratch, capacity)

            self.assertNotEqual(run(33), 0)
            self.assertEqual(list(output), [42] * 4)
            self.assertEqual(run(34), 0)
            self.assertEqual(list(output), [0, 2, -2, -2])
            activation[4] = 1e8
            self.assertNotEqual(run(34), 0)
            self.assertEqual(list(output), [0, 2, -2, -2])
            activation[4] = 127
            self.assertEqual(run(34), 0)


if __name__ == "__main__":
    unittest.main()
