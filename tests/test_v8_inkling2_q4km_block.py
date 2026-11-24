"""Asset-free composition tests for mixed packed Inkling decoder blocks."""

import ctypes
from itertools import product
import json
from pathlib import Path
import platform
import subprocess
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from tests.test_v8_inkling2_mini_causal_kv import entry
from version.v8.scripts.build_inkling2_decoder_block_v8 import (
    bind_mixed_decoder_weights, first_block_weights, first_decoder_block,
)
from tests.test_v8_inkling2_text_loop import CONFIG
from version.v8.scripts.build_inkling2_text_loop_v8 import resident_text_loop
from version.v8.scripts.checked_execution_exports_v8 import load_exports
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph
from version.v8.scripts.convert_gguf_to_bump_v8 import (
    GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
)


def packed_weights(rng, shape, dtype):
    """Write nonzero synthetic blocks with finite little-endian scales."""
    block_width, block_bytes = {
        "Q4_K": (256, 144), "Q6_K": (256, 210), "Q5_0": (32, 22), "Q8_0": (32, 34),
    }[dtype]
    count = int(np.prod(shape)) // block_width
    data = rng.integers(0, 256, (count, block_bytes), dtype=np.uint8)
    scale = np.array([0.002], dtype="<f2").view("u1")
    if dtype == "Q6_K":
        data[:, -18:-2] = 1
        data[:, -2:] = scale
    else:
        data[:, :2] = scale
        if dtype == "Q4_K":
            data[:, 2:4] = 0
            data[:, 4:16] = 1
    return data.ravel()


class MixedInklingBlockTests(unittest.TestCase):
    def test_q5_quantized_avx2_lane_reduction(self):
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            self.skipTest("AVX2 native dot regression requires x86 Linux")
        flags = next((line.split(":", 1)[1].split() for line in
                      Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("flags")), ())
        if not {"avx2", "fma", "f16c"}.issubset(flags):
            self.skipTest("AVX2, FMA and F16C execution unavailable")

        root = Path(__file__).resolve().parents[1]
        rng = np.random.default_rng(55408)
        rows, blocks = 5, 3
        codes = rng.integers(-16, 16, (rows, blocks, 32), dtype=np.int16)
        inputs = rng.integers(-127, 128, (blocks, 32), dtype=np.int16)
        scales = np.array([2**-8, -2**-10, 3 * 2**-11], dtype="<f2")
        input_scales = np.array([2**-5, 3 * 2**-7, 2**-6], dtype="<f2")
        encoded = (codes + 16).astype(np.uint8)
        weights = np.empty((rows, blocks, 22), dtype=np.uint8)
        weights[..., :2] = scales.view("u1").reshape(blocks, 2)
        weights[..., 2:6] = np.packbits(encoded >> 4, axis=-1, bitorder="little")
        weights[..., 6:] = (encoded[..., :16] & 15) | (encoded[..., 16:] << 4)
        activation = np.empty((blocks, 34), dtype=np.uint8)
        activation[:, :2] = input_scales.view("u1").reshape(blocks, 2)
        activation[:, 2:] = inputs.astype(np.int8).view("u1")

        lanes = np.zeros((rows, 8), dtype=np.float32)
        dots = (codes.astype(np.int32) * inputs).reshape(rows, blocks, 8, 4).sum(axis=-1)
        for block in range(blocks):
            # These binary scales and integer dots permit exact FP64 evaluation
            # of the product and sum before the single FP32 FMA rounding.
            scale = float(scales[block]) * float(input_scales[block])
            lanes[:] = lanes.astype(np.float64) + scale * dots[:, block]
        halves = lanes[:, :4] + lanes[:, 4:]
        pairs = halves[:, :2] + halves[:, 2:]
        expected = pairs[:, 0] + pairs[:, 1]

        with TemporaryDirectory() as temp:
            path = Path(temp) / "libq5-check.so"
            exports = Path(temp) / "exports.map"
            exports.write_text("{ global: gemv_q5_0_q8_0_checked_cpu; local: *; };")
            subprocess.run([
                "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                "-shared", "-fPIC", "-ffp-contract=off", "-ffunction-sections",
                "-Wl,--gc-sections", "-Wl,--no-undefined", f"-Wl,--version-script={exports}",
                "-I", str(root / "include"), str(root / "src/kernels/gemm_kernels_q5_0.c"),
                "-lm", "-o", str(path),
            ], check=True)
            gemv = ctypes.CDLL(str(path)).gemv_q5_0_q8_0_checked_cpu
            gemv.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2
            gemv.restype = None
            actual = np.full(rows, np.nan, dtype=np.float32)
            gemv(actual.ctypes.data, weights.ctypes.data, activation.ctypes.data, rows, blocks * 32)
            np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))

    def test_mixed_packed_block_shapes_and_continuation(self):
        rng = np.random.default_rng(1744)
        options = dict(width=256, kv_heads=1, query_heads=2, dim=128, relative_dim=2,
                       band=3, window=4, inner=32, experts=3, top_k=2, slots=3)
        tensor_types = {
            f"blk.0.{suffix}.weight": GGML_TYPE_Q4_K for suffix in
            ("attn_q", "attn_k", "attn_rel", "attn_sink_proj", "attn_gate", "attn_output", "ffn_gate_up_exps")
        }
        tensor_types["blk.0.attn_v.weight"] = GGML_TYPE_Q6_K
        hidden = rng.normal(0, .2, (3, 256)).astype("f4")
        for down, mode in product((GGML_TYPE_Q5_0, GGML_TYPE_Q8_0), ("ordered", "packed_hidden", "packed_down", "grouped", "grouped_pair", "pair_jobs")):
            packed = mode != "ordered"
            packed_down = mode in ("packed_down", "grouped", "grouped_pair", "pair_jobs")
            grouped = mode in ("grouped", "grouped_pair", "pair_jobs")
            paired = mode in ("grouped_pair", "pair_jobs")
            pair_jobs = mode == "pair_jobs"
            tensor_types["blk.0.ffn_down_exps.weight"] = down
            with self.subTest(down=down, activation_mode=mode), TemporaryDirectory() as temp:
                runs = {}
                for label, count in (("split", 2), ("full", 3)):
                    graph = bind_mixed_decoder_weights(first_decoder_block(prefill_tokens=count, **options),
                                                       tensor_types, q8_k_hidden=packed, q8_k_linear=packed,
                                                       q8_0_down=packed_down, q8_k_q6_linear=packed_down, group_experts=grouped, pair_expert_rows=paired, dispatch_expert_pairs=pair_jobs)
                    specs = first_block_weights(graph)
                    down_dtype = "Q5_0" if down == GGML_TYPE_Q5_0 else "Q8_0"
                    self.assertEqual(specs["blk.0.expert_bank.gate"][1], "Q4_K")
                    self.assertEqual(specs["blk.0.expert_bank.down"][1], down_dtype)
                    self.assertEqual(specs["blk.0.attn_v.weight"][1], "Q6_K")
                    if label == "split":
                        weights = {name: (rng.normal(0, .1, shape).astype("f4") if dtype == "F32"
                                          else packed_weights(rng, shape, dtype))
                                   for name, (shape, dtype) in specs.items()}
                    folder = Path(temp) / label
                    folder.mkdir()
                    _, arena, view, lib = compile_checked_weight_graph(folder, graph, weights, specs,
                                                                        extra_exports=("ck_set_num_threads",))
                    lib.ck_set_num_threads(4 if pair_jobs else 1)
                    calls = json.loads((folder / "call.json").read_text())
                    providers = {op["call_abi"]["kernel_id"] for op in calls["operations"]}
                    q6_provider = ("linear_rows_q6_k_q8_k_checked_f32" if packed_down
                                   else "linear_rows_q6_k_checked_f32")
                    self.assertIn(q6_provider, providers)
                    if packed:
                        self.assertIn("linear_rows_q4_k_q8_k_checked_f32", providers)
                    suffix = "_q8_k_hidden" if packed else ""
                    if packed_down:
                        suffix += "_q8_0_down"
                    if grouped:
                        group_suffix = "_pair_jobs" if pair_jobs else "_grouped_pair" if paired else "_grouped"
                        self.assertIn(f"routed_softcapped_experts_checked_q4_k_{down_dtype.lower()}{group_suffix}", providers)
                    self.assertIn(f"routed_softcapped_experts_checked_q4_k_{down_dtype.lower()}{suffix}", providers)
                    view("expert_slots", np.int32)[:] = [1, 2, 0]
                    view("router_bias")[:] = [.06, -.08, .12]
                    view("inkling2_prefill_slot0_hidden")[:] = hidden[:count]
                    self.assertEqual(entry(lib, "prefill", 0)(arena, len(arena)), 0)
                    if label == "split":
                        view("inkling2_append_slot0_hidden")[:] = hidden[count:]
                        self.assertEqual(entry(lib, "append", 0)(arena, len(arena)), 0)
                    runs[label] = view
                split, full = runs["split"], runs["full"]
                for stage in ("layer_output", "mixture", "query_raw"):
                    output = np.concatenate([split(f"inkling2_{phase}_slot0_{stage}")
                                             for phase in ("prefill", "append")])
                    expected = full(f"inkling2_prefill_slot0_{stage}")
                    self.assertTrue(np.isfinite(output).all())
                    self.assertGreater(float(np.max(np.abs(output))), 0)
                    np.testing.assert_array_equal(output, expected)
                for name in ("cache_k_0", "cache_v_0"):
                    np.testing.assert_array_equal(split(name), full(name)[:, :3])
                for kind in ("k_0", "v_0", "attn_post_conv", "ffn_post_conv"):
                    np.testing.assert_array_equal(split("history_" + kind), full("history_" + kind))


    def test_mixed_token_loop_embedding_output_and_continuation(self):
        rng = np.random.default_rng(451)
        cfg = {**CONFIG, "hidden_size": 256, "head_dim": 128, "swa_head_dim": 128}
        # Derive the fake GGUF directory from the circuit's existing logical names.
        logical = first_block_weights(resident_text_loop(cfg, 2, 8))
        types = {}
        for name, (_, dtype) in logical.items():
            if name.endswith("expert_bank.gate"):
                types[name.replace("expert_bank.gate", "ffn_gate_up_exps.weight")] = GGML_TYPE_Q4_K
            elif name.endswith("expert_bank.down"):
                gguf_name = name.replace("expert_bank.down", "ffn_down_exps.weight")
                types[gguf_name] = GGML_TYPE_Q5_0 if int(name.split(".")[1]) % 2 else GGML_TYPE_Q8_0
            elif dtype == "Q8_0" and not name.endswith("expert_bank.up"):
                types[name] = GGML_TYPE_Q6_K if name == "output.weight" else GGML_TYPE_Q4_K

        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            graph = resident_text_loop(cfg, 2, 8, tensor_types=types)
            specs = first_block_weights(graph)
            self.assertEqual(specs["token_embd.weight"][1], "Q4_K")
            self.assertEqual(specs["output.weight"][1], "Q6_K")
            weights = {name: (rng.normal(.02, .05, shape).astype("f4") if dtype == "F32"
                              else packed_weights(rng, shape, dtype)) for name, (shape, dtype) in specs.items()}
            _, arena, view, lib = compile_checked_weight_graph(folder, graph, weights, specs)
            exports = load_exports(folder, lib)
            self.assertEqual(exports["properties"]["weight_format"], "q4_k_m")
            calls = json.loads((folder / "call.json").read_text())
            providers = {op["call_abi"]["kernel_id"] for op in calls["operations"]}
            self.assertIn("embedding_lookup_q4_k_checked_f32", providers)
            self.assertIn("linear_rows_q6_k_checked_f32", providers)
            view("expert_slots", np.int32)[:] = np.arange(cfg["n_routed_experts"])
            for layer in range(cfg["num_hidden_layers"]):
                view(f"router_bias_{layer}")[:] = [.1, .2, .3, .4]

            def run(phase, position, ids):
                view(f"{phase}_request_position", np.int32)[:] = position
                view(f"{phase}_token_ids", np.int32)[:] = ids
                fn = getattr(lib, exports["entries"][phase])
                fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
                fn.restype = ctypes.c_int
                self.assertEqual(fn(arena, len(arena)), 0)
                return view(f"{phase}_logits").copy()

            run("prefill", 0, [2, 8])
            state_names = [name for name in graph["activation_buffers"]
                           if name.startswith(("cache_", "history_"))]
            snapshot = {name: view(name).copy() for name in state_names}
            expected = run("prefill", 2, [3, 7])
            end_state = {name: view(name).copy() for name in state_names}
            for name, values in snapshot.items():
                view(name)[:] = values
            actual = np.concatenate((run("append", 2, [3]), run("append", 3, [7])))
            self.assertTrue(np.isfinite(actual).all())
            self.assertGreater(float(np.ptp(actual)), 0.0)
            np.testing.assert_array_equal(actual, expected)
            for name, values in end_state.items():
                np.testing.assert_array_equal(view(name), values)


if __name__ == "__main__":
    unittest.main()
