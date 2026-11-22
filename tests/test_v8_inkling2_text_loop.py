import ctypes
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from tests.test_v8_inkling2_routed_moe import pack_quant
from version.v8.scripts.build_inkling2_decoder_block_v8 import first_block_weights
from version.v8.scripts.build_inkling2_text_loop_v8 import resident_text_loop, text_token_loop
from version.v8.scripts.checked_execution_exports_v8 import load_exports
from version.v8.scripts.replay_inkling2_text_loop_v8 import capture_layer_rows
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph


CONFIG = {
    "num_hidden_layers": 24, "n_layers_per_attn_res_block": 6,
    "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 4, "is_varloop": False},
    "local_layer_ids": [i for i in range(24) if i % 6 != 5],
    "hidden_size": 32, "n_routed_experts": 4, "num_experts_per_tok": 2, "intermediate_size": 32,
    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
    "swa_num_attention_heads": 2, "swa_num_key_value_heads": 1, "swa_head_dim": 16,
    "d_rel": 2, "rel_extent": 3, "sliding_window_size": 4, "sconv_kernel_size": 4,
    "vocab_size": 24, "final_logit_softcapping": 30.0,
    "rms_norm_eps": 1e-6, "log_scaling_n_floor": 2, "log_scaling_alpha": .1,
}


class InklingTextLoopTests(unittest.TestCase):
    def test_native_full_schedule_split_equivalence(self):
        rng = np.random.default_rng(516)
        ids = np.array([5, 11, 3, 14, 5], dtype=np.int32)
        views = {}
        with TemporaryDirectory() as directory:
            for label, count in (("split", 2), ("last", 2), ("full", 5)):
                graph = (text_token_loop(CONFIG, count, f64_norm_sum=True, reference_moe=True, fp32_static_conv=True) if label == "full"
                         else resident_text_loop(CONFIG, count, 6, f64_norm_sum=True,
                                                 last_prefill_logit=label == "last", reference_moe=True, fp32_static_conv=True))
                specs = first_block_weights(graph)
                if label == "split":
                    weights = {
                        name: pack_quant(rng.integers(-2, 3, shape, dtype=np.int8), "q8_0")
                        if dtype == "Q8_0" else rng.normal(.05, .2, shape).astype(np.float32)
                        for name, (shape, dtype) in specs.items()
                    }
                    schedule = graph["contract"]["runtime_invariants"]["layer_schedule"]
                    self.assertEqual(len(schedule), 48)
                    self.assertEqual(len({x["state_slot"] for x in schedule}), 48)
                    self.assertEqual(len({x["weight_layer"] for x in schedule}), 24)

                folder = Path(directory) / label
                folder.mkdir()
                _, arena, view, lib = compile_checked_weight_graph(
                    folder, graph, weights, specs, openmp=label == "last")
                if label == "last":
                    lib.omp_set_dynamic(0)
                    lib.omp_set_num_threads(2)
                exports = load_exports(folder, lib)
                self.assertEqual(exports["properties"]["supports_explicit_position"], label != "full")
                self.assertEqual(exports["ports"]["prefill_token_ids"]["dtype"], "i32")
                logit_rows = 1 if label == "last" else count
                self.assertEqual(exports["ports"]["prefill_logits"]["shape"], [logit_rows, CONFIG["vocab_size"]])
                self.assertEqual(exports["properties"]["expert_accumulation"], "router_slot")
                self.assertEqual(exports["properties"]["static_conv_accumulator"], "fp32")
                self.assertEqual(exports["properties"]["prefill_logits"],
                                 "last_row" if label == "last" else "all_rows")
                for layer in range(CONFIG["num_hidden_layers"]):
                    view(f"router_bias_{layer}")[:] = [-.02, .04, .08, .01]
                view("expert_slots", np.int32)[:] = np.arange(CONFIG["n_routed_experts"])

                def run(phase, rows, expected=0, position=None):
                    if position is not None:
                        view(f"{phase}_request_position", np.int32)[:] = position
                    view(f"{phase}_token_ids", np.int32)[:] = rows
                    fn = getattr(lib, exports["entries"][phase])
                    fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
                    fn.restype = ctypes.c_int
                    self.assertEqual(fn(arena, len(arena)), expected)

                if label == "full":
                    invalid = ids.copy()
                    invalid[-1] = CONFIG["vocab_size"]
                    run("prefill", invalid, expected=-3)
                    np.testing.assert_array_equal(view("cache_k_0"), 0)
                    np.testing.assert_array_equal(view("prefill_logits"), 0)
                if label != "full":
                    run("prefill", ids[:count], position=0)
                    initial = view("prefill_logits").copy()
                    history = view("history_attn_post_conv_0").copy()
                    run("prefill", ids[count:4], expected=-2, position=5)
                    np.testing.assert_array_equal(view("history_attn_post_conv_0"), history)
                    np.testing.assert_array_equal(view("prefill_logits"), initial)
                    run("prefill", ids[count:4], position=2)
                    continuation = view("prefill_logits").copy()
                    run("append", ids[4:], position=4)
                    if label == "split":
                        first, second = initial, continuation
                    else:
                        np.testing.assert_array_equal(initial[0].view("u4"), first[-1].view("u4"))
                        np.testing.assert_array_equal(continuation[0].view("u4"), second[-1].view("u4"))
                        archive = folder / "xray"
                        capture_layer_rows(archive, graph, view, "append", 4, 0)
                        index = json.loads((archive / "index.json").read_text())
                        self.assertEqual(index["position"], 4)
                        self.assertEqual([row["state_slot"] for row in index["checkpoints"]],
                                         [row["state_slot"] for row in schedule])
                        row = index["checkpoints"][18]
                        self.assertEqual(row["state_slot"], 24)
                        np.testing.assert_array_equal(
                            np.fromfile(archive / row["filename"], dtype="<f4"),
                            view(row["tensor"])[0])
                else:
                    run("prefill", ids)
                    embedding = view("prefill_embeddings")
                    mean = (np.sum(embedding * embedding, axis=1, dtype=np.float64) / 32).astype(np.float32)
                    scale = np.float32(1) / np.sqrt(mean + np.float32(CONFIG["rms_norm_eps"]))
                    norm = embedding * scale[:, None] * weights["token_embd_norm.weight"]
                    np.testing.assert_array_equal(view("prefill_hidden"), norm)
                views[label] = view

            split, full = views["split"], views["full"]
            actual = np.concatenate((first, second, split("append_logits")))
            self.assertTrue(np.isfinite(actual).all())
            self.assertGreater(float(np.max(np.abs(actual))), .05)
            self.assertEqual(actual.shape, (5, CONFIG["vocab_size"]))
            np.testing.assert_allclose(actual, 30 * np.tanh(full("prefill_raw_logits") / 30), atol=1e-6)
            np.testing.assert_array_equal(actual, full("prefill_logits"))
            last = views["last"]
            np.testing.assert_array_equal(last("append_logits"), split("append_logits"))
            for slot in range(48):
                for kind in ("k", "v"):
                    name = f"cache_{kind}_{slot}"
                    np.testing.assert_array_equal(split(name)[:, :5, :], full(name)[:, :5, :], err_msg=name)
                    np.testing.assert_array_equal(last(name), split(name), err_msg=name)
                for kind in ("k", "v", "attn_post_conv", "ffn_post_conv"):
                    name = f"history_{kind}_{slot}"
                    np.testing.assert_array_equal(split(name), full(name), err_msg=name)
                    np.testing.assert_array_equal(last(name), split(name), err_msg=name)
            self.assertFalse(np.array_equal(full("cache_k_6"), full("cache_k_24")))
            self.assertFalse(np.array_equal(full("cache_k_24"), full("cache_k_36")))


if __name__ == "__main__":
    unittest.main()
