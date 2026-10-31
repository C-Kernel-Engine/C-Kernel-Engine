#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODEGEN_PREFILL_PATH = ROOT / "version" / "v8" / "scripts" / "codegen_prefill_v8.py"


def _load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


codegen_prefill_v8 = _load_module("codegen_prefill_v8_tests", CODEGEN_PREFILL_PATH)
codegen_v8 = _load_module(
    "codegen_v8_prefill_phase_tests",
    ROOT / "version" / "v8" / "scripts" / "codegen_v8.py",
)
codegen_core_v8 = _load_module(
    "codegen_core_v8_prefill_append_tests",
    ROOT / "version" / "v8" / "scripts" / "codegen_core_v8.py",
)
chunked_compare = _load_module(
    "compare_chunked_prefill_v8_tests",
    ROOT / "version" / "v8" / "scripts" / "compare_chunked_prefill_v8.py",
)


class TestV8PrefillCodegen(unittest.TestCase):
    def test_layer_progress_is_emitted_after_real_layer_operations(self):
        ops = [
            {"op": "first", "function": "first", "layer": 0, "args": []},
            {"op": "second", "function": "second", "layer": 0, "args": []},
            {"op": "third", "function": "third", "layer": 1, "args": []},
            {"op": "logits", "function": "finish", "layer": -1, "args": []},
        ]
        emitted = codegen_prefill_v8.emit_prefill_function(
            ops, {"context_length": 16, "prefill_chunk_length": 16})
        first = emitted.index("first(")
        second = emitted.index("second(")
        one_done = emitted.index("ck_model_progress_report(1, 2, num_tokens, prefill_start_pos)")
        third = emitted.index("third(")
        two_done = emitted.index("ck_model_progress_report(2, 2, num_tokens, prefill_start_pos)")
        self.assertLess(first, second)
        self.assertLess(second, one_done)
        self.assertLess(one_done, third)
        self.assertLess(third, two_done)
        self.assertIn("ck_model_progress_report(0, 2, num_tokens, prefill_start_pos)", emitted)
        self.assertEqual(emitted.count("ck_model_progress_report("), 3)
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("C compiler unavailable")
        harness = r'''
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define MAX_SEQ_LEN 16
#define A_TOKEN_IDS 0
typedef struct { unsigned char bump[64]; int pos, rope_pos, encoder_kv_ready; } CKModel;
static int g_ck_skip_decode_logits, completed[3], total[3], chunk[3], start[3];
static int callbacks, operations;
static int ck_model_cancel_requested(void) { return 0; }
static void ck_model_progress_report(int a, int b, int c, int d) {
    if (callbacks < 3) {
        completed[callbacks] = a; total[callbacks] = b;
        chunk[callbacks] = c; start[callbacks] = d;
    }
    callbacks++;
}
static void ck_trace_pos(const char *a, int b, int c, int d, int e) {
    (void)a; (void)b; (void)c; (void)d; (void)e;
}
static void ck_parallel_prefill_release_transient_caches(void) {}
static int ck_env_truthy_or_qwen3vl_ocr_profile(const char *x) {
    (void)x; return 0;
}
static void ck_decode(CKModel *m, int32_t t) { (void)t; m->pos++; }
static void first(void) { operations++; }
static void second(void) { operations++; }
static void third(void) { operations++; }
static void finish(void) { operations++; }
''' + emitted + r'''
int main(void) {
    CKModel model = {0};
    int32_t ids[2] = {1, 2};
    ck_prefill(&model, ids, 2);
    if (model.pos != 2 || operations != 4 || callbacks != 3) return 1;
    for (int i = 0; i < 3; ++i)
        if (completed[i] != i || total[i] != 2 || chunk[i] != 2 || start[i] != 0)
            return 2;
    return 0;
}
'''
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            source = Path(directory) / "progress.c"
            executable = Path(directory) / "progress"
            source.write_text(harness)
            subprocess.run([compiler, "-std=c11", str(source), "-o", str(executable)],
                           check=True, capture_output=True, text=True)
            subprocess.run([str(executable)], check=True)
        interrupted = codegen_prefill_v8.emit_prefill_function(
            [ops[0], ops[-1], ops[1]],
            {"context_length": 16, "prefill_chunk_length": 16})
        self.assertNotIn("ck_model_progress_report(", interrupted)

    def test_prior_cache_attention_keeps_chunked_prefill_with_gate_projection(self):
        config = {"context_length": 32, "prefill_chunk_length": 8, "embed_dim": 16}
        ops = [
            {"op": "attn_sliding", "function": "append_attention", "layer": 0,
             "prefill_history_scope": "prior_cache", "args": []},
            {"op": "attention_gate_projection", "function": "gate_projection",
             "layer": 0, "args": []},
        ]
        emitted = codegen_prefill_v8.emit_prefill_function(ops, config)
        self.assertNotIn("Continue through decode", emitted)
        self.assertIn("ck_prefill_range(model, tokens + consumed, chunk, start_pos + consumed)", emitted)
        unknown = codegen_prefill_v8.emit_prefill_function(
            ops + [{"op": "attention_future", "function": "future_attention",
                    "layer": 1, "args": []}], config)
        self.assertIn("Continue through decode", unknown)
        ops[0]["prefill_history_scope"] = "current_chunk"
        fallback = codegen_prefill_v8.emit_prefill_function(ops, config)
        self.assertIn("Continue through decode", fallback)

    def test_selected_prefill_row_groups_define_checked_append_boundary(self):
        self.assertEqual(codegen_v8._prefill_row_group_alignment([]), 1)
        self.assertEqual(codegen_v8._prefill_row_group_alignment([
            {"prefill_row_group": 4}, {"prefill_row_group": 8}, {}]), 8)
        for bad in (0, -1, 1.5, "4", True):
            with self.subTest(bad=bad), self.assertRaisesRegex(RuntimeError, "row group"):
                codegen_v8._prefill_row_group_alignment(
                    [{"prefill_row_group": bad}])

    def test_grouped_append_rejects_before_mutation_and_uses_prefill_continuation(self):
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("C compiler unavailable")
        source = codegen_core_v8.emit_model_and_api(
            layout={"memory": {"activations": {"buffers": []}}},
            config={"prefill_append_row_group": 4},
        )
        ordinary = codegen_core_v8.emit_model_and_api(
            layout={"memory": {"activations": {"buffers": []}}}, config={},
        )
        self.assertIn("if (count > 1 &&", ordinary)
        self.assertNotIn("a prior tail reduction cannot be regrouped", ordinary)
        for bad in (0, -4, 1.5, "4", True):
            with self.subTest(bad=bad), self.assertRaisesRegex(
                    RuntimeError, "row group must be positive"):
                codegen_core_v8.emit_model_and_api(
                    layout={"memory": {"activations": {"buffers": []}}},
                    config={"prefill_append_row_group": bad},
                )
        api = source[
            source.index("CK_EXPORT int ck_model_prefill_append_row_group(void)"):
            source.index("/* Forward pass (after embed_tokens) */")
        ]
        harness = r'''
#include <stdint.h>
#include <stdlib.h>
#define CK_EXPORT
#define CK_HAS_PREFILL 1
typedef struct { int pos; int state; } CKModel;
static CKModel storage;
static CKModel *g_model = &storage;
static int g_ck_skip_decode_logits;
static int prefill_calls, decode_calls;
static int ck_model_cancel_requested(void) { return 0; }
static void ck_trace_pos(const char *stage, int token, int count, int before, int after) {
    (void)stage; (void)token; (void)count; (void)before; (void)after;
}
static void ck_parallel_prefill_release_transient_caches(void) {}
static void ck_prefill(CKModel *model, const int32_t *tokens, int count) {
    (void)tokens;
    ++prefill_calls;
    model->pos += count;
    model->state += count;
}
static void ck_decode(CKModel *model, int32_t token) {
    (void)token;
    ++decode_calls;
    ++model->pos;
    ++model->state;
}
''' + api + r'''
int main(void) {
    const int32_t token = 7;
    if (ck_model_prefill_append_row_group() != 4) return 1;
    storage.pos = 127; storage.state = 91;
    if (ck_model_embed_tokens(&token, 1) != -3 || storage.pos != 127 ||
        storage.state != 91 || prefill_calls || decode_calls) return 2;
    if (ck_model_embed_tokens(&token, 2) != -3 || storage.state != 91 ||
        prefill_calls || decode_calls) return 3;
    storage.pos = 124;
    if (ck_model_embed_tokens(&token, 1) != 0 || storage.pos != 125 ||
        prefill_calls != 1 || decode_calls) return 4;
    storage.pos = 0;
    if (ck_model_embed_tokens(&token, 1) != 0 || storage.pos != 1 ||
        prefill_calls != 1 || decode_calls != 1) return 5;
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="prefill_append_", dir=os.environ.get("TMPDIR")) as directory:
            path = Path(directory)
            (path / "test.c").write_text(harness, encoding="utf-8")
            subprocess.run([compiler, "-std=c11", "-Wall", "-Wextra", "-Werror",
                            str(path / "test.c"), "-o", str(path / "test")],
                           check=True, capture_output=True, text=True)
            subprocess.run([str(path / "test")], check=True)

    def test_execution_segments_cover_repeated_append_and_nonzero_position(self):
        tokens = list(range(10))
        self.assertEqual(chunked_compare._segments(tokens, None, 4), [
            (0, [0, 1, 2, 3]), (4, [4, 5, 6, 7]), (8, [8, 9])])
        self.assertEqual(chunked_compare._segments(tokens, 5, 3), [
            (0, [0, 1, 2]), (3, [3, 4]), (5, [5, 6, 7]), (8, [8, 9])])
        with self.assertRaisesRegex(ValueError, "positive"):
            chunked_compare._segments(tokens, None, 0)

    def test_execution_comparison_excludes_unexported_recurrent_state(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            layout = bundle / "layout_decode.json"
            def write(names):
                layout.write_text(json.dumps({"memory": {"activations": {"buffers": [
                    {"name": name, "lifetime": "sequence", "mutable": True}
                    for name in names]}}}), encoding="utf-8")
            write(["kv_cache"])
            self.assertEqual(chunked_compare._kv_only_state_contract(bundle)[
                "exported_state"], ["kv_cache"])
            write(["kv_cache", "recurrent_ssm_state"])
            with self.assertRaisesRegex(RuntimeError, "unexported persistent"):
                chunked_compare._kv_only_state_contract(bundle)
            layout.write_text(json.dumps({"memory": {"activations": {"buffers": [
                {"name": "kv_cache", "lifetime": "sequence", "mutable": True},
                {"name": "recurrent_ssm_state", "lifetime": "sequence",
                 "mutable": True, "shape": "[4, 0, 0]"},
            ]}}}), encoding="utf-8")
            for mode in ("decode", "prefill"):
                (bundle / f"lowered_{mode}_call.json").write_text(
                    json.dumps({"operations": []}), encoding="utf-8")
            self.assertEqual(chunked_compare._kv_only_state_contract(bundle)[
                "exported_state"], ["kv_cache"])
            (bundle / "lowered_prefill_call.json").write_text(json.dumps({
                "operations": [{"op": "state_update", "buffer_ref": "recurrent_ssm_state"}]
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "unexported persistent"):
                chunked_compare._kv_only_state_contract(bundle)

    def test_loaded_library_identity_rejects_replaced_backing(self):
        if not Path("/proc/self/maps").exists():
            self.skipTest("Linux process mappings required")
        system = ctypes.CDLL("libm.so.6")
        address = ctypes.cast(system.cos, ctypes.c_void_p).value
        system_path = None
        for row in Path("/proc/self/maps").read_text().splitlines():
            fields = row.split(maxsplit=5)
            if len(fields) == 6 and fields[5].startswith("/"):
                lo, hi = (int(part, 16) for part in fields[0].split("-", 1))
                if lo <= address < hi:
                    system_path = Path(fields[5])
                    break
        self.assertIsNotNone(system_path)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "libmodel.so"
            shutil.copy2(system_path, target)
            loaded = ctypes.CDLL(str(target))
            actual = chunked_compare._loaded_symbol_digest(
                loaded, "cos", target,
                Path("/proc/self/maps").read_text())
            self.assertEqual(actual, chunked_compare._sha256(target))
            replacement = target.with_suffix(".replacement")
            shutil.copy2(system_path, replacement)
            os.replace(replacement, target)
            with self.assertRaisesRegex(RuntimeError, "path mismatch|backing changed"):
                chunked_compare._loaded_symbol_digest(
                    loaded, "cos", target,
                    Path("/proc/self/maps").read_text())

    def test_chunked_comparison_detects_logit_and_kv_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left, right = root / "left", root / "right"
            left.mkdir()
            right.mkdir()
            np.array([1.0, 2.0], dtype="<f4").tofile(left / "prefill.f32")
            np.array([1.0, 3.0], dtype="<f4").tofile(right / "prefill.f32")
            a = {"position": 2, "top_token": 1, "kv_sha256": ["a" * 64, "b" * 64]}
            b = {"position": 2, "top_token": 1, "kv_sha256": ["a" * 64, "c" * 64]}
            result = chunked_compare._comparison(left, right, "prefill", a, b)
            self.assertFalse(result["logits_exact"])
            self.assertEqual(result["max_logit_abs"], 1.0)
            self.assertEqual(result["kv_exact_layers"], 1)
            self.assertEqual(result["kv_mismatch_layers"], [1])
            shifted = chunked_compare._comparison(
                left, right, "prefill", a, dict(b, position=3))
            self.assertEqual(shifted["xray"]["classification"], "POSITION_STATE_MISMATCH")
            np.array([0.0], dtype="<f4").tofile(left / "prefill.f32")
            np.array([-0.0], dtype="<f4").tofile(right / "prefill.f32")
            self.assertFalse(chunked_compare._comparison(
                left, right, "prefill", a, a)["logits_exact"])

    def test_prefill_phase_decode_uses_only_matching_map_declared_provider(self):
        group = {"provider_selection": {"equivalence_group": "attention.test.v1"}}
        prefill = [{
            "op": "attn_sliding", "layer": 0, "instance": 0,
            "prefill_history_scope": "current_chunk",
            "numerical_reduction_order": "padded_256",
            "resolved_execution": group,
        }]
        decode = [
            {"op": "dense_embedding_lookup", "layer": -1, "function": "embed"},
            {
                "op": "attn_sliding", "layer": 0, "instance": 0,
                "function": "attention_decode",
                "prefill_continuation": {
                    "function": "attention_decode_prefill_phase",
                    "reduction_order": "padded_256",
                },
                "resolved_execution": group,
            },
        ]
        selected = codegen_v8._prefill_phase_decode_ops(decode, prefill)
        self.assertEqual(
            [op["function"] for op in selected],
            ["embed", "attention_decode_prefill_phase"],
        )
        self.assertEqual(decode[1]["function"], "attention_decode")
        self.assertIsNone(codegen_v8._prefill_phase_decode_ops(
            [decode[0], {k: v for k, v in decode[1].items()
                         if k != "prefill_continuation"}], prefill))
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            codegen_v8._prefill_phase_decode_ops(
                decode, [dict(prefill[0], numerical_reduction_order="other")])
        self.assertIsNone(codegen_v8._prefill_phase_decode_ops(
            decode, [dict(prefill[0], layer=1)]))

    def test_prefill_projection_requires_matching_weight_shape_and_provider(self):
        group = {"provider_selection": {"equivalence_group": "attention.test.v1"}}
        prefill_attention = {
            "op": "attn", "layer": 2, "instance": 0,
            "prefill_history_scope": "current_chunk",
            "numerical_reduction_order": "online_query",
            "resolved_execution": group,
        }
        decode_attention = {
            "op": "attn", "layer": 2, "instance": 0,
            "function": "decode_attention",
            "prefill_continuation": {
                "function": "prefill_attention", "reduction_order": "online_query"},
            "resolved_execution": group,
        }
        prefill_projection = {
            "op": "mlp_down", "layer": 2, "instance": 0,
            "prefill_row_reduction": "packed_superblock.v1",
            "call_abi": {"kernel_id": "gemm_q4k"},
            "args": [
                {"name": "A", "source": "activation:a", "buffer_ref": "q8_rows",
                 "expr": "(const void*)(model->bump + A_Q8_ROWS)"},
                {"name": "B", "source": "weight:_first_weight",
                 "weight_ref": "layer.2.w2"},
                {"name": "bias", "source": "weight_f:_bias",
                 "weight_ref": "layer.2.b2",
                 "expr": "(float*)(model->bump + W_LAYER_2_B2)"},
                {"name": "C", "source": "output:c", "buffer_ref": "output",
                 "expr": "(float*)(model->bump + A_OUTPUT)"},
                {"name": "N", "expr": "896"}, {"name": "K", "expr": "4864"},
            ],
        }
        decode_projection = {
            "op": "mlp_down", "layer": 2, "instance": 0,
            "function": "ordinary_gemv",
            "prefill_continuation": {
                "function": "prefill_order_gemv",
                "tail_function": "prefill_tail_gemv",
                "prefill_kernel_id": "gemm_q4k",
                "reduction_order": "packed_superblock.v1",
            },
            "args": [
                {"name": "y", "source": "output:y", "buffer_ref": "output",
                 "expr": "(float*)(model->bump + A_OUTPUT)"},
                {"name": "W", "source": "weight:_first_weight",
                 "weight_ref": "layer.2.w2"},
                {"name": "x_q8", "source": "activation:x",
                 "buffer_ref": "q8_rows",
                 "expr": "(void*)(model->bump + A_Q8_ROWS)"},
                {"name": "M", "expr": "896"}, {"name": "K", "expr": "4864"},
            ],
        }
        prefill = [prefill_attention, prefill_projection]
        decode = [decode_attention, decode_projection]
        selected = codegen_v8._prefill_phase_decode_ops(decode, prefill)
        self.assertEqual([op["function"] for op in selected],
                         ["prefill_attention", "prefill_order_gemv"])
        self.assertEqual([arg["name"] for arg in selected[1]["args"]],
                         ["y", "W", "x_q8", "bias", "M", "K"])
        tail = codegen_v8._prefill_phase_decode_ops(
            decode, prefill, tail_reduction=True)
        self.assertEqual([op["function"] for op in tail],
                         ["prefill_attention", "prefill_tail_gemv"])
        self.assertEqual([arg["name"] for arg in tail[1]["args"]],
                         ["y", "W", "x_q8", "bias", "M", "K"])
        with self.assertRaisesRegex(RuntimeError, "no declared tail provider"):
            codegen_v8._prefill_phase_decode_ops(
                [decode_attention, {**decode_projection,
                                    "prefill_continuation": {
                                        k: v for k, v in decode_projection["prefill_continuation"].items()
                                        if k != "tail_function"}}],
                prefill, tail_reduction=True)
        selected[1]["resolved_execution"] = {
            "numerical_contract": "q4_k_x_q8_k",
            "implementation": {
                "weight_storage": {"format": "q4_k", "block_elements": 256,
                                   "block_bytes": 144},
                "activation_storage": {"format": "q8_k", "block_elements": 256},
                "diagnostic_providers": {"fp32_activation": "diagnostic_fp32"},
            },
        }
        emitted = codegen_v8.codegen_core_v8.emit_op(selected[1])
        self.assertIn("prefill_order_gemv(", emitted)
        self.assertIn("(float*)(model->bump + W_LAYER_2_B2),", emitted)
        without_bias = dict(prefill_projection, args=[
            arg for arg in prefill_projection["args"] if arg["name"] != "bias"])
        no_bias = codegen_v8._prefill_phase_decode_ops(
            decode, [prefill_attention, without_bias])
        self.assertEqual(no_bias[1]["args"][3]["expr"], "NULL")
        self.assertEqual(decode_projection["function"], "ordinary_gemv")
        for changed, expected in (
            ({"call_abi": {"kernel_id": "other"}}, "provider mismatch"),
            ({"prefill_row_reduction": "other"}, "provider mismatch"),
            ({"args": [*prefill_projection["args"][:1],
                       dict(prefill_projection["args"][1],
                            weight_ref="layer.3.w2"),
                       *prefill_projection["args"][2:]]}, "layout mismatch"),
            ({"args": [*prefill_projection["args"][:-1],
                       {"name": "K", "expr": "5120"}]}, "layout mismatch"),
            ({"args": [dict(prefill_projection["args"][0],
                            buffer_ref="other_rows"),
                       *prefill_projection["args"][1:]]}, "layout mismatch"),
            ({"args": [dict(prefill_projection["args"][0],
                            expr="(const void*)(model->bump + A_Q8_ROWS + 4)"),
                       *prefill_projection["args"][1:]]}, "layout mismatch"),
            ({"args": [*prefill_projection["args"][:2],
                       *prefill_projection["args"][2:3],
                       dict(prefill_projection["args"][3],
                            buffer_ref="other_output"),
                       *prefill_projection["args"][4:]]}, "layout mismatch"),
            ({"args": [*prefill_projection["args"][:2],
                       dict(prefill_projection["args"][2],
                            expr="(float*)(model->bump + W_LAYER_2_B2 + 4)"),
                       *prefill_projection["args"][3:]]}, "bias binding mismatch"),
        ):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(RuntimeError, expected):
                    codegen_v8._prefill_phase_decode_ops(
                        decode, [prefill_attention,
                                 {**prefill_projection, **changed}])

    def test_grouped_projection_aligns_only_intermediate_first_chunk(self):
        op = {
            "function": "noop_kernel", "op": "attention", "layer": 0,
            "args": [], "prefill_history_scope": "current_chunk",
        }
        emitted = codegen_prefill_v8.emit_prefill_function(
            [op], {"context_length": 32, "prefill_chunk_length": 7,
                   "prefill_initial_chunk_alignment": 4, "embed_dim": 4}
        )
        self.assertIn("if (consumed == 0 && start_pos == 0 && num_tokens > chunk)",
                      emitted)
        self.assertIn("chunk -= chunk % 4;", emitted)
        self.assertIn("consumed += chunk;", emitted)
        with_tail = codegen_prefill_v8.emit_prefill_function(
            [op], {"context_length": 32, "prefill_chunk_length": 7,
                   "prefill_initial_chunk_alignment": 4, "embed_dim": 4,
                   "prefill_continuation_entry": "ck_decode_prefill_continuation",
                   "prefill_tail_continuation_entry": "ck_decode_prefill_tail_continuation"})
        self.assertIn("((start_pos + num_tokens) / 4) * 4", with_tail)
        self.assertIn("ck_decode_prefill_tail_continuation(model", with_tail)
        with self.assertRaisesRegex(RuntimeError, "row group"):
            codegen_prefill_v8.emit_prefill_function(
                [op], {"context_length": 32, "prefill_chunk_length": 3,
                       "prefill_initial_chunk_alignment": 4, "embed_dim": 4}
            )

    def test_generated_embedded_prefill_scales_only_declared_text_rows(self):
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("C compiler unavailable")
        embedding = {"op": "dense_embedding_lookup", "layer": -1}
        mixed_schedule = {
            "segments": ["text_before", "visual", "text_after"],
            "cache_transition": "single_pass",
            "position_transition": "explicit_full_sequence",
        }
        segmented_schedule = dict(
            mixed_schedule,
            cache_transition="append_preserve",
            position_transition="segment_defined",
        )
        cases = (
            ("unified", {"prefill_batching": "unified_mixed", "prefill_schedule": mixed_schedule}),
            ("segmented", {"prefill_batching": "segmented_append", "prefill_schedule": segmented_schedule}),
            ("text", None),
        )
        for name, bridge in cases:
            with self.subTest(name=name):
                config = {"embed_dim": 4, "scale_embeddings_sqrt_dim": True}
                if bridge is not None:
                    config["multimodal_bridge_contract"] = dict(
                        bridge, prefix_policy="mixed_visual_text_prefill"
                    )
                emitted = codegen_prefill_v8.emit_prefill_from_embedded_function(
                    [embedding], config
                )
                if name == "unified":
                    self.assertIn("ck_multimodal_prefill_has_segment_plan(rows)", emitted)
                elif name == "segmented":
                    self.assertIn("bridge_noncausal_visual_chunk &&", emitted)
                    self.assertNotIn("ck_multimodal_prefill_has_segment_plan(rows)", emitted)
                else:
                    self.assertNotIn("ck_multimodal_prefill_has_segment_plan(rows)", emitted)
                source = r'''
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define EMBED_DIM 4
#define A_EMBEDDED_INPUT 0
#define MAX_SEQ_LEN 16
typedef struct { unsigned char *bump; int pos; int rope_pos; } CKModel;
static int ck_env_truthy_or_qwen3vl_ocr_profile(const char *name) {
    (void)name;
    return 0;
}
__EMITTED__
static int check(CKModel *model, float *data, int rows, int before, int visual) {
    for (int i = 0; i < rows * EMBED_DIM; ++i) data[i] = (float)(i + 1);
    model->pos = model->rope_pos = 0;
    ck_prefill_from_embedded(model, rows);
    if (model->pos != rows) return 1;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < EMBED_DIM; ++col) {
            const int i = row * EMBED_DIM + col;
            const float factor = row >= before && row < before + visual ? 1.0f : 2.0f;
            if (data[i] != (float)(i + 1) * factor) return 2;
        }
    }
    return 0;
}
int main(void) {
    float data[MAX_SEQ_LEN * EMBED_DIM] = {0};
    CKModel model = {(unsigned char *)data, 0, 0};
    int rc;
__CASES__
    return 0;
}
'''
                if name == "unified":
                    checks = r'''
    const int plans[][4] = {{0, 2, 1, 3}, {2, 2, 0, 4}, {0, 2, 0, 2}};
    for (int i = 0; i < 3; ++i) {
        g_multimodal_prefill_segment_lengths[0] = plans[i][0];
        g_multimodal_prefill_segment_lengths[1] = plans[i][1];
        g_multimodal_prefill_segment_lengths[2] = plans[i][2];
        g_multimodal_prefill_num_segments = 3;
        rc = check(&model, data, plans[i][3], plans[i][0], plans[i][1]);
        if (rc) return 10 + i;
    }
    setenv("CK_BRIDGE_NONCAUSAL_VISUAL_CHUNK", "1", 1);
    g_multimodal_prefill_segment_lengths[0] = 2;
    g_multimodal_prefill_segment_lengths[1] = 2;
    g_multimodal_prefill_segment_lengths[2] = 0;
    rc = check(&model, data, 4, 2, 2);
    if (rc) return 13;
    unsetenv("CK_BRIDGE_NONCAUSAL_VISUAL_CHUNK");
    g_multimodal_prefill_num_segments = 0;
    rc = check(&model, data, 3, 0, 0);
    if (rc) return 20;
    g_multimodal_prefill_segment_lengths[0] = 1;
    g_multimodal_prefill_segment_lengths[1] = 2;
    g_multimodal_prefill_segment_lengths[2] = 1;
    g_multimodal_prefill_num_segments = 3;
    rc = check(&model, data, 3, 0, 0);
    if (rc) return 21;
    g_multimodal_prefill_segment_lengths[0] = -1;
    rc = check(&model, data, 3, 0, 0);
    if (rc) return 22;
'''
                elif name == "segmented":
                    checks = r'''
    setenv("CK_BRIDGE_NONCAUSAL_VISUAL_CHUNK", "1", 1);
    setenv("CK_BRIDGE_VISUAL_START", "0", 1);
    setenv("CK_BRIDGE_VISUAL_TOKENS", "2", 1);
    rc = check(&model, data, 2, 0, 2);
    if (rc) return 30;
    setenv("CK_BRIDGE_NONCAUSAL_VISUAL_CHUNK", "0", 1);
    rc = check(&model, data, 2, 0, 0);
    if (rc) return 31;
'''
                else:
                    checks = r'''
    rc = check(&model, data, 3, 0, 0);
    if (rc) return 40;
'''
                source = source.replace("__EMITTED__", emitted).replace("__CASES__", checks)
                with tempfile.TemporaryDirectory() as tmpdir:
                    c_path = Path(tmpdir) / "prefill.c"
                    exe_path = Path(tmpdir) / "prefill"
                    c_path.write_text(source, encoding="utf-8")
                    subprocess.run(
                        [compiler, "-std=c11", "-w", str(c_path), "-lm", "-o", str(exe_path)],
                        check=True, capture_output=True, text=True,
                    )
                    subprocess.run([str(exe_path)], check=True, capture_output=True, text=True)

    def test_embedding_scale_preserves_visual_rows_for_causal_mixed_prefill(self):
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("C compiler unavailable")
        emitted = codegen_prefill_v8._emit_embedding_scale_block(
            "num_tokens", False, segment_plan_available=True
        )
        text_only = codegen_prefill_v8._emit_embedding_scale_block("num_tokens", False)
        self.assertNotIn("ck_multimodal_prefill_has_segment_plan", text_only)
        segmented_append = codegen_prefill_v8._emit_embedding_scale_block(
            "num_tokens", False, segment_plan_available=True, segmented_append=True
        )
        self.assertIn("bridge_noncausal_visual_chunk", segmented_append)
        self.assertNotIn("ck_multimodal_prefill_has_segment_plan", segmented_append)
        source = r'''
#include <math.h>
#include <stddef.h>
#define EMBED_DIM 4
#define A_EMBEDDED_INPUT 0
typedef struct { unsigned char *bump; } CKModel;
static int g_multimodal_prefill_segment_lengths[3] = {2, 2, 1};
static int g_multimodal_prefill_num_segments = 3;
static int ck_multimodal_prefill_has_segment_plan(int n) {
    return g_multimodal_prefill_num_segments == 3 &&
        g_multimodal_prefill_segment_lengths[0] >= 0 &&
        g_multimodal_prefill_segment_lengths[1] >= 0 &&
        g_multimodal_prefill_segment_lengths[2] >= 0 &&
        g_multimodal_prefill_segment_lengths[0] +
        g_multimodal_prefill_segment_lengths[1] +
        g_multimodal_prefill_segment_lengths[2] == n;
}
static void run(CKModel *model, int num_tokens) {
__EMITTED__
}
int main(void) {
    float data[20];
    for (int i = 0; i < 20; ++i) data[i] = (float)(i + 1);
    CKModel model = {(unsigned char *)data};
    run(&model, 5);
    for (int row = 0; row < 5; ++row) {
        for (int channel = 0; channel < 4; ++channel) {
            const int i = row * 4 + channel;
            const float expected = (float)(i + 1) * (row >= 2 && row < 4 ? 1.0f : 2.0f);
            if (data[i] != expected) return 1;
        }
    }
    g_multimodal_prefill_num_segments = 0;
    for (int i = 0; i < 20; ++i) data[i] = (float)(i + 1);
    run(&model, 5);
    for (int i = 0; i < 20; ++i) {
        if (data[i] != (float)(i + 1) * 2.0f) return 2;
    }
    return 0;
}
'''.replace("__EMITTED__", emitted)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            c_path = root / "embedding_scale.c"
            exe_path = root / "embedding_scale"
            c_path.write_text(source, encoding="utf-8")
            subprocess.run(
                [compiler, "-std=c11", "-Wall", "-Werror", str(c_path), "-lm", "-o", str(exe_path)],
                check=True, capture_output=True, text=True,
            )
            subprocess.run([str(exe_path)], check=True, capture_output=True, text=True)

    def test_chunk_local_attention_continuation_uses_decode_and_preserves_positions(self):
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("C compiler unavailable")
        op = {
            "function": "noop_kernel", "op": "mla_attention", "layer": 0, "args": [],
            "prefill_history_scope": "current_chunk",
        }
        emitted = codegen_prefill_v8.emit_prefill_function(
            [op], {"context_length": 32, "prefill_chunk_length": 2, "embed_dim": 4}
        )
        schedule = "static void ck_prefill(CKModel" + emitted.split(
            "static void ck_prefill(CKModel", 1
        )[1]
        source = """#include <stdint.h>
#define MAX_SEQ_LEN 32
typedef struct { int pos; } CKModel;
static int g_ck_skip_decode_logits;
static int prefill_calls, decode_calls, bad_position, bad_skip, bad_token;
static int expected_decode_start = 2, expected_decode_count = 3;
static int expected_token_start = 12;
static int cancel_after_decodes = -1;
static int ck_model_cancel_requested(void) {
    return cancel_after_decodes >= 0 && decode_calls >= cancel_after_decodes;
}
static void ck_prefill_range(CKModel *m, const int32_t *tokens, int count, int pos) {
    (void)tokens;
    ++prefill_calls;
    if (pos != m->pos) bad_position = 1;
    m->pos = pos + count;
}
static void ck_decode(CKModel *m, int32_t token) {
    if (m->pos != expected_decode_start + decode_calls) bad_position = 1;
    if (g_ck_skip_decode_logits != (decode_calls + 1 < expected_decode_count)) bad_skip = 1;
    if (token != expected_token_start + decode_calls) bad_token = 1;
    ++decode_calls;
    ++m->pos;
}
""" + schedule + """
int main(void) {
    const int32_t tokens[] = {10, 11, 12, 13, 14};
    CKModel model = {0};
    ck_prefill(&model, tokens, 5);
    if (prefill_calls != 1 || decode_calls != 3 || model.pos != 5 ||
        bad_position || bad_skip || bad_token || g_ck_skip_decode_logits) return 1;
    prefill_calls = decode_calls = bad_position = bad_skip = bad_token = 0;
    expected_decode_start = 3;
    expected_decode_count = 2;
    expected_token_start = 13;
    model.pos = 3;
    ck_prefill(&model, tokens + 3, 2);
    if (prefill_calls != 0 || decode_calls != 2 || model.pos != 5 ||
        bad_position || bad_skip || bad_token || g_ck_skip_decode_logits) return 2;
    prefill_calls = decode_calls = bad_position = bad_skip = bad_token = 0;
    expected_decode_start = 2;
    expected_decode_count = 3;
    expected_token_start = 12;
    cancel_after_decodes = 1;
    model.pos = 0;
    ck_prefill(&model, tokens, 5);
    if (prefill_calls != 1 || decode_calls != 1 || model.pos != 3 ||
        bad_position || bad_skip || bad_token || g_ck_skip_decode_logits) return 3;
    prefill_calls = decode_calls = bad_position = bad_skip = bad_token = 0;
    expected_decode_start = 3;
    expected_decode_count = 2;
    expected_token_start = 13;
    cancel_after_decodes = -1;
    ck_prefill(&model, tokens + 3, 2);
    if (prefill_calls != 0 || decode_calls != 2 || model.pos != 5 ||
        bad_position || bad_skip || bad_token || g_ck_skip_decode_logits) return 4;
    return 0;
}
"""
        with tempfile.TemporaryDirectory() as temp:
            c_path = Path(temp) / "chunk_continuation.c"
            binary = Path(temp) / "chunk_continuation"
            c_path.write_text(source)
            subprocess.run([compiler, "-std=c11", "-Wall", "-Werror", str(c_path),
                            "-o", str(binary)], check=True, capture_output=True)
            subprocess.run([str(binary)], check=True, capture_output=True)

    def test_chunk_local_continuation_reads_prior_cache(self):
        """The old chunk-only schedule must fail a history-sensitive output."""
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("C compiler unavailable")
        base_op = {"function": "noop_kernel", "op": "mla_attention", "layer": 0,
                   "args": []}
        config = {"context_length": 32, "prefill_chunk_length": 2, "embed_dim": 4}
        program = r'''#include <stdint.h>
#define MAX_SEQ_LEN 32
typedef struct { int pos; int cache[32]; int context_sum; } CKModel;
static int g_ck_skip_decode_logits;
static int ck_model_cancel_requested(void) { return 0; }
/* A current-chunk provider sees only the supplied rows. */
static void ck_prefill_range(CKModel *m, const int32_t *tokens, int count, int pos) {
    int local_sum = 0;
    for (int i = 0; i < count; ++i) {
        m->cache[pos + i] = tokens[i];
        local_sum += tokens[i];
    }
    m->context_sum = local_sum;
    m->pos = pos + count;
}
/* Decode consumes every previously stored row and appends its own. */
static void ck_decode(CKModel *m, int32_t token) {
    int total = token;
    for (int i = 0; i < m->pos; ++i) total += m->cache[i];
    m->cache[m->pos++] = token;
    m->context_sum = total;
}
__SCHEDULE__
int main(void) {
    (void)g_ck_skip_decode_logits;
    const int32_t tokens[] = {10, 11, 12, 13};
    CKModel model = {0};
    ck_prefill(&model, tokens, 4);
    if (model.pos != 4) return 1;
    if (model.context_sum != 46) return 2;
    /* The same continuation must notice a wrong prior cache owner/value. */
    model.pos = 2;
    model.cache[0] = 99;
    ck_prefill(&model, tokens + 2, 2);
    if (model.pos != 4 || model.context_sum == 46) return 3;
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for label, scope, expected_exit in (
                ("declared", "current_chunk", 0),
                ("undeclared_falls_back", None, 0),
                ("unsafe_chunk_replay_control", None, 2),
            ):
                op = dict(base_op)
                if scope is not None:
                    op["prefill_history_scope"] = scope
                emitted = codegen_prefill_v8.emit_prefill_function([op], config)
                schedule = "static void ck_prefill(CKModel" + emitted.split(
                    "static void ck_prefill(CKModel", 1)[1]
                if label == "unsafe_chunk_replay_control":
                    schedule = schedule.replace(
                        "if (consumed > 0 || start_pos > 0)", "if (0)"
                    )
                source = root / f"{label}.c"
                binary = root / label
                source.write_text(program.replace("__SCHEDULE__", schedule))
                subprocess.run([compiler, "-std=c11", "-Wall", "-Werror",
                                str(source), "-o", str(binary)], check=True,
                               capture_output=True, text=True)
                result = subprocess.run([str(binary)], capture_output=True)
                self.assertEqual(result.returncode, expected_exit, label)

    def test_composite_embedding_receives_declared_segment_plan(self):
        provider = json.loads((ROOT / "version/v8/kernel_maps/gemma4_per_layer_embed_forward.json").read_text())
        op = {
            "function": provider["impl"]["function"],
            "op": provider["op"],
            "layer": 0,
            "args": [
                {"name": arg["name"], "source": arg["source"], "expr": arg["name"]}
                for arg in provider["call_abi"]["params"]
            ],
        }
        config = {
            "multimodal_bridge_contract": {
                "prefill_schedule": {
                    "projection_row_group_boundaries": {
                        "policy": "restart_each_segment",
                        "operations": [provider["op"]],
                        "text_after_row_group": "single_token",
                    },
                },
            },
        }
        emitted = codegen_prefill_v8.emit_prefill_op(
            op, 0, config, segment_plan_available=True
        )
        self.assertIn("g_multimodal_prefill_segment_lengths", emitted)
        self.assertIn("g_multimodal_prefill_num_segments", emitted)
        self.assertIn("? g_multimodal_prefill_num_segments : 0", emitted)
        self.assertIn("\n        1\n    );", emitted)
        config["multimodal_bridge_contract"]["prefill_schedule"][
            "projection_row_group_boundaries"
        ]["operations"] = []
        with self.assertRaisesRegex(RuntimeError, "without a declared segment policy"):
            codegen_prefill_v8.emit_prefill_op(
                op, 0, config, segment_plan_available=True
            )

    def test_gemma4_mixed_bridge_emits_segmented_attention_schedule(self):
        path = ROOT / "version/v8/kernel_maps/attention_forward_causal_head_major_gqa_prefill_append_f16cache_sliding_gemma4.json"
        provider = json.loads(path.read_text(encoding="utf-8"))
        op = {
            "function": provider["impl"]["function"],
            "op": "attn_sliding",
            "layer": 0,
            "resolved_execution": {"implementation": provider["implementation"]},
            "args": [
                {"name": arg["name"], "source": arg["source"], "expr": arg["name"]}
                for arg in provider["call_abi"]["params"]
            ],
        }
        config = {
            "embed_dim": 16,
            "multimodal_bridge_contract": {
                "prefill_batching": "segmented_append",
                "prefill_schedule": {
                    "segments": ["text_before", "visual", "text_after"],
                    "cache_transition": "append_preserve",
                    "position_transition": "runtime_offset",
                },
            },
        }
        emitted = codegen_prefill_v8.emit_prefill_from_embedded_function([op], config)
        self.assertIn("static int g_multimodal_prefill_segment_lengths[3]", emitted)
        self.assertIn("ck_multimodal_prefill_has_segment_plan(num_tokens)", emitted)
        self.assertIn(provider["implementation"]["segmented_query_provider"]["function"] + "(", emitted)
        self.assertIn(provider["impl"]["function"] + "(", emitted)
        embedding = {
            "op": "dense_embedding_lookup",
            "function": "embedding_forward_q4_k",
            "args": [
                {"name": "token_ids", "expr": "tokens"},
                {"name": "token_embeddings", "expr": "weights"},
                {"name": "output", "expr": "embedded"},
            ],
        }
        bridge = codegen_prefill_v8.emit_multimodal_bridge_api([embedding], config)
        self.assertIn("g_multimodal_prefill_segment_lengths[0] = tokens_before_count", bridge)
        self.assertIn("g_multimodal_prefill_segment_lengths[1] = prefix_tokens", bridge)
        self.assertIn("g_multimodal_prefill_num_segments = 0", bridge)
        invalid = json.loads(json.dumps(config))
        invalid["multimodal_bridge_contract"]["prefill_schedule"]["cache_transition"] = "replace"
        self.assertFalse(codegen_prefill_v8._has_mixed_prefill_segment_contract(invalid))

    def test_hyper_prefill_exports_wide_boundaries_and_last_rows(self):
        args = {"rows": "num_tokens", "streams": "4", "hidden_dim": "2560",
                "dynamic_dim": "320", "normalized_scratch": "norm",
                "dynamic_scratch": "dynamic", "mix_scratch": "gate",
                "mixed_output": "mixed", "injection_output": "NULL", "output": "wide"}
        for op_name, label in [("hyper_mix_attn", "attn_hyper_norm"),
                               ("hyper_mix_mlp", "mlp_hyper_norm"),
                               ("hyper_mix_final", "final_hyper_norm"),
                               ("hyper_stream_expand", "hyper_stream"),
                               ("hyper_inject_attn", "after_attn_hyper"),
                               ("hyper_inject_mlp", "layer_out")]:
            with self.subTest(op=op_name):
                op = {"function": "test_provider", "op": op_name, "layer": 7,
                      "args": [{"name": k, "expr": v} for k, v in args.items()]}
                emitted = codegen_prefill_v8.emit_prefill_op(op, 0, {})
                self.assertIn(f'"{label}"', emitted)
                self.assertIn(f'"{label}_last"', emitted)
                self.assertIn("(4) * (2560)", emitted)
                self.assertNotIn('"final_injection_weights"', emitted)

    @staticmethod
    def _q4_segmented_projection_op(op_name: str = "recurrent_gate_proj") -> dict:
        return {
            "function": "gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch",
            "op": op_name,
            "layer": 0,
            "resolved_execution": {
                "numerical_contract": "q4_k_x_q8_k_repacked_matmul_fp32",
                "implementation": {
                    "weight_storage": {
                        "format": "q4_k",
                        "block_elements": 256,
                        "block_bytes": 144,
                    },
                    "activation_storage": {
                        "format": "q8_k",
                        "block_elements": 256,
                    },
                    "diagnostic_providers": {
                        "fp32_activation": "gemm_nt_q4_k",
                        "row_quantized": "gemv_q4_k_q8_k",
                    },
                    "segmented_row_provider": {
                        "function": "gemm_nt_q4_k_q8_k_segmented_pairwise_split_min_parallel_dispatch",
                        "segment_lengths_dtype": "i32",
                        "boundary_semantics": "restart_row_group_at_each_segment",
                        "fallback": "gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch",
                    },
                },
            },
            "args": [
                {"name": "A", "source": "activation:a", "expr": "A"},
                {"name": "B", "source": "weight:_first_weight", "expr": "B"},
                {"name": "bias", "source": "weight_f:_bias", "expr": "NULL"},
                {"name": "C", "source": "output:c", "expr": "C"},
                {"name": "M", "source": "dim:_m", "expr": "1035"},
                {"name": "N", "source": "dim:_output_dim", "expr": "6144"},
                {"name": "K", "source": "dim:_input_dim", "expr": "5120"},
            ],
        }

    @staticmethod
    def _segment_preserving_config() -> dict:
        return {
            "multimodal_bridge_contract": {
                "prefix_policy": "mixed_visual_text_prefill",
                "prefill_schedule": {
                    "projection_row_group_boundaries": {
                        "policy": "restart_each_segment",
                        "operations": [
                            "recurrent_qkv_proj",
                            "recurrent_gate_proj",
                            "mlp_gate_up",
                        ],
                    }
                },
            }
        }

    def test_q4_projection_uses_map_owned_runtime_segment_provider(self) -> None:
        emitted = codegen_prefill_v8.emit_prefill_op(
            self._q4_segmented_projection_op(),
            6,
            self._segment_preserving_config(),
            segment_plan_available=True,
        )
        self.assertIn(
            "gemm_nt_q4_k_q8_k_segmented_pairwise_split_min_parallel_dispatch(",
            emitted,
        )
        self.assertIn("g_multimodal_prefill_segment_lengths", emitted)
        self.assertIn("g_multimodal_prefill_num_segments", emitted)
        self.assertIn("gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch(", emitted)

    def test_q4_projection_without_circuit_contract_keeps_unified_provider(self) -> None:
        emitted = codegen_prefill_v8.emit_prefill_op(
            self._q4_segmented_projection_op(), 6, {}
        )
        self.assertNotIn("segmented_pairwise_split_min", emitted)
        self.assertIn("gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch(", emitted)

    def test_q6_projection_uses_map_owned_runtime_segment_provider(self) -> None:
        op = self._q4_segmented_projection_op("recurrent_qkv_proj")
        op["function"] = "gemm_nt_q6_k_q8_k"
        op["resolved_execution"]["implementation"].update(
            {
                "weight_storage": {
                    "format": "q6_k",
                    "block_elements": 256,
                    "block_bytes": 210,
                },
                "diagnostic_providers": {
                    "fp32_activation": "gemm_nt_q6_k",
                    "row_quantized": "gemv_q6_k_q8_k",
                },
                "segmented_row_provider": {
                    "function": "gemm_nt_q6_k_q8_k_segmented_parallel_dispatch",
                    "segment_lengths_dtype": "i32",
                    "boundary_semantics": "restart_row_group_at_each_segment",
                    "fallback": "gemm_nt_q6_k_q8_k_parallel_dispatch",
                },
            }
        )
        op["resolved_execution"]["numerical_contract"] = (
            "q6_k_weight_q8_k_input_llama_fp32_output"
        )

        emitted = codegen_prefill_v8.emit_prefill_op(
            op,
            4,
            self._segment_preserving_config(),
            segment_plan_available=True,
        )

        self.assertIn("gemm_nt_q6_k_q8_k_segmented_parallel_dispatch(", emitted)
        self.assertIn("g_multimodal_prefill_segment_lengths", emitted)
        self.assertIn("gemm_nt_q6_k_q8_k(", emitted)

    def test_q6_projection_selects_declared_singleton_tail_provider(self) -> None:
        op = self._q4_segmented_projection_op("v_proj")
        op["function"] = "gemm_nt_q6_k_q8_k"
        op["resolved_execution"]["numerical_contract"] = (
            "q6_k_weight_q8_k_input_llama_fp32_output"
        )
        implementation = op["resolved_execution"]["implementation"]
        implementation["weight_storage"] = {
            "format": "q6_k", "block_elements": 256, "block_bytes": 210,
        }
        implementation["segmented_row_provider"] = {
            "function": "gemm_nt_q6_k_q8_k_segmented_parallel_dispatch",
            "singleton_tail_function": (
                "gemm_nt_q6_k_q8_k_segmented_singleton_tail_parallel_dispatch"
            ),
            "segment_lengths_dtype": "i32",
            "boundary_semantics": "restart_row_group_at_each_segment",
            "fallback": "gemm_nt_q6_k_q8_k_parallel_dispatch",
        }
        config = self._segment_preserving_config()
        boundaries = config["multimodal_bridge_contract"]["prefill_schedule"][
            "projection_row_group_boundaries"
        ]
        boundaries["operations"] = ["v_proj"]
        boundaries["text_after_row_group"] = "single_token"
        emitted = codegen_prefill_v8.emit_prefill_op(
            op, 9, config, segment_plan_available=True
        )
        self.assertIn(
            "gemm_nt_q6_k_q8_k_segmented_singleton_tail_parallel_dispatch(", emitted
        )
        self.assertIn("g_multimodal_prefill_segment_lengths", emitted)
        self.assertIn("gemm_nt_q6_k_q8_k(", emitted)

        del implementation["segmented_row_provider"]["singleton_tail_function"]
        with self.assertRaisesRegex(RuntimeError, "map-owned singleton-tail provider"):
            codegen_prefill_v8.emit_prefill_op(
                op, 9, config, segment_plan_available=True
            )

    def test_q4_output_projection_restarts_rows_at_mixed_prefix_boundaries(self) -> None:
        op = self._q4_segmented_projection_op("out_proj")
        op["resolved_execution"]["implementation"]["segmented_row_provider"][
            "singleton_tail_function"
        ] = "gemm_nt_q4_k_q8_k_segmented_singleton_tail_parallel_dispatch"
        config = self._segment_preserving_config()
        boundaries = config["multimodal_bridge_contract"]["prefill_schedule"][
            "projection_row_group_boundaries"
        ]
        boundaries["operations"] = ["out_proj"]
        boundaries["text_after_row_group"] = "single_token"
        emitted = codegen_prefill_v8.emit_prefill_op(
            op, 0, config, segment_plan_available=True
        )
        self.assertIn(
            "gemm_nt_q4_k_q8_k_segmented_singleton_tail_parallel_dispatch(", emitted
        )
        self.assertIn("g_multimodal_prefill_segment_lengths", emitted)
        self.assertIn("_ck_projection_segments, 3", emitted)
        self.assertIn("g_multimodal_prefill_segment_lengths[2] +", emitted)
        self.assertIn("gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch(", emitted)

        del op["resolved_execution"]["implementation"]["segmented_row_provider"][
            "singleton_tail_function"
        ]
        with self.assertRaisesRegex(RuntimeError, "map-owned singleton-tail provider"):
            codegen_prefill_v8.emit_prefill_op(
                op, 0, config, segment_plan_available=True
            )
        op["resolved_execution"]["implementation"]["segmented_row_provider"][
            "singleton_tail_function"
        ] = "gemm_nt_q4_k_q8_k_segmented_singleton_tail_parallel_dispatch"

        boundaries["operations"] = ["q_proj", "k_proj", "v_proj"]
        unsegmented = codegen_prefill_v8.emit_prefill_op(
            op, 0, config, segment_plan_available=True
        )
        self.assertNotIn("segmented_singleton_tail_parallel_dispatch(", unsegmented)

    def test_compiled_projection_contract_selects_declared_provider(self) -> None:
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "compiled projection contract requires a C compiler")

        config = self._segment_preserving_config()
        boundaries = config["multimodal_bridge_contract"]["prefill_schedule"][
            "projection_row_group_boundaries"
        ]
        boundaries["operations"] = ["out_proj"]
        boundaries["text_after_row_group"] = "single_token"
        op = self._q4_segmented_projection_op("out_proj")
        op["resolved_execution"]["implementation"]["segmented_row_provider"][
            "singleton_tail_function"
        ] = "gemm_nt_q4_k_q8_k_segmented_singleton_tail_parallel_dispatch"
        for arg in op["args"]:
            if arg["name"] == "M":
                arg["expr"] = "rows"
                arg["source"] = "runtime:rows"
            elif arg["name"] == "N":
                arg["expr"] = "4"
            elif arg["name"] == "K":
                arg["expr"] = "8"

        declared = codegen_prefill_v8.emit_prefill_op(
            op, 0, config, segment_plan_available=True
        )
        boundaries["operations"] = []
        undeclared = codegen_prefill_v8.emit_prefill_op(
            op, 0, config, segment_plan_available=True
        )
        boundaries["operations"] = ["out_proj"]
        del op["resolved_execution"]["implementation"]["segmented_row_provider"][
            "singleton_tail_function"
        ]
        with self.assertRaisesRegex(RuntimeError, "map-owned singleton-tail provider"):
            codegen_prefill_v8.emit_prefill_op(
                op, 0, config, segment_plan_available=True
            )

        source = r'''
#include <stddef.h>
static int selected, saw_lengths[3], saw_count;
static int g_multimodal_prefill_segment_lengths[3] = {2, 2, 1};
static int debug_outproj_fp32;
static void *ck_debug_outproj_fp32_input;
static void *model;
#define EMBED_DIM 4
static int ck_multimodal_prefill_has_segment_plan(int n) { return n == 5; }
static void ck_debug_export_hidden(void *m, int layer, const char *name,
                                   const float *data, int count) {
    (void)m; (void)layer; (void)name; (void)data; (void)count;
}
static void gemm_nt_q4_k(const void *a, const void *b, const float *bias,
                         float *c, int m, int n, int k) {
    (void)a; (void)b; (void)bias; (void)c; (void)m; (void)n; (void)k;
    selected = 3;
}
static void gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch(
    const void *a, const void *b, const float *bias, float *c,
    int m, int n, int k) {
    (void)a; (void)b; (void)bias; (void)c; (void)m; (void)n; (void)k;
    selected = 1;
}
static void gemm_nt_q4_k_q8_k_segmented_singleton_tail_parallel_dispatch(
    const void *a, const void *b, const float *bias, float *c,
    int m, int n, int k, const int *lengths, int count) {
    (void)a; (void)b; (void)bias; (void)c; (void)m; (void)n; (void)k;
    selected = 2;
    saw_count = count;
    for (int i = 0; i < 3; ++i) saw_lengths[i] = lengths[i];
}
static void run_declared(int num_tokens) {
    float A[40] = {0}, B[40] = {0}, C[20] = {0};
    int rows = num_tokens;
    (void)A; (void)B; (void)C;
__DECLARED__
}
static void run_undeclared(int num_tokens) {
    float A[40] = {0}, B[40] = {0}, C[20] = {0};
    int rows = num_tokens;
    (void)A; (void)B; (void)C;
__UNDECLARED__
}
int main(void) {
    run_declared(5);
    if (selected != 2 || saw_count != 3 || saw_lengths[0] != 2 ||
        saw_lengths[1] != 2 || saw_lengths[2] != 1) return 1;
    selected = 0;
    run_declared(4);
    if (selected != 1) return 2;
    selected = 0;
    run_undeclared(5);
    if (selected != 1) return 3;
    return 0;
}
'''.replace("__DECLARED__", declared).replace("__UNDECLARED__", undeclared)
        with tempfile.TemporaryDirectory(prefix="cke_projection_contract_") as tmp:
            path = Path(tmp)
            (path / "fixture.c").write_text(source, encoding="utf-8")
            compiled = subprocess.run(
                [cc, "-std=c11", "-Wall", "-Werror", str(path / "fixture.c"),
                 "-o", str(path / "fixture")],
                capture_output=True, text=True,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            subprocess.run([str(path / "fixture")], check=True)

    def test_segmented_gate_up_contract_disables_uncontracted_fusion(self) -> None:
        op = self._q4_segmented_projection_op("mlp_gate_up")
        swiglu = {
            "op": "swiglu",
            "function": "silu_mul",
            "layer": 0,
            "args": [{"name": "output", "expr": "Y"}],
        }
        config = self._segment_preserving_config()
        bridge = config["multimodal_bridge_contract"]
        bridge["prefill_batching"] = "segmented_append"
        bridge["prefill_schedule"].update({
            "segments": ["text_before", "visual", "text_after"],
            "cache_transition": "append_preserve",
            "position_transition": "runtime_offset",
        })
        direct = codegen_prefill_v8.emit_prefill_op(
            op, 0, config, segment_plan_available=True
        )
        self.assertIn(
            "gemm_nt_q4_k_q8_k_segmented_pairwise_split_min_parallel_dispatch(",
            direct,
        )
        emitted = codegen_prefill_v8.emit_prefill_from_embedded_function(
            [op, swiglu], config
        )
        self.assertIn(
            "gemm_nt_q4_k_q8_k_segmented_pairwise_split_min_parallel_dispatch(",
            emitted,
        )
        self.assertNotIn("gemm_nt_q4_k_q8_k_gateup_swiglu_x16_parallel_dispatch(", emitted)

        del op["resolved_execution"]["implementation"]["segmented_row_provider"]
        with self.assertRaisesRegex(RuntimeError, "map-owned segmented-row provider"):
            codegen_prefill_v8.emit_prefill_from_embedded_function([op, swiglu], config)

    def test_segmented_gate_up_uses_schedule_without_prefix_policy(self) -> None:
        op = self._q4_segmented_projection_op("mlp_gate_up")
        config = self._segment_preserving_config()
        bridge = config["multimodal_bridge_contract"]
        del bridge["prefix_policy"]
        bridge["prefill_batching"] = "segmented_append"
        bridge["prefill_schedule"].update({
            "segments": ["text_before", "visual", "text_after"],
            "cache_transition": "append_preserve",
            "position_transition": "runtime_offset",
        })
        emitted = codegen_prefill_v8.emit_prefill_from_embedded_function(
            [op], config, dump=True
        )
        self.assertIn(
            "gemm_nt_q4_k_q8_k_segmented_pairwise_split_min_parallel_dispatch(",
            emitted,
        )
        self.assertIn('"ffn_gate_up"', emitted)
        self.assertNotIn('"down_proj"', emitted)

        bridge["prefill_schedule"]["projection_row_group_boundaries"]["operations"] = []
        unsegmented = codegen_prefill_v8.emit_prefill_from_embedded_function(
            [op], config
        )
        self.assertNotIn("segmented_pairwise_split_min_parallel_dispatch(", unsegmented)

    def test_segmented_down_uses_schedule_without_prefix_policy(self) -> None:
        op = self._q4_segmented_projection_op("mlp_down")
        op["function"] = "gemm_nt_q6_k_q8_k"
        op["resolved_execution"]["numerical_contract"] = (
            "q6_k_weight_q8_k_input_llama_fp32_output"
        )
        implementation = op["resolved_execution"]["implementation"]
        implementation["weight_storage"] = {
            "format": "q6_k", "block_elements": 256, "block_bytes": 210,
        }
        implementation["segmented_row_provider"] = {
            "function": "gemm_nt_q6_k_q8_k_segmented_parallel_dispatch",
            "singleton_tail_function": (
                "gemm_nt_q6_k_q8_k_segmented_singleton_tail_parallel_dispatch"
            ),
            "segment_lengths_dtype": "i32",
            "boundary_semantics": "restart_row_group_at_each_segment",
            "fallback": "gemm_nt_q6_k_q8_k_parallel_dispatch",
        }
        config = self._segment_preserving_config()
        bridge = config["multimodal_bridge_contract"]
        del bridge["prefix_policy"]
        bridge["prefill_batching"] = "segmented_append"
        bridge["prefill_schedule"].update({
            "segments": ["text_before", "visual", "text_after"],
            "cache_transition": "append_preserve",
            "position_transition": "runtime_offset",
        })
        boundaries = bridge["prefill_schedule"]["projection_row_group_boundaries"]
        boundaries["operations"] = ["mlp_down"]
        boundaries["text_after_row_group"] = "single_token"
        emitted = codegen_prefill_v8.emit_prefill_from_embedded_function(
            [op], config, dump=True
        )
        self.assertIn(
            "gemm_nt_q6_k_q8_k_segmented_singleton_tail_parallel_dispatch(",
            emitted,
        )
        self.assertIn('"down_proj"', emitted)

        del implementation["segmented_row_provider"]["singleton_tail_function"]
        with self.assertRaisesRegex(RuntimeError, "map-owned singleton-tail provider"):
            codegen_prefill_v8.emit_prefill_from_embedded_function([op], config)

    def test_segmented_projection_rejects_wrong_call_arity(self) -> None:
        op = self._q4_segmented_projection_op()
        op["args"].pop()
        with self.assertRaisesRegex(RuntimeError, "requires seven GEMM arguments"):
            codegen_prefill_v8.emit_prefill_op(
                op, 0, self._segment_preserving_config(),
                segment_plan_available=True,
            )

    def test_segmented_projection_rejects_incompatible_boundary_semantics(self) -> None:
        op = self._q4_segmented_projection_op()
        op["resolved_execution"]["implementation"]["segmented_row_provider"][
            "boundary_semantics"
        ] = "continue_row_group_across_segments"
        with self.assertRaisesRegex(RuntimeError, "incompatible boundary semantics"):
            codegen_prefill_v8.emit_prefill_op(
                op, 0, self._segment_preserving_config(),
                segment_plan_available=True,
            )

    def test_attention_uses_map_owned_segmented_query_provider(self) -> None:
        base_arg_names = [
            "q", "k_cache", "v_cache", "output", "num_heads", "num_kv_heads",
            "q_tokens", "past_tokens", "cache_capacity", "head_dim",
            "aligned_head_dim", "reduction", "token_workspace",
            "token_workspace_bytes",
        ]
        route_arg_names = [
            "gqa_workspace", "gqa_workspace_bytes", "route_num_heads",
            "route_num_kv_heads", "route_head_dim", "route_query_tokens",
            "route_min_kv_tokens", "route_workers", "route_query_tile_size",
            "route_concurrent_query_tiles",
        ]
        op = {
            "function": "attention_forward_causal_head_major_gqa_prefill_append_f16cache_auto_workspace",
            "op": "attn",
            "layer": 3,
            "resolved_execution": {
                "implementation": {
                    "segmented_query_provider": {
                        "function": "attention_forward_causal_head_major_gqa_prefill_segmented_f16cache_contract_workspace",
                        "base_args": base_arg_names,
                        "segment_lengths_dtype": "i32",
                        "boundary_semantics": "restart_query_tile_policy_at_each_segment",
                        "fallback": "reject_invalid_plan",
                    }
                }
            },
            "args": [
                {"name": name, "source": f"test:{name}", "expr": name}
                for name in base_arg_names + route_arg_names
            ],
        }
        emitted = codegen_prefill_v8.emit_prefill_op(
            op, 91, {}, segment_plan_available=True
        )
        self.assertIn(
            "attention_forward_causal_head_major_gqa_prefill_segmented_f16cache_contract_workspace(",
            emitted,
        )
        self.assertIn("g_multimodal_prefill_segment_lengths", emitted)
        self.assertIn("g_multimodal_prefill_num_segments", emitted)
        self.assertIn(
            "attention_forward_causal_head_major_gqa_prefill_append_f16cache_auto_workspace(",
            emitted,
        )
        segmented_call = emitted.split("} else {", 1)[0]
        self.assertNotIn("gqa_workspace,", segmented_call)
        self.assertNotIn("route_num_heads,", segmented_call)

    def test_kv_transpose_exports_attention_consumed_layout(self) -> None:
        config = {"num_kv_heads": 8, "head_dim": 128, "context_len": 1034}
        for is_k, label in ((True, "k_head_major"), (False, "v_head_major")):
            with self.subTest(label=label):
                emitted = codegen_prefill_v8.emit_prefill_op(
                    {
                        "function": "transpose_inplace",
                        "op": "transpose_kv_to_head_major",
                        "layer": 3,
                        "section": "body",
                        "_is_k": is_k,
                        "_num_kv_heads": 8,
                        "_head_dim": 128,
                    },
                    17,
                    config,
                )
                copy_back = "memcpy(buf, _temp_buf, (size_t)Hkv * num_tokens * D * sizeof(float));"
                export = f'"{label}"'
                self.assertIn(copy_back, emitted)
                self.assertIn(export, emitted)
                self.assertLess(emitted.index(copy_back), emitted.index(export))
                self.assertIn("Hkv * num_tokens * D", emitted)

    def test_last_logits_preserves_resolved_exact_gemm_provider(self) -> None:
        function = "gemm_nt_bf16_pytorch_onednn_brgemm_bf16_storage"
        op = {
            "function": function,
            "op": "logits",
            "layer": -1,
            "call_abi": {
                "version": 1,
                "owner": "kernel_map",
                "kernel_id": function,
                "last_token_dispatch": "preserve_provider",
            },
            "args": [
                {
                    "name": "A",
                    "source": "activation:a",
                    "expr": "(const float*)(model->bump + A_MAIN_STREAM)",
                },
                {
                    "name": "B",
                    "source": "weight:_first_weight",
                    "expr": "(const void*)(model->bump + W_LM_HEAD)",
                },
                {
                    "name": "C",
                    "source": "output:c",
                    "expr": "(float*)(model->bump + A_LOGITS)",
                },
                {"name": "M", "source": "dim:_m", "expr": "num_tokens"},
                {"name": "N", "source": "dim:_output_dim", "expr": "151936"},
                {"name": "K", "source": "dim:_input_dim", "expr": "4096"},
            ],
        }

        emitted = codegen_prefill_v8.emit_prefill_op(
            op,
            758,
            {"embed_dim": 4096, "vocab_size": 151936, "logits_layout": "last"},
        )

        self.assertIn("logits (last-only exact GEMM contract)", emitted)
        self.assertIn(function + "(", emitted)
        self.assertIn("(size_t)(num_tokens - 1) * 4096", emitted)
        self.assertNotIn(
            "gemv_bf16_pytorch_onednn_brgemm_bf16_storage",
            emitted,
        )

    def test_last_logits_quantized_provider_uses_q8_k_row_gemv(self) -> None:
        op = {
            "function": "gemm_nt_q6_k_q8_k",
            "op": "logits",
            "layer": -1,
            "call_abi": {
                "version": 1,
                "owner": "kernel_map",
                "kernel_id": "gemm_nt_q6_k_q8_k",
            },
            "resolved_execution": {
                "kernel_id": "gemm_nt_q6_k_q8_k",
                "implementation": {
                    "weight_storage": {
                        "format": "q6_k",
                        "block_elements": 256,
                        "block_bytes": 210,
                    },
                    "activation_storage": {
                        "format": "q8_k",
                        "block_elements": 256,
                    },
                    "diagnostic_providers": {
                        "fp32_activation": "gemm_nt_q6_k",
                        "row_quantized": "gemv_q6_k_q8_k",
                    },
                },
                "numerical_contract": "q6_k_x_q8_k_fp32_block_order",
            },
            "args": [
                {
                    "name": "A",
                    "source": "activation:a",
                    "expr": "(const void*)(model->bump + A_LAYER_INPUT)",
                },
                {
                    "name": "B",
                    "source": "weight:_first_weight",
                    "expr": "(const void*)(model->bump + W_OUTPUT_WEIGHT)",
                },
                {
                    "name": "bias",
                    "source": "weight_f:_bias",
                    "expr": "NULL",
                },
                {
                    "name": "C",
                    "source": "output:c",
                    "expr": "(float*)(model->bump + A_LOGITS)",
                },
            ],
        }

        emitted = codegen_prefill_v8.emit_prefill_op(
            op,
            921,
            {"embed_dim": 4096, "vocab_size": 151936, "logits_layout": "last"},
        )

        self.assertIn("logits (last-only)", emitted)
        self.assertIn("gemv_q6_k_q8_k(", emitted)
        self.assertIn("(4096 / QK_K) * sizeof(block_q8_K)", emitted)
        self.assertNotIn("(const float*)", emitted.split("ck_debug_export_hidden", 1)[0])
        self.assertNotIn("last-only exact GEMM contract", emitted)

    def test_last_logits_generic_bf16_preserves_registered_gemm(self) -> None:
        op = {
            "function": "gemm_nt_bf16",
            "op": "logits",
            "layer": -1,
            "call_abi": {
                "version": 1,
                "owner": "kernel_map",
                "kernel_id": "gemm_nt_bf16",
                "last_token_dispatch": "preserve_provider",
            },
            "args": [
                {
                    "name": "A",
                    "source": "activation:a",
                    "expr": "(const float*)(model->bump + A_MAIN_STREAM)",
                },
                {
                    "name": "B",
                    "source": "weight:_first_weight",
                    "expr": "(const void*)(model->bump + W_LM_HEAD)",
                },
                {
                    "name": "bias",
                    "source": "weight_f:_bias",
                    "expr": "NULL",
                },
                {
                    "name": "C",
                    "source": "output:c",
                    "expr": "(float*)(model->bump + A_LOGITS)",
                },
            ],
        }

        emitted = codegen_prefill_v8.emit_prefill_op(
            op,
            460,
            {"embed_dim": 2048, "vocab_size": 163840, "logits_layout": "last"},
        )

        self.assertIn("gemm_nt_bf16(", emitted)
        self.assertIn("(size_t)(num_tokens - 1) * 2048", emitted)
        self.assertNotIn("gemv_bf16(", emitted)

    def test_last_logits_q4_provider_uses_registered_row_gemv(self) -> None:
        op = {
            "function": "gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch",
            "op": "logits",
            "layer": -1,
            "call_abi": {
                "version": 1,
                "owner": "kernel_map",
                "kernel_id": "gemm_nt_q4_k_q8_k",
            },
            "resolved_execution": {
                "kernel_id": "gemm_nt_q4_k_q8_k",
                "implementation": {
                    "weight_storage": {
                        "format": "q4_k",
                        "block_elements": 256,
                        "block_bytes": 144,
                    },
                    "activation_storage": {
                        "format": "q8_k",
                        "block_elements": 256,
                    },
                    "diagnostic_providers": {
                        "fp32_activation": "gemm_nt_q4_k",
                        "row_quantized": "gemv_q4_k_q8_k",
                    },
                },
                "numerical_contract": "q4_k_x_q8_k_repacked_matmul_fp32",
            },
            "args": [
                {
                    "name": "A",
                    "source": "activation:a",
                    "expr": "(const void*)(model->bump + A_LAYER_INPUT)",
                },
                {
                    "name": "B",
                    "source": "weight:_first_weight",
                    "expr": "(const void*)(model->bump + W_TOKEN_EMB)",
                },
                {
                    "name": "C",
                    "source": "output:c",
                    "expr": "(float*)(model->bump + A_LOGITS)",
                },
            ],
        }

        emitted = codegen_prefill_v8.emit_prefill_op(
            op,
            1222,
            {"embed_dim": 2560, "vocab_size": 262144, "logits_layout": "last"},
        )

        self.assertIn("gemv_q4_k_q8_k(", emitted)
        self.assertNotIn("gemv_q4_k_q8_k_pairwise_split_min_parallel_dispatch", emitted)

    def test_last_logits_legacy_provider_keeps_bounded_row_dispatch(self) -> None:
        op = {
            "function": "gemm_nt_q8_0_q8_0",
            "op": "logits",
            "layer": -1,
            "call_abi": {
                "version": 0,
                "owner": "legacy_compatibility",
                "kernel_id": "gemm_nt_q8_0_q8_0",
                "source_file": "kernel_bindings*.json",
            },
            "args": [
                {
                    "name": "A",
                    "source": "activation:a",
                    "expr": "(const void*)(model->bump + A_MAIN_STREAM)",
                },
                {
                    "name": "B",
                    "source": "weight:_first_weight",
                    "expr": "(const void*)(model->bump + W_LM_HEAD)",
                },
                {
                    "name": "C",
                    "source": "output:c",
                    "expr": "(float*)(model->bump + A_LOGITS)",
                },
            ],
        }

        emitted = codegen_prefill_v8.emit_prefill_op(
            op,
            460,
            {"embed_dim": 2048, "vocab_size": 163840, "logits_layout": "last"},
        )

        self.assertIn("gemv_q8_0_q8_0(", emitted)
        self.assertNotIn("gemm_nt_q8_0_q8_0(", emitted)

    def test_residual_save_exports_prefill_layer_input_before_normalization(self) -> None:
        op = {
            "function": "memcpy",
            "op": "residual_save",
            "layer": 0,
            "op_instance_idx": 0,
            "args": [
                {"name": "dst", "expr": "RESIDUAL"},
                {"name": "src", "expr": "LAYER_INPUT"},
                {"name": "size", "source": "dim:_memcpy_bytes", "expr": "4096"},
            ],
        }

        emitted = codegen_prefill_v8.emit_prefill_op(op, 1, {"embed_dim": 4096})

        self.assertIn(
            'ck_debug_export_hidden(model, 0, "layer_input", '
            "(const float*)LAYER_INPUT, (num_tokens) * (EMBED_DIM))",
            emitted,
        )

        emitted_dump = codegen_prefill_v8.emit_prefill_op(
            op, 1, {"embed_dim": 4096}, dump=True
        )
        self.assertIn(
            'ck_dump_tensor((float*)LAYER_INPUT, 0, "layer_input", '
            "(num_tokens) * (4096))",
            emitted_dump,
        )

        op["op_instance_idx"] = 1
        emitted_after_attention = codegen_prefill_v8.emit_prefill_op(
            op, 16, {"embed_dim": 4096}
        )
        self.assertIn('"after_attn"', emitted_after_attention)
        self.assertNotIn('"layer_input"', emitted_after_attention)

    def test_fp16_cache_store_dump_uses_resolved_cache_geometry(self) -> None:
        op = {
            "function": "kv_cache_store_batch_f16",
            "op": "kv_cache_store_batch_f16",
            "layer": 3,
            "args": [
                {"name": "kv_cache_k", "expr": "K_CACHE"},
                {"name": "kv_cache_v", "expr": "V_CACHE"},
                {"name": "k", "expr": "K_SOURCE"},
                {"name": "v", "expr": "V_SOURCE"},
                {"name": "start_pos", "expr": "model->pos"},
                {"name": "num_tokens", "expr": "96"},
                {"name": "num_kv_heads", "expr": "2"},
                {"name": "head_dim", "expr": "256"},
                {"name": "max_seq_len", "expr": "108"},
            ],
        }
        source = codegen_prefill_v8.emit_prefill_op(op, 7, {"embed_dim": 2560}, dump=True)
        self.assertIn(
            'ck_dump_f16_cache_token_major((const uint16_t*)K_CACHE, 3, '
            '"Kcache_post_store", 2, ((model->pos) + (num_tokens)), 108, 256);',
            source,
        )
        self.assertIn('"Vcache_post_store"', source)
        self.assertLess(source.index("kv_cache_store_batch_f16("), source.index("Kcache_post_store"))
        self.assertNotIn("Kcache_post_store", codegen_prefill_v8.emit_prefill_op(op, 7, {}, dump=False))

    @staticmethod
    def _bridge_embedding_ops() -> list[dict]:
        return [{
            "op": "dense_embedding_lookup",
            "function": "embedding_forward_bf16_fp32",
            "args": [
                {"name": "token_ids", "expr": "TOKENS"},
                {"name": "token_embeddings", "expr": "WEIGHTS"},
                {"name": "output", "expr": "OUT"},
                {"name": "vocab_size", "expr": "VOCAB_SIZE"},
                {"name": "embed_dim", "expr": "EMBED_DIM"},
                {"name": "aligned_embed_dim", "expr": "EMBED_DIM"},
            ],
        }]

    def test_unified_mixed_bridge_emits_one_full_prefill(self) -> None:
        config = {
            "embed_dim": 16,
            "num_deepstack_layers": 3,
            "context_length": 32,
            "multimodal_bridge_contract": {
                "prefix_policy": "mixed_visual_text_prefill",
                "prefill_batching": "unified_mixed",
                "providers": {
                    "prefix_insert": {
                        "resolved_function": "ck_multimodal_prefix_insert_f32",
                    },
                    "position_builder": {
                        "resolved_function": "ck_multimodal_mrope_positions_2d",
                    },
                },
                "prefill_schedule": {
                    "segments": ["text_before", "visual", "text_after"],
                    "cache_transition": "single_pass",
                    "position_transition": "explicit_full_sequence",
                    "position_transform": {
                        "kernel_id": "mrope_qk_text_imrope_positions_bf16_pytorch_storage",
                        "contract_id": "text_imrope_positions_bf16_input_pytorch_bf16_compute_bf16_output",
                        "resolved_function": "mrope_qk_text_imrope_positions_bf16_pytorch_storage",
                    },
                    "deepstack_injection": {
                        "kernel_id": "ck_residual_add_token_major_bf16_storage",
                        "contract_id": "residual_add_bf16_input_fp32_add_bf16_output",
                        "resolved_function": "ck_residual_add_token_major_bf16_storage",
                        "target": "visual_rows_after_decoder_layer",
                        "layers_from_config": "num_deepstack_layers",
                    },
                },
            },
        }
        emitted = codegen_prefill_v8.emit_multimodal_bridge_api(
            self._bridge_embedding_ops(), config
        )
        self.assertEqual(
            emitted.count("ck_prefill_from_embedded(g_model, total_tokens);"), 2
        )
        self.assertNotIn(
            "ck_prefill_from_embedded_range(g_model, prefix_tokens", emitted
        )
        helpers = codegen_prefill_v8._emit_multimodal_prefill_bridge_helpers(
            config, "mrope_qk_text_imrope_bf16_pytorch_storage"
        )
        self.assertIn("mrope_qk_text_imrope_positions_bf16_pytorch_storage(q, k,", helpers)
        self.assertIn("ck_residual_add_token_major_bf16_storage(dst_row, src, dst_row,", helpers)
        self.assertNotIn("mrope_qk_imrope_positions(q, k,", helpers)

    def test_unified_mixed_bridge_rejects_unresolved_position_provider(self) -> None:
        config = {
            "embed_dim": 16,
            "num_deepstack_layers": 3,
            "multimodal_bridge_contract": {
                "prefix_policy": "mixed_visual_text_prefill",
                "prefill_batching": "unified_mixed",
                "prefill_schedule": {
                    "segments": ["text_before", "visual", "text_after"],
                    "cache_transition": "single_pass",
                    "position_transition": "explicit_full_sequence",
                },
            },
        }
        with self.assertRaisesRegex(RuntimeError, "positions-aware M-RoPE provider"):
            codegen_prefill_v8._emit_multimodal_prefill_bridge_helpers(
                config, "mrope_qk_text_imrope_bf16_pytorch_storage"
            )

    def test_segmented_bridge_retains_three_cache_preserving_prefills(self) -> None:
        config = {
            "embed_dim": 16,
            "num_deepstack_layers": 3,
            "context_length": 32,
            "multimodal_bridge_contract": {
                "prefix_policy": "mixed_visual_text_prefill",
                "prefill_batching": "segmented_append",
                "providers": {
                    "prefix_insert": {
                        "resolved_function": "ck_multimodal_prefix_insert_f32",
                    },
                },
                "prefill_schedule": {
                    "segments": ["text_before", "visual", "text_after"],
                    "cache_transition": "append_preserve",
                    "position_transition": "segment_defined",
                },
            },
        }
        emitted = codegen_prefill_v8.emit_multimodal_bridge_api(
            self._bridge_embedding_ops(), config
        )
        self.assertIn(
            "ck_prefill_from_embedded_range(g_model, prefix_tokens", emitted
        )
        self.assertEqual(
            emitted.count("ck_prefill_from_embedded(g_model, total_tokens);"), 1
        )

    def test_qwen35_post_attention_prefill_exports_full_checkpoint_extents(self) -> None:
        cases = [
            (
                {
                    "function": "attn_gate_sigmoid_mul_forward",
                    "op": "attn_gate_sigmoid_mul",
                    "layer": 3,
                    "args": [
                        {"name": "out", "expr": "ATTN"},
                        {"name": "rows", "expr": "1034"},
                        {"name": "num_heads", "expr": "8"},
                        {"name": "state_dim", "expr": "256"},
                    ],
                },
                '"attn_out", (const float*)ATTN, (8) * (num_tokens) * (256)',
            ),
            (
                {
                    "function": "rmsnorm_forward_llama_production",
                    "op": "post_attention_norm",
                    "layer": 3,
                    "args": [
                        {"name": "output", "expr": "NORM"},
                        {"name": "tokens", "expr": "1034"},
                        {"name": "d_model", "expr": "1024"},
                    ],
                },
                '"post_attn_norm", (const float*)NORM, (num_tokens) * (1024)',
            ),
            (
                {
                    "function": "swiglu_forward_ggml",
                    "op": "silu_mul",
                    "layer": 3,
                    "args": [
                        {"name": "output", "expr": "MLP"},
                        {"name": "tokens", "expr": "1034"},
                        {"name": "dim", "expr": "3584"},
                    ],
                },
                '"mlp_swiglu", (const float*)MLP, (num_tokens) * (3584)',
            ),
        ]
        for op, expected in cases:
            with self.subTest(op=op["op"]):
                emitted = codegen_prefill_v8.emit_prefill_op(op, 1, {"embed_dim": 1024})
                self.assertIn(expected, emitted)

    def test_registered_fp16_cache_batch_store_uses_physical_call_arguments(self) -> None:
        op = {
            "function": "kv_cache_store_batch_f16",
            "op": "kv_cache_store_batch_f16",
            "layer": 3,
            "section": "body",
            "args": [
                {"name": "kv_cache_k", "source": "runtime:kv_cache_k_layer_f16", "expr": "K_CACHE"},
                {"name": "kv_cache_v", "source": "runtime:kv_cache_v_layer_f16", "expr": "V_CACHE"},
                {"name": "k", "source": "activation:k_src", "expr": "K_SCRATCH"},
                {"name": "v", "source": "activation:v_src", "expr": "V_SCRATCH"},
                {"name": "start_pos", "source": "runtime:prefill_start_pos", "expr": "model->pos"},
                {"name": "num_tokens", "source": "dim:seq_len", "expr": "1034"},
                {"name": "num_kv_heads", "source": "dim:num_kv_heads", "expr": "8"},
                {"name": "head_dim", "source": "dim:head_dim", "expr": "64"},
                {"name": "max_seq_len", "source": "dim:max_seq_len", "expr": "1034"},
            ],
        }

        emitted = codegen_prefill_v8.emit_prefill_op(
            op,
            11,
            {
                "decode_kv_cache_dtype": "fp16",
                "num_kv_heads": 8,
                "head_dim": 64,
                "context_len": 1034,
            },
        )

        self.assertIn("kv_cache_store_batch_f16(", emitted)
        self.assertIn("K_CACHE", emitted)
        self.assertIn("V_CACHE", emitted)
        self.assertIn("K_SCRATCH", emitted)
        self.assertIn("V_SCRATCH", emitted)
        self.assertIn("prefill_start_pos", emitted)
        self.assertIn("num_tokens", emitted)

    def test_recurrent_prefill_seq_len_args_use_runtime_num_tokens(self) -> None:
        op = {
            "function": "recurrent_split_qkv_forward",
            "op": "recurrent_split_qkv",
            "layer": 0,
            "section": "body",
            "args": [
                {"name": "packed_qkv", "expr": "(const float*)(model->bump + A_RECURRENT_PACKED)"},
                {"name": "q", "expr": "(float*)(model->bump + A_RECURRENT_Q)"},
                {"name": "k", "expr": "(float*)(model->bump + A_RECURRENT_K)"},
                {"name": "v", "expr": "(float*)(model->bump + A_RECURRENT_V)"},
                {"name": "seq_len", "expr": "1034"},
                {"name": "q_dim", "expr": "2048"},
                {"name": "k_dim", "expr": "2048"},
                {"name": "v_dim", "expr": "2048"},
            ],
        }

        emitted = codegen_prefill_v8.emit_prefill_op(op, 7, {"embed_dim": 1024})

        self.assertIn("num_tokens", emitted)
        self.assertNotIn("\n        1034,", emitted)
        self.assertIn("\n        2048,", emitted)

    def test_scalar_constant_is_not_rewritten_as_runtime_token_count(self) -> None:
        op = {
            "function": "recurrent_dt_gate_forward",
            "op": "recurrent_dt_gate",
            "layer": 0,
            "section": "body",
            "args": [
                {"name": "alpha", "expr": "alpha"},
                {"name": "dt_bias", "expr": "dt_bias"},
                {"name": "a", "expr": "a"},
                {"name": "gate", "expr": "gate"},
                {"name": "rows", "source": "dim:seq_len", "expr": "1034"},
                {"name": "num_heads", "source": "dim:gate_dim", "expr": "16"},
                {"name": "state_dim", "source": "const:1", "expr": "1"},
            ],
        }

        emitted = codegen_prefill_v8.emit_prefill_op(op, 9, {"embed_dim": 1024})

        self.assertIn("\n        num_tokens,\n        16,\n        1\n", emitted)
        self.assertNotIn("\n        num_tokens,\n        16,\n        num_tokens\n", emitted)

    def test_recurrent_prefill_exports_full_batched_projection_boundary(self) -> None:
        op = {
            "function": "gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch",
            "op": "recurrent_gate_proj",
            "layer": 0,
            "section": "body",
            "args": [
                {"name": "A", "expr": "activation"},
                {"name": "B", "expr": "weights"},
                {"name": "bias", "expr": "NULL"},
                {"name": "C", "expr": "gate_output"},
                {"name": "M", "source": "dim:seq_len", "expr": "1034"},
                {"name": "N", "source": "dim:attn_gate_dim", "expr": "2048"},
                {"name": "K", "source": "dim:embed_dim", "expr": "1024"},
            ],
        }

        emitted = codegen_prefill_v8.emit_prefill_op(op, 3, {"embed_dim": 1024})

        self.assertIn('ck_debug_export_hidden(model, 0, "z"', emitted)
        self.assertIn("(num_tokens) * (2048)", emitted)

    def test_map_owned_weight_preparation_is_deduplicated_and_budgeted(self) -> None:
        preparation = {
            "function": "ck_q5_0_prepare_q8_0_weight",
            "arguments": {"B": "B", "N": "N", "K": "K"},
            "prepared_bytes": "N * (K / 32) * 34",
            "max_total_bytes": 1024,
            "min_remaining_memory_bytes": 256,
        }
        args = [
            {"name": "A", "expr": "input_q8"},
            {"name": "B", "expr": "model->weight_q5"},
            {"name": "bias", "expr": "NULL"},
            {"name": "C", "expr": "output"},
            {"name": "M", "expr": "num_tokens"},
            {"name": "N", "expr": "64"},
            {"name": "K", "expr": "128"},
        ]
        op = {"function": "gemm_nt_q5_0_q8_0", "args": args,
              "call_abi": {"weight_preparation": preparation}}

        emitted = codegen_prefill_v8.emit_prefill_weight_prepare_function([op, op])

        self.assertEqual(emitted.count("ck_q5_0_prepare_q8_0_weight("), 1)
        self.assertIn(
            "ck_model_preparation_budget((size_t)1024, (size_t)256)", emitted
        )
        self.assertIn("mapped_prepared_item_0_0 <= mapped_prepared_budget_0", emitted)
        self.assertIn(
            "mapped_prepared_bytes_0 <= mapped_prepared_budget_0 - mapped_prepared_item_0_0",
            emitted,
        )
        self.assertIn("mapped_prepared_skipped_0 += 1", emitted)
        self.assertIn("const int mapped_prepared_result_0_0", emitted)
        self.assertIn("if (mapped_prepared_result_0_0 > 0)", emitted)
        self.assertIn(
            "if (mapped_prepared_bytes_0 > 0 || mapped_prepared_skipped_0 > 0)",
            emitted,
        )
        self.assertIn(
            "prepared %zu bytes within runtime budget %zu (map max 1024, reserve 256); "
            "skipped %d weight(s)",
            emitted,
        )
        self.assertIn("model->weight_q5, 64, 128", emitted)

    def test_map_owned_weight_preparation_skips_only_items_beyond_budget(self) -> None:
        preparation = {
            "function": "prepare_weight",
            "arguments": {"B": "B", "N": "N", "K": "K"},
            "prepared_bytes": "N * K",
            "max_total_bytes": 1024,
        }

        def op(weight: str, n: int, k: int) -> dict:
            return {
                "function": "gemm_nt_synthetic",
                "args": [
                    {"name": "B", "expr": weight},
                    {"name": "N", "expr": str(n)},
                    {"name": "K", "expr": str(k)},
                ],
                "call_abi": {"weight_preparation": preparation},
            }

        emitted = codegen_prefill_v8.emit_prefill_weight_prepare_function(
            [op("model->small", 8, 8), op("model->large", 64, 64)]
        )

        self.assertIn("prepare_weight(model->small, 8, 8)", emitted)
        self.assertIn("prepare_weight(model->large, 64, 64)", emitted)
        self.assertEqual(emitted.count("mapped_prepared_skipped_0 += 1"), 4)
        self.assertNotIn("if (mapped_prepared_bytes_0 <=", emitted)


if __name__ == "__main__":
    unittest.main()
