"""Small compiled proof for serialized sequence state switching."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "version" / "v8" / "scripts"
sys.path.insert(0, str(SCRIPTS))
from sequence_state_codegen_v8 import emit_sequence_state_api  # noqa: E402
from sequence_state_contract_v8 import resolve_sequence_state_contract  # noqa: E402
import codegen_core_v8  # noqa: E402
import codegen_v8  # noqa: E402
from codegen_prefill_v8 import emit_prefill_op  # noqa: E402


def _layout(*, extra: list[dict] | None = None) -> dict:
    return {
        "memory": {
            "arena": {"total_size": 256},
            "activations": {
                "buffers": [
                    {"name": "kv_cache", "abs_offset": 64, "size": 32, "lifetime": "sequence", "mutable": True},
                    {"name": "logits", "abs_offset": 128, "size": 32, "lifetime": "call", "mutable": True},
                    *(extra or []),
                ]
            },
        }
    }


class SequenceStateContractTests(unittest.TestCase):
    def test_recurrent_handles_switch_all_state_and_reject_stale_reuse(self) -> None:
        contract = {
            "arena_bytes": 192,
            "recurrent_conv_state_arena_offset": 64,
            "recurrent_conv_state_bytes": 16,
            "recurrent_ssm_state_arena_offset": 128,
            "recurrent_ssm_state_bytes": 32,
        }
        source = r'''
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define CK_EXPORT
#define KV_CACHE_SIZE 32
#define A_KV_CACHE 64
#define A_RECURRENT_CONV_STATE 192
#define A_RECURRENT_SSM_STATE 256
typedef struct {
    uint8_t *bump;
    size_t bump_size;
    float *kv_cache;
    uint16_t *kv_cache_f16;
    float *recurrent_conv_state;
    float *recurrent_ssm_state;
    int pos;
    int rope_pos;
} CKModel;
static CKModel *g_model;
''' + emit_sequence_state_api(contract) + r'''
static void mutate(float *conv, float *ssm, float delta) {
    conv[0] += delta;
    ssm[0] += delta;
}
static void advance(CKModel *model, float delta) {
''' + codegen_core_v8.emit_op({
    "op": "probe", "function": "mutate", "layer": 0,
    "args": [
        {"name": "conv", "buffer_ref": "recurrent_conv_state",
         "expr": "(float*)(model->bump + A_RECURRENT_CONV_STATE)"},
        {"name": "ssm", "buffer_ref": "recurrent_ssm_state",
         "expr": "(float*)(model->bump + A_RECURRENT_SSM_STATE)"},
        {"name": "delta", "expr": "delta"},
    ],
}) + r'''
}
int main(void) {
    _Alignas(64) uint8_t bump[320] = {0};
    _Alignas(64) uint8_t first[192] = {0};
    _Alignas(64) uint8_t second[192] = {0};
    CKModel model = {.bump = bump, .bump_size = sizeof(bump),
        .kv_cache = (float *)(bump + A_KV_CACHE),
        .kv_cache_f16 = (uint16_t *)(bump + A_KV_CACHE),
        .recurrent_conv_state = (float *)(bump + A_RECURRENT_CONV_STATE),
        .recurrent_ssm_state = (float *)(bump + A_RECURRENT_SSM_STATE)};
    g_model = &model;
    size_t bytes = 0, alignment = 0;
    assert(ck_model_sequence_state_requirements(&bytes, &alignment) == 0);
    assert(bytes == 192 && alignment == 64);
    uint64_t original = ck_model_sequence_state_default(), a = 0, b = 0;
    assert(ck_model_sequence_state_create(bump + A_RECURRENT_CONV_STATE,
                                          bytes, &a) == -2);
    assert(ck_model_sequence_state_create(first, bytes, &a) == 0);
    assert(ck_model_sequence_state_create(second, bytes, &b) == 0);
    assert(ck_model_sequence_state_activate(a) == 0);
    ((uint8_t *)model.kv_cache)[0] = 11;
    advance(&model, 1.0f);
    model.pos = 3;
    assert(ck_model_sequence_state_activate(b) == 0);
    assert(((uint8_t *)model.kv_cache)[0] == 0);
    assert(model.recurrent_conv_state[0] == 0.0f);
    assert(model.recurrent_ssm_state[0] == 0.0f);
    ((uint8_t *)model.kv_cache)[0] = 21;
    advance(&model, 2.0f);
    assert(ck_model_sequence_state_activate(a) == 0);
    assert(model.pos == 3);
    assert(((uint8_t *)model.kv_cache)[0] == 11);
    assert(model.recurrent_conv_state[0] == 1.0f);
    assert(model.recurrent_ssm_state[0] == 1.0f);
    advance(&model, 3.0f);
    assert(model.recurrent_conv_state[0] == 4.0f);
    assert(model.recurrent_ssm_state[0] == 4.0f);
    assert(ck_model_sequence_state_activate(b) == 0);
    assert(model.recurrent_conv_state[0] == 2.0f);
    assert(model.recurrent_ssm_state[0] == 2.0f);
    assert(ck_model_sequence_state_destroy(a) == 0);
    assert(ck_model_sequence_state_activate(a) == -1);
    uint64_t stale = a;
    assert(ck_model_sequence_state_create(first, bytes, &a) == 0);
    assert(a != stale && ck_model_sequence_state_activate(stale) == -1);
    assert(ck_model_sequence_state_activate(a) == 0);
    assert(((uint8_t *)model.recurrent_ssm_state)[0] == 0);
    assert(ck_model_sequence_state_activate(original) == 0);
    assert(model.recurrent_conv_state == (float *)(bump + A_RECURRENT_CONV_STATE));
    assert(model.recurrent_ssm_state == (float *)(bump + A_RECURRENT_SSM_STATE));
    ck_sequence_release_all();
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sequence.c").write_text(source)
            compile_result = subprocess.run(
                ["cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
                 "sequence.c", "-o", "sequence"], cwd=root,
                capture_output=True, text=True)
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            subprocess.run([str(root / "sequence")], check=True, capture_output=True)

    def test_recurrent_regions_require_checked_operands(self) -> None:
        layout = _layout(extra=[
            {"name": "recurrent_conv_state", "abs_offset": 192, "size": 16,
             "shape": "[2, 2]", "dtype": "fp32", "lifetime": "sequence", "mutable": True},
            {"name": "recurrent_ssm_state", "abs_offset": 256, "size": 32,
             "shape": "[2, 4]", "dtype": "fp32", "lifetime": "sequence", "mutable": True},
        ])
        layout["memory"]["arena"]["total_size"] = 320
        def state_arg(ref: str, offset: int = 0) -> dict:
            macro = "A_" + ref.upper()
            suffix = f"({macro} + {offset})" if offset else macro
            return {"buffer_ref": ref, "expr": f"(float*)(model->bump + {suffix})"}
        ops = [
            {"layer": 0, "args": [state_arg("recurrent_conv_state"),
                                   state_arg("recurrent_ssm_state")]},
            {"layer": 1, "args": [state_arg("recurrent_conv_state", 8),
                                   state_arg("recurrent_ssm_state", 16)]},
        ]
        config = {"num_layers": 2}
        self.assertIsNone(resolve_sequence_state_contract(layout, config))
        contract = resolve_sequence_state_contract(layout, config, ops)
        self.assertEqual(contract["arena_bytes"], 192)
        self.assertEqual(contract["recurrent_conv_state_arena_offset"], 64)
        self.assertEqual(contract["recurrent_ssm_state_arena_offset"], 128)
        source = codegen_core_v8.emit_model_and_api(
            layout=layout, config={"num_layers": 2, "num_kv_heads": 1,
                                   "head_dim": 4}, decode_ops=ops)
        self.assertIn("ck_model_sequence_state_create", source)
        self.assertIn("memset(g_model->recurrent_conv_state", source)
        self.assertIn("memset(g_model->recurrent_ssm_state", source)
        capability = codegen_v8._emit_runtime_capability_api(
            {"config": config, "operations": ops}, layout, None)
        self.assertIn("CK_MODEL_CAP_SEQUENCE_STATE_SWITCH", capability)
        self.assertNotIn("CK_MODEL_CAP_BATCH_DECODE_2", capability)
        self.assertIsNone(resolve_sequence_state_contract(layout, config,
            [{"layer": 0, "args": [state_arg("recurrent_conv_state"),
                                    state_arg("recurrent_ssm_state", 32)]}]))
        self.assertIsNone(resolve_sequence_state_contract(layout, config,
            [{"layer": 0, "args": [state_arg("recurrent_conv_state"),
                                    {"buffer_ref": "recurrent_ssm_state", "expr": "unknown_pointer"}]}]))
        prefill_ops = [{**op, "mode": "prefill"} for op in ops]
        for phase, phase_ops in (("decode", ops), ("prefill", prefill_ops)):
            # A valid access elsewhere must not hide an untyped or mislabelled
            # access that remains attached to model->bump in emitted C.
            for bad in (
                {"expr": state_arg("recurrent_ssm_state")["expr"]},
                {"buffer_ref": "recurrent_conv_state",
                 "expr": state_arg("recurrent_ssm_state")["expr"]},
            ):
                with self.subTest(phase=phase, bad=bad):
                    poisoned = [*phase_ops, {"layer": 0, "args": [bad]}]
                    self.assertIsNone(resolve_sequence_state_contract(layout, config, poisoned))
            for wrong in (
                {"layer": 1, "args": [state_arg("recurrent_conv_state"),
                                        state_arg("recurrent_ssm_state", 16)]},
                {"layer": 0, "args": [state_arg("recurrent_conv_state"),
                                        state_arg("recurrent_ssm_state", 16)]},
            ):
                with self.subTest(phase=phase, wrong=wrong):
                    self.assertIsNone(resolve_sequence_state_contract(
                        layout, config, [*phase_ops, wrong]))
        layout["memory"]["activations"]["buffers"][2]["shape"] = "[4, 1]"
        self.assertIsNone(resolve_sequence_state_contract(layout, config, ops))
        layout["memory"]["activations"]["buffers"][2]["shape"] = "[2, 2]"
        self.assertIsNone(resolve_sequence_state_contract(layout, config,
            [{"layer": 0, "args": [state_arg("recurrent_conv_state"),
                                    state_arg("recurrent_ssm_state")]},
             {"layer": 1, "args": [state_arg("recurrent_conv_state", 8),
                                    state_arg("recurrent_ssm_state")]}]))
        self.assertIsNone(resolve_sequence_state_contract(_layout(),
            {"layer_recurrent_policy": ["deltanet"]}, ops))
        self.assertIn("model->recurrent_conv_state", codegen_core_v8.emit_op({
            "op": "probe", "function": "probe", "args": [state_arg("recurrent_conv_state")]}))
        self.assertIn("model->recurrent_ssm_state", codegen_core_v8.emit_op({
            "op": "probe", "function": "probe", "args": [state_arg("recurrent_ssm_state", 16)]}))
        self.assertIn("model->recurrent_ssm_state", emit_prefill_op({
            "op": "probe", "function": "probe", "args": [state_arg("recurrent_ssm_state", 16)]},
            0, {}))

    def test_generated_capability_matches_generated_api(self) -> None:
        layout = _layout()
        config = {"num_layers": 1, "num_kv_heads": 1, "head_dim": 4}
        dense = codegen_core_v8.emit_model_and_api(layout=layout, config=config)
        declared = codegen_v8._emit_runtime_capability_api(
            {"config": config, "operations": []}, layout, None
        )
        self.assertIn("ck_model_sequence_state_create", dense)
        self.assertIn("CK_MODEL_CAP_SEQUENCE_STATE_SWITCH", declared)

        hybrid_layout = _layout(
            extra=[{"name": "arbitrary_buffer", "abs_offset": 160, "size": 16,
                    "define": "A_ARBITRARY_BUFFER", "lifetime": "sequence", "mutable": True}]
        )
        hybrid = codegen_core_v8.emit_model_and_api(layout=hybrid_layout, config=config)
        hybrid_declared = codegen_v8._emit_runtime_capability_api(
            {"config": config, "operations": []}, hybrid_layout, None
        )
        self.assertNotIn("ck_model_sequence_state_create", hybrid)
        self.assertNotIn("CK_MODEL_CAP_SEQUENCE_STATE_SWITCH", hybrid_declared)
        self.assertIn("memset(g_model->bump + A_ARBITRARY_BUFFER, 0, 16)", hybrid)

        missing_binding = _layout(extra=[
            {"name": "unbound_history", "abs_offset": 160, "size": 16,
             "lifetime": "sequence", "mutable": True},
        ])
        with self.assertRaisesRegex(RuntimeError, "checked arena binding"):
            codegen_core_v8.emit_model_and_api(layout=missing_binding, config=config)

        for offset, size, reason in (
            (-16, 16, "range"), (250, 16, "planned arena"),
            (128, 16, "overlaps logits"), (160, 0, "reset extent"),
        ):
            with self.subTest(offset=offset, size=size):
                invalid = _layout(extra=[
                    {"name": "extra_history", "abs_offset": offset, "size": size,
                     "define": "A_EXTRA_HISTORY", "lifetime": "sequence", "mutable": True},
                ])
                with self.assertRaisesRegex(ValueError, reason):
                    codegen_core_v8.emit_model_and_api(layout=invalid, config=config)

        wrong_macro = _layout(extra=[
            {"name": "extra_history", "abs_offset": 160, "size": 16,
             "define": "A_LOGITS", "lifetime": "sequence", "mutable": True},
        ])
        with self.assertRaisesRegex(RuntimeError, "checked arena binding"):
            codegen_core_v8.emit_model_and_api(layout=wrong_macro, config=config)

        duplicate_macro = _layout(extra=[
            {"name": "extra_history", "abs_offset": 160, "size": 16,
             "define": "A_EXTRA_HISTORY", "lifetime": "sequence", "mutable": True},
        ])
        duplicate_macro["memory"]["activations"]["buffers"][1]["define"] = "A_EXTRA_HISTORY"
        with self.assertRaisesRegex(RuntimeError, "checked arena binding"):
            codegen_core_v8.emit_model_and_api(layout=duplicate_macro, config=config)

    def test_emitted_reset_clears_only_declared_sequence_regions(self) -> None:
        layout = _layout(extra=[
            {"name": "extra_history", "abs_offset": 160, "size": 16,
             "define": "A_EXTRA_HISTORY", "lifetime": "sequence", "mutable": True},
            {"name": "neighbor", "abs_offset": 192, "size": 16,
             "lifetime": "call", "mutable": True},
        ])
        emitted = codegen_core_v8.emit_model_and_api(
            layout=layout, config={"num_layers": 1, "num_kv_heads": 1, "head_dim": 4})
        start = emitted.index("CK_EXPORT void ck_model_kv_cache_reset(void) {")
        end = emitted.index("CK_EXPORT int ck_model_kv_cache_enable", start)
        reset = emitted[start:end]
        source = r'''
#include <assert.h>
#include <stdint.h>
#include <string.h>
#define CK_EXPORT
#define KV_CACHE_SIZE 32
#define A_EXTRA_HISTORY 160
typedef struct {
    uint8_t *bump;
    uint8_t *kv_cache;
    int pos;
    int rope_pos;
    int bridge_has_explicit_positions;
} CKModel;
static CKModel *g_model;
static uint8_t arena[256];
static CKModel model;
static void ck_trace_pos(const char *kind, int layer, int token, int before, int after) {
    (void)kind; (void)layer; (void)token; (void)before; (void)after;
}
''' + reset + r'''
int main(void) {
    memset(arena, 0x5a, sizeof(arena));
    model = (CKModel){.bump = arena, .kv_cache = arena + 64,
                      .pos = 7, .rope_pos = 11, .bridge_has_explicit_positions = 1};
    g_model = &model;
    for (int run = 0; run < 2; ++run) {
        memset(arena + 64, 0xa1, 32);
        memset(arena + 160, 0xb2, 16);
        ck_model_kv_cache_reset();
        for (int i = 64; i < 96; ++i) assert(arena[i] == 0);
        for (int i = 160; i < 176; ++i) assert(arena[i] == 0);
        for (int i = 0; i < 64; ++i) assert(arena[i] == 0x5a);
        for (int i = 96; i < 160; ++i) assert(arena[i] == 0x5a);
        for (int i = 176; i < 256; ++i) assert(arena[i] == 0x5a);
        assert(model.pos == 0 && model.rope_pos == 0);
        assert(model.bridge_has_explicit_positions == 0);
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "reset.c").write_text(source)
            build = subprocess.run(
                ["cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
                 "reset.c", "-o", "reset"], cwd=root,
                capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            subprocess.run([str(root / "reset")], check=True, capture_output=True)

    def test_named_kv_lookup_uses_active_handle(self) -> None:
        source = codegen_v8._inject_activation_lookup_api(
            "", _layout(), sequence_switch=True
        )
        self.assertIn('strcmp(name, "kv_cache") == 0', source)
        self.assertIn("return (uintptr_t)g_model->kv_cache", source)
        self.assertIn("g_model->kv_cache != (float *)(g_model->bump + A_KV_CACHE)", source)
        legacy = codegen_v8._inject_activation_lookup_api(
            "", _layout(), sequence_switch=False
        )
        self.assertNotIn("return (uintptr_t)g_model->kv_cache", legacy)

    def test_dense_kv_contract_comes_from_layout(self) -> None:
        self.assertEqual(resolve_sequence_state_contract(_layout(), {}), {
        "kv_offset": 64,
        "kv_bytes": 32,
        "kv_alignment": 64,
        })
        self.assertIsNotNone(resolve_sequence_state_contract(_layout(
            extra=[{"name": "shared_table", "abs_offset": 160, "size": 16,
                    "lifetime": "model", "mutable": False}]), {}))
        self.assertIsNone(resolve_sequence_state_contract(_layout(), {"uses_cross_attention": True}))
        self.assertIsNone(resolve_sequence_state_contract(
            _layout(extra=[{"name": "any_name", "abs_offset": 160, "size": 16, "lifetime": "sequence", "mutable": True}]), {}
        ))
        self.assertIsNone(resolve_sequence_state_contract(
            _layout(extra=[{"name": "anything", "abs_offset": 160, "size": 16}]), {}
        ))
        self.assertIsNone(resolve_sequence_state_contract(
            _layout(extra=[{"name": "state", "abs_offset": 160, "size": 16, "lifetime": "model", "mutable": True}]), {}
        ))

    def test_invalid_kv_regions_do_not_receive_switching(self) -> None:
        for kv in (
            {"name": "kv_cache", "abs_offset": 250, "size": 32, "lifetime": "sequence", "mutable": True},
            {"name": "kv_cache", "abs_offset": -1, "size": 32, "lifetime": "sequence", "mutable": True},
            {"name": "kv_cache", "abs_offset": 65, "size": 32, "lifetime": "sequence", "mutable": True},
            {"name": "kv_cache", "abs_offset": 64, "size": 0, "lifetime": "sequence", "mutable": True},
        ):
            layout = _layout()
            layout["memory"]["activations"]["buffers"][0] = kv
            with self.assertRaises(ValueError):
                resolve_sequence_state_contract(layout, {})

    def test_overlapping_activation_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlaps"):
            resolve_sequence_state_contract(
                _layout(extra=[{"name": "scratch", "abs_offset": 80, "size": 16, "lifetime": "call", "mutable": True}]), {}
            )

    def test_emitted_switching_keeps_two_sequences_independent(self) -> None:
        source = r'''
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define CK_EXPORT
#define A_KV_CACHE 0
#define KV_CACHE_SIZE 16
typedef struct {
    uint8_t *bump;
    size_t bump_size;
    float *kv_cache;
    uint16_t *kv_cache_f16;
    int pos;
    int rope_pos;
} CKModel;
static CKModel *g_model;
''' + emit_sequence_state_api() + r'''
static void advance(uint8_t token) {
    assert(g_model->pos < KV_CACHE_SIZE);
    ((uint8_t *)g_model->kv_cache)[g_model->pos++] = token;
    g_model->rope_pos++;
}
int main(void) {
    _Alignas(64) uint8_t bump[64] = {0};
    _Alignas(64) uint8_t arena_a[64] = {0};
    _Alignas(64) uint8_t arena_b[64] = {0};
    _Alignas(64) uint8_t arena_c[64] = {0};
    CKModel model = {.bump = bump, .bump_size = sizeof(bump),
                     .kv_cache = (float *)bump,
                     .kv_cache_f16 = (uint16_t *)bump};
    g_model = &model;
    size_t bytes = 0, alignment = 0;
    assert(ck_model_sequence_state_requirements(&bytes, &alignment) == 0);
    assert(bytes == 16 && alignment == 64);
    uint64_t original = ck_model_sequence_state_default();
    uint64_t a = 0, b = 0, rejected = 123;
    assert(original != 0);
    assert(ck_model_sequence_state_create(NULL, bytes, &rejected) == -2 && rejected == 0);
    assert(ck_model_sequence_state_create(arena_a + 1, bytes, &rejected) == -2);
    assert(ck_model_sequence_state_create(arena_a, bytes - 1, &rejected) == -2);
    assert(ck_model_sequence_state_create(arena_a, bytes, &a) == 0);
    assert(ck_model_sequence_state_create(arena_a, bytes, &rejected) == -2);
    assert(ck_model_sequence_state_create(arena_b, bytes, &b) == 0);
    assert(ck_model_sequence_state_create(arena_c, bytes, &rejected) == -3 && rejected == 0);
    assert(ck_model_sequence_state_create(bump, bytes, &rejected) == -2);
    assert(ck_model_sequence_state_create(arena_b, bytes, &rejected) == -2);
    assert(ck_model_sequence_state_activate(a) == 0);
    advance(11); advance(12);
    assert(ck_model_sequence_state_activate(b) == 0);
    advance(21);
    assert(ck_model_sequence_state_activate(a) == 0);
    assert(model.pos == 2 && model.rope_pos == 2);
    assert(((uint8_t *)model.kv_cache)[0] == 11);
    assert(((uint8_t *)model.kv_cache)[1] == 12);
    advance(13);
    assert(ck_model_sequence_state_activate(b) == 0);
    assert(model.pos == 1 && model.rope_pos == 1);
    assert(((uint8_t *)model.kv_cache)[0] == 21);
    assert(((uint8_t *)model.kv_cache)[1] == 0);
    memset(model.kv_cache, 0, KV_CACHE_SIZE);
    model.pos = 0;
    model.rope_pos = 0;
    assert(ck_model_sequence_state_activate(a) == 0);
    assert(model.pos == 3 && ((uint8_t *)model.kv_cache)[0] == 11);
    assert(ck_model_sequence_state_activate(b) == 0);
    assert(model.pos == 0 && ((uint8_t *)model.kv_cache)[0] == 0);
    assert(ck_model_sequence_state_activate(original) == 0);
    assert(model.pos == 0 && model.kv_cache == (float *)bump);
    assert(ck_model_sequence_state_destroy(a) == 0);
    assert(ck_model_sequence_state_activate(a) == -1);
    uint64_t old_a = a;
    assert(ck_model_sequence_state_create(arena_a, bytes, &a) == 0);
    assert(a != old_a && ck_model_sequence_state_activate(old_a) == -1);
    assert(ck_model_sequence_state_activate(a) == 0);
    assert(ck_model_sequence_state_destroy(a) == 0);
    assert(model.kv_cache == (float *)bump);
    assert(ck_model_sequence_state_destroy(a) == -1);
    assert(ck_model_sequence_state_destroy(b) == 0);
    assert(ck_model_sequence_state_destroy(original) == -1);
    ck_sequence_release_all();
    assert(ck_model_sequence_state_activate(original) == -1);
    uint64_t new_default = ck_model_sequence_state_default();
    assert(new_default != original);
    assert(ck_model_sequence_state_activate(original) == -1);
    assert(ck_model_sequence_state_activate(new_default) == 0);
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sequence.c").write_text(source)
            subprocess.run(
                ["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "sequence.c", "-o", "sequence"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run([str(root / "sequence")], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
