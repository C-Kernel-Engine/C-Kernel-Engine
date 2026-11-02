"""Compiled checks for process-local, planner-bounded sequence checkpoints."""

from __future__ import annotations

import os
import ctypes as C
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "version" / "v8" / "scripts"
sys.path.insert(0, str(SCRIPTS))
from sequence_state_codegen_v8 import emit_sequence_state_api  # noqa: E402
from certify_sequence_checkpoint_v8 import declared_state_regions, signature  # noqa: E402


class SequenceCheckpointTests(unittest.TestCase):
    def test_probe_requires_declared_recurrent_regions_and_extents(self) -> None:
        buffers = [
            {"name": "kv_cache", "abs_offset": 0, "size": 32,
             "lifetime": "sequence", "mutable": True},
            {"name": "recurrent_conv_state", "abs_offset": 64, "size": 16,
             "lifetime": "sequence", "mutable": True},
            {"name": "recurrent_ssm_state", "abs_offset": 128, "size": 32,
             "lifetime": "sequence", "mutable": True},
        ]
        layout = {"memory": {"arena": {"total_size": 256},
                             "activations": {"buffers": buffers}}}
        required = declared_state_regions(layout)
        self.assertEqual(required, {"kv_cache": 32,
                                    "recurrent_conv_state": 16,
                                    "recurrent_ssm_state": 32})
        storage = {name: C.create_string_buffer(size) for name, size in required.items()}

        class FakeModel:
            def __init__(self) -> None:
                self.pointers = {name: C.addressof(value) for name, value in storage.items()}
                self.extents = dict(required)

            def ck_model_get_named_activation_ptr(self, name: bytes) -> int:
                return self.pointers.get(name.decode(), 0)

            def ck_model_get_named_activation_nbytes(self, name: bytes) -> int:
                return self.extents.get(name.decode(), -1)

            def ck_model_get_active_tokens(self) -> int:
                return 3

        model = FakeModel()
        logits = (C.c_float * 4)(1, 2, 3, 4)
        self.assertEqual(set(signature(model, logits, required)["state"]), set(required))
        model.pointers["recurrent_conv_state"] = 0
        with self.assertRaisesRegex(RuntimeError, "recurrent_conv_state missing"):
            signature(model, logits, required)
        model.pointers["recurrent_conv_state"] = C.addressof(storage["recurrent_conv_state"])
        model.extents["recurrent_ssm_state"] = 31
        with self.assertRaisesRegex(RuntimeError, "recurrent_ssm_state missing"):
            signature(model, logits, required)
        self.assertEqual(declared_state_regions({"memory": {
            "arena": {"total_size": 64}, "activations": {"buffers": buffers[:1]}}}),
            {"kv_cache": 32})

    def test_kv_and_recurrent_checkpoints_restore_only_validated_state(self) -> None:
        for recurrent in (False, True):
            with self.subTest(recurrent=recurrent):
                contract = None
                if recurrent:
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
#define MAX_SEQ_LEN 32
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
static unsigned advance(CKModel *model, unsigned token) {
    uint8_t *cache = (uint8_t *)model->kv_cache;
    unsigned result = token + (unsigned)model->pos + cache[0];
#ifdef CK_SEQ_CONV_BYTES
    result += ((uint8_t *)model->recurrent_conv_state)[0];
    result += ((uint8_t *)model->recurrent_ssm_state)[0];
    ((uint8_t *)model->recurrent_conv_state)[0] ^= (uint8_t)token;
    ((uint8_t *)model->recurrent_ssm_state)[0] += (uint8_t)token;
#endif
    cache[model->pos++] = (uint8_t)result;
    ++model->rope_pos;
    return result;
}
static int same_logical_state(const uint8_t *left, const uint8_t *right) {
    if (memcmp(left, right, KV_CACHE_SIZE) != 0) return 0;
#ifdef CK_SEQ_CONV_BYTES
    if (memcmp(left + CK_SEQ_CONV_OFFSET, right + CK_SEQ_CONV_OFFSET,
               CK_SEQ_CONV_BYTES) != 0) return 0;
    if (memcmp(left + CK_SEQ_SSM_OFFSET, right + CK_SEQ_SSM_OFFSET,
               CK_SEQ_SSM_BYTES) != 0) return 0;
#endif
    return 1;
}
int main(void) {
    _Alignas(64) uint8_t bump[320] = {0};
    _Alignas(64) uint8_t first[192] = {0};
    _Alignas(64) uint8_t second[192] = {0};
    _Alignas(64) uint8_t snapshot[256] = {0};
    _Alignas(64) uint8_t inactive_snapshot[256] = {0};
    uint8_t prefix_state[192] = {0}, failed_state[192] = {0};
    uint8_t active_state[192] = {0};
    CKModel model = {.bump = bump, .bump_size = sizeof(bump),
        .kv_cache = (float *)(bump + A_KV_CACHE),
        .kv_cache_f16 = (uint16_t *)(bump + A_KV_CACHE),
        .recurrent_conv_state = (float *)(bump + A_RECURRENT_CONV_STATE),
        .recurrent_ssm_state = (float *)(bump + A_RECURRENT_SSM_STATE)};
    g_model = &model;
    size_t state_bytes = 0, alignment = 0, snapshot_bytes = 0;
    assert(ck_model_sequence_state_requirements(&state_bytes, &alignment) == 0);
    assert(ck_model_sequence_checkpoint_requirements(&snapshot_bytes, &alignment) == -1);
    assert(ck_model_sequence_checkpoint_bind_instance(0, 0) == -1);
    assert(ck_model_sequence_checkpoint_bind_instance(17, 29) == 0);
    assert(ck_model_sequence_checkpoint_bind_instance(19, 31) == -1);
    assert(ck_model_sequence_checkpoint_requirements(&snapshot_bytes, &alignment) == 0);
    assert(alignment == 64 && snapshot_bytes <= sizeof(snapshot));
    uint64_t original = ck_model_sequence_state_default(), a = 0, b = 0;
    assert(original != 0);
    assert(ck_model_sequence_state_create(first, state_bytes, &a) == 0);
    assert(ck_model_sequence_state_create(second, state_bytes, &b) == 0);
    assert(ck_model_sequence_state_activate(a) == 0);
    for (size_t i = 0; i < KV_CACHE_SIZE; ++i)
        ((uint8_t *)model.kv_cache)[i] = (uint8_t)(17 + i);
#ifdef CK_SEQ_CONV_BYTES
    for (size_t i = 0; i < CK_SEQ_CONV_BYTES; ++i)
        ((uint8_t *)model.recurrent_conv_state)[i] = (uint8_t)(23 + i);
    for (size_t i = 0; i < CK_SEQ_SSM_BYTES; ++i)
        ((uint8_t *)model.recurrent_ssm_state)[i] = (uint8_t)(29 + i);
#endif
    model.pos = 9;
    model.rope_pos = 12;
    assert(ck_model_sequence_checkpoint_save(a, snapshot, snapshot_bytes) == 0);
    memcpy(prefix_state, first, state_bytes);
    assert(ck_model_sequence_checkpoint_save(a, snapshot, snapshot_bytes - 1) == -2);
    assert(ck_model_sequence_checkpoint_save(a, first, snapshot_bytes) == -2);
    assert(ck_model_sequence_checkpoint_save(a, second, snapshot_bytes) == -2);
    assert(!ck_sequence_checkpoint_disjoint(&g_sequence_states[0], snapshot_bytes));
    assert(!ck_sequence_checkpoint_disjoint(&model, snapshot_bytes));
    model.pos = MAX_SEQ_LEN + 1;
    assert(ck_model_sequence_checkpoint_save(a, snapshot, snapshot_bytes) == -2);
    model.pos = 9;
    assert(ck_model_sequence_state_activate(b) == 0);
    memset(second, 99, state_bytes);
    model.pos = 2;
    model.rope_pos = 3;
    memcpy(failed_state, second, state_bytes);
    assert(ck_model_sequence_checkpoint_save(a, inactive_snapshot, snapshot_bytes) == 0);
    assert(memcmp(snapshot, inactive_snapshot, snapshot_bytes) == 0);
    assert(memcmp(second, failed_state, state_bytes) == 0);
    assert(model.pos == 2 && model.rope_pos == 3);
    assert(ck_model_sequence_checkpoint_restore(b, snapshot, snapshot_bytes - 1) == -2);
    assert(memcmp(second, failed_state, state_bytes) == 0);
    assert(model.pos == 2 && model.rope_pos == 3);
    assert(ck_model_sequence_checkpoint_restore(b, snapshot + 1, snapshot_bytes) == -2);
    assert(memcmp(second, failed_state, state_bytes) == 0);
    snapshot[sizeof(CKSequenceCheckpointV8)] ^= 1;
    assert(ck_model_sequence_checkpoint_restore(b, snapshot, snapshot_bytes) == -2);
    assert(memcmp(second, failed_state, state_bytes) == 0);
    snapshot[sizeof(CKSequenceCheckpointV8)] ^= 1;
#ifdef CK_SEQ_CONV_BYTES
    snapshot[sizeof(CKSequenceCheckpointV8) + CK_SEQ_SSM_OFFSET + 3] ^= 1;
    assert(ck_model_sequence_checkpoint_restore(b, snapshot, snapshot_bytes) == -2);
    assert(memcmp(second, failed_state, state_bytes) == 0);
    snapshot[sizeof(CKSequenceCheckpointV8) + CK_SEQ_SSM_OFFSET + 3] ^= 1;
#endif
    CKSequenceCheckpointV8 *head = (CKSequenceCheckpointV8 *)snapshot;
    ++head->instance_nonce_lo;
    head->checksum = ck_sequence_checkpoint_hash(
        head, snapshot + sizeof(CKSequenceCheckpointV8));
    assert(ck_model_sequence_checkpoint_restore(b, snapshot, snapshot_bytes) == -2);
    assert(memcmp(second, failed_state, state_bytes) == 0);
    --head->instance_nonce_lo;
    head->checksum = ck_sequence_checkpoint_hash(
        head, snapshot + sizeof(CKSequenceCheckpointV8));
    head->pos = MAX_SEQ_LEN + 1;
    head->checksum = ck_sequence_checkpoint_hash(
        head, snapshot + sizeof(CKSequenceCheckpointV8));
    assert(ck_model_sequence_checkpoint_restore(b, snapshot, snapshot_bytes) == -2);
    assert(memcmp(second, failed_state, state_bytes) == 0);
    assert(model.pos == 2 && model.rope_pos == 3);
    head->pos = 9;
    head->checksum = ck_sequence_checkpoint_hash(
        head, snapshot + sizeof(CKSequenceCheckpointV8));
    assert(ck_model_sequence_checkpoint_restore(b, first, snapshot_bytes) == -2);
    assert(ck_model_sequence_checkpoint_restore(b, snapshot, snapshot_bytes) == 0);
    assert(model.pos == 9 && model.rope_pos == 12);
    assert(same_logical_state(second, prefix_state));
#ifdef CK_SEQ_CONV_BYTES
    assert(((uint8_t *)model.recurrent_conv_state)[0] == 23);
    assert(((uint8_t *)model.recurrent_ssm_state)[0] == 29);
#endif
    unsigned resumed = advance(&model, 5);
    assert(ck_model_sequence_state_activate(a) == 0);
    assert(model.pos == 9 && ((uint8_t *)model.kv_cache)[0] == 17);
    unsigned isolated = advance(&model, 5);
    assert(resumed == isolated && model.pos == 10 && model.rope_pos == 13);
    assert(same_logical_state(first, second));
#ifdef CK_SEQ_CONV_BYTES
    assert(((uint8_t *)first)[CK_SEQ_CONV_OFFSET] ==
           ((uint8_t *)second)[CK_SEQ_CONV_OFFSET]);
    assert(((uint8_t *)first)[CK_SEQ_SSM_OFFSET] ==
           ((uint8_t *)second)[CK_SEQ_SSM_OFFSET]);
#endif
    memcpy(active_state, first, state_bytes);
    assert(ck_model_sequence_checkpoint_restore(b, inactive_snapshot, snapshot_bytes) == 0);
    assert(model.pos == 10 && model.rope_pos == 13);
    assert(memcmp(first, active_state, state_bytes) == 0);
    assert(same_logical_state(second, prefix_state));
    assert(ck_model_sequence_state_activate(b) == 0);
    assert(model.pos == 9 && model.rope_pos == 12);
    assert(ck_model_sequence_state_activate(a) == 0);
    assert(ck_model_sequence_state_destroy(b) == 0);
    assert(ck_model_sequence_checkpoint_restore(b, snapshot, snapshot_bytes) == -1);
    assert(ck_model_sequence_state_activate(original) == 0);
    ck_sequence_release_all();
    assert(ck_model_sequence_state_default() != 0);
    assert(ck_model_sequence_state_create(second, state_bytes, &b) == 0);
    assert(ck_model_sequence_checkpoint_bind_instance(17, 29) == 0);
    assert(ck_model_sequence_checkpoint_restore(b, snapshot, snapshot_bytes) == -2);
    assert(((uint8_t *)second)[0] == 0);
    return 0;
}
'''
                with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
                    root = Path(directory)
                    (root / "checkpoint.c").write_text(source)
                    compiled = subprocess.run(
                        ["cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
                         "checkpoint.c", "-o", "checkpoint"], cwd=root,
                        capture_output=True, text=True,
                    )
                    self.assertEqual(compiled.returncode, 0, compiled.stderr)
                    executed = subprocess.run([str(root / "checkpoint")], capture_output=True,
                                              text=True)
                    self.assertEqual(executed.returncode, 0, executed.stderr)

    def test_checkpoint_rejects_prior_shared_library_lifetime(self) -> None:
        contract = {
            "arena_bytes": 192,
            "recurrent_conv_state_arena_offset": 64,
            "recurrent_conv_state_bytes": 16,
            "recurrent_ssm_state_arena_offset": 128,
            "recurrent_ssm_state_bytes": 32,
        }
        library = r'''
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define CK_EXPORT __attribute__((visibility("default")))
#define MAX_SEQ_LEN 32
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
static _Alignas(64) uint8_t bump[320];
static CKModel model;
CK_EXPORT uint64_t test_epoch(void) { return g_sequence_model_epoch; }
CK_EXPORT int test_init(uint64_t hi, uint64_t lo) {
    memset(bump, 0, sizeof(bump));
    model = (CKModel){.bump = bump, .bump_size = sizeof(bump),
        .kv_cache = (float *)(bump + A_KV_CACHE),
        .kv_cache_f16 = (uint16_t *)(bump + A_KV_CACHE),
        .recurrent_conv_state = (float *)(bump + A_RECURRENT_CONV_STATE),
        .recurrent_ssm_state = (float *)(bump + A_RECURRENT_SSM_STATE)};
    g_model = &model;
    return ck_model_sequence_checkpoint_bind_instance(hi, lo);
}
CK_EXPORT int test_save(void *out, size_t bytes) {
    uint64_t handle = ck_model_sequence_state_default();
    for (size_t i = 0; i < KV_CACHE_SIZE; ++i)
        ((uint8_t *)model.kv_cache)[i] = (uint8_t)(10 + i);
    for (size_t i = 0; i < CK_SEQ_CONV_BYTES; ++i)
        ((uint8_t *)model.recurrent_conv_state)[i] = (uint8_t)(20 + i);
    for (size_t i = 0; i < CK_SEQ_SSM_BYTES; ++i)
        ((uint8_t *)model.recurrent_ssm_state)[i] = (uint8_t)(30 + i);
    model.pos = 9;
    model.rope_pos = 12;
    return ck_model_sequence_checkpoint_save(handle, out, bytes);
}
CK_EXPORT int test_restore(const void *snapshot, size_t bytes) {
    return ck_model_sequence_checkpoint_restore(
        ck_model_sequence_state_default(), snapshot, bytes);
}
CK_EXPORT int test_clean(void) {
    if (model.pos || model.rope_pos) return 0;
    for (size_t i = 0; i < KV_CACHE_SIZE; ++i)
        if (((uint8_t *)model.kv_cache)[i]) return 0;
    for (size_t i = 0; i < CK_SEQ_CONV_BYTES; ++i)
        if (((uint8_t *)model.recurrent_conv_state)[i]) return 0;
    for (size_t i = 0; i < CK_SEQ_SSM_BYTES; ++i)
        if (((uint8_t *)model.recurrent_ssm_state)[i]) return 0;
    return 1;
}
CK_EXPORT void test_teardown(void) {
    ck_sequence_release_all();
    g_model = NULL;
}
'''
        driver = r'''
#define _POSIX_C_SOURCE 200112L
#include <assert.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
typedef uint64_t (*epoch_fn)(void);
typedef int (*init_fn)(uint64_t, uint64_t);
typedef int (*requirements_fn)(size_t *, size_t *);
typedef int (*save_fn)(void *, size_t);
typedef int (*restore_fn)(const void *, size_t);
typedef int (*clean_fn)(void);
typedef void (*teardown_fn)(void);
#define SYMBOL(lib, type, name) ((type)dlsym((lib), (name)))
int main(int argc, char **argv) {
    assert(argc == 2);
    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    assert(lib);
    epoch_fn epoch = SYMBOL(lib, epoch_fn, "test_epoch");
    init_fn init = SYMBOL(lib, init_fn, "test_init");
    requirements_fn requirements = SYMBOL(lib, requirements_fn,
        "ck_model_sequence_checkpoint_requirements");
    save_fn save = SYMBOL(lib, save_fn, "test_save");
    teardown_fn teardown = SYMBOL(lib, teardown_fn, "test_teardown");
    assert(epoch && init && requirements && save && teardown && epoch() == 1);
    assert(init(101, 202) == 0);
    size_t bytes = 0, alignment = 0;
    assert(requirements(&bytes, &alignment) == 0 && alignment == 64);
    void *snapshot = NULL;
    assert(posix_memalign(&snapshot, alignment, bytes) == 0);
    assert(save(snapshot, bytes) == 0);
    teardown();
    assert(dlclose(lib) == 0);

    lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    assert(lib);
    epoch = SYMBOL(lib, epoch_fn, "test_epoch");
    init = SYMBOL(lib, init_fn, "test_init");
    restore_fn restore = SYMBOL(lib, restore_fn, "test_restore");
    clean_fn clean = SYMBOL(lib, clean_fn, "test_clean");
    teardown = SYMBOL(lib, teardown_fn, "test_teardown");
    /* Epoch 1 proves this is a fresh DSO lifetime, not a still-open handle. */
    assert(epoch && epoch() == 1 && init && restore && clean && teardown);
    assert(init(303, 404) == 0);
    assert(clean());
    assert(restore(snapshot, bytes) == -2);
    assert(clean());
    teardown();
    assert(dlclose(lib) == 0);
    free(snapshot);
    return 0;
}
'''
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            root = Path(directory)
            (root / "model.c").write_text(library)
            (root / "driver.c").write_text(driver)
            for command in (
                ["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-fPIC",
                 "-shared", "model.c", "-o", "libmodel.so"],
                ["cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
                 "driver.c", "-ldl", "-o", "driver"],
            ):
                compiled = subprocess.run(command, cwd=root, capture_output=True, text=True)
                self.assertEqual(compiled.returncode, 0, compiled.stderr)
            executed = subprocess.run([str(root / "driver"), str(root / "libmodel.so")],
                                      capture_output=True, text=True)
            self.assertEqual(executed.returncode, 0, executed.stderr)

    def test_native_loader_rejects_each_missing_checkpoint_export(self) -> None:
        root = Path(__file__).resolve().parents[1]
        model = r'''
#include "ck_model_abi_v8.h"
#include <stdint.h>
#include <stdatomic.h>
#include <string.h>
#include <sched.h>
static uint64_t bound_hi, bound_lo;
static uint64_t state_word;
static uint64_t embed_seen_state;
static int position, fail_embed_once, cancel_embed_once;
static const atomic_int *cancel_flag;
static atomic_int hold_restore, entered_restore;
static int32_t token_ids[2] = {1, 1};
static float logits[4] = {0, 1, 0, 0};
static const uint64_t caps = CK_MODEL_CAP_INIT |
    CK_MODEL_CAP_AUTOREGRESSIVE_DECODE | CK_MODEL_CAP_TEXT_ENCODE |
    CK_MODEL_CAP_TOKEN_DECODE | CK_MODEL_CAP_SEQUENCE_STATE_SWITCH |
    CK_MODEL_CAP_SEQUENCE_CHECKPOINT;
uint32_t ck_model_get_abi_version(void) { return CK_MODEL_ABI_V8_VERSION; }
uint64_t ck_model_get_capabilities(void) { return caps; }
int ck_model_get_runtime_descriptor(CKModelRuntimeDescriptorV8 *out, size_t size) {
    if (!out || size < sizeof(*out)) return -1;
    *out = (CKModelRuntimeDescriptorV8){.struct_size=sizeof(*out),
        .abi_version=CK_MODEL_ABI_V8_VERSION, .capabilities=caps,
        .artifact_role=CK_MODEL_ROLE_DECODER, .context_length=16, .vocab_size=4};
    return 0;
}
int ck_model_init(const char *weights) {
    (void)weights; bound_hi=bound_lo=state_word=0; position=0; return 0;
}
void ck_model_free(void) { bound_hi=bound_lo=0; }
int ck_model_embed_tokens(const int32_t *tokens, int count) {
    embed_seen_state = state_word;
    if (fail_embed_once) { fail_embed_once = 0; return -1; }
    if (count <= 0 || !tokens) return -1;
    for (int i=0; i<count; ++i) state_word += (uint64_t)tokens[i];
    position += count;
    if (cancel_embed_once) {
        cancel_embed_once = 0;
        if (cancel_flag) atomic_store((atomic_int *)cancel_flag, 1);
    }
    return 0;
}
void ck_model_kv_cache_reset(void) { state_word=0; position=0; }
void ck_model_set_cancel_flag(const atomic_int *flag) { cancel_flag=flag; }
int ck_model_forward(float *out) { if (out) memcpy(out, logits, sizeof(logits)); return 0; }
int ck_model_decode(int32_t token, float *out) {
    state_word += (uint64_t)token; ++position; return ck_model_forward(out);
}
int ck_model_get_active_tokens(void) { return position; }
int ck_model_get_vocab_size(void) { return 4; }
int ck_model_prefill_append_row_group(void) { return 1; }
int ck_model_prefill_append_min_prefix(void) { return 4; }
int ck_model_prefill_append_alignment(void) { return 4; }
float *ck_model_get_logits(void) { return logits; }
int ck_model_get_logits_stride(void) { return 0; }
int ck_model_can_encode_text(void) { return 1; }
int ck_model_encode_text(const char *text, int length) {
    (void)length; token_ids[0] = text && text[0] == 'B' ? 2 : 1;
    return text && text[0] == 'M' ? 2 : 1;
}
const int32_t *ck_model_get_token_buffer(void) { return token_ids; }
int ck_model_decode_tokens(const int32_t *tokens, int count, char *text, int capacity) {
    (void)tokens; (void)count;
    if (capacity < 2 || !text) return 1;
    text[0]='x'; text[1]=0; return 1;
}
#ifndef OMIT_DEFAULT
uint64_t ck_model_sequence_state_default(void) { return 7; }
#endif
#ifndef OMIT_BIND
int ck_model_sequence_checkpoint_bind_instance(uint64_t hi, uint64_t lo) {
    bound_hi=hi; bound_lo=lo; return (hi|lo) ? 0 : -1;
}
#endif
#ifndef OMIT_REQUIREMENTS
int ck_model_sequence_checkpoint_requirements(size_t *bytes, size_t *alignment) {
    if (!bytes || !alignment) return -1;
    *bytes=64; *alignment=64; return 0;
}
#endif
#ifndef OMIT_SAVE
int ck_model_sequence_checkpoint_save(uint64_t handle, void *out, size_t bytes) {
    if (handle != 7 || !out || bytes != 64) return -1;
    uint64_t data[4] = {bound_hi, bound_lo, state_word, (uint64_t)position};
    memcpy(out, data, sizeof(data));
    return 0;
}
#endif
#ifndef OMIT_RESTORE
int ck_model_sequence_checkpoint_restore(uint64_t handle, const void *in,
                                         size_t bytes) {
    if (handle != 7 || !in || bytes != 64) return -1;
    if (atomic_load(&hold_restore)) {
        atomic_store(&entered_restore, 1);
        while (atomic_load(&hold_restore)) sched_yield();
    }
    const uint64_t *data = (const uint64_t *)in;
    if (data[0] != bound_hi || data[1] != bound_lo) return -2;
    state_word = data[2];
    position = (int)data[3];
    return 0;
}
#endif
uint64_t ck_fake_bound_hi(void) { return bound_hi; }
uint64_t ck_fake_bound_lo(void) { return bound_lo; }
uint64_t ck_fake_state(void) { return state_word; }
uint64_t ck_fake_embed_seen_state(void) { return embed_seen_state; }
void ck_fake_set_state(uint64_t value) { state_word = value; }
void ck_fake_fail_embed_once(void) { fail_embed_once = 1; }
void ck_fake_cancel_embed_once(void) { cancel_embed_once = 1; }
void ck_fake_hold_restore(int hold) { atomic_store(&hold_restore, hold); }
int ck_fake_restore_entered(void) { return atomic_load(&entered_restore); }
void ck_fake_make_snapshot(void *out, uint64_t value, uint64_t pos) {
    uint64_t data[4] = {bound_hi, bound_lo, value, pos};
    memcpy(out, data, sizeof(data));
}
'''
        driver = r'''
#define _POSIX_C_SOURCE 200112L
#include "ck_session_v8.h"
#include <assert.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
typedef uint64_t (*nonce_fn)(void);
typedef void (*set_fn)(uint64_t);
typedef void (*snapshot_fn)(void *, uint64_t, uint64_t);
typedef void (*trigger_fn)(void);
typedef void (*hold_fn)(int);
typedef int (*entered_fn)(void);
typedef struct { CKSessionV8 *session; void *snapshot; size_t bytes; int status; } RestoreJob;
static void *restore_thread(void *arg) {
    RestoreJob *job = (RestoreJob *)arg;
    job->status = ck_session_v8_checkpoint_restore(
        job->session, job->snapshot, job->bytes);
    return NULL;
}
int main(int argc, char **argv) {
    assert(argc == 8);
    CKSessionConfigV8 config = {.struct_size=sizeof(config),
        .abi_version=CK_SESSION_ABI_V8_VERSION,
        .model_library_path=argv[1], .weights_path=argv[2],
        .manifest_path=argv[2], .context_length=16,
        .required_capabilities=CK_MODEL_CAP_SEQUENCE_CHECKPOINT};
    CKSessionV8 *session = NULL;
    assert(ck_session_v8_open(&config, &session) == CK_SESSION_V8_OK && session);
    void *model = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    assert(model);
    nonce_fn high = (nonce_fn)dlsym(model, "ck_fake_bound_hi");
    nonce_fn low = (nonce_fn)dlsym(model, "ck_fake_bound_lo");
    assert(high && low && (high() | low()));
    nonce_fn get_state = (nonce_fn)dlsym(model, "ck_fake_state");
    nonce_fn get_embed_seen = (nonce_fn)dlsym(model, "ck_fake_embed_seen_state");
    set_fn set_state = (set_fn)dlsym(model, "ck_fake_set_state");
    trigger_fn fail_embed = (trigger_fn)dlsym(model, "ck_fake_fail_embed_once");
    trigger_fn cancel_embed = (trigger_fn)dlsym(model, "ck_fake_cancel_embed_once");
    snapshot_fn make_snapshot = (snapshot_fn)dlsym(model, "ck_fake_make_snapshot");
    hold_fn hold_restore = (hold_fn)dlsym(model, "ck_fake_hold_restore");
    entered_fn restore_entered = (entered_fn)dlsym(model, "ck_fake_restore_entered");
    assert(get_state && get_embed_seen && set_state && make_snapshot &&
           fail_embed && cancel_embed && hold_restore && restore_entered);
    int32_t minimum = 0, append_alignment = 0, row_group = 0;
    assert(ck_session_v8_prefill_contract(session, &minimum,
                                         &append_alignment, &row_group) == 0);
    assert(minimum == 4 && append_alignment == 4 && row_group == 1);
    size_t bytes = 0, alignment = 0;
    assert(ck_session_v8_checkpoint_requirements(session, &bytes, &alignment) == 0);
    assert(bytes == 64 && alignment == 64);
    void *snapshot = NULL;
    assert(posix_memalign(&snapshot, alignment, bytes) == 0);
    assert(ck_session_v8_checkpoint_save(session, snapshot, bytes) ==
           CK_SESSION_V8_ERROR_INVALID_ARGUMENT);
    make_snapshot(snapshot, 123, 1);
    assert(ck_session_v8_checkpoint_restore(session, snapshot, bytes) == 0);
    assert(get_state() == 123);
    set_state(456);
    assert(ck_session_v8_checkpoint_save(session, snapshot, bytes) == 0);
    set_state(789);
    assert(ck_session_v8_checkpoint_restore(session, snapshot, bytes) == 0);
    assert(get_state() == 456);
    hold_restore(1);
    RestoreJob job = {.session=session, .snapshot=snapshot, .bytes=bytes, .status=123};
    pthread_t worker;
    assert(pthread_create(&worker, NULL, restore_thread, &job) == 0);
    while (!restore_entered()) sched_yield();
    assert(ck_session_v8_checkpoint_requirements(session, &bytes, &alignment) ==
           CK_SESSION_V8_ERROR_BUSY);
    assert(ck_session_v8_checkpoint_save(session, snapshot, bytes) ==
           CK_SESSION_V8_ERROR_BUSY);
    hold_restore(0);
    assert(pthread_join(worker, NULL) == 0 && job.status == 0);
    assert(ck_session_v8_reset(session) == 0);
    assert(ck_session_v8_checkpoint_save(session, snapshot, bytes) ==
           CK_SESSION_V8_ERROR_INVALID_ARGUMENT);
    int32_t prefix_ids[4] = {1, 1, 1, 1};
    int32_t invalid_ids[4] = {1, 1, 1, 4};
    assert(ck_session_v8_prefill_ids(session, prefix_ids, 3, 1, 0) ==
           CK_SESSION_V8_ERROR_INVALID_ARGUMENT);
    assert(ck_session_v8_prefill_ids(session, prefix_ids, 4, 13, 0) ==
           CK_SESSION_V8_ERROR_INVALID_ARGUMENT);
    assert(ck_session_v8_prefill_ids(session, invalid_ids, 4, 1, 0) ==
           CK_SESSION_V8_ERROR_INVALID_ARGUMENT);
    assert(get_state() == 0);
    assert(ck_session_v8_prefill_ids(session, prefix_ids, 4, 2, 0) == 0);
    assert(get_state() == 4);
    assert(ck_session_v8_checkpoint_save(session, snapshot, bytes) == 0);
    assert(ck_session_v8_prefill_ids(session, prefix_ids, 4, 2,
           CK_SESSION_PREFILL_CONTINUE_STATE) == 0);
    assert(get_state() == 8);
    assert(ck_session_v8_checkpoint_restore(session, snapshot, bytes) == 0);
    assert(get_state() == 4);
    CKSessionGenerateRequestV8 multi = {.struct_size=sizeof(multi),
        .abi_version=CK_SESSION_ABI_V8_VERSION, .user_text="MULTI",
        .max_tokens=1, .flags=CK_SESSION_REQUEST_RAW_PROMPT |
        CK_SESSION_REQUEST_CONTINUE_STATE |
        CK_SESSION_REQUEST_REJECT_PROMPT_TRUNCATION};
    CKSessionGenerateResultV8 multi_result = {.struct_size=sizeof(multi_result),
        .abi_version=CK_SESSION_ABI_V8_VERSION};
    assert(ck_session_v8_generate(session, &multi, NULL, NULL, &multi_result) == 0);
    assert(get_embed_seen() == 4 && get_state() == 6);
    assert(ck_session_v8_checkpoint_restore(session, snapshot, bytes) == 0);
    int32_t exact_suffix[2] = {1, 2};
    multi.user_text = "";
    assert(ck_session_v8_generate_ids(session, &multi, exact_suffix, 2,
                                      NULL, NULL, &multi_result) == 0);
    assert(get_embed_seen() == 4 && get_state() == 7);
    assert(ck_session_v8_checkpoint_restore(session, snapshot, bytes) == 0);
    exact_suffix[1] = 4;
    assert(ck_session_v8_generate_ids(session, &multi, exact_suffix, 2,
                                      NULL, NULL, &multi_result) ==
           CK_SESSION_V8_ERROR_INVALID_ARGUMENT);
    assert(get_state() == 4);
    exact_suffix[1] = 2;
    cancel_embed();
    assert(ck_session_v8_prefill_ids(session, prefix_ids, 4, 2,
           CK_SESSION_PREFILL_CONTINUE_STATE) == CK_SESSION_V8_ERROR_CANCELLED);
    assert(get_state() == 0);
    assert(ck_session_v8_checkpoint_save(session, snapshot, bytes) ==
           CK_SESSION_V8_ERROR_INVALID_ARGUMENT);
    CKSessionGenerateRequestV8 request = {.struct_size=sizeof(request),
        .abi_version=CK_SESSION_ABI_V8_VERSION, .user_text="A",
        .max_tokens=1, .flags=CK_SESSION_REQUEST_RAW_PROMPT |
        CK_SESSION_REQUEST_REJECT_PROMPT_TRUNCATION};
    CKSessionGenerateResultV8 result = {.struct_size=sizeof(result),
        .abi_version=CK_SESSION_ABI_V8_VERSION};
    assert(ck_session_v8_generate(session, &request, NULL, NULL, &result) == 0);
    assert(get_state() == 1 && result.stop_reason == CK_SESSION_STOP_TOKEN_LIMIT);
    assert(ck_session_v8_checkpoint_save(session, snapshot, bytes) == 0);
    void *saved_a = NULL;
    assert(posix_memalign(&saved_a, alignment, bytes) == 0);
    memcpy(saved_a, snapshot, bytes);
    request.user_text = "B";
    assert(ck_session_v8_generate(session, &request, NULL, NULL, &result) == 0);
    assert(get_state() == 2);
    assert(ck_session_v8_checkpoint_restore(session, saved_a, bytes) == 0);
    request.user_text = "suffix";
    request.flags = CK_SESSION_REQUEST_RAW_PROMPT |
                    CK_SESSION_REQUEST_CONTINUE_STATE |
                    CK_SESSION_REQUEST_REJECT_PROMPT_TRUNCATION;
    request.flags &= ~CK_SESSION_REQUEST_RAW_PROMPT;
    assert(ck_session_v8_generate(session, &request, NULL, NULL, &result) ==
           CK_SESSION_V8_ERROR_INVALID_ARGUMENT);
    assert(get_state() == 1);
    request.flags |= CK_SESSION_REQUEST_RAW_PROMPT;
    request.user_text = "MULTI";
    assert(ck_session_v8_generate(session, &request, NULL, NULL, &result) ==
           CK_SESSION_V8_ERROR_INVALID_ARGUMENT);
    assert(get_state() == 1);
    request.user_text = "suffix";
    assert(ck_session_v8_generate(session, &request, NULL, NULL, &result) == 0);
    assert(get_embed_seen() == 1 && get_state() == 2);
    assert(ck_session_v8_checkpoint_save(session, snapshot, bytes) == 0);
    make_snapshot(snapshot, 456, 15);
    assert(ck_session_v8_checkpoint_restore(session, snapshot, bytes) == 0);
    request.max_tokens = 2;
    assert(ck_session_v8_generate(session, &request, NULL, NULL, &result) ==
           CK_SESSION_V8_ERROR_INVALID_ARGUMENT);
    assert(strstr(ck_session_v8_last_error(session), "context") != NULL);
    assert(get_state() == 456);
    assert(ck_session_v8_checkpoint_save(session, snapshot, bytes) == 0);
    request.max_tokens = 1;
    assert(ck_session_v8_checkpoint_restore(session, saved_a, bytes) == 0);
    fail_embed();
    assert(ck_session_v8_generate(session, &request, NULL, NULL, &result) ==
           CK_SESSION_V8_ERROR_RUNTIME);
    assert(ck_session_v8_checkpoint_save(session, snapshot, bytes) ==
           CK_SESSION_V8_ERROR_INVALID_ARGUMENT);
    assert(ck_session_v8_checkpoint_restore(session, saved_a, bytes) == 0);
    cancel_embed();
    assert(ck_session_v8_generate(session, &request, NULL, NULL, &result) == 0);
    assert(result.stop_reason == CK_SESSION_STOP_CANCELLED);
    assert(ck_session_v8_checkpoint_save(session, snapshot, bytes) ==
           CK_SESSION_V8_ERROR_INVALID_ARGUMENT);
    request.flags = CK_SESSION_REQUEST_RAW_PROMPT;
    request.user_text = "A";
    assert(ck_session_v8_generate(session, &request, NULL, NULL, &result) == 0);
    assert(get_state() == 1);
    uint64_t first_hi=high(), first_lo=low();
    ck_session_v8_close(session);
    assert(ck_session_v8_open(&config, &session) == CK_SESSION_V8_OK && session);
    assert(high() != first_hi || low() != first_lo);
    set_state(999);
    assert(ck_session_v8_checkpoint_restore(session, snapshot, bytes) ==
           CK_SESSION_V8_ERROR_INVALID_ARGUMENT);
    assert(get_state() == 999);
    ck_session_v8_close(session);
    free(snapshot);
    free(saved_a);
    dlclose(model);
    for (int i=3; i<argc; ++i) {
        config.model_library_path=argv[i];
        session=NULL;
        assert(ck_session_v8_open(&config, &session) == CK_SESSION_V8_ERROR_LOAD);
        assert(!session);
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            folder = Path(directory)
            build = folder / "build"
            built = subprocess.run(
                ["make", "--no-print-directory", "ck-session-v8",
                 f"BUILD_DIR={build}"], cwd=root, capture_output=True, text=True,
                env={**os.environ, "TMPDIR": str(folder)},
            )
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            source = folder / "fake.c"
            source.write_text(model)
            paths = []
            for omitted in (None, "DEFAULT", "BIND", "REQUIREMENTS", "SAVE", "RESTORE"):
                path = folder / ("libvalid.so" if omitted is None else
                                 f"libmissing_{omitted.lower()}.so")
                command = ["cc", "-std=c11", "-shared", "-fPIC", "-I", str(root / "include")]
                if omitted:
                    command.append(f"-DOMIT_{omitted}")
                command.extend([str(source), "-o", str(path)])
                compiled = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(compiled.returncode, 0, compiled.stderr)
                paths.append(path)
            (folder / "weights.bump").write_bytes(b"")
            driver_source = folder / "driver.c"
            driver_source.write_text(driver)
            driver_exe = folder / "driver"
            compiled = subprocess.run(
                ["cc", "-std=c11", "-I", str(root / "include"),
                 str(driver_source), "-L", str(build), "-lck_session_v8",
                 "-ldl", "-pthread", f"-Wl,-rpath,{build}", "-o", str(driver_exe)],
                capture_output=True, text=True,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            executed = subprocess.run(
                [str(driver_exe), str(paths[0]), str(folder / "weights.bump"),
                 *(str(path) for path in paths[1:])],
                capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(executed.returncode, 0, executed.stderr)


if __name__ == "__main__":
    unittest.main()
