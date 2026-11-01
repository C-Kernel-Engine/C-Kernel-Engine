"""Compiled checks for process-local, planner-bounded sequence checkpoints."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "version" / "v8" / "scripts"
sys.path.insert(0, str(SCRIPTS))
from sequence_state_codegen_v8 import emit_sequence_state_api  # noqa: E402


class SequenceCheckpointTests(unittest.TestCase):
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
int main(void) {
    _Alignas(64) uint8_t bump[320] = {0};
    _Alignas(64) uint8_t first[192] = {0};
    _Alignas(64) uint8_t second[192] = {0};
    _Alignas(64) uint8_t snapshot[256] = {0};
    CKModel model = {.bump = bump, .bump_size = sizeof(bump),
        .kv_cache = (float *)(bump + A_KV_CACHE),
        .kv_cache_f16 = (uint16_t *)(bump + A_KV_CACHE),
        .recurrent_conv_state = (float *)(bump + A_RECURRENT_CONV_STATE),
        .recurrent_ssm_state = (float *)(bump + A_RECURRENT_SSM_STATE)};
    g_model = &model;
    size_t state_bytes = 0, alignment = 0, snapshot_bytes = 0;
    assert(ck_model_sequence_state_requirements(&state_bytes, &alignment) == 0);
    assert(ck_model_sequence_checkpoint_requirements(&snapshot_bytes, &alignment) == 0);
    assert(alignment == 64 && snapshot_bytes <= sizeof(snapshot));
    uint64_t original = ck_model_sequence_state_default(), a = 0, b = 0;
    assert(original != 0);
    assert(ck_model_sequence_state_create(first, state_bytes, &a) == 0);
    assert(ck_model_sequence_state_create(second, state_bytes, &b) == 0);
    assert(ck_model_sequence_state_activate(a) == 0);
    ((uint8_t *)model.kv_cache)[0] = 17;
#ifdef CK_SEQ_CONV_BYTES
    ((uint8_t *)model.recurrent_conv_state)[0] = 23;
    ((uint8_t *)model.recurrent_ssm_state)[0] = 29;
#endif
    model.pos = 9;
    model.rope_pos = 12;
    assert(ck_model_sequence_checkpoint_save(a, snapshot, snapshot_bytes) == 0);
    assert(ck_model_sequence_checkpoint_save(a, snapshot, snapshot_bytes - 1) == -2);
    assert(ck_model_sequence_checkpoint_save(a, first, snapshot_bytes) == -2);
    assert(ck_model_sequence_checkpoint_save(a, second, snapshot_bytes) == -2);
    assert(!ck_sequence_checkpoint_disjoint(&g_sequence_states[0], snapshot_bytes));
    assert(!ck_sequence_checkpoint_disjoint(&model, snapshot_bytes));
    model.pos = MAX_SEQ_LEN + 1;
    assert(ck_model_sequence_checkpoint_save(a, snapshot, snapshot_bytes) == -2);
    model.pos = 9;
    assert(ck_model_sequence_state_activate(b) == 0);
    ((uint8_t *)model.kv_cache)[0] = 99;
    model.pos = 2;
    assert(ck_model_sequence_checkpoint_restore(b, snapshot, snapshot_bytes - 1) == -2);
    assert(model.pos == 2 && ((uint8_t *)model.kv_cache)[0] == 99);
    snapshot[sizeof(CKSequenceCheckpointV8)] ^= 1;
    assert(ck_model_sequence_checkpoint_restore(b, snapshot, snapshot_bytes) == -2);
    assert(model.pos == 2 && ((uint8_t *)model.kv_cache)[0] == 99);
    snapshot[sizeof(CKSequenceCheckpointV8)] ^= 1;
    CKSequenceCheckpointV8 *head = (CKSequenceCheckpointV8 *)snapshot;
    ++head->model_cookie;
    head->checksum = ck_sequence_checkpoint_hash(
        head, snapshot + sizeof(CKSequenceCheckpointV8));
    assert(ck_model_sequence_checkpoint_restore(b, snapshot, snapshot_bytes) == -2);
    assert(model.pos == 2 && ((uint8_t *)model.kv_cache)[0] == 99);
    --head->model_cookie;
    head->checksum = ck_sequence_checkpoint_hash(
        head, snapshot + sizeof(CKSequenceCheckpointV8));
    assert(ck_model_sequence_checkpoint_restore(b, first, snapshot_bytes) == -2);
    assert(ck_model_sequence_checkpoint_restore(b, snapshot, snapshot_bytes) == 0);
    assert(model.pos == 9 && model.rope_pos == 12);
    assert(((uint8_t *)model.kv_cache)[0] == 17);
#ifdef CK_SEQ_CONV_BYTES
    assert(((uint8_t *)model.recurrent_conv_state)[0] == 23);
    assert(((uint8_t *)model.recurrent_ssm_state)[0] == 29);
#endif
    unsigned resumed = advance(&model, 5);
    assert(ck_model_sequence_state_activate(a) == 0);
    assert(model.pos == 9 && ((uint8_t *)model.kv_cache)[0] == 17);
    unsigned isolated = advance(&model, 5);
    assert(resumed == isolated && model.pos == 10 && model.rope_pos == 13);
    assert(((uint8_t *)first)[9] == ((uint8_t *)second)[9]);
#ifdef CK_SEQ_CONV_BYTES
    assert(((uint8_t *)first)[CK_SEQ_CONV_OFFSET] ==
           ((uint8_t *)second)[CK_SEQ_CONV_OFFSET]);
    assert(((uint8_t *)first)[CK_SEQ_SSM_OFFSET] ==
           ((uint8_t *)second)[CK_SEQ_SSM_OFFSET]);
#endif
    assert(ck_model_sequence_state_destroy(b) == 0);
    assert(ck_model_sequence_checkpoint_restore(b, snapshot, snapshot_bytes) == -1);
    assert(ck_model_sequence_state_activate(original) == 0);
    ck_sequence_release_all();
    assert(ck_model_sequence_state_default() != 0);
    assert(ck_model_sequence_state_create(second, state_bytes, &b) == 0);
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
                    subprocess.run([str(root / "checkpoint")], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
